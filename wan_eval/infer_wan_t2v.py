#!/usr/bin/env python3
"""Batch and multi-GPU inference for official Wan2.1 T2V models.

This script preserves the official Wan model weights, text encoder, VAE, and
sampling solvers. It adds three evaluation-side utilities:

1. batch-oriented prompt inference for official Wan2.1 T2V;
2. optional spatial RoPE lambda loading compatible with our DiffSynth training;
3. sample sharding across multiple GPUs by launching one model worker per GPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.cuda.amp as amp
from tqdm import tqdm
from peft import LoraConfig, inject_adapter_in_model

ROOT_DIR = Path(__file__).resolve().parents[1]
WAN_ROOT = ROOT_DIR / 'projects' / 'Wan2_1'
DIFFSYNTH_ROOT = ROOT_DIR / 'DiffSynth-Studio'
if str(WAN_ROOT) not in sys.path:
    sys.path.insert(0, str(WAN_ROOT))
if str(DIFFSYNTH_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFSYNTH_ROOT))

import wan  # noqa: E402
from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS  # noqa: E402
from wan.modules import model as wan_model_mod  # noqa: E402
from wan.utils.fm_solvers import (  # noqa: E402
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # noqa: E402
from wan.utils.utils import cache_video  # noqa: E402
from diffsynth.core import load_state_dict  # noqa: E402
from diffsynth.models.wan_video_dit import WanSpatialRopeLambda  # noqa: E402


@dataclass(frozen=True)
class PromptSample:
    sample_id: str
    prompt: str
    negative_prompt: str | None = None
    seed: int | None = None
    raw_record: dict | None = None


_LAMBDA_PATCH_INSTALLED = False


def parse_step_list(value: str | None) -> set[int] | None:
    if value is None:
        return None
    value = str(value).strip()
    if value == '':
        return None
    steps = set()
    for item in value.split(','):
        item = item.strip()
        if item == '':
            continue
        step = int(item)
        if step <= 0:
            raise ValueError(f'Lambda denoising steps must be positive 1-based indices, got {step}.')
        steps.add(step)
    return steps or None


def _rescale_official_rope_freqs(freqs: torch.Tensor, lambda_scales: torch.Tensor | None, num_heads: int) -> torch.Tensor:
    if lambda_scales is None:
        return freqs

    original_dtype = freqs.dtype
    if freqs.device.type == 'npu':
        freqs = freqs.to(torch.complex64)
    c = freqs.shape[-1]
    c_f = c - 2 * (c // 3)
    c_h = c // 3
    c_w = c // 3
    freqs_f, freqs_h, freqs_w = freqs.split([c_f, c_h, c_w], dim=-1)

    scales = lambda_scales.to(device=freqs.device, dtype=torch.float64)
    if scales.ndim == 1:
        scale_f, scale_h, scale_w = scales[0], scales[1], scales[2]
    elif scales.ndim == 2:
        if scales.shape[0] != int(num_heads) or scales.shape[1] != 3:
            raise ValueError(
                'Head-wise RoPE lambda scales must have shape [num_heads, 3], '
                f'got {tuple(scales.shape)} for num_heads={int(num_heads)}.'
            )
        scale_f = scales[:, 0].view(1, int(num_heads), 1)
        scale_h = scales[:, 1].view(1, int(num_heads), 1)
        scale_w = scales[:, 2].view(1, int(num_heads), 1)
    else:
        raise ValueError(
            'RoPE lambda scales must have shape [3] or [num_heads, 3], '
            f'got {tuple(scales.shape)}.'
        )

    target_dtype = original_dtype if getattr(original_dtype, 'is_complex', False) else torch.complex64

    def rescale_axis(axis_freqs: torch.Tensor, axis_scale: torch.Tensor) -> torch.Tensor:
        phase = torch.angle(axis_freqs.to(torch.complex128)) * axis_scale
        return torch.polar(torch.ones_like(phase), phase).to(dtype=target_dtype)

    return torch.cat(
        [
            rescale_axis(freqs_f, scale_f),
            rescale_axis(freqs_h, scale_h),
            rescale_axis(freqs_w, scale_w),
        ],
        dim=-1,
    )


@amp.autocast(enabled=False)
def rope_apply_with_lambda(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    rope_lambda_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    n = x.size(2)
    c = x.size(3) // 2
    freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat(
            [
                freqs_split[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs_split[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs_split[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        freqs_i = _rescale_official_rope_freqs(freqs_i, rope_lambda_scales, n)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).float()


def install_official_wan_lambda_patch() -> None:
    global _LAMBDA_PATCH_INSTALLED
    if _LAMBDA_PATCH_INSTALLED:
        return

    def patched_self_attn_forward(self, x, seq_lens, grid_sizes, freqs, rope_lambda_scales=None):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        x = wan_model_mod.flash_attention(
            q=rope_apply_with_lambda(q, grid_sizes, freqs, rope_lambda_scales),
            k=rope_apply_with_lambda(k, grid_sizes, freqs, rope_lambda_scales),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size,
        )
        x = x.flatten(2)
        x = self.o(x)
        return x

    def patched_block_forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        rope_lambda_scales=None,
    ):
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0],
            seq_lens,
            grid_sizes,
            freqs,
            rope_lambda_scales=rope_lambda_scales,
        )
        with amp.autocast(dtype=torch.float32):
            x = x + y * e[2]

        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
        with amp.autocast(dtype=torch.float32):
            x = x + y * e[5]
        return x

    def patched_model_forward(self, x, t, context, seq_len, clip_fea=None, y=None):
        if self.model_type == 'i2v' or self.model_type == 'flf2v':
            assert clip_fea is not None and y is not None
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(
            [
                torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
                for u in x
            ]
        )

        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(wan_model_mod.sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        context_lens = None
        context = self.text_embedding(
            torch.stack(
                [torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]
            )
        )

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)

        rope_lambda_scales_all = None
        rope_lambda_module = getattr(self, 'spatial_rope_lambda', None)
        lambda_enabled_for_step = getattr(self, '_wan_eval_lambda_enabled_for_step', True)
        if rope_lambda_module is not None and lambda_enabled_for_step:
            rope_lambda_scales_all = rope_lambda_module(t)

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
        )

        for block_id, block in enumerate(self.blocks):
            rope_lambda_scales = None
            if rope_lambda_scales_all is not None:
                if rope_lambda_scales_all.ndim == 1:
                    rope_lambda_scales = rope_lambda_scales_all
                else:
                    rope_lambda_scales = rope_lambda_scales_all[block_id]
            x = block(x, rope_lambda_scales=rope_lambda_scales, **kwargs)

        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    wan_model_mod.WanSelfAttention.forward = patched_self_attn_forward
    wan_model_mod.WanAttentionBlock.forward = patched_block_forward
    wan_model_mod.WanModel.forward = patched_model_forward
    _LAMBDA_PATCH_INSTALLED = True


def load_json(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


_STEP_RE = re.compile(r'^(step|epoch)-(\d+)(?:_lambda)?\.safetensors$')


def sort_checkpoint_paths(paths: list[Path]) -> list[Path]:
    def key(path: Path):
        match = _STEP_RE.match(path.name)
        if match is None:
            return (0, path.name)
        return (int(match.group(2)), path.name)
    return sorted(paths, key=key)


def parse_lora_target_modules(value: str | list[str] | tuple[str, ...] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        modules = [str(item).strip() for item in value if str(item).strip()]
        return modules or None
    value = str(value).strip()
    if value == '':
        return None
    return [item.strip() for item in value.split(',') if item.strip()]


def preset_to_lora_target_modules(preset: str | None) -> list[str] | None:
    if preset is None:
        return None
    preset = str(preset).strip()
    if preset == '':
        return None
    mapping = {
        'attn': ['q', 'k', 'v', 'o'],
        'ffn': ['ffn.0', 'ffn.2'],
        'attn_ffn': ['q', 'k', 'v', 'o', 'ffn.0', 'ffn.2'],
        'attn+ffn': ['q', 'k', 'v', 'o', 'ffn.0', 'ffn.2'],
    }
    if preset not in mapping:
        raise ValueError(f'Unsupported LoRA preset: {preset}')
    return mapping[preset]


def mapping_lora_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    mapped = {}
    for key, value in state_dict.items():
        if 'lora_A.weight' in key or 'lora_B.weight' in key:
            mapped_key = key.replace('lora_A.weight', 'lora_A.default.weight').replace('lora_B.weight', 'lora_B.default.weight')
            mapped[mapped_key] = value
        elif 'lora_A.default.weight' in key or 'lora_B.default.weight' in key:
            mapped[key] = value
    return mapped


def parse_experiment_name(experiment_name: str) -> dict:
    parsed = {
        'lambda_enabled': 'lambda' in experiment_name,
        'lora_enabled': False,
        'lora_rank': None,
        'lora_alpha': None,
        'lora_target_modules': None,
        'wan_spatial_rope_lambda_scope': None,
        'wan_spatial_rope_lambda_timestep_conditioned': None,
        'wan_spatial_rope_lambda_hidden_dim': None,
        'wan_spatial_rope_lambda_parametrization': None,
        'wan_spatial_rope_lambda_min': None,
        'wan_spatial_rope_lambda_init_eps': None,
        'wan_spatial_rope_lambda_fixed_h': None,
        'wan_spatial_rope_lambda_fixed_w': None,
        'wan_spatial_rope_lambda_global': None,
    }

    float_pattern = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
    patterns = {
        'lora_rank': r'(?:^|-)lora_rank_([^\-\s]+)',
        'lora_alpha': r'(?:^|-)lora_alpha_([^\-\s]+)',
        'lora_modules': r'(?:^|-)lora_modules_([^\-\s]+)',
        'lambda_scope': r'(?:^|-)lambda_scope_([^\-\s]+)',
        'lambda_tcond': r'(?:^|-)lambda_tcond_([^\-\s]+)',
        'lambda_hidden': r'(?:^|-)lambda_hidden_([^\-\s]+)',
        'lambda_parametrization': r'(?:^|-)lambda_param_([^\-\s]+)',
        'lambda_min': rf'(?:^|-)lambda_min_({float_pattern})',
        'lambda_init_eps': rf'(?:^|-)lambda_init_eps_({float_pattern})',
        'fixed_lambda_h': rf'(?:^|-)lambda_h_({float_pattern})',
        'fixed_lambda_w': rf'(?:^|-)lambda_w_({float_pattern})',
        'lambda_global': r'(?:^|-)lambda_global_([^\-\s]+)',
        'legacy_range_parametrization': r'(?:^|-)range_(.+?)_min_',
        'legacy_range_min': rf'(?:^|-)range_.+?_min_({float_pattern})(?=_eps_|-steps_|-epochs_|-warmup_|-adam_beta1_|-beta2_|-timestep_|-seed_|$)',
        'legacy_range_eps': rf'(?:^|-)range_.+?_min_{float_pattern}_eps_({float_pattern})(?=-steps_|-epochs_|-warmup_|-adam_beta1_|-beta2_|-timestep_|-seed_|$)',
    }

    matches = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, experiment_name)
        if match is not None:
            matches[key] = match.group(1)

    if 'lora_rank' in matches:
        parsed['lora_rank'] = int(matches['lora_rank'])
        parsed['lora_enabled'] = True
    if 'lora_alpha' in matches:
        parsed['lora_alpha'] = int(matches['lora_alpha'])
        parsed['lora_enabled'] = True
    if 'lora_modules' in matches:
        parsed['lora_target_modules'] = preset_to_lora_target_modules(matches['lora_modules'])
        parsed['lora_enabled'] = True
    if 'lambda_scope' in matches:
        parsed['wan_spatial_rope_lambda_scope'] = matches['lambda_scope']
    if 'lambda_tcond' in matches:
        parsed['wan_spatial_rope_lambda_timestep_conditioned'] = bool(int(matches['lambda_tcond']))
    if 'lambda_hidden' in matches:
        parsed['wan_spatial_rope_lambda_hidden_dim'] = int(matches['lambda_hidden'])
    if 'lambda_parametrization' in matches:
        parsed['wan_spatial_rope_lambda_parametrization'] = matches['lambda_parametrization']
    elif 'legacy_range_parametrization' in matches:
        parsed['wan_spatial_rope_lambda_parametrization'] = matches['legacy_range_parametrization']
    elif 'fixed_lambda' in experiment_name:
        parsed['wan_spatial_rope_lambda_parametrization'] = 'fixed'
    if 'lambda_min' in matches:
        parsed['wan_spatial_rope_lambda_min'] = float(matches['lambda_min'])
    elif 'legacy_range_min' in matches:
        parsed['wan_spatial_rope_lambda_min'] = float(matches['legacy_range_min'])
    if 'lambda_init_eps' in matches:
        parsed['wan_spatial_rope_lambda_init_eps'] = float(matches['lambda_init_eps'])
    elif 'legacy_range_eps' in matches:
        parsed['wan_spatial_rope_lambda_init_eps'] = float(matches['legacy_range_eps'])
    if 'fixed_lambda_h' in matches:
        parsed['wan_spatial_rope_lambda_fixed_h'] = float(matches['fixed_lambda_h'])
        parsed['wan_spatial_rope_lambda_parametrization'] = 'fixed'
    if 'fixed_lambda_w' in matches:
        parsed['wan_spatial_rope_lambda_fixed_w'] = float(matches['fixed_lambda_w'])
        parsed['wan_spatial_rope_lambda_parametrization'] = 'fixed'
    if 'lambda_global' in matches:
        value = str(matches['lambda_global']).strip().lower()
        parsed['wan_spatial_rope_lambda_global'] = value in {'1', 'true', 'yes', 'y'}

    return parsed


def collect_inferred_training_args(model_path: str | None, training_args_path: Path | None) -> dict:
    inferred = {}
    if training_args_path is not None:
        inferred.update(load_json(training_args_path))

    if model_path:
        path = Path(model_path)
        experiment_name = path.name if path.is_dir() else path.parent.name
        parsed_from_name = parse_experiment_name(experiment_name)
        for key, value in parsed_from_name.items():
            if key not in inferred or inferred.get(key) in (None, '', False):
                inferred[key] = value
    return inferred


def resolve_model_artifacts(model_path: str | None):
    result = {
        'training_args_path': None,
        'main_checkpoint_path': None,
        'lambda_checkpoint_path': None,
    }
    if not model_path:
        return result

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f'model_path does not exist: {model_path}')

    if path.is_dir():
        training_args_path = path / 'training_args.json'
        if training_args_path.exists():
            result['training_args_path'] = training_args_path
        main_candidates = [
            p for p in path.glob('*.safetensors')
            if not p.name.endswith('_lambda.safetensors')
        ]
        lambda_candidates = [p for p in path.glob('*_lambda.safetensors')]
        if lambda_candidates:
            result['lambda_checkpoint_path'] = sort_checkpoint_paths(lambda_candidates)[-1]
        if main_candidates:
            result['main_checkpoint_path'] = sort_checkpoint_paths(main_candidates)[-1]
    else:
        if path.name.endswith('_lambda.safetensors'):
            result['lambda_checkpoint_path'] = path
        else:
            result['main_checkpoint_path'] = path
            sibling_lambda = path.with_name(path.stem + '_lambda' + path.suffix)
            if sibling_lambda.exists():
                result['lambda_checkpoint_path'] = sibling_lambda
        training_args_path = path.parent / 'training_args.json'
        if training_args_path.exists():
            result['training_args_path'] = training_args_path
    return result


def infer_model_name(model_type: str, model_path: str | None) -> str:
    if not model_path:
        return 'wan2.1_1B3' if model_type in ('1.3B', '1B3') else 'wan2.1_14B'
    path = Path(model_path)
    if path.is_dir():
        return path.name
    return path.parent.name


def read_jsonl(path: str) -> list[PromptSample]:
    samples = []
    seen_ids = set()
    with open(path, 'r', encoding='utf-8') as handle:
        for line_id, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if 'id' not in record or 'prompt' not in record:
                raise ValueError(f'Line {line_id} in {path} must contain fields `id` and `prompt`.')
            sample_id = str(record['id'])
            if not sample_id:
                raise ValueError(f'Line {line_id} has an empty `id`.')
            if sample_id in seen_ids:
                raise ValueError(f'Duplicated sample id detected: {sample_id}')
            if '/' in sample_id or os.sep in sample_id:
                raise ValueError(f'Sample id `{sample_id}` contains path separators, which is not allowed.')
            samples.append(
                PromptSample(
                    sample_id=sample_id,
                    prompt=str(record['prompt']),
                    negative_prompt=None if 'negative_prompt' not in record else str(record['negative_prompt']),
                    seed=None if 'seed' not in record else int(record['seed']),
                    raw_record=dict(record),
                )
            )
            seen_ids.add(sample_id)
    return samples


def write_jsonl(path: Path, samples: list[PromptSample]) -> None:
    with open(path, 'w', encoding='utf-8') as handle:
        for sample in samples:
            payload = dict(sample.raw_record or {})
            payload['id'] = sample.sample_id
            payload['prompt'] = sample.prompt
            if sample.negative_prompt is not None:
                payload['negative_prompt'] = sample.negative_prompt
            elif 'negative_prompt' in payload:
                payload.pop('negative_prompt', None)
            if sample.seed is not None:
                payload['seed'] = sample.seed
            elif 'seed' in payload:
                payload.pop('seed', None)
            handle.write(json.dumps(payload, ensure_ascii=False) + '\n')


def write_result_jsonl(path: Path, samples: list[PromptSample], output_dir: Path) -> None:
    with open(path, 'w', encoding='utf-8') as handle:
        for sample in samples:
            video_path = (output_dir / f'{sample.sample_id}.mp4').resolve()
            if not video_path.exists():
                raise FileNotFoundError(f'Expected generated video does not exist: {video_path}')
            payload = dict(sample.raw_record or {})
            payload['id'] = sample.sample_id
            payload['prompt'] = sample.prompt
            if sample.negative_prompt is not None:
                payload['negative_prompt'] = sample.negative_prompt
            if sample.seed is not None:
                payload['seed'] = sample.seed
            payload['video_url'] = str(video_path)
            handle.write(json.dumps(payload, ensure_ascii=False) + '\n')


class BatchWanT2V(wan.WanT2V):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_loaded = False

    def _build_scheduler(self, sample_solver: str, sampling_steps: int, shift: float):
        if sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(sample_scheduler, device=self.device, sigmas=sampling_sigmas)
        else:
            raise NotImplementedError(f'Unsupported solver: {sample_solver}')
        return sample_scheduler, timesteps

    def _encode_prompts(self, prompts: list[str], negative_prompts: list[str], offload_model: bool):
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder(prompts, self.device)
            context_null = self.text_encoder(negative_prompts, self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            cpu_device = torch.device('cpu')
            context = self.text_encoder(prompts, cpu_device)
            context_null = self.text_encoder(negative_prompts, cpu_device)
            context = [tensor.to(self.device) for tensor in context]
            context_null = [tensor.to(self.device) for tensor in context_null]
        return context, context_null

    def _make_noises(self, target_shape: tuple[int, int, int, int], seeds: list[int]) -> list[torch.Tensor]:
        noises = []
        for seed in seeds:
            seed_g = torch.Generator(device=self.device)
            seed_g.manual_seed(int(seed))
            noises.append(
                torch.randn(
                    target_shape[0],
                    target_shape[1],
                    target_shape[2],
                    target_shape[3],
                    dtype=torch.float32,
                    device=self.device,
                    generator=seed_g,
                )
            )
        return noises

    def load_optional_checkpoint(
        self,
        model_path: str | None,
        spatial_rope_lambda_enabled: bool = False,
        spatial_rope_lambda_scope: str | None = None,
        spatial_rope_lambda_timestep_conditioned: bool | None = None,
        spatial_rope_lambda_hidden_dim: int | None = None,
        spatial_rope_lambda_checkpoint: str | None = None,
        spatial_rope_lambda_fixed_h: float | None = None,
        spatial_rope_lambda_fixed_w: float | None = None,
        lora_target_modules: str | None = None,
        lora_rank: int | None = None,
        lora_alpha: int | None = None,
    ):
        artifacts = resolve_model_artifacts(model_path)
        training_args = collect_inferred_training_args(model_path, artifacts['training_args_path'])

        lambda_enabled = bool(spatial_rope_lambda_enabled)
        if not lambda_enabled:
            lambda_enabled = bool(training_args.get('wan_spatial_rope_lambda_enabled', False))
        if not lambda_enabled:
            lambda_enabled = bool(training_args.get('lambda_enabled', False))
        if spatial_rope_lambda_checkpoint:
            lambda_enabled = True
        if artifacts['lambda_checkpoint_path'] is not None:
            lambda_enabled = True

        lambda_scope = spatial_rope_lambda_scope or training_args.get('wan_spatial_rope_lambda_scope', 'layer')
        lambda_tcond = spatial_rope_lambda_timestep_conditioned
        if lambda_tcond is None:
            lambda_tcond = training_args.get('wan_spatial_rope_lambda_timestep_conditioned')
        if lambda_tcond is None:
            lambda_tcond = False
        lambda_hidden_dim = spatial_rope_lambda_hidden_dim or int(training_args.get('wan_spatial_rope_lambda_hidden_dim', 128))
        lambda_parametrization = str(training_args.get('wan_spatial_rope_lambda_parametrization') or 'unconstrained')
        lambda_min = float(training_args.get('wan_spatial_rope_lambda_min', 0.5))
        lambda_init_eps = float(training_args.get('wan_spatial_rope_lambda_init_eps', 1e-4))
        lambda_fixed_h = float(
            spatial_rope_lambda_fixed_h if spatial_rope_lambda_fixed_h is not None
            else training_args.get('wan_spatial_rope_lambda_fixed_h', 1.0)
        )
        lambda_fixed_w = float(
            spatial_rope_lambda_fixed_w if spatial_rope_lambda_fixed_w is not None
            else training_args.get('wan_spatial_rope_lambda_fixed_w', 1.0)
        )
        manual_fixed_lambda = spatial_rope_lambda_fixed_h is not None or spatial_rope_lambda_fixed_w is not None
        if manual_fixed_lambda:
            lambda_enabled = True
            if lambda_parametrization != 'fixed':
                logging.warning(
                    'Manual fixed lambda override was provided; forcing spatial RoPE lambda parametrization to fixed at inference time.'
                )
                lambda_parametrization = 'fixed'
        lambda_checkpoint_path = spatial_rope_lambda_checkpoint or artifacts['lambda_checkpoint_path']
        resolved_lora_target_modules = parse_lora_target_modules(lora_target_modules)
        if resolved_lora_target_modules is None:
            resolved_lora_target_modules = parse_lora_target_modules(training_args.get('lora_target_modules'))
        if resolved_lora_target_modules is None:
            resolved_lora_target_modules = training_args.get('lora_target_modules')
        if isinstance(resolved_lora_target_modules, str):
            resolved_lora_target_modules = parse_lora_target_modules(resolved_lora_target_modules)
        if resolved_lora_target_modules is None:
            resolved_lora_target_modules = preset_to_lora_target_modules(training_args.get('lora_modules'))
        raw_lora_rank = lora_rank if lora_rank is not None else training_args.get('lora_rank')
        raw_lora_alpha = lora_alpha if lora_alpha is not None else training_args.get('lora_alpha')
        resolved_lora_rank = None if raw_lora_rank is None else int(raw_lora_rank)
        resolved_lora_alpha = None if raw_lora_alpha is None else int(raw_lora_alpha)

        main_checkpoint_path = artifacts['main_checkpoint_path']
        if main_checkpoint_path is not None:
            state = load_state_dict(str(main_checkpoint_path))
            lora_state = {key: value for key, value in state.items() if 'lora_A' in key or 'lora_B' in key}
            if lora_state:
                if resolved_lora_target_modules is None:
                    raise RuntimeError(
                        'Detected LoRA weights in the checkpoint, but no lora_target_modules were provided and training_args.json does not contain them.'
                    )
                if resolved_lora_rank is None:
                    resolved_lora_rank = 32
                if resolved_lora_alpha is None:
                    resolved_lora_alpha = resolved_lora_rank
                lora_config = LoraConfig(
                    r=resolved_lora_rank,
                    lora_alpha=resolved_lora_alpha,
                    target_modules=resolved_lora_target_modules,
                )
                self.model = inject_adapter_in_model(lora_config, self.model)
                mapped_lora_state = mapping_lora_state_dict(lora_state)
                missing, unexpected = self.model.load_state_dict(mapped_lora_state, strict=False)
                logging.info(
                    'Loaded LoRA checkpoint %s with %d keys, %d missing, and %d unexpected.',
                    main_checkpoint_path,
                    len(mapped_lora_state),
                    len(missing),
                    len(unexpected),
                )
            dit_state = {
                key: value
                for key, value in state.items()
                if not key.startswith('spatial_rope_lambda.') and 'lora_A' not in key and 'lora_B' not in key
            }
            if dit_state:
                try:
                    current_state = self.model.state_dict()
                    matched = sum(1 for key in dit_state if key in current_state and current_state[key].shape == dit_state[key].shape)
                    missing, unexpected = self.model.load_state_dict(dit_state, strict=False)
                except RuntimeError as error:
                    raise RuntimeError(
                        f'Failed to load model checkpoint from {main_checkpoint_path}. '
                        'This usually means the checkpoint structure is not compatible with the official Wan model.'
                    ) from error
                if matched == 0:
                    raise RuntimeError(
                        f'No compatible parameters were found when loading {main_checkpoint_path} into the official Wan model.'
                    )
                logging.info(
                    'Loaded model checkpoint %s with %d matched keys, %d missing keys, and %d unexpected keys.',
                    main_checkpoint_path,
                    matched,
                    len(missing),
                    len(unexpected),
                )
                if lambda_checkpoint_path is None and any(key.startswith('spatial_rope_lambda.') for key in state):
                    lambda_checkpoint_path = str(main_checkpoint_path)
                    lambda_enabled = True

        if lambda_enabled:
            install_official_wan_lambda_patch()
            module = WanSpatialRopeLambda(
                num_layers=len(self.model.blocks),
                num_heads=self.model.blocks[0].self_attn.num_heads,
                freq_dim=self.model.freq_dim,
                scope=lambda_scope,
                learn_f=False,
                timestep_conditioned=bool(lambda_tcond),
                hidden_dim=int(lambda_hidden_dim),
                parametrization=lambda_parametrization,
                lambda_min=lambda_min,
                init_eps=lambda_init_eps,
                fixed_h=lambda_fixed_h,
                fixed_w=lambda_fixed_w,
            ).to(device=self.device)
            self.model.spatial_rope_lambda = module
            if lambda_checkpoint_path:
                lambda_state = load_state_dict(str(lambda_checkpoint_path))
                cleaned = {}
                for key, value in lambda_state.items():
                    if key.startswith('pipe.dit.spatial_rope_lambda.'):
                        key = key[len('pipe.dit.spatial_rope_lambda.'):]
                    elif key.startswith('spatial_rope_lambda.'):
                        key = key[len('spatial_rope_lambda.'):]
                    else:
                        continue
                    cleaned[key] = value
                if not cleaned:
                    raise RuntimeError(
                        f'Lambda checkpoint {lambda_checkpoint_path} does not contain spatial_rope_lambda keys.'
                    )
                missing, unexpected = module.load_state_dict(cleaned, strict=False)
                logging.info(
                    'Loaded lambda checkpoint %s with %d keys, %d missing, %d unexpected.',
                    lambda_checkpoint_path,
                    len(cleaned),
                    len(missing),
                    len(unexpected),
                )
            else:
                if lambda_parametrization == 'fixed':
                    logging.info(
                        'Enabled fixed spatial RoPE lambda without a separate lambda checkpoint: scope=%s, fixed_h=%.6f, fixed_w=%.6f.',
                        lambda_scope,
                        lambda_fixed_h,
                        lambda_fixed_w,
                    )
                else:
                    logging.warning(
                        'Spatial RoPE lambda was enabled but no checkpoint was provided. The lambda module stays at initialization: parametrization=%s, scope=%s, timestep_conditioned=%s.',
                        lambda_parametrization,
                        lambda_scope,
                        bool(lambda_tcond),
                    )
            self.lambda_loaded = True

    def generate_batch(
        self,
        prompts: list[str],
        size: tuple[int, int] = (1280, 720),
        frame_num: int = 81,
        shift: float = 5.0,
        sample_solver: str = 'unipc',
        sampling_steps: int = 50,
        guide_scale: float = 5.0,
        negative_prompts: list[str] | None = None,
        seeds: list[int] | None = None,
        offload_model: bool = True,
        show_progress: bool = True,
        spatial_rope_lambda_steps: set[int] | None = None,
    ) -> list[torch.Tensor]:
        if not prompts:
            return []

        batch_size = len(prompts)
        if negative_prompts is None:
            negative_prompts = [self.sample_neg_prompt] * batch_size
        elif len(negative_prompts) != batch_size:
            raise ValueError(f'negative_prompts length mismatch: expected {batch_size}, got {len(negative_prompts)}.')

        if seeds is None:
            seeds = [random.randint(0, sys.maxsize) for _ in range(batch_size)]
        elif len(seeds) != batch_size:
            raise ValueError(f'seeds length mismatch: expected {batch_size}, got {len(seeds)}.')

        target_shape = (
            self.vae.model.z_dim,
            (frame_num - 1) // self.vae_stride[0] + 1,
            size[1] // self.vae_stride[1],
            size[0] // self.vae_stride[2],
        )
        seq_len = math.ceil(
            (target_shape[2] * target_shape[3])
            / (self.patch_size[1] * self.patch_size[2])
            * target_shape[1]
            / self.sp_size
        ) * self.sp_size

        context, context_null = self._encode_prompts(prompts, negative_prompts, offload_model=offload_model)
        latents = self._make_noises(target_shape, seeds)

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)
        sample_scheduler, timesteps = self._build_scheduler(sample_solver=sample_solver, sampling_steps=sampling_steps, shift=shift)
        arg_c = {'context': context, 'seq_len': seq_len}
        arg_null = {'context': context_null, 'seq_len': seq_len}

        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
            self.model.to(self.device)
            progress = tqdm(timesteps, disable=not show_progress, desc='Sampling')
            for progress_id, t in enumerate(progress, start=1):
                lambda_enabled_for_step = spatial_rope_lambda_steps is None or progress_id in spatial_rope_lambda_steps
                setattr(self.model, '_wan_eval_lambda_enabled_for_step', lambda_enabled_for_step)
                timestep = torch.stack([t] * batch_size)
                noise_pred_cond = self.model(latents, t=timestep, **arg_c)
                noise_pred_uncond = self.model(latents, t=timestep, **arg_null)
                noise_pred = torch.stack(
                    [
                        pred_uncond + guide_scale * (pred_cond - pred_uncond)
                        for pred_cond, pred_uncond in zip(noise_pred_cond, noise_pred_uncond)
                    ],
                    dim=0,
                )
                latent_batch = torch.stack(latents, dim=0)
                next_latent_batch = sample_scheduler.step(noise_pred, t, latent_batch, return_dict=False)[0]
                latents = [next_latent_batch[idx] for idx in range(batch_size)]

            if offload_model:
                self.model.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            setattr(self.model, '_wan_eval_lambda_enabled_for_step', True)
            videos = self.vae.decode(latents)

        del sample_scheduler
        if offload_model and torch.cuda.is_available():
            torch.cuda.synchronize()
        return videos


def iter_batches(items: list[PromptSample], batch_size: int) -> Iterable[list[PromptSample]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def save_videos(videos: list[torch.Tensor], batch: list[PromptSample], output_dir: Path, fps: int):
    for sample, video in zip(batch, videos):
        save_path = output_dir / f'{sample.sample_id}.mp4'
        cached_path = cache_video(
            tensor=video[None],
            save_file=str(save_path),
            fps=fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        if cached_path is None:
            raise RuntimeError(f'Failed to save video to {save_path}')


def split_samples_for_gpus(samples: list[PromptSample], num_gpus: int) -> list[list[PromptSample]]:
    shard_size = math.ceil(len(samples) / num_gpus)
    return [samples[i * shard_size : (i + 1) * shard_size] for i in range(num_gpus)]


def run_worker(args, samples: list[PromptSample]) -> None:
    logging.info('Worker starting on visible CUDA devices=%s with %d samples.', os.environ.get('CUDA_VISIBLE_DEVICES', ''), len(samples))
    cfg = WAN_CONFIGS[args.task]
    pipeline = BatchWanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=args.device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=args.t5_cpu,
    )
    pipeline.load_optional_checkpoint(
        model_path=args.model_path,
        spatial_rope_lambda_enabled=args.spatial_rope_lambda_enabled,
        spatial_rope_lambda_scope=args.spatial_rope_lambda_scope,
        spatial_rope_lambda_timestep_conditioned=args.spatial_rope_lambda_timestep_conditioned,
        spatial_rope_lambda_hidden_dim=args.spatial_rope_lambda_hidden_dim,
        spatial_rope_lambda_checkpoint=args.spatial_rope_lambda_checkpoint,
        spatial_rope_lambda_fixed_h=args.spatial_rope_lambda_fixed_h,
        spatial_rope_lambda_fixed_w=args.spatial_rope_lambda_fixed_w,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )

    size = SIZE_CONFIGS[args.size]
    lambda_steps = parse_step_list(args.spatial_rope_lambda_steps)
    inferred_for_lambda_steps = collect_inferred_training_args(
        args.model_path,
        resolve_model_artifacts(args.model_path)['training_args_path'] if args.model_path else None,
    ) if args.model_path else {}
    lambda_global = inferred_for_lambda_steps.get('wan_spatial_rope_lambda_global')
    if lambda_global is True:
        lambda_steps = None
        args.spatial_rope_lambda_steps = ''
        logging.info('Checkpoint name/config requests lambda_global=True; spatial RoPE lambda is forced active on all denoising steps.')
    elif lambda_global is False:
        logging.info('Checkpoint name/config requests lambda_global=False; respecting --spatial_rope_lambda_steps.')
    if lambda_steps is None:
        logging.info('Spatial RoPE lambda is active on all denoising steps.')
    else:
        logging.info(
            'Spatial RoPE lambda is restricted to denoising steps (1-based): %s',
            ','.join(str(step) for step in sorted(lambda_steps)),
        )
    total_batches = math.ceil(len(samples) / args.batch_size)
    for batch_id, batch in enumerate(iter_batches(samples, args.batch_size), start=1):
        prompts = [sample.prompt for sample in batch]
        negative_prompts = [
            sample.negative_prompt if sample.negative_prompt is not None else (args.negative_prompt if args.negative_prompt else pipeline.sample_neg_prompt)
            for sample in batch
        ]
        seeds = [
            sample.seed if sample.seed is not None else args.base_seed
            for sample in batch
        ]
        logging.info('Running batch %d/%d with %d samples.', batch_id, total_batches, len(batch))
        videos = pipeline.generate_batch(
            prompts=prompts,
            size=size,
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            negative_prompts=negative_prompts,
            seeds=seeds,
            offload_model=args.offload_model,
            show_progress=not args.disable_tqdm,
            spatial_rope_lambda_steps=lambda_steps,
        )
        save_videos(videos, batch, Path(args.output_dir), fps=cfg.sample_fps)
        logging.info('Saved batch %d outputs to %s', batch_id, args.output_dir)


def launch_multi_gpu_workers(args, samples: list[PromptSample]) -> None:
    gpu_ids = [gpu_id.strip() for gpu_id in args.gpu_ids.split(',') if gpu_id.strip()]
    if len(gpu_ids) <= 1:
        return run_worker(args, samples)

    shard_dir = Path(args.output_dir) / '.worker_shards'
    shard_dir.mkdir(parents=True, exist_ok=True)
    sample_shards = [shard for shard in split_samples_for_gpus(samples, len(gpu_ids)) if shard]
    processes = []
    for worker_id, (gpu_id, shard_samples) in enumerate(zip(gpu_ids, sample_shards)):
        shard_path = shard_dir / f'shard_{worker_id:02d}.jsonl'
        write_jsonl(shard_path, shard_samples)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            '--input_jsonl', str(shard_path),
            '--output_dir', args.output_dir,
            '--ckpt_dir', args.ckpt_dir,
            '--task', args.task,
            '--size', args.size,
            '--frame_num', str(args.frame_num),
            '--sample_solver', args.sample_solver,
            '--sample_steps', str(args.sample_steps),
            '--sample_shift', str(args.sample_shift),
            '--sample_guide_scale', str(args.sample_guide_scale),
            '--batch_size', str(args.batch_size),
            '--base_seed', str(args.base_seed),
            '--device_id', '0',
            '--worker_mode',
            '--disable_tqdm',
        ]
        if args.offload_model:
            cmd.append('--offload_model')
        else:
            cmd.append('--no_offload_model')
        if args.t5_cpu:
            cmd.append('--t5_cpu')
        if args.skip_existing:
            cmd.append('--skip_existing')
        if args.model_path:
            cmd.extend(['--model_path', args.model_path])
        if args.negative_prompt:
            cmd.extend(['--negative_prompt', args.negative_prompt])
        if args.spatial_rope_lambda_enabled:
            cmd.append('--spatial_rope_lambda_enabled')
        if args.spatial_rope_lambda_scope:
            cmd.extend(['--spatial_rope_lambda_scope', args.spatial_rope_lambda_scope])
        if args.spatial_rope_lambda_timestep_conditioned:
            cmd.append('--spatial_rope_lambda_timestep_conditioned')
        if args.spatial_rope_lambda_hidden_dim is not None:
            cmd.extend(['--spatial_rope_lambda_hidden_dim', str(args.spatial_rope_lambda_hidden_dim)])
        if args.spatial_rope_lambda_checkpoint:
            cmd.extend(['--spatial_rope_lambda_checkpoint', args.spatial_rope_lambda_checkpoint])
        if args.spatial_rope_lambda_steps:
            cmd.extend(['--spatial_rope_lambda_steps', args.spatial_rope_lambda_steps])
        if args.lora_target_modules:
            cmd.extend(['--lora_target_modules', args.lora_target_modules])
        if args.lora_rank is not None:
            cmd.extend(['--lora_rank', str(args.lora_rank)])
        if args.lora_alpha is not None:
            cmd.extend(['--lora_alpha', str(args.lora_alpha)])

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = gpu_id
        env['WAN_EVAL_WORKER_ID'] = str(worker_id)
        logging.info('Launching worker %d on physical GPU %s with %d samples.', worker_id, gpu_id, len(shard_samples))
        processes.append(subprocess.Popen(cmd, env=env))

    exit_codes = [proc.wait() for proc in processes]
    if any(code != 0 for code in exit_codes):
        raise RuntimeError(f'Multi-GPU inference failed. Worker exit codes: {exit_codes}')


def parse_args():
    parser = argparse.ArgumentParser(description='Batch and multi-GPU inference for official Wan2.1 T2V checkpoints.')
    parser.add_argument('--input_jsonl', type=str, required=True, help='JSONL file with required fields `id` and `prompt`.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to store generated videos as {id}.mp4.')
    parser.add_argument('--ckpt_dir', type=str, required=True, help='Official Wan checkpoint directory.')
    parser.add_argument('--task', type=str, default='t2v-1.3B', choices=('t2v-1.3B', 't2v-14B'), help='Official Wan task identifier.')
    parser.add_argument('--size', type=str, default='832*480', help='Video size key from official Wan SIZE_CONFIGS.')
    parser.add_argument('--frame_num', type=int, default=81, help='Number of frames. Should satisfy 4n+1.')
    parser.add_argument('--sample_solver', type=str, default='unipc', choices=('unipc', 'dpm++'), help='Official Wan sampling solver.')
    parser.add_argument('--sample_steps', type=int, default=50, help='Number of sampling steps.')
    parser.add_argument('--sample_shift', type=float, default=5.0, help='Official Wan scheduler shift parameter.')
    parser.add_argument('--sample_guide_scale', type=float, default=5.0, help='Classifier-free guidance scale.')
    parser.add_argument('--negative_prompt', type=str, default='', help='Global negative prompt. Empty string falls back to the model default.')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of prompts to denoise in parallel on each GPU worker.')
    parser.add_argument('--base_seed', type=int, default=0, help='Fallback base seed when a JSONL row does not provide `seed`.')
    parser.add_argument('--device_id', type=int, default=0, help='Visible CUDA device index used by the current worker process.')
    parser.add_argument('--gpu_ids', type=str, default='', help='Comma-separated physical GPU ids. If multiple ids are given, the script shards the JSONL and launches one worker per GPU.')
    parser.add_argument('--offload_model', dest='offload_model', action='store_true', default=True, help='Offload text encoder / DiT following the official inference style.')
    parser.add_argument('--no_offload_model', dest='offload_model', action='store_false', help='Keep text encoder / DiT on GPU during inference.')
    parser.add_argument('--t5_cpu', action='store_true', help='Place the T5 text encoder on CPU.')
    parser.add_argument('--skip_existing', action='store_true', help='Skip samples whose {id}.mp4 already exists in output_dir.')
    parser.add_argument('--model_path', type=str, default='', help='Optional fine-tuned checkpoint path or experiment directory. Empty means the original Wan checkpoint only.')
    parser.add_argument('--lora_target_modules', type=str, default='', help='Optional LoRA target modules, for example `q,k,v,o`. When omitted, the script tries to infer them from training_args.json.')
    parser.add_argument('--lora_rank', type=int, default=None, help='Optional LoRA rank. When omitted, the script tries to infer it from training_args.json.')
    parser.add_argument('--lora_alpha', type=int, default=None, help='Optional LoRA alpha. When omitted, the script tries to infer it from training_args.json.')
    parser.add_argument('--spatial_rope_lambda_enabled', action='store_true', help='Enable spatial RoPE lambda patching for official Wan inference.')
    parser.add_argument('--spatial_rope_lambda_scope', type=str, default='', choices=('', 'model', 'layer', 'head'), help='Sharing scope for spatial RoPE lambda parameters. Empty means auto-infer from training_args.json when possible.')
    parser.add_argument('--spatial_rope_lambda_timestep_conditioned', action='store_true', default=None, help='Enable timestep-conditioned lambda MLP g(e_t).')
    parser.add_argument('--spatial_rope_lambda_hidden_dim', type=int, default=None, help='Hidden dimension of the timestep-conditioned lambda MLP.')
    parser.add_argument('--spatial_rope_lambda_checkpoint', type=str, default='', help='Optional explicit lambda checkpoint path. When omitted, the script tries to infer it from model_path.')
    parser.add_argument('--spatial_rope_lambda_fixed_h', type=float, default=None, help='Optional manual override for fixed-lambda height scale at inference.')
    parser.add_argument('--spatial_rope_lambda_fixed_w', type=float, default=None, help='Optional manual override for fixed-lambda width scale at inference.')
    parser.add_argument('--spatial_rope_lambda_steps', type=str, default='', help='Optional comma-separated 1-based denoising step indices where lambda stays active, for example `1,2,3,4,5`. Empty means enable lambda on every denoising step.')
    parser.add_argument('--worker_mode', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--disable_tqdm', action='store_true', help=argparse.SUPPRESS)
    return parser.parse_args()


def validate_args(args):
    if args.task not in WAN_CONFIGS:
        raise ValueError(f'Unsupported task: {args.task}')
    if args.task not in SUPPORTED_SIZES:
        raise ValueError(f'Task {args.task} has no declared size table.')
    if args.size not in SUPPORTED_SIZES[args.task]:
        supported_sizes = ', '.join(SUPPORTED_SIZES[args.task])
        raise ValueError(f'Unsupported size {args.size} for task {args.task}. Supported sizes: {supported_sizes}')
    if args.frame_num <= 0 or (args.frame_num - 1) % 4 != 0:
        raise ValueError(f'frame_num must be positive and satisfy 4n+1, got {args.frame_num}.')
    if args.batch_size <= 0:
        raise ValueError(f'batch_size must be positive, got {args.batch_size}.')
    if args.spatial_rope_lambda_fixed_h is not None and args.spatial_rope_lambda_fixed_h <= 0:
        raise ValueError(f'spatial_rope_lambda_fixed_h must be positive, got {args.spatial_rope_lambda_fixed_h}.')
    if args.spatial_rope_lambda_fixed_w is not None and args.spatial_rope_lambda_fixed_w <= 0:
        raise ValueError(f'spatial_rope_lambda_fixed_w must be positive, got {args.spatial_rope_lambda_fixed_w}.')
    parse_step_list(args.spatial_rope_lambda_steps)
    if args.gpu_ids:
        gpu_ids = [gpu_id.strip() for gpu_id in args.gpu_ids.split(',') if gpu_id.strip()]
        if not gpu_ids:
            raise ValueError('gpu_ids was provided but no valid GPU ids were parsed.')


def main():
    args = parse_args()
    validate_args(args)
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = read_jsonl(str(input_path))
    samples = all_samples
    if args.skip_existing:
        samples = [sample for sample in all_samples if not (output_dir / f'{sample.sample_id}.mp4').exists()]
    logging.info('Loaded %d samples from %s', len(all_samples), input_path)
    if samples:
        if args.gpu_ids and not args.worker_mode and len([gpu_id for gpu_id in args.gpu_ids.split(',') if gpu_id.strip()]) > 1:
            launch_multi_gpu_workers(args, samples)
        else:
            run_worker(args, samples)
    else:
        logging.info('No samples need generation. Skip model inference and only export metadata JSONL.')

    if not args.worker_mode:
        result_jsonl_path = output_dir / f'{input_path.stem}_with_video_url.jsonl'
        write_result_jsonl(result_jsonl_path, all_samples, output_dir)
        logging.info('Wrote result JSONL with video_url to %s', result_jsonl_path)
    logging.info('Finished batch inference. Outputs are stored in %s', output_dir)


if __name__ == '__main__':
    main()
