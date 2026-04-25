"""Wan2.1-T2V experiment: self_attention_temporal_kernel.

Main entries:
- Wan21T2VSelfAttentionTemporalKernelState
- Wan21T2VSelfAttentionTemporalKernelPatchHandle
- _install_wan21_t2v_self_attention_temporal_kernel_patch
- run_wan21_t2v_self_attention_temporal_kernel

This module perturbs self-attention through temporal output mixing. The
experiment-specific patch state stays local here; runtime helpers are imported
from utils.py.
"""

import csv
import gc
import json
import math
import os
import random
import re
import sys
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F

from .utils import (
    Wan21T2VParallelConfig,
    _broadcast_seed_if_needed,
    _build_wan21_t2v_pipeline,
    _dedup_wan21_t2v_int_list,
    _ensure_dir,
    _generate_wan21_t2v_video,
    _init_wan21_t2v_runtime,
    _parse_wan21_t2v_layer_head_specs,
    _resolve_wan21_t2v_offload_model,
    _resolve_wan21_t2v_steps,
    _save_csv,
    _save_json,
    _save_wan21_t2v_video,
    _unwrap_wan21_t2v_dit_model_for_runtime_patch,
    _wan21_t2v_branch_matches,
)

from projects.Wan2_1.wan.modules.attention import flash_attention
from projects.Wan2_1.wan.modules.model import rope_apply

class Wan21T2VSelfAttentionTemporalKernelState:
    """Runtime state for self-attention interventions.

    Supported modes:
    - `postoutput_same_position_kernel`:
      keep Wan self-attention unchanged, then smooth selected head outputs
      along the latent-frame axis at the same spatial position.
    - `prelogit_token_temperature`:
      directly flatten selected heads' token-level attention distributions by
      scaling their q/k logits with an exact softmax temperature before
      flash-attention is evaluated.
    """

    def __init__(
        self,
        intervention_steps: Sequence[int],
        layers: Optional[Sequence[int]],
        layer_head_specs: Optional[Sequence[Tuple[int, int]]],
        branch: str,
        intervention_mode: str,
        kernel_radius: int,
        kernel_sigma: float,
        mix_alpha: float,
        token_temperature: float,
    ):
        if not intervention_steps:
            raise ValueError("intervention_steps must be non-empty.")
        intervention_mode = str(intervention_mode).strip().lower()
        if intervention_mode not in {"postoutput_same_position_kernel", "prelogit_token_temperature"}:
            raise ValueError(
                "intervention_mode must be one of "
                "{'postoutput_same_position_kernel', 'prelogit_token_temperature'}."
            )
        if kernel_radius < 1:
            raise ValueError("kernel_radius must be >= 1.")
        if kernel_sigma <= 0:
            raise ValueError("kernel_sigma must be > 0.")
        if not (0.0 <= float(mix_alpha) <= 1.0):
            raise ValueError("mix_alpha must be in [0, 1].")
        if float(token_temperature) <= 0.0:
            raise ValueError("token_temperature must be > 0.")

        self.intervention_steps = set(int(step) for step in intervention_steps)
        self.layers = None if layers is None else set(int(layer) for layer in layers)
        if layer_head_specs:
            layer_to_heads = defaultdict(set)
            for layer_idx, head_idx in layer_head_specs:
                layer_to_heads[int(layer_idx)].add(int(head_idx))
            self.layer_to_heads = {
                int(layer_idx): tuple(sorted(int(head_idx) for head_idx in head_set))
                for layer_idx, head_set in layer_to_heads.items()
            }
        else:
            self.layer_to_heads = None
        self.branch = str(branch).strip().lower()
        if self.branch not in {"cond", "uncond", "both"}:
            raise ValueError("branch must be one of cond/uncond/both.")
        self.intervention_mode = intervention_mode
        self.kernel_radius = int(kernel_radius)
        self.kernel_sigma = float(kernel_sigma)
        self.mix_alpha = float(mix_alpha)
        self.token_temperature = float(token_temperature)

        self.current_step = 0
        self.current_timestep_value = None
        self.forward_call_index_in_step = 0
        self.total_self_attention_calls = 0
        self.modified_self_attention_calls = 0
        self.modified_by_step_layer: Dict[Tuple[int, int], int] = defaultdict(int)
        self.modified_by_step_layer_head: Dict[Tuple[int, int, int], int] = defaultdict(int)

    def on_forward_start(self, t_tensor):
        """Update the current 1-based diffusion step and CFG forward index."""
        t_value = float(t_tensor.flatten()[0].item()) if t_tensor is not None else None
        if self.current_timestep_value is None or t_value != self.current_timestep_value:
            self.current_step += 1
            self.current_timestep_value = t_value
            self.forward_call_index_in_step = 0
        else:
            self.forward_call_index_in_step += 1

    def should_modify(self, layer_idx: int) -> bool:
        """Return True if this self-attention call should be temporally mixed."""
        if self.current_step not in self.intervention_steps:
            return False
        if self.layers is not None and int(layer_idx) not in self.layers:
            return False
        if not _wan21_t2v_branch_matches(self.branch, self.forward_call_index_in_step):
            return False
        return True

    def heads_for_layer(self, layer_idx: int, num_heads: int) -> Tuple[int, ...]:
        """Return selected head ids for one layer. Empty means no head is modified."""
        if self.layer_to_heads is None:
            return tuple(range(int(num_heads)))
        return tuple(
            head_idx
            for head_idx in self.layer_to_heads.get(int(layer_idx), tuple())
            if 0 <= int(head_idx) < int(num_heads)
        )

    def apply_token_temperature_to_qk(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        selected_heads: Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply exact token-level softmax temperature to selected heads.

        For selected heads, the attention logits become
        `<q, k> / (sqrt(d) * T)` by scaling both q and k with `T^(-1/2)`.
        """
        if self.intervention_mode != "prelogit_token_temperature":
            return query, key
        if not selected_heads:
            return query, key
        if abs(float(self.token_temperature) - 1.0) < 1e-8:
            return query, key

        q = query.clone()
        k = key.clone()
        scale = float(self.token_temperature) ** (-0.5)
        head_index_tensor = torch.tensor(list(selected_heads), device=q.device, dtype=torch.long)
        q[:, :, head_index_tensor, :] = q[:, :, head_index_tensor, :] * scale
        k[:, :, head_index_tensor, :] = k[:, :, head_index_tensor, :] * scale
        return q, k

    def temporal_mix_heads(
        self,
        y_heads: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor,
        selected_heads: Sequence[int],
    ) -> torch.Tensor:
        """Apply same-spatial-position temporal kernel mixing to selected heads in `[B, L, N, D]`."""
        if self.mix_alpha <= 0.0:
            return y_heads
        if not selected_heads:
            return y_heads

        out = y_heads.clone()
        device = y_heads.device
        dtype = y_heads.dtype
        offsets = torch.arange(
            -self.kernel_radius,
            self.kernel_radius + 1,
            device=device,
            dtype=torch.float32,
        )
        kernel = torch.exp(-0.5 * (offsets / float(self.kernel_sigma)).pow(2))
        kernel = kernel / kernel.sum().clamp_min(1e-8)
        selected_head_indices = torch.tensor(list(selected_heads), device=device, dtype=torch.long)

        for batch_index in range(int(y_heads.size(0))):
            seq_len = int(seq_lens[batch_index].item())
            if seq_len <= 0:
                continue
            frame_count, token_grid_height, token_grid_width = [
                int(v) for v in grid_sizes[batch_index].tolist()
            ]
            valid_len = min(seq_len, frame_count * token_grid_height * token_grid_width, y_heads.size(1))
            if valid_len <= 0:
                continue
            if valid_len != frame_count * token_grid_height * token_grid_width:
                continue
            if frame_count <= 1:
                continue

            y_fhwnd = out[batch_index, :valid_len].reshape(
                frame_count,
                token_grid_height,
                token_grid_width,
                y_heads.size(2),
                y_heads.size(3),
            )
            selected_head_values = y_fhwnd[:, :, :, selected_head_indices, :]
            # Conv1d runs on `[channels, frames]`, so each selected
            # (head, spatial, channel) stream is independently smoothed
            # along the latent-frame axis.
            y_chf = selected_head_values.permute(3, 1, 2, 4, 0).reshape(-1, 1, frame_count)
            padded = F.pad(
                y_chf.float(),
                (int(self.kernel_radius), int(self.kernel_radius)),
                mode="replicate",
            )
            smoothed = F.conv1d(
                padded,
                kernel.view(1, 1, -1),
                padding=0,
            ).to(dtype=dtype)
            smoothed_fhwnd = smoothed.reshape(
                len(selected_heads),
                token_grid_height,
                token_grid_width,
                y_heads.size(3),
                frame_count,
            ).permute(4, 1, 2, 0, 3)
            mixed = (1.0 - self.mix_alpha) * selected_head_values + self.mix_alpha * smoothed_fhwnd
            y_fhwnd[:, :, :, selected_head_indices, :] = mixed
            out[batch_index, :valid_len] = y_fhwnd.reshape(
                valid_len,
                y_heads.size(2),
                y_heads.size(3),
            )

        return out

class Wan21T2VSelfAttentionTemporalKernelPatchHandle:
    """Restore handle for self-attention temporal-kernel intervention."""

    def __init__(self, target_model, original_forward, original_self_forwards):
        self.target_model = target_model
        self.original_forward = original_forward
        self.original_self_forwards = original_self_forwards

    def restore(self):
        self.target_model.forward = self.original_forward
        for idx, block in enumerate(self.target_model.blocks):
            block.self_attn.forward = self.original_self_forwards[idx]

def _install_wan21_t2v_self_attention_temporal_kernel_patch(
    model,
    state: Wan21T2VSelfAttentionTemporalKernelState,
) -> Wan21T2VSelfAttentionTemporalKernelPatchHandle:
    """Install self-attention temporal output mixing without editing Wan source."""
    target = _unwrap_wan21_t2v_dit_model_for_runtime_patch(model)
    if not hasattr(target, "blocks"):
        raise RuntimeError("Invalid DiT model for self-attention temporal-kernel patch.")

    original_forward = target.forward

    def patched_dit_forward(this, *args, **kwargs):
        t = kwargs.get("t", None)
        if t is None and len(args) > 1:
            t = args[1]
        state.on_forward_start(t)
        return original_forward(*args, **kwargs)

    target.forward = MethodType(patched_dit_forward, target)

    original_self_forwards = []
    for layer_idx, block in enumerate(target.blocks):
        original_self = block.self_attn.forward
        original_self_forwards.append(original_self)

        def build_patched_self(layer_id: int, orig_self_fn):
            def patched_self(self, x, seq_lens, grid_sizes, freqs):
                state.total_self_attention_calls += 1
                if not state.should_modify(layer_id):
                    return orig_self_fn(x, seq_lens, grid_sizes, freqs)

                batch_size, seq_length = x.shape[:2]
                num_heads, head_dim = self.num_heads, self.head_dim
                selected_heads = state.heads_for_layer(layer_id, num_heads)
                if not selected_heads:
                    return orig_self_fn(x, seq_lens, grid_sizes, freqs)

                query = self.norm_q(self.q(x)).view(batch_size, seq_length, num_heads, head_dim)
                key = self.norm_k(self.k(x)).view(batch_size, seq_length, num_heads, head_dim)
                value = self.v(x).view(batch_size, seq_length, num_heads, head_dim)
                query, key = state.apply_token_temperature_to_qk(
                    query=query,
                    key=key,
                    selected_heads=selected_heads,
                )
                attended_heads = flash_attention(
                    q=rope_apply(query, grid_sizes, freqs),
                    k=rope_apply(key, grid_sizes, freqs),
                    v=value,
                    k_lens=seq_lens,
                    window_size=self.window_size,
                )
                if state.intervention_mode == "postoutput_same_position_kernel":
                    attended_heads = state.temporal_mix_heads(
                        y_heads=attended_heads,
                        seq_lens=seq_lens,
                        grid_sizes=grid_sizes,
                        selected_heads=selected_heads,
                    )
                mixed = self.o(attended_heads.flatten(2))
                state.modified_self_attention_calls += 1
                state.modified_by_step_layer[(int(state.current_step), int(layer_id))] += 1
                for head_idx in selected_heads:
                    state.modified_by_step_layer_head[(int(state.current_step), int(layer_id), int(head_idx))] += 1
                return mixed

            return patched_self

        block.self_attn.forward = MethodType(build_patched_self(layer_idx, original_self), block.self_attn)

    return Wan21T2VSelfAttentionTemporalKernelPatchHandle(
        target_model=target,
        original_forward=original_forward,
        original_self_forwards=original_self_forwards,
    )

def run_wan21_t2v_self_attention_temporal_kernel(
    wan21_root: str,
    ckpt_dir: str,
    output_dir: str,
    prompt: str,
    size: Tuple[int, int],
    task: str = "t2v-14B",
    frame_num: int = 81,
    shift: float = 8.0,
    sample_solver: str = "unipc",
    sampling_steps: int = 50,
    guide_scale: float = 12.0,
    seed: int = 0,
    device_id: Optional[int] = None,
    offload_model: bool = True,
    self_attn_kernel_steps: Sequence[int] = tuple(),
    self_attn_kernel_layers: Sequence[int] = tuple(),
    self_attn_kernel_heads: Sequence[str] = tuple(),
    self_attn_kernel_branch: str = "cond",
    self_attn_temporal_intervention_mode: str = "postoutput_same_position_kernel",
    self_attn_kernel_radius: int = 2,
    self_attn_kernel_sigma: float = 1.0,
    self_attn_kernel_mix_alpha: float = 0.25,
    self_attn_token_temperature: float = 1.0,
    parallel_cfg: Optional[Wan21T2VParallelConfig] = None,
):
    """Generate one video with self-attention temporal output-kernel intervention.

    Inputs:
        self_attn_kernel_steps: 1-based diffusion steps to intervene. Empty
            means all sampling steps.
        self_attn_kernel_layers: DiT layer indices to intervene. Empty means
            all layers.
        self_attn_kernel_heads: optional head specs `LxHy`. Empty means all
            heads inside the selected layers.
        self_attn_kernel_branch: `cond`, `uncond`, or `both` CFG branch.
        self_attn_temporal_intervention_mode:
            `postoutput_same_position_kernel` or `prelogit_token_temperature`.
        self_attn_kernel_radius: temporal radius in latent-token frames.
        self_attn_kernel_sigma: Gaussian-kernel width in latent-token frames.
        self_attn_kernel_mix_alpha: residual mixing coefficient. `0` is no-op;
            `1` fully replaces the self-attention output by the smoothed output.
        self_attn_token_temperature: exact token-level softmax temperature used
            by `prelogit_token_temperature`. `1.0` is no-op.

    Outputs:
        Generated video, intervention call table, and summary JSON.
    """
    parallel_cfg = parallel_cfg or Wan21T2VParallelConfig()
    runtime = _init_wan21_t2v_runtime(parallel_cfg, explicit_device_id=device_id)
    seed = _broadcast_seed_if_needed(seed, runtime)
    if runtime.use_usp:
        raise RuntimeError("self_attention_temporal_kernel currently requires use_usp=False.")

    resolved_steps = _resolve_wan21_t2v_steps(self_attn_kernel_steps, sampling_steps)
    resolved_layers = _dedup_wan21_t2v_int_list(self_attn_kernel_layers)
    resolved_layer_head_specs = _parse_wan21_t2v_layer_head_specs(self_attn_kernel_heads)

    pipeline, cfg = _build_wan21_t2v_pipeline(
        wan21_root=wan21_root,
        ckpt_dir=ckpt_dir,
        task=task,
        runtime=runtime,
        parallel_cfg=parallel_cfg,
    )
    offload_model = _resolve_wan21_t2v_offload_model(runtime, offload_model)
    state = Wan21T2VSelfAttentionTemporalKernelState(
        intervention_steps=resolved_steps,
        layers=resolved_layers if resolved_layers else None,
        layer_head_specs=resolved_layer_head_specs,
        branch=self_attn_kernel_branch,
        intervention_mode=self_attn_temporal_intervention_mode,
        kernel_radius=self_attn_kernel_radius,
        kernel_sigma=self_attn_kernel_sigma,
        mix_alpha=self_attn_kernel_mix_alpha,
        token_temperature=self_attn_token_temperature,
    )
    handle = _install_wan21_t2v_self_attention_temporal_kernel_patch(pipeline.model, state)

    try:
        video = _generate_wan21_t2v_video(
            pipeline=pipeline,
            prompt=prompt,
            size=size,
            frame_num=frame_num,
            shift=shift,
            sample_solver=sample_solver,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            seed=seed,
            offload_model=offload_model,
        )
    finally:
        handle.restore()

    if dist.is_initialized():
        dist.barrier()

    if runtime.rank == 0:
        _ensure_dir(output_dir)
        video_path = os.path.join(
            output_dir,
            f"wan21_t2v_self_attention_temporal_kernel_seed_{int(seed)}.mp4",
        )
        _save_wan21_t2v_video(video, video_path, fps=cfg.sample_fps)
        rows = [
            {
                "step": int(step),
                "layer": int(layer),
                "modified_calls": int(count),
            }
            for (step, layer), count in sorted(state.modified_by_step_layer.items())
        ]
        _save_csv(os.path.join(output_dir, "self_attention_temporal_kernel_calls.csv"), rows)
        head_rows = [
            {
                "step": int(step),
                "layer": int(layer),
                "head": int(head),
                "head_tag": f"L{int(layer)}H{int(head)}",
                "modified_calls": int(count),
            }
            for (step, layer, head), count in sorted(state.modified_by_step_layer_head.items())
        ]
        _save_csv(
            os.path.join(output_dir, "self_attention_temporal_kernel_head_calls.csv"),
            head_rows,
        )
        summary = {
            "experiment": "wan21_t2v_self_attention_temporal_kernel",
            "prompt": prompt,
            "seed": int(seed),
            "video_path": video_path,
            "self_attn_kernel_steps": [int(step) for step in resolved_steps],
            "self_attn_kernel_layers": [int(layer) for layer in resolved_layers],
            "self_attn_kernel_heads_input": [str(spec) for spec in self_attn_kernel_heads],
            "self_attn_kernel_heads_parsed": [
                {
                    "layer": int(layer_idx),
                    "head": int(head_idx),
                    "head_tag": f"L{int(layer_idx)}H{int(head_idx)}",
                }
                for layer_idx, head_idx in resolved_layer_head_specs
            ],
            "self_attn_kernel_branch": str(self_attn_kernel_branch),
            "self_attn_temporal_intervention_mode": str(self_attn_temporal_intervention_mode),
            "self_attn_kernel_radius": int(self_attn_kernel_radius),
            "self_attn_kernel_sigma": float(self_attn_kernel_sigma),
            "self_attn_kernel_mix_alpha": float(self_attn_kernel_mix_alpha),
            "self_attn_token_temperature": float(self_attn_token_temperature),
            "total_self_attention_calls": int(state.total_self_attention_calls),
            "modified_self_attention_calls": int(state.modified_self_attention_calls),
            "modified_by_step_layer": rows,
            "modified_by_step_layer_head_csv": os.path.join(
                output_dir,
                "self_attention_temporal_kernel_head_calls.csv",
            ),
        }
        _save_json(os.path.join(output_dir, "self_attention_temporal_kernel_summary.json"), summary)
        return summary
    return None
