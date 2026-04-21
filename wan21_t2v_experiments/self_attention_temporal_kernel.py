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
    _resolve_wan21_t2v_offload_model,
    _resolve_wan21_t2v_steps,
    _save_csv,
    _save_json,
    _save_wan21_t2v_video,
    _wan21_t2v_branch_matches,
)

class Wan21T2VSelfAttentionTemporalKernelState:
    """Runtime state for self-attention temporal output mixing.

    The experiment keeps Wan's original self-attention computation intact and
    then mixes a fraction of each self-attention output token with outputs from
    nearby frames at the same spatial token position. This is a lightweight
    proxy for injecting a smoother temporal self-attention kernel without
    materializing a full `[num_heads, L, L]` attention matrix.
    """

    def __init__(
        self,
        intervention_steps: Sequence[int],
        layers: Optional[Sequence[int]],
        branch: str,
        kernel_radius: int,
        kernel_sigma: float,
        mix_alpha: float,
    ):
        if not intervention_steps:
            raise ValueError("intervention_steps must be non-empty.")
        if kernel_radius < 1:
            raise ValueError("kernel_radius must be >= 1.")
        if kernel_sigma <= 0:
            raise ValueError("kernel_sigma must be > 0.")
        if not (0.0 <= float(mix_alpha) <= 1.0):
            raise ValueError("mix_alpha must be in [0, 1].")

        self.intervention_steps = set(int(step) for step in intervention_steps)
        self.layers = None if layers is None else set(int(layer) for layer in layers)
        self.branch = str(branch).strip().lower()
        if self.branch not in {"cond", "uncond", "both"}:
            raise ValueError("branch must be one of cond/uncond/both.")
        self.kernel_radius = int(kernel_radius)
        self.kernel_sigma = float(kernel_sigma)
        self.mix_alpha = float(mix_alpha)

        self.current_step = 0
        self.current_timestep_value = None
        self.forward_call_index_in_step = 0
        self.total_self_attention_calls = 0
        self.modified_self_attention_calls = 0
        self.modified_by_step_layer: Dict[Tuple[int, int], int] = defaultdict(int)

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

    def temporal_mix(self, y: torch.Tensor, seq_lens: torch.Tensor, grid_sizes: torch.Tensor) -> torch.Tensor:
        """Apply same-spatial-position temporal kernel mixing to `[B, L, C]` outputs."""
        if self.mix_alpha <= 0.0:
            return y

        out = y.clone()
        device = y.device
        dtype = y.dtype
        offsets = torch.arange(
            -self.kernel_radius,
            self.kernel_radius + 1,
            device=device,
            dtype=torch.float32,
        )
        kernel = torch.exp(-0.5 * (offsets / float(self.kernel_sigma)).pow(2))
        kernel = kernel / kernel.sum().clamp_min(1e-8)

        for batch_index in range(int(y.size(0))):
            seq_len = int(seq_lens[batch_index].item())
            if seq_len <= 0:
                continue
            frame_count, token_grid_height, token_grid_width = [
                int(v) for v in grid_sizes[batch_index].tolist()
            ]
            valid_len = min(seq_len, frame_count * token_grid_height * token_grid_width, y.size(1))
            if valid_len <= 0:
                continue
            if valid_len != frame_count * token_grid_height * token_grid_width:
                continue

            y_fhwc = y[batch_index, :valid_len].reshape(
                frame_count,
                token_grid_height,
                token_grid_width,
                y.size(-1),
            )
            # Conv1d runs on `[channels, frames]`, so each spatial/channel
            # stream is independently smoothed along the frame axis.
            y_chf = y_fhwc.permute(1, 2, 3, 0).reshape(-1, 1, frame_count)
            smoothed = F.conv1d(
                y_chf.float(),
                kernel.view(1, 1, -1),
                padding=int(self.kernel_radius),
            ).to(dtype=dtype)
            smoothed_fhwc = smoothed.reshape(
                token_grid_height,
                token_grid_width,
                y.size(-1),
                frame_count,
            ).permute(3, 0, 1, 2)
            mixed = (1.0 - self.mix_alpha) * y_fhwc + self.mix_alpha * smoothed_fhwc
            out[batch_index, :valid_len] = mixed.reshape(valid_len, y.size(-1))

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
    target = model.module if (hasattr(model, "module") and hasattr(model.module, "blocks")) else model
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
                y = orig_self_fn(x, seq_lens, grid_sizes, freqs)
                state.total_self_attention_calls += 1
                if not state.should_modify(layer_id):
                    return y
                mixed = state.temporal_mix(y, seq_lens, grid_sizes)
                state.modified_self_attention_calls += 1
                state.modified_by_step_layer[(int(state.current_step), int(layer_id))] += 1
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
    self_attn_kernel_branch: str = "cond",
    self_attn_kernel_radius: int = 2,
    self_attn_kernel_sigma: float = 1.0,
    self_attn_kernel_mix_alpha: float = 0.25,
    parallel_cfg: Optional[Wan21T2VParallelConfig] = None,
):
    """Generate one video with self-attention temporal output-kernel intervention.

    Inputs:
        self_attn_kernel_steps: 1-based diffusion steps to intervene. Empty
            means all sampling steps.
        self_attn_kernel_layers: DiT layer indices to intervene. Empty means
            all layers.
        self_attn_kernel_branch: `cond`, `uncond`, or `both` CFG branch.
        self_attn_kernel_radius: temporal radius in latent-token frames.
        self_attn_kernel_sigma: Gaussian-kernel width in latent-token frames.
        self_attn_kernel_mix_alpha: residual mixing coefficient. `0` is no-op;
            `1` fully replaces the self-attention output by the smoothed output.

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
        branch=self_attn_kernel_branch,
        kernel_radius=self_attn_kernel_radius,
        kernel_sigma=self_attn_kernel_sigma,
        mix_alpha=self_attn_kernel_mix_alpha,
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
        summary = {
            "experiment": "wan21_t2v_self_attention_temporal_kernel",
            "prompt": prompt,
            "seed": int(seed),
            "video_path": video_path,
            "self_attn_kernel_steps": [int(step) for step in resolved_steps],
            "self_attn_kernel_layers": [int(layer) for layer in resolved_layers],
            "self_attn_kernel_branch": str(self_attn_kernel_branch),
            "self_attn_kernel_radius": int(self_attn_kernel_radius),
            "self_attn_kernel_sigma": float(self_attn_kernel_sigma),
            "self_attn_kernel_mix_alpha": float(self_attn_kernel_mix_alpha),
            "total_self_attention_calls": int(state.total_self_attention_calls),
            "modified_self_attention_calls": int(state.modified_self_attention_calls),
        }
        _save_json(os.path.join(output_dir, "self_attention_temporal_kernel_summary.json"), summary)
        return summary
    return None
