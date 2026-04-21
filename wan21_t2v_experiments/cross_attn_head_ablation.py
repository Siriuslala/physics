"""Wan2.1-T2V experiment: cross_attn_head_ablation.

Main entries:
- Wan21T2VCrossAttnHeadAblationState
- Wan21T2VCrossAttnHeadAblationPatchHandle
- _install_wan21_t2v_cross_attn_head_ablation_patch
- run_wan21_t2v_cross_attn_head_ablation

This module implements targeted cross-attention head ablation. It keeps the
experiment-specific patch state and install logic local to this file, while
reusing shared runtime helpers from utils.py.
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
    _save_csv,
    _save_json,
    _save_wan21_t2v_video,
    _unwrap_wan21_t2v_dit_model_for_runtime_patch,
)

class Wan21T2VCrossAttnHeadAblationState:
    """Runtime state for cross-attention head ablation."""

    def __init__(
        self,
        ablate_layer_heads: Sequence[Tuple[int, int]],
        head_ablation_steps: Optional[Sequence[int]] = None,
    ):
        layer_to_heads = defaultdict(set)
        for layer_idx, head_idx in ablate_layer_heads:
            layer_to_heads[int(layer_idx)].add(int(head_idx))
        self.ablate_layer_heads = {
            int(layer_idx): tuple(sorted(int(h) for h in heads))
            for layer_idx, heads in layer_to_heads.items()
        }
        if head_ablation_steps:
            step_set = set(int(s) for s in head_ablation_steps)
            if any(s < 1 for s in step_set):
                raise ValueError("head_ablation_steps must be >= 1.")
            self.head_ablation_steps = step_set
        else:
            self.head_ablation_steps = None

        self.current_step = 0
        self.current_timestep_value = None
        self.forward_call_index_in_step = 0

        self.total_forward_calls = 0
        self.cross_attn_forward_calls = 0
        self.cross_attn_ablation_calls = 0
        self.ablated_head_instances = 0

    def on_forward_start(self, t_tensor):
        t_value = float(t_tensor.flatten()[0].item()) if t_tensor is not None else None
        self.total_forward_calls += 1
        if self.current_timestep_value is None or t_value != self.current_timestep_value:
            self.current_step += 1
            self.current_timestep_value = t_value
            self.forward_call_index_in_step = 0
        else:
            self.forward_call_index_in_step += 1

    def heads_for_layer(self, layer_idx: int) -> Tuple[int, ...]:
        return self.ablate_layer_heads.get(int(layer_idx), tuple())

    def should_ablate_layer(self, layer_idx: int) -> bool:
        if int(layer_idx) not in self.ablate_layer_heads:
            return False
        if self.head_ablation_steps is None:
            return True
        return int(self.current_step) in self.head_ablation_steps

class Wan21T2VCrossAttnHeadAblationPatchHandle:
    """Restore handle for cross-attention head ablation patch."""

    def __init__(self, target_model, state, original_forward, original_cross_attn_forwards):
        self.target_model = target_model
        self.state = state
        self.original_forward = original_forward
        self.original_cross_attn_forwards = original_cross_attn_forwards

    def restore(self):
        self.target_model.forward = self.original_forward
        for idx, block in enumerate(self.target_model.blocks):
            block.cross_attn.forward = self.original_cross_attn_forwards[idx]

def _install_wan21_t2v_cross_attn_head_ablation_patch(
    model,
    ablate_layer_heads: Sequence[Tuple[int, int]],
    head_ablation_steps: Optional[Sequence[int]] = None,
) -> Wan21T2VCrossAttnHeadAblationPatchHandle:
    """Install runtime patch to ablate selected cross-attention heads."""
    from projects.Wan2_1.wan.modules.attention import flash_attention
    from projects.Wan2_1.wan.modules.model import T5_CONTEXT_TOKEN_NUMBER

    target = _unwrap_wan21_t2v_dit_model_for_runtime_patch(model)
    if not hasattr(target, "blocks"):
        raise RuntimeError("Invalid DiT model: missing blocks.")
    if not ablate_layer_heads:
        raise ValueError("ablate_layer_heads must be non-empty.")

    num_layers = len(target.blocks)
    out_of_range_layers = sorted(set(int(l) for l, _ in ablate_layer_heads if int(l) < 0 or int(l) >= num_layers))
    if out_of_range_layers:
        raise ValueError(f"Layer indices out of range: {out_of_range_layers}, num_layers={num_layers}")

    for layer_idx, head_idx in ablate_layer_heads:
        num_heads = int(target.blocks[int(layer_idx)].cross_attn.num_heads)
        if int(head_idx) < 0 or int(head_idx) >= num_heads:
            raise ValueError(
                f"Head index out of range for layer {layer_idx}: head={head_idx}, num_heads={num_heads}"
            )

    state = Wan21T2VCrossAttnHeadAblationState(
        ablate_layer_heads=ablate_layer_heads,
        head_ablation_steps=head_ablation_steps,
    )

    original_forward = target.forward

    def patched_dit_forward(this, *args, **kwargs):
        t = kwargs.get("t", None)
        if t is None and len(args) > 1:
            t = args[1]
        state.on_forward_start(t)
        return original_forward(*args, **kwargs)

    target.forward = MethodType(patched_dit_forward, target)

    original_cross_attn_forwards = []

    def build_patched_cross_attn_forward(layer_idx: int, heads_to_ablate: Tuple[int, ...], original_cross_forward):
        def patched_cross_attn_forward(self, x, context, context_lens):
            state.cross_attn_forward_calls += 1
            if (not heads_to_ablate) or (not state.should_ablate_layer(layer_idx)):
                return original_cross_forward(x, context, context_lens)

            state.cross_attn_ablation_calls += 1
            state.ablated_head_instances += int(len(heads_to_ablate))

            b, n, d = x.size(0), self.num_heads, self.head_dim

            q = self.norm_q(self.q(x)).view(b, -1, n, d)
            if hasattr(self, "k_img") and hasattr(self, "v_img"):
                image_context_length = context.shape[1] - T5_CONTEXT_TOKEN_NUMBER
                context_img = context[:, :image_context_length]
                context_txt = context[:, image_context_length:]

                k = self.norm_k(self.k(context_txt)).view(b, -1, n, d)
                v = self.v(context_txt).view(b, -1, n, d)
                k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
                v_img = self.v_img(context_img).view(b, -1, n, d)

                x = flash_attention(q, k, v, k_lens=context_lens)
                x = x + flash_attention(q, k_img, v_img, k_lens=None)
            else:
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                x = flash_attention(q, k, v, k_lens=context_lens)

            x = x.clone()
            x[:, :, list(heads_to_ablate), :] = 0
            x = x.flatten(2)
            x = self.o(x)
            return x

        return patched_cross_attn_forward

    for layer_idx, block in enumerate(target.blocks):
        original_cross_attn_forwards.append(block.cross_attn.forward)
        heads_to_ablate = state.heads_for_layer(layer_idx)
        block.cross_attn.forward = MethodType(
            build_patched_cross_attn_forward(layer_idx, heads_to_ablate, block.cross_attn.forward),
            block.cross_attn,
        )

    return Wan21T2VCrossAttnHeadAblationPatchHandle(
        target_model=target,
        state=state,
        original_forward=original_forward,
        original_cross_attn_forwards=original_cross_attn_forwards,
    )

def run_wan21_t2v_cross_attn_head_ablation(
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
    ablate_heads: Sequence[str] = tuple(),
    head_ablation_steps: Sequence[int] = tuple(),
    parallel_cfg: Optional[Wan21T2VParallelConfig] = None,
):
    """Run Wan2.1-T2V cross-attention head ablation experiment."""
    if not ablate_heads:
        raise ValueError("ablate_heads must be non-empty.")

    parsed_heads = _parse_wan21_t2v_layer_head_specs(ablate_heads)
    if not parsed_heads:
        raise ValueError("No valid heads in ablate_heads.")

    resolved_steps = _dedup_wan21_t2v_int_list(head_ablation_steps) if head_ablation_steps else []
    if any(s < 1 or s > int(sampling_steps) for s in resolved_steps):
        raise ValueError(
            f"head_ablation_steps must be in [1, {sampling_steps}], got {resolved_steps}"
        )
    steps_for_patch = resolved_steps if resolved_steps else None

    parallel_cfg = parallel_cfg or Wan21T2VParallelConfig()
    runtime = _init_wan21_t2v_runtime(parallel_cfg, explicit_device_id=device_id)
    seed = _broadcast_seed_if_needed(seed, runtime)

    pipeline, cfg = _build_wan21_t2v_pipeline(
        wan21_root=wan21_root,
        ckpt_dir=ckpt_dir,
        task=task,
        runtime=runtime,
        parallel_cfg=parallel_cfg,
    )
    offload_model = _resolve_wan21_t2v_offload_model(runtime, offload_model)

    handle = _install_wan21_t2v_cross_attn_head_ablation_patch(
        model=pipeline.model,
        ablate_layer_heads=parsed_heads,
        head_ablation_steps=steps_for_patch,
    )
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
        state = handle.state
    finally:
        handle.restore()

    if dist.is_initialized():
        dist.barrier()

    if runtime.rank == 0:
        _ensure_dir(output_dir)

        head_tags = [f"L{layer_idx}H{head_idx}" for layer_idx, head_idx in parsed_heads]
        steps_tag = "all" if not resolved_steps else "_".join(f"{s:03d}" for s in resolved_steps)
        video_path = os.path.join(
            output_dir,
            f"wan21_t2v_cross_attn_head_ablation_steps_{steps_tag}_seed_{seed}.mp4",
        )
        _save_wan21_t2v_video(video, video_path, fps=cfg.sample_fps)

        head_rows = [
            {"layer": int(layer_idx), "head": int(head_idx), "head_spec": f"L{layer_idx}H{head_idx}"}
            for layer_idx, head_idx in parsed_heads
        ]
        _save_csv(os.path.join(output_dir, "cross_attn_head_ablation_heads.csv"), head_rows)

        row = {
            "video_path": video_path,
            "ablate_heads": ",".join(head_tags),
            "head_ablation_steps": "all" if not resolved_steps else ",".join(str(s) for s in resolved_steps),
            "cross_attn_forward_calls": int(state.cross_attn_forward_calls),
            "cross_attn_ablation_calls": int(state.cross_attn_ablation_calls),
            "ablated_head_instances": int(state.ablated_head_instances),
            "total_forward_calls": int(state.total_forward_calls),
            "sampling_steps": int(sampling_steps),
            "seed": int(seed),
            "task": task,
            "sample_solver": sample_solver,
            "shift": float(shift),
            "guide_scale": float(guide_scale),
            "frame_num": int(frame_num),
            "size": f"{size[0]}x{size[1]}",
        }
        _save_csv(os.path.join(output_dir, "cross_attn_head_ablation_summary.csv"), [row])

        summary = {
            "experiment": "wan21_t2v_cross_attn_head_ablation",
            "prompt": prompt,
            "video_path": video_path,
            "ablate_heads_input": [str(x) for x in ablate_heads],
            "ablate_heads_parsed": [
                {"layer": int(layer_idx), "head": int(head_idx), "head_spec": f"L{layer_idx}H{head_idx}"}
                for layer_idx, head_idx in parsed_heads
            ],
            "head_ablation_steps": [int(s) for s in resolved_steps],
            "head_ablation_all_steps": bool(not resolved_steps),
            "cross_attn_forward_calls": int(state.cross_attn_forward_calls),
            "cross_attn_ablation_calls": int(state.cross_attn_ablation_calls),
            "ablated_head_instances": int(state.ablated_head_instances),
            "total_forward_calls": int(state.total_forward_calls),
            "runtime_device_id": int(runtime.device_id),
            "runtime_local_rank": int(runtime.local_rank),
            "runtime_world_size": int(runtime.world_size),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        }
        _save_json(os.path.join(output_dir, "cross_attn_head_ablation_summary.json"), summary)
        return summary
    return None
