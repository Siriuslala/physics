"""Wan2.1-T2V experiment: step_window_ffn_off.

Main entry:
- run_wan21_t2v_step_window_ffn_off

This module removes FFN residual updates in selected denoising windows. It uses
shared runtime helpers and the step-window ablation patch core defined in
utils.py.
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
    _install_wan21_t2v_step_window_ablation_patch,
    _resolve_wan21_t2v_offload_model,
    _save_csv,
    _save_json,
    _save_wan21_t2v_video,
    _unwrap_wan21_t2v_dit_model_for_runtime_patch,
)

def run_wan21_t2v_step_window_ffn_off(
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
    timestep_idx_to_remove_ffn: Sequence[int] = (1,),
    layer_idx_to_remove_ffn: Optional[Sequence[int]] = None,
    ffn_remove_scope: str = "single_step",  # single_step or from_step
    parallel_cfg: Optional[Wan21T2VParallelConfig] = None,
):
    """Run Wan2.1-T2V step-window FFN ablation experiment.

    For each target step `s`, FFN residual updates are removed only when
    `current_step == s`, on all layers or selected layer indices.
    """
    if ffn_remove_scope not in {"single_step", "from_step"}:
        raise ValueError("ffn_remove_scope must be one of: single_step, from_step")
    if not timestep_idx_to_remove_ffn:
        raise ValueError("timestep_idx_to_remove_ffn must be non-empty.")
    remove_steps = _dedup_wan21_t2v_int_list(timestep_idx_to_remove_ffn)
    if any(s < 1 for s in remove_steps):
        raise ValueError("timestep_idx_to_remove_ffn must be >= 1.")

    layer_indices = None
    if layer_idx_to_remove_ffn is not None and len(layer_idx_to_remove_ffn) > 0:
        layer_indices = _dedup_wan21_t2v_int_list(layer_idx_to_remove_ffn)
        if any(i < 0 for i in layer_indices):
            raise ValueError("layer_idx_to_remove_ffn must be >= 0.")

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

    dit_target = _unwrap_wan21_t2v_dit_model_for_runtime_patch(pipeline.model)
    num_layers = len(dit_target.blocks)
    if layer_indices is not None:
        out_of_range = [i for i in layer_indices if i >= num_layers]
        if out_of_range:
            raise ValueError(
                f"layer_idx_to_remove_ffn out of range: {out_of_range}, num_layers={num_layers}"
            )

    rows = []
    for remove_step in remove_steps:
        handle = _install_wan21_t2v_step_window_ablation_patch(
            model=pipeline.model,
            cross_attn_remove_start_step=None,
            cross_attn_step_scope="from_step",
            ffn_remove_step=int(remove_step),
            ffn_step_scope=ffn_remove_scope,
            ffn_remove_layers=layer_indices,
            reuse_removed_cond_for_uncond=False,
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

        if runtime.rank == 0:
            layer_tag = (
                "all"
                if layer_indices is None
                else "_".join(f"{idx:02d}" for idx in layer_indices)
            )
            video_path = os.path.join(
                output_dir,
                f"wan21_t2v_step_window_ffn_off_step_{remove_step:03d}_layers_{layer_tag}_seed_{seed}.mp4",
            )
            _save_wan21_t2v_video(video, video_path, fps=cfg.sample_fps)
            rows.append(
                {
                    "timestep_idx_to_remove_ffn": int(remove_step),
                    "layer_idx_to_remove_ffn": "all" if layer_indices is None else ",".join(str(i) for i in layer_indices),
                    "video_path": video_path,
                    "cross_attn_removed_calls": int(state.cross_attn_removed_calls),
                    "ffn_removed_calls": int(state.ffn_removed_calls),
                    "uncond_reused_calls": int(state.uncond_reused_calls),
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
            )

    if dist.is_initialized():
        dist.barrier()

    if runtime.rank == 0:
        _ensure_dir(output_dir)
        summary = {
            "experiment": "wan21_t2v_step_window_ffn_off",
            "prompt": prompt,
            "rows": rows,
            "timestep_idx_to_remove_ffn": [int(s) for s in remove_steps],
            "layer_idx_to_remove_ffn": None if layer_indices is None else [int(i) for i in layer_indices],
            "ffn_remove_scope": ffn_remove_scope,
            "num_layers": int(num_layers),
            "runtime_device_id": int(runtime.device_id),
            "runtime_local_rank": int(runtime.local_rank),
            "runtime_world_size": int(runtime.world_size),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        }
        _save_json(os.path.join(output_dir, "step_window_ffn_off_summary.json"), summary)
        _save_csv(os.path.join(output_dir, "step_window_ffn_off_summary.csv"), rows)
        return summary
    return None
