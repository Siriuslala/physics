"""Wan2.1-T2V experiment: rope_axis_ablation.

Main entry:
- run_wan21_t2v_rope_axis_ablation

This module evaluates how different 3D RoPE axis configurations affect Wan2.1
text-to-video generation. It depends on the shared runtime/pipeline helpers in
utils.py and on the composable DiT patch-stack configuration objects from
wan21_t2v_experiment_patch.py.
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
    _ensure_dir,
    _generate_wan21_t2v_video,
    _init_wan21_t2v_runtime,
    _resolve_wan21_t2v_offload_model,
    _run_wan21_t2v_once_with_patch,
    _save_csv,
    _save_json,
    _save_wan21_t2v_video,
)

from .wan21_t2v_experiment_patch import (
    Wan21T2VAttentionProbeConfig,
    Wan21T2VCausalAttentionConfig,
    Wan21T2VPatchBundleConfig,
    Wan21T2VRopePatchConfig,
)

def run_wan21_t2v_rope_axis_ablation(
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
    rope_modes: Sequence[str] = ("full", "no_f", "no_h", "no_w", "only_f", "only_hw"),
    parallel_cfg: Optional[Wan21T2VParallelConfig] = None,
):
    """Run RoPE axis ablation for Wan2.1-T2V."""
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

    rows = []
    for mode in rope_modes:
        if mode == "full":
            # Strict baseline: do not install any monkey patch in full mode.
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
        else:
            patch_cfg = Wan21T2VPatchBundleConfig(
                rope=Wan21T2VRopePatchConfig(enabled=True, mode=mode),
                probe=Wan21T2VAttentionProbeConfig(enabled=False),
                causal=Wan21T2VCausalAttentionConfig(enabled=False),
            )

            video, _ = _run_wan21_t2v_once_with_patch(
                pipeline=pipeline,
                patch_cfg=patch_cfg,
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

        if runtime.rank == 0:
            video_path = os.path.join(output_dir, f"wan21_{task}-rope_axis-mode_{mode}-seed_{seed}-shift_{shift}-scale_{guide_scale}-frame_{frame_num}.mp4")
            _save_wan21_t2v_video(video, video_path, fps=cfg.sample_fps)
            rows.append({
                "rope_mode": mode,
                "video_path": video_path,
                "seed": seed,
                "frame_num": frame_num,
                "size": f"{size[0]}x{size[1]}",
                "sampling_steps": sampling_steps,
                "guide_scale": guide_scale,
                "shift": shift,
                "sample_solver": sample_solver,
                "task": task,
            })

    if dist.is_initialized():
        dist.barrier()

    if runtime.rank == 0:
        _ensure_dir(output_dir)
        meta = {
            "experiment": "wan21_t2v_rope_axis_ablation",
            "prompt": prompt,
            "rows": rows,
        }
        _save_json(os.path.join(output_dir, "rope_axis_ablation_summary.json"), meta)
        _save_csv(os.path.join(output_dir, "rope_axis_ablation_summary.csv"), rows)
        return meta
    
    return None
