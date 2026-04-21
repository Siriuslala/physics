"""Wan2.1-T2V experiment: attention_dt_profile.

Main entry:
- run_wan21_t2v_attention_dt_profile

This module profiles temporal distance preference P(|dt|) in early denoising
self-attention. It uses the shared runtime helpers and patch-stack runner from
utils.py, plus the patch bundle config objects from wan21_t2v_experiment_patch.py.
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
    _export_wan21_t2v_dt_profile_visualizations,
    _init_wan21_t2v_runtime,
    _resolve_wan21_t2v_offload_model,
    _run_wan21_t2v_once_with_patch,
    _save_json,
    _save_wan21_t2v_video,
    _summarize_wan21_t2v_dt_hist,
)

from .wan21_t2v_experiment_patch import (
    Wan21T2VAttentionProbeConfig,
    Wan21T2VCausalAttentionConfig,
    Wan21T2VPatchBundleConfig,
    Wan21T2VRopePatchConfig,
)

def run_wan21_t2v_attention_dt_profile(
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
    probe_steps: Sequence[int] = (1, 2, 3),
    query_frame_count: int = 8,
    query_mode: str = "center",
    probe_branch: str = "uncond",
    object_token_trajectory: Optional[Dict[int, Tuple[float, float]]] = None,
    parallel_cfg: Optional[Wan21T2VParallelConfig] = None,
):
    """Run early-step attention dt profiling (P(|dt|))."""
    if query_mode == "object_guided" and not object_token_trajectory:
        raise ValueError("query_mode=object_guided requires object_token_trajectory.")

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

    patch_cfg = Wan21T2VPatchBundleConfig(
        rope=Wan21T2VRopePatchConfig(enabled=True, mode="full"),
        probe=Wan21T2VAttentionProbeConfig(
            enabled=True,
            probe_steps=tuple(probe_steps),
            query_frame_count=query_frame_count,
            query_mode=query_mode,
            probe_branch=probe_branch,
            use_abs_dt=True,
            object_token_trajectory=object_token_trajectory,
            collect_maas_maps=False,
        ),
        causal=Wan21T2VCausalAttentionConfig(enabled=False),
    )

    video, state = _run_wan21_t2v_once_with_patch(
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

    if dist.is_initialized():
        dist.barrier()

    if runtime.rank == 0:
        _ensure_dir(output_dir)

        video_path = os.path.join(output_dir, f"wan21_t2v_attention_dt_profile_seed_{seed}.mp4")
        _save_wan21_t2v_video(video, video_path, fps=cfg.sample_fps)

        dt_tensors = state.export_dt_histograms()
        dt_path = os.path.join(output_dir, "attention_dt_histograms.pt")
        torch.save(dt_tensors, dt_path)
        dt_viz = _export_wan21_t2v_dt_profile_visualizations(
            dt_tensors=dt_tensors,
            output_dir=output_dir,
            file_prefix="attention_dt_profile",
        )

        summary = {
            "experiment": "wan21_t2v_attention_dt_profile",
            "prompt": prompt,
            "video_path": video_path,
            "dt_hist_path": dt_path,
            "dt_hist_summary": _summarize_wan21_t2v_dt_hist(dt_tensors),
            "dt_visualizations": dt_viz,
            "probe_steps": list(probe_steps),
            "query_frame_count": query_frame_count,
            "query_mode": query_mode,
            "probe_branch": probe_branch,
        }
        _save_json(os.path.join(output_dir, "attention_dt_profile_summary.json"), summary)
        return summary
    return None
