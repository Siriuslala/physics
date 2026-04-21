"""Wan2.1-T2V experiment: causal_schedule.

Main entry:
- run_wan21_t2v_causal_schedule

This module compares early-step bidirectional attention and causal attention
schedules. It reuses the shared runtime helpers and patch-stack runner from
utils.py, together with the patch bundle config objects from
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

def run_wan21_t2v_causal_schedule(
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
    causal_first_n_steps: int = 3,
    causal_mode: str = "flat",  # flat or temporal
    attention_backend: str = "auto",  # auto / flash / torch_sdpa
    run_baseline: bool = True,
    collect_dt_profile: bool = True,
    probe_steps: Sequence[int] = (1, 2, 3),
    query_frame_count: int = 8,
    parallel_cfg: Optional[Wan21T2VParallelConfig] = None,
):
    """Run causal schedule experiment with configurable causal mode/backend.

    This function supports both:
    - flat token causal,
    - strict temporal causal.

    It also supports non-flash backend via `attention_backend=torch_sdpa`.
    """
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
    if runtime.rank == 0 and causal_mode == "temporal" and attention_backend == "torch_sdpa":
        print(
            "[wan21_t2v_experiments] causal_mode=temporal with attention_backend=torch_sdpa "
            "is memory-heavy. If you hit OOM, try attention_backend=auto."
        )

    variants: List[Tuple[str, bool, int, str]] = []
    if run_baseline:
        variants.append(("bidirectional", False, 0, "none"))
    variants.append((f"causal_{causal_mode}_{causal_first_n_steps}", True, causal_first_n_steps, causal_mode))

    rows = []
    for name, enabled, k, mode in variants:
        patch_cfg = Wan21T2VPatchBundleConfig(
            rope=Wan21T2VRopePatchConfig(enabled=True, mode="full"),
            probe=Wan21T2VAttentionProbeConfig(
                enabled=collect_dt_profile,
                probe_steps=tuple(probe_steps),
                query_frame_count=query_frame_count,
                use_abs_dt=True,
                collect_maas_maps=False,
            ),
            causal=Wan21T2VCausalAttentionConfig(
                enabled=enabled,
                causal_first_n_steps=k,
                mode=mode,
                backend=attention_backend,
            ),
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

        if runtime.rank == 0:
            video_path = os.path.join(output_dir, f"wan21_t2v_causal_schedule_{name}_seed_{seed}.mp4")
            _save_wan21_t2v_video(video, video_path, fps=cfg.sample_fps)

            dt_path = None
            dt_summary = None
            if collect_dt_profile:
                dt_tensors = state.export_dt_histograms()
                dt_path = os.path.join(output_dir, f"wan21_t2v_causal_schedule_{name}_dt_histograms.pt")
                torch.save(dt_tensors, dt_path)
                dt_summary = _summarize_wan21_t2v_dt_hist(dt_tensors)

            rows.append(
                {
                    "variant": name,
                    "causal_enabled": enabled,
                    "causal_first_n_steps": k,
                    "causal_mode": mode,
                    "attention_backend": attention_backend,
                    "video_path": video_path,
                    "dt_hist_path": dt_path,
                    "dt_hist_summary": dt_summary,
                }
            )

    if dist.is_initialized():
        dist.barrier()

    if runtime.rank == 0:
        _ensure_dir(output_dir)
        summary = {
            "experiment": "wan21_t2v_causal_schedule",
            "prompt": prompt,
            "rows": rows,
            "probe_steps": list(probe_steps),
            "query_frame_count": query_frame_count,
            "runtime_device_id": int(runtime.device_id),
            "runtime_local_rank": int(runtime.local_rank),
            "runtime_world_size": int(runtime.world_size),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        }
        _save_json(os.path.join(output_dir, "causal_schedule_summary.json"), summary)
        return summary
    return None
