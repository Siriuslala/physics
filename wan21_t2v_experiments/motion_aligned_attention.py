"""Wan2.1-T2V experiment: motion_aligned_attention.

Main entry:
- run_wan21_t2v_motion_aligned_attention

This module measures whether self-attention mass aligns with motion targets.
It reuses the shared runtime helpers, DT visualization helpers, and motion
trajectory projection helpers from utils.py, together with the patch-stack
configuration objects from wan21_t2v_experiment_patch.py.
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
    _export_wan21_t2v_motion_alignment_visualizations,
    _extract_wan21_t2v_motion_centroid_trajectory,
    _init_wan21_t2v_runtime,
    _map_wan21_t2v_token_frame_to_video_frame_label,
    _project_wan21_t2v_pixel_traj_to_tokens,
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

def run_wan21_t2v_motion_aligned_attention(
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
    maas_layers: Sequence[int] = (0, 10, 20, 30, 39),
    maas_radius: int = 1,
    motion_target_source: str = "motion_centroid",  # motion_centroid or object_token_trajectory
    object_token_trajectory: Optional[Dict[int, Tuple[float, float]]] = None,
    export_motion_alignment_visualizations: bool = True,
    parallel_cfg: Optional[Wan21T2VParallelConfig] = None,
):
    """Run MAAS analysis from early-step attention and motion targets."""
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
            collect_maas_maps=True,
            maas_steps=tuple(probe_steps),
            maas_layers=tuple(maas_layers),
            maas_radius=maas_radius,
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

        video_path = os.path.join(output_dir, f"wan21_t2v_motion_aligned_attention_seed_{seed}.mp4")
        _save_wan21_t2v_video(video, video_path, fps=cfg.sample_fps)

        token_motion_trajectory: Dict[int, Tuple[int, int]] = {}
        if motion_target_source == "object_token_trajectory":
            if not object_token_trajectory:
                raise ValueError("motion_target_source=object_token_trajectory requires object_token_trajectory.")
            if state.maas_grid_size is not None:
                _, token_h, token_w = state.maas_grid_size
                for frame, (y, x) in object_token_trajectory.items():
                    yy = max(0, min(token_h - 1, int(round(float(y)))))
                    xx = max(0, min(token_w - 1, int(round(float(x)))))
                    token_motion_trajectory[int(frame)] = (yy, xx)
            object_rows = [
                {"frame": int(frame), "y": float(y), "x": float(x)}
                for frame, (y, x) in sorted(token_motion_trajectory.items())
            ]
            _save_csv(os.path.join(output_dir, "object_token_trajectory.csv"), object_rows)
        elif motion_target_source == "motion_centroid":
            pixel_trajectory = _extract_wan21_t2v_motion_centroid_trajectory(video)
            pixel_rows = [{"frame": i, "y": float(y), "x": float(x)} for i, (y, x) in enumerate(pixel_trajectory)]
            _save_csv(os.path.join(output_dir, "motion_centroid_trajectory.csv"), pixel_rows)
            if state.maas_grid_size is not None:
                token_f, token_h, token_w = state.maas_grid_size
                _, _, video_h, video_w = video.shape
                token_motion_trajectory = _project_wan21_t2v_pixel_traj_to_tokens(
                    pixel_traj=pixel_trajectory,
                    video_h=video_h,
                    video_w=video_w,
                    token_f=token_f,
                    token_h=token_h,
                    token_w=token_w,
                    temporal_stride=4,
                )
        else:
            raise ValueError(f"Unknown motion_target_source: {motion_target_source}")

        if state.maas_grid_size is None:
            motion_alignment_rows = []
            motion_alignment_summary = {
                "maas_mean_local_ratio": 0.0,
                "maas_std_local_ratio": 0.0,
                "maas_mean_local_mass": 0.0,
                "maas_std_local_mass": 0.0,
                "maas_num_rows": 0,
            }
        else:
            motion_alignment_summary, motion_alignment_rows = state.compute_maas(
                token_motion_trajectory,
                radius=maas_radius,
            )

        if state.maas_grid_size is not None:
            token_frame_count = int(state.maas_grid_size[0])
            for row in motion_alignment_rows:
                query_frame_token = int(row.get("query_frame", 0))
                target_frame_token = query_frame_token + 1
                row["query_frame_token"] = query_frame_token
                row["target_frame_token"] = target_frame_token
                row["query_frame_video"] = _map_wan21_t2v_token_frame_to_video_frame_label(
                    token_frame_idx=query_frame_token,
                    token_frame_count=token_frame_count,
                    video_frame_count=frame_num,
                )
                row["target_frame_video"] = _map_wan21_t2v_token_frame_to_video_frame_label(
                    token_frame_idx=target_frame_token,
                    token_frame_count=token_frame_count,
                    video_frame_count=frame_num,
                )

        _save_csv(os.path.join(output_dir, "motion_aligned_attention_rows.csv"), motion_alignment_rows)
        if export_motion_alignment_visualizations:
            _export_wan21_t2v_motion_alignment_visualizations(
                state=state,
                motion_alignment_rows=motion_alignment_rows,
                token_trajectory=token_motion_trajectory,
                radius=maas_radius,
                output_dir=os.path.join(output_dir, "motion_alignment_visualizations"),
                video_frame_count=frame_num,
            )

        dt_tensors = state.export_dt_histograms()
        dt_path = os.path.join(output_dir, "motion_aligned_attention_dt_histograms.pt")
        torch.save(dt_tensors, dt_path)
        dt_viz = _export_wan21_t2v_dt_profile_visualizations(
            dt_tensors=dt_tensors,
            output_dir=output_dir,
            file_prefix="motion_aligned_attention_dt_profile",
        )

        summary = {
            "experiment": "wan21_t2v_motion_aligned_attention",
            "prompt": prompt,
            "video_path": video_path,
            "maas_summary": motion_alignment_summary,
            "dt_hist_path": dt_path,
            "dt_visualizations": dt_viz,
            "probe_steps": list(probe_steps),
            "query_frame_count": query_frame_count,
            "query_mode": query_mode,
            "probe_branch": probe_branch,
            "maas_layers": list(maas_layers),
            "maas_radius": maas_radius,
            "motion_target_source": motion_target_source,
            "maas_token_frame_count": int(state.maas_grid_size[0]) if state.maas_grid_size is not None else None,
            "video_frame_count": int(frame_num),
            "export_motion_alignment_visualizations": bool(export_motion_alignment_visualizations),
        }
        _save_json(os.path.join(output_dir, "motion_aligned_attention_summary.json"), summary)
        return summary
    return None
