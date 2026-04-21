"""Wan2.1-T2V experiment: joint_attention_suite.

Main entry:
- run_wan21_t2v_joint_attention_suite

This module is an orchestration layer that chains cross_attention_token_viz,
attention_dt_profile, and motion_aligned_attention. It intentionally contains
workflow composition logic rather than duplicating lower-level analysis code.
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
    _ensure_dir,
    _load_wan21_t2v_csv_rows,
    _save_json,
)

from .attention_dt_profile import (
    run_wan21_t2v_attention_dt_profile,
)

from .cross_attention_token_viz import (
    run_wan21_t2v_cross_attention_token_viz,
)

from .motion_aligned_attention import (
    run_wan21_t2v_motion_aligned_attention,
)

def _build_wan21_t2v_object_token_trajectory_from_csv(
    trajectory_csv_path: str,
    target_object_words: Sequence[str],
    step: Optional[int] = None,
    layer: Optional[int] = None,
    head: str = "mean",
) -> Tuple[Dict[int, Tuple[float, float]], Dict[str, object]]:
    """Load and aggregate object-token trajectory from cross-attention trajectory CSV."""
    rows = _load_wan21_t2v_csv_rows(trajectory_csv_path)
    if not rows:
        raise ValueError(f"Trajectory CSV is empty or missing: {trajectory_csv_path}")

    object_words = set(w.strip() for w in target_object_words if w.strip())
    filtered = []
    for row in rows:
        token_type = row.get("token_type", "")
        token_name = row.get("token", "")
        if token_type != "object":
            continue
        if object_words and token_name not in object_words:
            continue
        if str(row.get("head", "")) != str(head):
            continue
        filtered.append(row)
    if not filtered:
        raise ValueError(f"No object-token trajectory rows found for head={head}.")

    step_values = sorted(set(int(row["step"]) for row in filtered))
    if step is not None:
        selected_step = int(step)
        if selected_step not in step_values:
            raise ValueError(
                f"Requested object trajectory step={selected_step} is unavailable. "
                f"available_steps={step_values}"
            )
    else:
        # Default to the latest diffusion step (usually sharpest object localization).
        selected_step = step_values[-1]
    filtered = [row for row in filtered if int(row["step"]) == selected_step]
    if not filtered:
        raise ValueError(f"No rows found for selected step={selected_step}.")

    layer_values = sorted(set(int(row["layer"]) for row in filtered))
    if layer is not None:
        selected_layer = int(layer)
        if selected_layer not in layer_values:
            raise ValueError(
                f"Requested object trajectory layer={selected_layer} is unavailable at step={selected_step}. "
                f"available_layers={layer_values}"
            )
    else:
        # Prefer layer 27 if available (empirically stable), else use the deepest available layer.
        selected_layer = 27 if 27 in layer_values else layer_values[-1]
    filtered = [row for row in filtered if int(row["layer"]) == selected_layer]
    if not filtered:
        raise ValueError(f"No rows found for selected layer={selected_layer}.")

    by_frame = defaultdict(list)
    for row in filtered:
        frame = int(row["frame"])
        y = float(row["y"])
        x = float(row["x"])
        by_frame[frame].append((y, x))

    trajectory = {}
    for frame, points in by_frame.items():
        ys = [p[0] for p in points]
        xs = [p[1] for p in points]
        trajectory[int(frame)] = (float(sum(ys) / len(ys)), float(sum(xs) / len(xs)))

    if not trajectory:
        raise ValueError("Failed to build object-token trajectory from CSV.")

    metadata = {
        "selected_step": selected_step,
        "selected_layer": selected_layer,
        "selected_head": str(head),
        "selection_policy": "latest_step + prefer_layer_27_else_last_layer",
        "available_steps": step_values,
        "available_layers_at_selected_step": layer_values,
        "num_frames": len(trajectory),
    }
    return dict(sorted(trajectory.items())), metadata

def run_wan21_t2v_joint_attention_suite(
    wan21_root: str,
    ckpt_dir: str,
    output_dir: str,
    prompt: str,
    size: Tuple[int, int],
    target_object_words: Sequence[str],
    target_verb_words: Sequence[str],
    task: str = "t2v-14B",
    frame_num: int = 81,
    shift: float = 8.0,
    sample_solver: str = "unipc",
    sampling_steps: int = 50,
    guide_scale: float = 12.0,
    seed: int = 0,
    device_id: Optional[int] = None,
    offload_model: bool = True,
    collect_steps: Sequence[int] = (1, 2, 3),
    layers_to_collect: Optional[Sequence[int]] = None,
    query_frame_count: int = 8,
    maas_layers: Sequence[int] = (0, 10, 20, 30, 39),
    maas_radius: int = 1,
    probe_branch: str = "cond",
    motion_target_source: str = "object_token_trajectory",  # object_token_trajectory or motion_centroid
    object_trajectory_step: Optional[int] = None,
    object_trajectory_layer: Optional[int] = None,
    object_trajectory_head: str = "mean",
    reuse_cross_attention_dir: Optional[str] = None,
    parallel_cfg: Optional[Wan21T2VParallelConfig] = None,
):
    """Run a unified 3-stage suite: cross-attn -> object-guided dt profile -> motion-aligned attention."""
    _ensure_dir(output_dir)

    cross_attention_reused = bool(reuse_cross_attention_dir and str(reuse_cross_attention_dir).strip())
    cross_attention_dir = (
        os.path.abspath(str(reuse_cross_attention_dir))
        if cross_attention_reused
        else os.path.join(output_dir, "cross_attention_token_viz")
    )
    dt_profile_dir = os.path.join(output_dir, "object_guided_attention_dt_profile")
    motion_alignment_dir = os.path.join(output_dir, "motion_aligned_attention")

    cross_attention_summary = None
    if cross_attention_reused:
        trajectory_csv_path = os.path.join(cross_attention_dir, "cross_attention_token_trajectory.csv")
        if not os.path.exists(trajectory_csv_path):
            raise FileNotFoundError(
                f"reuse_cross_attention_dir is set but missing trajectory CSV: {trajectory_csv_path}"
            )
        summary_path = os.path.join(cross_attention_dir, "cross_attention_token_viz_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                cross_attention_summary = json.load(f)
        else:
            cross_attention_summary = {
                "experiment": "wan21_t2v_cross_attention_token_viz",
                "reused_cross_attention_dir": cross_attention_dir,
                "cross_attention_token_trajectory_csv": trajectory_csv_path,
            }
    else:
        cross_attention_summary = run_wan21_t2v_cross_attention_token_viz(
            wan21_root=wan21_root,
            ckpt_dir=ckpt_dir,
            output_dir=cross_attention_dir,
            prompt=prompt,
            size=size,
            target_object_words=target_object_words,
            target_verb_words=target_verb_words,
            task=task,
            frame_num=frame_num,
            shift=shift,
            sample_solver=sample_solver,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            seed=seed,
            device_id=device_id,
            offload_model=offload_model,
            collect_steps=collect_steps,
            num_viz_frames=5,
            layers_to_collect=layers_to_collect,
            chunk_size=1024,
            trajectory_enable=True,
            trajectory_style="glow_arrow",
            trajectory_num_frames=0,
            trajectory_smooth_radius=2,
            trajectory_power=1.5,
            trajectory_quantile=0.8,
            trajectory_arrow_stride=4,
            trajectory_include_head_mean=True,
            save_attention_pdfs=True,
            save_trajectory_pdfs=True,
            save_video=True,
            parallel_cfg=parallel_cfg,
        )

    rank = dist.get_rank() if dist.is_initialized() else 0
    object_token_trajectory = None
    object_trajectory_metadata = None
    if rank == 0:
        csv_point_mode = ""
        if isinstance(cross_attention_summary, dict):
            csv_point_mode = str(cross_attention_summary.get("trajectory_csv_point_mode", "")).strip()
        if csv_point_mode and csv_point_mode != "raw_region_center_per_frame":
            raise ValueError(
                "cross_attention trajectory CSV was not exported from raw per-frame centers. "
                "Please rerun cross_attention_token_viz with current code to regenerate "
                "cross_attention_token_trajectory.csv."
            )

        trajectory_csv_path = os.path.join(cross_attention_dir, "cross_attention_token_trajectory.csv")
        object_token_trajectory, object_trajectory_metadata = _build_wan21_t2v_object_token_trajectory_from_csv(
            trajectory_csv_path=trajectory_csv_path,
            target_object_words=target_object_words,
            step=object_trajectory_step,
            layer=object_trajectory_layer,
            head=object_trajectory_head,
        )
        object_trajectory_metadata["trajectory_source_csv"] = trajectory_csv_path
        object_trajectory_metadata["trajectory_source_dir"] = cross_attention_dir
        object_trajectory_metadata["cross_attention_reused"] = cross_attention_reused
        object_trajectory_metadata["trajectory_csv_point_mode"] = (
            csv_point_mode if csv_point_mode else "unknown(no-summary)"
        )
    if dist.is_initialized():
        payload = [object_token_trajectory]
        dist.broadcast_object_list(payload, src=0)
        object_token_trajectory = payload[0]

    dt_profile_summary = run_wan21_t2v_attention_dt_profile(
        wan21_root=wan21_root,
        ckpt_dir=ckpt_dir,
        output_dir=dt_profile_dir,
        prompt=prompt,
        size=size,
        task=task,
        frame_num=frame_num,
        shift=shift,
        sample_solver=sample_solver,
        sampling_steps=sampling_steps,
        guide_scale=guide_scale,
        seed=seed,
        device_id=device_id,
        offload_model=offload_model,
        probe_steps=collect_steps,
        query_frame_count=query_frame_count,
        query_mode="object_guided",
        probe_branch=probe_branch,
        object_token_trajectory=object_token_trajectory,
        parallel_cfg=parallel_cfg,
    )

    motion_alignment_summary = run_wan21_t2v_motion_aligned_attention(
        wan21_root=wan21_root,
        ckpt_dir=ckpt_dir,
        output_dir=motion_alignment_dir,
        prompt=prompt,
        size=size,
        task=task,
        frame_num=frame_num,
        shift=shift,
        sample_solver=sample_solver,
        sampling_steps=sampling_steps,
        guide_scale=guide_scale,
        seed=seed,
        device_id=device_id,
        offload_model=offload_model,
        probe_steps=collect_steps,
        query_frame_count=query_frame_count,
        query_mode="object_guided",
        probe_branch=probe_branch,
        maas_layers=maas_layers,
        maas_radius=maas_radius,
        motion_target_source=motion_target_source,
        object_token_trajectory=object_token_trajectory,
        export_motion_alignment_visualizations=True,
        parallel_cfg=parallel_cfg,
    )

    if rank == 0:
        joint_summary = {
            "experiment": "wan21_t2v_joint_attention_suite",
            "prompt": prompt,
            "target_object_words": list(target_object_words),
            "target_verb_words": list(target_verb_words),
            "collect_steps": list(collect_steps),
            "probe_branch": probe_branch,
            "motion_target_source": motion_target_source,
            "object_trajectory_request": {
                "step": object_trajectory_step,
                "layer": object_trajectory_layer,
                "head": object_trajectory_head,
            },
            "cross_attention_reused": cross_attention_reused,
            "reuse_cross_attention_dir": cross_attention_dir if cross_attention_reused else "",
            "object_trajectory_metadata": object_trajectory_metadata,
            "reports": {
                "cross_attention_token_viz": os.path.join(cross_attention_dir, "cross_attention_token_viz_summary.json"),
                "object_guided_attention_dt_profile": os.path.join(dt_profile_dir, "attention_dt_profile_summary.json"),
                "motion_aligned_attention": os.path.join(motion_alignment_dir, "motion_aligned_attention_summary.json"),
            },
            "sub_summaries": {
                "cross_attention_token_viz": cross_attention_summary,
                "object_guided_attention_dt_profile": dt_profile_summary,
                "motion_aligned_attention": motion_alignment_summary,
            },
        }
        _save_json(os.path.join(output_dir, "joint_attention_suite_summary.json"), joint_summary)
        return joint_summary
    return None
