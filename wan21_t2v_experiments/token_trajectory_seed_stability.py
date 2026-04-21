"""Wan2.1-T2V experiment: token_trajectory_seed_stability.

Main entry:
- run_wan21_t2v_token_trajectory_seed_stability

This module aggregates multiple cross_attention_token_viz runs to measure
seed-level trajectory stability. It depends on the visualization experiment for
per-seed outputs and on shared trajectory helpers from utils.py.
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
    _dtw_wan21_t2v_trajectory_distance,
    _ensure_dir,
    _load_wan21_t2v_csv_rows,
    _resample_wan21_t2v_trajectory,
    _sanitize_wan21_t2v_token_name,
    _save_csv,
    _save_json,
)

from .cross_attention_token_viz import (
    run_wan21_t2v_cross_attention_token_viz,
)

def run_wan21_t2v_token_trajectory_seed_stability(
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
    seed_list: Sequence[int] = (0, 1, 2, 3),
    device_id: Optional[int] = None,
    offload_model: bool = True,
    collect_steps: Sequence[int] = (1, 2, 3),
    num_viz_frames: int = 5,
    layers_to_collect: Optional[Sequence[int]] = None,
    chunk_size: int = 1024,
    stability_num_points: int = 41,
    stability_head: str = "mean",
    trajectory_style: str = "glow_arrow",
    trajectory_smooth_radius: int = 2,
    trajectory_power: float = 1.5,
    trajectory_quantile: float = 0.8,
    trajectory_arrow_stride: int = 4,
    parallel_cfg: Optional[Wan21T2VParallelConfig] = None,
):
    """Run multi-seed trajectory stability analysis based on object-token cross-attention trajectories."""
    if not target_object_words:
        raise ValueError("token_trajectory_seed_stability requires non-empty target_object_words.")
    if not seed_list:
        raise ValueError("seed_list is empty.")

    _ensure_dir(output_dir)
    seed_rows = []
    trajectories_by_key = defaultdict(dict)  # (step, layer, token, head) -> {seed: [(y, x), ...]}

    for seed in seed_list:
        seed_output_dir = os.path.join(output_dir, f"seed_{int(seed):06d}")
        summary = run_wan21_t2v_cross_attention_token_viz(
            wan21_root=wan21_root,
            ckpt_dir=ckpt_dir,
            output_dir=seed_output_dir,
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
            seed=int(seed),
            device_id=device_id,
            offload_model=offload_model,
            collect_steps=collect_steps,
            num_viz_frames=num_viz_frames,
            layers_to_collect=layers_to_collect,
            chunk_size=chunk_size,
            trajectory_enable=True,
            trajectory_style=trajectory_style,
            trajectory_num_frames=0,
            trajectory_smooth_radius=trajectory_smooth_radius,
            trajectory_power=trajectory_power,
            trajectory_quantile=trajectory_quantile,
            trajectory_arrow_stride=trajectory_arrow_stride,
            trajectory_include_head_mean=True,
            save_attention_pdfs=False,
            save_trajectory_pdfs=False,
            save_video=False,
            parallel_cfg=parallel_cfg,
        )
        seed_rows.append({"seed": int(seed), "seed_output_dir": seed_output_dir, "summary_path": os.path.join(seed_output_dir, "cross_attention_token_viz_summary.json")})

        if summary is None:
            continue

        trajectory_csv_path = os.path.join(seed_output_dir, "cross_attention_token_trajectory.csv")
        trajectory_rows = _load_wan21_t2v_csv_rows(trajectory_csv_path)
        grouped = defaultdict(dict)  # key -> frame -> (y, x)
        for row in trajectory_rows:
            if row.get("token_type") != "object":
                continue
            if str(row.get("head", "")) != str(stability_head):
                continue
            key = (
                int(row["step"]),
                int(row["layer"]),
                row["token"],
                str(row["head"]),
            )
            grouped[key][int(row["frame"])] = (float(row["y"]), float(row["x"]))

        for key, frame_to_point in grouped.items():
            ordered = [frame_to_point[f] for f in sorted(frame_to_point.keys())]
            if ordered:
                trajectories_by_key[key][int(seed)] = ordered

    if dist.is_initialized():
        dist.barrier()

    # This function delegates heavy generation to sub-runs and aggregates only on rank 0.
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        return None

    _save_csv(os.path.join(output_dir, "seed_run_index.csv"), seed_rows)

    stability_rows = []
    plots_dir = os.path.join(output_dir, "stability_plots")
    _ensure_dir(plots_dir)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for key in sorted(trajectories_by_key.keys()):
        seed_to_traj = trajectories_by_key[key]
        seeds = sorted(seed_to_traj.keys())
        if len(seeds) < 2:
            continue

        resampled = [
            _resample_wan21_t2v_trajectory(seed_to_traj[seed], num_points=stability_num_points)
            for seed in seeds
        ]
        data = torch.tensor(resampled, dtype=torch.float32)  # [S, T, 2]
        mean_traj = data.mean(dim=0)  # [T, 2]
        residual = (data - mean_traj.unsqueeze(0)).pow(2).sum(dim=-1).sqrt()  # [S, T]
        pointwise_dispersion = float(residual.mean().item())

        path_lengths = []
        for seq in resampled:
            length = 0.0
            for i in range(len(seq) - 1):
                y0, x0 = seq[i]
                y1, x1 = seq[i + 1]
                length += ((y1 - y0) ** 2 + (x1 - x0) ** 2) ** 0.5
            path_lengths.append(length)
        path_len_t = torch.tensor(path_lengths, dtype=torch.float32)
        mean_path = float(path_len_t.mean().item())
        std_path = float(path_len_t.std(unbiased=False).item())
        path_cv = float(std_path / max(mean_path, 1e-6))

        velocity = data[:, 1:, :] - data[:, :-1, :]
        speed = velocity.pow(2).sum(dim=-1).sqrt().clamp_min(1e-8)
        unit_velocity = velocity / speed.unsqueeze(-1)
        direction_consistency = float(unit_velocity.mean(dim=0).pow(2).sum(dim=-1).sqrt().mean().item())

        dtw_values = []
        dtw_matrix = torch.zeros((len(seeds), len(seeds)), dtype=torch.float32)
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                d = _dtw_wan21_t2v_trajectory_distance(resampled[i], resampled[j])
                dtw_values.append(d)
                dtw_matrix[i, j] = float(d)
                dtw_matrix[j, i] = float(d)
        mean_dtw = float(torch.tensor(dtw_values).mean().item()) if dtw_values else 0.0

        step, layer, token, head = key
        stability_rows.append(
            {
                "step": int(step),
                "layer": int(layer),
                "token": token,
                "head": head,
                "num_seeds": len(seeds),
                "pointwise_dispersion": pointwise_dispersion,
                "path_length_mean": mean_path,
                "path_length_std": std_path,
                "path_length_cv": path_cv,
                "direction_consistency": direction_consistency,
                "pairwise_dtw_mean": mean_dtw,
            }
        )

        key_tag = f"step_{step:03d}_layer_{layer:02d}_{_sanitize_wan21_t2v_token_name(token)}_{head}"
        fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.6))
        for idx, seed in enumerate(seeds):
            seq = resampled[idx]
            ys = [p[0] for p in seq]
            xs = [p[1] for p in seq]
            ax.plot(xs, ys, alpha=0.28, linewidth=1.0, label=f"seed={seed}" if idx < 8 else None)
        mean_y = mean_traj[:, 0].tolist()
        mean_x = mean_traj[:, 1].tolist()
        ax.plot(mean_x, mean_y, color="#ff3b30", linewidth=2.2, label="mean")
        ax.set_title(f"Trajectory Stability: step={step} layer={layer} token={token} head={head}")
        ax.set_xlabel("token-x")
        ax.set_ylabel("token-y")
        ax.grid(alpha=0.2, linestyle="--")
        if len(seeds) <= 8:
            ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{key_tag}_trajectory_overlay.pdf"), format="pdf")
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(5.8, 5.1))
        im = ax.imshow(dtw_matrix.numpy(), cmap="magma")
        ax.set_title(f"Pairwise DTW: step={step} layer={layer} token={token} head={head}")
        ax.set_xlabel("seed index")
        ax.set_ylabel("seed index")
        fig.colorbar(im, ax=ax, shrink=0.82)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{key_tag}_dtw_matrix.pdf"), format="pdf")
        plt.close(fig)

    _save_csv(os.path.join(output_dir, "seed_stability_summary.csv"), stability_rows)
    summary = {
        "experiment": "wan21_t2v_token_trajectory_seed_stability",
        "prompt": prompt,
        "target_object_words": list(target_object_words),
        "target_verb_words": list(target_verb_words),
        "seed_list": [int(s) for s in seed_list],
        "stability_head": stability_head,
        "stability_num_points": int(stability_num_points),
        "num_summary_rows": len(stability_rows),
    }
    _save_json(os.path.join(output_dir, "seed_stability_summary.json"), summary)
    return summary
