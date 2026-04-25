"""Wan2.1-T2V experiment: head_trajectory_dynamics.

Main entry:
- run_wan21_t2v_head_trajectory_dynamics

This module performs offline analysis of head consensus, pairwise trajectory
similarity, and attractor-style dynamics from saved cross-attention maps. It
uses shared map helpers from utils.py and keeps its analysis-specific plotting
functions local.
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
    _dedup_wan21_t2v_int_list,
    _ensure_dir,
    _extract_wan21_t2v_attention_region_center_trajectory,
    _js_wan21_t2v_distance_per_frame,
    _load_wan21_t2v_cross_attention_mean_maps_from_disk,
    _load_wan21_t2v_cross_attention_token_meta,
    _mean_wan21_t2v_head_maps_for_words,
    _normalize_wan21_t2v_attention_map_per_frame,
    _parse_wan21_t2v_layer_head_specs,
    _save_csv,
    _save_json,
    _trajectory_distance_wan21_t2v_soft_centers,
)

from .head_evolution import (
    _extract_wan21_t2v_reference_peak_and_centroid_trajectory,
    _preprocess_wan21_t2v_attention_map_fhw,
)

def _plot_wan21_t2v_head_trajectory_dynamics_curve(
    rows: Sequence[Dict[str, object]],
    save_file: str,
    metric_key: str,
    title: str,
    y_label: str,
):
    """Plot one step-wise dynamics curve from row dictionaries containing `step` and a metric."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_rows = [row for row in rows if metric_key in row]
    if not plot_rows:
        return ""

    plot_rows = sorted(plot_rows, key=lambda row: int(row["step"]))
    x_steps = [int(row["step"]) for row in plot_rows]
    y_values = [float(row[metric_key]) for row in plot_rows]

    fig, axis = plt.subplots(1, 1, figsize=(7.8, 4.8))
    axis.plot(x_steps, y_values, marker="o", linewidth=1.8, color="#0f766e")
    axis.set_title(title)
    axis.set_xlabel("diffusion step")
    axis.set_ylabel(y_label)
    axis.grid(alpha=0.22, linestyle="--")
    fig.tight_layout()

    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file

def _plot_wan21_t2v_head_trajectory_dynamics_heatmap(
    matrix_rows: Sequence[Dict[str, object]],
    save_file: str,
    title: str,
    row_key: str,
    col_key: str,
    value_key: str,
    row_label: str,
    col_label: str,
):
    """Plot a simple heatmap from flat rows containing row/column/value fields."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not matrix_rows:
        return ""

    row_values = sorted(set(int(row[row_key]) for row in matrix_rows))
    col_values = sorted(set(int(row[col_key]) for row in matrix_rows))
    row_to_index = {value: idx for idx, value in enumerate(row_values)}
    col_to_index = {value: idx for idx, value in enumerate(col_values)}

    heatmap = torch.full((len(row_values), len(col_values)), float("nan"), dtype=torch.float32)
    for row in matrix_rows:
        heatmap[row_to_index[int(row[row_key])], col_to_index[int(row[col_key])]] = float(row[value_key])

    fig_width = max(6.2, 0.28 * len(col_values))
    fig_height = max(4.8, 0.24 * len(row_values))
    fig, axis = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    image = axis.imshow(heatmap.numpy(), cmap="viridis", aspect="auto")
    axis.set_title(title)
    axis.set_xlabel(col_label)
    axis.set_ylabel(row_label)
    axis.set_xticks(list(range(len(col_values))))
    axis.set_xticklabels([str(v) for v in col_values], rotation=45, ha="right", fontsize=8)
    axis.set_yticks(list(range(len(row_values))))
    axis.set_yticklabels([str(v) for v in row_values], fontsize=8)
    fig.colorbar(image, ax=axis, shrink=0.82)
    fig.tight_layout()

    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file

def _plot_wan21_t2v_head_trajectory_dynamics_multihead_curve(
    rows: Sequence[Dict[str, object]],
    save_file: str,
    metric_key: str,
    title: str,
    y_label: str,
):
    """Plot one layer-level curve figure with one line per head across diffusion steps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped_rows: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        if metric_key not in row:
            continue
        grouped_rows[str(row["head_tag"])].append(row)
    if not grouped_rows:
        return ""

    head_tags = sorted(grouped_rows.keys())
    fig, axis = plt.subplots(1, 1, figsize=(8.6, 5.2))
    color_map = plt.get_cmap("gist_ncar", max(1, len(head_tags)))
    for color_index, head_tag in enumerate(head_tags):
        head_rows = sorted(grouped_rows[head_tag], key=lambda row: int(row["step"]))
        x_steps = [int(row["step"]) for row in head_rows]
        y_values = [float(row[metric_key]) for row in head_rows]
        axis.plot(
            x_steps,
            y_values,
            linewidth=1.35,
            alpha=0.92,
            color=color_map(color_index),
            label=head_tag,
        )

    axis.set_title(title)
    axis.set_xlabel("diffusion step")
    axis.set_ylabel(y_label)
    axis.grid(alpha=0.22, linestyle="--")
    if len(head_tags) <= 24:
        axis.legend(fontsize=6.6, ncol=3)
    fig.tight_layout()

    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file

def _plot_wan21_t2v_head_trajectory_centers(
    center_rows: Sequence[Dict[str, object]],
    save_file: str,
    title: str,
):
    """Plot center trajectories for selected heads on a token-grid plane."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not center_rows:
        return ""

    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    all_y = []
    all_x = []
    for row in center_rows:
        head_tag = str(row["head_tag"])
        grouped[head_tag].append(row)
        all_y.append(float(row["center_y"]))
        all_x.append(float(row["center_x"]))

    fig, axis = plt.subplots(1, 1, figsize=(7.2, 6.0))
    color_map = plt.get_cmap("tab20", max(1, len(grouped)))
    for color_index, head_tag in enumerate(sorted(grouped.keys())):
        rows_for_head = sorted(grouped[head_tag], key=lambda row: int(row["frame"]))
        ys = [float(row["center_y"]) for row in rows_for_head]
        xs = [float(row["center_x"]) for row in rows_for_head]
        axis.plot(xs, ys, linewidth=1.8, alpha=0.92, color=color_map(color_index), label=head_tag)
        axis.scatter([xs[0]], [ys[0]], s=18, color=color_map(color_index), alpha=0.95)

    axis.set_title(title)
    axis.set_xlabel("token-x")
    axis.set_ylabel("token-y")
    axis.grid(alpha=0.2, linestyle="--")
    axis.invert_yaxis()
    if len(grouped) <= 20:
        axis.legend(fontsize=7, ncol=2)

    if all_x and all_y:
        axis.set_xlim(min(all_x) - 0.5, max(all_x) + 0.5)
        axis.set_ylim(max(all_y) + 0.5, min(all_y) - 0.5)

    fig.tight_layout()
    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file

def _plot_wan21_t2v_head_trajectory_center_overlay(
    probability_map_fhw: torch.Tensor,
    center_f2: torch.Tensor,
    save_file: str,
    title: str,
    num_frames: int = 10,
):
    """Visualize per-frame probability maps with overlaid extracted centers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if probability_map_fhw.dim() != 3:
        raise ValueError(f"Expected [F, H, W], got shape={tuple(probability_map_fhw.shape)}")
    if center_f2.dim() != 2 or center_f2.size(-1) != 2:
        raise ValueError(f"Expected [F, 2], got shape={tuple(center_f2.shape)}")

    frame_count = int(probability_map_fhw.size(0))
    if frame_count <= 0:
        return ""

    if int(num_frames) <= 0 or int(num_frames) >= frame_count:
        frame_indices = list(range(frame_count))
    else:
        frame_indices = (
            torch.linspace(0, frame_count - 1, steps=int(num_frames))
            .round()
            .long()
            .unique(sorted=True)
            .tolist()
        )

    num_panels = len(frame_indices)
    fig_width = max(2.8 * num_panels, 8.0)
    fig, axes = plt.subplots(1, num_panels, figsize=(fig_width, 3.2))
    if num_panels == 1:
        axes = [axes]

    global_max = float(probability_map_fhw.max().item()) if probability_map_fhw.numel() > 0 else 1.0
    global_max = max(global_max, 1e-8)

    for axis, frame_index in zip(axes, frame_indices):
        frame_map = probability_map_fhw[int(frame_index)].detach().cpu().float()
        center_y = float(center_f2[int(frame_index), 0].item())
        center_x = float(center_f2[int(frame_index), 1].item())
        axis.imshow(frame_map.numpy(), cmap="viridis", vmin=0.0, vmax=global_max)
        axis.scatter(
            [center_x],
            [center_y],
            s=36,
            c=["#ff3b30"],
            marker="o",
            edgecolors="white",
            linewidths=0.9,
        )
        axis.set_title(f"frame={int(frame_index)}", fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file

def _format_wan21_t2v_value_for_filename(value) -> str:
    """Format a scalar config value into a filesystem-friendly token."""
    text = str(value).strip().lower()
    text = text.replace("-", "m").replace(".", "p")
    return re.sub(r"[^a-z0-9_]+", "_", text).strip("_")

def _build_wan21_t2v_head_trajectory_cache_basename(
    center_method: str,
    center_power: float,
    center_quantile: float,
    preprocessed_center_mode: str,
    preprocess_winsorize_quantile: float,
    preprocess_despike_quantile: float,
    preprocess_min_component_area: int,
) -> str:
    """Build a descriptive cache basename from center-extraction settings."""
    method_name = str(center_method).strip().lower()
    if method_name == "region_centroid":
        parts = [
            "head_trajectory_dynamics_trajectory_cache",
            "region_centroid",
            f"q_{_format_wan21_t2v_value_for_filename(center_quantile)}",
            f"p_{_format_wan21_t2v_value_for_filename(center_power)}",
        ]
    elif method_name == "preprocessed_component_center":
        parts = [
            "head_trajectory_dynamics_trajectory_cache",
            "preprocessed_component_center",
            f"mode_{_format_wan21_t2v_value_for_filename(preprocessed_center_mode)}",
            f"q_{_format_wan21_t2v_value_for_filename(center_quantile)}",
            f"p_{_format_wan21_t2v_value_for_filename(center_power)}",
            f"wq_{_format_wan21_t2v_value_for_filename(preprocess_winsorize_quantile)}",
            f"dq_{_format_wan21_t2v_value_for_filename(preprocess_despike_quantile)}",
            f"mca_{int(preprocess_min_component_area)}",
        ]
    else:
        raise ValueError(
            "head_trajectory_dynamics_center_method must be one of "
            "{'region_centroid', 'preprocessed_component_center'}, "
            f"got: {center_method}"
        )
    return "_".join(parts) + ".json"

def _load_wan21_t2v_head_trajectory_cache(cache_path: str) -> Dict[str, object]:
    """Load one trajectory-cache JSON file if it exists, otherwise create an empty cache payload."""
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            loaded.setdefault("trajectories", {})
            return loaded
    return {"trajectories": {}}

def _save_wan21_t2v_head_trajectory_cache(cache_path: str, payload: Dict[str, object]):
    """Write one trajectory-cache JSON payload to disk."""
    _ensure_dir(os.path.dirname(cache_path))
    with open(cache_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

def _get_wan21_t2v_cached_center_trajectory(
    cache_payload: Dict[str, object],
    step: int,
    layer: int,
    head: int,
) -> Optional[List[Tuple[float, float]]]:
    """Return cached trajectory if present."""
    trajectories = cache_payload.get("trajectories", {})
    if not isinstance(trajectories, dict):
        return None
    step_payload = trajectories.get(str(int(step)), {})
    layer_payload = step_payload.get(str(int(layer)), {}) if isinstance(step_payload, dict) else {}
    head_payload = layer_payload.get(str(int(head))) if isinstance(layer_payload, dict) else None
    if not isinstance(head_payload, list):
        return None
    out: List[Tuple[float, float]] = []
    for point in head_payload:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        out.append((float(point[0]), float(point[1])))
    return out

def _set_wan21_t2v_cached_center_trajectory(
    cache_payload: Dict[str, object],
    step: int,
    layer: int,
    head: int,
    trajectory: Sequence[Tuple[float, float]],
):
    """Insert or overwrite one cached center trajectory."""
    trajectories = cache_payload.setdefault("trajectories", {})
    step_payload = trajectories.setdefault(str(int(step)), {})
    layer_payload = step_payload.setdefault(str(int(layer)), {})
    layer_payload[str(int(head))] = [
        [float(point_y), float(point_x)]
        for point_y, point_x in trajectory
    ]

def _extract_wan21_t2v_head_trajectory_centers(
    map_fhw: torch.Tensor,
    center_method: str,
    center_power: float,
    center_quantile: float,
    preprocessed_center_mode: str,
    preprocess_winsorize_quantile: float,
    preprocess_despike_quantile: float,
    preprocess_min_component_area: int,
) -> Tuple[List[Tuple[float, float]], Dict[str, object]]:
    """Extract one frame-wise center trajectory with the selected method."""
    method_name = str(center_method).strip().lower()
    stats: Dict[str, object] = {
        "center_method": method_name,
        "center_power": float(center_power),
        "center_quantile": float(center_quantile),
    }

    if method_name == "region_centroid":
        trajectory = _extract_wan21_t2v_attention_region_center_trajectory(
            map_fhw=map_fhw,
            power=float(center_power),
            quantile=float(center_quantile),
        )
        stats["preprocess_enabled"] = 0
        stats["preprocessed_center_mode"] = ""
        return trajectory, stats

    if method_name != "preprocessed_component_center":
        raise ValueError(
            "head_trajectory_dynamics_center_method must be one of "
            "{'region_centroid', 'preprocessed_component_center'}, "
            f"got: {center_method}"
        )

    center_mode = str(preprocessed_center_mode).strip().lower()
    if center_mode not in {"peak", "centroid", "geometric_center"}:
        raise ValueError(
            "head_trajectory_dynamics_preprocessed_center_mode must be one of "
            "{'peak', 'centroid', 'geometric_center'}, "
            f"got: {preprocessed_center_mode}"
        )

    preprocessed_map, preprocess_stats = _preprocess_wan21_t2v_attention_map_fhw(
        map_fhw=map_fhw,
        winsorize_quantile=float(preprocess_winsorize_quantile),
        despike_quantile=float(preprocess_despike_quantile),
        min_component_area=int(preprocess_min_component_area),
    )
    trajectory_data = _extract_wan21_t2v_reference_peak_and_centroid_trajectory(
        map_fhw=preprocessed_map,
        power=float(center_power),
        quantile=float(center_quantile),
    )
    if center_mode == "peak":
        trajectory = trajectory_data["peak_centers"]
    elif center_mode == "centroid":
        trajectory = trajectory_data["centroid_centers"]
    else:
        trajectory = trajectory_data["geometric_centers"]

    stats.update(
        {
            "preprocess_enabled": 1,
            "preprocessed_center_mode": center_mode,
            "preprocess_stats": preprocess_stats,
        }
    )
    return trajectory, stats

def _center_trajectory_wan21_t2v_to_tensor(
    trajectory: Sequence[Tuple[float, float]],
) -> torch.Tensor:
    """Convert one `(y, x)` trajectory into tensor `[F, 2]`."""
    if not trajectory:
        return torch.zeros((0, 2), dtype=torch.float32)
    return torch.tensor([[float(y), float(x)] for y, x in trajectory], dtype=torch.float32)

def _center_trajectory_wan21_t2v_distance_per_frame(
    center_traj_a: torch.Tensor,
    center_traj_b: torch.Tensor,
) -> torch.Tensor:
    """Compute per-frame L2 distance between two `[F, 2]` center trajectories."""
    if tuple(center_traj_a.shape) != tuple(center_traj_b.shape):
        raise ValueError(
            "Center trajectory shapes must match, "
            f"got {tuple(center_traj_a.shape)} vs {tuple(center_traj_b.shape)}"
        )
    if center_traj_a.dim() != 2 or center_traj_a.size(-1) != 2:
        raise ValueError(f"Expected [F, 2], got shape={tuple(center_traj_a.shape)}")
    return (center_traj_a - center_traj_b).pow(2).sum(dim=-1).sqrt()

def _hellinger_wan21_t2v_distance_per_frame(
    probability_map_a_fhw: torch.Tensor,
    probability_map_b_fhw: torch.Tensor,
) -> torch.Tensor:
    """Compute frame-wise Hellinger distance between two `[F, H, W]` distributions."""
    if tuple(probability_map_a_fhw.shape) != tuple(probability_map_b_fhw.shape):
        raise ValueError(
            "Expected same shapes for Hellinger distance, "
            f"got {tuple(probability_map_a_fhw.shape)} vs {tuple(probability_map_b_fhw.shape)}"
        )
    flat_a = probability_map_a_fhw.reshape(probability_map_a_fhw.size(0), -1).clamp_min(0.0)
    flat_b = probability_map_b_fhw.reshape(probability_map_b_fhw.size(0), -1).clamp_min(0.0)
    return (0.5 * (flat_a.sqrt() - flat_b.sqrt()).pow(2).sum(dim=1)).clamp_min(0.0).sqrt()

def _marginal_wasserstein_wan21_t2v_distance_per_frame(
    probability_map_a_fhw: torch.Tensor,
    probability_map_b_fhw: torch.Tensor,
) -> torch.Tensor:
    """Compute an efficient map-level Wasserstein proxy from row/column marginals.

    This is not full 2D OT. It averages the 1D Wasserstein-1 distance of the
    row marginals and the column marginals.
    """
    if tuple(probability_map_a_fhw.shape) != tuple(probability_map_b_fhw.shape):
        raise ValueError(
            "Expected same shapes for marginal Wasserstein distance, "
            f"got {tuple(probability_map_a_fhw.shape)} vs {tuple(probability_map_b_fhw.shape)}"
        )
    row_a = probability_map_a_fhw.sum(dim=2)
    row_b = probability_map_b_fhw.sum(dim=2)
    col_a = probability_map_a_fhw.sum(dim=1)
    col_b = probability_map_b_fhw.sum(dim=1)
    row_w1 = (row_a.cumsum(dim=1) - row_b.cumsum(dim=1)).abs().sum(dim=1)
    col_w1 = (col_a.cumsum(dim=1) - col_b.cumsum(dim=1)).abs().sum(dim=1)
    return 0.5 * (row_w1 + col_w1)

def _support_overlap_iou_wan21_t2v_per_frame(
    probability_map_a_fhw: torch.Tensor,
    probability_map_b_fhw: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    """Compute frame-wise IoU of high-response support masks."""
    if tuple(probability_map_a_fhw.shape) != tuple(probability_map_b_fhw.shape):
        raise ValueError(
            "Expected same shapes for support-overlap IoU, "
            f"got {tuple(probability_map_a_fhw.shape)} vs {tuple(probability_map_b_fhw.shape)}"
        )
    flat_a = probability_map_a_fhw.reshape(probability_map_a_fhw.size(0), -1)
    flat_b = probability_map_b_fhw.reshape(probability_map_b_fhw.size(0), -1)
    threshold_a = torch.quantile(flat_a, q=float(quantile), dim=1, keepdim=True)
    threshold_b = torch.quantile(flat_b, q=float(quantile), dim=1, keepdim=True)
    mask_a = flat_a >= threshold_a
    mask_b = flat_b >= threshold_b
    intersection = (mask_a & mask_b).sum(dim=1).float()
    union = (mask_a | mask_b).sum(dim=1).float().clamp_min(1.0)
    return intersection / union

def run_wan21_t2v_head_trajectory_dynamics(
    wan21_root: str,
    ckpt_dir: str,
    output_dir: str,
    prompt: str,
    size: Tuple[int, int],
    target_object_words: Sequence[str],
    target_verb_words: Sequence[str] = tuple(),
    task: str = "t2v-14B",
    frame_num: int = 81,
    shift: float = 8.0,
    sample_solver: str = "unipc",
    sampling_steps: int = 50,
    guide_scale: float = 12.0,
    seed: int = 0,
    device_id: Optional[int] = None,
    offload_model: bool = True,
    head_trajectory_dynamics_heads: Sequence[str] = tuple(),
    head_trajectory_dynamics_steps: Sequence[int] = tuple(),
    head_trajectory_dynamics_distance_metrics: Sequence[str] = tuple(),
    head_trajectory_dynamics_reference_step: int = 50,
    head_trajectory_dynamics_reference_layer: int = 27,
    head_trajectory_dynamics_support_quantile: float = 0.9,
    head_trajectory_dynamics_attractor_window: int = 3,
    head_trajectory_dynamics_center_method: str = "region_centroid",
    head_trajectory_dynamics_center_power: float = 1.5,
    head_trajectory_dynamics_center_quantile: float = 0.8,
    head_trajectory_dynamics_preprocessed_center_mode: str = "geometric_center",
    head_trajectory_dynamics_preprocess_winsorize_quantile: float = 0.995,
    head_trajectory_dynamics_preprocess_despike_quantile: float = 0.98,
    head_trajectory_dynamics_preprocess_min_component_area: int = 2,
    head_trajectory_dynamics_center_viz_step: int = -1,
    head_trajectory_dynamics_center_viz_layer: int = -1,
    head_trajectory_dynamics_center_viz_heads: Sequence[str] = tuple(),
    head_trajectory_dynamics_center_viz_num_frames: int = 10,
    reuse_cross_attention_dir: Optional[str] = None,
    parallel_cfg: Optional[Wan21T2VParallelConfig] = None,
):
    """Run offline head-trajectory dynamics analysis from saved cross-attention maps.

    Inputs:
        reuse_cross_attention_dir: existing cross_attention_token_viz directory.
        target_object_words: object words used to aggregate cross-attention maps.
        head_trajectory_dynamics_heads: optional CSV-like head specs `LxHy`; empty means all heads.
        head_trajectory_dynamics_steps: optional step list; empty means all available steps.
        head_trajectory_dynamics_distance_metrics: subset of {"js", "wasserstein"}; empty means both.
        head_trajectory_dynamics_support_quantile: quantile used to define high-response
            support masks for support-overlap IoU.
        head_trajectory_dynamics_attractor_window: future-step window for multi-step
            attractor metrics.
        head_trajectory_dynamics_center_method: one of
            `region_centroid` or `preprocessed_component_center`.
        head_trajectory_dynamics_center_power/head_trajectory_dynamics_center_quantile:
            region-center extraction parameters.
        head_trajectory_dynamics_preprocessed_center_mode: for the preprocessed method,
            choose `peak`, `centroid`, or `geometric_center`.
        head_trajectory_dynamics_center_viz_step/layer/heads: optional selection for per-head
            center-overlay PDFs used to inspect center quality.

    Outputs:
        CSV files for pairwise distances, consensus curves, attractor scores, final-trajectory distance,
        plus summary JSON and PDF visualizations.
    """
    del wan21_root, ckpt_dir, task, frame_num, size, shift, sample_solver, sampling_steps, guide_scale
    del seed, device_id, offload_model, parallel_cfg, prompt

    if dist.is_initialized() and dist.get_rank() != 0:
        dist.barrier()
        return None

    object_words = [str(word).strip() for word in target_object_words if str(word).strip()]
    if not object_words:
        raise ValueError("head_trajectory_dynamics requires non-empty target_object_words.")
    object_words = list(dict.fromkeys(object_words))

    verb_words = [str(word).strip() for word in target_verb_words if str(word).strip()]
    verb_words = list(dict.fromkeys(verb_words))

    if not reuse_cross_attention_dir or (not str(reuse_cross_attention_dir).strip()):
        raise ValueError(
            "head_trajectory_dynamics is an offline analysis and requires --reuse_cross_attention_dir "
            "pointing to an existing cross_attention_token_viz output directory."
        )

    cross_attention_dir = os.path.abspath(str(reuse_cross_attention_dir))
    loaded_maps_raw, loaded_maps_source = _load_wan21_t2v_cross_attention_mean_maps_from_disk(
        output_dir=cross_attention_dir,
        draw_attention_maps_path="",
    )
    if not loaded_maps_raw:
        raise ValueError(
            f"No valid cross-attention maps found under reuse_cross_attention_dir={cross_attention_dir}."
        )

    mean_maps: Dict[Tuple[int, int, str], torch.Tensor] = {}
    for key, value in loaded_maps_raw.items():
        if not isinstance(key, (tuple, list)) or len(key) != 3:
            continue
        step_index = int(key[0])
        layer_index = int(key[1])
        token_name = str(key[2])
        mean_maps[(step_index, layer_index, token_name)] = value.float()
    if not mean_maps:
        raise ValueError(
            f"Loaded maps exist but no valid (step, layer, token) keys found in: {loaded_maps_source}"
        )

    _ensure_dir(output_dir)
    available_steps = sorted({int(key[0]) for key in mean_maps.keys()})
    available_layers = sorted({int(key[1]) for key in mean_maps.keys()})
    words_in_maps = sorted({str(key[2]) for key in mean_maps.keys()})

    word_to_positions, word_to_type, prompt_tokens = _load_wan21_t2v_cross_attention_token_meta(
        output_dir=cross_attention_dir,
        words_in_maps=words_in_maps,
        target_object_words=object_words,
        target_verb_words=verb_words,
    )
    del word_to_positions

    object_words_in_maps = [word for word in object_words if word in set(words_in_maps)]
    if not object_words_in_maps:
        object_words_in_maps = [word for word in words_in_maps if word_to_type.get(word) == "object"]
    if not object_words_in_maps:
        raise ValueError(
            "None of target_object_words found in reused cross-attention maps and no fallback object token type found. "
            f"target_object_words={object_words}, words_in_maps={words_in_maps[:50]}"
        )

    if head_trajectory_dynamics_steps:
        resolved_steps = _dedup_wan21_t2v_int_list(head_trajectory_dynamics_steps)
    else:
        resolved_steps = list(available_steps)
    missing_steps = [step for step in resolved_steps if step not in set(available_steps)]
    if missing_steps:
        raise ValueError(
            "Some head_trajectory_dynamics_steps are not present in reused maps: "
            f"{missing_steps}; available={available_steps}"
        )

    requested_distance_metrics = [str(x).strip().lower() for x in head_trajectory_dynamics_distance_metrics if str(x).strip()]
    if not requested_distance_metrics:
        requested_distance_metrics = ["js", "hellinger", "wasserstein_map", "support_overlap", "center_l2"]
    requested_distance_metrics = list(dict.fromkeys(requested_distance_metrics))
    metric_aliases = {"wasserstein": "center_l2"}
    requested_distance_metrics = [metric_aliases.get(metric_name, metric_name) for metric_name in requested_distance_metrics]
    requested_distance_metrics = list(dict.fromkeys(requested_distance_metrics))
    for metric_name in requested_distance_metrics:
        if metric_name not in {"js", "hellinger", "wasserstein_map", "support_overlap", "center_l2"}:
            raise ValueError(
                "head_trajectory_dynamics_distance_metrics must be chosen from "
                "{'js', 'hellinger', 'wasserstein_map', 'support_overlap', 'center_l2'} "
                "(legacy alias: 'wasserstein' -> 'center_l2'), "
                f"got `{metric_name}`."
            )

    parsed_heads = _parse_wan21_t2v_layer_head_specs(head_trajectory_dynamics_heads)
    requested_head_set = set(parsed_heads)
    parsed_center_viz_heads = _parse_wan21_t2v_layer_head_specs(head_trajectory_dynamics_center_viz_heads)
    requested_center_viz_head_set = set(parsed_center_viz_heads)
    center_method_name = str(head_trajectory_dynamics_center_method).strip().lower()
    cache_basename = _build_wan21_t2v_head_trajectory_cache_basename(
        center_method=center_method_name,
        center_power=float(head_trajectory_dynamics_center_power),
        center_quantile=float(head_trajectory_dynamics_center_quantile),
        preprocessed_center_mode=str(head_trajectory_dynamics_preprocessed_center_mode).strip().lower(),
        preprocess_winsorize_quantile=float(head_trajectory_dynamics_preprocess_winsorize_quantile),
        preprocess_despike_quantile=float(head_trajectory_dynamics_preprocess_despike_quantile),
        preprocess_min_component_area=int(head_trajectory_dynamics_preprocess_min_component_area),
    )
    center_cache_path = os.path.join(output_dir, cache_basename)
    center_cache_payload = _load_wan21_t2v_head_trajectory_cache(center_cache_path)
    center_cache_payload["center_method"] = center_method_name
    center_cache_payload["algorithm_params"] = {
        "center_power": float(head_trajectory_dynamics_center_power),
        "center_quantile": float(head_trajectory_dynamics_center_quantile),
        "preprocessed_center_mode": str(head_trajectory_dynamics_preprocessed_center_mode).strip().lower(),
        "preprocess_winsorize_quantile": float(head_trajectory_dynamics_preprocess_winsorize_quantile),
        "preprocess_despike_quantile": float(head_trajectory_dynamics_preprocess_despike_quantile),
        "preprocess_min_component_area": int(head_trajectory_dynamics_preprocess_min_component_area),
    }
    cache_hits = 0
    cache_misses = 0

    if head_trajectory_dynamics_reference_step not in set(available_steps):
        raise ValueError(
            f"head_trajectory_dynamics_reference_step={head_trajectory_dynamics_reference_step} "
            f"not found in maps. available_steps={available_steps}"
        )
    if head_trajectory_dynamics_reference_layer not in set(available_layers):
        raise ValueError(
            f"head_trajectory_dynamics_reference_layer={head_trajectory_dynamics_reference_layer} "
            f"not found in maps. available_layers={available_layers}"
        )

    reference_head_maps = _mean_wan21_t2v_head_maps_for_words(
        mean_maps=mean_maps,
        step=int(head_trajectory_dynamics_reference_step),
        layer=int(head_trajectory_dynamics_reference_layer),
        words=object_words_in_maps,
    )
    if reference_head_maps is None:
        raise ValueError(
            "Cannot construct reference object map for head_trajectory_dynamics. "
            f"step={head_trajectory_dynamics_reference_step}, "
            f"layer={head_trajectory_dynamics_reference_layer}, object_words={object_words_in_maps}"
        )
    reference_head_mean_map = reference_head_maps.mean(dim=0)
    reference_probability_map = _normalize_wan21_t2v_attention_map_per_frame(reference_head_mean_map)
    reference_center_trajectory, reference_center_stats = _extract_wan21_t2v_head_trajectory_centers(
        map_fhw=reference_head_mean_map,
        center_method=center_method_name,
        center_power=float(head_trajectory_dynamics_center_power),
        center_quantile=float(head_trajectory_dynamics_center_quantile),
        preprocessed_center_mode=str(head_trajectory_dynamics_preprocessed_center_mode).strip().lower(),
        preprocess_winsorize_quantile=float(head_trajectory_dynamics_preprocess_winsorize_quantile),
        preprocess_despike_quantile=float(head_trajectory_dynamics_preprocess_despike_quantile),
        preprocess_min_component_area=int(head_trajectory_dynamics_preprocess_min_component_area),
    )
    final_reference_center = _center_trajectory_wan21_t2v_to_tensor(reference_center_trajectory)

    head_map_records = []
    pairwise_rows = []
    consensus_rows = []
    attractor_rows = []
    reference_distance_rows = []
    convergence_rows = []
    center_rows = []

    probability_maps_by_step_layer_head: Dict[Tuple[int, int, int], torch.Tensor] = {}
    center_trajectories_by_step_layer_head: Dict[Tuple[int, int, int], torch.Tensor] = {}

    for step_index in resolved_steps:
        for layer_index in available_layers:
            object_head_maps = _mean_wan21_t2v_head_maps_for_words(
                mean_maps=mean_maps,
                step=int(step_index),
                layer=int(layer_index),
                words=object_words_in_maps,
            )
            if object_head_maps is None:
                continue
            cache_dirty = False
            for head_index in range(int(object_head_maps.size(0))):
                if requested_head_set and (int(layer_index), int(head_index)) not in requested_head_set:
                    continue
                map_fhw = object_head_maps[head_index]
                probability_map = _normalize_wan21_t2v_attention_map_per_frame(map_fhw)
                cached_trajectory = _get_wan21_t2v_cached_center_trajectory(
                    cache_payload=center_cache_payload,
                    step=int(step_index),
                    layer=int(layer_index),
                    head=int(head_index),
                )
                if cached_trajectory is None:
                    extracted_trajectory, _ = _extract_wan21_t2v_head_trajectory_centers(
                        map_fhw=map_fhw,
                        center_method=center_method_name,
                        center_power=float(head_trajectory_dynamics_center_power),
                        center_quantile=float(head_trajectory_dynamics_center_quantile),
                        preprocessed_center_mode=str(head_trajectory_dynamics_preprocessed_center_mode).strip().lower(),
                        preprocess_winsorize_quantile=float(head_trajectory_dynamics_preprocess_winsorize_quantile),
                        preprocess_despike_quantile=float(head_trajectory_dynamics_preprocess_despike_quantile),
                        preprocess_min_component_area=int(head_trajectory_dynamics_preprocess_min_component_area),
                    )
                    _set_wan21_t2v_cached_center_trajectory(
                        cache_payload=center_cache_payload,
                        step=int(step_index),
                        layer=int(layer_index),
                        head=int(head_index),
                        trajectory=extracted_trajectory,
                    )
                    cached_trajectory = extracted_trajectory
                    cache_dirty = True
                    cache_misses += 1
                else:
                    cache_hits += 1
                center_trajectory = _center_trajectory_wan21_t2v_to_tensor(cached_trajectory)
                key = (int(step_index), int(layer_index), int(head_index))
                probability_maps_by_step_layer_head[key] = probability_map
                center_trajectories_by_step_layer_head[key] = center_trajectory
                head_map_records.append(
                    {
                        "step": int(step_index),
                        "layer": int(layer_index),
                        "head": int(head_index),
                        "head_tag": f"L{int(layer_index)}H{int(head_index)}",
                        "frame_count": int(probability_map.size(0)),
                        "token_grid_h": int(probability_map.size(1)),
                        "token_grid_w": int(probability_map.size(2)),
                    }
                )
                reference_row = {
                    "step": int(step_index),
                    "layer": int(layer_index),
                    "head": int(head_index),
                    "head_tag": f"L{int(layer_index)}H{int(head_index)}",
                }
                if "js" in requested_distance_metrics:
                    reference_row["js_reference_distance"] = float(
                        _js_wan21_t2v_distance_per_frame(probability_map, reference_probability_map).mean().item()
                    )
                if "hellinger" in requested_distance_metrics:
                    reference_row["hellinger_reference_distance"] = float(
                        _hellinger_wan21_t2v_distance_per_frame(probability_map, reference_probability_map).mean().item()
                    )
                if "wasserstein_map" in requested_distance_metrics:
                    reference_row["wasserstein_map_reference_distance"] = float(
                        _marginal_wasserstein_wan21_t2v_distance_per_frame(probability_map, reference_probability_map).mean().item()
                    )
                if "support_overlap" in requested_distance_metrics:
                    reference_support_iou = _support_overlap_iou_wan21_t2v_per_frame(
                        probability_map,
                        reference_probability_map,
                        quantile=float(head_trajectory_dynamics_support_quantile),
                    )
                    reference_row["support_overlap_reference_iou"] = float(reference_support_iou.mean().item())
                    reference_row["support_overlap_reference_distance"] = float((1.0 - reference_support_iou).mean().item())
                if "center_l2" in requested_distance_metrics:
                    reference_row["center_l2_reference_distance"] = float(
                        _trajectory_distance_wan21_t2v_soft_centers(center_trajectory, final_reference_center)
                    )
                reference_distance_rows.append(reference_row)
                for frame_index in range(int(center_trajectory.size(0))):
                    center_rows.append(
                        {
                            "step": int(step_index),
                            "layer": int(layer_index),
                            "head": int(head_index),
                            "head_tag": f"L{int(layer_index)}H{int(head_index)}",
                            "frame": int(frame_index),
                            "center_y": float(center_trajectory[frame_index, 0].item()),
                            "center_x": float(center_trajectory[frame_index, 1].item()),
                        }
                    )
            if cache_dirty:
                _save_wan21_t2v_head_trajectory_cache(center_cache_path, center_cache_payload)

    if not head_map_records:
        raise ValueError(
            "No head maps remain after applying head filters. "
            f"requested_heads={list(head_trajectory_dynamics_heads)}"
        )

    per_step_layer_heads: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for step_index, layer_index, head_index in probability_maps_by_step_layer_head.keys():
        per_step_layer_heads[(int(step_index), int(layer_index))].append(int(head_index))

    for key in per_step_layer_heads:
        per_step_layer_heads[key] = sorted(set(per_step_layer_heads[key]))

    for (step_index, layer_index), head_indices in sorted(per_step_layer_heads.items()):
        metric_to_pairwise_values: Dict[str, List[float]] = {metric_name: [] for metric_name in requested_distance_metrics}
        for head_i_idx in range(len(head_indices)):
            head_i = int(head_indices[head_i_idx])
            prob_i = probability_maps_by_step_layer_head[(step_index, layer_index, head_i)]
            for head_j_idx in range(head_i_idx + 1, len(head_indices)):
                head_j = int(head_indices[head_j_idx])
                prob_j = probability_maps_by_step_layer_head[(step_index, layer_index, head_j)]
                row = {
                    "step": int(step_index),
                    "layer": int(layer_index),
                    "head_a": int(head_i),
                    "head_b": int(head_j),
                    "head_tag_a": f"L{int(layer_index)}H{int(head_i)}",
                    "head_tag_b": f"L{int(layer_index)}H{int(head_j)}",
                }
                if "js" in requested_distance_metrics:
                    js_distance = _js_wan21_t2v_distance_per_frame(prob_i, prob_j).mean().item()
                    row["js_distance"] = float(js_distance)
                    metric_to_pairwise_values["js"].append(float(js_distance))
                if "hellinger" in requested_distance_metrics:
                    hellinger_distance = _hellinger_wan21_t2v_distance_per_frame(prob_i, prob_j).mean().item()
                    row["hellinger_distance"] = float(hellinger_distance)
                    metric_to_pairwise_values["hellinger"].append(float(hellinger_distance))
                if "wasserstein_map" in requested_distance_metrics:
                    wasserstein_map_distance = _marginal_wasserstein_wan21_t2v_distance_per_frame(prob_i, prob_j).mean().item()
                    row["wasserstein_map_distance"] = float(wasserstein_map_distance)
                    metric_to_pairwise_values["wasserstein_map"].append(float(wasserstein_map_distance))
                if "support_overlap" in requested_distance_metrics:
                    support_iou = _support_overlap_iou_wan21_t2v_per_frame(
                        prob_i,
                        prob_j,
                        quantile=float(head_trajectory_dynamics_support_quantile),
                    )
                    row["support_overlap_iou"] = float(support_iou.mean().item())
                    row["support_overlap_distance"] = float((1.0 - support_iou).mean().item())
                    metric_to_pairwise_values["support_overlap"].append(float((1.0 - support_iou).mean().item()))
                if "center_l2" in requested_distance_metrics:
                    center_i = center_trajectories_by_step_layer_head[(step_index, layer_index, head_i)]
                    center_j = center_trajectories_by_step_layer_head[(step_index, layer_index, head_j)]
                    center_distance = _center_trajectory_wan21_t2v_distance_per_frame(center_i, center_j).mean().item()
                    row["center_l2_distance"] = float(center_distance)
                    metric_to_pairwise_values["center_l2"].append(float(center_distance))
                pairwise_rows.append(row)

        consensus_row = {
            "step": int(step_index),
            "layer": int(layer_index),
            "num_heads": int(len(head_indices)),
        }
        for metric_name in requested_distance_metrics:
            values = metric_to_pairwise_values[metric_name]
            if values:
                mean_distance = float(sum(values) / len(values))
                consensus_row[f"{metric_name}_pairwise_distance_mean"] = mean_distance
                consensus_row[f"{metric_name}_consensus"] = float(1.0 / (1.0 + mean_distance))
            else:
                consensus_row[f"{metric_name}_pairwise_distance_mean"] = 0.0
                consensus_row[f"{metric_name}_consensus"] = 1.0
        consensus_rows.append(consensus_row)

    sorted_resolved_steps = sorted(int(step) for step in resolved_steps)
    step_to_index = {step: idx for idx, step in enumerate(sorted_resolved_steps)}
    for step_index in sorted_resolved_steps[:-1]:
        step_pos = step_to_index[int(step_index)]
        future_steps = sorted_resolved_steps[step_pos + 1: step_pos + 1 + max(1, int(head_trajectory_dynamics_attractor_window))]
        next_step = future_steps[0] if future_steps else None
        for layer_index in available_layers:
            head_indices = per_step_layer_heads.get((int(step_index), int(layer_index)), [])
            if not head_indices or not future_steps:
                continue
            future_head_sets = {
                int(future_step): set(per_step_layer_heads.get((int(future_step), int(layer_index)), []))
                for future_step in future_steps
            }
            for leader_head in head_indices:
                leader_key = (int(step_index), int(layer_index), int(leader_head))
                if leader_key not in center_trajectories_by_step_layer_head:
                    continue
                leader_traj = center_trajectories_by_step_layer_head[leader_key]
                one_step_deltas = []
                window_mean_deltas = []
                best_future_deltas = []
                for follower_head in head_indices:
                    if int(follower_head) == int(leader_head):
                        continue
                    follower_current_key = (int(step_index), int(layer_index), int(follower_head))
                    if follower_current_key not in center_trajectories_by_step_layer_head:
                        continue
                    follower_current = center_trajectories_by_step_layer_head[follower_current_key]
                    current_distance = _trajectory_distance_wan21_t2v_soft_centers(follower_current, leader_traj)
                    future_distances = []
                    for future_step in future_steps:
                        if int(follower_head) not in future_head_sets[int(future_step)]:
                            continue
                        follower_future = center_trajectories_by_step_layer_head[(int(future_step), int(layer_index), int(follower_head))]
                        future_distances.append(
                            (
                                int(future_step),
                                float(_trajectory_distance_wan21_t2v_soft_centers(follower_future, leader_traj)),
                            )
                        )
                    if not future_distances:
                        continue
                    if next_step is not None and future_distances[0][0] == int(next_step):
                        one_step_deltas.append(float(current_distance - future_distances[0][1]))
                    window_mean_deltas.append(
                        float(
                            current_distance
                            - sum(distance for _, distance in future_distances) / float(len(future_distances))
                        )
                    )
                    best_future_deltas.append(
                        float(current_distance - min(distance for _, distance in future_distances))
                    )

                method_to_deltas = {
                    "one_step": one_step_deltas,
                    "window_mean": window_mean_deltas,
                    "best_future": best_future_deltas,
                }
                for method_name, deltas in method_to_deltas.items():
                    if not deltas:
                        continue
                    attractor_rows.append(
                        {
                            "step": int(step_index),
                            "next_step": int(next_step) if next_step is not None else -1,
                            "window_end_step": int(future_steps[-1]),
                            "layer": int(layer_index),
                            "head": int(leader_head),
                            "head_tag": f"L{int(layer_index)}H{int(leader_head)}",
                            "attractor_method": method_name,
                            "attractor_score_mean": float(sum(deltas) / len(deltas)),
                            "attractor_score_max": float(max(deltas)),
                            "attractor_score_min": float(min(deltas)),
                            "num_followers": int(len(deltas)),
                        }
                    )

    metric_to_reference_key = {
        "js": "js_reference_distance",
        "hellinger": "hellinger_reference_distance",
        "wasserstein_map": "wasserstein_map_reference_distance",
        "support_overlap": "support_overlap_reference_distance",
        "center_l2": "center_l2_reference_distance",
    }
    for layer_index in available_layers:
        layer_heads = sorted(
            set(
                int(row["head"])
                for row in reference_distance_rows
                if int(row["layer"]) == int(layer_index)
            )
        )
        for head_index in layer_heads:
            head_rows = sorted(
                [
                    row
                    for row in reference_distance_rows
                    if int(row["layer"]) == int(layer_index) and int(row["head"]) == int(head_index)
                ],
                key=lambda row: int(row["step"]),
            )
            if not head_rows:
                continue
            for metric_name, metric_key in metric_to_reference_key.items():
                if metric_name not in requested_distance_metrics:
                    continue
                values = [float(row[metric_key]) for row in head_rows if metric_key in row]
                if not values:
                    continue
                initial_distance = float(values[0])
                final_distance = float(values[-1])
                distance_gap = max(0.0, initial_distance - final_distance)
                lock_in_step_rho_0p2 = ""
                lock_in_step_rho_0p5 = ""
                threshold_0p2 = final_distance + 0.2 * distance_gap
                threshold_0p5 = final_distance + 0.5 * distance_gap
                for row in head_rows:
                    step_value = int(row["step"])
                    metric_value = float(row[metric_key])
                    if lock_in_step_rho_0p2 == "" and metric_value <= threshold_0p2:
                        lock_in_step_rho_0p2 = step_value
                    if lock_in_step_rho_0p5 == "" and metric_value <= threshold_0p5:
                        lock_in_step_rho_0p5 = step_value
                convergence_rows.append(
                    {
                        "layer": int(layer_index),
                        "head": int(head_index),
                        "head_tag": f"L{int(layer_index)}H{int(head_index)}",
                        "metric": metric_name,
                        "initial_reference_distance": initial_distance,
                        "final_reference_distance": final_distance,
                        "reference_distance_auc": float(sum(values) / len(values)),
                        "lock_in_step_rho_0p2": lock_in_step_rho_0p2,
                        "lock_in_step_rho_0p5": lock_in_step_rho_0p5,
                    }
                )

    _save_csv(os.path.join(output_dir, "head_trajectory_dynamics_head_maps.csv"), head_map_records)
    _save_csv(os.path.join(output_dir, "head_trajectory_dynamics_pairwise.csv"), pairwise_rows)
    _save_csv(os.path.join(output_dir, "head_trajectory_dynamics_consensus.csv"), consensus_rows)
    _save_csv(os.path.join(output_dir, "head_trajectory_dynamics_attractor.csv"), attractor_rows)
    _save_csv(os.path.join(output_dir, "head_trajectory_dynamics_reference_distance.csv"), reference_distance_rows)
    _save_csv(os.path.join(output_dir, "head_trajectory_dynamics_convergence.csv"), convergence_rows)
    trajectory_centers_csv_path = os.path.join(output_dir, "head_trajectory_dynamics_trajectory_centers.csv")
    legacy_soft_centers_csv_path = os.path.join(output_dir, "head_trajectory_dynamics_soft_centers.csv")
    _save_csv(trajectory_centers_csv_path, center_rows)
    _save_csv(legacy_soft_centers_csv_path, center_rows)

    plot_paths = []
    plots_dir = os.path.join(output_dir, "head_trajectory_dynamics_plots")

    for metric_name in requested_distance_metrics:
        for layer_index in available_layers:
            layer_rows = [row for row in consensus_rows if int(row["layer"]) == int(layer_index)]
            if not layer_rows:
                continue
            plot_path = _plot_wan21_t2v_head_trajectory_dynamics_curve(
                rows=layer_rows,
                save_file=os.path.join(
                    plots_dir,
                    "consensus_curves",
                    metric_name,
                    f"consensus_layer_{int(layer_index):02d}_{metric_name}.pdf",
                ),
                metric_key=f"{metric_name}_consensus",
                title=f"Head Trajectory Consensus ({metric_name}) | layer={int(layer_index)}",
                y_label=f"{metric_name} consensus",
            )
            if plot_path:
                plot_paths.append(plot_path)

        heatmap_path = _plot_wan21_t2v_head_trajectory_dynamics_heatmap(
            matrix_rows=consensus_rows,
            save_file=os.path.join(
                plots_dir,
                "consensus_heatmaps",
                f"consensus_heatmap_{metric_name}.pdf",
            ),
            title=f"Head Trajectory Consensus Heatmap ({metric_name})",
            row_key="layer",
            col_key="step",
            value_key=f"{metric_name}_consensus",
            row_label="layer",
            col_label="diffusion step",
        )
        if heatmap_path:
            plot_paths.append(heatmap_path)

    attractor_methods = sorted(set(str(row["attractor_method"]) for row in attractor_rows))
    for method_name in attractor_methods:
        for layer_index in available_layers:
            layer_attractor_rows = [
                row for row in attractor_rows
                if int(row["layer"]) == int(layer_index) and str(row["attractor_method"]) == method_name
            ]
            if not layer_attractor_rows:
                continue
            by_head: Dict[str, List[Dict[str, object]]] = defaultdict(list)
            for row in layer_attractor_rows:
                by_head[str(row["head_tag"])].append(row)

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axis = plt.subplots(1, 1, figsize=(8.2, 5.0))
            head_tags = sorted(by_head.keys())
            color_map = plt.get_cmap("tab20", max(1, len(head_tags)))
            for color_index, head_tag in enumerate(head_tags):
                head_rows = sorted(by_head[head_tag], key=lambda row: int(row["step"]))
                x_steps = [int(row["step"]) for row in head_rows]
                y_values = [float(row["attractor_score_mean"]) for row in head_rows]
                axis.plot(
                    x_steps,
                    y_values,
                    linewidth=1.4,
                    alpha=0.92,
                    color=color_map(color_index),
                    label=head_tag,
                )
            axis.set_title(f"Head Attractor Score ({method_name}) | layer={int(layer_index)}")
            axis.set_xlabel("diffusion step")
            axis.set_ylabel("attractor score")
            axis.grid(alpha=0.22, linestyle="--")
            if len(head_tags) <= 20:
                axis.legend(fontsize=7, ncol=2)
            fig.tight_layout()
            plot_path = os.path.join(
                plots_dir,
                "attractor_curves",
                method_name,
                f"attractor_layer_{int(layer_index):02d}.pdf",
            )
            _ensure_dir(os.path.dirname(plot_path))
            fig.savefig(plot_path, format="pdf")
            plt.close(fig)
            plot_paths.append(plot_path)

    metric_to_reference_key = {
        "js": "js_reference_distance",
        "hellinger": "hellinger_reference_distance",
        "wasserstein_map": "wasserstein_map_reference_distance",
        "support_overlap": "support_overlap_reference_distance",
        "center_l2": "center_l2_reference_distance",
    }
    for metric_name, metric_key in metric_to_reference_key.items():
        if metric_name not in requested_distance_metrics:
            continue
        for layer_index in available_layers:
            layer_reference_curve_rows = [
                row
                for row in reference_distance_rows
                if int(row["layer"]) == int(layer_index) and metric_key in row
            ]
            if layer_reference_curve_rows:
                curve_path = _plot_wan21_t2v_head_trajectory_dynamics_multihead_curve(
                    rows=layer_reference_curve_rows,
                    save_file=os.path.join(
                        plots_dir,
                        "reference_distance_curves",
                        metric_name,
                        f"reference_distance_layer_{int(layer_index):02d}.pdf",
                    ),
                    metric_key=metric_key,
                    title=f"Reference Distance Curves ({metric_name}) | layer={int(layer_index)}",
                    y_label=f"{metric_name} reference distance",
                )
                if curve_path:
                    plot_paths.append(curve_path)

            layer_reference_rows = [
                {
                    "step": int(row["step"]),
                    "head": int(row["head"]),
                    metric_key: float(row[metric_key]),
                }
                for row in reference_distance_rows
                if int(row["layer"]) == int(layer_index) and metric_key in row
            ]
            if not layer_reference_rows:
                continue
            heatmap_path = _plot_wan21_t2v_head_trajectory_dynamics_heatmap(
                matrix_rows=layer_reference_rows,
                save_file=os.path.join(
                    plots_dir,
                    "reference_distance_heatmaps",
                    metric_name,
                    f"reference_distance_layer_{int(layer_index):02d}.pdf",
                ),
                title=f"Reference Distance Heatmap ({metric_name}) | layer={int(layer_index)}",
                row_key="head",
                col_key="step",
                value_key=metric_key,
                row_label="head",
                col_label="diffusion step",
            )
            if heatmap_path:
                plot_paths.append(heatmap_path)

        metric_convergence_rows = [
            row
            for row in convergence_rows
            if str(row["metric"]) == metric_name
        ]
        if metric_convergence_rows:
            auc_heatmap_rows = [
                {
                    "layer": int(row["layer"]),
                    "head": int(row["head"]),
                    "reference_distance_auc": float(row["reference_distance_auc"]),
                }
                for row in metric_convergence_rows
            ]
            auc_heatmap_path = _plot_wan21_t2v_head_trajectory_dynamics_heatmap(
                matrix_rows=auc_heatmap_rows,
                save_file=os.path.join(
                    plots_dir,
                    "convergence_heatmaps",
                    metric_name,
                    "reference_distance_auc.pdf",
                ),
                title=f"Reference Distance AUC ({metric_name})",
                row_key="layer",
                col_key="head",
                value_key="reference_distance_auc",
                row_label="layer",
                col_label="head",
            )
            if auc_heatmap_path:
                plot_paths.append(auc_heatmap_path)

            for lock_in_key in ("lock_in_step_rho_0p2", "lock_in_step_rho_0p5"):
                lock_in_rows = [
                    {
                        "layer": int(row["layer"]),
                        "head": int(row["head"]),
                        lock_in_key: float(row[lock_in_key]),
                    }
                    for row in metric_convergence_rows
                    if str(row[lock_in_key]) != ""
                ]
                if not lock_in_rows:
                    continue
                lock_in_path = _plot_wan21_t2v_head_trajectory_dynamics_heatmap(
                    matrix_rows=lock_in_rows,
                    save_file=os.path.join(
                        plots_dir,
                        "convergence_heatmaps",
                        metric_name,
                        f"{lock_in_key}.pdf",
                    ),
                    title=f"{lock_in_key} ({metric_name})",
                    row_key="layer",
                    col_key="head",
                    value_key=lock_in_key,
                    row_label="layer",
                    col_label="head",
                )
                if lock_in_path:
                    plot_paths.append(lock_in_path)

    center_overlay_dir = os.path.join(output_dir, "head_trajectory_dynamics_head_center_overlays")
    center_overlay_specs: List[Tuple[int, int, int]] = []
    explicit_center_viz = (
        int(head_trajectory_dynamics_center_viz_step) >= 1
        and int(head_trajectory_dynamics_center_viz_layer) >= 0
    )
    if explicit_center_viz:
        selected_step = int(head_trajectory_dynamics_center_viz_step)
        selected_layer = int(head_trajectory_dynamics_center_viz_layer)
        candidate_heads = sorted(
            head_idx
            for step_idx, layer_idx, head_idx in probability_maps_by_step_layer_head.keys()
            if int(step_idx) == selected_step and int(layer_idx) == selected_layer
        )
        if requested_center_viz_head_set:
            candidate_heads = [
                head_idx
                for head_idx in candidate_heads
                if (selected_layer, int(head_idx)) in requested_center_viz_head_set
            ]
        center_overlay_specs = [
            (selected_step, selected_layer, int(head_idx))
            for head_idx in candidate_heads
        ]
    elif requested_center_viz_head_set:
        center_overlay_specs = sorted(
            [
                (int(step_idx), int(layer_idx), int(head_idx))
                for step_idx, layer_idx, head_idx in probability_maps_by_step_layer_head.keys()
                if (int(layer_idx), int(head_idx)) in requested_center_viz_head_set
            ]
        )
    elif requested_head_set:
        center_overlay_specs = sorted(
            [
                (int(step_idx), int(layer_idx), int(head_idx))
                for step_idx, layer_idx, head_idx in probability_maps_by_step_layer_head.keys()
                if (int(layer_idx), int(head_idx)) in requested_head_set
            ]
        )

    for step_index, layer_index, head_idx in center_overlay_specs:
        probability_map = probability_maps_by_step_layer_head[(int(step_index), int(layer_index), int(head_idx))]
        center_trajectory = center_trajectories_by_step_layer_head[(int(step_index), int(layer_index), int(head_idx))]
        plot_path = _plot_wan21_t2v_head_trajectory_center_overlay(
            probability_map_fhw=probability_map,
            center_f2=center_trajectory,
            save_file=os.path.join(
                center_overlay_dir,
                f"step_{int(step_index):03d}",
                f"layer_{int(layer_index):02d}",
                f"center_overlay_step_{int(step_index):03d}_layer_{int(layer_index):02d}_head_{int(head_idx):02d}.pdf",
            ),
            title=f"Center Overlay | step={int(step_index)} layer={int(layer_index)} head={int(head_idx)}",
            num_frames=head_trajectory_dynamics_center_viz_num_frames,
        )
        if plot_path:
            plot_paths.append(plot_path)

    summary = {
        "experiment": "wan21_t2v_head_trajectory_dynamics",
        "target_object_words": list(object_words),
        "target_verb_words": list(verb_words),
        "prompt_tokens": prompt_tokens,
        "token_types": word_to_type,
        "object_words_in_maps": object_words_in_maps,
        "reuse_cross_attention_dir": cross_attention_dir,
        "loaded_maps_source": loaded_maps_source,
        "available_steps": [int(step) for step in available_steps],
        "available_layers": [int(layer) for layer in available_layers],
        "head_trajectory_dynamics_steps": [int(step) for step in resolved_steps],
        "head_trajectory_dynamics_heads_input": [str(x) for x in head_trajectory_dynamics_heads],
        "head_trajectory_dynamics_heads_parsed": [
            {"layer": int(layer_index), "head": int(head_index), "head_tag": f"L{int(layer_index)}H{int(head_index)}"}
            for layer_index, head_index in parsed_heads
        ],
        "head_trajectory_dynamics_distance_metrics": list(requested_distance_metrics),
        "reference_step": int(head_trajectory_dynamics_reference_step),
        "reference_layer": int(head_trajectory_dynamics_reference_layer),
        "center_viz_step": int(head_trajectory_dynamics_center_viz_step),
        "center_viz_layer": int(head_trajectory_dynamics_center_viz_layer),
        "center_viz_heads_input": [str(x) for x in head_trajectory_dynamics_center_viz_heads],
        "center_viz_heads_parsed": [
            {"layer": int(layer_index), "head": int(head_index), "head_tag": f"L{int(layer_index)}H{int(head_index)}"}
            for layer_index, head_index in parsed_center_viz_heads
        ],
        "center_viz_num_frames": int(head_trajectory_dynamics_center_viz_num_frames),
        "center_overlay_dir": center_overlay_dir,
        "num_center_overlay_pdfs": int(len(center_overlay_specs)),
        "num_head_records": int(len(head_map_records)),
        "num_pairwise_rows": int(len(pairwise_rows)),
        "num_consensus_rows": int(len(consensus_rows)),
        "num_attractor_rows": int(len(attractor_rows)),
        "num_reference_distance_rows": int(len(reference_distance_rows)),
        "num_convergence_rows": int(len(convergence_rows)),
        "num_center_rows": int(len(center_rows)),
        "plot_paths": plot_paths,
        "head_maps_csv": os.path.join(output_dir, "head_trajectory_dynamics_head_maps.csv"),
        "pairwise_csv": os.path.join(output_dir, "head_trajectory_dynamics_pairwise.csv"),
        "consensus_csv": os.path.join(output_dir, "head_trajectory_dynamics_consensus.csv"),
        "attractor_csv": os.path.join(output_dir, "head_trajectory_dynamics_attractor.csv"),
        "reference_distance_csv": os.path.join(output_dir, "head_trajectory_dynamics_reference_distance.csv"),
        "convergence_csv": os.path.join(output_dir, "head_trajectory_dynamics_convergence.csv"),
        "trajectory_centers_csv": trajectory_centers_csv_path,
        "legacy_soft_centers_csv": legacy_soft_centers_csv_path,
        "center_cache_json": center_cache_path,
        "center_method": center_method_name,
        "support_quantile": float(head_trajectory_dynamics_support_quantile),
        "attractor_window": int(head_trajectory_dynamics_attractor_window),
        "center_method_params": {
            "center_power": float(head_trajectory_dynamics_center_power),
            "center_quantile": float(head_trajectory_dynamics_center_quantile),
            "preprocessed_center_mode": str(head_trajectory_dynamics_preprocessed_center_mode).strip().lower(),
            "preprocess_winsorize_quantile": float(head_trajectory_dynamics_preprocess_winsorize_quantile),
            "preprocess_despike_quantile": float(head_trajectory_dynamics_preprocess_despike_quantile),
            "preprocess_min_component_area": int(head_trajectory_dynamics_preprocess_min_component_area),
        },
        "center_cache_hits": int(cache_hits),
        "center_cache_misses": int(cache_misses),
        "reference_center_stats": reference_center_stats,
    }
    _save_json(os.path.join(output_dir, "head_trajectory_dynamics_summary.json"), summary)
    if dist.is_initialized():
        dist.barrier()
    return summary
