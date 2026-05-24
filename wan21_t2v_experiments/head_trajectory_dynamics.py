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
    _map_wan21_t2v_token_frame_to_video_frame_label,
    _mean_wan21_t2v_head_maps_for_words,
    _normalize_wan21_t2v_attention_map_per_frame,
    _parse_wan21_t2v_layer_head_specs,
    _iter_wan21_t2v_parallel_results,
    _resolve_wan21_t2v_viz_frame_indices,
    _resolve_wan21_t2v_num_workers,
    _save_csv,
    _save_json,
    _trajectory_distance_wan21_t2v_soft_centers,
)

from .head_evolution import (
    _extract_wan21_t2v_connected_components,
    _extract_wan21_t2v_reference_peak_and_centroid_trajectory,
    _preprocess_wan21_t2v_attention_map_fhw,
)

def _get_wan21_t2v_visible_line_colors(num_colors: int) -> List[str]:
    """Return a clean bright palette based on gist_ncar, guaranteed to have `num_colors` entries."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    if int(num_colors) <= 0:
        return []

    color_map = plt.get_cmap("gist_ncar")
    def _adjust_gist_ncar_rgb(red: float, green: float, blue: float) -> Tuple[float, float, float]:
        """Keep gist_ncar's vivid hue structure while avoiding near-white and near-black colors."""
        hue, sat, val = mcolors.rgb_to_hsv((red, green, blue))
        sat = max(0.58, float(sat))
        val = min(max(0.72, float(val)), 0.95)
        red_adj, green_adj, blue_adj = mcolors.hsv_to_rgb((hue, sat, val))

        luminance = 0.2126 * red_adj + 0.7152 * green_adj + 0.0722 * blue_adj
        if luminance > 0.80:
            scale = 0.80 / max(luminance, 1e-8)
            red_adj *= scale
            green_adj *= scale
            blue_adj *= scale
        elif luminance < 0.22:
            mix = (0.22 - luminance) / max(1e-8, 1.0 - luminance)
            red_adj = red_adj + (1.0 - red_adj) * mix
            green_adj = green_adj + (1.0 - green_adj) * mix
            blue_adj = blue_adj + (1.0 - blue_adj) * mix
        return float(red_adj), float(green_adj), float(blue_adj)

    colors: List[str] = []
    # Sample enough positions across the full map and adjust every sample instead of dropping many of them.
    for idx in range(int(num_colors)):
        position = (float(idx) + 0.5) / float(int(num_colors))
        rgba = color_map(position)
        red, green, blue = _adjust_gist_ncar_rgb(float(rgba[0]), float(rgba[1]), float(rgba[2]))
        colors.append(mcolors.to_hex((red, green, blue)))
    return colors


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
    line_colors = _get_wan21_t2v_visible_line_colors(len(head_tags))
    for color_index, head_tag in enumerate(head_tags):
        head_rows = sorted(grouped_rows[head_tag], key=lambda row: int(row["step"]))
        x_steps = [int(row["step"]) for row in head_rows]
        y_values = [float(row[metric_key]) for row in head_rows]
        axis.plot(
            x_steps,
            y_values,
            linewidth=1.35,
            alpha=0.92,
            color=line_colors[color_index],
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
    line_colors = _get_wan21_t2v_visible_line_colors(len(grouped))
    for color_index, head_tag in enumerate(sorted(grouped.keys())):
        rows_for_head = sorted(grouped[head_tag], key=lambda row: int(row["frame"]))
        ys = [float(row["center_y"]) for row in rows_for_head]
        xs = [float(row["center_x"]) for row in rows_for_head]
        axis.plot(xs, ys, linewidth=1.8, alpha=0.92, color=line_colors[color_index], label=head_tag)
        axis.scatter([xs[0]], [ys[0]], s=18, color=line_colors[color_index], alpha=0.95)

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
    video_frame_count: Optional[int] = None,
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
        frame_labels = [
            _map_wan21_t2v_token_frame_to_video_frame_label(
                token_frame_idx=int(frame_index),
                token_frame_count=int(frame_count),
                video_frame_count=(
                    int(video_frame_count)
                    if video_frame_count is not None else int(frame_count)
                ),
            )
            for frame_index in frame_indices
        ]
    else:
        frame_indices, frame_labels = _resolve_wan21_t2v_viz_frame_indices(
            attention_frame_count=int(frame_count),
            video_frame_count=int(video_frame_count) if video_frame_count is not None else int(frame_count),
            num_frames=int(num_frames),
            explicit_indices=None,
        )
        frame_pairs = [
            (int(frame_index), int(frame_label))
            for frame_index, frame_label in zip(frame_indices, frame_labels)
            if 0 <= int(frame_index) < frame_count
        ]
        frame_indices = [int(frame_index) for frame_index, _ in frame_pairs]
        frame_labels = [int(frame_label) for _, frame_label in frame_pairs]

    num_panels = len(frame_indices)
    if num_panels <= 0:
        return ""
    fig_width = max(2.8 * num_panels, 8.0)
    fig, axes = plt.subplots(1, num_panels, figsize=(fig_width, 3.2))
    if num_panels == 1:
        axes = [axes]

    global_max = float(probability_map_fhw.max().item()) if probability_map_fhw.numel() > 0 else 1.0
    global_max = max(global_max, 1e-8)

    for axis, frame_index, frame_label in zip(axes, frame_indices, frame_labels):
        frame_map = probability_map_fhw[int(frame_index)].detach().cpu().float()
        center_y = float(center_f2[int(frame_index), 0].item())
        center_x = float(center_f2[int(frame_index), 1].item())
        axis.imshow(frame_map.numpy(), cmap="magma", vmin=0.0, vmax=global_max, alpha=0.92)
        axis.scatter(
            [center_x],
            [center_y],
            s=36,
            c=["#22c55e"],
            marker="o",
            edgecolors="white",
            linewidths=0.9,
        )
        axis.set_title(f"frame={int(frame_label)}", fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file


def _build_wan21_t2v_support_mask_fhw(
    probability_map_fhw: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    """Build the exact binary support masks used by support-overlap IoU."""
    if probability_map_fhw.dim() != 3:
        raise ValueError(f"Expected [F, H, W], got shape={tuple(probability_map_fhw.shape)}")
    flat_map = probability_map_fhw.reshape(probability_map_fhw.size(0), -1)
    threshold = torch.quantile(flat_map, q=float(quantile), dim=1, keepdim=True)
    return (flat_map >= threshold).reshape_as(probability_map_fhw)


def _build_wan21_t2v_motion_planning_region_mask_fhw(
    probability_map_fhw: torch.Tensor,
    quantile: float,
    min_component_area: int,
) -> torch.Tensor:
    """Build the denoised support-region mask used as a motion-planning region."""
    support_mask_fhw = _build_wan21_t2v_support_mask_fhw(
        probability_map_fhw=probability_map_fhw,
        quantile=float(quantile),
    )
    if support_mask_fhw.dim() != 3:
        raise ValueError(f"Expected [F, H, W], got shape={tuple(support_mask_fhw.shape)}")

    frame_count = int(support_mask_fhw.size(0))
    filtered_mask = torch.zeros_like(support_mask_fhw, dtype=torch.bool)
    area_threshold = max(1, int(min_component_area))
    for frame_index in range(frame_count):
        frame_mask = _build_wan21_t2v_filtered_support_mask_hw(
            binary_mask_hw=support_mask_fhw[frame_index],
            min_component_area=area_threshold,
        )
        if bool(frame_mask.any().item()):
            filtered_mask[frame_index] = frame_mask
        else:
            filtered_mask[frame_index] = support_mask_fhw[frame_index]
    return filtered_mask


def _build_wan21_t2v_motion_planning_region_mask_task(
    task: Tuple[int, int, int, torch.Tensor, float, int],
) -> Tuple[Tuple[int, int, int], torch.Tensor]:
    """Worker task used to build one motion-planning-region mask in parallel."""
    step, layer, head, probability_map_fhw, quantile, min_component_area = task
    key = (int(step), int(layer), int(head))
    mask = _build_wan21_t2v_motion_planning_region_mask_fhw(
        probability_map_fhw=probability_map_fhw,
        quantile=float(quantile),
        min_component_area=int(min_component_area),
    )
    return key, mask


def _build_wan21_t2v_head_center_extraction_task(
    task: Tuple[int, int, int, torch.Tensor, Dict[str, object]],
) -> Tuple[Tuple[int, int, int], List[Tuple[float, float]]]:
    """Worker task used to extract one raw head center trajectory."""
    step, layer, head, map_fhw, center_config = task
    key = (int(step), int(layer), int(head))
    trajectory, _ = _extract_wan21_t2v_head_trajectory_centers(
        map_fhw=map_fhw,
        center_method=str(center_config["center_method"]),
        center_power=float(center_config["center_power"]),
        center_quantile=float(center_config["center_quantile"]),
        preprocessed_center_mode=str(center_config["preprocessed_center_mode"]),
        preprocess_winsorize_quantile=float(center_config["preprocess_winsorize_quantile"]),
        preprocess_despike_quantile=float(center_config["preprocess_despike_quantile"]),
        preprocess_min_component_area=int(center_config["preprocess_min_component_area"]),
    )
    return key, trajectory


def _render_wan21_t2v_center_overlay_task(
    task: Tuple[torch.Tensor, torch.Tensor, str, str, int, Optional[int]],
) -> str:
    """Worker task used to render one center-overlay PDF."""
    probability_map_fhw, center_f2, save_file, title, num_frames, video_frame_count = task
    return _plot_wan21_t2v_head_trajectory_center_overlay(
        probability_map_fhw=probability_map_fhw,
        center_f2=center_f2,
        save_file=save_file,
        title=title,
        num_frames=int(num_frames),
        video_frame_count=video_frame_count,
    )


def _render_wan21_t2v_support_overlay_task(
    task: Tuple[torch.Tensor, str, str, int, int, Optional[int]],
) -> str:
    """Worker task used to render one support-overlay PDF."""
    binary_mask_fhw, save_file, title, num_frames, contour_min_component_area, video_frame_count = task
    return _plot_wan21_t2v_support_overlap_mask_panels(
        binary_mask_fhw=binary_mask_fhw,
        save_file=save_file,
        title=title,
        num_frames=int(num_frames),
        contour_min_component_area=int(contour_min_component_area),
        draw_contours=True,
        video_frame_count=video_frame_count,
    )


def _resolve_wan21_t2v_motion_planning_region_num_workers(requested_num_workers: int, task_count: int) -> int:
    """Resolve the effective number of worker processes for support-cache building."""
    return _resolve_wan21_t2v_num_workers(
        requested_num_workers=int(requested_num_workers),
        task_count=int(task_count),
    )


def _materialize_wan21_t2v_motion_planning_region_masks(
    probability_maps_by_step_layer_head: Dict[Tuple[int, int, int], torch.Tensor],
    cache_payload: Dict[str, object],
    cache_path: str,
    support_quantile: float,
    min_component_area: int,
    num_workers: int,
    progress_desc: str,
    cache_save_interval: int = 256,
) -> Tuple[Dict[Tuple[int, int, int], torch.Tensor], int, int]:
    """Load cached motion-planning masks and build the missing ones, optionally in parallel."""
    mask_by_key: Dict[Tuple[int, int, int], torch.Tensor] = {}
    cache_hits = 0
    cache_misses = 0
    pending_cache_writes = 0
    keys = sorted(probability_maps_by_step_layer_head.keys())
    if not keys:
        return mask_by_key, cache_hits, cache_misses

    progress_bar = None
    try:
        from tqdm import tqdm

        progress_bar = tqdm(
            total=int(len(keys)),
            desc=str(progress_desc),
            unit="head",
            leave=True,
        )
    except Exception:
        progress_bar = None

    missing_tasks: List[Tuple[int, int, int, torch.Tensor, float, int]] = []
    try:
        for step_index, layer_index, head_index in keys:
            cached_mask = _get_wan21_t2v_cached_motion_planning_region_mask(
                cache_payload=cache_payload,
                step=int(step_index),
                layer=int(layer_index),
                head=int(head_index),
            )
            if cached_mask is not None:
                mask_by_key[(int(step_index), int(layer_index), int(head_index))] = cached_mask
                cache_hits += 1
                if progress_bar is not None:
                    progress_bar.update(1)
                continue
            missing_tasks.append(
                (
                    int(step_index),
                    int(layer_index),
                    int(head_index),
                    probability_maps_by_step_layer_head[(int(step_index), int(layer_index), int(head_index))],
                    float(support_quantile),
                    int(min_component_area),
                )
            )

        effective_num_workers = _resolve_wan21_t2v_motion_planning_region_num_workers(
            requested_num_workers=int(num_workers),
            task_count=int(len(missing_tasks)),
        )
        for key, mask in _iter_wan21_t2v_parallel_results(
            tasks=missing_tasks,
            worker_fn=_build_wan21_t2v_motion_planning_region_mask_task,
            num_workers=int(effective_num_workers),
        ):
            mask_by_key[key] = mask
            _set_wan21_t2v_cached_motion_planning_region_mask(
                cache_payload=cache_payload,
                step=int(key[0]),
                layer=int(key[1]),
                head=int(key[2]),
                mask_fhw=mask,
            )
            cache_misses += 1
            pending_cache_writes += 1
            if pending_cache_writes >= int(cache_save_interval):
                _save_wan21_t2v_motion_planning_region_cache(cache_path, cache_payload)
                pending_cache_writes = 0
            if progress_bar is not None:
                progress_bar.update(1)
        if pending_cache_writes > 0:
            _save_wan21_t2v_motion_planning_region_cache(cache_path, cache_payload)
            pending_cache_writes = 0
    finally:
        if progress_bar is not None:
            progress_bar.close()
    return mask_by_key, cache_hits, cache_misses


def _build_wan21_t2v_motion_planning_region_masks_with_progress(
    probability_maps_by_step_layer_head: Dict[Tuple[int, int, int], torch.Tensor],
    cache_payload: Dict[str, object],
    cache_path: str,
    support_quantile: float,
    min_component_area: int,
    num_workers: int,
    cache_save_interval: int,
):
    """Wrapper used by the top-level stage to expose the exact user-facing progress-bar name."""
    return _materialize_wan21_t2v_motion_planning_region_masks(
        probability_maps_by_step_layer_head=probability_maps_by_step_layer_head,
        cache_payload=cache_payload,
        cache_path=cache_path,
        support_quantile=float(support_quantile),
        min_component_area=int(min_component_area),
        num_workers=int(num_workers),
        progress_desc="head_trajectory motion-planning regions",
        cache_save_interval=int(cache_save_interval),
    )


def _materialize_wan21_t2v_motion_planning_filtered_maps(
    probability_maps_by_step_layer_head: Dict[Tuple[int, int, int], torch.Tensor],
    motion_planning_region_masks_by_step_layer_head: Dict[Tuple[int, int, int], torch.Tensor],
    progress_desc: str,
) -> Dict[Tuple[int, int, int], torch.Tensor]:
    """Apply motion-planning masks to all heads with one visible progress bar."""
    filtered_probability_maps_by_key: Dict[Tuple[int, int, int], torch.Tensor] = {}
    keys = sorted(probability_maps_by_step_layer_head.keys())
    if not keys:
        return filtered_probability_maps_by_key

    progress_bar = None
    try:
        from tqdm import tqdm

        progress_bar = tqdm(
            total=int(len(keys)),
            desc=str(progress_desc),
            unit="head",
            leave=True,
        )
    except Exception:
        progress_bar = None

    try:
        for key in keys:
            raw_probability_map = probability_maps_by_step_layer_head[key]
            motion_planning_region_mask = motion_planning_region_masks_by_step_layer_head[key]
            filtered_probability_maps_by_key[key] = _apply_wan21_t2v_motion_planning_region_to_probability_map(
                probability_map_fhw=raw_probability_map,
                motion_planning_region_mask_fhw=motion_planning_region_mask,
            )
            if progress_bar is not None:
                progress_bar.update(1)
    finally:
        if progress_bar is not None:
            progress_bar.close()
    return filtered_probability_maps_by_key


def _materialize_wan21_t2v_motion_planning_filtered_centers(
    filtered_probability_maps_by_step_layer_head: Dict[Tuple[int, int, int], torch.Tensor],
    filtered_center_cache_payload: Dict[str, object],
    filtered_center_cache_path: str,
    center_config: Dict[str, object],
    num_workers: int,
    progress_desc: str,
    cache_save_interval: int = 256,
) -> Tuple[Dict[Tuple[int, int, int], torch.Tensor], int, int]:
    """Extract filtered centers from already-filtered maps with one visible progress bar."""
    filtered_center_trajectories_by_key: Dict[Tuple[int, int, int], torch.Tensor] = {}
    cache_hits = 0
    cache_misses = 0
    pending_cache_writes = 0
    keys = sorted(filtered_probability_maps_by_step_layer_head.keys())
    if not keys:
        return filtered_center_trajectories_by_key, cache_hits, cache_misses

    progress_bar = None
    try:
        from tqdm import tqdm

        progress_bar = tqdm(
            total=int(len(keys)),
            desc=str(progress_desc),
            unit="head",
            leave=True,
        )
    except Exception:
        progress_bar = None

    try:
        missing_tasks: List[Tuple[int, int, int, torch.Tensor, Dict[str, object]]] = []
        for step_index, layer_index, head_index in keys:
            key = (int(step_index), int(layer_index), int(head_index))
            cached_trajectory = _get_wan21_t2v_cached_center_trajectory(
                cache_payload=filtered_center_cache_payload,
                step=int(step_index),
                layer=int(layer_index),
                head=int(head_index),
            )
            if cached_trajectory is None:
                missing_tasks.append(
                    (
                        int(step_index),
                        int(layer_index),
                        int(head_index),
                        filtered_probability_maps_by_step_layer_head[key],
                        dict(center_config),
                    )
                )
            else:
                filtered_center_trajectories_by_key[key] = _center_trajectory_wan21_t2v_to_tensor(cached_trajectory)
                cache_hits += 1
                if progress_bar is not None:
                    progress_bar.update(1)

        for key, filtered_center_trajectory in _iter_wan21_t2v_parallel_results(
            tasks=missing_tasks,
            worker_fn=_build_wan21_t2v_head_center_extraction_task,
            num_workers=int(num_workers),
        ):
            _set_wan21_t2v_cached_center_trajectory(
                cache_payload=filtered_center_cache_payload,
                step=int(key[0]),
                layer=int(key[1]),
                head=int(key[2]),
                trajectory=filtered_center_trajectory,
            )
            filtered_center_trajectories_by_key[key] = _center_trajectory_wan21_t2v_to_tensor(
                filtered_center_trajectory
            )
            cache_misses += 1
            pending_cache_writes += 1
            if pending_cache_writes >= int(cache_save_interval):
                _save_wan21_t2v_head_trajectory_cache(
                    filtered_center_cache_path,
                    filtered_center_cache_payload,
                )
                pending_cache_writes = 0
            if progress_bar is not None:
                progress_bar.update(1)
        if pending_cache_writes > 0 and filtered_center_cache_path:
            _save_wan21_t2v_head_trajectory_cache(
                filtered_center_cache_path,
                filtered_center_cache_payload,
            )
    finally:
        if progress_bar is not None:
            progress_bar.close()
    return filtered_center_trajectories_by_key, cache_hits, cache_misses


def _extract_wan21_t2v_support_connected_components(
    binary_mask_hw: torch.Tensor,
) -> List[List[Tuple[int, int]]]:
    """Extract 4-neighborhood connected components for support-mask contour filtering."""
    if binary_mask_hw.dim() != 2:
        raise ValueError(f"Expected [H, W], got shape={tuple(binary_mask_hw.shape)}")

    token_grid_height, token_grid_width = binary_mask_hw.shape
    visited_mask = torch.zeros_like(binary_mask_hw, dtype=torch.bool)
    components: List[List[Tuple[int, int]]] = []
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for y_index in range(int(token_grid_height)):
        for x_index in range(int(token_grid_width)):
            if not bool(binary_mask_hw[y_index, x_index].item()):
                continue
            if bool(visited_mask[y_index, x_index].item()):
                continue
            visited_mask[y_index, x_index] = True
            queue = deque([(y_index, x_index)])
            current_component: List[Tuple[int, int]] = []
            while queue:
                current_y, current_x = queue.popleft()
                if not bool(binary_mask_hw[current_y, current_x].item()):
                    continue
                current_component.append((int(current_y), int(current_x)))
                for delta_y, delta_x in neighbors:
                    next_y = current_y + delta_y
                    next_x = current_x + delta_x
                    if next_y < 0 or next_y >= int(token_grid_height) or next_x < 0 or next_x >= int(token_grid_width):
                        continue
                    if bool(visited_mask[next_y, next_x].item()):
                        continue
                    visited_mask[next_y, next_x] = True
                    queue.append((next_y, next_x))
            if current_component:
                components.append(current_component)
    return components


def _build_wan21_t2v_filtered_support_mask_hw(
    binary_mask_hw: torch.Tensor,
    min_component_area: int,
) -> torch.Tensor:
    """Keep only sufficiently large 4-connected support components for contour drawing."""
    if binary_mask_hw.dim() != 2:
        raise ValueError(f"Expected [H, W], got shape={tuple(binary_mask_hw.shape)}")
    components = _extract_wan21_t2v_support_connected_components(binary_mask_hw.bool())
    filtered_mask = torch.zeros_like(binary_mask_hw, dtype=torch.bool)
    area_threshold = max(1, int(min_component_area))
    for component in components:
        if len(component) < area_threshold:
            continue
        for y_index, x_index in component:
            filtered_mask[int(y_index), int(x_index)] = True
    return filtered_mask


def _build_wan21_t2v_motion_planning_region_cache_basename(
    support_quantile: float,
    contour_min_component_area: int,
) -> str:
    """Build a descriptive cache basename for support-region masks."""
    return "_".join(
        [
            "head_trajectory_dynamics_motion_planning_region_cache",
            f"q_{_format_wan21_t2v_value_for_filename(support_quantile)}",
            f"mca_{int(contour_min_component_area)}",
        ]
    ) + ".json"


def _load_wan21_t2v_motion_planning_region_cache(cache_path: str) -> Dict[str, object]:
    """Load one motion-planning-region cache file if present."""
    candidate_paths = [cache_path, f"{cache_path}.bak"]
    for candidate_path in candidate_paths:
        if not os.path.exists(candidate_path):
            continue
        try:
            with open(candidate_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                loaded.setdefault("masks", {})
                return loaded
        except Exception:
            continue
    return {"masks": {}}


def _save_wan21_t2v_motion_planning_region_cache(cache_path: str, payload: Dict[str, object]):
    """Save one motion-planning-region cache file."""
    _ensure_dir(os.path.dirname(cache_path))
    tmp_path = f"{cache_path}.tmp"
    bak_path = f"{cache_path}.bak"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    if os.path.exists(cache_path):
        try:
            if os.path.exists(bak_path):
                os.remove(bak_path)
        except Exception:
            pass
        try:
            os.replace(cache_path, bak_path)
        except Exception:
            pass
    os.replace(tmp_path, cache_path)


def _get_wan21_t2v_cached_motion_planning_region_mask(
    cache_payload: Dict[str, object],
    step: int,
    layer: int,
    head: int,
) -> Optional[torch.Tensor]:
    """Return cached motion-planning mask if present."""
    masks = cache_payload.get("masks", {})
    if not isinstance(masks, dict):
        return None
    step_payload = masks.get(str(int(step)), {})
    layer_payload = step_payload.get(str(int(layer)), {}) if isinstance(step_payload, dict) else {}
    head_payload = layer_payload.get(str(int(head))) if isinstance(layer_payload, dict) else None
    if not isinstance(head_payload, dict):
        return None

    shape = head_payload.get("shape", [])
    positive_indices_by_frame = head_payload.get("positive_indices_by_frame", [])
    if not isinstance(shape, (list, tuple)) or len(shape) != 3:
        return None
    if not isinstance(positive_indices_by_frame, list):
        return None

    try:
        frame_count = int(shape[0])
        token_grid_height = int(shape[1])
        token_grid_width = int(shape[2])
    except Exception:
        return None

    mask = torch.zeros((frame_count, token_grid_height, token_grid_width), dtype=torch.bool)
    for frame_index, positive_indices in enumerate(positive_indices_by_frame):
        if not isinstance(positive_indices, list):
            return None
        flat = mask[frame_index].reshape(-1)
        for flat_index in positive_indices:
            try:
                flat[int(flat_index)] = True
            except Exception:
                return None
    return mask


def _set_wan21_t2v_cached_motion_planning_region_mask(
    cache_payload: Dict[str, object],
    step: int,
    layer: int,
    head: int,
    mask_fhw: torch.Tensor,
):
    """Insert or overwrite one cached motion-planning-region mask."""
    if mask_fhw.dim() != 3:
        raise ValueError(f"Expected [F, H, W], got shape={tuple(mask_fhw.shape)}")
    masks = cache_payload.setdefault("masks", {})
    step_payload = masks.setdefault(str(int(step)), {})
    layer_payload = step_payload.setdefault(str(int(layer)), {})
    layer_payload[str(int(head))] = {
        "shape": [int(mask_fhw.size(0)), int(mask_fhw.size(1)), int(mask_fhw.size(2))],
        "positive_indices_by_frame": [
            [int(index) for index in torch.nonzero(mask_fhw[frame_index].reshape(-1), as_tuple=False).reshape(-1).tolist()]
            for frame_index in range(int(mask_fhw.size(0)))
        ],
    }


def _plot_wan21_t2v_support_overlap_mask_panels(
    binary_mask_fhw: torch.Tensor,
    save_file: str,
    title: str,
    num_frames: int = 10,
    contour_min_component_area: int = 4,
    draw_contours: bool = False,
    video_frame_count: Optional[int] = None,
) -> str:
    """Visualize support-overlap binary masks with optional green connected-component contours."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if binary_mask_fhw.dim() != 3:
        raise ValueError(f"Expected [F, H, W], got shape={tuple(binary_mask_fhw.shape)}")

    frame_count = int(binary_mask_fhw.size(0))
    if frame_count <= 0:
        return ""

    if int(num_frames) <= 0 or int(num_frames) >= frame_count:
        frame_indices = list(range(frame_count))
        frame_labels = [
            _map_wan21_t2v_token_frame_to_video_frame_label(
                token_frame_idx=int(frame_index),
                token_frame_count=int(frame_count),
                video_frame_count=(
                    int(video_frame_count)
                    if video_frame_count is not None else int(frame_count)
                ),
            )
            for frame_index in frame_indices
        ]
    else:
        frame_indices, frame_labels = _resolve_wan21_t2v_viz_frame_indices(
            attention_frame_count=int(frame_count),
            video_frame_count=int(video_frame_count) if video_frame_count is not None else int(frame_count),
            num_frames=int(num_frames),
            explicit_indices=None,
        )
        frame_pairs = [
            (int(frame_index), int(frame_label))
            for frame_index, frame_label in zip(frame_indices, frame_labels)
            if 0 <= int(frame_index) < frame_count
        ]
        frame_indices = [int(frame_index) for frame_index, _ in frame_pairs]
        frame_labels = [int(frame_label) for _, frame_label in frame_pairs]

    num_panels = len(frame_indices)
    if num_panels <= 0:
        return ""
    fig_width = max(2.75 * num_panels, 8.0)
    fig, axes = plt.subplots(1, num_panels, figsize=(fig_width, 3.25), facecolor="white")
    if num_panels == 1:
        axes = [axes]

    for axis, frame_index, frame_label in zip(axes, frame_indices, frame_labels):
        frame_mask = binary_mask_fhw[int(frame_index)].detach().cpu().bool()
        axis.imshow(frame_mask.float().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        if bool(draw_contours):
            filtered_mask = _build_wan21_t2v_filtered_support_mask_hw(
                binary_mask_hw=frame_mask,
                min_component_area=int(contour_min_component_area),
            )
            if bool(filtered_mask.any().item()):
                axis.contour(
                    filtered_mask.float().numpy(),
                    levels=[0.5],
                    colors=["#22c55e"],
                    linewidths=1.6,
                )
        axis.set_title(f"frame={int(frame_label)}", fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_facecolor("white")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf", facecolor="white")
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


def _build_wan21_t2v_filtered_center_cache_basename(
    ordinary_center_config: Dict[str, object],
    reference_center_config: Dict[str, object],
    support_quantile: float,
    support_viz_contour_min_component_area: int,
    reference_step: int,
    reference_layer: int,
) -> str:
    """Build a cache basename for motion-planning filtered center trajectories."""
    ordinary_method = str(ordinary_center_config["center_method"]).strip().lower()
    if ordinary_method == "region_centroid":
        ordinary_part = "_".join(
            [
                "ordinary_head",
                "region_centroid",
                f"q_{_format_wan21_t2v_value_for_filename(ordinary_center_config['center_quantile'])}",
                f"p_{_format_wan21_t2v_value_for_filename(ordinary_center_config['center_power'])}",
            ]
        )
    elif ordinary_method == "preprocessed_component_center":
        ordinary_part = "_".join(
            [
                "ordinary_head",
                "preprocessed_component_center",
                f"mode_{_format_wan21_t2v_value_for_filename(ordinary_center_config['preprocessed_center_mode'])}",
                f"q_{_format_wan21_t2v_value_for_filename(ordinary_center_config['center_quantile'])}",
                f"p_{_format_wan21_t2v_value_for_filename(ordinary_center_config['center_power'])}",
                f"wq_{_format_wan21_t2v_value_for_filename(ordinary_center_config['preprocess_winsorize_quantile'])}",
                f"dq_{_format_wan21_t2v_value_for_filename(ordinary_center_config['preprocess_despike_quantile'])}",
                f"mca_{int(ordinary_center_config['preprocess_min_component_area'])}",
            ]
        )
    else:
        raise ValueError(
            "ordinary_center_config['center_method'] must be one of "
            "{'region_centroid', 'preprocessed_component_center'}, "
            f"got: {ordinary_center_config['center_method']}"
        )

    parts = [
        "head_trajectory_dynamics_filtered_trajectory_cache",
        ordinary_part,
        f"support_q_{_format_wan21_t2v_value_for_filename(support_quantile)}",
        f"support_mca_{int(support_viz_contour_min_component_area)}",
        f"ref_s_{int(reference_step)}",
        f"ref_l_{int(reference_layer)}",
    ]
    return "_".join(parts) + ".json"


def _build_wan21_t2v_head_trajectory_center_overlay_dir(
    output_dir: str,
    use_motion_planning_region_before_metrics: bool,
    use_preprocessed_component_center: bool,
    preprocessed_center_mode: str,
) -> str:
    """Build the center-overlay directory name with preprocessing flags."""
    mode_name = _format_wan21_t2v_value_for_filename(preprocessed_center_mode)
    return os.path.join(
        output_dir,
        "_".join(
            [
                "head_trajectory_dynamics_head_center_overlays",
                "motion_planning_region_on" if bool(use_motion_planning_region_before_metrics) else "motion_planning_region_off",
                "preprocessed_on" if bool(use_preprocessed_component_center) else "preprocessed_off",
                f"center_mode_{mode_name}",
            ]
        ),
    )


def _build_wan21_t2v_head_trajectory_metrics_output_dir(
    output_dir: str,
    hypothesis_name: str,
    use_motion_planning_region_before_metrics: bool,
    use_preprocessed_component_center: bool,
    preprocessed_center_mode: str,
) -> str:
    """Build the metric-output directory name with preprocessing flags."""
    mode_name = _format_wan21_t2v_value_for_filename(preprocessed_center_mode)
    return os.path.join(
        output_dir,
        "_".join(
            [
                "head_trajectory_dynamics_metrics",
                f"hypothesis_{_format_wan21_t2v_value_for_filename(hypothesis_name)}",
                "motion_planning_region_on" if bool(use_motion_planning_region_before_metrics) else "motion_planning_region_off",
                "preprocessed_on" if bool(use_preprocessed_component_center) else "preprocessed_off",
                f"center_mode_{mode_name}",
            ]
        ),
    )

def _load_wan21_t2v_head_trajectory_cache(cache_path: str) -> Dict[str, object]:
    """Load one trajectory-cache JSON file if it exists, otherwise create an empty cache payload."""
    candidate_paths = [cache_path, f"{cache_path}.bak"]
    for candidate_path in candidate_paths:
        if not os.path.exists(candidate_path):
            continue
        try:
            with open(candidate_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                loaded.setdefault("trajectories", {})
                return loaded
        except Exception:
            continue
    return {"trajectories": {}}

def _save_wan21_t2v_head_trajectory_cache(cache_path: str, payload: Dict[str, object]):
    """Write one trajectory-cache JSON payload to disk."""
    _ensure_dir(os.path.dirname(cache_path))
    tmp_path = f"{cache_path}.tmp"
    bak_path = f"{cache_path}.bak"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    if os.path.exists(cache_path):
        try:
            if os.path.exists(bak_path):
                os.remove(bak_path)
        except Exception:
            pass
        try:
            os.replace(cache_path, bak_path)
        except Exception:
            pass
    os.replace(tmp_path, cache_path)


def _build_wan21_t2v_head_trajectory_metrics_subdir(
    hypothesis_name: str,
    use_motion_planning_region_before_metrics: bool,
    use_preprocessed_component_center: bool,
    preprocessed_center_mode: str,
) -> str:
    """Build the metrics-output subdirectory name."""
    return "_".join(
        [
            f"hypothesis_{_format_wan21_t2v_value_for_filename(hypothesis_name)}",
            (
                "motion_planning_region_on"
                if bool(use_motion_planning_region_before_metrics)
                else "motion_planning_region_off"
            ),
            "preprocessed_on" if bool(use_preprocessed_component_center) else "preprocessed_off",
            f"center_mode_{_format_wan21_t2v_value_for_filename(preprocessed_center_mode)}",
        ]
    )


def _apply_wan21_t2v_motion_planning_region_to_probability_map(
    probability_map_fhw: torch.Tensor,
    motion_planning_region_mask_fhw: torch.Tensor,
) -> torch.Tensor:
    """Zero out probabilities outside the motion-planning region and renormalize each frame."""
    if tuple(probability_map_fhw.shape) != tuple(motion_planning_region_mask_fhw.shape):
        raise ValueError(
            "Motion-planning-region mask shape must match probability map shape, "
            f"got {tuple(probability_map_fhw.shape)} vs {tuple(motion_planning_region_mask_fhw.shape)}"
        )
    filtered_map = probability_map_fhw.detach().float() * motion_planning_region_mask_fhw.detach().float()
    frame_sums = filtered_map.reshape(filtered_map.size(0), -1).sum(dim=1)
    if bool((frame_sums <= 1e-12).any().item()):
        fallback = probability_map_fhw.detach().float()
        empty_frames = (frame_sums <= 1e-12).nonzero(as_tuple=False).reshape(-1).tolist()
        for frame_index in empty_frames:
            filtered_map[int(frame_index)] = fallback[int(frame_index)]
    return _normalize_wan21_t2v_attention_map_per_frame(filtered_map)

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


def _get_wan21_t2v_cached_reference_center_trajectory(
    cache_payload: Dict[str, object],
) -> Optional[List[Tuple[float, float]]]:
    """Return the cached filtered reference trajectory if present."""
    payload = cache_payload.get("reference_center_trajectory", None)
    if not isinstance(payload, list):
        return None
    out: List[Tuple[float, float]] = []
    for point in payload:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        out.append((float(point[0]), float(point[1])))
    return out


def _set_wan21_t2v_cached_reference_center_trajectory(
    cache_payload: Dict[str, object],
    trajectory: Sequence[Tuple[float, float]],
):
    """Insert or overwrite the cached filtered reference trajectory."""
    cache_payload["reference_center_trajectory"] = [
        [float(point_y), float(point_x)]
        for point_y, point_x in trajectory
    ]

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

def _resolve_wan21_t2v_head_trajectory_center_config(
    center_method: str,
    center_power: float,
    center_quantile: float,
    preprocessed_center_mode: str,
    preprocess_winsorize_quantile: float,
    preprocess_despike_quantile: float,
    preprocess_min_component_area: int,
) -> Dict[str, object]:
    """Normalize one center-extraction config into a validated dictionary."""
    method_name = str(center_method).strip().lower()
    if method_name not in {"region_centroid", "preprocessed_component_center"}:
        raise ValueError(
            "center_method must be one of {'region_centroid', 'preprocessed_component_center'}, "
            f"got: {center_method}"
        )

    center_mode_name = str(preprocessed_center_mode).strip().lower()
    if center_mode_name not in {"peak", "centroid", "geometric_center"}:
        raise ValueError(
            "preprocessed_center_mode must be one of {'peak', 'centroid', 'geometric_center'}, "
            f"got: {preprocessed_center_mode}"
        )

    return {
        "center_method": method_name,
        "center_power": float(center_power),
        "center_quantile": float(center_quantile),
        "preprocessed_center_mode": center_mode_name,
        "preprocess_winsorize_quantile": float(preprocess_winsorize_quantile),
        "preprocess_despike_quantile": float(preprocess_despike_quantile),
        "preprocess_min_component_area": int(preprocess_min_component_area),
    }

def _resolve_wan21_t2v_head_trajectory_reference_center_config(
    ordinary_center_config: Dict[str, object],
    reference_center_method: str,
    reference_center_power: float,
    reference_center_quantile: float,
    reference_preprocessed_center_mode: str,
    reference_preprocess_winsorize_quantile: float,
    reference_preprocess_despike_quantile: float,
    reference_preprocess_min_component_area: int,
) -> Dict[str, object]:
    """Resolve reference-center config with fallback to ordinary-head settings."""
    method_raw = str(reference_center_method).strip().lower()
    if method_raw in {"", "same_as_head"}:
        method_name = str(ordinary_center_config["center_method"])
    elif method_raw in {"region_centroid", "preprocessed_component_center"}:
        method_name = method_raw
    else:
        raise ValueError(
            "head_trajectory_dynamics_reference_center_method must be one of "
            "{'same_as_head', 'region_centroid', 'preprocessed_component_center'}, "
            f"got: {reference_center_method}"
        )

    mode_raw = str(reference_preprocessed_center_mode).strip().lower()
    if mode_raw in {"", "same_as_head"}:
        center_mode_name = str(ordinary_center_config["preprocessed_center_mode"])
    elif mode_raw in {"peak", "centroid", "geometric_center"}:
        center_mode_name = mode_raw
    else:
        raise ValueError(
            "head_trajectory_dynamics_reference_preprocessed_center_mode must be one of "
            "{'same_as_head', 'peak', 'centroid', 'geometric_center'}, "
            f"got: {reference_preprocessed_center_mode}"
        )

    return {
        "center_method": method_name,
        "center_power": (
            float(ordinary_center_config["center_power"])
            if float(reference_center_power) < 0.0
            else float(reference_center_power)
        ),
        "center_quantile": (
            float(ordinary_center_config["center_quantile"])
            if float(reference_center_quantile) < 0.0
            else float(reference_center_quantile)
        ),
        "preprocessed_center_mode": center_mode_name,
        "preprocess_winsorize_quantile": (
            float(ordinary_center_config["preprocess_winsorize_quantile"])
            if float(reference_preprocess_winsorize_quantile) < 0.0
            else float(reference_preprocess_winsorize_quantile)
        ),
        "preprocess_despike_quantile": (
            float(ordinary_center_config["preprocess_despike_quantile"])
            if float(reference_preprocess_despike_quantile) < 0.0
            else float(reference_preprocess_despike_quantile)
        ),
        "preprocess_min_component_area": (
            int(ordinary_center_config["preprocess_min_component_area"])
            if int(reference_preprocess_min_component_area) < 0
            else int(reference_preprocess_min_component_area)
        ),
    }

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


def _support_overlap_mask_iou_wan21_t2v_per_frame(
    support_mask_a_fhw: torch.Tensor,
    support_mask_b_fhw: torch.Tensor,
) -> torch.Tensor:
    """Compute frame-wise IoU between two binary motion-planning-region masks."""
    if tuple(support_mask_a_fhw.shape) != tuple(support_mask_b_fhw.shape):
        raise ValueError(
            "Expected same shapes for support-overlap IoU, "
            f"got {tuple(support_mask_a_fhw.shape)} vs {tuple(support_mask_b_fhw.shape)}"
        )
    flat_a = support_mask_a_fhw.reshape(support_mask_a_fhw.size(0), -1).bool()
    flat_b = support_mask_b_fhw.reshape(support_mask_b_fhw.size(0), -1).bool()
    intersection = (flat_a & flat_b).sum(dim=1).float()
    union = (flat_a | flat_b).sum(dim=1).float().clamp_min(1.0)
    return intersection / union


def _compute_wan21_t2v_head_trajectory_distance(
    metric_name: str,
    probability_map_a_fhw: torch.Tensor,
    probability_map_b_fhw: torch.Tensor,
    center_traj_a: torch.Tensor,
    center_traj_b: torch.Tensor,
    support_quantile: float,
    use_motion_planning_region_for_support_overlap: bool = False,
    support_mask_a_fhw: Optional[torch.Tensor] = None,
    support_mask_b_fhw: Optional[torch.Tensor] = None,
) -> float:
    """Compute one scalar head-to-head distance using the requested trajectory metric."""
    metric_name = str(metric_name).strip().lower()
    if metric_name == "js":
        return float(_js_wan21_t2v_distance_per_frame(probability_map_a_fhw, probability_map_b_fhw).mean().item())
    if metric_name == "hellinger":
        return float(_hellinger_wan21_t2v_distance_per_frame(probability_map_a_fhw, probability_map_b_fhw).mean().item())
    if metric_name == "wasserstein_map":
        return float(
            _marginal_wasserstein_wan21_t2v_distance_per_frame(probability_map_a_fhw, probability_map_b_fhw).mean().item()
        )
    if metric_name == "support_overlap":
        if bool(use_motion_planning_region_for_support_overlap):
            if support_mask_a_fhw is None or support_mask_b_fhw is None:
                raise ValueError("support_overlap requires explicit motion-planning-region masks.")
            support_iou = _support_overlap_mask_iou_wan21_t2v_per_frame(
                support_mask_a_fhw,
                support_mask_b_fhw,
            )
        else:
            support_iou = _support_overlap_iou_wan21_t2v_per_frame(
                probability_map_a_fhw,
                probability_map_b_fhw,
                quantile=float(support_quantile),
            )
        return float((1.0 - support_iou).mean().item())
    if metric_name == "center_l2":
        return float(_trajectory_distance_wan21_t2v_soft_centers(center_traj_a, center_traj_b))
    raise ValueError(
        "head_trajectory_dynamics_attractor_distance_metric must be one of "
        "{'js', 'hellinger', 'wasserstein_map', 'support_overlap', 'center_l2'}."
    )


def _load_wan21_t2v_csv_rows(csv_path: str) -> List[Dict[str, str]]:
    """Load one CSV file into a list of row dictionaries."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing CSV required for plotting: {csv_path}")
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        return [dict(row) for row in reader]


def _load_wan21_t2v_json_if_exists(json_path: str) -> Dict[str, object]:
    """Load one JSON file if present, otherwise return an empty dict."""
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return loaded if isinstance(loaded, dict) else {}


def _maybe_skip_wan21_t2v_existing_plot(save_file: str, skip_existing_plots: bool) -> bool:
    """Return True when one plot should be skipped because it already exists."""
    return bool(skip_existing_plots) and os.path.exists(save_file)


def _resolve_wan21_t2v_overlay_specs(
    probability_maps_by_step_layer_head: Dict[Tuple[int, int, int], torch.Tensor],
    requested_head_set: Sequence[Tuple[int, int]],
    requested_center_viz_head_set: Sequence[Tuple[int, int]],
    requested_support_viz_head_set: Sequence[Tuple[int, int]],
    head_trajectory_dynamics_center_viz_enable: bool,
    head_trajectory_dynamics_center_viz_step: int,
    head_trajectory_dynamics_center_viz_layer: int,
    head_trajectory_dynamics_support_viz_enable: bool,
    head_trajectory_dynamics_support_viz_step: int,
    head_trajectory_dynamics_support_viz_layer: int,
) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    """Resolve which `(step, layer, head)` specs should be rendered for center/support overlays."""
    requested_head_lookup = set((int(layer_idx), int(head_idx)) for layer_idx, head_idx in requested_head_set)
    requested_center_lookup = set((int(layer_idx), int(head_idx)) for layer_idx, head_idx in requested_center_viz_head_set)
    requested_support_lookup = set((int(layer_idx), int(head_idx)) for layer_idx, head_idx in requested_support_viz_head_set)
    all_available_heads_lookup = set(
        (int(layer_idx), int(head_idx))
        for _, layer_idx, head_idx in probability_maps_by_step_layer_head.keys()
    )

    center_overlay_specs: List[Tuple[int, int, int]] = []
    support_viz_specs: List[Tuple[int, int, int]] = []

    if bool(head_trajectory_dynamics_center_viz_enable):
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
            if requested_center_lookup:
                candidate_heads = [
                    head_idx
                    for head_idx in candidate_heads
                    if (selected_layer, int(head_idx)) in requested_center_lookup
                ]
            center_overlay_specs = [
                (selected_step, selected_layer, int(head_idx))
                for head_idx in candidate_heads
            ]
        elif requested_center_lookup:
            center_overlay_specs = sorted(
                [
                    (int(step_idx), int(layer_idx), int(head_idx))
                    for step_idx, layer_idx, head_idx in probability_maps_by_step_layer_head.keys()
                    if (int(layer_idx), int(head_idx)) in requested_center_lookup
                ]
            )
        elif requested_head_lookup:
            center_overlay_specs = sorted(
                [
                    (int(step_idx), int(layer_idx), int(head_idx))
                    for step_idx, layer_idx, head_idx in probability_maps_by_step_layer_head.keys()
                    if (int(layer_idx), int(head_idx)) in requested_head_lookup
                ]
            )
        else:
            center_overlay_specs = sorted(
                [
                    (int(step_idx), int(layer_idx), int(head_idx))
                    for step_idx, layer_idx, head_idx in probability_maps_by_step_layer_head.keys()
                    if (int(layer_idx), int(head_idx)) in all_available_heads_lookup
                ]
            )

    if bool(head_trajectory_dynamics_support_viz_enable):
        explicit_support_viz = (
            int(head_trajectory_dynamics_support_viz_step) >= 1
            and int(head_trajectory_dynamics_support_viz_layer) >= 0
        )
        if explicit_support_viz:
            selected_step = int(head_trajectory_dynamics_support_viz_step)
            selected_layer = int(head_trajectory_dynamics_support_viz_layer)
            candidate_heads = sorted(
                head_idx
                for step_idx, layer_idx, head_idx in probability_maps_by_step_layer_head.keys()
                if int(step_idx) == selected_step and int(layer_idx) == selected_layer
            )
            if requested_support_lookup:
                candidate_heads = [
                    head_idx
                    for head_idx in candidate_heads
                    if (selected_layer, int(head_idx)) in requested_support_lookup
                ]
            support_viz_specs = [
                (selected_step, selected_layer, int(head_idx))
                for head_idx in candidate_heads
            ]
        elif requested_support_lookup:
            support_viz_specs = sorted(
                [
                    (int(step_idx), int(layer_idx), int(head_idx))
                    for step_idx, layer_idx, head_idx in probability_maps_by_step_layer_head.keys()
                    if (int(layer_idx), int(head_idx)) in requested_support_lookup
                ]
            )
        elif requested_head_lookup:
            support_viz_specs = sorted(
                [
                    (int(step_idx), int(layer_idx), int(head_idx))
                    for step_idx, layer_idx, head_idx in probability_maps_by_step_layer_head.keys()
                    if (int(layer_idx), int(head_idx)) in requested_head_lookup
                ]
            )
        else:
            support_viz_specs = sorted(
                [
                    (int(step_idx), int(layer_idx), int(head_idx))
                    for step_idx, layer_idx, head_idx in probability_maps_by_step_layer_head.keys()
                    if (int(layer_idx), int(head_idx)) in all_available_heads_lookup
                ]
            )

    return center_overlay_specs, support_viz_specs


def _render_wan21_t2v_head_trajectory_overlays(
    *,
    probability_maps_by_step_layer_head: Dict[Tuple[int, int, int], torch.Tensor],
    center_trajectories_by_step_layer_head: Dict[Tuple[int, int, int], torch.Tensor],
    output_dir: str,
    requested_head_set: Sequence[Tuple[int, int]],
    requested_center_viz_head_set: Sequence[Tuple[int, int]],
    requested_support_viz_head_set: Sequence[Tuple[int, int]],
    head_trajectory_dynamics_center_viz_enable: bool,
    head_trajectory_dynamics_center_viz_step: int,
    head_trajectory_dynamics_center_viz_layer: int,
    head_trajectory_dynamics_center_viz_num_frames: int,
    head_trajectory_dynamics_support_viz_enable: bool,
    head_trajectory_dynamics_support_viz_step: int,
    head_trajectory_dynamics_support_viz_layer: int,
    head_trajectory_dynamics_support_viz_num_frames: int,
    head_trajectory_dynamics_support_viz_contour_min_component_area: int,
    head_trajectory_dynamics_support_quantile: float,
    head_trajectory_dynamics_skip_existing_plots: bool,
    reuse_video_frame_count: int,
    overlay_num_workers: int,
    use_motion_planning_region_before_metrics: bool,
    use_preprocessed_component_center: bool,
    preprocessed_center_mode: str,
) -> Tuple[List[str], str, str, int, int]:
    """Render center/support overlays from raw maps, raw centers, and precomputed support masks."""
    center_overlay_dir = _build_wan21_t2v_head_trajectory_center_overlay_dir(
        output_dir=output_dir,
        use_motion_planning_region_before_metrics=bool(use_motion_planning_region_before_metrics),
        use_preprocessed_component_center=bool(use_preprocessed_component_center),
        preprocessed_center_mode=str(preprocessed_center_mode),
    )
    support_overlap_mask_dir = os.path.join(output_dir, "head_trajectory_dynamics_support_overlap_masks")
    center_overlay_specs, support_viz_specs = _resolve_wan21_t2v_overlay_specs(
        probability_maps_by_step_layer_head=probability_maps_by_step_layer_head,
        requested_head_set=requested_head_set,
        requested_center_viz_head_set=requested_center_viz_head_set,
        requested_support_viz_head_set=requested_support_viz_head_set,
        head_trajectory_dynamics_center_viz_enable=bool(head_trajectory_dynamics_center_viz_enable),
        head_trajectory_dynamics_center_viz_step=int(head_trajectory_dynamics_center_viz_step),
        head_trajectory_dynamics_center_viz_layer=int(head_trajectory_dynamics_center_viz_layer),
        head_trajectory_dynamics_support_viz_enable=bool(head_trajectory_dynamics_support_viz_enable),
        head_trajectory_dynamics_support_viz_step=int(head_trajectory_dynamics_support_viz_step),
        head_trajectory_dynamics_support_viz_layer=int(head_trajectory_dynamics_support_viz_layer),
    )
    plot_paths: List[str] = []
    center_overlay_progress_bar = None
    support_overlay_progress_bar = None
    if int(len(center_overlay_specs)) > 0:
        try:
            from tqdm import tqdm
            center_overlay_progress_bar = tqdm(
                total=int(len(center_overlay_specs)),
                desc="head_trajectory center overlays",
                unit="plot",
                leave=True,
            )
        except Exception:
            center_overlay_progress_bar = None
    if int(len(support_viz_specs)) > 0:
        try:
            from tqdm import tqdm
            support_overlay_progress_bar = tqdm(
                total=int(len(support_viz_specs)),
                desc="head_trajectory support overlays",
                unit="plot",
                leave=True,
            )
        except Exception:
            support_overlay_progress_bar = None

    center_overlay_tasks: List[Tuple[torch.Tensor, torch.Tensor, str, str, int, Optional[int]]] = []
    support_overlay_tasks: List[Tuple[torch.Tensor, str, str, int, int, Optional[int]]] = []

    try:
        for step_index, layer_index, head_idx in center_overlay_specs:
            probability_map = probability_maps_by_step_layer_head[(int(step_index), int(layer_index), int(head_idx))]
            center_trajectory = center_trajectories_by_step_layer_head[(int(step_index), int(layer_index), int(head_idx))]
            save_file = os.path.join(
                center_overlay_dir,
                f"step_{int(step_index):03d}",
                f"layer_{int(layer_index):02d}",
                f"center_overlay_step_{int(step_index):03d}_layer_{int(layer_index):02d}_head_{int(head_idx):02d}.pdf",
            )
            if _maybe_skip_wan21_t2v_existing_plot(save_file, bool(head_trajectory_dynamics_skip_existing_plots)):
                plot_paths.append(save_file)
                if center_overlay_progress_bar is not None:
                    center_overlay_progress_bar.update(1)
            else:
                center_overlay_tasks.append(
                    (
                        probability_map,
                        center_trajectory,
                        save_file,
                        f"Center Overlay | step={int(step_index)} layer={int(layer_index)} head={int(head_idx)}",
                        int(head_trajectory_dynamics_center_viz_num_frames),
                        int(reuse_video_frame_count),
                    )
                )

        for step_index, layer_index, head_idx in support_viz_specs:
            support_mask_fhw = _build_wan21_t2v_support_mask_fhw(
                probability_maps_by_step_layer_head[(int(step_index), int(layer_index), int(head_idx))],
                quantile=float(head_trajectory_dynamics_support_quantile),
            )
            mask_dir = os.path.join(
                support_overlap_mask_dir,
                f"step_{int(step_index):03d}",
                f"layer_{int(layer_index):02d}",
            )
            contour_save_file = os.path.join(
                mask_dir,
                f"support_mask_contour_step_{int(step_index):03d}_layer_{int(layer_index):02d}_head_{int(head_idx):02d}.pdf",
            )
            if _maybe_skip_wan21_t2v_existing_plot(contour_save_file, bool(head_trajectory_dynamics_skip_existing_plots)):
                plot_paths.append(contour_save_file)
                if support_overlay_progress_bar is not None:
                    support_overlay_progress_bar.update(1)
            else:
                support_overlay_tasks.append(
                    (
                        support_mask_fhw,
                        contour_save_file,
                        (
                        f"Support Mask + Contour | step={int(step_index)} layer={int(layer_index)} "
                        f"head={int(head_idx)} q={float(head_trajectory_dynamics_support_quantile):.3f}"
                        ),
                        int(head_trajectory_dynamics_support_viz_num_frames),
                        int(head_trajectory_dynamics_support_viz_contour_min_component_area),
                        int(reuse_video_frame_count),
                    )
                )

        for plot_path in _iter_wan21_t2v_parallel_results(
            tasks=center_overlay_tasks,
            worker_fn=_render_wan21_t2v_center_overlay_task,
            num_workers=int(overlay_num_workers),
        ):
            if plot_path:
                plot_paths.append(plot_path)
            if center_overlay_progress_bar is not None:
                center_overlay_progress_bar.update(1)

        for plot_path in _iter_wan21_t2v_parallel_results(
            tasks=support_overlay_tasks,
            worker_fn=_render_wan21_t2v_support_overlay_task,
            num_workers=int(overlay_num_workers),
        ):
            if plot_path:
                plot_paths.append(plot_path)
            if support_overlay_progress_bar is not None:
                support_overlay_progress_bar.update(1)
    finally:
        if center_overlay_progress_bar is not None:
            center_overlay_progress_bar.close()
        if support_overlay_progress_bar is not None:
            support_overlay_progress_bar.close()

    return (
        plot_paths,
        center_overlay_dir,
        support_overlap_mask_dir,
        int(len(center_overlay_specs)),
        int(len(support_viz_specs)),
    )


def _infer_wan21_t2v_head_trajectory_distance_metrics(
    consensus_rows: Sequence[Dict[str, object]],
    reference_distance_rows: Sequence[Dict[str, object]],
    convergence_rows: Sequence[Dict[str, object]],
) -> List[str]:
    """Infer which distance metrics are present in saved CSV rows."""
    metric_candidates = ["js", "hellinger", "wasserstein_map", "support_overlap", "center_l2"]
    requested = []
    for metric_name in metric_candidates:
        consensus_key = f"{metric_name}_consensus"
        reference_key = f"{metric_name}_reference_distance"
        if any(str(row.get(consensus_key, "")).strip() != "" for row in consensus_rows):
            requested.append(metric_name)
            continue
        if any(str(row.get(reference_key, "")).strip() != "" for row in reference_distance_rows):
            requested.append(metric_name)
            continue
        if any(str(row.get("metric", "")).strip().lower() == metric_name for row in convergence_rows):
            requested.append(metric_name)
    return requested


def _infer_wan21_t2v_attractor_distance_metrics(
    attractor_rows: Sequence[Dict[str, object]],
) -> List[str]:
    """Infer which attractor distance metrics are actually present in saved CSV rows."""
    metric_names = []
    for row in attractor_rows:
        metric_name = str(row.get("attractor_distance_metric", "")).strip().lower()
        if not metric_name:
            metric_name = "center_l2"
        if metric_name not in metric_names:
            metric_names.append(metric_name)
    return metric_names


def _render_wan21_t2v_head_trajectory_metric_plots(
    consensus_rows: Sequence[Dict[str, object]],
    attractor_rows: Sequence[Dict[str, object]],
    reference_distance_rows: Sequence[Dict[str, object]],
    convergence_rows: Sequence[Dict[str, object]],
    output_dir: str,
    requested_distance_metrics: Sequence[str],
    skip_existing_plots: bool = False,
) -> List[str]:
    """Render the metric plots that depend only on saved CSV rows."""
    plot_paths: List[str] = []
    plots_dir = os.path.join(output_dir, "head_trajectory_dynamics_plots")
    plot_specs_total = 0
    available_layers = sorted(
        {
            int(row["layer"])
            for row in list(consensus_rows) + list(attractor_rows) + list(reference_distance_rows) + list(convergence_rows)
            if str(row.get("layer", "")).strip() != ""
        }
    )

    for metric_name in requested_distance_metrics:
        plot_specs_total += len(available_layers)
        plot_specs_total += 1
    normalized_attractor_rows = []
    for row in attractor_rows:
        normalized_row = dict(row)
        metric_name = str(normalized_row.get("attractor_distance_metric", "")).strip().lower()
        normalized_row["attractor_distance_metric"] = metric_name if metric_name else "center_l2"
        normalized_attractor_rows.append(normalized_row)
    attractor_metric_names = sorted(
        {
            str(row["attractor_distance_metric"])
            for row in normalized_attractor_rows
            if str(row.get("attractor_distance_metric", "")).strip() != ""
        }
    )
    attractor_methods = sorted(set(str(row["attractor_method"]) for row in normalized_attractor_rows))
    plot_specs_total += len(attractor_metric_names) * len(attractor_methods) * (1 + len(available_layers))
    metric_to_reference_key = {
        "js": "js_reference_distance",
        "hellinger": "hellinger_reference_distance",
        "wasserstein_map": "wasserstein_map_reference_distance",
        "support_overlap": "support_overlap_reference_distance",
        "center_l2": "center_l2_reference_distance",
    }
    plot_specs_total += sum(
        len(available_layers)
        for metric_name in metric_to_reference_key.keys()
        if metric_name in requested_distance_metrics
    )

    plot_progress_bar = None
    if plot_specs_total > 0:
        try:
            from tqdm import tqdm
            plot_progress_bar = tqdm(
                total=int(plot_specs_total),
                desc="head_trajectory_dynamics plots",
                unit="plot",
                leave=True,
            )
        except Exception:
            plot_progress_bar = None

    def _mark_plot_done():
        if plot_progress_bar is not None:
            plot_progress_bar.update(1)

    try:
        for metric_name in requested_distance_metrics:
            for layer_index in available_layers:
                layer_rows = [row for row in consensus_rows if int(row["layer"]) == int(layer_index)]
                if not layer_rows:
                    _mark_plot_done()
                    continue
                save_file = os.path.join(
                    plots_dir,
                    "consensus_curves",
                    metric_name,
                    f"consensus_layer_{int(layer_index):02d}_{metric_name}.pdf",
                )
                if _maybe_skip_wan21_t2v_existing_plot(save_file, skip_existing_plots):
                    plot_paths.append(save_file)
                else:
                    plot_path = _plot_wan21_t2v_head_trajectory_dynamics_curve(
                        rows=layer_rows,
                        save_file=save_file,
                        metric_key=f"{metric_name}_consensus",
                        title=f"Head Trajectory Consensus ({metric_name}) | layer={int(layer_index)}",
                        y_label=f"{metric_name} consensus",
                    )
                    if plot_path:
                        plot_paths.append(plot_path)
                _mark_plot_done()

            save_file = os.path.join(
                plots_dir,
                "consensus_heatmaps",
                f"consensus_heatmap_{metric_name}.pdf",
            )
            if _maybe_skip_wan21_t2v_existing_plot(save_file, skip_existing_plots):
                plot_paths.append(save_file)
            else:
                heatmap_path = _plot_wan21_t2v_head_trajectory_dynamics_heatmap(
                    matrix_rows=consensus_rows,
                    save_file=save_file,
                    title=f"Head Trajectory Consensus Heatmap ({metric_name})",
                    row_key="layer",
                    col_key="step",
                    value_key=f"{metric_name}_consensus",
                    row_label="layer",
                    col_label="diffusion step",
                )
                if heatmap_path:
                    plot_paths.append(heatmap_path)
            _mark_plot_done()

        for attractor_metric_name in attractor_metric_names:
            for method_name in attractor_methods:
                method_attractor_rows = [
                    row
                    for row in normalized_attractor_rows
                    if str(row["attractor_method"]) == method_name
                    and str(row["attractor_distance_metric"]) == str(attractor_metric_name)
                ]
                if not method_attractor_rows:
                    _mark_plot_done()
                    for _ in available_layers:
                        _mark_plot_done()
                    continue

                save_file = os.path.join(
                    plots_dir,
                    "attractor_curves",
                    attractor_metric_name,
                    method_name,
                    "attractor_all_heads.pdf",
                )
                if _maybe_skip_wan21_t2v_existing_plot(save_file, skip_existing_plots):
                    plot_paths.append(save_file)
                else:
                    plot_path = _plot_wan21_t2v_head_trajectory_dynamics_all_heads_curve(
                        rows=method_attractor_rows,
                        save_file=save_file,
                        metric_key="attractor_score_mean",
                        title=(
                            f"Head Attractor Score ({method_name}, metric={str(attractor_metric_name)}) | "
                            "all analyzed heads"
                        ),
                        y_label="attractor score",
                    )
                    if plot_path:
                        plot_paths.append(plot_path)
                _mark_plot_done()

                for layer_index in available_layers:
                    layer_attractor_rows = [
                        row for row in method_attractor_rows
                        if int(row["layer"]) == int(layer_index)
                    ]
                    if not layer_attractor_rows:
                        _mark_plot_done()
                        continue
                    by_head: Dict[str, List[Dict[str, object]]] = defaultdict(list)
                    for row in layer_attractor_rows:
                        by_head[str(row["head_tag"])].append(row)

                    save_file = os.path.join(
                        plots_dir,
                        "attractor_curves",
                        attractor_metric_name,
                        method_name,
                        f"attractor_layer_{int(layer_index):02d}.pdf",
                    )
                    if _maybe_skip_wan21_t2v_existing_plot(save_file, skip_existing_plots):
                        plot_paths.append(save_file)
                        _mark_plot_done()
                        continue

                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, axis = plt.subplots(1, 1, figsize=(8.2, 5.0))
                    head_tags = sorted(by_head.keys())
                    line_colors = _get_wan21_t2v_visible_line_colors(len(head_tags))
                    for color_index, head_tag in enumerate(head_tags):
                        head_rows = sorted(by_head[head_tag], key=lambda row: int(row["step"]))
                        x_steps = [int(row["step"]) for row in head_rows]
                        y_values = [float(row["attractor_score_mean"]) for row in head_rows]
                        axis.plot(
                            x_steps,
                            y_values,
                            linewidth=1.4,
                            alpha=0.92,
                            color=line_colors[color_index],
                            label=head_tag,
                        )
                    axis.set_title(
                        f"Head Attractor Score ({method_name}, metric={str(attractor_metric_name)}) | "
                        f"layer={int(layer_index)}"
                    )
                    axis.set_xlabel("diffusion step")
                    axis.set_ylabel("attractor score")
                    axis.grid(alpha=0.22, linestyle="--")
                    if len(head_tags) <= 20:
                        axis.legend(fontsize=7, ncol=2)
                    fig.tight_layout()
                    _ensure_dir(os.path.dirname(save_file))
                    fig.savefig(save_file, format="pdf")
                    plt.close(fig)
                    plot_paths.append(save_file)
                    _mark_plot_done()

        for metric_name, metric_key in metric_to_reference_key.items():
            if metric_name not in requested_distance_metrics:
                continue
            normalized_curve_rows = []
            raw_curve_rows = []
            per_head_rows: Dict[Tuple[int, int], List[Dict[str, object]]] = defaultdict(list)
            for row in reference_distance_rows:
                if str(row.get(metric_key, "")).strip() == "":
                    continue
                per_head_rows[(int(row["layer"]), int(row["head"]))].append(row)
            for (layer_index, head_index), head_rows in per_head_rows.items():
                ordered_rows = sorted(head_rows, key=lambda row: int(row["step"]))
                distances = [float(row[metric_key]) for row in ordered_rows]
                if not distances:
                    continue
                initial_distance = float(distances[0])
                final_distance = float(distances[-1])
                gap = float(initial_distance - final_distance)
                for row, distance_value in zip(ordered_rows, distances):
                    if abs(gap) > 1e-8:
                        normalized_value = float((distance_value - final_distance) / gap)
                    else:
                        normalized_value = 0.0
                    normalized_curve_rows.append(
                        {
                            "step": int(row["step"]),
                            "layer": int(layer_index),
                            "head": int(head_index),
                            "head_tag": f"L{int(layer_index)}H{int(head_index)}",
                            "value": normalized_value,
                        }
                    )
                    raw_curve_rows.append(
                        {
                            "step": int(row["step"]),
                            "layer": int(layer_index),
                            "head": int(head_index),
                            "head_tag": f"L{int(layer_index)}H{int(head_index)}",
                            "value": float(distance_value),
                        }
                    )

            for layer_index in available_layers:
                layer_reference_curve_rows = [
                    row
                    for row in reference_distance_rows
                    if int(row["layer"]) == int(layer_index) and str(row.get(metric_key, "")).strip() != ""
                ]
                save_file = os.path.join(
                    plots_dir,
                    "reference_distance_curves",
                    metric_name,
                    f"reference_distance_layer_{int(layer_index):02d}.pdf",
                )
                if layer_reference_curve_rows:
                    if _maybe_skip_wan21_t2v_existing_plot(save_file, skip_existing_plots):
                        plot_paths.append(save_file)
                    else:
                        curve_path = _plot_wan21_t2v_head_trajectory_dynamics_multihead_curve(
                            rows=layer_reference_curve_rows,
                            save_file=save_file,
                            metric_key=metric_key,
                            title=f"Reference Distance Curves ({metric_name}) | layer={int(layer_index)}",
                            y_label=f"{metric_name} reference distance",
                        )
                        if curve_path:
                            plot_paths.append(curve_path)
                _mark_plot_done()
    finally:
        if plot_progress_bar is not None:
            plot_progress_bar.close()

    return plot_paths


def _plot_wan21_t2v_head_trajectory_dynamics_convergence_curve(
    rows: Sequence[Dict[str, object]],
    save_file: str,
    title: str,
    y_label: str,
    curve_group_key: Optional[str] = None,
):
    """Plot convergence curves from rows containing `step` and `value`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return ""

    grouped_rows: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    if curve_group_key is None:
        grouped_rows["all"] = list(rows)
    else:
        for row in rows:
            grouped_rows[str(row[curve_group_key])].append(row)

    fig, axis = plt.subplots(1, 1, figsize=(8.4, 5.0))
    group_names = sorted(grouped_rows.keys(), key=lambda x: (x != "all", x))
    line_colors = _get_wan21_t2v_visible_line_colors(len(group_names))
    for color_index, group_name in enumerate(group_names):
        step_to_values: Dict[int, List[float]] = defaultdict(list)
        for row in grouped_rows[group_name]:
            step_to_values[int(row["step"])].append(float(row["value"]))
        x_steps = sorted(step_to_values.keys())
        y_values = [
            float(sum(step_to_values[step_value]) / len(step_to_values[step_value]))
            for step_value in x_steps
        ]
        axis.plot(
            x_steps,
            y_values,
            linewidth=1.8,
            alpha=0.96,
            color=line_colors[color_index],
            label=group_name,
        )

    axis.set_title(title)
    axis.set_xlabel("diffusion step")
    axis.set_ylabel(y_label)
    axis.grid(alpha=0.22, linestyle="--")
    if curve_group_key is not None and len(group_names) <= 16:
        axis.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()

    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file


def _plot_wan21_t2v_head_trajectory_dynamics_all_heads_curve(
    rows: Sequence[Dict[str, object]],
    save_file: str,
    metric_key: str,
    title: str,
    y_label: str,
):
    """Plot one global curve figure with one line per LxHy head tag across all layers."""
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
    line_colors = _get_wan21_t2v_visible_line_colors(len(head_tags))
    fig, axis = plt.subplots(1, 1, figsize=(10.2, 5.8))
    for color_index, head_tag in enumerate(head_tags):
        head_rows = sorted(grouped_rows[head_tag], key=lambda row: int(row["step"]))
        x_steps = [int(row["step"]) for row in head_rows]
        y_values = [float(row[metric_key]) for row in head_rows]
        axis.plot(
            x_steps,
            y_values,
            linewidth=1.15,
            alpha=0.90,
            color=line_colors[color_index],
            label=head_tag,
        )

    axis.set_title(title)
    axis.set_xlabel("diffusion step")
    axis.set_ylabel(y_label)
    axis.grid(alpha=0.22, linestyle="--")
    if len(head_tags) <= 32:
        axis.legend(fontsize=6.2, ncol=4)
    fig.tight_layout()

    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file

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
    head_trajectory_dynamics_attractor_distance_metrics: Sequence[str] = tuple(),
    head_trajectory_dynamics_center_method: str = "region_centroid",
    head_trajectory_dynamics_center_power: float = 1.5,
    head_trajectory_dynamics_center_quantile: float = 0.8,
    head_trajectory_dynamics_preprocessed_center_mode: str = "geometric_center",
    head_trajectory_dynamics_preprocess_winsorize_quantile: float = 0.995,
    head_trajectory_dynamics_preprocess_despike_quantile: float = 0.98,
    head_trajectory_dynamics_preprocess_min_component_area: int = 2,
    head_trajectory_dynamics_reference_center_method: str = "same_as_head",
    head_trajectory_dynamics_reference_center_power: float = -1.0,
    head_trajectory_dynamics_reference_center_quantile: float = -1.0,
    head_trajectory_dynamics_reference_preprocessed_center_mode: str = "same_as_head",
    head_trajectory_dynamics_reference_preprocess_winsorize_quantile: float = -1.0,
    head_trajectory_dynamics_reference_preprocess_despike_quantile: float = -1.0,
    head_trajectory_dynamics_reference_preprocess_min_component_area: int = -1,
    head_trajectory_dynamics_center_viz_enable: bool = False,
    head_trajectory_dynamics_center_viz_step: int = -1,
    head_trajectory_dynamics_center_viz_layer: int = -1,
    head_trajectory_dynamics_center_viz_heads: Sequence[str] = tuple(),
    head_trajectory_dynamics_center_viz_num_frames: int = 10,
    head_trajectory_dynamics_support_viz_enable: bool = False,
    head_trajectory_dynamics_support_viz_step: int = -1,
    head_trajectory_dynamics_support_viz_layer: int = -1,
    head_trajectory_dynamics_support_viz_heads: Sequence[str] = tuple(),
    head_trajectory_dynamics_support_viz_num_frames: int = 10,
    head_trajectory_dynamics_support_viz_contour_min_component_area: int = 4,
    head_trajectory_dynamics_support_cache_num_workers: int = 0,
    head_trajectory_dynamics_center_cache_num_workers: int = 0,
    head_trajectory_dynamics_overlay_num_workers: int = 0,
    head_trajectory_dynamics_cache_save_interval: int = 512,
    head_trajectory_dynamics_hypothesis: str = "attractor",
    head_trajectory_dynamics_traj_type: str = "",
    head_trajectory_dynamics_use_motion_planning_region_before_metrics: bool = False,
    head_trajectory_dynamics_plot_only_from_csv: bool = False,
    head_trajectory_dynamics_overlay_only: bool = False,
    head_trajectory_dynamics_skip_existing_plots: bool = True,
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
        head_trajectory_dynamics_attractor_distance_metrics: distance metrics used by
            attractor analysis when measuring whether followers move closer to the
            current leader head. Empty means all supported metrics.
        head_trajectory_dynamics_center_method: ordinary-head center method, one of
            `region_centroid` or `preprocessed_component_center`.
        head_trajectory_dynamics_center_power/head_trajectory_dynamics_center_quantile:
            ordinary-head center extraction parameters.
        head_trajectory_dynamics_preprocessed_center_mode: for the ordinary-head preprocessed method,
            choose `peak`, `centroid`, or `geometric_center`.
        head_trajectory_dynamics_reference_center_*:
            optional reference-trajectory center config. When left as `same_as_head`
            / negative sentinel values, fallback to ordinary-head settings.
        head_trajectory_dynamics_center_viz_enable: whether to render per-head
            center-overlay PDFs at all.
        head_trajectory_dynamics_center_viz_step/layer/heads: optional selection for per-head
            center-overlay PDFs used to inspect center quality.
        head_trajectory_dynamics_support_viz_*: optional selection for per-head
            support-overlap mask / contour PDFs used to inspect the quantile-thresholded support region.
        head_trajectory_dynamics_hypothesis: hypothesis label used to name the metrics subdirectory.
        head_trajectory_dynamics_use_motion_planning_region_before_metrics: if true, metrics are
            computed from attention maps masked by the contour-filtered support region.
        head_trajectory_dynamics_support_cache_num_workers: number of CPU worker processes used
            to build missing motion-planning-region masks. A non-positive value means use os.cpu_count().
        head_trajectory_dynamics_center_cache_num_workers: number of CPU worker processes used
            to extract missing raw head center trajectories. A non-positive value means use os.cpu_count().
        head_trajectory_dynamics_overlay_num_workers: number of CPU worker processes used
            to render center/support overlay PDFs. A non-positive value means use os.cpu_count().
        head_trajectory_dynamics_cache_save_interval: flush caches every N newly materialized
            entries. Larger values reduce disk-write frequency but increase the amount of work
            lost if the run is interrupted mid-cache-build.
        head_trajectory_dynamics_traj_type: trajectory subset label used by the bash wrapper to
            record which head family was selected for the run.
        head_trajectory_dynamics_overlay_only: reuse saved cross-attention maps and center cache to
            render overlays only, skipping metric recomputation and CSV metric plotting.

    Outputs:
        CSV files for pairwise distances, consensus curves, attractor scores, final-trajectory distance,
        plus summary JSON and PDF visualizations.
    """
    default_video_frame_count = max(1, int(frame_num))
    del wan21_root, ckpt_dir, task, frame_num, size, shift, sample_solver, sampling_steps, guide_scale
    del seed, device_id, offload_model, parallel_cfg, prompt

    if dist.is_initialized() and dist.get_rank() != 0:
        dist.barrier()
        return None

    if bool(head_trajectory_dynamics_plot_only_from_csv):
        _ensure_dir(metrics_output_dir)
        consensus_csv_path = os.path.join(metrics_output_dir, "head_trajectory_dynamics_consensus.csv")
        attractor_csv_path = os.path.join(metrics_output_dir, "head_trajectory_dynamics_attractor.csv")
        reference_distance_csv_path = os.path.join(metrics_output_dir, "head_trajectory_dynamics_reference_distance.csv")
        convergence_csv_path = os.path.join(metrics_output_dir, "head_trajectory_dynamics_convergence.csv")
        summary_path = os.path.join(metrics_output_dir, "head_trajectory_dynamics_summary.json")

        consensus_rows = _load_wan21_t2v_csv_rows(consensus_csv_path)
        attractor_rows = _load_wan21_t2v_csv_rows(attractor_csv_path)
        reference_distance_rows = _load_wan21_t2v_csv_rows(reference_distance_csv_path)
        convergence_rows = _load_wan21_t2v_csv_rows(convergence_csv_path)

        requested_distance_metrics = [str(x).strip().lower() for x in head_trajectory_dynamics_distance_metrics if str(x).strip()]
        if not requested_distance_metrics:
            requested_distance_metrics = _infer_wan21_t2v_head_trajectory_distance_metrics(
                consensus_rows=consensus_rows,
                reference_distance_rows=reference_distance_rows,
                convergence_rows=convergence_rows,
            )
        requested_attractor_distance_metrics = [
            str(x).strip().lower()
            for x in head_trajectory_dynamics_attractor_distance_metrics
            if str(x).strip()
        ]
        available_attractor_distance_metrics = _infer_wan21_t2v_attractor_distance_metrics(attractor_rows)
        if requested_attractor_distance_metrics:
            missing_attractor_metrics = [
                metric_name
                for metric_name in requested_attractor_distance_metrics
                if metric_name not in available_attractor_distance_metrics
            ]
            if missing_attractor_metrics:
                raise ValueError(
                    "head_trajectory_dynamics_plot_only_from_csv=True cannot synthesize attractor metrics "
                    "that are absent from the existing attractor CSV. "
                    f"requested={requested_attractor_distance_metrics} "
                    f"available_in_csv={available_attractor_distance_metrics} "
                    "Please rerun with head_trajectory_dynamics_plot_only_from_csv=False to recompute them."
                )
        plot_paths = _render_wan21_t2v_head_trajectory_metric_plots(
            consensus_rows=consensus_rows,
            attractor_rows=attractor_rows,
            reference_distance_rows=reference_distance_rows,
            convergence_rows=convergence_rows,
            output_dir=metrics_output_dir,
            requested_distance_metrics=requested_distance_metrics,
            skip_existing_plots=bool(head_trajectory_dynamics_skip_existing_plots),
        )

        summary = _load_wan21_t2v_json_if_exists(summary_path)
        summary.update(
            {
                "experiment": "wan21_t2v_head_trajectory_dynamics",
                "head_trajectory_dynamics_plot_only_from_csv": True,
                "head_trajectory_dynamics_skip_existing_plots": bool(head_trajectory_dynamics_skip_existing_plots),
                "head_trajectory_dynamics_hypothesis": str(head_trajectory_dynamics_hypothesis),
                "head_trajectory_dynamics_traj_type": str(head_trajectory_dynamics_traj_type),
                "head_trajectory_dynamics_use_motion_planning_region_before_metrics": bool(
                    head_trajectory_dynamics_use_motion_planning_region_before_metrics
                ),
                "metrics_output_dir": metrics_output_dir,
                "head_trajectory_dynamics_distance_metrics": list(requested_distance_metrics),
                "available_attractor_distance_metrics_in_csv": list(available_attractor_distance_metrics),
                "plot_paths": plot_paths,
            }
        )
        _save_json(summary_path, summary)
        if dist.is_initialized():
            dist.barrier()
        return summary

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
    cross_attention_summary_path = os.path.join(cross_attention_dir, "cross_attention_token_viz_summary.json")
    cross_attention_summary = _load_wan21_t2v_json_if_exists(cross_attention_summary_path)
    reuse_video_frame_count_raw = cross_attention_summary.get("frame_num", default_video_frame_count)
    try:
        reuse_video_frame_count = max(1, int(reuse_video_frame_count_raw))
    except Exception:
        reuse_video_frame_count = int(default_video_frame_count)
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
    requested_attractor_distance_metrics = [
        str(x).strip().lower()
        for x in head_trajectory_dynamics_attractor_distance_metrics
        if str(x).strip()
    ]
    if not requested_attractor_distance_metrics:
        requested_attractor_distance_metrics = ["js", "hellinger", "wasserstein_map", "support_overlap", "center_l2"]
    requested_attractor_distance_metrics = list(dict.fromkeys(requested_attractor_distance_metrics))
    for metric_name in requested_attractor_distance_metrics:
        if metric_name not in {"js", "hellinger", "wasserstein_map", "support_overlap", "center_l2"}:
            raise ValueError(
                "head_trajectory_dynamics_attractor_distance_metrics must be chosen from "
                "{'js', 'hellinger', 'wasserstein_map', 'support_overlap', 'center_l2'}, "
                f"got `{metric_name}`."
            )

    parsed_heads = _parse_wan21_t2v_layer_head_specs(head_trajectory_dynamics_heads)
    requested_head_set = set(parsed_heads)
    parsed_center_viz_heads = _parse_wan21_t2v_layer_head_specs(head_trajectory_dynamics_center_viz_heads)
    requested_center_viz_head_set = set(parsed_center_viz_heads)
    parsed_support_viz_heads = _parse_wan21_t2v_layer_head_specs(head_trajectory_dynamics_support_viz_heads)
    requested_support_viz_head_set = set(parsed_support_viz_heads)

    ordinary_center_config = _resolve_wan21_t2v_head_trajectory_center_config(
        center_method=str(head_trajectory_dynamics_center_method),
        center_power=float(head_trajectory_dynamics_center_power),
        center_quantile=float(head_trajectory_dynamics_center_quantile),
        preprocessed_center_mode=str(head_trajectory_dynamics_preprocessed_center_mode),
        preprocess_winsorize_quantile=float(head_trajectory_dynamics_preprocess_winsorize_quantile),
        preprocess_despike_quantile=float(head_trajectory_dynamics_preprocess_despike_quantile),
        preprocess_min_component_area=int(head_trajectory_dynamics_preprocess_min_component_area),
    )
    reference_center_config = _resolve_wan21_t2v_head_trajectory_reference_center_config(
        ordinary_center_config=ordinary_center_config,
        reference_center_method=str(head_trajectory_dynamics_reference_center_method),
        reference_center_power=float(head_trajectory_dynamics_reference_center_power),
        reference_center_quantile=float(head_trajectory_dynamics_reference_center_quantile),
        reference_preprocessed_center_mode=str(head_trajectory_dynamics_reference_preprocessed_center_mode),
        reference_preprocess_winsorize_quantile=float(head_trajectory_dynamics_reference_preprocess_winsorize_quantile),
        reference_preprocess_despike_quantile=float(head_trajectory_dynamics_reference_preprocess_despike_quantile),
        reference_preprocess_min_component_area=int(head_trajectory_dynamics_reference_preprocess_min_component_area),
    )
    use_preprocessed_component_center = (
        str(ordinary_center_config["center_method"]).strip().lower() == "preprocessed_component_center"
    )
    metrics_output_dir = _build_wan21_t2v_head_trajectory_metrics_output_dir(
        output_dir=output_dir,
        hypothesis_name=str(head_trajectory_dynamics_hypothesis),
        use_motion_planning_region_before_metrics=bool(
            head_trajectory_dynamics_use_motion_planning_region_before_metrics
        ),
        use_preprocessed_component_center=bool(use_preprocessed_component_center),
        preprocessed_center_mode=str(ordinary_center_config["preprocessed_center_mode"]),
    )
    need_motion_planning_region_masks = (
        bool(head_trajectory_dynamics_support_viz_enable)
        or bool(head_trajectory_dynamics_use_motion_planning_region_before_metrics)
        or ("support_overlap" in requested_distance_metrics)
        or ("support_overlap" in requested_attractor_distance_metrics)
    )
    motion_planning_region_cache_basename = _build_wan21_t2v_motion_planning_region_cache_basename(
        support_quantile=float(head_trajectory_dynamics_support_quantile),
        contour_min_component_area=int(head_trajectory_dynamics_support_viz_contour_min_component_area),
    )
    motion_planning_region_cache_path = os.path.join(output_dir, motion_planning_region_cache_basename)
    motion_planning_region_cache_payload = _load_wan21_t2v_motion_planning_region_cache(
        motion_planning_region_cache_path
    )
    motion_planning_region_cache_payload["support_quantile"] = float(head_trajectory_dynamics_support_quantile)
    motion_planning_region_cache_payload["min_component_area"] = int(
        head_trajectory_dynamics_support_viz_contour_min_component_area
    )
    motion_planning_region_cache_hits = 0
    motion_planning_region_cache_misses = 0
    motion_planning_region_pending_cache_writes = 0
    motion_planning_region_cache_save_interval = max(1, int(head_trajectory_dynamics_cache_save_interval))
    center_method_name = str(ordinary_center_config["center_method"])
    cache_basename = _build_wan21_t2v_head_trajectory_cache_basename(
        center_method=center_method_name,
        center_power=float(ordinary_center_config["center_power"]),
        center_quantile=float(ordinary_center_config["center_quantile"]),
        preprocessed_center_mode=str(ordinary_center_config["preprocessed_center_mode"]),
        preprocess_winsorize_quantile=float(ordinary_center_config["preprocess_winsorize_quantile"]),
        preprocess_despike_quantile=float(ordinary_center_config["preprocess_despike_quantile"]),
        preprocess_min_component_area=int(ordinary_center_config["preprocess_min_component_area"]),
    )
    center_cache_path = os.path.join(output_dir, cache_basename)
    center_cache_payload = _load_wan21_t2v_head_trajectory_cache(center_cache_path)
    center_cache_payload["center_method"] = center_method_name
    center_cache_payload["algorithm_params"] = dict(ordinary_center_config)
    cache_hits = 0
    cache_misses = 0
    cache_save_interval = max(1, int(head_trajectory_dynamics_cache_save_interval))
    pending_cache_writes = 0

    filtered_center_cache_path = ""
    filtered_center_cache_payload: Dict[str, object] = {"trajectories": {}}
    filtered_center_cache_hits = 0
    filtered_center_cache_misses = 0
    filtered_center_cache_enabled = (
        bool(head_trajectory_dynamics_use_motion_planning_region_before_metrics)
        and (not bool(use_preprocessed_component_center))
    )
    if filtered_center_cache_enabled:
        filtered_center_cache_basename = _build_wan21_t2v_filtered_center_cache_basename(
            ordinary_center_config=ordinary_center_config,
            reference_center_config=reference_center_config,
            support_quantile=float(head_trajectory_dynamics_support_quantile),
            support_viz_contour_min_component_area=int(head_trajectory_dynamics_support_viz_contour_min_component_area),
            reference_step=int(head_trajectory_dynamics_reference_step),
            reference_layer=int(head_trajectory_dynamics_reference_layer),
        )
        filtered_center_cache_path = os.path.join(output_dir, filtered_center_cache_basename)
        filtered_center_cache_payload = _load_wan21_t2v_head_trajectory_cache(filtered_center_cache_path)
        filtered_center_cache_payload["center_method"] = center_method_name
        filtered_center_cache_payload["algorithm_params"] = dict(ordinary_center_config)
        filtered_center_cache_payload["reference_algorithm_params"] = dict(reference_center_config)
        filtered_center_cache_payload["support_quantile"] = float(head_trajectory_dynamics_support_quantile)
        filtered_center_cache_payload["support_viz_contour_min_component_area"] = int(
            head_trajectory_dynamics_support_viz_contour_min_component_area
        )
        filtered_center_cache_payload["reference_step"] = int(head_trajectory_dynamics_reference_step)
        filtered_center_cache_payload["reference_layer"] = int(head_trajectory_dynamics_reference_layer)
    else:
        filtered_center_cache_path = ""

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
        center_method=str(reference_center_config["center_method"]),
        center_power=float(reference_center_config["center_power"]),
        center_quantile=float(reference_center_config["center_quantile"]),
        preprocessed_center_mode=str(reference_center_config["preprocessed_center_mode"]),
        preprocess_winsorize_quantile=float(reference_center_config["preprocess_winsorize_quantile"]),
        preprocess_despike_quantile=float(reference_center_config["preprocess_despike_quantile"]),
        preprocess_min_component_area=int(reference_center_config["preprocess_min_component_area"]),
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
    extraction_task_count = 0
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
            for head_index in range(int(object_head_maps.size(0))):
                if requested_head_set and (int(layer_index), int(head_index)) not in requested_head_set:
                    continue
                extraction_task_count += 1

    extraction_progress_bar = None
    if extraction_task_count > 0:
        try:
            from tqdm import tqdm
            extraction_progress_bar = tqdm(
                total=int(extraction_task_count),
                desc="head_trajectory centers",
                unit="head",
                leave=True,
            )
        except Exception:
            extraction_progress_bar = None

    center_missing_tasks: List[Tuple[int, int, int, torch.Tensor, Dict[str, object]]] = []
    try:
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
                        center_missing_tasks.append(
                            (
                                int(step_index),
                                int(layer_index),
                                int(head_index),
                                map_fhw,
                                dict(ordinary_center_config),
                            )
                        )
                    else:
                        cache_hits += 1
                    key = (int(step_index), int(layer_index), int(head_index))
                    probability_maps_by_step_layer_head[key] = probability_map
                    if cached_trajectory is not None:
                        center_trajectories_by_step_layer_head[key] = _center_trajectory_wan21_t2v_to_tensor(
                            cached_trajectory
                        )
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

        for key, extracted_trajectory in _iter_wan21_t2v_parallel_results(
            tasks=center_missing_tasks,
            worker_fn=_build_wan21_t2v_head_center_extraction_task,
            num_workers=int(head_trajectory_dynamics_center_cache_num_workers),
        ):
            _set_wan21_t2v_cached_center_trajectory(
                cache_payload=center_cache_payload,
                step=int(key[0]),
                layer=int(key[1]),
                head=int(key[2]),
                trajectory=extracted_trajectory,
            )
            center_trajectories_by_step_layer_head[key] = _center_trajectory_wan21_t2v_to_tensor(extracted_trajectory)
            cache_misses += 1
            pending_cache_writes += 1
            if pending_cache_writes >= int(cache_save_interval):
                _save_wan21_t2v_head_trajectory_cache(center_cache_path, center_cache_payload)
                pending_cache_writes = 0
            if extraction_progress_bar is not None:
                extraction_progress_bar.update(1)

        if extraction_progress_bar is not None:
            extraction_progress_bar.update(int(cache_hits))
    finally:
        if pending_cache_writes > 0:
            _save_wan21_t2v_head_trajectory_cache(center_cache_path, center_cache_payload)
        if extraction_progress_bar is not None:
            extraction_progress_bar.close()

    if not head_map_records:
        raise ValueError(
            "No head maps remain after applying head filters. "
            f"requested_heads={list(head_trajectory_dynamics_heads)}"
        )

    support_masks_by_key: Dict[Tuple[int, int, int], torch.Tensor] = {}
    if bool(need_motion_planning_region_masks):
        support_masks_by_key, cache_hits, cache_misses = _build_wan21_t2v_motion_planning_region_masks_with_progress(
            probability_maps_by_step_layer_head=probability_maps_by_step_layer_head,
            cache_payload=motion_planning_region_cache_payload,
            cache_path=motion_planning_region_cache_path,
            support_quantile=float(head_trajectory_dynamics_support_quantile),
            min_component_area=int(head_trajectory_dynamics_support_viz_contour_min_component_area),
            num_workers=int(head_trajectory_dynamics_support_cache_num_workers),
            cache_save_interval=int(motion_planning_region_cache_save_interval),
        )
        motion_planning_region_cache_hits += int(cache_hits)
        motion_planning_region_cache_misses += int(cache_misses)

    (
        overlay_plot_paths,
        center_overlay_dir,
        support_overlap_mask_dir,
        num_center_overlay_pdfs,
        num_support_overlap_mask_pdfs,
    ) = _render_wan21_t2v_head_trajectory_overlays(
        probability_maps_by_step_layer_head=probability_maps_by_step_layer_head,
        center_trajectories_by_step_layer_head=center_trajectories_by_step_layer_head,
        output_dir=output_dir,
        requested_head_set=parsed_heads,
        requested_center_viz_head_set=parsed_center_viz_heads,
        requested_support_viz_head_set=parsed_support_viz_heads,
        head_trajectory_dynamics_center_viz_enable=bool(head_trajectory_dynamics_center_viz_enable),
        head_trajectory_dynamics_center_viz_step=int(head_trajectory_dynamics_center_viz_step),
        head_trajectory_dynamics_center_viz_layer=int(head_trajectory_dynamics_center_viz_layer),
        head_trajectory_dynamics_center_viz_num_frames=int(head_trajectory_dynamics_center_viz_num_frames),
        head_trajectory_dynamics_support_viz_enable=bool(head_trajectory_dynamics_support_viz_enable),
        head_trajectory_dynamics_support_viz_step=int(head_trajectory_dynamics_support_viz_step),
        head_trajectory_dynamics_support_viz_layer=int(head_trajectory_dynamics_support_viz_layer),
        head_trajectory_dynamics_support_viz_num_frames=int(head_trajectory_dynamics_support_viz_num_frames),
        head_trajectory_dynamics_support_viz_contour_min_component_area=int(
            head_trajectory_dynamics_support_viz_contour_min_component_area
        ),
        head_trajectory_dynamics_support_quantile=float(head_trajectory_dynamics_support_quantile),
        head_trajectory_dynamics_skip_existing_plots=bool(head_trajectory_dynamics_skip_existing_plots),
        reuse_video_frame_count=int(reuse_video_frame_count),
        overlay_num_workers=int(head_trajectory_dynamics_overlay_num_workers),
        use_motion_planning_region_before_metrics=bool(
            head_trajectory_dynamics_use_motion_planning_region_before_metrics
        ),
        use_preprocessed_component_center=bool(use_preprocessed_component_center),
        preprocessed_center_mode=str(ordinary_center_config["preprocessed_center_mode"]),
    )

    if bool(head_trajectory_dynamics_overlay_only):
        summary_path = os.path.join(output_dir, "head_trajectory_dynamics_summary.json")
        summary = _load_wan21_t2v_json_if_exists(summary_path)
        summary.update(
            {
                "experiment": "wan21_t2v_head_trajectory_dynamics",
                "head_trajectory_dynamics_overlay_only": True,
                "head_trajectory_dynamics_skip_existing_plots": bool(head_trajectory_dynamics_skip_existing_plots),
                "head_trajectory_dynamics_center_viz_enable": bool(head_trajectory_dynamics_center_viz_enable),
                "head_trajectory_dynamics_support_viz_enable": bool(head_trajectory_dynamics_support_viz_enable),
                "head_trajectory_dynamics_traj_type": str(head_trajectory_dynamics_traj_type),
                "center_overlay_dir": center_overlay_dir,
                "support_overlap_mask_dir": support_overlap_mask_dir,
                "num_center_overlay_pdfs": int(num_center_overlay_pdfs),
                "num_support_overlap_mask_pdfs": int(num_support_overlap_mask_pdfs),
                "center_cache_json": center_cache_path,
                "motion_planning_region_cache_json": motion_planning_region_cache_path,
                "head_trajectory_dynamics_center_cache_num_workers": int(head_trajectory_dynamics_center_cache_num_workers),
                "head_trajectory_dynamics_overlay_num_workers": int(head_trajectory_dynamics_overlay_num_workers),
                "plot_paths": overlay_plot_paths,
                "reuse_cross_attention_dir": cross_attention_dir,
                "reuse_cross_attention_summary_path": cross_attention_summary_path,
                "reuse_video_frame_count": int(reuse_video_frame_count),
                "loaded_maps_source": loaded_maps_source,
            }
        )
        _save_json(summary_path, summary)
        if dist.is_initialized():
            dist.barrier()
        return summary

    reference_motion_planning_region_mask = _build_wan21_t2v_motion_planning_region_mask_fhw(
        probability_map_fhw=reference_probability_map,
        quantile=float(head_trajectory_dynamics_support_quantile),
        min_component_area=int(head_trajectory_dynamics_support_viz_contour_min_component_area),
    )

    if bool(head_trajectory_dynamics_use_motion_planning_region_before_metrics):
        (
            filtered_probability_maps_by_step_layer_head,
        ) = (
            _materialize_wan21_t2v_motion_planning_filtered_maps(
                probability_maps_by_step_layer_head=probability_maps_by_step_layer_head,
                motion_planning_region_masks_by_step_layer_head=support_masks_by_key,
                progress_desc="head_trajectory apply mask",
            ),
        )
        probability_maps_by_step_layer_head = filtered_probability_maps_by_step_layer_head
        if bool(use_preprocessed_component_center):
            filtered_center_trajectories_by_step_layer_head = center_trajectories_by_step_layer_head
        else:
            (
                filtered_center_trajectories_by_step_layer_head,
                filter_cache_hits,
                filter_cache_misses,
            ) = _materialize_wan21_t2v_motion_planning_filtered_centers(
                filtered_probability_maps_by_step_layer_head=filtered_probability_maps_by_step_layer_head,
                filtered_center_cache_payload=filtered_center_cache_payload,
                filtered_center_cache_path=filtered_center_cache_path,
                center_config=ordinary_center_config,
                num_workers=int(head_trajectory_dynamics_center_cache_num_workers),
                progress_desc="head_trajectory filtered centers",
                cache_save_interval=int(cache_save_interval),
            )
            filtered_center_cache_hits += int(filter_cache_hits)
            filtered_center_cache_misses += int(filter_cache_misses)
            center_trajectories_by_step_layer_head = filtered_center_trajectories_by_step_layer_head
        reference_probability_map = _apply_wan21_t2v_motion_planning_region_to_probability_map(
            probability_map_fhw=reference_probability_map,
            motion_planning_region_mask_fhw=reference_motion_planning_region_mask,
        )
        if bool(use_preprocessed_component_center):
            filtered_reference_center_trajectory = reference_center_trajectory
            final_reference_center = _center_trajectory_wan21_t2v_to_tensor(filtered_reference_center_trajectory)
        else:
            filtered_reference_center_trajectory = _get_wan21_t2v_cached_reference_center_trajectory(
                cache_payload=filtered_center_cache_payload
            )
            if filtered_reference_center_trajectory is None:
                reference_center_trajectory, reference_center_stats = _extract_wan21_t2v_head_trajectory_centers(
                    map_fhw=reference_probability_map,
                    center_method=str(reference_center_config["center_method"]),
                    center_power=float(reference_center_config["center_power"]),
                    center_quantile=float(reference_center_config["center_quantile"]),
                    preprocessed_center_mode=str(reference_center_config["preprocessed_center_mode"]),
                    preprocess_winsorize_quantile=float(reference_center_config["preprocess_winsorize_quantile"]),
                    preprocess_despike_quantile=float(reference_center_config["preprocess_despike_quantile"]),
                    preprocess_min_component_area=int(reference_center_config["preprocess_min_component_area"]),
                )
                filtered_reference_center_trajectory = reference_center_trajectory
                _set_wan21_t2v_cached_reference_center_trajectory(
                    cache_payload=filtered_center_cache_payload,
                    trajectory=filtered_reference_center_trajectory,
                )
                filtered_center_cache_payload["reference_center_stats"] = dict(reference_center_stats)
                filtered_center_cache_misses += 1
                _save_wan21_t2v_head_trajectory_cache(filtered_center_cache_path, filtered_center_cache_payload)
            else:
                filtered_center_cache_hits += 1
                reference_center_stats = filtered_center_cache_payload.get("reference_center_stats", {})
                if not isinstance(reference_center_stats, dict):
                    reference_center_stats = {
                        "center_method": str(reference_center_config["center_method"]),
                        "center_power": float(reference_center_config["center_power"]),
                        "center_quantile": float(reference_center_config["center_quantile"]),
                        "preprocess_enabled": 1
                        if str(reference_center_config["center_method"]).strip().lower() == "preprocessed_component_center"
                        else 0,
                        "preprocessed_center_mode": str(reference_center_config["preprocessed_center_mode"]).strip().lower(),
                    }
            final_reference_center = _center_trajectory_wan21_t2v_to_tensor(filtered_reference_center_trajectory)

    reference_distance_rows = []
    center_rows = []
    for step_index, layer_index, head_index in sorted(probability_maps_by_step_layer_head.keys()):
        probability_map = probability_maps_by_step_layer_head[(int(step_index), int(layer_index), int(head_index))]
        center_trajectory = center_trajectories_by_step_layer_head[(int(step_index), int(layer_index), int(head_index))]
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
            if bool(head_trajectory_dynamics_use_motion_planning_region_before_metrics):
                reference_support_iou = _support_overlap_mask_iou_wan21_t2v_per_frame(
                    support_masks_by_key[(int(step_index), int(layer_index), int(head_index))],
                    reference_motion_planning_region_mask,
                )
            else:
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

    per_step_layer_heads: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for step_index, layer_index, head_index in probability_maps_by_step_layer_head.keys():
        per_step_layer_heads[(int(step_index), int(layer_index))].append(int(head_index))

    for key in per_step_layer_heads:
        per_step_layer_heads[key] = sorted(set(per_step_layer_heads[key]))

    pairwise_task_count = 0
    for head_indices in per_step_layer_heads.values():
        pairwise_task_count += int(len(head_indices) * max(0, len(head_indices) - 1) / 2)
    attractor_task_count = 0
    sorted_resolved_steps = sorted(int(step) for step in resolved_steps)
    for step_index in sorted_resolved_steps[:-1]:
        for layer_index in available_layers:
            head_indices = per_step_layer_heads.get((int(step_index), int(layer_index)), [])
            if not head_indices:
                continue
            attractor_task_count += int(len(head_indices) * len(requested_attractor_distance_metrics))
    convergence_task_count = 0
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
        convergence_task_count += int(len(layer_heads) * len(requested_distance_metrics))

    metric_progress_bar = None
    total_metric_tasks = int(pairwise_task_count + attractor_task_count + convergence_task_count)
    if total_metric_tasks > 0:
        try:
            from tqdm import tqdm
            metric_progress_bar = tqdm(
                total=int(total_metric_tasks),
                desc="head_trajectory_dynamics metrics",
                unit="task",
                leave=True,
            )
        except Exception:
            metric_progress_bar = None

    try:
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
                        if bool(head_trajectory_dynamics_use_motion_planning_region_before_metrics):
                            support_iou = _support_overlap_mask_iou_wan21_t2v_per_frame(
                                support_masks_by_key[(int(step_index), int(layer_index), int(head_i))],
                                support_masks_by_key[(int(step_index), int(layer_index), int(head_j))],
                            )
                        else:
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
                    if metric_progress_bar is not None:
                        metric_progress_bar.update(1)

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
                    leader_prob = probability_maps_by_step_layer_head[leader_key]
                    for attractor_distance_metric in requested_attractor_distance_metrics:
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
                            follower_current_prob = probability_maps_by_step_layer_head[follower_current_key]
                            current_distance = _compute_wan21_t2v_head_trajectory_distance(
                                metric_name=attractor_distance_metric,
                                probability_map_a_fhw=follower_current_prob,
                                probability_map_b_fhw=leader_prob,
                                center_traj_a=follower_current,
                                center_traj_b=leader_traj,
                                support_quantile=float(head_trajectory_dynamics_support_quantile),
                                use_motion_planning_region_for_support_overlap=bool(
                                    head_trajectory_dynamics_use_motion_planning_region_before_metrics
                                ),
                                support_mask_a_fhw=(
                                    support_masks_by_key[follower_current_key]
                                    if follower_current_key in support_masks_by_key else None
                                ),
                                support_mask_b_fhw=(
                                    support_masks_by_key[leader_key]
                                    if leader_key in support_masks_by_key else None
                                ),
                            )
                            future_distances = []
                            for future_step in future_steps:
                                future_head_set = future_head_sets[int(future_step)]
                                if (
                                    int(follower_head) not in future_head_set
                                    or int(leader_head) not in future_head_set
                                ):
                                    continue
                                follower_future_key = (int(future_step), int(layer_index), int(follower_head))
                                leader_future_key = (int(future_step), int(layer_index), int(leader_head))
                                if (
                                    follower_future_key not in center_trajectories_by_step_layer_head
                                    or leader_future_key not in center_trajectories_by_step_layer_head
                                    or follower_future_key not in probability_maps_by_step_layer_head
                                    or leader_future_key not in probability_maps_by_step_layer_head
                                ):
                                    continue
                                follower_future = center_trajectories_by_step_layer_head[follower_future_key]
                                follower_future_prob = probability_maps_by_step_layer_head[follower_future_key]
                                leader_future = center_trajectories_by_step_layer_head[leader_future_key]
                                leader_future_prob = probability_maps_by_step_layer_head[leader_future_key]
                                future_distances.append(
                                    (
                                        int(future_step),
                                        _compute_wan21_t2v_head_trajectory_distance(
                                            metric_name=attractor_distance_metric,
                                            probability_map_a_fhw=follower_future_prob,
                                            probability_map_b_fhw=leader_future_prob,
                                            center_traj_a=follower_future,
                                            center_traj_b=leader_future,
                                            support_quantile=float(head_trajectory_dynamics_support_quantile),
                                            use_motion_planning_region_for_support_overlap=bool(
                                                head_trajectory_dynamics_use_motion_planning_region_before_metrics
                                            ),
                                            support_mask_a_fhw=(
                                                support_masks_by_key[follower_future_key]
                                                if follower_future_key in support_masks_by_key else None
                                            ),
                                            support_mask_b_fhw=(
                                                support_masks_by_key[leader_future_key]
                                                if leader_future_key in support_masks_by_key else None
                                            ),
                                        ),
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
                                    "attractor_distance_metric": str(attractor_distance_metric),
                                    "attractor_score_mean": float(sum(deltas) / len(deltas)),
                                    "attractor_score_max": float(max(deltas)),
                                    "attractor_score_min": float(min(deltas)),
                                    "num_followers": int(len(deltas)),
                                }
                            )
                        if metric_progress_bar is not None:
                            metric_progress_bar.update(1)

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
                    if metric_progress_bar is not None:
                        metric_progress_bar.update(1)
    finally:
        if metric_progress_bar is not None:
            metric_progress_bar.close()

    _ensure_dir(metrics_output_dir)
    _save_csv(os.path.join(metrics_output_dir, "head_trajectory_dynamics_head_maps.csv"), head_map_records)
    _save_csv(os.path.join(metrics_output_dir, "head_trajectory_dynamics_pairwise.csv"), pairwise_rows)
    _save_csv(os.path.join(metrics_output_dir, "head_trajectory_dynamics_consensus.csv"), consensus_rows)
    _save_csv(os.path.join(metrics_output_dir, "head_trajectory_dynamics_attractor.csv"), attractor_rows)
    _save_csv(os.path.join(metrics_output_dir, "head_trajectory_dynamics_reference_distance.csv"), reference_distance_rows)
    _save_csv(os.path.join(metrics_output_dir, "head_trajectory_dynamics_convergence.csv"), convergence_rows)
    trajectory_centers_csv_path = os.path.join(metrics_output_dir, "head_trajectory_dynamics_trajectory_centers.csv")
    legacy_soft_centers_csv_path = os.path.join(metrics_output_dir, "head_trajectory_dynamics_soft_centers.csv")
    _save_csv(trajectory_centers_csv_path, center_rows)
    _save_csv(legacy_soft_centers_csv_path, center_rows)

    metric_plot_paths = _render_wan21_t2v_head_trajectory_metric_plots(
        consensus_rows=consensus_rows,
        attractor_rows=attractor_rows,
        reference_distance_rows=reference_distance_rows,
        convergence_rows=convergence_rows,
        output_dir=metrics_output_dir,
        requested_distance_metrics=requested_distance_metrics,
        skip_existing_plots=bool(head_trajectory_dynamics_skip_existing_plots),
    )
    plot_paths = list(overlay_plot_paths) + list(metric_plot_paths)

    summary = {
        "experiment": "wan21_t2v_head_trajectory_dynamics",
        "target_object_words": list(object_words),
        "target_verb_words": list(verb_words),
        "prompt_tokens": prompt_tokens,
        "token_types": word_to_type,
        "object_words_in_maps": object_words_in_maps,
        "reuse_cross_attention_dir": cross_attention_dir,
        "reuse_cross_attention_summary_path": cross_attention_summary_path,
        "reuse_video_frame_count": int(reuse_video_frame_count),
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
        "head_trajectory_dynamics_attractor_distance_metrics": list(requested_attractor_distance_metrics),
        "head_trajectory_dynamics_hypothesis": str(head_trajectory_dynamics_hypothesis),
        "head_trajectory_dynamics_traj_type": str(head_trajectory_dynamics_traj_type),
        "head_trajectory_dynamics_use_motion_planning_region_before_metrics": bool(
            head_trajectory_dynamics_use_motion_planning_region_before_metrics
        ),
        "head_trajectory_dynamics_plot_only_from_csv": False,
        "head_trajectory_dynamics_skip_existing_plots": bool(head_trajectory_dynamics_skip_existing_plots),
        "metrics_output_dir": metrics_output_dir,
        "reference_step": int(head_trajectory_dynamics_reference_step),
        "reference_layer": int(head_trajectory_dynamics_reference_layer),
        "head_trajectory_dynamics_center_viz_enable": bool(head_trajectory_dynamics_center_viz_enable),
        "center_viz_step": int(head_trajectory_dynamics_center_viz_step),
        "center_viz_layer": int(head_trajectory_dynamics_center_viz_layer),
        "center_viz_heads_input": [str(x) for x in head_trajectory_dynamics_center_viz_heads],
        "center_viz_heads_parsed": [
            {"layer": int(layer_index), "head": int(head_index), "head_tag": f"L{int(layer_index)}H{int(head_index)}"}
            for layer_index, head_index in parsed_center_viz_heads
        ],
        "center_viz_num_frames": int(head_trajectory_dynamics_center_viz_num_frames),
        "center_overlay_dir": center_overlay_dir,
        "center_overlay_path_policy": "motion_planning_region_on_or_off_preprocessed_on_or_off_center_mode",
        "num_center_overlay_pdfs": int(num_center_overlay_pdfs),
        "head_trajectory_dynamics_support_viz_enable": bool(head_trajectory_dynamics_support_viz_enable),
        "support_viz_step": int(head_trajectory_dynamics_support_viz_step),
        "support_viz_layer": int(head_trajectory_dynamics_support_viz_layer),
        "support_viz_heads_input": [str(x) for x in head_trajectory_dynamics_support_viz_heads],
        "support_viz_heads_parsed": [
            {"layer": int(layer_index), "head": int(head_index), "head_tag": f"L{int(layer_index)}H{int(head_index)}"}
            for layer_index, head_index in parsed_support_viz_heads
        ],
        "support_viz_num_frames": int(head_trajectory_dynamics_support_viz_num_frames),
        "support_viz_contour_min_component_area": int(head_trajectory_dynamics_support_viz_contour_min_component_area),
        "support_overlap_mask_dir": support_overlap_mask_dir,
        "num_support_overlap_mask_pdfs": int(num_support_overlap_mask_pdfs),
        "motion_planning_region_cache_json": motion_planning_region_cache_path,
        "motion_planning_region_cache_hits": int(motion_planning_region_cache_hits),
        "motion_planning_region_cache_misses": int(motion_planning_region_cache_misses),
        "filtered_center_cache_json": filtered_center_cache_path,
        "filtered_center_cache_enabled": bool(filtered_center_cache_enabled),
        "head_trajectory_dynamics_support_cache_num_workers": int(head_trajectory_dynamics_support_cache_num_workers),
        "head_trajectory_dynamics_center_cache_num_workers": int(head_trajectory_dynamics_center_cache_num_workers),
        "head_trajectory_dynamics_overlay_num_workers": int(head_trajectory_dynamics_overlay_num_workers),
        "head_trajectory_dynamics_cache_save_interval": int(cache_save_interval),
        "num_head_records": int(len(head_map_records)),
        "num_pairwise_rows": int(len(pairwise_rows)),
        "num_consensus_rows": int(len(consensus_rows)),
        "num_attractor_rows": int(len(attractor_rows)),
        "num_reference_distance_rows": int(len(reference_distance_rows)),
        "num_convergence_rows": int(len(convergence_rows)),
        "num_center_rows": int(len(center_rows)),
        "plot_paths": plot_paths,
        "head_maps_csv": os.path.join(metrics_output_dir, "head_trajectory_dynamics_head_maps.csv"),
        "pairwise_csv": os.path.join(metrics_output_dir, "head_trajectory_dynamics_pairwise.csv"),
        "consensus_csv": os.path.join(metrics_output_dir, "head_trajectory_dynamics_consensus.csv"),
        "attractor_csv": os.path.join(metrics_output_dir, "head_trajectory_dynamics_attractor.csv"),
        "reference_distance_csv": os.path.join(metrics_output_dir, "head_trajectory_dynamics_reference_distance.csv"),
        "convergence_csv": os.path.join(metrics_output_dir, "head_trajectory_dynamics_convergence.csv"),
        "trajectory_centers_csv": trajectory_centers_csv_path,
        "legacy_soft_centers_csv": legacy_soft_centers_csv_path,
        "center_cache_json": center_cache_path,
        "filtered_center_cache_json": filtered_center_cache_path,
        "center_method": center_method_name,
        "support_quantile": float(head_trajectory_dynamics_support_quantile),
        "attractor_window": int(head_trajectory_dynamics_attractor_window),
        "ordinary_head_center_config": dict(ordinary_center_config),
        "reference_center_config": dict(reference_center_config),
        "center_cache_hits": int(cache_hits),
        "center_cache_misses": int(cache_misses),
        "filtered_center_cache_hits": int(filtered_center_cache_hits),
        "filtered_center_cache_misses": int(filtered_center_cache_misses),
        "reference_center_stats": reference_center_stats,
    }
    _save_json(os.path.join(metrics_output_dir, "head_trajectory_dynamics_summary.json"), summary)
    if dist.is_initialized():
        dist.barrier()
    return summary
