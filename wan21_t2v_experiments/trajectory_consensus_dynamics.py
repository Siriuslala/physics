"""Wan2.1-T2V experiment: trajectory_consensus_dynamics.

Main entry:
- run_wan21_t2v_trajectory_consensus_dynamics

This experiment is a stage-based 2.0 framework for motion-planning analysis.
The current implementation focuses on three stages:
1) candidate_consensus: offline candidate-region extraction and winner-gap analysis
2) head_contribution: runtime head-wise contribution analysis via exact zero-ablation,
   first-order Taylor approximation, and direct-proxy readout
3) self_attention_coupling: runtime self-attention candidate-coupling analysis,
   winner-versus-loser feature extraction, and temporal precedence summaries

Later stages documented in the technical note can be added on the same
engineering scaffold without modifying the official Wan2.1 source tree.
"""

import json
import math
import os
import pickle
import random
import re
import sys
import gc
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MethodType
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist

from .head_evolution import (
    _build_wan21_t2v_trajectory_support_mask_from_centers,
    _extract_wan21_t2v_connected_components,
    _extract_wan21_t2v_reference_peak_and_centroid_trajectory,
    _preprocess_wan21_t2v_attention_map_fhw,
)
from .self_attention_distribution import (
    _build_wan21_t2v_self_attention_distribution_reference_support,
)
from .utils import (
    Wan21T2VParallelConfig,
    _map_wan21_t2v_token_frame_to_video_frame_label,
    _broadcast_seed_if_needed,
    _build_wan21_t2v_pipeline,
    _dedup_wan21_t2v_int_list,
    _encode_wan21_t2v_text_context_once,
    _ensure_dir,
    _generate_wan21_t2v_video,
    _generate_wan21_t2v_video_with_initial_noise,
    _init_wan21_t2v_runtime,
    _iter_wan21_t2v_parallel_results,
    _load_wan21_t2v_cross_attention_mean_maps_from_disk,
    _load_wan21_t2v_cross_attention_token_meta,
    _load_wan21_t2v_csv_rows,
    _mean_wan21_t2v_head_maps_for_words,
    _mean_wan21_t2v_headmean_map_for_words,
    _normalize_wan21_t2v_attention_map_per_frame,
    _parse_wan21_t2v_layer_head_specs,
    _resolve_wan21_t2v_branch_from_forward_call_index,
    _resolve_wan21_t2v_num_workers,
    _resolve_wan21_t2v_offload_model,
    _resolve_wan21_t2v_viz_frame_indices,
    _run_wan21_t2v_once_with_patch,
    _save_csv,
    _save_json,
    _unwrap_wan21_t2v_dit_model_for_runtime_patch,
    _wan21_t2v_branch_matches,
)
from .wan21_t2v_experiment_patch import (
    Wan21T2VAttentionProbeConfig,
    Wan21T2VCausalAttentionConfig,
    Wan21T2VEarlyStopRequested,
    Wan21T2VPatchBundleConfig,
    Wan21T2VRopePatchConfig,
)
from projects.Wan2_1.wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from projects.Wan2_1.wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

def _maybe_skip_wan21_t2v_existing_plot(save_file: str, skip_existing_plots: bool) -> bool:
    """Return True when a plot already exists and should be reused."""
    return bool(skip_existing_plots) and os.path.exists(save_file)


def _load_wan21_t2v_torch_cache(path: str) -> Any:
    """Load one local torch cache with compatibility for PyTorch 2.6+.

    Newer PyTorch versions default `weights_only=True`, which rejects our older
    candidate-cache files because they may contain numpy arrays. These files are
    generated locally by this experiment, so it is safe to fall back to
    `weights_only=False` for trusted local caches.
    """
    try:
        return torch.load(path, map_location="cpu")
    except pickle.UnpicklingError:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")


def _resolve_wan21_t2v_trajectory_consensus_viz_frames(
    attention_frame_count: int,
    video_frame_count: int,
    num_frames: int,
) -> Tuple[List[int], List[int]]:
    """Resolve token-frame indices and display them using token-frame labels.

    The selected frame subset still reuses the same sampling rule as
    `cross_attention_token_viz`, but trajectory-consensus figures now display
    the token-frame indices directly, that is, `0 .. attention_frame_count-1`.
    """
    attention_frame_indices, _ = _resolve_wan21_t2v_viz_frame_indices(
        attention_frame_count=int(attention_frame_count),
        video_frame_count=int(video_frame_count),
        num_frames=int(num_frames),
        explicit_indices=None,
    )
    return (
        [int(frame_index) for frame_index in attention_frame_indices],
        [int(frame_index) for frame_index in attention_frame_indices],
    )


def _resolve_wan21_t2v_selected_head_specs_from_layer_counts(
    explicit_head_specs: Optional[Sequence[str]],
    num_heads_per_layer: Dict[int, int],
) -> List[Tuple[int, int]]:
    """Resolve explicit head specs with `all` / `none` semantics.

    Semantics:
    - `None`: analyze no heads
    - empty sequence: analyze all heads in all provided layers
    - non-empty sequence: analyze the explicitly listed heads
    """
    if explicit_head_specs is None:
        return []
    if explicit_head_specs:
        return _parse_wan21_t2v_layer_head_specs(explicit_head_specs)
    resolved: List[Tuple[int, int]] = []
    for layer in sorted(int(x) for x in num_heads_per_layer.keys()):
        for head in range(int(num_heads_per_layer[int(layer)])):
            resolved.append((int(layer), int(head)))
    return resolved

def _plot_wan21_t2v_trajectory_consensus_heatmap(
    matrix_rows: Sequence[Dict[str, object]],
    save_file: str,
    title: str,
    row_key: str,
    col_key: str,
    value_key: str,
    row_label: str,
    col_label: str,
    cmap: str = "bwr",
):
    """Render a heatmap from flat row dictionaries."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    if not matrix_rows:
        return ""

    row_values = sorted(set(int(row[row_key]) for row in matrix_rows))
    col_values = sorted(set(int(row[col_key]) for row in matrix_rows))
    row_to_index = {value: idx for idx, value in enumerate(row_values)}
    col_to_index = {value: idx for idx, value in enumerate(col_values)}

    heatmap = torch.full((len(row_values), len(col_values)), float("nan"), dtype=torch.float32)
    for row in matrix_rows:
        if value_key not in row or row[value_key] == "":
            continue
        heatmap[row_to_index[int(row[row_key])], col_to_index[int(row[col_key])]] = float(row[value_key])

    fig_width = max(6.4, 0.32 * len(col_values))
    fig_height = max(4.8, 0.25 * len(row_values))
    fig, axis = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    heatmap_np = heatmap.numpy()
    finite_values = heatmap[~torch.isnan(heatmap)]
    cmap_name = str(cmap).strip() or "bwr"
    if finite_values.numel() > 0:
        data_min = float(finite_values.min().item())
        data_max = float(finite_values.max().item())
        if cmap_name.lower() == "blues":
            image = axis.imshow(heatmap_np, cmap=cmap_name, aspect="auto", vmin=0.0, vmax=max(data_max, 1e-12))
        elif data_min < 0.0 < data_max:
            bound = max(abs(data_min), abs(data_max))
            image = axis.imshow(heatmap_np, cmap=cmap_name, aspect="auto", vmin=-bound, vmax=bound)
        else:
            image = axis.imshow(heatmap_np, cmap=cmap_name, aspect="auto", vmin=data_min, vmax=data_max)
    else:
        image = axis.imshow(heatmap_np, cmap=cmap_name, aspect="auto")
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


def _trajectory_consensus_extract_candidate_task(task: Tuple) -> Tuple[int, int, Dict[str, object]]:
    """Worker task for candidate-region extraction on one `(step, layer)` pair."""
    (
        step,
        layer,
        headmean_map,
        base_quantile,
        split_quantiles,
        min_component_area,
        smooth_radius,
        stable_peak_min_levels,
        peak_merge_distance,
        preprocess_winsorize_quantile,
        preprocess_despike_quantile,
        preprocess_min_component_area,
    ) = task
    preprocessed_headmean_map, preprocess_stats = _preprocess_wan21_t2v_attention_map_fhw(
        map_fhw=headmean_map.float(),
        winsorize_quantile=float(preprocess_winsorize_quantile),
        despike_quantile=float(preprocess_despike_quantile),
        min_component_area=int(preprocess_min_component_area),
    )
    candidate_data = _extract_wan21_t2v_candidate_regions_for_map(
        map_fhw=preprocessed_headmean_map,
        base_quantile=float(base_quantile),
        split_quantiles=split_quantiles,
        min_component_area=int(min_component_area),
        smooth_radius=int(smooth_radius),
        stable_peak_min_levels=int(stable_peak_min_levels),
        peak_merge_distance=float(peak_merge_distance),
    )
    return (
        int(step),
        int(layer),
        {
            # Returning torch tensors from forked workers can hang in this
            # environment. Return numpy / Python objects and rebuild torch
            # tensors on the parent process.
            "label_map_fhw_np": candidate_data["label_map_fhw"].detach().cpu().numpy().astype(np.int16, copy=False),
            "frame_metadata": candidate_data["frame_metadata"],
        },
    )


def _trajectory_consensus_extract_head_specific_candidate_task(
    task: Tuple,
) -> Tuple[int, int, int, Dict[str, object]]:
    """Worker task for per-head candidate extraction on one `(step, layer, head)`."""
    (
        step,
        layer,
        head,
        head_map,
        base_quantile,
        split_quantiles,
        min_component_area,
        smooth_radius,
        stable_peak_min_levels,
        peak_merge_distance,
        preprocess_winsorize_quantile,
        preprocess_despike_quantile,
        preprocess_min_component_area,
    ) = task
    preprocessed_head_map, _ = _preprocess_wan21_t2v_attention_map_fhw(
        map_fhw=head_map.float(),
        winsorize_quantile=float(preprocess_winsorize_quantile),
        despike_quantile=float(preprocess_despike_quantile),
        min_component_area=int(preprocess_min_component_area),
    )
    candidate_data = _extract_wan21_t2v_candidate_regions_for_map(
        map_fhw=preprocessed_head_map,
        base_quantile=float(base_quantile),
        split_quantiles=split_quantiles,
        min_component_area=int(min_component_area),
        smooth_radius=int(smooth_radius),
        stable_peak_min_levels=int(stable_peak_min_levels),
        peak_merge_distance=float(peak_merge_distance),
    )
    return (
        int(step),
        int(layer),
        int(head),
        {
            "label_map_fhw_np": candidate_data["label_map_fhw"].detach().cpu().numpy().astype(np.int16, copy=False),
            "frame_metadata": candidate_data["frame_metadata"],
        },
    )


def _trajectory_consensus_extract_layer_head_candidates_task(
    task: Tuple,
) -> Tuple[int, int, List[Tuple[int, Dict[str, object]]]]:
    """Worker task for per-head candidate extraction on one `(step, layer)` group."""
    (
        step,
        layer,
        head_payloads,
        base_quantile,
        split_quantiles,
        min_component_area,
        smooth_radius,
        stable_peak_min_levels,
        peak_merge_distance,
        preprocess_winsorize_quantile,
        preprocess_despike_quantile,
        preprocess_min_component_area,
    ) = task
    results: List[Tuple[int, Dict[str, object]]] = []
    for head, head_map in head_payloads:
        preprocessed_head_map, _ = _preprocess_wan21_t2v_attention_map_fhw(
            map_fhw=head_map.float(),
            winsorize_quantile=float(preprocess_winsorize_quantile),
            despike_quantile=float(preprocess_despike_quantile),
            min_component_area=int(preprocess_min_component_area),
        )
        candidate_data = _extract_wan21_t2v_candidate_regions_for_map(
            map_fhw=preprocessed_head_map,
            base_quantile=float(base_quantile),
            split_quantiles=split_quantiles,
            min_component_area=int(min_component_area),
            smooth_radius=int(smooth_radius),
            stable_peak_min_levels=int(stable_peak_min_levels),
            peak_merge_distance=float(peak_merge_distance),
        )
        results.append(
            (
                int(head),
                {
                    "label_map_fhw_np": candidate_data["label_map_fhw"].detach().cpu().numpy().astype(
                        np.int16,
                        copy=False,
                    ),
                    "frame_metadata": candidate_data["frame_metadata"],
                },
            )
        )
    return int(step), int(layer), results


def _trajectory_consensus_render_candidate_viz_task(task: Tuple) -> str:
    """Worker task for one candidate-region visualization."""
    (
        raw_map_fhw,
        label_map_fhw,
        save_file,
        title,
        attention_frame_indices,
        video_frame_labels,
        draw_candidate_contours,
        raw_map_cmap,
    ) = task
    return _plot_wan21_t2v_candidate_region_viz(
        raw_map_fhw=raw_map_fhw,
        label_map_fhw=label_map_fhw,
        save_file=save_file,
        title=title,
        attention_frame_indices=attention_frame_indices,
        video_frame_labels=video_frame_labels,
        draw_candidate_contours=bool(draw_candidate_contours),
        raw_map_cmap=str(raw_map_cmap),
    )

def _pack_wan21_t2v_candidate_viz_arrays(
    raw_map_fhw: torch.Tensor,
    label_map_fhw: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert plot inputs into compact CPU numpy arrays for process workers."""
    raw_map_np = np.ascontiguousarray(
        raw_map_fhw.detach().cpu().numpy().astype(np.float16, copy=False)
    )
    label_map_np = np.ascontiguousarray(
        label_map_fhw.detach().cpu().numpy().astype(np.int16, copy=False)
    )
    return raw_map_np, label_map_np


def _trajectory_consensus_render_head_specific_candidate_viz_task(task: Tuple) -> str:
    """Worker task that renders one per-head candidate-region plot."""
    (
        raw_map_fhw,
        label_map_fhw,
        save_file,
        title,
        attention_frame_indices,
        video_frame_labels,
        draw_candidate_contours,
        raw_map_cmap,
    ) = task
    return _plot_wan21_t2v_candidate_region_viz(
        raw_map_fhw=raw_map_fhw,
        label_map_fhw=label_map_fhw,
        save_file=save_file,
        title=title,
        attention_frame_indices=attention_frame_indices,
        video_frame_labels=video_frame_labels,
        draw_candidate_contours=bool(draw_candidate_contours),
        raw_map_cmap=str(raw_map_cmap),
    )


def _plot_wan21_t2v_trajectory_consensus_curve(
    rows: Sequence[Dict[str, object]],
    save_file: str,
    x_key: str,
    y_key: str,
    title: str,
    x_label: str,
    y_label: str,
    group_key: str = "",
):
    """Render one or multiple line curves from row dictionaries."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return ""

    if not group_key:
        grouped_rows = {"all": list(rows)}
    else:
        grouped_rows = defaultdict(list)
        for row in rows:
            grouped_rows[str(row[group_key])].append(row)

    fig, axis = plt.subplots(1, 1, figsize=(8.2, 5.0))
    color_map = plt.get_cmap("gist_ncar")
    group_names = sorted(grouped_rows.keys())
    for group_index, group_name in enumerate(group_names):
        plot_rows = [row for row in grouped_rows[group_name] if x_key in row and y_key in row and row[y_key] != ""]
        if not plot_rows:
            continue
        plot_rows = sorted(plot_rows, key=lambda row: float(row[x_key]))
        xs = [float(row[x_key]) for row in plot_rows]
        ys = [float(row[y_key]) for row in plot_rows]
        axis.plot(xs, ys, marker="o", linewidth=1.5, color=color_map((group_index + 0.5) / max(1, len(group_names))), label=group_name)

    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axis.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    axis.grid(alpha=0.22, linestyle="--")
    if group_key and len(group_names) <= 20:
        axis.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file


def _classify_wan21_t2v_scatter_split_class(
    x_value: float,
    y_value: float,
) -> str:
    """Assign one point to the requested split-class bucket."""
    if float(y_value) > 500.0:
        return "high_contribution"
    if float(x_value) <= 0.1:
        return "low_speed"
    return "high_speed"


_SPLIT_CLASS_STYLE = {
    "high_contribution": {
        "label": "contribution > 500",
        "color": "#8fdf93",
        "edge": "#000000",
    },
    "low_speed": {
        "label": "contribution <= 500 and speed <= 0.1",
        "color": "#f0d84f",
        "edge": "#000000",
    },
    "high_speed": {
        "label": "contribution <= 500 and speed > 0.1",
        "color": "#7fb3f0",
        "edge": "#000000",
    },
}


def _plot_wan21_t2v_trajectory_consensus_scatter(
    rows: Sequence[Dict[str, object]],
    save_file: str,
    x_key: str,
    y_key: str,
    title: str,
    x_label: str,
    y_label: str,
    split_classes: bool = False,
):
    """Render a scatter plot from row dictionaries."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    plot_rows = [row for row in rows if row.get(x_key, "") != "" and row.get(y_key, "") != ""]
    if not plot_rows:
        return ""

    xs = [float(row[x_key]) for row in plot_rows]
    ys = [float(row[y_key]) for row in plot_rows]

    fig, axis = plt.subplots(1, 1, figsize=(6.8, 5.2))
    if bool(split_classes):
        class_to_points = defaultdict(lambda: {"x": [], "y": []})
        for x_value, y_value in zip(xs, ys):
            class_name = _classify_wan21_t2v_scatter_split_class(x_value, y_value)
            class_to_points[class_name]["x"].append(float(x_value))
            class_to_points[class_name]["y"].append(float(y_value))
        for class_name in ("high_contribution", "low_speed", "high_speed"):
            payload = class_to_points.get(class_name)
            if not payload or not payload["x"]:
                continue
            style = _SPLIT_CLASS_STYLE[class_name]
            axis.scatter(
                payload["x"],
                payload["y"],
                s=22,
                alpha=0.72,
                color=style["color"],
                edgecolors=style["edge"],
                linewidths=0.5,
                label=style["label"],
            )
        axis.legend(fontsize=8, frameon=True)
    else:
        axis.scatter(
            xs,
            ys,
            s=20,
            alpha=0.82,
            color="#0f766e",
            edgecolors="none",
        )
    if len(xs) >= 2:
        xs_np = np.asarray(xs, dtype=np.float64)
        ys_np = np.asarray(ys, dtype=np.float64)
        if np.isfinite(xs_np).all() and np.isfinite(ys_np).all() and float(np.std(xs_np)) > 1e-12:
            slope, intercept = np.polyfit(xs_np, ys_np, deg=1)
            fit_xs = np.linspace(float(xs_np.min()), float(xs_np.max()), num=200, dtype=np.float64)
            fit_ys = slope * fit_xs + intercept
            axis.plot(
                fit_xs,
                fit_ys,
                color="#dc2626",
                linewidth=1.8,
                alpha=0.92,
            )
    axis.set_title(title)
    axis.set_xlabel(x_label, fontsize=17)
    axis.set_ylabel(y_label, fontsize=17)
    axis.tick_params(axis="both", labelsize=14)
    axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    sci_formatter = ScalarFormatter(useMathText=True)
    sci_formatter.set_powerlimits((0, 0))
    axis.yaxis.set_major_formatter(sci_formatter)
    axis.grid(alpha=0.22, linestyle="--")
    fig.tight_layout()
    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file


def _plot_wan21_t2v_trajectory_consensus_interactive_scatter(
    rows: Sequence[Dict[str, object]],
    save_file: str,
    x_key: str,
    y_key: str,
    title: str,
    x_label: str,
    y_label: str,
    split_classes: bool = False,
):
    """Render an interactive HTML scatter plot with hover tips and head search."""
    plot_rows = [row for row in rows if row.get(x_key, "") != "" and row.get(y_key, "") != ""]
    if not plot_rows:
        return ""

    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except Exception:
        return ""

    ordered_rows = sorted(
        plot_rows,
        key=lambda row: (int(row.get("step", -1)), str(row.get("head_tag", ""))),
    )
    xs = [float(row[x_key]) for row in ordered_rows]
    ys = [float(row[y_key]) for row in ordered_rows]
    steps = [int(row.get("step", -1)) for row in ordered_rows]
    head_tags = [str(row.get("head_tag", "")) for row in ordered_rows]
    labels = [
        f"T{int(row.get('step', -1))}{str(row.get('head_tag', ''))}"
        for row in ordered_rows
    ]
    customdata = [
        [
            labels[index],
            steps[index],
            str(row.get("module", "")),
            str(row.get("branch", "")),
            str(row.get("metric", "")),
            head_tags[index],
        ]
        for index, row in enumerate(ordered_rows)
    ]

    if bool(split_classes):
        marker_colors = []
        marker_line_colors = []
        for x_value, y_value in zip(xs, ys):
            class_name = _classify_wan21_t2v_scatter_split_class(x_value, y_value)
            style = _SPLIT_CLASS_STYLE[class_name]
            marker_colors.append(style["color"])
            marker_line_colors.append(style["edge"])
    else:
        marker_colors = steps
        marker_line_colors = ["rgba(0,0,0,0)"] * len(xs)

    figure = go.Figure(
        data=[
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(
                    size=7,
                    color=marker_colors,
                    colorscale=None if bool(split_classes) else "Viridis",
                    showscale=bool(not split_classes),
                    colorbar=dict(title="step") if not split_classes else None,
                    opacity=0.76 if split_classes else 0.82,
                    line=dict(color=marker_line_colors, width=0.75 if split_classes else 0.0),
                ),
                customdata=customdata,
                hovertemplate=(
                    "id=%{customdata[0]}<br>"
                    "step=%{customdata[1]}<br>"
                    "module=%{customdata[2]}<br>"
                    "branch=%{customdata[3]}<br>"
                    "metric=%{customdata[4]}<br>"
                    "head=%{customdata[5]}<br>"
                    f"{x_label}=%{{x:.6g}}<br>"
                    f"{y_label}=%{{y:.6g}}<extra></extra>"
                ),
                showlegend=False,
            ),
            go.Scatter(
                x=[],
                y=[],
                mode="markers",
                marker=dict(
                    size=12,
                    color="red",
                    opacity=0.96,
                    line=dict(color="white", width=0.9),
                ),
                customdata=[],
                hovertemplate=(
                    "id=%{customdata[0]}<br>"
                    "step=%{customdata[1]}<br>"
                    "module=%{customdata[2]}<br>"
                    "branch=%{customdata[3]}<br>"
                    "metric=%{customdata[4]}<br>"
                    "head=%{customdata[5]}<br>"
                    f"{x_label}=%{{x:.6g}}<br>"
                    f"{y_label}=%{{y:.6g}}<extra></extra>"
                ),
                showlegend=False,
            ),
        ]
    )
    if len(xs) >= 2:
        xs_np = np.asarray(xs, dtype=np.float64)
        ys_np = np.asarray(ys, dtype=np.float64)
        if np.isfinite(xs_np).all() and np.isfinite(ys_np).all() and float(np.std(xs_np)) > 1e-12:
            slope, intercept = np.polyfit(xs_np, ys_np, deg=1)
            fit_xs = np.linspace(float(xs_np.min()), float(xs_np.max()), num=200, dtype=np.float64)
            fit_ys = slope * fit_xs + intercept
            figure.add_trace(
                go.Scatter(
                    x=fit_xs.tolist(),
                    y=fit_ys.tolist(),
                    mode="lines",
                    line=dict(color="#dc2626", width=1.8),
                    opacity=0.92,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    if bool(split_classes):
        for class_name in ("high_contribution", "low_speed", "high_speed"):
            style = _SPLIT_CLASS_STYLE[class_name]
            figure.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=style["color"],
                        opacity=0.76,
                        line=dict(color=style["edge"], width=0.75),
                    ),
                    name=style["label"],
                    hoverinfo="skip",
                    showlegend=True,
                )
            )
    figure.update_layout(
        title=title,
        xaxis=dict(
            title=dict(text=x_label, font=dict(size=20)),
            tickfont=dict(size=14),
        ),
        yaxis=dict(
            title=dict(text=y_label, font=dict(size=20)),
            tickfont=dict(size=14),
            exponentformat="e",
            showexponent="all",
        ),
        template="plotly_white",
        width=760,
        height=560,
        margin=dict(l=72, r=28, t=72, b=68),
        legend=dict(
            bgcolor="rgba(255,255,255,0.86)",
            bordercolor="rgba(0,0,0,0.08)",
            borderwidth=1,
        ),
    )
    _ensure_dir(os.path.dirname(save_file))
    plot_div_id = f"traj_consensus_scatter_{abs(hash(save_file))}"
    search_input_id = f"{plot_div_id}_search"
    search_button_id = f"{plot_div_id}_button"
    search_status_id = f"{plot_div_id}_status"
    plot_html = pio.to_html(
        figure,
        include_plotlyjs="cdn",
        full_html=False,
        div_id=plot_div_id,
        default_width="760px",
        default_height="560px",
    )
    html = rf"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
</head>
<body>
  <div style="margin: 12px auto 10px auto; width: 760px; font-family: sans-serif;">
    <label for="{search_input_id}" style="margin-right: 8px;"><strong>Find head</strong></label>
    <input
      id="{search_input_id}"
      type="text"
      placeholder="L20H5 or 20,5 or L20H5,L21H3"
      style="width: 320px; padding: 4px 8px;"
    />
    <button id="{search_button_id}" type="button" style="margin-left: 8px; padding: 4px 10px;">
      Highlight
    </button>
    <span id="{search_status_id}" style="margin-left: 12px; color: #555;"></span>
  </div>
  <div style="width: 760px; margin: 0 auto;">
    {plot_html}
  </div>
  <script>
    (function() {{
      const plotDiv = document.getElementById("{plot_div_id}");
      const inputEl = document.getElementById("{search_input_id}");
      const buttonEl = document.getElementById("{search_button_id}");
      const statusEl = document.getElementById("{search_status_id}");

      function normalizeHeadQuery(query) {{
        const text = (query || "").trim().toUpperCase();
        if (!text) {{
          return "";
        }}
        const explicit = text.match(/^L\s*(\d+)\s*H\s*(\d+)$/);
        if (explicit) {{
          return `L${{parseInt(explicit[1], 10)}}H${{parseInt(explicit[2], 10)}}`;
        }}
        const nums = text.match(/\d+/g);
        if (nums && nums.length >= 2) {{
          return `L${{parseInt(nums[0], 10)}}H${{parseInt(nums[1], 10)}}`;
        }}
        return text.replace(/\s+/g, "");
      }}

      function parseHeadQueries(query) {{
        const rawText = String(query || "").trim();
        if (!rawText) {{
          return [];
        }}
        const normalizedWhole = normalizeHeadQuery(rawText);
        if (/^\s*\d+\s*,\s*\d+\s*$/.test(rawText) || /^\s*L/i.test(rawText) === false && rawText.indexOf(",") >= 0 && rawText.split(",").length === 2 && /^L\d+H\d+$/.test(normalizedWhole)) {{
          return normalizedWhole ? [normalizedWhole] : [];
        }}
        const rawParts = rawText.split(",");
        const normalizedParts = [];
        const seen = new Set();
        for (let i = 0; i < rawParts.length; i += 1) {{
          const normalized = normalizeHeadQuery(rawParts[i]);
          if (!normalized || seen.has(normalized)) {{
            continue;
          }}
          seen.add(normalized);
          normalizedParts.push(normalized);
        }}
        return normalizedParts;
      }}

      function highlightHead() {{
        const normalizedQueries = parseHeadQueries(inputEl.value);
        const normalizedSet = new Set(normalizedQueries);
        const baseTrace = plotDiv.data[0];
        const matchedX = [];
        const matchedY = [];
        const matchedCustom = [];
        if (normalizedQueries.length === 0) {{
          Plotly.restyle(plotDiv, {{ x: [[]], y: [[]], customdata: [[]] }}, [1]);
          statusEl.textContent = "highlight cleared";
          return;
        }}
        const xs = baseTrace.x || [];
        const ys = baseTrace.y || [];
        const custom = baseTrace.customdata || [];
        for (let i = 0; i < custom.length; i += 1) {{
          const headTag = String(custom[i][5] || "").toUpperCase();
          if (normalizedSet.has(headTag)) {{
            matchedX.push(xs[i]);
            matchedY.push(ys[i]);
            matchedCustom.push(custom[i]);
          }}
        }}
        Plotly.restyle(plotDiv, {{
          x: [matchedX],
          y: [matchedY],
          customdata: [matchedCustom],
        }}, [1]);
        const querySummary = normalizedQueries.join(", ");
        statusEl.textContent = matchedX.length > 0
          ? `highlighted ${{matchedX.length}} point(s) for ${{querySummary}}`
          : `no point found for ${{querySummary}}`;
      }}

      buttonEl.addEventListener("click", highlightHead);
      inputEl.addEventListener("keydown", function(event) {{
        if (event.key === "Enter") {{
          highlightHead();
        }}
      }});
    }})();
  </script>
</body>
</html>
"""
    with open(save_file, 'w', encoding='utf-8') as handle:
        handle.write(html)
    return save_file


def _smooth_wan21_t2v_map_fhw(map_fhw: torch.Tensor, smooth_radius: int) -> torch.Tensor:
    """Apply small average smoothing over spatial dimensions when requested."""
    if int(smooth_radius) <= 0:
        return map_fhw.detach().float()
    kernel_size = 2 * int(smooth_radius) + 1
    x = map_fhw.detach().float().unsqueeze(1)
    x = torch.nn.functional.avg_pool2d(
        x,
        kernel_size=kernel_size,
        stride=1,
        padding=int(smooth_radius),
    )
    return x.squeeze(1)


def _wan21_t2v_candidate_mask_from_points(
    points_hw: Sequence[Tuple[int, int]],
    shape_hw: Tuple[int, int],
) -> torch.Tensor:
    """Build a boolean mask from a list of token-grid coordinates."""
    mask = torch.zeros((int(shape_hw[0]), int(shape_hw[1])), dtype=torch.bool)
    for point_y, point_x in points_hw:
        if 0 <= int(point_y) < int(shape_hw[0]) and 0 <= int(point_x) < int(shape_hw[1]):
            mask[int(point_y), int(point_x)] = True
    return mask


def _wan21_t2v_candidate_mask_largest_component(mask_hw: torch.Tensor) -> torch.Tensor:
    """Keep only the largest 8-neighborhood connected component of one mask."""
    components = _extract_wan21_t2v_connected_components(mask_hw)
    if not components:
        return torch.zeros_like(mask_hw, dtype=torch.bool)
    largest_component = max(components, key=len)
    out = torch.zeros_like(mask_hw, dtype=torch.bool)
    for point_y, point_x in largest_component:
        out[int(point_y), int(point_x)] = True
    return out


def _wan21_t2v_candidate_mask_bbox_stats(mask_hw: torch.Tensor) -> Dict[str, float]:
    """Compute bounding-box statistics for one boolean candidate mask."""
    if mask_hw.dim() != 2:
        raise ValueError(f"Expected [H, W] mask, got shape={tuple(mask_hw.shape)}")
    points = torch.nonzero(mask_hw, as_tuple=False)
    if int(points.numel()) <= 0:
        return {
            "bbox_height": 0.0,
            "bbox_width": 0.0,
            "bbox_y_min": 0.0,
            "bbox_y_max": 0.0,
            "bbox_x_min": 0.0,
            "bbox_x_max": 0.0,
        }
    y_min = int(points[:, 0].min().item())
    y_max = int(points[:, 0].max().item())
    x_min = int(points[:, 1].min().item())
    x_max = int(points[:, 1].max().item())
    return {
        "bbox_height": float(y_max - y_min + 1),
        "bbox_width": float(x_max - x_min + 1),
        "bbox_y_min": float(y_min),
        "bbox_y_max": float(y_max),
        "bbox_x_min": float(x_min),
        "bbox_x_max": float(x_max),
    }


def _wan21_t2v_candidate_local_maxima_mask(
    score_hw: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Return an 8-neighborhood local-maxima mask above a score threshold."""
    if score_hw.dim() != 2:
        raise ValueError(f"Expected [H, W], got shape={tuple(score_hw.shape)}")
    if float(score_hw.max().item()) <= float(threshold):
        return torch.zeros_like(score_hw, dtype=torch.bool)
    pooled = torch.nn.functional.max_pool2d(
        score_hw.unsqueeze(0).unsqueeze(0),
        kernel_size=3,
        stride=1,
        padding=1,
    ).squeeze(0).squeeze(0)
    return (score_hw >= float(threshold)) & torch.isclose(score_hw, pooled)


@dataclass
class Wan21T2VCandidateSeedProposal:
    """One local-maximum proposal used to seed candidate clustering."""

    y: float
    x: float
    score: float
    support_level: float
    support_count: int = 1
    weight_sum: float = 0.0

    def __post_init__(self):
        if float(self.weight_sum) <= 0.0:
            self.weight_sum = float(self.score)

    @property
    def center(self) -> torch.Tensor:
        return torch.tensor([float(self.y), float(self.x)], dtype=torch.float32)


def _wan21_t2v_merge_candidate_seed_proposals(
    proposals: Sequence[Wan21T2VCandidateSeedProposal],
    merge_distance: float,
    min_support_levels: int,
    max_seeds: int,
) -> List[Wan21T2VCandidateSeedProposal]:
    """Greedily merge seed proposals using a spatial non-maximum suppression rule."""
    if not proposals:
        return []

    kept: List[Wan21T2VCandidateSeedProposal] = []
    max_distance = max(1e-6, float(merge_distance))
    for proposal in sorted(proposals, key=lambda item: (float(item.score), float(item.support_count)), reverse=True):
        assigned_index = -1
        for idx, kept_seed in enumerate(kept):
            distance = float(torch.norm(proposal.center - kept_seed.center).item())
            if distance <= max_distance:
                assigned_index = idx
                break
        if assigned_index < 0:
            kept.append(
                Wan21T2VCandidateSeedProposal(
                    y=float(proposal.y),
                    x=float(proposal.x),
                    score=float(proposal.score),
                    support_level=float(proposal.support_level),
                    support_count=int(proposal.support_count),
                    weight_sum=float(proposal.weight_sum),
                )
            )
            continue

        seed = kept[assigned_index]
        new_support_count = int(seed.support_count) + int(proposal.support_count)
        new_weight_sum = float(seed.weight_sum) + float(proposal.weight_sum)
        if new_weight_sum <= 0.0:
            new_weight_sum = float(seed.weight_sum + proposal.weight_sum + 1e-12)
        seed.y = float(
            (float(seed.y) * float(seed.weight_sum) + float(proposal.y) * float(proposal.weight_sum))
            / max(1e-12, new_weight_sum)
        )
        seed.x = float(
            (float(seed.x) * float(seed.weight_sum) + float(proposal.x) * float(proposal.weight_sum))
            / max(1e-12, new_weight_sum)
        )
        seed.score = float(max(float(seed.score), float(proposal.score)))
        seed.support_level = float(max(float(seed.support_level), float(proposal.support_level)))
        seed.support_count = int(new_support_count)
        seed.weight_sum = float(new_weight_sum)

    filtered = [seed for seed in kept if int(seed.support_count) >= int(max(1, min_support_levels))]
    if not filtered and kept:
        filtered = [max(kept, key=lambda item: (int(item.support_count), float(item.score)))]
    filtered = sorted(filtered, key=lambda item: (int(item.support_count), float(item.score)), reverse=True)
    if int(max_seeds) > 0:
        filtered = filtered[: int(max_seeds)]
    return filtered


def _wan21_t2v_weighted_kmeans_from_seeds(
    point_yx: torch.Tensor,
    point_weights: torch.Tensor,
    initial_centers_yx: torch.Tensor,
    max_iter: int = 10,
    center_tol: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run a seeded weighted k-means update on token-grid points.

    The cluster assignment is computed on the high-confidence support set only.
    Cluster centers are updated by weighted averaging, using the local attention
    score as point weight.
    """
    if point_yx.numel() <= 0 or initial_centers_yx.numel() <= 0:
        return torch.zeros((0, 2), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)
    centers = initial_centers_yx.detach().float().clone()
    points = point_yx.detach().float()
    weights = point_weights.detach().float().clamp_min(0.0)
    max_iter = max(1, int(max_iter))
    center_tol = max(0.0, float(center_tol))

    for _ in range(max_iter):
        distances = torch.cdist(points, centers)
        assignments = distances.argmin(dim=1)
        new_centers: List[torch.Tensor] = []
        max_shift = 0.0
        for center_index in range(int(centers.shape[0])):
            cluster_mask = assignments == int(center_index)
            if not bool(cluster_mask.any().item()):
                continue
            cluster_points = points[cluster_mask]
            cluster_weights = weights[cluster_mask]
            weight_sum = float(cluster_weights.sum().item())
            if weight_sum <= 1e-12:
                new_center = cluster_points.mean(dim=0)
            else:
                new_center = (cluster_points * cluster_weights.unsqueeze(1)).sum(dim=0) / weight_sum
            new_centers.append(new_center)
            max_shift = max(max_shift, float(torch.norm(new_center - centers[center_index]).item()))
        if not new_centers:
            return torch.zeros((0, 2), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)
        new_centers_tensor = torch.stack(new_centers, dim=0)
        if int(new_centers_tensor.shape[0]) == int(centers.shape[0]) and max_shift <= center_tol:
            centers = new_centers_tensor
            break
        centers = new_centers_tensor

    final_distances = torch.cdist(points, centers)
    final_assignments = final_distances.argmin(dim=1)
    return centers, final_assignments


def _extract_wan21_t2v_candidate_regions_for_frame(
    frame_hw: torch.Tensor,
    base_quantile: float,
    split_quantiles: Sequence[float],
    min_component_area: int,
    stable_peak_min_levels: int,
    peak_merge_distance: float,
) -> Tuple[torch.Tensor, List[Dict[str, object]]]:
    """Extract candidate regions on one frame as an integer label map.

    The current implementation uses a peak-seeded weighted clustering scheme:
    1. suppress background with a quantile-based contrast transform;
    2. propose local maxima at several seed-support levels;
    3. greedily merge repeated proposals into stable seeds;
    4. run seeded weighted k-means on the high-confidence support set;
    5. trim each cluster to a compact core and reject wide / thin / tiny regions.
    """
    frame = frame_hw.detach().float().clamp_min(0.0)
    token_grid_height, token_grid_width = frame.shape
    base_quantile = max(0.0, min(1.0, float(base_quantile)))
    area_threshold = max(1, int(min_component_area))
    max_seed_count = 5
    background_quantile = max(0.5, min(float(base_quantile) - 0.10, 0.92))
    seed_quantiles = sorted(
        {
            max(float(base_quantile), min(0.995, float(level)))
            for level in split_quantiles
            if str(level).strip() != ""
        }
    )
    if not seed_quantiles:
        seed_quantiles = [max(float(base_quantile), min(0.995, float(base_quantile) + 0.08))]
    seed_quantiles = seed_quantiles[: max_seed_count]
    core_mass_fraction = 0.80
    max_width_ratio = 0.40
    max_height_ratio = 0.95
    if int(peak_merge_distance) <= 0:
        peak_merge_distance = 2.0
    seed_merge_distance = max(1.0, float(peak_merge_distance))

    if float(frame.max().item()) <= 0.0:
        label_map = torch.zeros_like(frame, dtype=torch.int64)
        peak_index = int(frame.reshape(-1).argmax().item())
        peak_y = int(peak_index // int(token_grid_width))
        peak_x = int(peak_index % int(token_grid_width))
        label_map[peak_y, peak_x] = 1
        return label_map, [{
            "candidate_index": 1,
            "area": 1,
            "peak_y": float(peak_y),
            "peak_x": float(peak_x),
            "centroid_y": float(peak_y),
            "centroid_x": float(peak_x),
        }]

    background_threshold = float(torch.quantile(frame.reshape(-1), background_quantile).item())
    contrast_map = torch.clamp(frame - background_threshold, min=0.0)
    if float(contrast_map.max().item()) <= 0.0:
        peak_index = int(frame.reshape(-1).argmax().item())
        peak_y = int(peak_index // int(token_grid_width))
        peak_x = int(peak_index % int(token_grid_width))
        label_map = torch.zeros_like(frame, dtype=torch.int64)
        label_map[peak_y, peak_x] = 1
        return label_map, [{
            "candidate_index": 1,
            "area": 1,
            "peak_y": float(peak_y),
            "peak_x": float(peak_x),
            "centroid_y": float(peak_y),
            "centroid_x": float(peak_x),
        }]
    contrast_max = float(contrast_map.max().item())
    if contrast_max > 0.0:
        contrast_map = contrast_map / contrast_max
    contrast_map = contrast_map.pow(2.0)

    positive_values = contrast_map[contrast_map > 0.0]
    if int(positive_values.numel()) <= 0:
        peak_index = int(frame.reshape(-1).argmax().item())
        peak_y = int(peak_index // int(token_grid_width))
        peak_x = int(peak_index % int(token_grid_width))
        label_map = torch.zeros_like(frame, dtype=torch.int64)
        label_map[peak_y, peak_x] = 1
        return label_map, [{
            "candidate_index": 1,
            "area": 1,
            "peak_y": float(peak_y),
            "peak_x": float(peak_x),
            "centroid_y": float(peak_y),
            "centroid_x": float(peak_x),
        }]

    support_threshold = float(torch.quantile(positive_values.reshape(-1), base_quantile).item())
    support_mask = contrast_map >= support_threshold
    support_points = torch.nonzero(support_mask, as_tuple=False).long()
    support_weights = contrast_map[support_mask].reshape(-1)
    if int(support_points.shape[0]) <= 0 or float(support_weights.sum().item()) <= 0.0:
        peak_index = int(frame.reshape(-1).argmax().item())
        peak_y = int(peak_index // int(token_grid_width))
        peak_x = int(peak_index % int(token_grid_width))
        label_map = torch.zeros_like(frame, dtype=torch.int64)
        label_map[peak_y, peak_x] = 1
        return label_map, [{
            "candidate_index": 1,
            "area": 1,
            "peak_y": float(peak_y),
            "peak_x": float(peak_x),
            "centroid_y": float(peak_y),
            "centroid_x": float(peak_x),
        }]

    seed_proposals: List[Wan21T2VCandidateSeedProposal] = []
    for support_level in seed_quantiles:
        level_threshold = float(torch.quantile(positive_values.reshape(-1), support_level).item())
        level_mask = support_mask & (contrast_map >= level_threshold)
        local_max_mask = _wan21_t2v_candidate_local_maxima_mask(
            contrast_map,
            threshold=float(level_threshold),
        ) & level_mask
        if not bool(local_max_mask.any().item()):
            continue
        for component in _extract_wan21_t2v_connected_components(local_max_mask):
            if not component:
                continue
            peak_y, peak_x = max(
                component,
                key=lambda point: float(contrast_map[int(point[0]), int(point[1])].item()),
            )
            peak_score = float(contrast_map[int(peak_y), int(peak_x)].item())
            if peak_score <= 0.0:
                continue
            seed_proposals.append(
                Wan21T2VCandidateSeedProposal(
                    y=float(peak_y),
                    x=float(peak_x),
                    score=float(peak_score),
                    support_level=float(support_level),
                    support_count=1,
                    weight_sum=float(peak_score),
                )
            )

    if not seed_proposals:
        peak_index = int(frame.reshape(-1).argmax().item())
        peak_y = int(peak_index // int(token_grid_width))
        peak_x = int(peak_index % int(token_grid_width))
        label_map = torch.zeros_like(frame, dtype=torch.int64)
        label_map[peak_y, peak_x] = 1
        return label_map, [{
            "candidate_index": 1,
            "area": 1,
            "peak_y": float(peak_y),
            "peak_x": float(peak_x),
            "centroid_y": float(peak_y),
            "centroid_x": float(peak_x),
        }]

    stable_seeds = _wan21_t2v_merge_candidate_seed_proposals(
        proposals=seed_proposals,
        merge_distance=float(seed_merge_distance),
        min_support_levels=int(stable_peak_min_levels),
        max_seeds=int(max_seed_count),
    )
    if not stable_seeds:
        stable_seeds = [
            max(seed_proposals, key=lambda item: (float(item.score), int(item.support_count)))
        ]

    stable_seed_tensor = torch.stack([seed.center for seed in stable_seeds], dim=0).to(dtype=torch.float32)

    label_map = torch.zeros_like(frame, dtype=torch.int64)
    candidate_metadata: List[Dict[str, object]] = []
    next_candidate_index = 1

    cluster_centers, cluster_assignments = _wan21_t2v_weighted_kmeans_from_seeds(
        point_yx=support_points.to(dtype=torch.float32),
        point_weights=support_weights.to(dtype=torch.float32),
        initial_centers_yx=stable_seed_tensor,
        max_iter=10,
        center_tol=0.25,
    )

    if int(cluster_centers.shape[0]) <= 0:
        peak_index = int(frame.reshape(-1).argmax().item())
        peak_y = int(peak_index // int(token_grid_width))
        peak_x = int(peak_index % int(token_grid_width))
        label_map[peak_y, peak_x] = 1
        candidate_metadata.append({
            "candidate_index": 1,
            "area": 1,
            "mass": float(frame[peak_y, peak_x].item()),
            "density": float(frame[peak_y, peak_x].item()),
            "bbox_height": 1.0,
            "bbox_width": 1.0,
            "bbox_y_min": float(peak_y),
            "bbox_y_max": float(peak_y),
            "bbox_x_min": float(peak_x),
            "bbox_x_max": float(peak_x),
            "peak_y": float(peak_y),
            "peak_x": float(peak_x),
            "centroid_y": float(peak_y),
            "centroid_x": float(peak_x),
            "seed_y": float(peak_y),
            "seed_x": float(peak_x),
            "seed_score": float(frame[peak_y, peak_x].item()),
            "support_count": 1,
        })
        return label_map, candidate_metadata

    cluster_infos: List[Dict[str, object]] = []
    for cluster_index in range(int(cluster_centers.shape[0])):
        cluster_mask = cluster_assignments == int(cluster_index)
        if not bool(cluster_mask.any().item()):
            continue
        cluster_points = support_points[cluster_mask]
        cluster_weights = support_weights[cluster_mask]
        cluster_center = cluster_centers[cluster_index]
        cluster_mass = float(cluster_weights.sum().item())
        if cluster_mass <= 0.0:
            continue
        distances = torch.norm(cluster_points.to(dtype=torch.float32) - cluster_center.unsqueeze(0), dim=1)
        sorted_indices = torch.argsort(distances)
        sorted_weights = cluster_weights[sorted_indices]
        cumulative_mass = torch.cumsum(sorted_weights, dim=0)
        target_mass = float(cluster_mass * float(core_mass_fraction))
        keep_count = int(torch.searchsorted(cumulative_mass, torch.tensor(target_mass, dtype=torch.float32), right=False).item()) + 1
        keep_count = max(1, min(int(keep_count), int(sorted_indices.numel())))
        selected_points = cluster_points[sorted_indices[:keep_count]]
        selected_weights = cluster_weights[sorted_indices[:keep_count]]
        selected_mask = _wan21_t2v_candidate_mask_from_points(
            points_hw=[(int(point[0].item()), int(point[1].item())) for point in selected_points],
            shape_hw=(int(token_grid_height), int(token_grid_width)),
        )
        selected_mask = _wan21_t2v_candidate_mask_largest_component(selected_mask)
        selected_points = torch.nonzero(selected_mask, as_tuple=False).long()
        if int(selected_points.shape[0]) < int(area_threshold):
            continue
        selected_weights = contrast_map[selected_mask].reshape(-1)
        if float(selected_weights.sum().item()) <= 0.0:
            continue
        candidate_mass = float(selected_weights.sum().item())
        candidate_area = int(selected_points.shape[0])
        candidate_density = float(candidate_mass / max(1, candidate_area))
        support_density = float(cluster_mass / max(1, int(cluster_points.shape[0])))
        if candidate_density < support_density:
            continue
        peak_y, peak_x = max(
            [(int(point[0].item()), int(point[1].item())) for point in selected_points],
            key=lambda point: float(contrast_map[int(point[0]), int(point[1])].item()),
        )
        bbox_stats = _wan21_t2v_candidate_mask_bbox_stats(selected_mask)
        bbox_height_ratio = float(bbox_stats["bbox_height"]) / max(1.0, float(token_grid_height))
        bbox_width_ratio = float(bbox_stats["bbox_width"]) / max(1.0, float(token_grid_width))
        if bbox_width_ratio > float(max_width_ratio) or bbox_height_ratio > float(max_height_ratio):
            continue
        candidate_mask = torch.zeros_like(frame, dtype=torch.bool)
        for point_y, point_x in selected_points.tolist():
            candidate_mask[int(point_y), int(point_x)] = True
        candidate_points = torch.nonzero(candidate_mask, as_tuple=False).long()
        if int(candidate_points.shape[0]) < int(area_threshold):
            continue
        label_map[candidate_mask] = int(next_candidate_index)
        centroid_y = float(candidate_points[:, 0].float().mean().item())
        centroid_x = float(candidate_points[:, 1].float().mean().item())
        cluster_seed = min(
            stable_seeds,
            key=lambda seed: float((seed.center - cluster_center).norm().item()),
        )
        cluster_infos.append({
            "candidate_index": int(next_candidate_index),
            "area": int(candidate_points.shape[0]),
            "mass": float(candidate_mass),
            "density": float(candidate_density),
            "bbox_height": float(bbox_stats["bbox_height"]),
            "bbox_width": float(bbox_stats["bbox_width"]),
            "bbox_y_min": float(bbox_stats["bbox_y_min"]),
            "bbox_y_max": float(bbox_stats["bbox_y_max"]),
            "bbox_x_min": float(bbox_stats["bbox_x_min"]),
            "bbox_x_max": float(bbox_stats["bbox_x_max"]),
            "peak_y": float(peak_y),
            "peak_x": float(peak_x),
            "centroid_y": float(centroid_y),
            "centroid_x": float(centroid_x),
            "seed_y": float(cluster_seed.y),
            "seed_x": float(cluster_seed.x),
            "seed_score": float(cluster_seed.score),
            "support_count": int(cluster_seed.support_count),
            "support_level": float(cluster_seed.support_level),
        })
        next_candidate_index += 1

    candidate_metadata.extend(cluster_infos)

    if not candidate_metadata:
        peak_index = int(frame.reshape(-1).argmax().item())
        peak_y = int(peak_index // int(token_grid_width))
        peak_x = int(peak_index % int(token_grid_width))
        label_map[peak_y, peak_x] = 1
        candidate_metadata.append({
            "candidate_index": 1,
            "area": 1,
            "mass": float(frame[peak_y, peak_x].item()),
            "density": float(frame[peak_y, peak_x].item()),
            "bbox_height": 1.0,
            "bbox_width": 1.0,
            "bbox_y_min": float(peak_y),
            "bbox_y_max": float(peak_y),
            "bbox_x_min": float(peak_x),
            "bbox_x_max": float(peak_x),
            "peak_y": float(peak_y),
            "peak_x": float(peak_x),
            "centroid_y": float(peak_y),
            "centroid_x": float(peak_x),
            "seed_y": float(peak_y),
            "seed_x": float(peak_x),
            "seed_score": float(frame[peak_y, peak_x].item()),
            "support_count": 1,
            "support_level": float(base_quantile),
        })

    return label_map, candidate_metadata


def _extract_wan21_t2v_candidate_regions_for_map(
    map_fhw: torch.Tensor,
    base_quantile: float,
    split_quantiles: Sequence[float],
    min_component_area: int,
    smooth_radius: int,
    stable_peak_min_levels: int,
    peak_merge_distance: float,
) -> Dict[str, object]:
    """Extract candidate-region label maps for all frames of one `[F, H, W]` map."""
    preprocessed = _smooth_wan21_t2v_map_fhw(map_fhw, smooth_radius=int(smooth_radius))
    frame_count = int(preprocessed.shape[0])
    label_maps = []
    frame_metadata: List[List[Dict[str, object]]] = []
    for frame_index in range(frame_count):
        label_map_hw, metadata = _extract_wan21_t2v_candidate_regions_for_frame(
            frame_hw=preprocessed[frame_index],
            base_quantile=float(base_quantile),
            split_quantiles=split_quantiles,
            min_component_area=int(min_component_area),
            stable_peak_min_levels=int(stable_peak_min_levels),
            peak_merge_distance=float(peak_merge_distance),
        )
        label_maps.append(label_map_hw.to(torch.int64))
        frame_metadata.append(metadata)
    return {
        "label_map_fhw": torch.stack(label_maps, dim=0),
        "frame_metadata": frame_metadata,
        "preprocessed_map_fhw": preprocessed,
    }


def _compute_wan21_t2v_candidate_weights_for_head_map(
    probability_map_fhw: torch.Tensor,
    label_map_fhw: torch.Tensor,
) -> List[List[float]]:
    """Return candidate weights per frame for one normalized head map."""
    frame_count = int(probability_map_fhw.shape[0])
    all_weights: List[List[float]] = []
    for frame_index in range(frame_count):
        frame_prob = probability_map_fhw[frame_index]
        frame_labels = label_map_fhw[frame_index]
        candidate_count = int(frame_labels.max().item())
        frame_weights: List[float] = []
        for candidate_index in range(1, candidate_count + 1):
            frame_weights.append(float(frame_prob[frame_labels == int(candidate_index)].sum().item()))
        all_weights.append(frame_weights)
    return all_weights


def _build_wan21_t2v_reference_object_boxes(
    reference_map_fhw: torch.Tensor,
    center_mode: str = "centroid",
    traj_power: float = 1.5,
    traj_quantile: float = 0.8,
    support_radius_mode: str = "adaptive_area",
    support_radius_fixed: float = 2.0,
    support_radius_alpha: float = 1.5,
    support_radius_min: float = 1.0,
    support_radius_max_ratio: float = 0.25,
) -> List[Dict[str, float]]:
    """Construct one reference object box per token frame from a reference head-mean map."""
    reference_trajectory_data = _extract_wan21_t2v_reference_peak_and_centroid_trajectory(
        map_fhw=reference_map_fhw,
        power=float(traj_power),
        quantile=float(traj_quantile),
    )
    center_mode = str(center_mode).strip().lower()
    if center_mode == "peak":
        center_trajectory = reference_trajectory_data["peak_centers"]
    elif center_mode == "geometric_center":
        center_trajectory = reference_trajectory_data["geometric_centers"]
    else:
        center_trajectory = reference_trajectory_data["centroid_centers"]

    _, support_radius_per_frame = _build_wan21_t2v_trajectory_support_mask_from_centers(
        center_trajectory=center_trajectory,
        component_areas=reference_trajectory_data["component_areas"],
        token_grid_height=int(reference_map_fhw.shape[1]),
        token_grid_width=int(reference_map_fhw.shape[2]),
        support_radius_mode=str(support_radius_mode),
        support_radius_fixed=float(support_radius_fixed),
        support_radius_alpha=float(support_radius_alpha),
        support_radius_min=float(support_radius_min),
        support_radius_max_ratio=float(support_radius_max_ratio),
    )

    token_grid_height = int(reference_map_fhw.shape[1])
    token_grid_width = int(reference_map_fhw.shape[2])
    boxes: List[Dict[str, float]] = []
    for frame_index, ((center_y, center_x), radius_value) in enumerate(zip(center_trajectory, support_radius_per_frame)):
        half_extent = float(max(1.0, float(radius_value)))
        y0 = max(0.0, float(center_y) - half_extent)
        y1 = min(float(token_grid_height - 1), float(center_y) + half_extent)
        x0 = max(0.0, float(center_x) - half_extent)
        x1 = min(float(token_grid_width - 1), float(center_x) + half_extent)
        boxes.append(
            {
                "frame": float(frame_index),
                "center_y": float(center_y),
                "center_x": float(center_x),
                "radius": float(radius_value),
                "y0": float(y0),
                "y1": float(y1),
                "x0": float(x0),
                "x1": float(x1),
            }
        )
    return boxes


def _merge_wan21_t2v_candidate_regions_by_reference_box(
    label_map_fhw: torch.Tensor,
    frame_metadata: Sequence[Sequence[Dict[str, object]]],
    reference_boxes: Sequence[Dict[str, float]],
    min_overlap_ratio: float = 0.75,
) -> Tuple[torch.Tensor, List[List[Dict[str, object]]]]:
    """Merge fragmented candidate regions that mostly lie inside the reference object box.

    A candidate is merge-eligible only if a sufficient fraction of its own area
    lies inside the frame-wise reference object box. If multiple candidates
    satisfy this overlap-ratio rule, they are merged directly into one object
    region. No additional center-distance, adjacency, or ranking heuristics are
    applied here.
    """
    label_map = label_map_fhw.detach().cpu().to(torch.int64).clone()
    merged_metadata: List[List[Dict[str, object]]] = []
    frame_count = int(label_map.shape[0])
    min_overlap_ratio = max(0.0, min(1.0, float(min_overlap_ratio)))

    for frame_index in range(frame_count):
        frame_label_map = label_map[frame_index]
        original_frame_label_map = frame_label_map.clone()
        frame_box = reference_boxes[frame_index] if frame_index < len(reference_boxes) else None
        if frame_box is None:
            merged_metadata.append([dict(row) for row in frame_metadata[frame_index]])
            continue

        token_grid_height = int(frame_label_map.shape[0])
        token_grid_width = int(frame_label_map.shape[1])
        y0 = max(0, min(token_grid_height - 1, int(math.floor(float(frame_box["y0"])))))
        y1 = max(0, min(token_grid_height - 1, int(math.ceil(float(frame_box["y1"])))))
        x0 = max(0, min(token_grid_width - 1, int(math.floor(float(frame_box["x0"])))))
        x1 = max(0, min(token_grid_width - 1, int(math.ceil(float(frame_box["x1"])))))
        box_mask = torch.zeros_like(frame_label_map, dtype=torch.bool)
        box_mask[y0 : y1 + 1, x0 : x1 + 1] = True

        eligible_candidate_ids: List[int] = []
        frame_candidate_count = int(frame_label_map.max().item())
        original_rows_by_candidate = {
            int(row.get("candidate_index", -1)): dict(row)
            for row in frame_metadata[frame_index]
        }
        candidate_masks: Dict[int, torch.Tensor] = {}
        merged_source_indices: Dict[int, List[int]] = {
            int(candidate_index): [int(candidate_index)]
            for candidate_index in range(1, frame_candidate_count + 1)
        }

        for candidate_index in range(1, frame_candidate_count + 1):
            candidate_mask = original_frame_label_map == int(candidate_index)
            candidate_area = int(candidate_mask.sum().item())
            if candidate_area <= 0:
                continue
            candidate_masks[int(candidate_index)] = candidate_mask
            overlap_area = int(torch.logical_and(candidate_mask, box_mask).sum().item())
            overlap_ratio = float(overlap_area / max(1, candidate_area))
            if overlap_ratio < min_overlap_ratio:
                continue
            eligible_candidate_ids.append(int(candidate_index))

        eligible_candidate_ids = sorted(set(int(candidate_index) for candidate_index in eligible_candidate_ids))
        if len(eligible_candidate_ids) >= 2:
            anchor_index = int(eligible_candidate_ids[0])
            for candidate_index in eligible_candidate_ids[1:]:
                frame_label_map[candidate_masks[int(candidate_index)]] = int(anchor_index)
                merged_source_indices[int(anchor_index)].extend(
                    merged_source_indices.get(int(candidate_index), [int(candidate_index)])
                )
                merged_source_indices[int(candidate_index)] = []

        unique_candidate_indices = sorted(
            int(candidate_index)
            for candidate_index in torch.unique(frame_label_map).tolist()
            if int(candidate_index) > 0
        )
        remapped_label_map = torch.zeros_like(frame_label_map, dtype=torch.int64)
        remapped_rows: List[Dict[str, object]] = []
        for new_candidate_index, old_candidate_index in enumerate(unique_candidate_indices, start=1):
            candidate_mask = frame_label_map == int(old_candidate_index)
            remapped_label_map[candidate_mask] = int(new_candidate_index)
            points = torch.nonzero(candidate_mask, as_tuple=False).float()
            candidate_area = int(points.shape[0])
            bbox_stats = _wan21_t2v_candidate_mask_bbox_stats(candidate_mask)
            centroid_y = float(points[:, 0].mean().item()) if candidate_area > 0 else 0.0
            centroid_x = float(points[:, 1].mean().item()) if candidate_area > 0 else 0.0
            source_rows = [
                original_rows_by_candidate[source_index]
                for source_index in merged_source_indices.get(int(old_candidate_index), [int(old_candidate_index)])
                if int(source_index) in original_rows_by_candidate
            ]
            if source_rows:
                representative_row = max(
                    source_rows,
                    key=lambda row: (
                        float(row.get("mass", float("-inf"))),
                        int(row.get("area", 0)),
                    ),
                )
                merged_mass = float(
                    sum(float(row.get("mass", 0.0)) for row in source_rows if math.isfinite(float(row.get("mass", float("nan")))))
                )
                if not math.isfinite(merged_mass) or merged_mass <= 0.0:
                    merged_mass = float("nan")
                merged_density = (
                    float(merged_mass / max(1, candidate_area))
                    if math.isfinite(merged_mass)
                    else float("nan")
                )
            else:
                representative_row = None
                merged_mass = float("nan")
                merged_density = float("nan")
            remapped_rows.append(
                {
                    "candidate_index": int(new_candidate_index),
                    "area": int(candidate_area),
                    "mass": float(merged_mass),
                    "density": float(merged_density),
                    "bbox_height": float(bbox_stats["bbox_height"]),
                    "bbox_width": float(bbox_stats["bbox_width"]),
                    "bbox_y_min": float(bbox_stats["bbox_y_min"]),
                    "bbox_y_max": float(bbox_stats["bbox_y_max"]),
                    "bbox_x_min": float(bbox_stats["bbox_x_min"]),
                    "bbox_x_max": float(bbox_stats["bbox_x_max"]),
                    "peak_y": float(representative_row.get("peak_y", centroid_y)) if representative_row is not None else float(centroid_y),
                    "peak_x": float(representative_row.get("peak_x", centroid_x)) if representative_row is not None else float(centroid_x),
                    "centroid_y": float(centroid_y),
                    "centroid_x": float(centroid_x),
                    "seed_y": float(representative_row.get("seed_y", centroid_y)) if representative_row is not None else float(centroid_y),
                    "seed_x": float(representative_row.get("seed_x", centroid_x)) if representative_row is not None else float(centroid_x),
                    "seed_score": float(representative_row.get("seed_score", float("nan"))) if representative_row is not None else float("nan"),
                    "support_count": int(
                        sum(int(row.get("support_count", 0)) for row in source_rows)
                    ) if source_rows else 0,
                    "support_level": float(
                        max(float(row.get("support_level", float("-inf"))) for row in source_rows)
                    ) if source_rows else float("nan"),
                }
            )
        label_map[frame_index] = remapped_label_map
        merged_metadata.append(remapped_rows)

    return label_map, merged_metadata


def _plot_wan21_t2v_candidate_region_viz(
    raw_map_fhw: torch.Tensor,
    label_map_fhw: torch.Tensor,
    save_file: str,
    title: str,
    attention_frame_indices: Sequence[int],
    video_frame_labels: Optional[Sequence[int]] = None,
    draw_candidate_contours: bool = False,
    raw_map_cmap: str = "magma",
):
    """Render two-row candidate-region visualization for one map.

    The first row shows the raw attention map with per-frame autoscaling.
    The second row shows the extracted binary candidate support.
    When `draw_candidate_contours=True`, each candidate label is additionally
    outlined with a green contour so that touching candidates remain visible.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not attention_frame_indices:
        return ""

    raw_map_fhw = torch.as_tensor(raw_map_fhw)
    label_map_fhw = torch.as_tensor(label_map_fhw)
    raw_map_cmap = str(raw_map_cmap).strip() or "magma"
    if video_frame_labels is None:
        video_frame_labels = [int(frame_index) for frame_index in attention_frame_indices]
    num_frames = len(attention_frame_indices)
    fig, axes = plt.subplots(2, num_frames, figsize=(max(3.0 * num_frames, 6.0), 5.6))
    if num_frames == 1:
        axes = axes.reshape(2, 1)

    for col_index, frame_index in enumerate(attention_frame_indices):
        display_frame_label = int(video_frame_labels[col_index]) if col_index < len(video_frame_labels) else int(frame_index)
        raw_frame = raw_map_fhw[int(frame_index)].detach().cpu().float()
        label_frame = label_map_fhw[int(frame_index)].detach().cpu().long()
        support_frame = (label_frame > 0).float()

        axis_raw = axes[0, col_index]
        axis_raw.imshow(raw_frame.numpy(), cmap=raw_map_cmap)
        axis_raw.set_title(f"frame={int(display_frame_label)}", fontsize=9)
        axis_raw.set_xticks([])
        axis_raw.set_yticks([])

        axis_mask = axes[1, col_index]
        axis_mask.imshow(support_frame.numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        if bool(draw_candidate_contours):
            candidate_count = int(label_frame.max().item())
            for candidate_index in range(1, candidate_count + 1):
                candidate_mask = (label_frame == int(candidate_index))
                if not bool(candidate_mask.any().item()):
                    continue
                axis_mask.contour(
                    candidate_mask.float().numpy(),
                    levels=[0.5],
                    colors=["#22c55e"],
                    linewidths=1.3,
                )
        axis_mask.set_xticks([])
        axis_mask.set_yticks([])
        axis_mask.set_xlabel(f"K={int(label_frame.max().item())}", fontsize=24, labelpad=10)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file


def _upsample_wan21_t2v_token_mask_to_vpred(
    token_mask_fhw: torch.Tensor,
    patch_size: Sequence[int],
) -> torch.Tensor:
    """Upsample token-grid mask to Wan `v_pred` spatial resolution."""
    patch_t, patch_h, patch_w = [int(x) for x in patch_size]
    out = token_mask_fhw.detach().float()
    if patch_t > 1:
        out = out.repeat_interleave(int(patch_t), dim=0)
    if patch_h > 1:
        out = out.repeat_interleave(int(patch_h), dim=1)
    if patch_w > 1:
        out = out.repeat_interleave(int(patch_w), dim=2)
    return out


def _subset_wan21_t2v_token_mask_by_frames(
    token_mask_fhw: torch.Tensor,
    selected_frame_indices: Sequence[int],
) -> torch.Tensor:
    """Keep only the requested latent-frame slices inside a token-grid mask."""
    frame_count = int(token_mask_fhw.shape[0])
    out = torch.zeros_like(token_mask_fhw, dtype=token_mask_fhw.dtype)
    if frame_count <= 0:
        return out
    kept = sorted(
        {
            int(frame_index)
            for frame_index in selected_frame_indices
            if 0 <= int(frame_index) < frame_count
        }
    )
    if not kept:
        return out
    out[kept] = token_mask_fhw[kept]
    return out


def _token_grid_mask_to_wan21_t2v_sequence_indices(
    token_mask_fhw: torch.Tensor,
) -> torch.Tensor:
    """Convert a `[F, H, W]` token-grid mask into Wan sequence-token indices."""
    flat_mask = token_mask_fhw.detach().reshape(-1) > 0
    if not bool(flat_mask.any().item()):
        return torch.zeros((0,), dtype=torch.long)
    return torch.nonzero(flat_mask, as_tuple=False).reshape(-1).long().cpu()


def _compute_wan21_t2v_attribution_patch_object_metrics(
    masked_clean_vpred: torch.Tensor,
    head_writes: torch.Tensor,
    head_writes_grad_obj: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute object-only attribution-patching metrics for all heads."""
    clean_obj = masked_clean_vpred.detach().float().reshape(-1)
    dot_obj_clean = float(torch.dot(clean_obj, clean_obj).item())
    head_writes = head_writes.detach().float()
    head_writes_grad_obj = head_writes_grad_obj.detach().float()
    dot_obj = -torch.einsum("bthd,bthd->h", head_writes_grad_obj, head_writes)
    ablate_dot_obj = torch.full_like(dot_obj, dot_obj_clean) - dot_obj
    return {
        "dot_obj": dot_obj.cpu(),
        "ablate_dot_obj": ablate_dot_obj.cpu(),
    }


def _safe_wan21_t2v_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return cosine similarity between two tensors flattened to vectors."""
    a_flat = a.detach().float().reshape(-1)
    b_flat = b.detach().float().reshape(-1)
    denom = float(a_flat.norm().item()) * float(b_flat.norm().item())
    if denom <= 1e-12:
        return 0.0
    return float(torch.dot(a_flat, b_flat).item() / denom)


def _safe_wan21_t2v_dot(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return flattened dot product between two tensors."""
    return float(torch.dot(a.detach().float().reshape(-1), b.detach().float().reshape(-1)).item())


def _compute_wan21_t2v_contribution_metrics(
    delta_vpred: torch.Tensor,
    clean_vpred: torch.Tensor,
    object_mask_fhw: Optional[torch.Tensor],
    ablated_vpred: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute full-field and object-masked contribution metrics."""
    delta_vpred = delta_vpred.detach().float()
    clean_vpred = clean_vpred.detach().float()
    ablated_vpred_local = None if ablated_vpred is None else ablated_vpred.detach().float()
    metrics = {
        "cos_full": abs(_safe_wan21_t2v_cosine(delta_vpred, clean_vpred)),
        "dot_full": abs(_safe_wan21_t2v_dot(delta_vpred, clean_vpred)),
    }
    if ablated_vpred_local is not None:
        metrics["ablate_cos_full"] = _safe_wan21_t2v_cosine(ablated_vpred_local, clean_vpred)
        metrics["ablate_dot_full"] = _safe_wan21_t2v_dot(ablated_vpred_local, clean_vpred)
    else:
        metrics["ablate_cos_full"] = float("nan")
        metrics["ablate_dot_full"] = float("nan")
    if object_mask_fhw is None:
        metrics["cos_obj"] = float("nan")
        metrics["dot_obj"] = float("nan")
        metrics["ablate_cos_obj"] = float("nan")
        metrics["ablate_dot_obj"] = float("nan")
        return metrics

    mask = object_mask_fhw.detach().float().unsqueeze(0)
    masked_delta = delta_vpred * mask
    masked_clean = clean_vpred * mask
    metrics["cos_obj"] = abs(_safe_wan21_t2v_cosine(masked_delta, masked_clean))
    metrics["dot_obj"] = abs(_safe_wan21_t2v_dot(masked_delta, masked_clean))
    if ablated_vpred_local is not None:
        masked_ablated = ablated_vpred_local * mask
        metrics["ablate_cos_obj"] = _safe_wan21_t2v_cosine(masked_ablated, masked_clean)
        metrics["ablate_dot_obj"] = _safe_wan21_t2v_dot(masked_ablated, masked_clean)
    else:
        metrics["ablate_cos_obj"] = float("nan")
        metrics["ablate_dot_obj"] = float("nan")
    return metrics


@dataclass
class Wan21T2VTrajectoryConsensusContributionState:
    """Runtime state for one targeted contribution run."""

    target_step: int
    target_layer: int
    target_head: int
    target_module: str
    target_branch: str
    ablate_position: str
    ablate_head: bool
    capture_selected_head_write: bool
    capture_all_head_writes: bool
    capture_suffix_payload: bool

    current_step: int = 0
    current_timestep_value: Optional[float] = None
    forward_call_index_in_step: int = 0
    captured_vpred: Optional[torch.Tensor] = None
    captured_grid_sizes: Optional[torch.Tensor] = None
    captured_head_e: Optional[torch.Tensor] = None
    captured_selected_head_write: Optional[torch.Tensor] = None
    captured_all_head_writes: Optional[torch.Tensor] = None
    captured_suffix_payload: Optional[Dict[str, Any]] = None

    def on_forward_start(self, t_tensor):
        """Track current diffusion step and branch index."""
        t_value = float(t_tensor.flatten()[0].item()) if t_tensor is not None else None
        if self.current_timestep_value is None or t_value != self.current_timestep_value:
            self.current_step += 1
            self.current_timestep_value = t_value
            self.forward_call_index_in_step = 0
        else:
            self.forward_call_index_in_step += 1

    @property
    def current_branch(self) -> str:
        return _resolve_wan21_t2v_branch_from_forward_call_index(self.forward_call_index_in_step)

    def is_target_forward(self) -> bool:
        """Whether the currently executing DiT forward is the target step and branch."""
        return (
            int(self.current_step) == int(self.target_step)
            and _wan21_t2v_branch_matches(self.target_branch, self.forward_call_index_in_step)
        )


@dataclass
class Wan21T2VTrajectoryConsensusAttributionState:
    """Runtime state for one attribution-patching clean forward."""

    target_step: int
    target_layer: int
    target_module: str
    target_branch: str
    attribution_position: str
    token_indices: Optional[torch.Tensor] = None

    current_step: int = 0
    current_timestep_value: Optional[float] = None
    forward_call_index_in_step: int = 0

    captured_clean_vpred: Optional[torch.Tensor] = None
    captured_head_writes: Optional[torch.Tensor] = None
    captured_head_writes_grad: Optional[torch.Tensor] = None

    def on_forward_start(self, t_tensor):
        """Track current diffusion step and branch index."""
        t_value = float(t_tensor.flatten()[0].item()) if t_tensor is not None else None
        if self.current_timestep_value is None or t_value != self.current_timestep_value:
            self.current_step += 1
            self.current_timestep_value = t_value
            self.forward_call_index_in_step = 0
        else:
            self.forward_call_index_in_step += 1

    def is_target_forward(self) -> bool:
        """Whether the currently executing DiT forward matches the target scope."""
        return (
            int(self.current_step) == int(self.target_step)
            and _wan21_t2v_branch_matches(self.target_branch, self.forward_call_index_in_step)
        )


@dataclass
class Wan21T2VTrajectoryConsensusGlobalAttributionState:
    """Runtime state for one global attribution-patching clean forward on a whole step."""

    target_step: int
    target_branch: str
    attribution_position: str
    selected_modules: Tuple[str, ...]
    token_indices: Optional[torch.Tensor] = None

    current_step: int = 0
    current_timestep_value: Optional[float] = None
    forward_call_index_in_step: int = 0

    captured_clean_vpred: Optional[torch.Tensor] = None
    captured_head_writes: Dict[Tuple[int, str], torch.Tensor] = field(default_factory=dict)
    captured_head_write_means: Dict[Tuple[int, str], torch.Tensor] = field(default_factory=dict)
    captured_head_writes_grad_obj: Dict[Tuple[int, str], torch.Tensor] = field(default_factory=dict)

    def on_forward_start(self, t_tensor):
        """Track current diffusion step and branch index."""
        t_value = float(t_tensor.flatten()[0].item()) if t_tensor is not None else None
        if self.current_timestep_value is None or t_value != self.current_timestep_value:
            self.current_step += 1
            self.current_timestep_value = t_value
            self.forward_call_index_in_step = 0
        else:
            self.forward_call_index_in_step += 1

    def is_target_forward(self) -> bool:
        """Whether the currently executing DiT forward matches the target step and branch."""
        return (
            int(self.current_step) == int(self.target_step)
            and _wan21_t2v_branch_matches(self.target_branch, self.forward_call_index_in_step)
        )


@dataclass
class Wan21T2VTrajectoryConsensusGlobalDirectProxyState:
    """Runtime state for one global direct-proxy clean forward on a whole step."""

    target_step: int
    target_branch: str
    selected_modules: Tuple[str, ...]

    current_step: int = 0
    current_timestep_value: Optional[float] = None
    forward_call_index_in_step: int = 0

    captured_clean_vpred: Optional[torch.Tensor] = None
    captured_head_e: Optional[torch.Tensor] = None
    captured_grid_sizes: Optional[torch.Tensor] = None
    captured_post_o_head_writes: Dict[Tuple[int, str], torch.Tensor] = field(default_factory=dict)

    def on_forward_start(self, t_tensor):
        """Track current diffusion step and branch index."""
        t_value = float(t_tensor.flatten()[0].item()) if t_tensor is not None else None
        if self.current_timestep_value is None or t_value != self.current_timestep_value:
            self.current_step += 1
            self.current_timestep_value = t_value
            self.forward_call_index_in_step = 0
        else:
            self.forward_call_index_in_step += 1

    def is_target_forward(self) -> bool:
        """Whether the currently executing DiT forward matches the target step and branch."""
        return (
            int(self.current_step) == int(self.target_step)
            and _wan21_t2v_branch_matches(self.target_branch, self.forward_call_index_in_step)
        )


class Wan21T2VTrajectoryConsensusContributionHandle:
    """Restore handle for one targeted contribution patch."""

    def __init__(self, target_model, state, original_forward, original_head_forward, original_unpatchify, original_block_forward):
        self.target_model = target_model
        self.state = state
        self.original_forward = original_forward
        self.original_head_forward = original_head_forward
        self.original_unpatchify = original_unpatchify
        self.original_block_forward = original_block_forward

    def restore(self):
        self.target_model.forward = self.original_forward
        self.target_model.head.forward = self.original_head_forward
        self.target_model.unpatchify = self.original_unpatchify
        self.target_model.blocks[int(self.state.target_layer)].forward = self.original_block_forward


class Wan21T2VTrajectoryConsensusAttributionHandle:
    """Restore handle for one targeted attribution patch."""

    def __init__(self, target_model, state, original_forward, original_block_forward, original_downstream_block_forwards):
        self.target_model = target_model
        self.state = state
        self.original_forward = original_forward
        self.original_block_forward = original_block_forward
        self.original_downstream_block_forwards = original_downstream_block_forwards

    def restore(self):
        self.target_model.forward = self.original_forward
        self.target_model.blocks[int(self.state.target_layer)].forward = self.original_block_forward
        for block_index, original_forward in self.original_downstream_block_forwards.items():
            self.target_model.blocks[int(block_index)].forward = original_forward


class Wan21T2VTrajectoryConsensusGlobalAttributionHandle:
    """Restore handle for one global attribution patch over multiple layers/modules."""

    def __init__(self, target_model, state, original_forward, original_block_forwards, original_downstream_block_forwards):
        self.target_model = target_model
        self.state = state
        self.original_forward = original_forward
        self.original_block_forwards = original_block_forwards
        self.original_downstream_block_forwards = original_downstream_block_forwards

    def restore(self):
        self.target_model.forward = self.original_forward
        for block_index, original_forward in self.original_block_forwards.items():
            self.target_model.blocks[int(block_index)].forward = original_forward
        for block_index, original_forward in self.original_downstream_block_forwards.items():
            self.target_model.blocks[int(block_index)].forward = original_forward


class Wan21T2VTrajectoryConsensusGlobalDirectProxyHandle:
    """Restore handle for one global direct-proxy patch over multiple layers/modules."""

    def __init__(
        self,
        target_model,
        state,
        original_forward,
        original_head_forward,
        original_unpatchify,
        original_block_forwards,
    ):
        self.target_model = target_model
        self.state = state
        self.original_forward = original_forward
        self.original_head_forward = original_head_forward
        self.original_unpatchify = original_unpatchify
        self.original_block_forwards = original_block_forwards

    def restore(self):
        self.target_model.forward = self.original_forward
        self.target_model.head.forward = self.original_head_forward
        self.target_model.unpatchify = self.original_unpatchify
        for block_index, original_forward in self.original_block_forwards.items():
            self.target_model.blocks[int(block_index)].forward = original_forward


def _install_wan21_t2v_trajectory_consensus_contribution_patch(
    model,
    target_step: int,
    target_layer: int,
    target_head: int,
    target_module: str,
    target_branch: str,
    ablate_position: str,
    ablate_head: bool,
    capture_selected_head_write: bool,
    capture_all_head_writes: bool = False,
    capture_suffix_payload: bool = False,
) -> Wan21T2VTrajectoryConsensusContributionHandle:
    """Install a targeted runtime patch for one contribution run."""
    from projects.Wan2_1.wan.modules.attention import flash_attention
    from projects.Wan2_1.wan.modules.model import T5_CONTEXT_TOKEN_NUMBER, rope_apply

    target = _unwrap_wan21_t2v_dit_model_for_runtime_patch(model)
    if not hasattr(target, "blocks"):
        raise RuntimeError("Invalid DiT model: missing blocks.")
    if int(target_layer) < 0 or int(target_layer) >= len(target.blocks):
        raise ValueError(f"target_layer out of range: {target_layer}, num_layers={len(target.blocks)}")

    block = target.blocks[int(target_layer)]
    attn_module = block.self_attn if str(target_module) == "self" else block.cross_attn
    if int(target_head) < 0 or int(target_head) >= int(attn_module.num_heads):
        raise ValueError(
            f"target_head out of range for module={target_module}, layer={target_layer}: "
            f"head={target_head}, num_heads={attn_module.num_heads}"
        )

    state = Wan21T2VTrajectoryConsensusContributionState(
        target_step=int(target_step),
        target_layer=int(target_layer),
        target_head=int(target_head),
        target_module=str(target_module),
        target_branch=str(target_branch),
        ablate_position=str(ablate_position).strip().lower(),
        ablate_head=bool(ablate_head),
        capture_selected_head_write=bool(capture_selected_head_write),
        capture_all_head_writes=bool(capture_all_head_writes),
        capture_suffix_payload=bool(capture_suffix_payload),
    )

    original_forward = target.forward
    original_head_forward = target.head.forward
    original_unpatchify = target.unpatchify
    original_block_forward = block.forward

    def patched_dit_forward(this, *args, **kwargs):
        t = kwargs.get("t", None)
        if t is None and len(args) > 1:
            t = args[1]
        state.on_forward_start(t)
        result = original_forward(*args, **kwargs)
        if state.is_target_forward():
            if isinstance(result, list) and result:
                state.captured_vpred = result[0].detach().float().cpu()
            raise Wan21T2VEarlyStopRequested(
                completed_step=int(state.current_step),
                requested_last_step=int(state.target_step),
            )
        return result

    def patched_head_forward(self, x, e):
        if state.is_target_forward():
            state.captured_head_e = e.detach().float().cpu()
        return original_head_forward(x, e)

    def patched_unpatchify(self, x, grid_sizes):
        if state.is_target_forward():
            state.captured_grid_sizes = grid_sizes.detach().cpu()
        return original_unpatchify(x, grid_sizes)

    def _project_selected_head_write(z_selected: torch.Tensor, linear_module) -> torch.Tensor:
        start = int(state.target_head) * int(linear_module.head_dim)
        end = start + int(linear_module.head_dim)
        weight_slice = linear_module.o.weight[:, start:end].transpose(0, 1).contiguous()
        return torch.matmul(z_selected, weight_slice)

    def _project_all_head_writes(
        z_bthd: torch.Tensor,
        linear_module,
        post_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_heads = int(linear_module.num_heads)
        head_dim = int(linear_module.head_dim)
        weight = linear_module.o.weight.view(
            linear_module.o.out_features,
            num_heads,
            head_dim,
        ).permute(1, 2, 0).contiguous()
        head_writes = torch.einsum("bthd,hdo->btho", z_bthd, weight)
        if post_scale is not None:
            head_writes = head_writes * post_scale.unsqueeze(2)
        return head_writes

    def _run_manual_cross_attn(cross_attn_module, x_input, context, context_lens):
        b, n, d = x_input.size(0), cross_attn_module.num_heads, cross_attn_module.head_dim
        q = cross_attn_module.norm_q(cross_attn_module.q(x_input)).view(b, -1, n, d)
        if hasattr(cross_attn_module, "k_img") and hasattr(cross_attn_module, "v_img"):
            image_context_length = context.shape[1] - T5_CONTEXT_TOKEN_NUMBER
            context_img = context[:, :image_context_length]
            context_txt = context[:, image_context_length:]

            k = cross_attn_module.norm_k(cross_attn_module.k(context_txt)).view(b, -1, n, d)
            v = cross_attn_module.v(context_txt).view(b, -1, n, d)
            k_img = cross_attn_module.norm_k_img(cross_attn_module.k_img(context_img)).view(b, -1, n, d)
            v_img = cross_attn_module.v_img(context_img).view(b, -1, n, d)

            head_output = flash_attention(q, k, v, k_lens=context_lens)
            head_output = head_output + flash_attention(q, k_img, v_img, k_lens=None)
        else:
            k = cross_attn_module.norm_k(cross_attn_module.k(context)).view(b, -1, n, d)
            v = cross_attn_module.v(context).view(b, -1, n, d)
            head_output = flash_attention(q, k, v, k_lens=context_lens)
        return head_output

    def patched_block_forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        if not state.is_target_forward():
            return original_block_forward(x, e, seq_lens, grid_sizes, freqs, context, context_lens)

        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            modulation = (self.modulation + e).chunk(6, dim=1)

        if str(state.target_module) == "self":
            sa_input = self.norm1(x).float() * (1 + modulation[1]) + modulation[0]
            batch_size, seq_len = sa_input.shape[:2]
            num_heads = self.self_attn.num_heads
            head_dim = self.self_attn.head_dim
            q = self.self_attn.norm_q(self.self_attn.q(sa_input)).view(batch_size, seq_len, num_heads, head_dim)
            k = self.self_attn.norm_k(self.self_attn.k(sa_input)).view(batch_size, seq_len, num_heads, head_dim)
            v = self.self_attn.v(sa_input).view(batch_size, seq_len, num_heads, head_dim)
            head_output = flash_attention(
                q=rope_apply(q, grid_sizes, freqs),
                k=rope_apply(k, grid_sizes, freqs),
                v=v,
                k_lens=seq_lens,
                window_size=self.self_attn.window_size,
            )

            if state.capture_selected_head_write:
                selected_head_output = head_output[:, :, int(state.target_head), :]
                selected_write = _project_selected_head_write(selected_head_output, self.self_attn)
                with amp.autocast(dtype=torch.float32):
                    selected_write = selected_write * modulation[2]
                state.captured_selected_head_write = selected_write.detach().float().cpu()

            all_head_writes = None
            need_all_head_writes = (
                state.capture_all_head_writes
                or state.capture_suffix_payload
                or str(state.ablate_position) == "post_o"
            )
            if need_all_head_writes:
                all_head_writes = _project_all_head_writes(
                    head_output,
                    self.self_attn,
                    post_scale=modulation[2],
                )
                if state.capture_all_head_writes:
                    state.captured_all_head_writes = all_head_writes.detach().float().cpu()

            if state.capture_suffix_payload:
                state.captured_suffix_payload = {
                    "target_layer": int(state.target_layer),
                    "x_before": x.detach().float().cpu(),
                    "e": e.detach().float().cpu(),
                    "head_e": e[:, 0, :].detach().float().cpu(),
                    "seq_lens": seq_lens.detach().cpu(),
                    "grid_sizes": grid_sizes.detach().cpu(),
                    "freqs": freqs.detach().float().cpu(),
                    "context": context.detach().float().cpu(),
                    "context_lens": None if context_lens is None else context_lens.detach().cpu(),
                    "modulation2": modulation[2].detach().float().cpu(),
                    "modulation3": modulation[3].detach().float().cpu(),
                    "modulation4": modulation[4].detach().float().cpu(),
                    "modulation5": modulation[5].detach().float().cpu(),
                    "norm3_module": self.norm3,
                    "cross_attn_module": self.cross_attn,
                    "norm2_module": self.norm2,
                    "ffn_module": self.ffn,
                    "all_head_writes": None if all_head_writes is None else all_head_writes.detach().float().cpu(),
                }

            if str(state.ablate_position) == "post_o":
                if all_head_writes is None:
                    all_head_writes = _project_all_head_writes(
                        head_output,
                        self.self_attn,
                        post_scale=modulation[2],
                    )
                if state.ablate_head:
                    all_head_writes = all_head_writes.clone()
                    all_head_writes[:, :, int(state.target_head), :] = 0
                sa_output = all_head_writes.sum(dim=2)
                with amp.autocast(dtype=torch.float32):
                    x = x + sa_output
            else:
                if state.ablate_head:
                    head_output = head_output.clone()
                    head_output[:, :, int(state.target_head), :] = 0
                sa_output = self.self_attn.o(head_output.flatten(2))
                with amp.autocast(dtype=torch.float32):
                    x = x + sa_output * modulation[2]
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x).float() * (1 + modulation[4]) + modulation[3])
            with amp.autocast(dtype=torch.float32):
                x = x + y * modulation[5]
            return x

        sa_output = self.self_attn(
            self.norm1(x).float() * (1 + modulation[1]) + modulation[0],
            seq_lens,
            grid_sizes,
            freqs,
        )
        with amp.autocast(dtype=torch.float32):
            x = x + sa_output * modulation[2]

        cross_input = self.norm3(x)
        head_output = _run_manual_cross_attn(self.cross_attn, cross_input, context, context_lens)

        if state.capture_selected_head_write:
            selected_head_output = head_output[:, :, int(state.target_head), :]
            selected_write = _project_selected_head_write(selected_head_output, self.cross_attn)
            state.captured_selected_head_write = selected_write.detach().float().cpu()

        all_head_writes = None
        need_all_head_writes = (
            state.capture_all_head_writes
            or state.capture_suffix_payload
            or str(state.ablate_position) == "post_o"
        )
        if need_all_head_writes:
            all_head_writes = _project_all_head_writes(head_output, self.cross_attn, post_scale=None)
            if state.capture_all_head_writes:
                state.captured_all_head_writes = all_head_writes.detach().float().cpu()

        if state.capture_suffix_payload:
            state.captured_suffix_payload = {
                "target_layer": int(state.target_layer),
                "x_before_cross": x.detach().float().cpu(),
                "e": e.detach().float().cpu(),
                "head_e": e[:, 0, :].detach().float().cpu(),
                "seq_lens": seq_lens.detach().cpu(),
                "grid_sizes": grid_sizes.detach().cpu(),
                "freqs": freqs.detach().float().cpu(),
                "context": context.detach().float().cpu(),
                "context_lens": None if context_lens is None else context_lens.detach().cpu(),
                "modulation3": modulation[3].detach().float().cpu(),
                "modulation4": modulation[4].detach().float().cpu(),
                "modulation5": modulation[5].detach().float().cpu(),
                "norm2_module": self.norm2,
                "ffn_module": self.ffn,
                "all_head_writes": None if all_head_writes is None else all_head_writes.detach().float().cpu(),
            }

        if str(state.ablate_position) == "post_o":
            if all_head_writes is None:
                all_head_writes = _project_all_head_writes(head_output, self.cross_attn, post_scale=None)
            if state.ablate_head:
                all_head_writes = all_head_writes.clone()
                all_head_writes[:, :, int(state.target_head), :] = 0
            cross_output = all_head_writes.sum(dim=2)
        else:
            if state.ablate_head:
                head_output = head_output.clone()
                head_output[:, :, int(state.target_head), :] = 0
            cross_output = self.cross_attn.o(head_output.flatten(2))
        x = x + cross_output
        y = self.ffn(self.norm2(x).float() * (1 + modulation[4]) + modulation[3])
        with amp.autocast(dtype=torch.float32):
            x = x + y * modulation[5]
        return x

    target.forward = MethodType(patched_dit_forward, target)
    target.head.forward = MethodType(patched_head_forward, target.head)
    target.unpatchify = MethodType(patched_unpatchify, target)
    target.blocks[int(target_layer)].forward = MethodType(patched_block_forward, target.blocks[int(target_layer)])

    return Wan21T2VTrajectoryConsensusContributionHandle(
        target_model=target,
        state=state,
        original_forward=original_forward,
        original_head_forward=original_head_forward,
        original_unpatchify=original_unpatchify,
        original_block_forward=original_block_forward,
    )


def _install_wan21_t2v_trajectory_consensus_attribution_patch(
    model,
    target_step: int,
    target_layer: int,
    target_module: str,
    target_branch: str,
    attribution_position: str,
    token_indices: Optional[torch.Tensor],
    use_gradient_checkpointing: bool,
) -> Wan21T2VTrajectoryConsensusAttributionHandle:
    """Install the clean attribution patch for one `(step, layer, module, branch)` scope.

    This patch follows standard attribution patching:
    - run one clean forward for the selected step and branch;
    - capture all heads in the target module as residual-stream writes
      `U[B, T, H, D]`;
    - retain gradients on that clean activation tensor;
    - continue the same clean forward to the final `v_pred`.
    """
    from projects.Wan2_1.wan.modules.attention import flash_attention
    from projects.Wan2_1.wan.modules.model import T5_CONTEXT_TOKEN_NUMBER, rope_apply
    from torch.utils.checkpoint import checkpoint

    target = _unwrap_wan21_t2v_dit_model_for_runtime_patch(model)
    if not hasattr(target, "blocks"):
        raise RuntimeError("Invalid DiT model: missing blocks.")
    if int(target_layer) < 0 or int(target_layer) >= len(target.blocks):
        raise ValueError(f"target_layer out of range: {target_layer}, num_layers={len(target.blocks)}")

    block = target.blocks[int(target_layer)]
    state = Wan21T2VTrajectoryConsensusAttributionState(
        target_step=int(target_step),
        target_layer=int(target_layer),
        target_module=str(target_module),
        target_branch=str(target_branch),
        attribution_position=str(attribution_position).strip().lower(),
        token_indices=None if token_indices is None else token_indices.detach().long().cpu(),
    )

    original_forward = target.forward
    original_block_forward = block.forward
    original_downstream_block_forwards: Dict[int, Any] = {}

    def patched_dit_forward(this, *args, **kwargs):
        t = kwargs.get("t", None)
        if t is None and len(args) > 1:
            t = args[1]
        state.on_forward_start(t)
        result = original_forward(*args, **kwargs)
        if state.is_target_forward():
            if isinstance(result, list) and result:
                state.captured_clean_vpred = result[0]
        return result

    def _project_all_head_writes(z_bthd: torch.Tensor, linear_module, post_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Project per-head values into per-head residual writes.

        Args:
            z_bthd: `[B, T, H, d_head]`
            post_scale: optional multiplicative scale broadcastable to `[B, T, 1, D]`

        Returns:
            `[B, T, H, D]`
        """
        num_heads = int(linear_module.num_heads)
        head_dim = int(linear_module.head_dim)
        weight = linear_module.o.weight.view(linear_module.o.out_features, num_heads, head_dim).permute(1, 2, 0).contiguous()
        head_writes = torch.einsum("bthd,hdo->btho", z_bthd, weight)
        if post_scale is not None:
            head_writes = head_writes * post_scale.unsqueeze(2)
        return head_writes

    def _run_manual_cross_attn(cross_attn_module, x_input, context, context_lens):
        b, n, d = x_input.size(0), cross_attn_module.num_heads, cross_attn_module.head_dim
        q = cross_attn_module.norm_q(cross_attn_module.q(x_input)).view(b, -1, n, d)
        if hasattr(cross_attn_module, "k_img") and hasattr(cross_attn_module, "v_img"):
            image_context_length = context.shape[1] - T5_CONTEXT_TOKEN_NUMBER
            context_img = context[:, :image_context_length]
            context_txt = context[:, image_context_length:]

            k = cross_attn_module.norm_k(cross_attn_module.k(context_txt)).view(b, -1, n, d)
            v = cross_attn_module.v(context_txt).view(b, -1, n, d)
            k_img = cross_attn_module.norm_k_img(cross_attn_module.k_img(context_img)).view(b, -1, n, d)
            v_img = cross_attn_module.v_img(context_img).view(b, -1, n, d)

            head_output = flash_attention(q, k, v, k_lens=context_lens)
            head_output = head_output + flash_attention(q, k_img, v_img, k_lens=None)
        else:
            k = cross_attn_module.norm_k(cross_attn_module.k(context)).view(b, -1, n, d)
            v = cross_attn_module.v(context).view(b, -1, n, d)
            head_output = flash_attention(q, k, v, k_lens=context_lens)
        return head_output

    def _capture_grad(grad: torch.Tensor):
        state.captured_head_writes_grad = grad.detach().float().cpu()

    def _build_gradient_subset_activation(activation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if state.token_indices is None or int(state.token_indices.numel()) <= 0:
            activation_leaf = activation.detach().requires_grad_(True)
            return activation_leaf, activation_leaf
        token_indices_device = state.token_indices.to(device=activation.device)
        activation_subset = activation.index_select(dim=1, index=token_indices_device).detach().requires_grad_(True)
        activation_full = activation.detach().clone()
        activation_full.index_copy_(1, token_indices_device, activation_subset)
        return activation_full, activation_subset

    def patched_block_forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        if not state.is_target_forward():
            return original_block_forward(x, e, seq_lens, grid_sizes, freqs, context, context_lens)

        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            modulation = (self.modulation + e).chunk(6, dim=1)

        if str(state.target_module).strip().lower() == "self":
            sa_input = self.norm1(x).float() * (1 + modulation[1]) + modulation[0]
            batch_size, seq_len = sa_input.shape[:2]
            num_heads = self.self_attn.num_heads
            head_dim = self.self_attn.head_dim
            q = self.self_attn.norm_q(self.self_attn.q(sa_input)).view(batch_size, seq_len, num_heads, head_dim)
            k = self.self_attn.norm_k(self.self_attn.k(sa_input)).view(batch_size, seq_len, num_heads, head_dim)
            v = self.self_attn.v(sa_input).view(batch_size, seq_len, num_heads, head_dim)
            z = flash_attention(
                q=rope_apply(q, grid_sizes, freqs),
                k=rope_apply(k, grid_sizes, freqs),
                v=v,
                k_lens=seq_lens,
                window_size=self.self_attn.window_size,
            )
            if str(state.attribution_position) == "post_o":
                full_activation, tracked_activation = _build_gradient_subset_activation(
                    _project_all_head_writes(z, self.self_attn, post_scale=modulation[2])
                )
                state.captured_head_writes = tracked_activation.detach().float().cpu()
                tracked_activation.register_hook(_capture_grad)
                sa_output = full_activation.sum(dim=2)
                with amp.autocast(dtype=torch.float32):
                    x = x + sa_output
            else:
                full_activation, tracked_activation = _build_gradient_subset_activation(z)
                state.captured_head_writes = tracked_activation.detach().float().cpu()
                tracked_activation.register_hook(_capture_grad)
                sa_output = self.self_attn.o(full_activation.flatten(2))
                with amp.autocast(dtype=torch.float32):
                    x = x + sa_output * modulation[2]
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x).float() * (1 + modulation[4]) + modulation[3])
            with amp.autocast(dtype=torch.float32):
                x = x + y * modulation[5]
            return x

        sa_output = self.self_attn(
            self.norm1(x).float() * (1 + modulation[1]) + modulation[0],
            seq_lens,
            grid_sizes,
            freqs,
        )
        with amp.autocast(dtype=torch.float32):
            x = x + sa_output * modulation[2]

        cross_input = self.norm3(x)
        z = _run_manual_cross_attn(self.cross_attn, cross_input, context, context_lens)
        if str(state.attribution_position) == "post_o":
            full_activation, tracked_activation = _build_gradient_subset_activation(
                _project_all_head_writes(z, self.cross_attn, post_scale=None)
            )
            state.captured_head_writes = tracked_activation.detach().float().cpu()
            tracked_activation.register_hook(_capture_grad)
            cross_output = full_activation.sum(dim=2)
        else:
            full_activation, tracked_activation = _build_gradient_subset_activation(z)
            state.captured_head_writes = tracked_activation.detach().float().cpu()
            tracked_activation.register_hook(_capture_grad)
            cross_output = self.cross_attn.o(full_activation.flatten(2))
        x = x + cross_output
        y = self.ffn(self.norm2(x).float() * (1 + modulation[4]) + modulation[3])
        with amp.autocast(dtype=torch.float32):
            x = x + y * modulation[5]
        return x

    if bool(use_gradient_checkpointing):
        for downstream_block_index in range(int(target_layer) + 1, len(target.blocks)):
            downstream_block = target.blocks[int(downstream_block_index)]
            original_downstream_block_forwards[int(downstream_block_index)] = downstream_block.forward

            def _make_checkpointed_block_forward(original_block_forward_fn):
                def checkpointed_block_forward(
                    self,
                    x,
                    e,
                    seq_lens,
                    grid_sizes,
                    freqs,
                    context,
                    context_lens,
                ):
                    def block_fn(
                        x_tensor,
                        e_tensor,
                        seq_lens_tensor,
                        grid_sizes_tensor,
                        freqs_tensor,
                        context_tensor,
                    ):
                        return original_block_forward_fn(
                            x_tensor,
                            e_tensor,
                            seq_lens_tensor,
                            grid_sizes_tensor,
                            freqs_tensor,
                            context_tensor,
                            context_lens,
                        )
                    return checkpoint(
                        block_fn,
                        x,
                        e,
                        seq_lens,
                        grid_sizes,
                        freqs,
                        context,
                        use_reentrant=False,
                    )
                return checkpointed_block_forward

            downstream_block.forward = MethodType(
                _make_checkpointed_block_forward(original_downstream_block_forwards[int(downstream_block_index)]),
                downstream_block,
            )

    target.forward = MethodType(patched_dit_forward, target)
    target.blocks[int(target_layer)].forward = MethodType(patched_block_forward, target.blocks[int(target_layer)])

    return Wan21T2VTrajectoryConsensusAttributionHandle(
        target_model=target,
        state=state,
        original_forward=original_forward,
        original_block_forward=original_block_forward,
        original_downstream_block_forwards=original_downstream_block_forwards,
    )


def _run_wan21_t2v_trajectory_consensus_contribution_forward(
    pipeline,
    prompt: str,
    size: Tuple[int, int],
    frame_num: int,
    shift: float,
    sample_solver: str,
    sampling_steps: int,
    guide_scale: float,
    seed: int,
    offload_model: bool,
    target_step: int,
    target_layer: int,
    target_head: int,
    target_module: str,
    target_branch: str,
    ablate_position: str,
    ablate_head: bool,
    capture_selected_head_write: bool,
    capture_all_head_writes: bool = False,
    capture_suffix_payload: bool = False,
) -> Wan21T2VTrajectoryConsensusContributionState:
    """Run one early-stopped contribution forward pass and return captured state."""
    handle = _install_wan21_t2v_trajectory_consensus_contribution_patch(
        model=pipeline.model,
        target_step=int(target_step),
        target_layer=int(target_layer),
        target_head=int(target_head),
        target_module=str(target_module),
        target_branch=str(target_branch),
        ablate_position=str(ablate_position),
        ablate_head=bool(ablate_head),
        capture_selected_head_write=bool(capture_selected_head_write),
        capture_all_head_writes=bool(capture_all_head_writes),
        capture_suffix_payload=bool(capture_suffix_payload),
    )
    try:
        _generate_wan21_t2v_video(
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
    except Wan21T2VEarlyStopRequested:
        pass
    finally:
        handle.restore()
    return handle.state


def _run_wan21_t2v_local_clean_vpred(
    pipeline,
    latent_input: torch.Tensor,
    timestep_value: torch.Tensor,
    seq_len: int,
    context: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Run one local denoiser forward from a cached step latent and return clean `v_pred`."""
    target_model = _unwrap_wan21_t2v_dit_model_for_runtime_patch(pipeline.model)
    target_model.to(pipeline.device)
    with amp.autocast(dtype=pipeline.param_dtype), torch.no_grad():
        latent = latent_input.to(device=pipeline.device, dtype=torch.float32).detach()
        timestep = torch.stack([timestep_value.to(device=pipeline.device)])
        context_device = [u.to(device=pipeline.device, dtype=torch.float32) for u in context]
        result = pipeline.model([latent], t=timestep, context=context_device, seq_len=int(seq_len))
    if not isinstance(result, list) or not result:
        raise RuntimeError("Unexpected clean local forward output format.")
    return result[0].detach().float().cpu()


def _run_wan21_t2v_local_contribution_forward(
    pipeline,
    latent_input: torch.Tensor,
    timestep_value: torch.Tensor,
    seq_len: int,
    context: Sequence[torch.Tensor],
    branch: str,
    target_layer: int,
    target_head: int,
    target_module: str,
    ablate_position: str,
    ablate_head: bool,
    capture_selected_head_write: bool,
    capture_all_head_writes: bool = False,
    capture_suffix_payload: bool = False,
) -> Wan21T2VTrajectoryConsensusContributionState:
    """Run one patched local denoiser forward from a cached step latent.

    The branch-specific behavior is already encoded by the supplied `context`.
    Therefore this helper always installs the patch with synthetic branch
    `cond` and synthetic step `1`, then runs exactly one local DiT forward.
    """
    branch_name = str(branch).strip().lower()
    if branch_name not in {"cond", "uncond"}:
        raise ValueError(f"Unsupported branch for local contribution forward: {branch}")

    target_model = _unwrap_wan21_t2v_dit_model_for_runtime_patch(pipeline.model)
    target_model.to(pipeline.device)
    handle = _install_wan21_t2v_trajectory_consensus_contribution_patch(
        model=pipeline.model,
        target_step=1,
        target_layer=int(target_layer),
        target_head=int(target_head),
        target_module=str(target_module),
        target_branch="cond",
        ablate_position=str(ablate_position),
        ablate_head=bool(ablate_head),
        capture_selected_head_write=bool(capture_selected_head_write),
        capture_all_head_writes=bool(capture_all_head_writes),
        capture_suffix_payload=bool(capture_suffix_payload),
    )
    try:
        with amp.autocast(dtype=pipeline.param_dtype), torch.no_grad():
            latent = latent_input.to(device=pipeline.device, dtype=torch.float32).detach()
            timestep = torch.stack([timestep_value.to(device=pipeline.device)])
            context_device = [u.to(device=pipeline.device, dtype=torch.float32) for u in context]
            result = pipeline.model([latent], t=timestep, context=context_device, seq_len=int(seq_len))
            if handle.state.captured_vpred is None and isinstance(result, list) and result:
                handle.state.captured_vpred = result[0].detach().float().cpu()
    except Wan21T2VEarlyStopRequested:
        pass
    finally:
        handle.restore()
    return handle.state


def _project_wan21_t2v_all_head_writes_to_vpred(
    pipeline,
    per_head_writes: torch.Tensor,
    head_e: torch.Tensor,
    grid_sizes: torch.Tensor,
) -> torch.Tensor:
    """Project all per-head residual writes in one layer/module to per-head `v_pred'`.

    Args:
        per_head_writes: `[B, T, H, D]`
        head_e: clean run final-head modulation input, shape `[B, D]`
        grid_sizes: clean run grid sizes used by `unpatchify`

    Returns:
        Tensor of shape `[H, C, F, H_out, W_out]`.
    """
    target = _unwrap_wan21_t2v_dit_model_for_runtime_patch(pipeline.model)
    device = target.head.head.weight.device
    writes = per_head_writes.to(device=device, dtype=torch.float32)
    head_e_device = head_e.to(device=device, dtype=torch.float32)
    grid_sizes_device = grid_sizes.to(device=device)
    num_heads = int(writes.shape[2])
    outputs: List[torch.Tensor] = []
    with torch.no_grad():
        for head_index in range(num_heads):
            projected = target.head(writes[:, :, head_index, :], head_e_device)
            vpred = target.unpatchify(projected, grid_sizes_device)
            if not isinstance(vpred, list) or not vpred:
                raise RuntimeError("Unexpected direct projection output format.")
            outputs.append(vpred[0].detach().float().cpu())
    return torch.stack(outputs, dim=0)


def _compute_wan21_t2v_global_direct_proxy_metrics(
    pipeline,
    clean_vpred: torch.Tensor,
    post_o_head_writes: Dict[Tuple[int, str], torch.Tensor],
    head_e: torch.Tensor,
    grid_sizes: torch.Tensor,
    object_mask_fhw: Optional[torch.Tensor],
    chunk_num_heads: int = 8,
) -> Dict[Tuple[int, str], Dict[str, torch.Tensor]]:
    """Compute direct-proxy metrics for all selected heads with chunked final readout.

    This function treats all selected heads in one diffusion step as one global
    head list, then evaluates the final readout in head chunks to control GPU
    memory. No extra denoiser forward is run here.
    """
    target = _unwrap_wan21_t2v_dit_model_for_runtime_patch(pipeline.model)
    device = target.head.head.weight.device
    clean_vpred_cpu = clean_vpred.detach().float().cpu()
    metric_rows: List[Tuple[Tuple[int, str], int]] = []
    for key, tensor in sorted(post_o_head_writes.items()):
        num_heads = int(tensor.shape[2])
        for head_index in range(num_heads):
            metric_rows.append((key, int(head_index)))
    if not metric_rows:
        return {}

    result: Dict[Tuple[int, str], Dict[str, List[float]]] = {}
    repeated_head_e_base = head_e.detach().float().cpu()
    repeated_grid_sizes_base = grid_sizes.detach().cpu()

    for chunk_start in range(0, len(metric_rows), max(1, int(chunk_num_heads))):
        chunk_specs = metric_rows[chunk_start: chunk_start + max(1, int(chunk_num_heads))]
        chunk_writes = torch.stack(
            [
                post_o_head_writes[key][:, :, head_index, :].squeeze(0).detach().float().cpu()
                for key, head_index in chunk_specs
            ],
            dim=0,
        )
        if repeated_head_e_base.ndim == 1:
            chunk_head_e = repeated_head_e_base.unsqueeze(0).repeat(int(chunk_writes.shape[0]), 1)
        elif repeated_head_e_base.ndim == 2:
            if int(repeated_head_e_base.shape[0]) == 1:
                chunk_head_e = repeated_head_e_base.repeat(int(chunk_writes.shape[0]), 1)
            elif int(repeated_head_e_base.shape[0]) == int(chunk_writes.shape[0]):
                chunk_head_e = repeated_head_e_base
            else:
                raise RuntimeError(
                    "Unexpected head_e batch shape for direct proxy: "
                    f"{tuple(repeated_head_e_base.shape)} vs chunk={int(chunk_writes.shape[0])}"
                )
        else:
            raise RuntimeError(
                f"Unexpected head_e ndim for direct proxy: {int(repeated_head_e_base.ndim)}"
            )
        chunk_grid_sizes = repeated_grid_sizes_base.repeat(int(chunk_writes.shape[0]), 1)
        with torch.no_grad():
            projected = target.head(
                chunk_writes.to(device=device, dtype=torch.float32),
                chunk_head_e.to(device=device, dtype=torch.float32),
            )
            vpred_list = target.unpatchify(projected, chunk_grid_sizes.to(device=device))
        if not isinstance(vpred_list, list) or len(vpred_list) != int(chunk_writes.shape[0]):
            raise RuntimeError("Unexpected direct projection batch output format.")
        for (key, head_index), proxy_vpred in zip(chunk_specs, vpred_list):
            metrics = _compute_wan21_t2v_contribution_metrics(
                delta_vpred=proxy_vpred.detach().float().cpu(),
                clean_vpred=clean_vpred_cpu,
                object_mask_fhw=object_mask_fhw,
            )
            bucket = result.setdefault(
                key,
                {
                    "proj_cos_full": [],
                    "proj_dot_full": [],
                    "proj_cos_obj": [],
                    "proj_dot_obj": [],
                },
            )
            bucket["proj_cos_full"].append(float(metrics["cos_full"]))
            bucket["proj_dot_full"].append(float(metrics["dot_full"]))
            bucket["proj_cos_obj"].append(float(metrics["cos_obj"]))
            bucket["proj_dot_obj"].append(float(metrics["dot_obj"]))

    return {
        key: {metric_name: torch.tensor(values, dtype=torch.float32) for metric_name, values in metric_dict.items()}
        for key, metric_dict in result.items()
    }


def _run_wan21_t2v_to_target_step_latent(
    pipeline,
    prompt: str,
    size: Tuple[int, int],
    frame_num: int,
    shift: float,
    sample_solver: str,
    sampling_steps: int,
    guide_scale: float,
    seed: int,
    target_step: int,
    offload_model: bool,
) -> Dict[str, Any]:
    """Backward-compatible wrapper around the multi-step latent cache builder."""
    cache = _run_wan21_t2v_collect_target_step_latents(
        pipeline=pipeline,
        prompt=prompt,
        size=size,
        frame_num=frame_num,
        shift=shift,
        sample_solver=sample_solver,
        sampling_steps=sampling_steps,
        guide_scale=guide_scale,
        seed=seed,
        target_steps=[int(target_step)],
        offload_model=offload_model,
    )
    if int(target_step) not in cache:
        raise RuntimeError(f"Failed to capture target-step latent for step={target_step}")
    return cache[int(target_step)]


def _run_wan21_t2v_collect_target_step_latents(
    pipeline,
    prompt: str,
    size: Tuple[int, int],
    frame_num: int,
    shift: float,
    sample_solver: str,
    sampling_steps: int,
    guide_scale: float,
    seed: int,
    target_steps: Sequence[int],
    offload_model: bool,
) -> Dict[int, Dict[str, Any]]:
    """Run one monotonic diffusion scan and cache the input latent for target steps."""
    requested_steps = sorted({int(step) for step in target_steps})
    if not requested_steps:
        return {}
    if int(requested_steps[0]) < 1:
        raise ValueError(f"target_steps must be >= 1, got {requested_steps}")

    target_shape = (
        pipeline.vae.model.z_dim,
        (int(frame_num) - 1) // pipeline.vae_stride[0] + 1,
        int(size[1]) // pipeline.vae_stride[1],
        int(size[0]) // pipeline.vae_stride[2],
    )
    seq_len = math.ceil(
        (target_shape[2] * target_shape[3])
        / (pipeline.patch_size[1] * pipeline.patch_size[2])
        * target_shape[1]
        / pipeline.sp_size
    ) * pipeline.sp_size

    n_prompt = pipeline.sample_neg_prompt
    seed = int(seed) if int(seed) >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=pipeline.device)
    seed_g.manual_seed(seed)

    context = _encode_wan21_t2v_text_context_once(
        pipeline=pipeline,
        text=prompt,
        offload_model=bool(offload_model),
    )
    context_null = _encode_wan21_t2v_text_context_once(
        pipeline=pipeline,
        text=n_prompt,
        offload_model=bool(offload_model),
    )

    noise = [
        torch.randn(
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device=pipeline.device,
            generator=seed_g,
        )
    ]

    if sample_solver == "unipc":
        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=pipeline.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        sample_scheduler.set_timesteps(sampling_steps, device=pipeline.device, shift=shift)
        timesteps = sample_scheduler.timesteps
    elif sample_solver == "dpm++":
        sample_scheduler = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=pipeline.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
        timesteps, _ = retrieve_timesteps(
            sample_scheduler,
            device=pipeline.device,
            sigmas=sampling_sigmas,
        )
    else:
        raise NotImplementedError("Unsupported solver.")

    max_target_step = int(max(requested_steps))
    if int(max_target_step) > int(len(timesteps)):
        raise ValueError(
            f"target_steps={requested_steps} exceed available sampling steps={len(timesteps)}"
        )

    arg_c = {"context": context, "seq_len": seq_len}
    arg_null = {"context": context_null, "seq_len": seq_len}
    latents = noise
    context_cpu = [u.detach().float().cpu() for u in context]
    context_null_cpu = [u.detach().float().cpu() for u in context_null]
    cache: Dict[int, Dict[str, Any]] = {}

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(pipeline.model, "no_sync", noop_no_sync)
    with amp.autocast(dtype=pipeline.param_dtype), torch.no_grad(), no_sync():
        for step_index, t in enumerate(timesteps, start=1):
            if int(step_index) in requested_steps:
                cache[int(step_index)] = {
                    "latent_input": latents[0].detach().float().cpu(),
                    "timestep": t.detach().clone(),
                    "seq_len": int(seq_len),
                    "context": context_cpu,
                    "context_null": context_null_cpu,
                }
                if int(step_index) >= int(max_target_step) and len(cache) == len(requested_steps):
                    return cache

            latent_model_input = latents
            timestep = torch.stack([t])
            pipeline.model.to(pipeline.device)
            noise_pred_cond = pipeline.model(latent_model_input, t=timestep, **arg_c)[0]
            noise_pred_uncond = pipeline.model(latent_model_input, t=timestep, **arg_null)[0]
            noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0),
                t,
                latents[0].unsqueeze(0),
                return_dict=False,
                generator=seed_g,
            )[0]
            latents = [temp_x0.squeeze(0)]

    missing_steps = [step for step in requested_steps if int(step) not in cache]
    raise RuntimeError(f"Failed to capture target-step latents for steps={missing_steps}")


def _run_wan21_t2v_attribution_clean_forward(
    pipeline,
    latent_input: torch.Tensor,
    timestep_value: torch.Tensor,
    seq_len: int,
    context: Sequence[torch.Tensor],
    branch: str,
    target_step: int,
    target_layer: int,
    target_module: str,
    attribution_position: str,
    token_indices: Optional[torch.Tensor],
    use_gradient_checkpointing: bool,
    offload_model: bool,
) -> Wan21T2VTrajectoryConsensusAttributionState:
    """Run one clean local denoiser forward with gradients for attribution patching."""
    branch_name = str(branch).strip().lower()
    if branch_name not in {"cond", "uncond"}:
        raise ValueError(f"Unsupported branch for attribution clean forward: {branch}")

    target_model = _unwrap_wan21_t2v_dit_model_for_runtime_patch(pipeline.model)
    target_model.to(pipeline.device)
    original_param_requires_grad: List[bool] = [bool(param.requires_grad) for param in target_model.parameters()]
    for param in target_model.parameters():
        param.requires_grad_(False)
    handle = _install_wan21_t2v_trajectory_consensus_attribution_patch(
        model=pipeline.model,
        target_step=1,
        target_layer=int(target_layer),
        target_module=str(target_module),
        target_branch="cond",
        attribution_position=str(attribution_position),
        token_indices=token_indices,
        use_gradient_checkpointing=bool(use_gradient_checkpointing),
    )
    try:
        with amp.autocast(dtype=pipeline.param_dtype):
            latent = latent_input.to(device=pipeline.device, dtype=torch.float32).detach()
            timestep = torch.stack([timestep_value.to(device=pipeline.device)])
            context_device = [u.to(device=pipeline.device, dtype=torch.float32) for u in context]
            result = pipeline.model([latent], t=timestep, context=context_device, seq_len=int(seq_len))
            if handle.state.captured_clean_vpred is None and isinstance(result, list) and result:
                handle.state.captured_clean_vpred = result[0]
    finally:
        handle.restore()
        for param, requires_grad in zip(target_model.parameters(), original_param_requires_grad):
            param.requires_grad_(requires_grad)
    return handle.state


def _install_wan21_t2v_trajectory_consensus_global_attribution_patch(
    model,
    target_step: int,
    target_branch: str,
    selected_targets: Dict[int, Tuple[str, ...]],
    attribution_position: str,
    token_indices: Optional[torch.Tensor],
    use_gradient_checkpointing: bool,
) -> Wan21T2VTrajectoryConsensusGlobalAttributionHandle:
    """Install a global attribution patch that captures all selected layers/modules in one forward."""
    from projects.Wan2_1.wan.modules.attention import flash_attention
    from projects.Wan2_1.wan.modules.model import T5_CONTEXT_TOKEN_NUMBER, rope_apply
    from torch.utils.checkpoint import checkpoint

    target = _unwrap_wan21_t2v_dit_model_for_runtime_patch(model)
    if not hasattr(target, "blocks"):
        raise RuntimeError("Invalid DiT model: missing blocks.")

    normalized_targets: Dict[int, Tuple[str, ...]] = {}
    for layer, modules in selected_targets.items():
        layer_index = int(layer)
        if layer_index < 0 or layer_index >= len(target.blocks):
            raise ValueError(f"target_layer out of range: {layer_index}, num_layers={len(target.blocks)}")
        normalized_targets[layer_index] = tuple(
            sorted({str(module).strip().lower() for module in modules if str(module).strip().lower() in {"self", "cross"}})
        )
    normalized_targets = {layer: modules for layer, modules in normalized_targets.items() if modules}
    if not normalized_targets:
        raise ValueError("selected_targets must be non-empty for global attribution patch.")

    state = Wan21T2VTrajectoryConsensusGlobalAttributionState(
        target_step=int(target_step),
        target_branch=str(target_branch),
        attribution_position=str(attribution_position).strip().lower(),
        selected_modules=tuple(sorted({module for modules in normalized_targets.values() for module in modules})),
        token_indices=None if token_indices is None else token_indices.detach().long().cpu(),
    )

    original_forward = target.forward
    original_block_forwards: Dict[int, Any] = {}
    original_downstream_block_forwards: Dict[int, Any] = {}

    def patched_dit_forward(this, *args, **kwargs):
        t = kwargs.get("t", None)
        if t is None and len(args) > 1:
            t = args[1]
        state.on_forward_start(t)
        result = original_forward(*args, **kwargs)
        if state.is_target_forward():
            if isinstance(result, list) and result:
                state.captured_clean_vpred = result[0]
        return result

    def _project_all_head_writes(z_bthd: torch.Tensor, linear_module, post_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        num_heads = int(linear_module.num_heads)
        head_dim = int(linear_module.head_dim)
        weight = linear_module.o.weight.view(
            linear_module.o.out_features,
            num_heads,
            head_dim,
        ).permute(1, 2, 0).contiguous()
        head_writes = torch.einsum("bthd,hdo->btho", z_bthd, weight)
        if post_scale is not None:
            head_writes = head_writes * post_scale.unsqueeze(2)
        return head_writes

    def _run_manual_cross_attn(cross_attn_module, x_input, context, context_lens):
        b, n, d = x_input.size(0), cross_attn_module.num_heads, cross_attn_module.head_dim
        q = cross_attn_module.norm_q(cross_attn_module.q(x_input)).view(b, -1, n, d)
        if hasattr(cross_attn_module, "k_img") and hasattr(cross_attn_module, "v_img"):
            image_context_length = context.shape[1] - T5_CONTEXT_TOKEN_NUMBER
            context_img = context[:, :image_context_length]
            context_txt = context[:, image_context_length:]
            k = cross_attn_module.norm_k(cross_attn_module.k(context_txt)).view(b, -1, n, d)
            v = cross_attn_module.v(context_txt).view(b, -1, n, d)
            k_img = cross_attn_module.norm_k_img(cross_attn_module.k_img(context_img)).view(b, -1, n, d)
            v_img = cross_attn_module.v_img(context_img).view(b, -1, n, d)
            head_output = flash_attention(q, k, v, k_lens=context_lens)
            head_output = head_output + flash_attention(q, k_img, v_img, k_lens=None)
        else:
            k = cross_attn_module.norm_k(cross_attn_module.k(context)).view(b, -1, n, d)
            v = cross_attn_module.v(context).view(b, -1, n, d)
            head_output = flash_attention(q, k, v, k_lens=context_lens)
        return head_output

    def _capture_grad(key: Tuple[int, str], grad: torch.Tensor):
        state.captured_head_writes_grad_obj[key] = grad.detach().float().cpu()

    def _build_gradient_subset_activation(activation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if state.token_indices is None or int(state.token_indices.numel()) <= 0:
            activation_leaf = activation.detach().requires_grad_(True)
            return activation_leaf, activation_leaf
        token_indices_device = state.token_indices.to(device=activation.device)
        activation_subset = activation.index_select(dim=1, index=token_indices_device).detach().requires_grad_(True)
        activation_full = activation.detach().clone()
        activation_full.index_copy_(1, token_indices_device, activation_subset)
        return activation_full, activation_subset

    def _make_target_block_forward(layer_index: int, original_block_forward_fn):
        def target_block_forward(
            self,
            x,
            e,
            seq_lens,
            grid_sizes,
            freqs,
            context,
            context_lens,
        ):
            if not state.is_target_forward():
                return original_block_forward_fn(x, e, seq_lens, grid_sizes, freqs, context, context_lens)

            selected_modules = normalized_targets.get(int(layer_index), tuple())
            if not selected_modules:
                return original_block_forward_fn(x, e, seq_lens, grid_sizes, freqs, context, context_lens)

            assert e.dtype == torch.float32
            with amp.autocast(dtype=torch.float32):
                modulation = (self.modulation + e).chunk(6, dim=1)

            if "self" in selected_modules:
                sa_input = self.norm1(x).float() * (1 + modulation[1]) + modulation[0]
                batch_size, seq_len = sa_input.shape[:2]
                num_heads = self.self_attn.num_heads
                head_dim = self.self_attn.head_dim
                q = self.self_attn.norm_q(self.self_attn.q(sa_input)).view(batch_size, seq_len, num_heads, head_dim)
                k = self.self_attn.norm_k(self.self_attn.k(sa_input)).view(batch_size, seq_len, num_heads, head_dim)
                v = self.self_attn.v(sa_input).view(batch_size, seq_len, num_heads, head_dim)
                z_self = flash_attention(
                    q=rope_apply(q, grid_sizes, freqs),
                    k=rope_apply(k, grid_sizes, freqs),
                    v=v,
                    k_lens=seq_lens,
                    window_size=self.self_attn.window_size,
                )
                if str(state.attribution_position) == "post_o":
                    projected_self = _project_all_head_writes(z_self, self.self_attn, post_scale=modulation[2])
                    state.captured_head_write_means[(int(layer_index), "self")] = projected_self.detach().float().mean(dim=(0, 1), keepdim=True).cpu()
                    full_activation, tracked_activation = _build_gradient_subset_activation(projected_self)
                    state.captured_head_writes[(int(layer_index), "self")] = tracked_activation.detach().float().cpu()
                    tracked_activation.register_hook(lambda grad, key=(int(layer_index), "self"): _capture_grad(key, grad))
                    sa_output = full_activation.sum(dim=2)
                    with amp.autocast(dtype=torch.float32):
                        x = x + sa_output
                else:
                    state.captured_head_write_means[(int(layer_index), "self")] = z_self.detach().float().mean(dim=(0, 1), keepdim=True).cpu()
                    full_activation, tracked_activation = _build_gradient_subset_activation(z_self)
                    state.captured_head_writes[(int(layer_index), "self")] = tracked_activation.detach().float().cpu()
                    tracked_activation.register_hook(lambda grad, key=(int(layer_index), "self"): _capture_grad(key, grad))
                    sa_output = self.self_attn.o(full_activation.flatten(2))
                    with amp.autocast(dtype=torch.float32):
                        x = x + sa_output * modulation[2]
            else:
                sa_output = self.self_attn(
                    self.norm1(x).float() * (1 + modulation[1]) + modulation[0],
                    seq_lens,
                    grid_sizes,
                    freqs,
                )
                with amp.autocast(dtype=torch.float32):
                    x = x + sa_output * modulation[2]

            if "cross" in selected_modules:
                cross_input = self.norm3(x)
                z_cross = _run_manual_cross_attn(self.cross_attn, cross_input, context, context_lens)
                if str(state.attribution_position) == "post_o":
                    projected_cross = _project_all_head_writes(z_cross, self.cross_attn, post_scale=None)
                    state.captured_head_write_means[(int(layer_index), "cross")] = projected_cross.detach().float().mean(dim=(0, 1), keepdim=True).cpu()
                    full_activation, tracked_activation = _build_gradient_subset_activation(projected_cross)
                    state.captured_head_writes[(int(layer_index), "cross")] = tracked_activation.detach().float().cpu()
                    tracked_activation.register_hook(lambda grad, key=(int(layer_index), "cross"): _capture_grad(key, grad))
                    cross_output = full_activation.sum(dim=2)
                else:
                    state.captured_head_write_means[(int(layer_index), "cross")] = z_cross.detach().float().mean(dim=(0, 1), keepdim=True).cpu()
                    full_activation, tracked_activation = _build_gradient_subset_activation(z_cross)
                    state.captured_head_writes[(int(layer_index), "cross")] = tracked_activation.detach().float().cpu()
                    tracked_activation.register_hook(lambda grad, key=(int(layer_index), "cross"): _capture_grad(key, grad))
                    cross_output = self.cross_attn.o(full_activation.flatten(2))
                x = x + cross_output
            else:
                x = x + self.cross_attn(self.norm3(x), context, context_lens)

            y = self.ffn(self.norm2(x).float() * (1 + modulation[4]) + modulation[3])
            with amp.autocast(dtype=torch.float32):
                x = x + y * modulation[5]
            return x

        return target_block_forward

    base_block_forwards: Dict[int, Any] = {}
    for layer_index in range(len(target.blocks)):
        block_module = target.blocks[int(layer_index)]
        original_block_forwards[int(layer_index)] = block_module.forward
        if int(layer_index) in normalized_targets:
            base_block_forwards[int(layer_index)] = MethodType(
                _make_target_block_forward(int(layer_index), original_block_forwards[int(layer_index)]),
                block_module,
            )
        else:
            base_block_forwards[int(layer_index)] = original_block_forwards[int(layer_index)]

    first_target_layer = min(normalized_targets.keys())
    for layer_index in range(first_target_layer, len(target.blocks)):
        block_module = target.blocks[int(layer_index)]
        base_forward = base_block_forwards[int(layer_index)]
        if bool(use_gradient_checkpointing):
            original_downstream_block_forwards[int(layer_index)] = block_module.forward

            def _make_checkpointed_block_forward(base_forward_fn):
                def checkpointed_block_forward(
                    self,
                    x,
                    e,
                    seq_lens,
                    grid_sizes,
                    freqs,
                    context,
                    context_lens,
                ):
                    def block_fn(
                        x_tensor,
                        e_tensor,
                        seq_lens_tensor,
                        grid_sizes_tensor,
                        freqs_tensor,
                        context_tensor,
                    ):
                        return base_forward_fn(
                            x_tensor,
                            e_tensor,
                            seq_lens_tensor,
                            grid_sizes_tensor,
                            freqs_tensor,
                            context_tensor,
                            context_lens,
                        )
                    return checkpoint(
                        block_fn,
                        x,
                        e,
                        seq_lens,
                        grid_sizes,
                        freqs,
                        context,
                        use_reentrant=False,
                    )
                return checkpointed_block_forward

            block_module.forward = MethodType(_make_checkpointed_block_forward(base_forward), block_module)
        else:
            block_module.forward = base_forward

    target.forward = MethodType(patched_dit_forward, target)

    return Wan21T2VTrajectoryConsensusGlobalAttributionHandle(
        target_model=target,
        state=state,
        original_forward=original_forward,
        original_block_forwards=original_block_forwards,
        original_downstream_block_forwards=original_downstream_block_forwards,
    )


def _run_wan21_t2v_global_attribution_clean_forward(
    pipeline,
    latent_input: torch.Tensor,
    timestep_value: torch.Tensor,
    seq_len: int,
    context: Sequence[torch.Tensor],
    branch: str,
    target_step: int,
    selected_targets: Dict[int, Tuple[str, ...]],
    attribution_position: str,
    token_indices: Optional[torch.Tensor],
    use_gradient_checkpointing: bool,
    offload_model: bool,
) -> Wan21T2VTrajectoryConsensusGlobalAttributionState:
    """Run one clean local denoiser forward with gradients and collect all selected layers/modules at once."""
    branch_name = str(branch).strip().lower()
    if branch_name not in {"cond", "uncond"}:
        raise ValueError(f"Unsupported branch for attribution clean forward: {branch}")

    target_model = _unwrap_wan21_t2v_dit_model_for_runtime_patch(pipeline.model)
    target_model.to(pipeline.device)
    original_param_requires_grad: List[bool] = [bool(param.requires_grad) for param in target_model.parameters()]
    for param in target_model.parameters():
        param.requires_grad_(False)
    handle = _install_wan21_t2v_trajectory_consensus_global_attribution_patch(
        model=pipeline.model,
        target_step=1,
        target_branch="cond",
        selected_targets=selected_targets,
        attribution_position=str(attribution_position),
        token_indices=token_indices,
        use_gradient_checkpointing=bool(use_gradient_checkpointing),
    )
    try:
        with amp.autocast(dtype=pipeline.param_dtype):
            latent = latent_input.to(device=pipeline.device, dtype=torch.float32).detach()
            timestep = torch.stack([timestep_value.to(device=pipeline.device)])
            context_device = [u.to(device=pipeline.device, dtype=torch.float32) for u in context]
            result = pipeline.model([latent], t=timestep, context=context_device, seq_len=int(seq_len))
            if handle.state.captured_clean_vpred is None and isinstance(result, list) and result:
                handle.state.captured_clean_vpred = result[0]
    finally:
        handle.restore()
        for param, requires_grad in zip(target_model.parameters(), original_param_requires_grad):
            param.requires_grad_(requires_grad)
    return handle.state


def _run_wan21_t2v_global_direct_proxy_clean_forward(
    pipeline,
    latent_input: torch.Tensor,
    timestep_value: torch.Tensor,
    seq_len: int,
    context: Sequence[torch.Tensor],
    branch: str,
    target_step: int,
    selected_targets: Dict[int, Tuple[str, ...]],
    offload_model: bool,
) -> Wan21T2VTrajectoryConsensusGlobalDirectProxyState:
    """Run one clean local denoiser forward and capture post-o writes for all selected heads."""
    branch_name = str(branch).strip().lower()
    if branch_name not in {"cond", "uncond"}:
        raise ValueError(f"Unsupported branch for direct-proxy clean forward: {branch}")

    from projects.Wan2_1.wan.modules.attention import flash_attention
    from projects.Wan2_1.wan.modules.model import T5_CONTEXT_TOKEN_NUMBER, rope_apply

    target = _unwrap_wan21_t2v_dit_model_for_runtime_patch(pipeline.model)
    if not hasattr(target, "blocks"):
        raise RuntimeError("Invalid DiT model: missing blocks.")

    normalized_targets: Dict[int, Tuple[str, ...]] = {}
    for layer, modules in selected_targets.items():
        layer_index = int(layer)
        if layer_index < 0 or layer_index >= len(target.blocks):
            raise ValueError(f"target_layer out of range: {layer_index}, num_layers={len(target.blocks)}")
        normalized_targets[layer_index] = tuple(
            sorted({str(module).strip().lower() for module in modules if str(module).strip().lower() in {"self", "cross"}})
        )
    normalized_targets = {layer: modules for layer, modules in normalized_targets.items() if modules}
    if not normalized_targets:
        raise ValueError("selected_targets must be non-empty for global direct proxy.")

    state = Wan21T2VTrajectoryConsensusGlobalDirectProxyState(
        target_step=1,
        target_branch="cond",
        selected_modules=tuple(sorted({module for modules in normalized_targets.values() for module in modules})),
    )

    original_forward = target.forward
    original_head_forward = target.head.forward
    original_unpatchify = target.unpatchify
    original_block_forwards: Dict[int, Any] = {}

    def patched_dit_forward(this, *args, **kwargs):
        t = kwargs.get("t", None)
        if t is None and len(args) > 1:
            t = args[1]
        state.on_forward_start(t)
        result = original_forward(*args, **kwargs)
        if state.is_target_forward():
            if isinstance(result, list) and result:
                state.captured_clean_vpred = result[0].detach().float().cpu()
        return result

    def patched_head_forward(self, x, e):
        if state.is_target_forward():
            state.captured_head_e = e.detach().float().cpu()
        return original_head_forward(x, e)

    def patched_unpatchify(self, x, grid_sizes):
        if state.is_target_forward():
            state.captured_grid_sizes = grid_sizes.detach().cpu()
        return original_unpatchify(x, grid_sizes)

    def _project_all_head_writes(z_bthd: torch.Tensor, linear_module, post_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        num_heads = int(linear_module.num_heads)
        head_dim = int(linear_module.head_dim)
        weight = linear_module.o.weight.view(
            linear_module.o.out_features,
            num_heads,
            head_dim,
        ).permute(1, 2, 0).contiguous()
        head_writes = torch.einsum("bthd,hdo->btho", z_bthd, weight)
        if post_scale is not None:
            head_writes = head_writes * post_scale.unsqueeze(2)
        return head_writes

    def _run_manual_cross_attn(cross_attn_module, x_input, context, context_lens):
        b, n, d = x_input.size(0), cross_attn_module.num_heads, cross_attn_module.head_dim
        q = cross_attn_module.norm_q(cross_attn_module.q(x_input)).view(b, -1, n, d)
        if hasattr(cross_attn_module, "k_img") and hasattr(cross_attn_module, "v_img"):
            image_context_length = context.shape[1] - T5_CONTEXT_TOKEN_NUMBER
            context_img = context[:, :image_context_length]
            context_txt = context[:, image_context_length:]
            k = cross_attn_module.norm_k(cross_attn_module.k(context_txt)).view(b, -1, n, d)
            v = cross_attn_module.v(context_txt).view(b, -1, n, d)
            k_img = cross_attn_module.norm_k_img(cross_attn_module.k_img(context_img)).view(b, -1, n, d)
            v_img = cross_attn_module.v_img(context_img).view(b, -1, n, d)
            head_output = flash_attention(q, k, v, k_lens=context_lens)
            head_output = head_output + flash_attention(q, k_img, v_img, k_lens=None)
        else:
            k = cross_attn_module.norm_k(cross_attn_module.k(context)).view(b, -1, n, d)
            v = cross_attn_module.v(context).view(b, -1, n, d)
            head_output = flash_attention(q, k, v, k_lens=context_lens)
        return head_output

    def _make_target_block_forward(layer_index: int, original_block_forward_fn):
        def target_block_forward(
            self,
            x,
            e,
            seq_lens,
            grid_sizes,
            freqs,
            context,
            context_lens,
        ):
            if not state.is_target_forward():
                return original_block_forward_fn(x, e, seq_lens, grid_sizes, freqs, context, context_lens)

            selected_modules = normalized_targets.get(int(layer_index), tuple())
            if not selected_modules:
                return original_block_forward_fn(x, e, seq_lens, grid_sizes, freqs, context, context_lens)

            assert e.dtype == torch.float32
            with amp.autocast(dtype=torch.float32):
                modulation = (self.modulation + e).chunk(6, dim=1)

            if "self" in selected_modules:
                sa_input = self.norm1(x).float() * (1 + modulation[1]) + modulation[0]
                batch_size, seq_len = sa_input.shape[:2]
                num_heads = self.self_attn.num_heads
                head_dim = self.self_attn.head_dim
                q = self.self_attn.norm_q(self.self_attn.q(sa_input)).view(batch_size, seq_len, num_heads, head_dim)
                k = self.self_attn.norm_k(self.self_attn.k(sa_input)).view(batch_size, seq_len, num_heads, head_dim)
                v = self.self_attn.v(sa_input).view(batch_size, seq_len, num_heads, head_dim)
                z_self = flash_attention(
                    q=rope_apply(q, grid_sizes, freqs),
                    k=rope_apply(k, grid_sizes, freqs),
                    v=v,
                    k_lens=seq_lens,
                    window_size=self.self_attn.window_size,
                )
                post_o_self = _project_all_head_writes(z_self, self.self_attn, post_scale=modulation[2])
                state.captured_post_o_head_writes[(int(layer_index), "self")] = post_o_self.detach().float().cpu()
                sa_output = post_o_self.sum(dim=2)
                with amp.autocast(dtype=torch.float32):
                    x = x + sa_output
            else:
                sa_output = self.self_attn(
                    self.norm1(x).float() * (1 + modulation[1]) + modulation[0],
                    seq_lens,
                    grid_sizes,
                    freqs,
                )
                with amp.autocast(dtype=torch.float32):
                    x = x + sa_output * modulation[2]

            if "cross" in selected_modules:
                cross_input = self.norm3(x)
                z_cross = _run_manual_cross_attn(self.cross_attn, cross_input, context, context_lens)
                post_o_cross = _project_all_head_writes(z_cross, self.cross_attn, post_scale=None)
                state.captured_post_o_head_writes[(int(layer_index), "cross")] = post_o_cross.detach().float().cpu()
                x = x + post_o_cross.sum(dim=2)
            else:
                x = x + self.cross_attn(self.norm3(x), context, context_lens)

            y = self.ffn(self.norm2(x).float() * (1 + modulation[4]) + modulation[3])
            with amp.autocast(dtype=torch.float32):
                x = x + y * modulation[5]
            return x

        return target_block_forward

    for layer_index in sorted(normalized_targets.keys()):
        block_module = target.blocks[int(layer_index)]
        original_block_forwards[int(layer_index)] = block_module.forward
        block_module.forward = MethodType(
            _make_target_block_forward(int(layer_index), original_block_forwards[int(layer_index)]),
            block_module,
        )

    target.forward = MethodType(patched_dit_forward, target)
    target.head.forward = MethodType(patched_head_forward, target.head)
    target.unpatchify = MethodType(patched_unpatchify, target)

    target_model = _unwrap_wan21_t2v_dit_model_for_runtime_patch(pipeline.model)
    target_model.to(pipeline.device)
    original_param_requires_grad: List[bool] = [bool(param.requires_grad) for param in target_model.parameters()]
    for param in target_model.parameters():
        param.requires_grad_(False)
    handle = Wan21T2VTrajectoryConsensusGlobalDirectProxyHandle(
        target_model=target,
        state=state,
        original_forward=original_forward,
        original_head_forward=original_head_forward,
        original_unpatchify=original_unpatchify,
        original_block_forwards=original_block_forwards,
    )
    try:
        with amp.autocast(dtype=pipeline.param_dtype):
            latent = latent_input.to(device=pipeline.device, dtype=torch.float32).detach()
            timestep = torch.stack([timestep_value.to(device=pipeline.device)])
            context_device = [u.to(device=pipeline.device, dtype=torch.float32) for u in context]
            result = pipeline.model([latent], t=timestep, context=context_device, seq_len=int(seq_len))
            if handle.state.captured_clean_vpred is None and isinstance(result, list) and result:
                handle.state.captured_clean_vpred = result[0].detach().float().cpu()
    finally:
        handle.restore()
        for param, requires_grad in zip(target_model.parameters(), original_param_requires_grad):
            param.requires_grad_(requires_grad)
        if offload_model:
            pipeline.model.cpu()
            torch.cuda.empty_cache()
    return handle.state


def _move_wan21_t2v_tree_to_device(obj, device: torch.device):
    """Recursively move a nested payload of tensors to the requested device."""
    if torch.is_tensor(obj):
        return obj.to(device=device)
    if isinstance(obj, dict):
        return {key: _move_wan21_t2v_tree_to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_move_wan21_t2v_tree_to_device(value, device) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_move_wan21_t2v_tree_to_device(value, device) for value in obj)
    return obj


def _compute_wan21_t2v_attribution_patch_dot_metrics(
    clean_vpred: torch.Tensor,
    head_writes: torch.Tensor,
    head_writes_grad_full: torch.Tensor,
    head_writes_grad_obj: Optional[torch.Tensor],
    object_mask_fhw: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Compute all-head attribution-patching dot metrics from clean activations and gradients.

    Inputs:
    - `clean_vpred`: clean final prediction, shape `[C, F, H, W]`
    - `head_writes`: clean residual writes for all heads, shape `[B, T, H_head, D]`
    - `head_writes_grad_full`: gradient of clean full-field dot metric w.r.t. `head_writes`
    - `head_writes_grad_obj`: gradient of clean object-masked dot metric w.r.t. `head_writes`

    Returns:
    - tensors of shape `[H_head]` for:
      `dot_full`, `ablate_dot_full`, `dot_obj`, `ablate_dot_obj`
    """
    head_writes = head_writes.detach().float()
    head_writes_grad_full = head_writes_grad_full.detach().float()
    dot_full_clean = float(torch.dot(clean_vpred.detach().float().reshape(-1), clean_vpred.detach().float().reshape(-1)).item())
    dot_full = -torch.einsum("bthd,bthd->h", head_writes_grad_full, head_writes)
    ablate_dot_full = torch.full_like(dot_full, dot_full_clean) - dot_full

    if object_mask_fhw is None or head_writes_grad_obj is None:
        nan_vec = torch.full_like(dot_full, float("nan"))
        return {
            "dot_full": dot_full.cpu(),
            "ablate_dot_full": ablate_dot_full.cpu(),
            "dot_obj": nan_vec.cpu(),
            "ablate_dot_obj": nan_vec.cpu(),
        }

    mask = object_mask_fhw.detach().float().unsqueeze(0)
    masked_clean = clean_vpred.detach().float() * mask
    dot_obj_clean = float(torch.dot(masked_clean.reshape(-1), masked_clean.reshape(-1)).item())
    head_writes_grad_obj = head_writes_grad_obj.detach().float()
    dot_obj = -torch.einsum("bthd,bthd->h", head_writes_grad_obj, head_writes)
    ablate_dot_obj = torch.full_like(dot_obj, dot_obj_clean) - dot_obj

    return {
        "dot_full": dot_full.cpu(),
        "ablate_dot_full": ablate_dot_full.cpu(),
        "dot_obj": dot_obj.cpu(),
        "ablate_dot_obj": ablate_dot_obj.cpu(),
    }


def _build_wan21_t2v_taylor_scalar_metric(
    clean_vpred: torch.Tensor,
    metric_scope: str,
    patching_metric: str,
    object_mask_fhw: Optional[torch.Tensor],
    clean_uncond_vpred: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build the clean-only scalar metric used by Taylor attribution patching."""
    metric_scope = str(metric_scope).strip().lower()
    patching_metric = str(patching_metric).strip().lower()
    metric_tensor = clean_vpred
    metric_mask = None
    if metric_scope == "obj":
        if object_mask_fhw is None:
            raise ValueError("object_mask_fhw is required when trajectory_consensus_taylor_metric_scope='obj'.")
        metric_mask = object_mask_fhw.detach().to(device=clean_vpred.device, dtype=clean_vpred.dtype).unsqueeze(0)
        metric_tensor = metric_tensor * metric_mask
    elif metric_scope != "global":
        raise ValueError(f"Unsupported Taylor metric scope: {metric_scope}")

    if patching_metric == "v_sum":
        return metric_tensor.sum()
    if patching_metric == "ref_dot":
        clean_reference = metric_tensor.detach()
        return (metric_tensor * clean_reference).sum()
    if patching_metric == "sem_obj":
        if metric_scope != "obj":
            raise ValueError("trajectory_consensus_taylor_patching_metric='sem_obj' requires trajectory_consensus_taylor_metric_scope='obj'.")
        if clean_uncond_vpred is None:
            raise ValueError("clean_uncond_vpred is required when trajectory_consensus_taylor_patching_metric='sem_obj'.")
        uncond_vpred_device = clean_uncond_vpred.detach().to(device=clean_vpred.device, dtype=clean_vpred.dtype)
        semantic_reference = clean_vpred.detach() - uncond_vpred_device
        if metric_mask is not None:
            semantic_reference = semantic_reference * metric_mask
        return (metric_tensor * semantic_reference).sum()
    raise ValueError(f"Unsupported Taylor patching metric: {patching_metric}")


def _compute_wan21_t2v_taylor_contribution_metrics(
    head_writes: torch.Tensor,
    head_writes_grad: torch.Tensor,
    ablation_mode: str = "zero_ablation",
    full_token_head_mean: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Compute first-order Taylor contributions for all heads.

    Let `A_h` be the clean tracked activation of head `h` and let `A_h^base`
    denote the Taylor baseline.

    - `zero_ablation`: `A_h^base = 0`
    - `mean_ablation`: `A_h^base` is the per-head mean activation vector over
      all latent tokens at the current step, broadcast back to every tracked token.

    The first-order scalar change is

    `contribution_h = <∇_{A_h} M(clean), A_h^base - A_h(clean)>`.

    The returned tensor is one absolute-magnitude scalar per head.
    """
    head_writes = head_writes.detach().float()
    head_writes_grad = head_writes_grad.detach().float()
    baseline_mode = str(ablation_mode).strip().lower()
    if baseline_mode == "zero_ablation":
        delta = -head_writes
    elif baseline_mode == "mean_ablation":
        if full_token_head_mean is None:
            head_mean = head_writes.mean(dim=(0, 1), keepdim=True)
        else:
            head_mean = full_token_head_mean.detach().float()
            if head_mean.dim() != 4 or int(head_mean.shape[0]) != 1 or int(head_mean.shape[1]) != 1:
                raise ValueError("full_token_head_mean must have shape [1, 1, H, d].")
        delta = head_mean - head_writes
    else:
        raise ValueError(f"Unsupported Taylor ablation mode: {ablation_mode}")
    contribution_sum = torch.abs(torch.einsum("bthd,bthd->h", head_writes_grad, delta))
    num_tracked_positions = max(1, int(head_writes.shape[0]) * int(head_writes.shape[1]))
    contribution = contribution_sum / float(num_tracked_positions)
    return {"contribution": contribution.cpu()}


def _project_wan21_t2v_pre_o_heads_to_post_o_writes(
    pipeline,
    module_name: str,
    layer_index: int,
    per_head_pre_o: torch.Tensor,
    self_post_scale: Optional[torch.Tensor],
) -> torch.Tensor:
    """Convert stored pre-`o` head outputs into post-`o` per-head residual writes."""
    target = _unwrap_wan21_t2v_dit_model_for_runtime_patch(pipeline.model)
    block = target.blocks[int(layer_index)]
    module_name = str(module_name).strip().lower()
    if module_name == "self":
        linear_module = block.self_attn
        post_scale = None if self_post_scale is None else self_post_scale.detach().float().cpu()
    elif module_name == "cross":
        linear_module = block.cross_attn
        post_scale = None
    else:
        raise ValueError(f"Unsupported module for direct projection: {module_name}")

    num_heads = int(linear_module.num_heads)
    head_dim = int(linear_module.head_dim)
    weight = linear_module.o.weight.detach().float().cpu().view(
        linear_module.o.out_features,
        num_heads,
        head_dim,
    ).permute(1, 2, 0).contiguous()
    head_writes = torch.einsum("bthd,hdo->btho", per_head_pre_o.detach().float().cpu(), weight)
    if post_scale is not None:
        head_writes = head_writes * post_scale.unsqueeze(2)
    return head_writes


def _replay_wan21_t2v_suffix_from_head_write(
    pipeline,
    suffix_payload: Dict[str, Any],
    target_module: str,
    target_head: int,
    injected_head_write: torch.Tensor,
) -> torch.Tensor:
    """Replay the clean downstream suffix from one injected head write.

    The replay starts immediately after the targeted head write has been formed.
    For a self-attention head this means replacing the selected slice inside the
    self-attention residual write and then executing:
    - the residual add of self-attention;
    - the block's cross-attention and FFN;
    - all later transformer blocks;
    - the final output head and unpatchify.

    For a cross-attention head this means replacing the selected slice inside the
    cross-attention residual write and then executing:
    - the residual add of cross-attention;
    - the block FFN;
    - all later transformer blocks;
    - the final output head and unpatchify.
    """
    target = _unwrap_wan21_t2v_dit_model_for_runtime_patch(pipeline.model)
    model_device = target.head.head.weight.device
    payload = _move_wan21_t2v_tree_to_device(suffix_payload, model_device)
    head_write = injected_head_write.to(device=model_device)
    module_name = str(target_module).strip().lower()
    head_index = int(target_head)

    if module_name == "self":
        x_before = payload["x_before"]
        all_head_writes = payload["all_head_writes"].clone()
        all_head_writes[:, :, head_index, :] = head_write
        sa_write_sum = all_head_writes.sum(dim=2)
        with amp.autocast(dtype=torch.float32):
            x = x_before + sa_write_sum
        x = x + payload["cross_attn_module"](payload["norm3_module"](x), payload["context"], payload["context_lens"])
        y = payload["ffn_module"](payload["norm2_module"](x).float() * (1 + payload["modulation4"]) + payload["modulation3"])
        with amp.autocast(dtype=torch.float32):
            x = x + y * payload["modulation5"]
    elif module_name == "cross":
        x_before_cross = payload["x_before_cross"]
        all_head_writes = payload["all_head_writes"].clone()
        all_head_writes[:, :, head_index, :] = head_write
        cross_write_sum = all_head_writes.sum(dim=2)
        x = x_before_cross + cross_write_sum
        y = payload["ffn_module"](payload["norm2_module"](x).float() * (1 + payload["modulation4"]) + payload["modulation3"])
        with amp.autocast(dtype=torch.float32):
            x = x + y * payload["modulation5"]
    else:
        raise ValueError(f"Unsupported target_module for suffix replay: {target_module}")

    kwargs = {
        "seq_lens": payload["seq_lens"],
        "grid_sizes": payload["grid_sizes"],
        "freqs": payload["freqs"],
        "context": payload["context"],
        "context_lens": payload["context_lens"],
    }
    for block_index in range(int(payload["target_layer"]) + 1, len(target.blocks)):
        x = target.blocks[block_index](x, payload["e"], **kwargs)

    head_output = target.head(x, payload["head_e"])
    vpred_list = target.unpatchify(head_output, payload["grid_sizes"])
    if not isinstance(vpred_list, list) or not vpred_list:
        raise RuntimeError("Unexpected suffix replay output format.")
    return vpred_list[0]


def _compute_wan21_t2v_taylor_approx_scalar_metrics(
    pipeline,
    suffix_payload: Dict[str, Any],
    selected_head_write: torch.Tensor,
    clean_vpred: torch.Tensor,
    object_mask_fhw: Optional[torch.Tensor],
    target_module: str,
    target_head: int,
) -> Dict[str, float]:
    r"""Compute low-memory first-order Taylor metrics for head contribution.

    Instead of approximating the whole vector-valued `Delta v_pred`, this path
    follows the attribution-patching style scalar approximation:

    \[
    \Delta m_{s,\ell,h}^{\mathrm{ablate}}
    \approx
    \left\langle
    \nabla_U m\!\big(F_{s,\ell,h}(U_{s,\ell,h})\big),
    U_{s,\ell,h}
    \right\rangle .
    \]

    This avoids constructing the full JVP of the video-sized output and is far
    more memory efficient than the vector-valued implementation.
    """
    target = _unwrap_wan21_t2v_dit_model_for_runtime_patch(pipeline.model)
    model_device = target.head.head.weight.device
    clean_write = selected_head_write.detach().to(device=model_device, dtype=torch.float32).requires_grad_(True)
    clean_vpred_device = clean_vpred.detach().to(device=model_device, dtype=torch.float32)
    object_mask_device = None
    if object_mask_fhw is not None:
        object_mask_device = object_mask_fhw.detach().to(device=model_device, dtype=torch.float32).unsqueeze(0)

    def suffix_fn(write_tensor: torch.Tensor) -> torch.Tensor:
        return _replay_wan21_t2v_suffix_from_head_write(
            pipeline=pipeline,
            suffix_payload=suffix_payload,
            target_module=str(target_module),
            target_head=int(target_head),
            injected_head_write=write_tensor,
        )

    vpred_clean_from_suffix = suffix_fn(clean_write)
    dot_full_clean = torch.dot(vpred_clean_from_suffix.reshape(-1), clean_vpred_device.reshape(-1))
    grad_dot_full = torch.autograd.grad(dot_full_clean, clean_write, retain_graph=True, create_graph=False)[0]
    delta_dot_full = torch.dot(grad_dot_full.reshape(-1), clean_write.reshape(-1))

    metrics = {
        "dot_full": float(delta_dot_full.detach().cpu().item()),
        "ablate_dot_full": float((dot_full_clean - delta_dot_full).detach().cpu().item()),
        "cos_full": float("nan"),
        "ablate_cos_full": float("nan"),
    }

    if object_mask_device is None:
        metrics["dot_obj"] = float("nan")
        metrics["ablate_dot_obj"] = float("nan")
        metrics["cos_obj"] = float("nan")
        metrics["ablate_cos_obj"] = float("nan")
        return metrics

    masked_vpred = vpred_clean_from_suffix * object_mask_device
    masked_clean = clean_vpred_device * object_mask_device
    dot_obj_clean = torch.dot(masked_vpred.reshape(-1), masked_clean.reshape(-1))
    grad_dot_obj = torch.autograd.grad(dot_obj_clean, clean_write, retain_graph=False, create_graph=False)[0]
    delta_dot_obj = torch.dot(grad_dot_obj.reshape(-1), clean_write.reshape(-1))

    metrics["dot_obj"] = float(delta_dot_obj.detach().cpu().item())
    metrics["ablate_dot_obj"] = float((dot_obj_clean - delta_dot_obj).detach().cpu().item())
    metrics["cos_obj"] = float("nan")
    metrics["ablate_cos_obj"] = float("nan")
    return metrics


def _resolve_wan21_t2v_steps_and_layers_from_maps(
    mean_maps: Dict[Tuple[int, int, str], torch.Tensor],
    requested_steps: Sequence[int],
    requested_layers: Sequence[int],
) -> Tuple[List[int], List[int]]:
    """Resolve step/layer scopes from saved cross-attention maps."""
    available_steps = sorted({int(key[0]) for key in mean_maps.keys()})
    available_layers = sorted({int(key[1]) for key in mean_maps.keys()})

    if requested_steps:
        steps = _dedup_wan21_t2v_int_list(requested_steps)
        missing_steps = [step for step in steps if int(step) not in available_steps]
        if missing_steps:
            raise ValueError(f"Requested steps missing from reused maps: {missing_steps}")
    else:
        steps = list(available_steps)

    if requested_layers:
        layers = _dedup_wan21_t2v_int_list(requested_layers)
        missing_layers = [layer for layer in layers if int(layer) not in available_layers]
        if missing_layers:
            raise ValueError(f"Requested layers missing from reused maps: {missing_layers}")
    else:
        layers = list(available_layers)
    return [int(step) for step in steps], [int(layer) for layer in layers]


def _resolve_wan21_t2v_candidate_viz_frame_indices(frame_count: int, num_frames: int) -> List[int]:
    """Backward-compatible wrapper that returns attention-frame indices only."""
    frame_indices, _ = _resolve_wan21_t2v_trajectory_consensus_viz_frames(
        attention_frame_count=int(frame_count),
        video_frame_count=81,
        num_frames=int(num_frames),
    )
    return [int(frame_index) for frame_index in frame_indices]


def _load_wan21_t2v_head_trajectory_alignment_summary(
    reuse_head_trajectory_dynamics_dir: Optional[str],
    distance_metric: str,
    selected_steps: Sequence[int],
    alignment_summary_steps: int,
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """Load early-alignment summaries from head_trajectory_dynamics reference-distance CSV."""
    if not reuse_head_trajectory_dynamics_dir:
        return {}
    csv_path = os.path.join(
        reuse_head_trajectory_dynamics_dir,
        "head_trajectory_dynamics_reference_distance.csv",
    )
    rows = _load_wan21_t2v_csv_rows(csv_path)
    if not rows:
        return {}

    metric_key = f"{str(distance_metric).strip().lower()}_reference_distance"
    grouped_values: Dict[Tuple[int, int], List[Tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if metric_key not in row or str(row.get(metric_key, "")).strip() == "":
            continue
        step = int(row["step"])
        layer = int(row["layer"])
        head = int(row["head"])
        grouped_values[(layer, head)].append((int(step), float(row[metric_key])))

    summary = {}
    for key, step_value_pairs in grouped_values.items():
        if not step_value_pairs:
            continue
        sorted_pairs = sorted(step_value_pairs, key=lambda item: int(item[0]))
        first_step_values = [float(value) for _, value in sorted_pairs[:max(1, int(alignment_summary_steps))]]
        if not first_step_values:
            continue
        mean_value = float(sum(first_step_values) / len(first_step_values))
        summary[key] = {
            "convergence_speed": float(-mean_value),
            "alignment_mean": float(mean_value),
            "alignment_family": "distance",
        }
    return summary


def _load_wan21_t2v_head_evolution_support_quality_summary(
    reuse_head_evolution_dir: Optional[str],
    selected_steps: Sequence[int],
    alignment_summary_steps: int,
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """Load early support-quality summaries from head_evolution_headwise.csv."""
    if not reuse_head_evolution_dir:
        return {}
    csv_path = os.path.join(
        reuse_head_evolution_dir,
        "head_evolution_headwise.csv",
    )
    rows = _load_wan21_t2v_csv_rows(csv_path)
    if not rows:
        return {}

    grouped_values: Dict[Tuple[int, int], List[Tuple[int, float]]] = defaultdict(list)
    for row in rows:
        metric_text = str(row.get("support_quality_video", "")).strip()
        if not metric_text:
            continue
        step = int(row["step"])
        layer = int(row["layer"])
        head = int(row["head"])
        grouped_values[(layer, head)].append((int(step), float(metric_text)))

    summary = {}
    for key, step_value_pairs in grouped_values.items():
        if not step_value_pairs:
            continue
        sorted_pairs = sorted(step_value_pairs, key=lambda item: int(item[0]))
        first_step_values = [float(value) for _, value in sorted_pairs[:max(1, int(alignment_summary_steps))]]
        if not first_step_values:
            continue
        mean_value = float(sum(first_step_values) / len(first_step_values))
        summary[key] = {
            "convergence_speed": float(mean_value),
            "alignment_mean": float(mean_value),
            "alignment_family": "quality",
        }
    return summary


def _load_wan21_t2v_reference_distance_summaries(
    reuse_head_trajectory_dynamics_dir: Optional[str],
    reuse_head_evolution_dir: Optional[str],
    distance_metrics: Sequence[str],
    selected_steps: Sequence[int],
    alignment_summary_steps: int,
) -> Dict[str, Dict[Tuple[int, int], Dict[str, float]]]:
    """Load early-alignment summaries for multiple trajectory-consensus scatter metrics."""
    summaries: Dict[str, Dict[Tuple[int, int], Dict[str, float]]] = {}
    for metric_name in distance_metrics:
        metric_key = str(metric_name).strip()
        if not metric_key:
            continue
        normalized_metric_key = metric_key.lower()
        if normalized_metric_key in {"support_quality", "support_quality_video"}:
            summaries[metric_key] = _load_wan21_t2v_head_evolution_support_quality_summary(
                reuse_head_evolution_dir=reuse_head_evolution_dir,
                selected_steps=selected_steps,
                alignment_summary_steps=alignment_summary_steps,
            )
        else:
            summaries[metric_key] = _load_wan21_t2v_head_trajectory_alignment_summary(
                reuse_head_trajectory_dynamics_dir=reuse_head_trajectory_dynamics_dir,
                distance_metric=metric_key,
                selected_steps=selected_steps,
                alignment_summary_steps=alignment_summary_steps,
            )
    return summaries


def _append_wan21_t2v_alignment_scatter_row(
    scatter_rows: List[Dict[str, object]],
    *,
    alignment_metric_name: str,
    analysis_method: str,
    module_name: str,
    branch_name: str,
    metric_name: str,
    head_tag: str,
    step: int,
    metric_value: object,
    alignment_summary: Dict[str, float],
):
    """Append one alignment-scatter row when the metric value is finite."""
    metric_text = str(metric_value).strip()
    if not metric_text or metric_text.lower() == "nan":
        return
    scatter_rows.append({
        "alignment_metric_name": str(alignment_metric_name),
        "alignment_family": str(alignment_summary.get("alignment_family", "distance")),
        "analysis_method": str(analysis_method),
        "module": str(module_name),
        "branch": str(branch_name),
        "metric": str(metric_name),
        "head_tag": str(head_tag),
        "step": int(step),
        "convergence_speed": float(alignment_summary["convergence_speed"]),
        "alignment_mean": float(alignment_summary["alignment_mean"]),
        "metric_value": float(metric_value),
    })


def _normalize_wan21_t2v_path_component(text: str) -> str:
    """Convert a free-form label into a filesystem-friendly path component."""
    cleaned = str(text).strip().lower()
    cleaned = cleaned.replace("/", "_")
    cleaned = re.sub(r"[^a-z0-9_.-]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "metric"


def _apply_wan21_t2v_abs_to_ablation_contribution_row(
    row: Dict[str, object],
) -> Dict[str, object]:
    """Normalize ablation-derived contribution magnitudes to absolute values."""
    method_name = str(row.get("analysis_method", "")).strip().lower()
    if method_name == "taylor_approx":
        metric_names = ("contribution",)
    else:
        metric_names = tuple()
    for metric_name in metric_names:
        metric_text = str(row.get(metric_name, "")).strip()
        if not metric_text or metric_text.lower() == "nan":
            continue
        row[metric_name] = abs(float(metric_text))
    return row



def _format_wan21_t2v_filter_threshold_for_filename(value: float) -> str:
    """Convert one numeric threshold into a stable filename component."""
    scalar = float(value)
    if math.isinf(scalar):
        return "neg_inf" if scalar < 0 else "pos_inf"
    if math.isnan(scalar):
        return "nan"
    text = f"{scalar:.6g}"
    text = text.replace("-", "neg_")
    text = text.replace(".", "p")
    return _normalize_wan21_t2v_path_component(text)


def _resolve_wan21_t2v_filter_contribution_metric(
    analysis_method: str,
    explicit_metric_name: str,
) -> str:
    """Resolve the contribution column used by filtered-head export."""
    metric_name = str(explicit_metric_name).strip()
    if metric_name:
        return metric_name
    normalized_method = str(analysis_method).strip().lower()
    if normalized_method == "taylor_approx":
        return "contribution"
    if normalized_method == "exact_ablation":
        return "ablate_dot_obj"
    if normalized_method == "direct_proxy":
        return "proj_dot_obj"
    raise ValueError(f"Unsupported analysis method for filtered-head export: {analysis_method}")


def _parse_wan21_t2v_filter_rule(
    rule_text: str,
    *,
    default_direction: str,
    rule_name: str,
) -> Dict[str, object]:
    """Parse one threshold rule such as `gt_0.2` or `lt_10`.

    `gt_x` means value >= x and `lt_x` means value <= x.
    Bare numeric strings are still accepted for backward compatibility and are
    interpreted using `default_direction`.
    """
    normalized_default = str(default_direction).strip().lower()
    if normalized_default not in {"gt", "lt"}:
        raise ValueError(f"Unsupported default filter direction: {default_direction}")
    text = str(rule_text).strip()
    if not text:
        text = f"{normalized_default}_{'-inf' if normalized_default == 'gt' else 'inf'}"
    match = re.fullmatch(r"(gt|lt)\s*_\s*([-+]?((\d+\.?\d*)|(\.\d+))(?:[eE][-+]?\d+)?|inf|-inf)", text, flags=re.IGNORECASE)
    if match:
        direction = str(match.group(1)).strip().lower()
        value = float(str(match.group(2)).strip().lower())
    else:
        try:
            value = float(text)
        except ValueError as exc:
            raise ValueError(
                f"{rule_name} must be a rule like `gt_0.2` or `lt_10`, got: {rule_text}"
            ) from exc
        direction = normalized_default
    normalized_text = f"{direction}_{value:.6g}" if math.isfinite(value) else f"{direction}_{str(value).lower()}"
    return {
        "direction": direction,
        "value": float(value),
        "text": normalized_text,
    }


def _wan21_t2v_filter_rule_matches(value: float, rule: Dict[str, object]) -> bool:
    """Return whether one scalar passes the parsed filter rule."""
    direction = str(rule.get("direction", "")).strip().lower()
    threshold = float(rule.get("value", float("nan")))
    scalar = float(value)
    if direction == "gt":
        return scalar >= threshold
    if direction == "lt":
        return scalar <= threshold
    raise ValueError(f"Unsupported filter direction: {direction}")


def _export_wan21_t2v_filtered_heads(
    output_dir: str,
    head_rows: Sequence[Dict[str, object]],
    *,
    analysis_method: str,
    module_name: str,
    branch_name: str,
    reuse_head_evolution_dir: Optional[str],
    alignment_summary_steps: int,
    filter_step: int,
    convergence_speed_rule: str,
    contribution_rule: str,
    contribution_metric_name: str,
) -> str:
    """Export one comma-separated `LxHy` head list filtered by convergence speed and contribution."""
    if not head_rows:
        return ""
    normalized_method = str(analysis_method).strip().lower()
    normalized_module = str(module_name).strip().lower()
    normalized_branch = str(branch_name).strip().lower()
    metric_name = _resolve_wan21_t2v_filter_contribution_metric(normalized_method, contribution_metric_name)
    parsed_convergence_rule = _parse_wan21_t2v_filter_rule(
        str(convergence_speed_rule),
        default_direction="gt",
        rule_name="convergence_speed_rule",
    )
    parsed_contribution_rule = _parse_wan21_t2v_filter_rule(
        str(contribution_rule),
        default_direction="lt",
        rule_name="contribution_rule",
    )
    support_quality_summary = _load_wan21_t2v_head_evolution_support_quality_summary(
        reuse_head_evolution_dir=reuse_head_evolution_dir,
        selected_steps=tuple(),
        alignment_summary_steps=int(alignment_summary_steps),
    )
    if not support_quality_summary:
        raise FileNotFoundError(
            "Filtered-head export requires cached head_evolution support-quality results, but none were found."
        )

    matched_rows: List[Tuple[int, int, str]] = []
    for row in head_rows:
        if str(row.get("analysis_method", "")).strip().lower() != normalized_method:
            continue
        if str(row.get("module", "")).strip().lower() != normalized_module:
            continue
        if str(row.get("branch", "")).strip().lower() != normalized_branch:
            continue
        if int(row.get("step", -1)) != int(filter_step):
            continue
        metric_text = str(row.get(metric_name, "")).strip()
        if not metric_text or metric_text.lower() == "nan":
            continue
        layer = int(row["layer"])
        head = int(row["head"])
        summary_key = (layer, head)
        if summary_key not in support_quality_summary:
            continue
        convergence_speed = float(support_quality_summary[summary_key]["convergence_speed"])
        contribution_value = float(metric_text)
        if (
            _wan21_t2v_filter_rule_matches(convergence_speed, parsed_convergence_rule)
            and _wan21_t2v_filter_rule_matches(contribution_value, parsed_contribution_rule)
        ):
            matched_rows.append((layer, head, str(row.get("head_tag", f"L{layer}H{head}"))))

    matched_rows = sorted(dict.fromkeys(matched_rows), key=lambda item: (int(item[0]), int(item[1])))
    plots_root_dir = os.path.join(output_dir, "trajectory_consensus_head_contribution_plots")
    _ensure_dir(plots_root_dir)
    file_name = (
        f"filtered_heads_{normalized_module}_{normalized_branch}_step_{int(filter_step):03d}_"
        f"support_quality_video_convergence_speed_{_normalize_wan21_t2v_path_component(str(parsed_convergence_rule['text']))}_"
        f"{_normalize_wan21_t2v_path_component(metric_name)}_"
        f"{_normalize_wan21_t2v_path_component(str(parsed_contribution_rule['text']))}.txt"
    )
    save_path = os.path.join(plots_root_dir, file_name)
    with open(save_path, "w", encoding="utf-8") as handle:
        handle.write(",".join(head_tag for _, _, head_tag in matched_rows))
    return save_path


def _render_wan21_t2v_candidate_consensus_plots(
    output_dir: str,
    winner_gap_rows: Sequence[Dict[str, object]],
    skip_existing_plots: bool = False,
):
    """Render candidate-consensus summary heatmaps."""
    if not winner_gap_rows:
        return []

    mean_rows = []
    grouped_gap = defaultdict(list)
    grouped_entropy = defaultdict(list)
    for row in winner_gap_rows:
        grouped_gap[(int(row["step"]), int(row["layer"]))].append(float(row["winner_gap"]))
        grouped_entropy[(int(row["step"]), int(row["layer"]))].append(float(row["candidate_entropy"]))
    for (step, layer), values in sorted(grouped_gap.items()):
        mean_rows.append({
            "step": int(step),
            "layer": int(layer),
            "winner_gap_mean": float(sum(values) / len(values)),
            "candidate_entropy_mean": float(sum(grouped_entropy[(step, layer)]) / len(grouped_entropy[(step, layer)])),
        })

    plots_dir = os.path.join(output_dir, "trajectory_consensus_candidate_plots")
    plot_paths = []
    winner_gap_save = os.path.join(plots_dir, "winner_gap_step_layer_heatmap.pdf")
    entropy_save = os.path.join(plots_dir, "candidate_entropy_step_layer_heatmap.pdf")
    if _maybe_skip_wan21_t2v_existing_plot(winner_gap_save, skip_existing_plots):
        plot_paths.append(winner_gap_save)
    else:
        plot_paths.append(
            _plot_wan21_t2v_trajectory_consensus_heatmap(
                matrix_rows=mean_rows,
                save_file=winner_gap_save,
                title="Candidate Winner Gap (step-layer mean)",
                row_key="layer",
                col_key="step",
                value_key="winner_gap_mean",
                row_label="layer",
                col_label="step",
            )
        )
    if _maybe_skip_wan21_t2v_existing_plot(entropy_save, skip_existing_plots):
        plot_paths.append(entropy_save)
    else:
        plot_paths.append(
            _plot_wan21_t2v_trajectory_consensus_heatmap(
                matrix_rows=mean_rows,
                save_file=entropy_save,
                title="Candidate Entropy (step-layer mean)",
                row_key="layer",
                col_key="step",
                value_key="candidate_entropy_mean",
                row_label="layer",
                col_label="step",
            )
        )
    return [path for path in plot_paths if path]


def _render_wan21_t2v_head_contribution_plots(
    output_dir: str,
    head_rows: Sequence[Dict[str, object]],
    scatter_rows: Sequence[Dict[str, object]],
    scatter_outlier_heads_by_module: Optional[Dict[str, Sequence[str]]] = None,
    alignment_summary_steps: int = 10,
    skip_existing_plots: bool = False,
):
    """Render contribution heatmaps, curves, and scatter plots."""
    if not head_rows:
        return []

    plots_root_dir = os.path.join(output_dir, "trajectory_consensus_head_contribution_plots")
    heatmap_plots_dir = os.path.join(plots_root_dir, "heatmaps")
    plot_paths: List[str] = []
    normalized_outlier_by_module: Dict[str, set] = {}
    for module_name, head_tags in (scatter_outlier_heads_by_module or {}).items():
        normalized_outlier_by_module[str(module_name).strip().lower()] = {
            str(head_tag).strip().upper()
            for head_tag in head_tags
            if str(head_tag).strip()
        }

    method_metric_keys = {
        "exact_ablation": [
            "cos_full",
            "dot_full",
            "cos_obj",
            "dot_obj",
            "ablate_cos_full",
            "ablate_dot_full",
            "ablate_cos_obj",
            "ablate_dot_obj",
        ],
        "taylor_approx": ["contribution"],
        "direct_proxy": [
            "proj_cos_full",
            "proj_dot_full",
            "proj_cos_obj",
            "proj_dot_obj",
            "proj_share_full",
            "proj_share_obj",
        ],
    }
    grouped_by_method_module_branch = defaultdict(list)
    for row in head_rows:
        method_name = str(row.get("analysis_method") or row.get("contribution_method") or "").strip().lower()
        if not method_name:
            continue
        grouped_by_method_module_branch[(method_name, str(row["module"]), str(row["branch"]))].append(row)

    for (method_name, module_name, branch_name), rows in sorted(grouped_by_method_module_branch.items()):
        for metric_key in method_metric_keys.get(method_name, []):
            metric_rows = [row for row in rows if row.get(metric_key, "") != "" and str(row.get(metric_key, "")).lower() != "nan"]
            if not metric_rows:
                continue
            metric_dir = os.path.join(heatmap_plots_dir, module_name, branch_name, metric_key)
            metric_cmap = "Blues" if metric_key == "contribution" else "bwr"

            mean_rows = []
            grouped = defaultdict(list)
            for row in metric_rows:
                grouped[(int(row["step"]), int(row["layer"]))].append(float(row[metric_key]))
            for (step, layer), values in sorted(grouped.items()):
                mean_rows.append({
                    "step": int(step),
                    "layer": int(layer),
                    "value": float(sum(values) / len(values)),
                })

            save_file = os.path.join(metric_dir, f"{metric_key}_step_layer_heatmap.pdf")
            if _maybe_skip_wan21_t2v_existing_plot(save_file, skip_existing_plots):
                plot_paths.append(save_file)
            else:
                plot_paths.append(
                    _plot_wan21_t2v_trajectory_consensus_heatmap(
                        matrix_rows=mean_rows,
                        save_file=save_file,
                        title=(
                            f"{metric_key} ({method_name}, {module_name}, {branch_name})"
                        ),
                        row_key="layer",
                        col_key="step",
                        value_key="value",
                        row_label="layer",
                        col_label="step",
                        cmap=metric_cmap,
                    )
                )
            if metric_key == "contribution" and str(module_name).strip().lower() == "self":
                deep_layer_cutoff = 26
                shallow_mean_rows = [
                    row
                    for row in mean_rows
                    if int(row["layer"]) < int(deep_layer_cutoff)
                ]
                if shallow_mean_rows:
                    shallow_save_file = os.path.join(
                        metric_dir,
                        f"{metric_key}_step_layer_heatmap-remove_deep_layers.pdf",
                    )
                    if _maybe_skip_wan21_t2v_existing_plot(shallow_save_file, skip_existing_plots):
                        plot_paths.append(shallow_save_file)
                    else:
                        plot_paths.append(
                            _plot_wan21_t2v_trajectory_consensus_heatmap(
                                matrix_rows=shallow_mean_rows,
                                save_file=shallow_save_file,
                                title=(
                                    f"{metric_key} with layers<26 ({method_name}, {module_name}, {branch_name})"
                                ),
                                row_key="layer",
                                col_key="step",
                                value_key="value",
                                row_label="layer",
                                col_label="step",
                                cmap="Blues",
                            )
                        )

            steps = sorted(set(int(row["step"]) for row in metric_rows))
            layers = sorted(set(int(row["layer"]) for row in metric_rows))
            for step in steps:
                step_rows = [
                    {
                        "layer": int(row["layer"]),
                        "head": int(row["head"]),
                        "value": float(row[metric_key]),
                    }
                    for row in metric_rows
                    if int(row["step"]) == int(step)
                ]
                step_dir = os.path.join(metric_dir, f"step_{int(step):03d}")
                save_file = os.path.join(step_dir, f"{metric_key}_layer_head_step_{int(step):03d}.pdf")
                if _maybe_skip_wan21_t2v_existing_plot(save_file, skip_existing_plots):
                    plot_paths.append(save_file)
                else:
                    plot_paths.append(
                        _plot_wan21_t2v_trajectory_consensus_heatmap(
                            matrix_rows=step_rows,
                            save_file=save_file,
                            title=(
                                f"{metric_key} at step={int(step)} ({method_name}, {module_name}, "
                                f"{branch_name})"
                            ),
                            row_key="layer",
                            col_key="head",
                            value_key="value",
                            row_label="layer",
                            col_label="head",
                            cmap=metric_cmap,
                        )
                    )
                if metric_key == "contribution" and str(module_name).strip().lower() == "self":
                    deep_layer_cutoff = 26
                    shallow_step_rows = [
                        row
                        for row in step_rows
                        if int(row["layer"]) < int(deep_layer_cutoff)
                    ]
                    if shallow_step_rows:
                        shallow_save_file = os.path.join(
                            step_dir,
                            f"{metric_key}_layer_head-remove_deep_layers-step_{int(step):03d}.pdf",
                        )
                        if _maybe_skip_wan21_t2v_existing_plot(shallow_save_file, skip_existing_plots):
                            plot_paths.append(shallow_save_file)
                        else:
                            plot_paths.append(
                                _plot_wan21_t2v_trajectory_consensus_heatmap(
                                    matrix_rows=shallow_step_rows,
                                    save_file=shallow_save_file,
                                    title=(
                                        f"{metric_key} at step={int(step)} with layers<26 "
                                        f"({method_name}, {module_name}, {branch_name})"
                                    ),
                                    row_key="layer",
                                    col_key="head",
                                    value_key="value",
                                    row_label="layer",
                                    col_label="head",
                                    cmap="Blues",
                                )
                            )
            for layer in layers:
                layer_rows = [
                    {
                        "step": int(row["step"]),
                        "head": int(row["head"]),
                        "value": float(row[metric_key]),
                    }
                    for row in metric_rows
                    if int(row["layer"]) == int(layer)
                ]
                save_file = os.path.join(metric_dir, f"{metric_key}_step_head_layer_{int(layer):02d}.pdf")
                if _maybe_skip_wan21_t2v_existing_plot(save_file, skip_existing_plots):
                    plot_paths.append(save_file)
                else:
                    plot_paths.append(
                        _plot_wan21_t2v_trajectory_consensus_heatmap(
                            matrix_rows=layer_rows,
                            save_file=save_file,
                            title=(
                                f"{metric_key} at layer={int(layer)} ({method_name}, {module_name}, "
                                f"{branch_name})"
                            ),
                            row_key="head",
                            col_key="step",
                            value_key="value",
                            row_label="head",
                            col_label="step",
                            cmap=metric_cmap,
                        )
                    )

    grouped_scatter = defaultdict(list)
    for row in scatter_rows:
        method_name = str(row.get("analysis_method") or row.get("contribution_method") or "").strip().lower()
        if not method_name:
            continue
        grouped_scatter[(
            str(row.get("alignment_metric_name", "")).strip(),
            method_name,
            str(row["module"]),
            str(row["branch"]),
            str(row["metric"]),
        )].append(row)
    for (alignment_metric_name, method_name, module_name, branch_name, metric_name), rows in sorted(grouped_scatter.items()):
        alignment_metric_tag = _normalize_wan21_t2v_path_component(alignment_metric_name)
        scatter_plots_dir = os.path.join(plots_root_dir, f"scatter_{alignment_metric_tag}")
        metric_dir = os.path.join(scatter_plots_dir, module_name, branch_name, metric_name)
        alignment_family = str(rows[0].get("alignment_family", "distance")).strip().lower() if rows else "distance"
        if alignment_family == "quality":
            convergence_x_label = f"convergence speed"
        else:
            convergence_x_label = f"convergence speed"
        scatter_outlier_head_set = normalized_outlier_by_module.get(str(module_name).strip().lower(), set())
        scatter_variants = [("", rows)]
        if scatter_outlier_head_set:
            filtered_rows = [
                row for row in rows
                if str(row.get("head_tag", "")).strip().upper() not in scatter_outlier_head_set
            ]
            scatter_variants.append(("_del_outlier", filtered_rows))
        split_class_scatter = (
            str(alignment_metric_name).strip().lower() in {"support_quality", "support_quality_video"}
            and str(metric_name).strip().lower() == "contribution"
        )
        for suffix, variant_rows in scatter_variants:
            scatter_save = os.path.join(metric_dir, f"{metric_name}_vs_convergence_speed{suffix}.pdf")
            scatter_html_save = os.path.join(metric_dir, f"{metric_name}_vs_convergence_speed{suffix}.html")
            if _maybe_skip_wan21_t2v_existing_plot(scatter_save, skip_existing_plots):
                plot_paths.append(scatter_save)
            else:
                plot_paths.append(
                    _plot_wan21_t2v_trajectory_consensus_scatter(
                        rows=variant_rows,
                        save_file=scatter_save,
                        x_key="convergence_speed",
                        y_key="metric_value",
                        title=(
                            f"{metric_name} vs convergence speed ({method_name}, {module_name}, "
                            f"{branch_name}; alignment_metric={alignment_metric_name})"
                        ),
                        x_label=convergence_x_label,
                        y_label=metric_name,
                    )
                )
            if _maybe_skip_wan21_t2v_existing_plot(scatter_html_save, skip_existing_plots):
                plot_paths.append(scatter_html_save)
            else:
                html_path = _plot_wan21_t2v_trajectory_consensus_interactive_scatter(
                    rows=variant_rows,
                    save_file=scatter_html_save,
                    x_key="convergence_speed",
                    y_key="metric_value",
                    title=(
                        f"{metric_name} vs convergence speed ({method_name}, {module_name}, "
                        f"{branch_name}; alignment_metric={alignment_metric_name})"
                    ),
                    x_label=convergence_x_label,
                    y_label=metric_name,
                )
                if html_path:
                    plot_paths.append(html_path)
            if split_class_scatter:
                split_scatter_save = os.path.join(metric_dir, f"{metric_name}_vs_convergence_speed{suffix}_split_classes.pdf")
                split_scatter_html_save = os.path.join(metric_dir, f"{metric_name}_vs_convergence_speed{suffix}_split_classes.html")
                if _maybe_skip_wan21_t2v_existing_plot(split_scatter_save, skip_existing_plots):
                    plot_paths.append(split_scatter_save)
                else:
                    plot_paths.append(
                        _plot_wan21_t2v_trajectory_consensus_scatter(
                            rows=variant_rows,
                            save_file=split_scatter_save,
                            x_key="convergence_speed",
                            y_key="metric_value",
                            title=(
                                f"{metric_name} vs convergence speed ({method_name}, {module_name}, "
                                f"{branch_name}; alignment_metric={alignment_metric_name}; split classes)"
                            ),
                            x_label=convergence_x_label,
                            y_label=metric_name,
                            split_classes=True,
                        )
                    )
                if _maybe_skip_wan21_t2v_existing_plot(split_scatter_html_save, skip_existing_plots):
                    plot_paths.append(split_scatter_html_save)
                else:
                    html_path = _plot_wan21_t2v_trajectory_consensus_interactive_scatter(
                        rows=variant_rows,
                        save_file=split_scatter_html_save,
                        x_key="convergence_speed",
                        y_key="metric_value",
                        title=(
                            f"{metric_name} vs convergence speed ({method_name}, {module_name}, "
                            f"{branch_name}; alignment_metric={alignment_metric_name}; split classes)"
                        ),
                        x_label=convergence_x_label,
                        y_label=metric_name,
                        split_classes=True,
                    )
                    if html_path:
                        plot_paths.append(html_path)
        per_step_groups = defaultdict(list)
        for row in rows:
            per_step_groups[int(row["step"])].append(row)
        for step, step_rows in sorted(per_step_groups.items()):
            step_dir = os.path.join(metric_dir, f"step_{int(step):03d}")
            step_variants = [("", step_rows)]
            if scatter_outlier_head_set:
                filtered_step_rows = [
                    row for row in step_rows
                    if str(row.get("head_tag", "")).strip().upper() not in scatter_outlier_head_set
                ]
                step_variants.append(("_del_outlier", filtered_step_rows))
            for suffix, variant_step_rows in step_variants:
                step_scatter_save = os.path.join(step_dir, f"{metric_name}_vs_convergence_speed{suffix}.pdf")
                step_scatter_html_save = os.path.join(step_dir, f"{metric_name}_vs_convergence_speed{suffix}.html")
                if _maybe_skip_wan21_t2v_existing_plot(step_scatter_save, skip_existing_plots):
                    plot_paths.append(step_scatter_save)
                else:
                    plot_paths.append(
                        _plot_wan21_t2v_trajectory_consensus_scatter(
                            rows=variant_step_rows,
                            save_file=step_scatter_save,
                            x_key="convergence_speed",
                            y_key="metric_value",
                            title=(
                                f"{metric_name} vs convergence speed at step={int(step)} "
                                f"({method_name}, {module_name}, {branch_name}; "
                                f"alignment_metric={alignment_metric_name})"
                            ),
                            x_label=convergence_x_label,
                            y_label=metric_name,
                        )
                    )
                if _maybe_skip_wan21_t2v_existing_plot(step_scatter_html_save, skip_existing_plots):
                    plot_paths.append(step_scatter_html_save)
                else:
                    html_path = _plot_wan21_t2v_trajectory_consensus_interactive_scatter(
                        rows=variant_step_rows,
                        save_file=step_scatter_html_save,
                        x_key="convergence_speed",
                        y_key="metric_value",
                        title=(
                            f"{metric_name} vs convergence speed at step={int(step)} "
                            f"({method_name}, {module_name}, {branch_name}; "
                            f"alignment_metric={alignment_metric_name})"
                        ),
                        x_label=convergence_x_label,
                        y_label=metric_name,
                    )
                    if html_path:
                        plot_paths.append(html_path)
                if split_class_scatter:
                    step_split_scatter_save = os.path.join(step_dir, f"{metric_name}_vs_convergence_speed{suffix}_split_classes.pdf")
                    step_split_scatter_html_save = os.path.join(step_dir, f"{metric_name}_vs_convergence_speed{suffix}_split_classes.html")
                    if _maybe_skip_wan21_t2v_existing_plot(step_split_scatter_save, skip_existing_plots):
                        plot_paths.append(step_split_scatter_save)
                    else:
                        plot_paths.append(
                            _plot_wan21_t2v_trajectory_consensus_scatter(
                                rows=variant_step_rows,
                                save_file=step_split_scatter_save,
                                x_key="convergence_speed",
                                y_key="metric_value",
                                title=(
                                    f"{metric_name} vs convergence speed at step={int(step)} "
                                    f"({method_name}, {module_name}, {branch_name}; "
                                    f"alignment_metric={alignment_metric_name}; split classes)"
                                ),
                                x_label=convergence_x_label,
                                y_label=metric_name,
                                split_classes=True,
                            )
                        )
                    if _maybe_skip_wan21_t2v_existing_plot(step_split_scatter_html_save, skip_existing_plots):
                        plot_paths.append(step_split_scatter_html_save)
                    else:
                        html_path = _plot_wan21_t2v_trajectory_consensus_interactive_scatter(
                            rows=variant_step_rows,
                            save_file=step_split_scatter_html_save,
                            x_key="convergence_speed",
                            y_key="metric_value",
                            title=(
                                f"{metric_name} vs convergence speed at step={int(step)} "
                                f"({method_name}, {module_name}, {branch_name}; "
                                f"alignment_metric={alignment_metric_name}; split classes)"
                            ),
                            x_label=convergence_x_label,
                            y_label=metric_name,
                            split_classes=True,
                        )
                        if html_path:
                            plot_paths.append(html_path)
    return [path for path in plot_paths if path]


def _plot_wan21_t2v_trajectory_consensus_bar(
    rows: Sequence[Dict[str, object]],
    save_file: str,
    x_key: str,
    y_key: str,
    title: str,
    x_label: str,
    y_label: str,
):
    """Render a simple bar chart from row dictionaries."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_rows = [row for row in rows if row.get(x_key, "") != "" and row.get(y_key, "") != ""]
    if not plot_rows:
        return ""

    ordered_rows = sorted(plot_rows, key=lambda row: float(row[y_key]))
    x_values = [str(row[x_key]) for row in ordered_rows]
    y_values = [float(row[y_key]) for row in ordered_rows]

    fig_width = max(8.4, 0.42 * len(x_values))
    fig, axis = plt.subplots(1, 1, figsize=(fig_width, 5.0))
    axis.bar(list(range(len(x_values))), y_values, color="#0f766e", alpha=0.88)
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_xticks(list(range(len(x_values))))
    axis.set_xticklabels(x_values, rotation=45, ha="right", fontsize=8)
    axis.grid(axis="y", alpha=0.22, linestyle="--")
    fig.tight_layout()
    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file


def _load_wan21_t2v_trajectory_consensus_candidate_cache(
    output_dir: str,
) -> Tuple[
    Dict[Tuple[int, int], Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
]:
    """Load cached candidate-consensus outputs from one trajectory-consensus directory."""
    candidate_regions_pt_path = os.path.join(output_dir, "trajectory_consensus_candidate_regions.pt")
    candidate_regions_csv_path = os.path.join(output_dir, "trajectory_consensus_candidate_regions.csv")
    candidate_weights_csv_path = os.path.join(output_dir, "trajectory_consensus_candidate_weights.csv")
    winner_gap_csv_path = os.path.join(output_dir, "trajectory_consensus_winner_gap.csv")
    if not os.path.exists(candidate_regions_pt_path):
        raise FileNotFoundError(
            "self_attention_coupling requires cached candidate regions from candidate_consensus: "
            f"{candidate_regions_pt_path}"
        )
    candidate_region_cache = {
        (int(step), int(layer)): {
            "label_map_fhw": (
                torch.from_numpy(np.asarray(candidate_payload["label_map_fhw_np"])).to(torch.int64)
                if "label_map_fhw_np" in candidate_payload
                else candidate_payload["label_map_fhw"].detach().cpu().to(torch.int64)
            )
        }
        for (step, layer), candidate_payload in _load_wan21_t2v_torch_cache(candidate_regions_pt_path).items()
    }
    candidate_region_rows = _load_wan21_t2v_csv_rows(candidate_regions_csv_path) if os.path.exists(candidate_regions_csv_path) else []
    candidate_weight_rows = _load_wan21_t2v_csv_rows(candidate_weights_csv_path) if os.path.exists(candidate_weights_csv_path) else []
    winner_gap_rows = _load_wan21_t2v_csv_rows(winner_gap_csv_path) if os.path.exists(winner_gap_csv_path) else []
    return candidate_region_cache, candidate_region_rows, candidate_weight_rows, winner_gap_rows


def _safe_wan21_t2v_float(value: object, default: float = float("nan")) -> float:
    """Best-effort conversion to float with a fallback."""
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _mean_wan21_t2v_finite(values: Sequence[float]) -> float:
    """Average finite values only."""
    finite_values = [float(v) for v in values if math.isfinite(float(v))]
    if not finite_values:
        return float("nan")
    return float(sum(finite_values) / len(finite_values))


def _compute_wan21_t2v_binary_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute AUROC without sklearn."""
    paired = [
        (int(label), float(score))
        for label, score in zip(labels, scores)
        if math.isfinite(float(score))
    ]
    positives = [score for label, score in paired if int(label) == 1]
    negatives = [score for label, score in paired if int(label) == 0]
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    total = float(len(positives) * len(negatives))
    for positive_score in positives:
        for negative_score in negatives:
            if positive_score > negative_score:
                wins += 1.0
            elif positive_score == negative_score:
                wins += 0.5
    return float(wins / max(1.0, total))


def _build_wan21_t2v_anchor_union_payload(
    candidate_region_cache: Dict[Tuple[int, int], Dict[str, object]],
    anchor_step: int,
    anchor_layer: int,
) -> Dict[str, object]:
    """Build the anchor union mask and per-frame centroids."""
    anchor_payload = candidate_region_cache.get((int(anchor_step), int(anchor_layer)))
    if anchor_payload is None:
        raise KeyError(
            "Missing anchor candidate-region cache for self_attention_coupling: "
            f"step={int(anchor_step)} layer={int(anchor_layer)}"
        )
    anchor_label_map = anchor_payload["label_map_fhw"].detach().cpu().to(torch.int64)
    anchor_union_mask = anchor_label_map > 0
    anchor_centers: List[Tuple[float, float]] = []
    for frame_index in range(int(anchor_union_mask.shape[0])):
        frame_points = torch.nonzero(anchor_union_mask[frame_index], as_tuple=False)
        if int(frame_points.numel()) <= 0:
            anchor_centers.append((float("nan"), float("nan")))
            continue
        anchor_centers.append(
            (
                float(frame_points[:, 0].float().mean().item()),
                float(frame_points[:, 1].float().mean().item()),
            )
        )
    return {
        "mask_fhw": anchor_union_mask,
        "centers": anchor_centers,
    }


def _compute_wan21_t2v_pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Compute Pearson correlation on finite pairs only."""
    paired = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(paired) < 2:
        return float("nan")
    x_values = np.asarray([x for x, _ in paired], dtype=np.float64)
    y_values = np.asarray([y for _, y in paired], dtype=np.float64)
    x_centered = x_values - x_values.mean()
    y_centered = y_values - y_values.mean()
    denom = float(np.sqrt((x_centered ** 2).sum()) * np.sqrt((y_centered ** 2).sum()))
    if denom <= 1e-12:
        return float("nan")
    return float((x_centered * y_centered).sum() / denom)


def _smooth_wan21_t2v_curve_values(
    xs: Sequence[float],
    ys: Sequence[float],
    window_radius: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a simple local-mean smoothed curve on sorted finite points."""
    paired = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if not paired:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    paired = sorted(paired, key=lambda item: float(item[0]))
    x_values = np.asarray([x for x, _ in paired], dtype=np.float64)
    y_values = np.asarray([y for _, y in paired], dtype=np.float64)
    radius = max(0, int(window_radius))
    if radius <= 0 or len(y_values) <= 2:
        return x_values, y_values
    smoothed = np.zeros_like(y_values)
    for index in range(len(y_values)):
        left = max(0, int(index - radius))
        right = min(len(y_values), int(index + radius + 1))
        smoothed[index] = float(y_values[left:right].mean())
    return x_values, smoothed


def _trajectory_consensus_self_attention_feature_display_name(feature_name: str) -> str:
    """Return a readable display label for one self-attention coupling feature."""
    feature_key = str(feature_name)
    display_names = {
        "proposal_pi": "CA proposal strength",
        "proposal_vote_share": "CA vote share",
        "local_avg_covered_mass": "SA local covered mass",
        "local_avg_covered_mass_filtered": "SA local covered mass (filtered)",
        "global_avg_covered_mass": "SA global covered mass",
        "global_avg_covered_mass_filtered": "SA global covered mass (filtered)",
        "local_entropy": "SA local entropy",
        "local_entropy_filtered": "SA local entropy (filtered)",
        "global_entropy": "SA global entropy",
        "global_entropy_filtered": "SA global entropy (filtered)",
        "local_dominant_link_ratio": "SA local dominant-link ratio",
        "local_dominant_link_ratio_filtered": "SA local dominant-link ratio (filtered)",
        "global_dominant_link_ratio": "SA global dominant-link ratio",
        "global_dominant_link_ratio_filtered": "SA global dominant-link ratio (filtered)",
        "local_link_margin": "SA local link margin",
        "local_link_margin_filtered": "SA local link margin (filtered)",
        "global_link_margin": "SA global link margin",
        "global_link_margin_filtered": "SA global link margin (filtered)",
        "local_head_agreement": "SA local head agreement",
        "local_head_agreement_filtered": "SA local head agreement (filtered)",
        "global_head_agreement": "SA global head agreement",
        "global_head_agreement_filtered": "SA global head agreement (filtered)",
        "local_soft_head_vote_share": "SA local soft head-vote share",
        "global_soft_head_vote_share": "SA global soft head-vote share",
        "local_soft_head_agreement": "SA local soft head agreement",
        "global_soft_head_agreement": "SA global soft head agreement",
        "local_compatibility": "SA local compatibility",
        "global_compatibility": "SA global compatibility",
        "local_chainability": "SA local chainability",
        "global_chainability": "SA global chainability",
        "local_incoming_support": "SA local incoming support",
        "global_incoming_support": "SA global incoming support",
        "local_incoming_preference_share": "SA local incoming preference share",
        "global_incoming_preference_share": "SA global incoming preference share",
        "local_incoming_vote_share": "SA local incoming vote share",
        "global_incoming_vote_share": "SA global incoming vote share",
        "local_mutual_consistency": "SA local mutual consistency",
        "global_mutual_consistency": "SA global mutual consistency",
        "anchor_iou": "anchor IoU",
        "anchor_distance": "anchor-distance score",
        "anchor_center_l2": "anchor-distance score",
    }
    return display_names.get(feature_key, feature_key.replace("_", " "))


def _trajectory_consensus_self_attention_feature_plot_range(
    feature_name: str,
    finite_values: Sequence[float],
) -> Tuple[float, float]:
    """Return a stable color range for one overlay feature."""
    values = [
        float(value)
        for value in finite_values
        if math.isfinite(float(value))
    ]
    if not values:
        return (0.0, 1.0)
    feature_key = str(feature_name)
    bounded_zero_one = {
        "proposal_pi",
        "proposal_vote_share",
        "proposal_agreement_frame",
        "local_avg_covered_mass",
        "local_avg_covered_mass_filtered",
        "global_avg_covered_mass",
        "global_avg_covered_mass_filtered",
        "local_dominant_link_ratio",
        "local_dominant_link_ratio_filtered",
        "global_dominant_link_ratio",
        "global_dominant_link_ratio_filtered",
        "local_link_margin",
        "local_link_margin_filtered",
        "global_link_margin",
        "global_link_margin_filtered",
        "local_head_agreement",
        "local_head_agreement_filtered",
        "global_head_agreement",
        "global_head_agreement_filtered",
        "local_soft_head_vote_share",
        "local_soft_head_vote_share_filtered",
        "global_soft_head_vote_share",
        "global_soft_head_vote_share_filtered",
        "local_soft_head_agreement",
        "local_soft_head_agreement_filtered",
        "global_soft_head_agreement",
        "global_soft_head_agreement_filtered",
        "local_chainability",
        "global_chainability",
        "local_incoming_preference_share",
        "global_incoming_preference_share",
        "local_incoming_vote_share",
        "global_incoming_vote_share",
        "local_mutual_consistency",
        "global_mutual_consistency",
    }
    bounded_zero_two = {
        "local_compatibility",
        "global_compatibility",
    }
    if feature_key in bounded_zero_one:
        return (0.0, 1.0)
    if feature_key in bounded_zero_two:
        return (0.0, 2.0)
    vmin = float(min(values))
    vmax = float(max(values))
    if abs(vmax - vmin) < 1e-8:
        center = float(values[0])
        if abs(center) < 1e-8:
            return (0.0, 1.0)
        if center > 0.0:
            return (0.0, float(center))
        return (float(center), 0.0)
    return (vmin, vmax)


def _trajectory_consensus_prepare_display_candidate_mask(candidate_mask: np.ndarray) -> np.ndarray:
    """Return a visualization-friendly candidate mask with interior holes filled."""
    mask_bool = np.asarray(candidate_mask, dtype=np.bool_)
    if not np.any(mask_bool):
        return mask_bool
    try:
        from scipy.ndimage import binary_fill_holes
        return np.asarray(binary_fill_holes(mask_bool), dtype=np.bool_)
    except Exception:
        return mask_bool


def _trajectory_consensus_candidate_edge_color(fill_color: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Derive a contour color that stays visually distinct from the fill color."""
    import colorsys

    red, green, blue = [max(0.0, min(1.0, float(channel))) for channel in fill_color]
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)
    comp_hue = (float(hue) + 0.5) % 1.0
    comp_lightness = min(0.78, max(0.28, 0.52 if lightness > 0.62 else 0.42))
    comp_saturation = min(1.0, max(0.55, float(saturation)))
    comp_red, comp_green, comp_blue = colorsys.hls_to_rgb(
        comp_hue,
        comp_lightness,
        comp_saturation,
    )
    return (float(comp_red), float(comp_green), float(comp_blue))


def _resolve_wan21_t2v_overlay_anchor_role_candidates(
    frame_rows: Sequence[Dict[str, object]],
    anchor_mode: str,
) -> Tuple[int, int]:
    """Resolve overlay winner/loser using one anchor-based ranking rule."""
    mode = str(anchor_mode).strip().lower()
    if mode not in {"anchor_iou", "anchor_distance"}:
        raise ValueError(f"Unsupported overlay anchor mode: {anchor_mode}")
    scored_rows: List[Tuple[float, int]] = []
    for row in frame_rows:
        candidate_index = int(row["candidate_index"])
        if mode == "anchor_iou":
            score_value = _safe_wan21_t2v_float(row.get("anchor_iou", float("nan")))
            if math.isfinite(score_value):
                scored_rows.append((float(score_value), int(candidate_index)))
        else:
            distance_value = _safe_wan21_t2v_float(row.get("anchor_distance", float("nan")))
            if math.isfinite(distance_value):
                scored_rows.append((float(-distance_value), int(candidate_index)))
    if not scored_rows:
        return (-1, -1)
    scored_rows = sorted(
        scored_rows,
        key=lambda item: (float(item[0]), -int(item[1])),
        reverse=True,
    )
    winner_candidate = int(scored_rows[0][1])
    loser_candidate = int(scored_rows[1][1]) if len(scored_rows) >= 2 else -1
    return (winner_candidate, loser_candidate)


def _build_wan21_t2v_self_attention_pairwise_layer_stats(
    pairwise_rows: Sequence[Dict[str, object]],
) -> Tuple[
    Dict[Tuple[int, int, int], np.ndarray],
    Dict[Tuple[int, int, int], Dict[str, float]],
]:
    """Build layer-mean pairwise coupling vectors and summary metrics.

    For each `(query_frame, query_candidate, key_frame)` group, this helper
    averages the candidate-normalized coupling vectors across all selected
    self-attention heads in the layer and computes the corresponding layer-level
    sharpness metrics.
    """
    selected_heads: List[int] = sorted({int(row["head"]) for row in pairwise_rows})
    pairwise_vectors: Dict[Tuple[int, int, int], Dict[int, np.ndarray]] = defaultdict(dict)
    pairwise_covered: Dict[Tuple[int, int, int], Dict[int, float]] = defaultdict(dict)
    for row in pairwise_rows:
        query_frame = int(row["query_frame"])
        query_candidate = int(row["query_candidate"])
        key_frame = int(row["key_frame"])
        key_candidate = int(row["key_candidate"])
        key_candidate_count = int(row["key_candidate_count"])
        head_index = int(row["head"])
        group_key = (int(query_frame), int(query_candidate), int(key_frame))
        if head_index not in pairwise_vectors[group_key]:
            pairwise_vectors[group_key][head_index] = np.zeros((int(key_candidate_count),), dtype=np.float32)
        pairwise_vectors[group_key][head_index][int(key_candidate - 1)] = np.float32(
            _safe_wan21_t2v_float(row.get("normalized_coupling", 0.0), default=0.0)
        )
        pairwise_covered[group_key][head_index] = float(
            _safe_wan21_t2v_float(row.get("covered_mass", 0.0), default=0.0)
        )

    pairwise_layer_vectors: Dict[Tuple[int, int, int], np.ndarray] = {}
    pairwise_layer_metrics: Dict[Tuple[int, int, int], Dict[str, float]] = {}
    for group_key, head_map in pairwise_vectors.items():
        if not head_map:
            continue
        key_candidate_count = max(int(vector.shape[0]) for vector in head_map.values())
        valid_vote_targets: List[int] = []
        layer_vectors: List[np.ndarray] = []
        layer_covered_values: List[float] = []
        for head_index in selected_heads:
            vector = np.zeros((int(key_candidate_count),), dtype=np.float32)
            if head_index in head_map:
                stored_vector = head_map[head_index]
                vector[: int(stored_vector.shape[0])] = stored_vector
            covered_mass = float(pairwise_covered.get(group_key, {}).get(int(head_index), 0.0))
            if float(vector.sum()) > 1e-8:
                valid_vote_targets.append(int(vector.argmax()) + 1)
            layer_vectors.append(vector)
            layer_covered_values.append(float(covered_mass))

        layer_matrix = np.stack(layer_vectors, axis=0)
        layer_mean_vector = np.mean(layer_matrix, axis=0)
        pairwise_layer_vectors[group_key] = layer_mean_vector
        if float(layer_mean_vector.sum()) > 1e-8:
            layer_entropy = float(-(layer_mean_vector * np.log(np.clip(layer_mean_vector, 1e-8, None))).sum())
            layer_dominant = float(layer_mean_vector.max())
            sorted_layer_vector = np.sort(layer_mean_vector)[::-1]
            layer_margin = float(
                sorted_layer_vector[0] - (sorted_layer_vector[1] if int(sorted_layer_vector.shape[0]) > 1 else 0.0)
            )
        else:
            layer_entropy = float("nan")
            layer_dominant = float("nan")
            layer_margin = float("nan")
        vote_counts: Dict[int, int] = defaultdict(int)
        for target_candidate in valid_vote_targets:
            vote_counts[int(target_candidate)] += 1
        head_agreement = (
            float(max(vote_counts.values()) / max(1, len(valid_vote_targets)))
            if vote_counts
            else float("nan")
        )
        hard_vote_winner = (
            max(vote_counts.items(), key=lambda item: int(item[1]))[0]
            if vote_counts
            else (int(layer_mean_vector.argmax()) + 1 if float(layer_mean_vector.sum()) > 1e-8 else -1)
        )
        if hard_vote_winner > 0 and int(hard_vote_winner - 1) < int(layer_matrix.shape[1]):
            soft_head_vote_share = float(layer_matrix[:, int(hard_vote_winner - 1)].mean())
        else:
            soft_head_vote_share = float("nan")

        valid_distribution_vectors = [
            np.asarray(vector, dtype=np.float64)
            for vector in layer_vectors
            if float(np.asarray(vector, dtype=np.float64).sum()) > 1e-8
        ]
        if len(valid_distribution_vectors) <= 0:
            soft_head_agreement = float("nan")
        elif len(valid_distribution_vectors) == 1:
            soft_head_agreement = 1.0
        else:
            pairwise_cosines: List[float] = []
            for left_index in range(len(valid_distribution_vectors)):
                left_vector = valid_distribution_vectors[left_index]
                left_norm = float(np.linalg.norm(left_vector))
                if left_norm <= 1e-12:
                    continue
                for right_index in range(left_index + 1, len(valid_distribution_vectors)):
                    right_vector = valid_distribution_vectors[right_index]
                    right_norm = float(np.linalg.norm(right_vector))
                    if right_norm <= 1e-12:
                        continue
                    pairwise_cosines.append(
                        float(np.dot(left_vector, right_vector) / max(1e-12, left_norm * right_norm))
                    )
            soft_head_agreement = float(_mean_wan21_t2v_finite(pairwise_cosines))
        pairwise_layer_metrics[group_key] = {
            "covered_mass": float(_mean_wan21_t2v_finite(layer_covered_values)),
            "entropy": float(layer_entropy),
            "dominant_link_ratio": float(layer_dominant),
            "link_margin": float(layer_margin),
            "head_agreement": float(head_agreement),
            "soft_head_vote_share": float(soft_head_vote_share),
            "soft_head_agreement": float(soft_head_agreement),
        }
    return pairwise_layer_vectors, pairwise_layer_metrics


def _build_wan21_t2v_self_attention_pairwise_layer_value_vectors(
    pairwise_rows: Sequence[Dict[str, object]],
    value_key: str,
) -> Dict[Tuple[int, int, int], np.ndarray]:
    """Average one pairwise scalar field across selected heads on the shared head-mean partition."""
    selected_heads: List[int] = sorted({int(row["head"]) for row in pairwise_rows})
    per_group_values: Dict[Tuple[int, int, int], Dict[int, np.ndarray]] = defaultdict(dict)
    for row in pairwise_rows:
        query_frame = int(row["query_frame"])
        query_candidate = int(row["query_candidate"])
        key_frame = int(row["key_frame"])
        key_candidate = int(row["key_candidate"])
        key_candidate_count = int(row["key_candidate_count"])
        head_index = int(row["head"])
        group_key = (int(query_frame), int(query_candidate), int(key_frame))
        if head_index not in per_group_values[group_key]:
            per_group_values[group_key][head_index] = np.zeros((int(key_candidate_count),), dtype=np.float32)
        per_group_values[group_key][head_index][int(key_candidate - 1)] = np.float32(
            _safe_wan21_t2v_float(row.get(value_key, 0.0), default=0.0)
        )
    group_mean_vectors: Dict[Tuple[int, int, int], np.ndarray] = {}
    for group_key, head_map in per_group_values.items():
        key_candidate_count = max(int(vector.shape[0]) for vector in head_map.values())
        layer_vectors: List[np.ndarray] = []
        for head_index in selected_heads:
            vector = np.zeros((int(key_candidate_count),), dtype=np.float32)
            if head_index in head_map:
                stored_vector = head_map[head_index]
                vector[: int(stored_vector.shape[0])] = stored_vector
            layer_vectors.append(vector)
        group_mean_vectors[group_key] = np.mean(np.stack(layer_vectors, axis=0), axis=0)
    return group_mean_vectors


def _build_wan21_t2v_self_attention_observation_order(
    candidate_feature_rows: Sequence[Dict[str, object]],
) -> List[Tuple[int, int]]:
    """Return the ordered `(step, layer)` observation list."""
    return sorted({(int(row["step"]), int(row["layer"])) for row in candidate_feature_rows})


def _build_wan21_t2v_self_attention_feature_gap_rows(
    candidate_feature_rows: Sequence[Dict[str, object]],
    feature_names: Sequence[str],
) -> List[Dict[str, object]]:
    """Convert candidate-level rows into winner/loser feature traces."""
    observation_order = _build_wan21_t2v_self_attention_observation_order(candidate_feature_rows)
    observation_to_index = {
        (int(step), int(layer)): int(index)
        for index, (step, layer) in enumerate(observation_order)
    }
    rows_by_step_layer_frame: Dict[Tuple[int, int, int], List[Dict[str, object]]] = defaultdict(list)
    for row in candidate_feature_rows:
        rows_by_step_layer_frame[(int(row["step"]), int(row["layer"]), int(row["frame"]))].append(row)

    gap_rows: List[Dict[str, object]] = []
    for (step, layer, frame_index), group_rows in sorted(rows_by_step_layer_frame.items()):
        winner_rows = [row for row in group_rows if int(row.get("is_winner_aligned", 0)) == 1]
        loser_rows = [row for row in group_rows if int(row.get("is_strongest_loser", 0)) == 1]
        if not winner_rows or not loser_rows:
            continue
        winner_row = winner_rows[0]
        loser_row = loser_rows[0]
        observation_index = int(observation_to_index[(int(step), int(layer))])
        for feature_name in feature_names:
            winner_value = _safe_wan21_t2v_float(winner_row.get(feature_name, float("nan")))
            loser_value = _safe_wan21_t2v_float(loser_row.get(feature_name, float("nan")))
            gap_rows.append({
                "step": int(step),
                "layer": int(layer),
                "observation_index": int(observation_index),
                "frame": int(frame_index),
                "feature": str(feature_name),
                "winner_value": float(winner_value),
                "loser_value": float(loser_value),
                "gap": (
                    float(winner_value - loser_value)
                    if math.isfinite(winner_value) and math.isfinite(loser_value)
                    else float("nan")
                ),
            })
    return gap_rows


def _build_wan21_t2v_self_attention_stepwise_gap_rows(
    gap_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Average winner-minus-loser gaps over frames for each `(feature, layer, step)`."""
    grouped: Dict[Tuple[str, int, int], List[float]] = defaultdict(list)
    for row in gap_rows:
        gap_value = _safe_wan21_t2v_float(row.get("gap", float("nan")))
        if not math.isfinite(gap_value):
            continue
        grouped[(str(row["feature"]), int(row["layer"]), int(row["step"]))].append(float(gap_value))
    out_rows: List[Dict[str, object]] = []
    for (feature_name, layer, step), values in sorted(grouped.items()):
        out_rows.append({
            "feature": str(feature_name),
            "layer": int(layer),
            "step": int(step),
            "mean_gap": float(_mean_wan21_t2v_finite(values)),
        })
    return out_rows


def _build_wan21_t2v_self_attention_layerwise_gap_rows(
    gap_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Average winner-minus-loser gaps over frames for each `(feature, step, layer)`."""
    grouped: Dict[Tuple[str, int, int], List[float]] = defaultdict(list)
    for row in gap_rows:
        gap_value = _safe_wan21_t2v_float(row.get("gap", float("nan")))
        if not math.isfinite(gap_value):
            continue
        grouped[(str(row["feature"]), int(row["step"]), int(row["layer"]))].append(float(gap_value))
    out_rows: List[Dict[str, object]] = []
    for (feature_name, step, layer), values in sorted(grouped.items()):
        out_rows.append({
            "feature": str(feature_name),
            "step": int(step),
            "layer": int(layer),
            "mean_gap": float(_mean_wan21_t2v_finite(values)),
        })
    return out_rows


def _sample_wan21_t2v_evenly_spaced_indices(total_count: int, max_count: int) -> List[int]:
    """Sample up to `max_count` approximately evenly spaced indices."""
    if total_count <= 0 or max_count <= 0:
        return []
    if total_count <= max_count:
        return list(range(total_count))
    samples = np.linspace(0, total_count - 1, num=max_count)
    return sorted({int(round(float(x))) for x in samples})


def _select_wan21_t2v_self_attention_representative_observation(
    feature_summary_rows: Sequence[Dict[str, object]],
    fallback_rows: Sequence[Dict[str, object]],
    feature_name: str = "global_chainability",
) -> Tuple[int, int]:
    """Pick one representative observation for qualitative figures."""
    target_rows = [
        row for row in feature_summary_rows
        if str(row.get("feature", "")) == str(feature_name)
    ]
    if target_rows:
        ranked = sorted(
            target_rows,
            key=lambda row: (
                -1e9 if not math.isfinite(_safe_wan21_t2v_float(row.get("auroc", float("nan"))))
                else float(row["auroc"]),
                -1e9 if not math.isfinite(_safe_wan21_t2v_float(row.get("winner_loser_gap", float("nan"))))
                else float(row["winner_loser_gap"]),
                int(row["step"]),
                int(row["layer"]),
            ),
            reverse=True,
        )
        return int(ranked[0]["step"]), int(ranked[0]["layer"])
    observation_order = _build_wan21_t2v_self_attention_observation_order(fallback_rows)
    if not observation_order:
        return (0, 0)
    return observation_order[-1]


def _trajectory_consensus_compute_candidate_feature_task(
    task: Tuple,
) -> Tuple[int, int, List[Dict[str, object]]]:
    """Worker task that converts one `(step, layer)` pairwise coupling group into candidate-level feature rows."""
    (
        step,
        layer,
        pairwise_rows,
        label_map_fhw_np,
        anchor_union_fhw_np,
        anchor_centers,
        proposal_rows,
        covered_mass_min,
    ) = task

    label_map_fhw = np.asarray(label_map_fhw_np, dtype=np.int64)
    anchor_union_fhw = np.asarray(anchor_union_fhw_np, dtype=np.bool_)
    frame_count = int(label_map_fhw.shape[0])
    candidate_counts = [int(label_map_fhw[frame_index].max()) for frame_index in range(frame_count)]

    proposal_weights: Dict[Tuple[int, int], float] = {}
    proposal_mean_weight_by_frame: Dict[int, Dict[int, float]] = defaultdict(dict)
    proposal_vote_share_by_frame: Dict[int, Dict[int, float]] = defaultdict(dict)
    proposal_agreement_by_frame: Dict[int, float] = {}
    if proposal_rows:
        per_frame_head_weights: Dict[int, Dict[int, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
        for row in proposal_rows:
            frame_index = int(row["frame"])
            candidate_index = int(row["candidate_index"])
            head_index = int(row["head"])
            candidate_weight = _safe_wan21_t2v_float(row.get("candidate_weight", float("nan")))
            per_frame_head_weights[frame_index][head_index][candidate_index] = candidate_weight

        for frame_index, head_payload in per_frame_head_weights.items():
            candidate_weight_lists: Dict[int, List[float]] = defaultdict(list)
            vote_counts: Dict[int, int] = defaultdict(int)
            for head_index, candidate_map in head_payload.items():
                del head_index
                if not candidate_map:
                    continue
                best_candidate = max(candidate_map.items(), key=lambda item: float(item[1]))[0]
                vote_counts[int(best_candidate)] += 1
                for candidate_index, weight_value in candidate_map.items():
                    candidate_weight_lists[int(candidate_index)].append(float(weight_value))

            candidate_mean_weights = {
                int(candidate_index): float(sum(weight_values) / len(weight_values))
                for candidate_index, weight_values in candidate_weight_lists.items()
                if weight_values
            }
            total_mean_weight = float(sum(candidate_mean_weights.values()))
            for candidate_index, mean_weight in candidate_mean_weights.items():
                proposal_mean_weight_by_frame[int(frame_index)][int(candidate_index)] = float(mean_weight)
                proposal_weights[(int(frame_index), int(candidate_index))] = float(
                    mean_weight / max(1e-8, total_mean_weight)
                )
            total_votes = float(sum(vote_counts.values()))
            if total_votes > 0.0:
                for candidate_index, vote_count in vote_counts.items():
                    proposal_vote_share_by_frame[int(frame_index)][int(candidate_index)] = float(vote_count / total_votes)
                proposal_agreement_by_frame[int(frame_index)] = float(
                    max(float(vote_count) / total_votes for vote_count in vote_counts.values())
                )

    pairwise_layer_vectors, pairwise_layer_metrics = _build_wan21_t2v_self_attention_pairwise_layer_stats(
        pairwise_rows
    )
    raw_pairwise_layer_vectors = _build_wan21_t2v_self_attention_pairwise_layer_value_vectors(
        pairwise_rows,
        value_key="raw_coupling",
    )
    frame_to_frame_raw_support_vectors: Dict[Tuple[int, int], np.ndarray] = {}
    for source_frame in range(frame_count):
        for target_frame in range(frame_count):
            if int(source_frame) == int(target_frame):
                continue
            support_vectors: List[np.ndarray] = []
            for source_candidate in range(1, int(candidate_counts[source_frame]) + 1):
                support_vector = raw_pairwise_layer_vectors.get(
                    (int(source_frame), int(source_candidate), int(target_frame))
                )
                if support_vector is None:
                    continue
                support_vectors.append(np.asarray(support_vector, dtype=np.float64))
            if support_vectors:
                frame_to_frame_raw_support_vectors[(int(source_frame), int(target_frame))] = np.sum(
                    np.stack(support_vectors, axis=0),
                    axis=0,
                )

    candidate_feature_rows: List[Dict[str, object]] = []

    def _aggregate_pairwise_metric(
        query_frame: int,
        query_candidate: int,
        metric_name: str,
        frame_indices: Sequence[int],
        filtered_only: bool,
    ) -> float:
        values: List[float] = []
        for key_frame in frame_indices:
            layer_key = (int(query_frame), int(query_candidate), int(key_frame))
            metrics = pairwise_layer_metrics.get(layer_key)
            if metrics is None:
                continue
            if bool(filtered_only) and float(metrics.get("covered_mass", float("nan"))) < float(covered_mass_min):
                continue
            values.append(float(metrics.get(metric_name, float("nan"))))
        return _mean_wan21_t2v_finite(values)

    def _candidate_center(mask_hw: np.ndarray) -> Tuple[float, float]:
        points = np.argwhere(mask_hw)
        if int(points.size) <= 0:
            return (float("nan"), float("nan"))
        return (float(points[:, 0].mean()), float(points[:, 1].mean()))

    def _aggregate_incoming_support(
        source_frames: Sequence[int],
        target_frame: int,
        target_candidate: int,
    ) -> float:
        values: List[float] = []
        for source_frame in source_frames:
            support_vector = frame_to_frame_raw_support_vectors.get((int(source_frame), int(target_frame)))
            if support_vector is None or int(target_candidate - 1) >= int(support_vector.shape[0]):
                continue
            values.append(float(support_vector[int(target_candidate - 1)]))
        return _mean_wan21_t2v_finite(values)

    def _aggregate_incoming_preference_share(
        source_frames: Sequence[int],
        target_frame: int,
        target_candidate: int,
    ) -> float:
        values: List[float] = []
        for source_frame in source_frames:
            support_vector = frame_to_frame_raw_support_vectors.get((int(source_frame), int(target_frame)))
            if support_vector is None or int(target_candidate - 1) >= int(support_vector.shape[0]):
                continue
            total_support = float(np.asarray(support_vector, dtype=np.float64).sum())
            if total_support <= 1e-12:
                continue
            values.append(float(support_vector[int(target_candidate - 1)] / total_support))
        return _mean_wan21_t2v_finite(values)

    def _aggregate_incoming_vote_share(
        source_frames: Sequence[int],
        target_frame: int,
        target_candidate: int,
    ) -> float:
        votes: List[float] = []
        for source_frame in source_frames:
            support_vector = frame_to_frame_raw_support_vectors.get((int(source_frame), int(target_frame)))
            if support_vector is None or int(support_vector.shape[0]) <= 0:
                continue
            total_support = float(np.asarray(support_vector, dtype=np.float64).sum())
            if total_support <= 1e-12:
                continue
            voted_candidate = int(np.asarray(support_vector, dtype=np.float64).argmax()) + 1
            votes.append(1.0 if int(voted_candidate) == int(target_candidate) else 0.0)
        return _mean_wan21_t2v_finite(votes)

    for frame_index in range(frame_count):
        frame_candidate_count = int(candidate_counts[frame_index])
        if frame_candidate_count <= 0:
            continue

        anchor_center_y, anchor_center_x = anchor_centers[int(frame_index)]
        candidate_iou_scores: Dict[int, float] = {}
        candidate_rows_in_frame: List[Dict[str, object]] = []

        global_in_frames = [int(g) for g in range(frame_count) if int(g) < int(frame_index)]
        global_out_frames = [int(g) for g in range(frame_count) if int(g) > int(frame_index)]
        local_frames = [int(g) for g in range(frame_count) if abs(int(g) - int(frame_index)) == 1]
        incoming_local_source_frames = [int(g) for g in range(frame_count) if abs(int(g) - int(frame_index)) == 1]
        incoming_global_source_frames = [int(g) for g in range(frame_count) if int(g) != int(frame_index)]

        for candidate_index in range(1, frame_candidate_count + 1):
            candidate_mask = (label_map_fhw[frame_index] == int(candidate_index))
            candidate_area = int(candidate_mask.sum())
            centroid_y, centroid_x = _candidate_center(candidate_mask)
            anchor_union_mask = anchor_union_fhw[frame_index]
            intersection = float(np.logical_and(candidate_mask, anchor_union_mask).sum())
            union = float(np.logical_or(candidate_mask, anchor_union_mask).sum())
            anchor_iou = float(intersection / max(1.0, union))
            candidate_iou_scores[int(candidate_index)] = float(anchor_iou)
            if math.isfinite(anchor_center_y) and math.isfinite(anchor_center_x) and math.isfinite(centroid_y) and math.isfinite(centroid_x):
                anchor_center_l2 = float(math.sqrt((centroid_y - anchor_center_y) ** 2 + (centroid_x - anchor_center_x) ** 2))
            else:
                anchor_center_l2 = float("nan")

            local_avg_covered_mass = _aggregate_pairwise_metric(frame_index, candidate_index, "covered_mass", local_frames, False)
            local_avg_covered_mass_filtered = _aggregate_pairwise_metric(frame_index, candidate_index, "covered_mass", local_frames, True)
            global_avg_covered_mass = _aggregate_pairwise_metric(frame_index, candidate_index, "covered_mass", global_in_frames + global_out_frames, False)
            global_avg_covered_mass_filtered = _aggregate_pairwise_metric(frame_index, candidate_index, "covered_mass", global_in_frames + global_out_frames, True)
            local_entropy = _aggregate_pairwise_metric(frame_index, candidate_index, "entropy", local_frames, False)
            local_entropy_filtered = _aggregate_pairwise_metric(frame_index, candidate_index, "entropy", local_frames, True)
            global_entropy = _aggregate_pairwise_metric(frame_index, candidate_index, "entropy", global_in_frames + global_out_frames, False)
            global_entropy_filtered = _aggregate_pairwise_metric(frame_index, candidate_index, "entropy", global_in_frames + global_out_frames, True)
            local_dominant = _aggregate_pairwise_metric(frame_index, candidate_index, "dominant_link_ratio", local_frames, False)
            local_dominant_filtered = _aggregate_pairwise_metric(frame_index, candidate_index, "dominant_link_ratio", local_frames, True)
            global_dominant = _aggregate_pairwise_metric(frame_index, candidate_index, "dominant_link_ratio", global_in_frames + global_out_frames, False)
            global_dominant_filtered = _aggregate_pairwise_metric(frame_index, candidate_index, "dominant_link_ratio", global_in_frames + global_out_frames, True)
            local_link_margin = _aggregate_pairwise_metric(frame_index, candidate_index, "link_margin", local_frames, False)
            local_link_margin_filtered = _aggregate_pairwise_metric(frame_index, candidate_index, "link_margin", local_frames, True)
            global_link_margin = _aggregate_pairwise_metric(frame_index, candidate_index, "link_margin", global_in_frames + global_out_frames, False)
            global_link_margin_filtered = _aggregate_pairwise_metric(frame_index, candidate_index, "link_margin", global_in_frames + global_out_frames, True)
            local_head_agreement = _aggregate_pairwise_metric(frame_index, candidate_index, "head_agreement", local_frames, False)
            local_head_agreement_filtered = _aggregate_pairwise_metric(frame_index, candidate_index, "head_agreement", local_frames, True)
            global_head_agreement = _aggregate_pairwise_metric(frame_index, candidate_index, "head_agreement", global_in_frames + global_out_frames, False)
            global_head_agreement_filtered = _aggregate_pairwise_metric(frame_index, candidate_index, "head_agreement", global_in_frames + global_out_frames, True)
            local_soft_head_vote_share = _aggregate_pairwise_metric(frame_index, candidate_index, "soft_head_vote_share", local_frames, False)
            local_soft_head_vote_share_filtered = _aggregate_pairwise_metric(frame_index, candidate_index, "soft_head_vote_share", local_frames, True)
            global_soft_head_vote_share = _aggregate_pairwise_metric(frame_index, candidate_index, "soft_head_vote_share", global_in_frames + global_out_frames, False)
            global_soft_head_vote_share_filtered = _aggregate_pairwise_metric(frame_index, candidate_index, "soft_head_vote_share", global_in_frames + global_out_frames, True)
            local_soft_head_agreement = _aggregate_pairwise_metric(frame_index, candidate_index, "soft_head_agreement", local_frames, False)
            local_soft_head_agreement_filtered = _aggregate_pairwise_metric(frame_index, candidate_index, "soft_head_agreement", local_frames, True)
            global_soft_head_agreement = _aggregate_pairwise_metric(frame_index, candidate_index, "soft_head_agreement", global_in_frames + global_out_frames, False)
            global_soft_head_agreement_filtered = _aggregate_pairwise_metric(frame_index, candidate_index, "soft_head_agreement", global_in_frames + global_out_frames, True)

            local_in = float("nan")
            local_out = float("nan")
            if int(frame_index) - 1 >= 0:
                prev_frame = int(frame_index) - 1
                incoming_values = []
                for prev_candidate in range(1, int(candidate_counts[prev_frame]) + 1):
                    incoming_vector = raw_pairwise_layer_vectors.get((int(prev_frame), int(prev_candidate), int(frame_index)))
                    if incoming_vector is None or int(candidate_index - 1) >= int(incoming_vector.shape[0]):
                        continue
                    incoming_values.append(float(incoming_vector[int(candidate_index - 1)]))
                local_in = float(sum(incoming_values)) if incoming_values else float("nan")
            if int(frame_index) + 1 < frame_count:
                next_frame = int(frame_index) + 1
                outgoing_vector = raw_pairwise_layer_vectors.get((int(frame_index), int(candidate_index), int(next_frame)))
                if outgoing_vector is not None:
                    local_out = float(outgoing_vector.sum())

            def _global_direction(frames: Sequence[int], incoming: bool) -> float:
                direction_values: List[float] = []
                for other_frame in frames:
                    if incoming:
                        incoming_values = []
                        for other_candidate in range(1, int(candidate_counts[int(other_frame)]) + 1):
                            incoming_vector = raw_pairwise_layer_vectors.get((int(other_frame), int(other_candidate), int(frame_index)))
                            if incoming_vector is None or int(candidate_index - 1) >= int(incoming_vector.shape[0]):
                                continue
                            incoming_values.append(float(incoming_vector[int(candidate_index - 1)]))
                        if incoming_values:
                            direction_values.append(float(sum(incoming_values)))
                    else:
                        outgoing_vector = raw_pairwise_layer_vectors.get((int(frame_index), int(candidate_index), int(other_frame)))
                        if outgoing_vector is not None:
                            direction_values.append(float(outgoing_vector.sum()))
                return _mean_wan21_t2v_finite(direction_values)

            global_in = _global_direction(global_in_frames, incoming=True)
            global_out = _global_direction(global_out_frames, incoming=False)
            local_compatibility = float(
                sum(value for value in [local_in, local_out] if math.isfinite(value))
            ) if any(math.isfinite(value) for value in [local_in, local_out]) else float("nan")
            local_chainability = float(min(local_in, local_out)) if math.isfinite(local_in) and math.isfinite(local_out) else float("nan")
            global_compatibility = float(
                sum(value for value in [global_in, global_out] if math.isfinite(value))
            ) if any(math.isfinite(value) for value in [global_in, global_out]) else float("nan")
            global_chainability = float(min(global_in, global_out)) if math.isfinite(global_in) and math.isfinite(global_out) else float("nan")
            local_incoming_support = _aggregate_incoming_support(
                source_frames=incoming_local_source_frames,
                target_frame=int(frame_index),
                target_candidate=int(candidate_index),
            )
            global_incoming_support = _aggregate_incoming_support(
                source_frames=incoming_global_source_frames,
                target_frame=int(frame_index),
                target_candidate=int(candidate_index),
            )
            local_incoming_preference_share = _aggregate_incoming_preference_share(
                source_frames=incoming_local_source_frames,
                target_frame=int(frame_index),
                target_candidate=int(candidate_index),
            )
            global_incoming_preference_share = _aggregate_incoming_preference_share(
                source_frames=incoming_global_source_frames,
                target_frame=int(frame_index),
                target_candidate=int(candidate_index),
            )
            local_incoming_vote_share = _aggregate_incoming_vote_share(
                source_frames=incoming_local_source_frames,
                target_frame=int(frame_index),
                target_candidate=int(candidate_index),
            )
            global_incoming_vote_share = _aggregate_incoming_vote_share(
                source_frames=incoming_global_source_frames,
                target_frame=int(frame_index),
                target_candidate=int(candidate_index),
            )

            def _mutual_consistency_max(key_frame: int) -> float:
                forward_vector = pairwise_layer_vectors.get((int(frame_index), int(candidate_index), int(key_frame)))
                if forward_vector is None:
                    return float("nan")
                reciprocal_values: List[float] = []
                for key_candidate in range(1, int(candidate_counts[int(key_frame)]) + 1):
                    reverse_vector = pairwise_layer_vectors.get((int(key_frame), int(key_candidate), int(frame_index)))
                    if reverse_vector is None or int(candidate_index - 1) >= int(reverse_vector.shape[0]):
                        continue
                    if int(key_candidate - 1) >= int(forward_vector.shape[0]):
                        continue
                    reciprocal_values.append(
                        float(forward_vector[int(key_candidate - 1)] * reverse_vector[int(candidate_index - 1)])
                    )
                return float(max(reciprocal_values)) if reciprocal_values else float("nan")

            local_mc_max = _mean_wan21_t2v_finite([_mutual_consistency_max(key_frame) for key_frame in local_frames])
            global_mc_max = _mean_wan21_t2v_finite([
                _mutual_consistency_max(key_frame)
                for key_frame in (global_in_frames + global_out_frames)
            ])

            candidate_rows_in_frame.append({
                "step": int(step),
                "layer": int(layer),
                "frame": int(frame_index),
                "candidate_index": int(candidate_index),
                "candidate_count": int(frame_candidate_count),
                "candidate_area": int(candidate_area),
                "centroid_y": float(centroid_y),
                "centroid_x": float(centroid_x),
                "anchor_iou": float(anchor_iou),
                "anchor_distance": float(anchor_center_l2),
                "anchor_center_l2": float(anchor_center_l2),
                "proposal_pi": float(proposal_weights.get((int(frame_index), int(candidate_index)), 0.0)),
                "proposal_vote_share": float(
                    proposal_vote_share_by_frame.get(int(frame_index), {}).get(int(candidate_index), 0.0)
                ),
                "proposal_agreement_frame": float(proposal_agreement_by_frame.get(int(frame_index), float("nan"))),
                "local_avg_covered_mass": float(local_avg_covered_mass),
                "local_avg_covered_mass_filtered": float(local_avg_covered_mass_filtered),
                "global_avg_covered_mass": float(global_avg_covered_mass),
                "global_avg_covered_mass_filtered": float(global_avg_covered_mass_filtered),
                "local_entropy": float(local_entropy),
                "local_entropy_filtered": float(local_entropy_filtered),
                "global_entropy": float(global_entropy),
                "global_entropy_filtered": float(global_entropy_filtered),
                "local_dominant_link_ratio": float(local_dominant),
                "local_dominant_link_ratio_filtered": float(local_dominant_filtered),
                "global_dominant_link_ratio": float(global_dominant),
                "global_dominant_link_ratio_filtered": float(global_dominant_filtered),
                "local_link_margin": float(local_link_margin),
                "local_link_margin_filtered": float(local_link_margin_filtered),
                "global_link_margin": float(global_link_margin),
                "global_link_margin_filtered": float(global_link_margin_filtered),
                "local_head_agreement": float(local_head_agreement),
                "local_head_agreement_filtered": float(local_head_agreement_filtered),
                "global_head_agreement": float(global_head_agreement),
                "global_head_agreement_filtered": float(global_head_agreement_filtered),
                "local_soft_head_vote_share": float(local_soft_head_vote_share),
                "local_soft_head_vote_share_filtered": float(local_soft_head_vote_share_filtered),
                "global_soft_head_vote_share": float(global_soft_head_vote_share),
                "global_soft_head_vote_share_filtered": float(global_soft_head_vote_share_filtered),
                "local_soft_head_agreement": float(local_soft_head_agreement),
                "local_soft_head_agreement_filtered": float(local_soft_head_agreement_filtered),
                "global_soft_head_agreement": float(global_soft_head_agreement),
                "global_soft_head_agreement_filtered": float(global_soft_head_agreement_filtered),
                "local_compatibility": float(local_compatibility),
                "local_chainability": float(local_chainability),
                "global_compatibility": float(global_compatibility),
                "global_chainability": float(global_chainability),
                "local_incoming_support": float(local_incoming_support),
                "global_incoming_support": float(global_incoming_support),
                "local_incoming_preference_share": float(local_incoming_preference_share),
                "global_incoming_preference_share": float(global_incoming_preference_share),
                "local_incoming_vote_share": float(local_incoming_vote_share),
                "global_incoming_vote_share": float(global_incoming_vote_share),
                "local_mutual_consistency": float(local_mc_max),
                "global_mutual_consistency": float(global_mc_max),
            })

        if candidate_rows_in_frame:
            winner_candidate = max(candidate_rows_in_frame, key=lambda row: float(row["anchor_iou"]))["candidate_index"]
            loser_candidates = [
                row
                for row in candidate_rows_in_frame
                if int(row["candidate_index"]) != int(winner_candidate)
            ]
            strongest_loser = None
            if loser_candidates:
                strongest_loser = max(
                    loser_candidates,
                    key=lambda row: (
                        -1e9 if not math.isfinite(_safe_wan21_t2v_float(row.get("proposal_pi", float("nan"))))
                        else float(row["proposal_pi"])
                    ),
                )["candidate_index"]
            for row in candidate_rows_in_frame:
                row["is_winner_aligned"] = int(int(row["candidate_index"]) == int(winner_candidate))
                row["is_strongest_loser"] = int(
                    strongest_loser is not None and int(row["candidate_index"]) == int(strongest_loser)
                )
                candidate_feature_rows.append(row)

    return int(step), int(layer), candidate_feature_rows


def _summarize_wan21_t2v_self_attention_candidate_features(
    candidate_feature_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Build per-step-layer feature-summary rows from candidate-level features."""
    feature_names = [
        "proposal_pi",
        "proposal_vote_share",
        "local_avg_covered_mass",
        "local_avg_covered_mass_filtered",
        "global_avg_covered_mass",
        "global_avg_covered_mass_filtered",
        "local_entropy",
        "local_entropy_filtered",
        "global_entropy",
        "global_entropy_filtered",
        "local_dominant_link_ratio",
        "local_dominant_link_ratio_filtered",
        "global_dominant_link_ratio",
        "global_dominant_link_ratio_filtered",
        "local_link_margin",
        "local_link_margin_filtered",
        "global_link_margin",
        "global_link_margin_filtered",
        "local_head_agreement",
        "local_head_agreement_filtered",
        "global_head_agreement",
        "global_head_agreement_filtered",
        "local_soft_head_vote_share",
        "local_soft_head_vote_share_filtered",
        "global_soft_head_vote_share",
        "global_soft_head_vote_share_filtered",
        "local_soft_head_agreement",
        "local_soft_head_agreement_filtered",
        "global_soft_head_agreement",
        "global_soft_head_agreement_filtered",
        "local_compatibility",
        "global_compatibility",
        "local_chainability",
        "global_chainability",
        "local_incoming_support",
        "global_incoming_support",
        "local_incoming_preference_share",
        "global_incoming_preference_share",
        "local_incoming_vote_share",
        "global_incoming_vote_share",
        "local_mutual_consistency",
        "global_mutual_consistency",
    ]
    rows_by_step_layer: Dict[Tuple[int, int], List[Dict[str, object]]] = defaultdict(list)
    for row in candidate_feature_rows:
        rows_by_step_layer[(int(row["step"]), int(row["layer"]))].append(row)

    summary_rows: List[Dict[str, object]] = []
    for (step, layer), group_rows in sorted(rows_by_step_layer.items()):
        for feature_name in feature_names:
            labels = [int(row.get("is_winner_aligned", 0)) for row in group_rows]
            scores = [_safe_wan21_t2v_float(row.get(feature_name, float("nan"))) for row in group_rows]
            auroc_value = _compute_wan21_t2v_binary_auroc(labels, scores)
            anchor_iou_values = [_safe_wan21_t2v_float(row.get("anchor_iou", float("nan"))) for row in group_rows]
            anchor_iou_correlation = _compute_wan21_t2v_pearson_correlation(scores, anchor_iou_values)
            anchor_distance_values = [_safe_wan21_t2v_float(row.get("anchor_distance", float("nan"))) for row in group_rows]
            anchor_distance_correlation = _compute_wan21_t2v_pearson_correlation(scores, anchor_distance_values)

            rows_by_frame: Dict[int, List[Dict[str, object]]] = defaultdict(list)
            for row in group_rows:
                rows_by_frame[int(row["frame"])].append(row)
            gap_values: List[float] = []
            for frame_index, frame_rows in rows_by_frame.items():
                del frame_index
                winner_rows = [row for row in frame_rows if int(row.get("is_winner_aligned", 0)) == 1]
                loser_rows = [row for row in frame_rows if int(row.get("is_strongest_loser", 0)) == 1]
                if not winner_rows or not loser_rows:
                    continue
                winner_value = _safe_wan21_t2v_float(winner_rows[0].get(feature_name, float("nan")))
                loser_value = _safe_wan21_t2v_float(loser_rows[0].get(feature_name, float("nan")))
                if math.isfinite(winner_value) and math.isfinite(loser_value):
                    gap_values.append(float(winner_value - loser_value))
            summary_rows.append({
                "step": int(step),
                "layer": int(layer),
                "feature": str(feature_name),
                "winner_loser_gap": float(_mean_wan21_t2v_finite(gap_values)),
                "auroc": float(auroc_value),
                "anchor_iou_correlation": float(anchor_iou_correlation),
                "anchor_distance_correlation": float(anchor_distance_correlation),
            })
    return summary_rows


def _summarize_wan21_t2v_self_attention_temporal_precedence(
    candidate_feature_rows: Sequence[Dict[str, object]],
    persistence_window: int,
) -> List[Dict[str, object]]:
    """Summarize temporal precedence from winner-minus-loser feature gaps."""
    feature_names = [
        "proposal_pi",
        "proposal_vote_share",
        "local_compatibility",
        "global_compatibility",
        "local_chainability",
        "global_chainability",
        "local_incoming_support",
        "global_incoming_support",
        "local_incoming_preference_share",
        "global_incoming_preference_share",
        "local_incoming_vote_share",
        "global_incoming_vote_share",
        "local_mutual_consistency",
        "global_mutual_consistency",
        "local_head_agreement",
        "global_head_agreement",
        "local_soft_head_vote_share",
        "global_soft_head_vote_share",
        "local_soft_head_agreement",
        "global_soft_head_agreement",
    ]
    observation_order = sorted(
        {
            (int(row["step"]), int(row["layer"]))
            for row in candidate_feature_rows
        }
    )
    observation_to_index = {
        (int(step), int(layer)): int(index)
        for index, (step, layer) in enumerate(observation_order)
    }
    rows_by_step_layer_frame: Dict[Tuple[int, int, int], List[Dict[str, object]]] = defaultdict(list)
    for row in candidate_feature_rows:
        rows_by_step_layer_frame[(int(row["step"]), int(row["layer"]), int(row["frame"]))].append(row)

    temporal_rows: List[Dict[str, object]] = []
    frame_ids = sorted({int(row["frame"]) for row in candidate_feature_rows})
    for feature_name in feature_names:
        for frame_index in frame_ids:
            deltas: List[float] = []
            observation_trace: List[Tuple[int, int]] = []
            for step, layer in observation_order:
                group_rows = rows_by_step_layer_frame.get((int(step), int(layer), int(frame_index)), [])
                winner_rows = [row for row in group_rows if int(row.get("is_winner_aligned", 0)) == 1]
                loser_rows = [row for row in group_rows if int(row.get("is_strongest_loser", 0)) == 1]
                if not winner_rows or not loser_rows:
                    deltas.append(float("nan"))
                    observation_trace.append((int(step), int(layer)))
                    continue
                winner_value = _safe_wan21_t2v_float(winner_rows[0].get(feature_name, float("nan")))
                loser_value = _safe_wan21_t2v_float(loser_rows[0].get(feature_name, float("nan")))
                deltas.append(
                    float(winner_value - loser_value)
                    if math.isfinite(winner_value) and math.isfinite(loser_value)
                    else float("nan")
                )
                observation_trace.append((int(step), int(layer)))

            precedence_index = -1
            precedence_step = -1
            precedence_layer = -1
            persistence = max(1, int(persistence_window))
            for start_index in range(len(deltas)):
                window = deltas[start_index:start_index + persistence]
                if len(window) < persistence:
                    continue
                if all(math.isfinite(value) and value > 0.0 for value in window):
                    precedence_index = int(start_index)
                    precedence_step, precedence_layer = observation_trace[start_index]
                    break
            temporal_rows.append({
                "feature": str(feature_name),
                "frame": int(frame_index),
                "precedence_observation_index": int(precedence_index),
                "precedence_step": int(precedence_step),
                "precedence_layer": int(precedence_layer),
            })
    return temporal_rows


def _build_wan21_t2v_self_attention_signed_offset_rows(
    pairwise_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Aggregate layer-mean pairwise metrics by signed frame offset."""
    rows_by_step_layer: Dict[Tuple[int, int], List[Dict[str, object]]] = defaultdict(list)
    for row in pairwise_rows:
        rows_by_step_layer[(int(row["step"]), int(row["layer"]))].append(row)

    aggregated_rows: List[Dict[str, object]] = []
    for (step, layer), observation_rows in sorted(rows_by_step_layer.items()):
        _, pairwise_layer_metrics = _build_wan21_t2v_self_attention_pairwise_layer_stats(observation_rows)
        by_offset: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        for (query_frame, query_candidate, key_frame), metric_map in pairwise_layer_metrics.items():
            del query_candidate
            signed_offset = int(key_frame) - int(query_frame)
            for metric_name, metric_value in metric_map.items():
                by_offset[int(signed_offset)][str(metric_name)].append(float(metric_value))
        for signed_offset, metric_lists in sorted(by_offset.items()):
            aggregated_rows.append({
                "step": int(step),
                "layer": int(layer),
                "offset": int(signed_offset),
                "covered_mass": float(_mean_wan21_t2v_finite(metric_lists.get("covered_mass", []))),
                "entropy": float(_mean_wan21_t2v_finite(metric_lists.get("entropy", []))),
                "dominant_link_ratio": float(
                    _mean_wan21_t2v_finite(metric_lists.get("dominant_link_ratio", []))
                ),
                "link_margin": float(_mean_wan21_t2v_finite(metric_lists.get("link_margin", []))),
                "head_agreement": float(_mean_wan21_t2v_finite(metric_lists.get("head_agreement", []))),
                "soft_head_vote_share": float(_mean_wan21_t2v_finite(metric_lists.get("soft_head_vote_share", []))),
                "soft_head_agreement": float(_mean_wan21_t2v_finite(metric_lists.get("soft_head_agreement", []))),
            })
    return aggregated_rows


def _plot_wan21_t2v_self_attention_candidate_score_overlay(
    label_map_fhw: torch.Tensor,
    feature_rows: Sequence[Dict[str, object]],
    feature_name: str,
    anchor_mode: str,
    save_file: str,
    title: str,
    frame_indices: Sequence[int],
    video_frame_labels: Optional[Sequence[int]] = None,
) -> str:
    """Render candidate masks colored by one candidate-level scalar."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mpl_colors

    if not frame_indices or not feature_rows:
        return ""
    if video_frame_labels is None:
        video_frame_labels = [int(frame_index) for frame_index in frame_indices]

    label_map_fhw = torch.as_tensor(label_map_fhw).detach().cpu().to(torch.int64)
    rows_by_frame: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for row in feature_rows:
        rows_by_frame[int(row["frame"])].append(row)
    finite_values = [
        _safe_wan21_t2v_float(row.get(feature_name, float("nan")))
        for row in feature_rows
        if math.isfinite(_safe_wan21_t2v_float(row.get(feature_name, float("nan"))))
    ]
    if not finite_values:
        return ""
    vmin, vmax = _trajectory_consensus_self_attention_feature_plot_range(
        feature_name=str(feature_name),
        finite_values=finite_values,
    )
    if abs(vmax - vmin) < 1e-8:
        vmax = vmin + 1e-8

    num_frames = len(frame_indices)
    fig, axes = plt.subplots(
        1,
        num_frames,
        figsize=(max(4.4 * num_frames, 12.0), 5.2),
        gridspec_kw={"wspace": 0.03},
    )
    if num_frames == 1:
        axes = [axes]
    image = None
    cmap = plt.get_cmap("viridis")
    value_norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax)
    text_pad_x = 2.2
    right_margin = 7.0
    for axis, frame_index, video_frame_label in zip(axes, frame_indices, video_frame_labels):
        label_frame = label_map_fhw[int(frame_index)].numpy()
        canvas = np.full(label_frame.shape, np.nan, dtype=np.float32)
        feature_by_candidate: Dict[int, float] = {}
        display_payloads: List[Tuple[int, np.ndarray, np.ndarray, float]] = []
        frame_rows = rows_by_frame.get(int(frame_index), [])
        winner_candidate, loser_candidate = _resolve_wan21_t2v_overlay_anchor_role_candidates(
            frame_rows=frame_rows,
            anchor_mode=str(anchor_mode),
        )
        for row in frame_rows:
            candidate_index = int(row["candidate_index"])
            feature_value = _safe_wan21_t2v_float(row.get(feature_name, float("nan")))
            if (not math.isfinite(feature_value)) and str(feature_name) in {"proposal_pi", "proposal_vote_share"}:
                feature_value = 0.0
            feature_by_candidate[int(candidate_index)] = float(feature_value)

        candidate_count = int(label_frame.max())
        for candidate_index in range(1, candidate_count + 1):
            candidate_mask = (label_frame == int(candidate_index))
            if not np.any(candidate_mask):
                continue
            display_mask = _trajectory_consensus_prepare_display_candidate_mask(candidate_mask)
            points = np.argwhere(display_mask)
            feature_value = _safe_wan21_t2v_float(feature_by_candidate.get(int(candidate_index), float("nan")))
            if math.isfinite(feature_value):
                canvas[display_mask] = np.float32(feature_value)
            display_payloads.append((int(candidate_index), display_mask, points, float(feature_value)))

        axis.imshow(np.ones_like(label_frame, dtype=np.float32), cmap="gray", vmin=0.0, vmax=1.0, alpha=0.12)
        image = axis.imshow(np.ma.masked_invalid(canvas), cmap="viridis", vmin=vmin, vmax=vmax)
        for candidate_index, display_mask, points, feature_value in display_payloads:
            if math.isfinite(feature_value):
                fill_color = cmap(value_norm(float(feature_value)))[:3]
                edge_color = _trajectory_consensus_candidate_edge_color(fill_color)
            else:
                edge_color = (0.45, 0.45, 0.45)
            line_style = "-"
            line_width = 1.3
            role_suffix = ""
            if int(candidate_index) == int(winner_candidate):
                line_width = 2.1
                line_style = "-"
                role_suffix = " [W]"
            elif int(candidate_index) == int(loser_candidate):
                line_width = 2.1
                line_style = "--"
                role_suffix = " [L]"
            axis.contour(
                display_mask.astype(np.float32),
                levels=[0.5],
                colors=[edge_color],
                linewidths=line_width,
                linestyles=[line_style],
            )
            if points.size > 0 and math.isfinite(feature_value):
                x_anchor = float(points[:, 1].max()) + text_pad_x
                y_anchor = float(points[:, 0].mean())
                axis.text(
                    x_anchor,
                    y_anchor,
                    f"K{candidate_index}:{feature_value:.2f}{role_suffix}",
                    fontsize=11,
                    color="black",
                    ha="left",
                    va="center",
                )
            elif points.size > 0:
                x_anchor = float(points[:, 1].max()) + text_pad_x
                y_anchor = float(points[:, 0].mean())
                axis.text(
                    x_anchor,
                    y_anchor,
                    f"K{candidate_index}:NA{role_suffix}",
                    fontsize=11,
                    color="black",
                    ha="left",
                    va="center",
                )
        axis.set_title(f"frame={int(video_frame_label)}", fontsize=30)
        axis.set_xlim(-0.5, float(label_frame.shape[1]) - 0.5 + right_margin)
        axis.set_xticks([])
        axis.set_yticks([])
    if image is not None:
        fig.subplots_adjust(left=0.02, right=0.94, top=0.84, bottom=0.06, wspace=0.03)
        colorbar_axis = fig.add_axes([0.947, 0.14, 0.012, 0.62])
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.ax.tick_params(labelsize=12)
    else:
        fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.06, wspace=0.03)
    fig.suptitle(title, fontsize=17)
    fig.text(
        0.02,
        0.015,
        (
            "W/L criterion: anchor IoU ranking"
            if str(anchor_mode).strip().lower() == "anchor_iou"
            else "W/L criterion: anchor center-distance ranking"
        ),
        fontsize=10,
        color="#334155",
        ha="left",
        va="bottom",
    )
    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file


def _plot_wan21_t2v_self_attention_coupling_storyboard(
    label_map_fhw: torch.Tensor,
    pairwise_layer_vectors: Dict[Tuple[int, int, int], np.ndarray],
    query_frame: int,
    winner_candidate: int,
    loser_candidate: int,
    target_frames: Sequence[int],
    save_file: str,
    title: str,
    value_name: str,
    query_video_frame_label: Optional[int] = None,
    target_video_frame_labels: Optional[Sequence[int]] = None,
) -> str:
    """Render winner-versus-loser routing patterns across target frames."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if winner_candidate <= 0 or loser_candidate <= 0 or not target_frames:
        return ""
    if query_video_frame_label is None:
        query_video_frame_label = int(query_frame)
    if target_video_frame_labels is None:
        target_video_frame_labels = [int(frame_index) for frame_index in target_frames]

    label_map_fhw = torch.as_tensor(label_map_fhw).detach().cpu().to(torch.int64)
    candidate_specs = [
        ("winner-aligned", int(winner_candidate)),
        ("strongest loser", int(loser_candidate)),
    ]
    vmax_candidates: List[float] = []
    for _, candidate_index in candidate_specs:
        for target_frame in target_frames:
            vector = pairwise_layer_vectors.get((int(query_frame), int(candidate_index), int(target_frame)))
            if vector is not None:
                vmax_candidates.extend([float(x) for x in vector if math.isfinite(float(x))])
    vmax = float(max(vmax_candidates)) if vmax_candidates else 1.0
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(
        len(candidate_specs),
        1 + len(target_frames),
        figsize=(max(4.3 * (1 + len(target_frames)), 12.0), 6.8),
        gridspec_kw={"wspace": 0.03, "hspace": 0.12},
    )
    if len(candidate_specs) == 1:
        axes = np.asarray([axes])

    query_cmap = plt.get_cmap("YlOrBr")
    key_cmap = plt.get_cmap("viridis")
    for row_index, (row_name, candidate_index) in enumerate(candidate_specs):
        query_label = label_map_fhw[int(query_frame)].numpy()
        query_axis = axes[row_index, 0]
        query_axis.set_facecolor(query_cmap(0.0))
        query_axis.imshow(np.ones_like(query_label, dtype=np.float32), cmap="gray", vmin=0.0, vmax=1.0, alpha=0.12)
        query_mask = _trajectory_consensus_prepare_display_candidate_mask(query_label == int(candidate_index))
        query_canvas = np.zeros_like(query_label, dtype=np.float32)
        query_canvas[query_mask] = 1.0
        query_axis.imshow(query_canvas, cmap="YlOrBr", vmin=0.0, vmax=1.0)
        for other_candidate in range(1, int(query_label.max()) + 1):
            candidate_mask = _trajectory_consensus_prepare_display_candidate_mask(query_label == int(other_candidate))
            if np.any(candidate_mask):
                if int(other_candidate) == int(candidate_index):
                    continue
                query_axis.contour(
                    candidate_mask.astype(np.float32),
                    levels=[0.5],
                    colors=["#cbd5e1"],
                    linewidths=0.8,
                )
        query_edge_color = _trajectory_consensus_candidate_edge_color(query_cmap(1.0)[:3])
        query_axis.contour(
            query_mask.astype(np.float32),
            levels=[0.5],
            colors=[query_edge_color],
            linewidths=2.2,
            linestyles=["-" if row_name == "winner-aligned" else "--"],
        )
        query_axis.set_title(
            f"query frame={int(query_video_frame_label)}\n{row_name} K{int(candidate_index)}",
            fontsize=14,
        )
        query_axis.set_xlim(-0.5, float(query_label.shape[1]) - 0.5 + 7.0)
        query_axis.set_xticks([])
        query_axis.set_yticks([])

        for col_index, (target_frame, target_video_frame_label) in enumerate(
            zip(target_frames, target_video_frame_labels),
            start=1,
        ):
            axis = axes[row_index, col_index]
            label_frame = label_map_fhw[int(target_frame)].numpy()
            vector = pairwise_layer_vectors.get((int(query_frame), int(candidate_index), int(target_frame)))
            canvas = np.full(label_frame.shape, np.nan, dtype=np.float32)
            if vector is not None:
                for key_candidate in range(1, int(label_frame.max()) + 1):
                    if int(key_candidate - 1) < int(vector.shape[0]):
                        display_mask = _trajectory_consensus_prepare_display_candidate_mask(label_frame == int(key_candidate))
                        canvas[display_mask] = np.float32(vector[int(key_candidate - 1)])
            axis.imshow(np.ones_like(label_frame, dtype=np.float32), cmap="gray", vmin=0.0, vmax=1.0, alpha=0.12)
            axis.imshow(np.ma.masked_invalid(canvas), cmap="viridis", vmin=0.0, vmax=vmax)
            for key_candidate in range(1, int(label_frame.max()) + 1):
                candidate_mask = _trajectory_consensus_prepare_display_candidate_mask(label_frame == int(key_candidate))
                if not np.any(candidate_mask):
                    continue
                points = np.argwhere(candidate_mask)
                metric_value = (
                    float(vector[int(key_candidate - 1)])
                    if vector is not None and int(key_candidate - 1) < int(vector.shape[0])
                    else float("nan")
                )
                if math.isfinite(metric_value):
                    fill_color = key_cmap(float(metric_value) / max(vmax, 1e-8))[:3]
                    edge_color = _trajectory_consensus_candidate_edge_color(fill_color)
                else:
                    edge_color = (0.45, 0.45, 0.45)
                axis.contour(
                    candidate_mask.astype(np.float32),
                    levels=[0.5],
                    colors=[edge_color],
                    linewidths=1.2,
                )
                if points.size > 0 and math.isfinite(metric_value):
                    center_y = float(points[:, 0].mean())
                    x_anchor = float(points[:, 1].max()) + 2.0
                    axis.text(
                        x_anchor,
                        center_y,
                        f"K{key_candidate}:{metric_value:.2f}",
                        fontsize=10,
                        color="black",
                        ha="left",
                        va="center",
                    )
            axis.set_title(f"key frame={int(target_video_frame_label)}", fontsize=14)
            axis.set_xlim(-0.5, float(label_frame.shape[1]) - 0.5 + 7.0)
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle(f"{title} ({value_name})", fontsize=17)
    fig.subplots_adjust(left=0.02, right=0.99, top=0.86, bottom=0.05, wspace=0.03, hspace=0.12)
    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file


def _select_wan21_t2v_storyboard_target_frames(
    frame_count: int,
    query_frame: int,
    mode: str,
) -> List[int]:
    """Select storyboard key frames using one named heuristic."""
    frame_count = int(frame_count)
    query_frame = int(query_frame)
    mode = str(mode).strip().lower()
    if frame_count <= 0:
        return []

    if mode == "fixed_local_global":
        candidate_frames = [0, int(frame_count // 2), int(frame_count - 1), int(query_frame - 1), int(query_frame + 1)]
    elif mode == "next6":
        candidate_frames = [int(query_frame + offset) for offset in range(1, 7)]
        if len([frame for frame in candidate_frames if 0 <= int(frame) < frame_count]) < 6:
            deficit = 6 - len([frame for frame in candidate_frames if 0 <= int(frame) < frame_count])
            left_candidates = [int(query_frame - offset) for offset in range(1, frame_count)]
            candidate_frames.extend(left_candidates[:deficit])
    elif mode == "around5":
        candidate_frames = [int(query_frame - 2), int(query_frame - 1), int(query_frame + 1), int(query_frame + 2), int(query_frame + 3)]
        valid_count = len([frame for frame in candidate_frames if 0 <= int(frame) < frame_count])
        if valid_count < 5:
            deficit = 5 - valid_count
            extra_pool = []
            left_candidates = [int(query_frame - offset) for offset in range(3, frame_count)]
            right_candidates = [int(query_frame + offset) for offset in range(4, frame_count)]
            extra_pool.extend(left_candidates)
            extra_pool.extend(right_candidates)
            candidate_frames.extend(extra_pool[:deficit])
    else:
        raise ValueError(f"Unsupported storyboard target-frame mode: {mode}")

    filtered_frames = [
        int(frame)
        for frame in candidate_frames
        if 0 <= int(frame) < frame_count and int(frame) != int(query_frame)
    ]
    unique_frames = sorted(dict.fromkeys(filtered_frames))
    if mode == "next6":
        # Keep the first 6 frames after deduplication/fallback expansion.
        return unique_frames[:6]
    if mode == "around5":
        return unique_frames[:5]
    return unique_frames


def _plot_wan21_t2v_self_attention_feature_evolution_panel(
    rows: Sequence[Dict[str, object]],
    feature_names: Sequence[str],
    axis_mode: str,
    fixed_value: int,
    save_file: str,
    title: str,
) -> str:
    """Render winner-minus-loser evolution curves either step-wise or layer-wise."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feature_names = [str(feature_name) for feature_name in feature_names]
    if not feature_names:
        return ""
    num_cols = min(3, max(1, len(feature_names)))
    num_rows = int(math.ceil(len(feature_names) / float(num_cols)))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5.0 * num_cols, 3.6 * num_rows))
    axes = np.atleast_1d(axes).reshape(num_rows, num_cols)
    axis_mode = str(axis_mode).strip().lower()
    if axis_mode not in {"step", "layer"}:
        raise ValueError(f"Unsupported axis_mode: {axis_mode}")

    for panel_index, feature_name in enumerate(feature_names):
        axis = axes[panel_index // num_cols, panel_index % num_cols]
        if axis_mode == "step":
            feature_rows = [
                row for row in rows
                if str(row["feature"]) == str(feature_name) and int(row["layer"]) == int(fixed_value)
            ]
            feature_rows = sorted(feature_rows, key=lambda row: int(row["step"]))
            xs = [int(row["step"]) for row in feature_rows]
            ys = [_safe_wan21_t2v_float(row.get("mean_gap", float("nan"))) for row in feature_rows]
            x_label = "step"
        else:
            feature_rows = [
                row for row in rows
                if str(row["feature"]) == str(feature_name) and int(row["step"]) == int(fixed_value)
            ]
            feature_rows = sorted(feature_rows, key=lambda row: int(row["layer"]))
            xs = [int(row["layer"]) for row in feature_rows]
            ys = [_safe_wan21_t2v_float(row.get("mean_gap", float("nan"))) for row in feature_rows]
            x_label = "layer"
        if not feature_rows:
            axis.set_axis_off()
            continue
        axis.plot(xs, ys, marker="o", linewidth=1.4, color="#2563eb", alpha=0.55, label="raw gap")
        smooth_xs, smooth_ys = _smooth_wan21_t2v_curve_values(xs, ys, window_radius=2)
        if smooth_xs.size > 0:
            axis.plot(smooth_xs, smooth_ys, linewidth=2.4, color="#dc2626", alpha=0.95, label="smoothed gap")
        axis.axhline(0.0, color="#64748b", linewidth=1.0, linestyle="--", alpha=0.75)
        axis.set_title(_trajectory_consensus_self_attention_feature_display_name(feature_name), fontsize=10)
        axis.set_xlabel(x_label)
        axis.set_ylabel("winner-minus-loser gap")
        axis.grid(alpha=0.22, linestyle="--")
        axis.legend(fontsize=8)

    for panel_index in range(len(feature_names), num_rows * num_cols):
        axes[panel_index // num_cols, panel_index % num_cols].set_axis_off()

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file


def _plot_wan21_t2v_self_attention_signed_offset_panel(
    signed_offset_rows: Sequence[Dict[str, object]],
    selected_observations: Sequence[Tuple[int, int]],
    save_file: str,
    title: str,
) -> str:
    """Render signed-offset trend curves for representative observations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("entropy", "entropy"),
        ("dominant_link_ratio", "dominant-link ratio"),
        ("link_margin", "link margin"),
        ("head_agreement", "head agreement"),
        ("soft_head_vote_share", "soft head-vote share"),
        ("soft_head_agreement", "soft head agreement"),
    ]
    if not signed_offset_rows or not selected_observations:
        return ""
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 10.8))
    axes = axes.reshape(3, 2)
    color_map = plt.get_cmap("tab10")
    for metric_index, (metric_key, metric_label) in enumerate(metrics):
        axis = axes[metric_index // 2, metric_index % 2]
        for obs_index, (step, layer) in enumerate(selected_observations):
            observation_rows = [
                row for row in signed_offset_rows
                if int(row["step"]) == int(step) and int(row["layer"]) == int(layer)
            ]
            observation_rows = sorted(observation_rows, key=lambda row: int(row["offset"]))
            xs = [int(row["offset"]) for row in observation_rows if math.isfinite(_safe_wan21_t2v_float(row.get(metric_key, float("nan"))))]
            ys = [_safe_wan21_t2v_float(row.get(metric_key, float("nan"))) for row in observation_rows if math.isfinite(_safe_wan21_t2v_float(row.get(metric_key, float("nan"))))]
            if not xs:
                continue
            axis.plot(
                xs,
                ys,
                marker="o",
                linewidth=1.6,
                color=color_map(obs_index % 10),
                label=f"step={int(step)}, layer={int(layer)}",
            )
        axis.set_title(metric_label, fontsize=10)
        axis.set_xlabel("signed offset d = g - f")
        axis.set_ylabel(metric_label)
        axis.grid(alpha=0.22, linestyle="--")
        axis.legend(fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.08, wspace=0.24, hspace=0.28)
    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file


def _plot_wan21_t2v_self_attention_competition_curves(
    rows: Sequence[Dict[str, object]],
    axis_mode: str,
    fixed_value: int,
    normalized: bool,
    save_file: str,
    title: str,
) -> str:
    """Render CA-vs-SA competition curves on step-wise or layer-wise axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected_features = [
        "proposal_pi",
        "proposal_vote_share",
        "local_compatibility",
        "global_compatibility",
        "local_chainability",
        "global_chainability",
        "global_mutual_consistency",
        "global_head_agreement",
        "global_soft_head_vote_share",
        "global_soft_head_agreement",
    ]
    feature_labels = {
        "proposal_pi": "CA proposal strength",
        "proposal_vote_share": "CA vote share",
        "local_compatibility": "SA local compatibility",
        "global_compatibility": "SA global compatibility",
        "local_chainability": "SA local chainability",
        "global_chainability": "SA global chainability",
        "global_mutual_consistency": "SA global mutual consistency",
        "global_head_agreement": "SA global head agreement",
        "global_soft_head_vote_share": "SA global soft head-vote share",
        "global_soft_head_agreement": "SA global soft head agreement",
    }
    feature_colors = {
        "proposal_pi": "#1d4ed8",
        "proposal_vote_share": "#38bdf8",
        "local_compatibility": "#f59e0b",
        "global_compatibility": "#b45309",
        "local_chainability": "#f59e0b",
        "global_chainability": "#dc2626",
        "global_mutual_consistency": "#7c3aed",
        "global_head_agreement": "#059669",
        "global_soft_head_vote_share": "#0f766e",
        "global_soft_head_agreement": "#14b8a6",
    }
    axis_mode = str(axis_mode).strip().lower()
    if axis_mode not in {"step", "layer"}:
        raise ValueError(f"Unsupported axis_mode for competition curves: {axis_mode}")
    if axis_mode == "step":
        x_key = "step"
        filter_key = "layer"
        x_label = "step"
    else:
        x_key = "layer"
        filter_key = "step"
        x_label = "layer"

    series_rows_by_feature: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        if int(row[filter_key]) != int(fixed_value):
            continue
        feature_name = str(row["feature"])
        if feature_name not in selected_features:
            continue
        gap_value = _safe_wan21_t2v_float(row.get("mean_gap", float("nan")))
        if not math.isfinite(gap_value):
            continue
        series_rows_by_feature[feature_name].append(row)
    if not series_rows_by_feature:
        return ""

    fig, axis = plt.subplots(1, 1, figsize=(10.0, 5.4))
    for feature_name in selected_features:
        feature_rows = sorted(
            series_rows_by_feature.get(feature_name, []),
            key=lambda row: int(row[x_key]),
        )
        if not feature_rows:
            continue
        xs = [int(row[x_key]) for row in feature_rows]
        ys = [_safe_wan21_t2v_float(row.get("mean_gap", float("nan"))) for row in feature_rows]
        if bool(normalized):
            finite_abs = [abs(float(value)) for value in ys if math.isfinite(float(value))]
            robust_scale = float(np.quantile(np.asarray(finite_abs, dtype=np.float64), 0.95)) if finite_abs else 1.0
            robust_scale = max(robust_scale, 1e-8)
            ys = [float(value) / robust_scale for value in ys]
        axis.plot(
            xs,
            ys,
            marker="o",
            linewidth=1.0,
            color=feature_colors.get(feature_name, "#0f766e"),
            alpha=0.22,
            label=feature_labels.get(feature_name, feature_name),
        )
        smooth_xs, smooth_ys = _smooth_wan21_t2v_curve_values(xs, ys, window_radius=4)
        if smooth_xs.size > 0:
            axis.plot(
                smooth_xs,
                smooth_ys,
                linewidth=3.2,
                color=feature_colors.get(feature_name, "#0f766e"),
                alpha=0.98,
            )
    axis.axhline(0.0, color="#64748b", linewidth=1.0, linestyle="--", alpha=0.75)
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel("normalized winner-minus-loser gap" if bool(normalized) else "winner-minus-loser gap")
    axis.grid(alpha=0.22, linestyle="--")
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file


def _plot_wan21_t2v_self_attention_feature_target_scatter(
    candidate_feature_rows: Sequence[Dict[str, object]],
    feature_name: str,
    target_key: str,
    target_label: str,
    save_file: str,
    title: str,
    selected_step: Optional[int] = None,
) -> str:
    """Scatter one feature against one target quantity for diagnostic inspection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_rows = []
    for row in candidate_feature_rows:
        if selected_step is not None and int(row["step"]) != int(selected_step):
            continue
        feature_value = _safe_wan21_t2v_float(row.get(feature_name, float("nan")))
        target_value = _safe_wan21_t2v_float(row.get(target_key, float("nan")))
        if not math.isfinite(feature_value) or not math.isfinite(target_value):
            continue
        if int(row.get("is_winner_aligned", 0)) == 1:
            role = "winner"
            color = "#16a34a"
        elif int(row.get("is_strongest_loser", 0)) == 1:
            role = "strongest loser"
            color = "#dc2626"
        else:
            role = "other"
            color = "#94a3b8"
        plot_rows.append((feature_value, target_value, role, color))
    if not plot_rows:
        return ""

    fig, axis = plt.subplots(1, 1, figsize=(6.9, 5.4))
    for role_name in ["other", "strongest loser", "winner"]:
        role_points = [(x, y, c) for x, y, role, c in plot_rows if role == role_name]
        if not role_points:
            continue
        xs = [x for x, _, _ in role_points]
        ys = [y for _, y, _ in role_points]
        color = role_points[0][2]
        axis.scatter(xs, ys, s=18 if role_name == "other" else 24, alpha=0.72, color=color, edgecolors="none", label=role_name)
    all_xs = np.asarray([x for x, _, _, _ in plot_rows], dtype=np.float64)
    all_ys = np.asarray([y for _, y, _, _ in plot_rows], dtype=np.float64)
    if all_xs.size >= 2 and np.unique(all_xs).size >= 2:
        slope, intercept = np.polyfit(all_xs, all_ys, deg=1)
        fit_xs = np.linspace(float(all_xs.min()), float(all_xs.max()), num=200)
        fit_ys = slope * fit_xs + intercept
        axis.plot(
            fit_xs,
            fit_ys,
            color="black",
            linewidth=1.5,
            alpha=0.9,
            label="linear fit",
        )
    axis.set_title(title)
    axis.set_xlabel(_trajectory_consensus_self_attention_feature_display_name(feature_name))
    axis.set_ylabel(str(target_label))
    axis.grid(alpha=0.22, linestyle="--")
    axis.legend(fontsize=8)
    fig.tight_layout()
    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file


def _plot_wan21_t2v_self_attention_feature_target_scatter_points(
    feature_values: np.ndarray,
    target_values: np.ndarray,
    role_codes: np.ndarray,
    feature_name: str,
    target_label: str,
    save_file: str,
    title: str,
) -> str:
    """Render one scatter plot from compact per-point arrays."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feature_values = np.asarray(feature_values, dtype=np.float64)
    target_values = np.asarray(target_values, dtype=np.float64)
    role_codes = np.asarray(role_codes, dtype=np.int8)
    if feature_values.size == 0 or target_values.size == 0 or role_codes.size == 0:
        return ""

    fig, axis = plt.subplots(1, 1, figsize=(6.9, 5.4))
    role_specs = [
        (0, "other", "#94a3b8"),
        (1, "strongest loser", "#dc2626"),
        (2, "winner", "#16a34a"),
    ]
    for role_code, role_name, role_color in role_specs:
        role_mask = role_codes == int(role_code)
        if not np.any(role_mask):
            continue
        xs = feature_values[role_mask]
        ys = target_values[role_mask]
        axis.scatter(
            xs,
            ys,
            s=18 if role_code == 0 else 24,
            alpha=0.72,
            color=role_color,
            edgecolors="none",
            label=role_name,
        )

    if feature_values.size >= 2 and np.unique(feature_values).size >= 2:
        slope, intercept = np.polyfit(feature_values, target_values, deg=1)
        fit_xs = np.linspace(float(feature_values.min()), float(feature_values.max()), num=200)
        fit_ys = slope * fit_xs + intercept
        axis.plot(
            fit_xs,
            fit_ys,
            color="black",
            linewidth=1.5,
            alpha=0.9,
            label="linear fit",
        )
    axis.set_title(title)
    axis.set_xlabel(_trajectory_consensus_self_attention_feature_display_name(feature_name))
    axis.set_ylabel(str(target_label))
    axis.grid(alpha=0.22, linestyle="--")
    axis.legend(fontsize=8)
    fig.tight_layout()
    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file


def _build_wan21_t2v_self_attention_scatter_family_payload(
    candidate_feature_rows: Sequence[Dict[str, object]],
    feature_name: str,
    target_key: str,
    min_candidate_count: int = 1,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Build compact per-point arrays for one scatter family."""
    feature_values: List[float] = []
    target_values: List[float] = []
    role_codes: List[int] = []
    step_values: List[int] = []
    for row in candidate_feature_rows:
        if int(row.get("candidate_count", 0)) < max(1, int(min_candidate_count)):
            continue
        feature_value = _safe_wan21_t2v_float(row.get(feature_name, float("nan")))
        target_value = _safe_wan21_t2v_float(row.get(target_key, float("nan")))
        if not math.isfinite(feature_value) or not math.isfinite(target_value):
            continue
        if int(row.get("is_winner_aligned", 0)) == 1:
            role_code = 2
        elif int(row.get("is_strongest_loser", 0)) == 1:
            role_code = 1
        else:
            role_code = 0
        feature_values.append(float(feature_value))
        target_values.append(float(target_value))
        role_codes.append(int(role_code))
        step_values.append(int(row["step"]))
    if not feature_values:
        return None
    return (
        np.asarray(feature_values, dtype=np.float32),
        np.asarray(target_values, dtype=np.float32),
        np.asarray(role_codes, dtype=np.int8),
        np.asarray(step_values, dtype=np.int16),
    )


def _trajectory_consensus_render_self_attention_observation_plots_task(
    task: Tuple,
) -> List[str]:
    """Render all per-observation self-attention plots for one `(step, layer)`."""
    (
        step,
        layer,
        label_map_fhw_np,
        feature_rows_for_obs,
        pairwise_rows_for_obs,
        overlay_features,
        overlay_anchor_modes,
        overlay_frame_indices,
        overlay_video_frame_labels,
        storyboard_modes,
        layer1_overlay_dir,
        layer1_storyboard_dir,
        skip_existing_plots,
    ) = task

    plot_paths: List[str] = []
    label_map_fhw = torch.from_numpy(np.asarray(label_map_fhw_np)).to(torch.int64)
    step = int(step)
    layer = int(layer)
    frame_count = int(label_map_fhw.shape[0])

    for anchor_mode in overlay_anchor_modes:
        for feature_name in overlay_features:
            save_file = os.path.join(
                str(layer1_overlay_dir),
                str(anchor_mode),
                f"step_{int(step):03d}",
                f"layer_{int(layer):02d}",
                f"candidate_score_overlay_{str(feature_name)}.pdf",
            )
            if _maybe_skip_wan21_t2v_existing_plot(save_file, bool(skip_existing_plots)):
                plot_paths.append(save_file)
            else:
                plot_path = _plot_wan21_t2v_self_attention_candidate_score_overlay(
                    label_map_fhw=label_map_fhw,
                    feature_rows=feature_rows_for_obs,
                    feature_name=str(feature_name),
                    anchor_mode=str(anchor_mode),
                    save_file=save_file,
                    title=(
                        f"Candidate score overlay at step={int(step)}, layer={int(layer)} "
                        f"({str(feature_name)}, {str(anchor_mode)})"
                    ),
                    frame_indices=overlay_frame_indices,
                    video_frame_labels=overlay_video_frame_labels,
                )
                if plot_path:
                    plot_paths.append(plot_path)

    if pairwise_rows_for_obs:
        raw_coupling_vectors = _build_wan21_t2v_self_attention_pairwise_layer_value_vectors(
            pairwise_rows_for_obs,
            value_key="raw_coupling",
        )
        normalized_coupling_vectors = _build_wan21_t2v_self_attention_pairwise_layer_value_vectors(
            pairwise_rows_for_obs,
            value_key="normalized_coupling",
        )
        frame_rows_by_frame: Dict[int, List[Dict[str, object]]] = defaultdict(list)
        for row in feature_rows_for_obs:
            frame_rows_by_frame[int(row["frame"])].append(row)
        for frame_index, frame_rows in sorted(frame_rows_by_frame.items()):
            winner_rows = [row for row in frame_rows if int(row.get("is_winner_aligned", 0)) == 1]
            loser_rows = [row for row in frame_rows if int(row.get("is_strongest_loser", 0)) == 1]
            if not winner_rows or not loser_rows:
                continue
            query_video_frame_label = int(frame_index)
            winner_candidate = int(winner_rows[0]["candidate_index"])
            loser_candidate = int(loser_rows[0]["candidate_index"])
            for value_name, value_vectors in [
                ("raw_coupling", raw_coupling_vectors),
                ("normalized_coupling", normalized_coupling_vectors),
            ]:
                storyboard_save = os.path.join(
                    str(layer1_storyboard_dir),
                    f"step_{int(step):03d}",
                    f"layer_{int(layer):02d}",
                    f"frame_{int(query_video_frame_label):03d}",
                    f"winner_loser_coupling_storyboard_{str(value_name)}.pdf",
                )
                if _maybe_skip_wan21_t2v_existing_plot(storyboard_save, bool(skip_existing_plots)):
                    plot_paths.append(storyboard_save)
                else:
                    candidate_target_frames = _select_wan21_t2v_storyboard_target_frames(
                        frame_count=int(frame_count),
                        query_frame=int(frame_index),
                        mode="fixed_local_global",
                    )
                    candidate_target_video_labels = [int(candidate_frame) for candidate_frame in candidate_target_frames]
                    plot_path = _plot_wan21_t2v_self_attention_coupling_storyboard(
                        label_map_fhw=label_map_fhw,
                        pairwise_layer_vectors=value_vectors,
                        query_frame=int(frame_index),
                        winner_candidate=int(winner_candidate),
                        loser_candidate=int(loser_candidate),
                        target_frames=candidate_target_frames,
                        save_file=storyboard_save,
                        title=(
                            f"Winner-versus-loser coupling storyboard at step={int(step)}, "
                            f"layer={int(layer)}, query frame={int(query_video_frame_label)}"
                        ),
                        value_name=str(value_name),
                        query_video_frame_label=int(query_video_frame_label),
                        target_video_frame_labels=tuple(int(x) for x in candidate_target_video_labels),
                    )
                    if plot_path:
                        plot_paths.append(plot_path)
            if "next6" in {str(mode) for mode in storyboard_modes} or "around5" in {str(mode) for mode in storyboard_modes}:
                normalized_value_vectors = normalized_coupling_vectors
                storyboard_variants = [
                    ("next6", "fixed next-6 frames", "winner_loser_coupling_storyboard_normalized_coupling_next6.pdf"),
                    ("around5", "local 5-frame neighborhood", "winner_loser_coupling_storyboard_normalized_coupling_around5.pdf"),
                ]
                for mode, mode_title, filename in storyboard_variants:
                    storyboard_save = os.path.join(
                        str(layer1_storyboard_dir),
                        f"step_{int(step):03d}",
                        f"layer_{int(layer):02d}",
                        f"frame_{int(frame_index):03d}",
                        filename,
                    )
                    if _maybe_skip_wan21_t2v_existing_plot(storyboard_save, bool(skip_existing_plots)):
                        plot_paths.append(storyboard_save)
                        continue
                    candidate_target_frames = _select_wan21_t2v_storyboard_target_frames(
                        frame_count=int(frame_count),
                        query_frame=int(frame_index),
                        mode=str(mode),
                    )
                    if not candidate_target_frames:
                        continue
                    candidate_target_video_labels = [int(candidate_frame) for candidate_frame in candidate_target_frames]
                    plot_path = _plot_wan21_t2v_self_attention_coupling_storyboard(
                        label_map_fhw=label_map_fhw,
                        pairwise_layer_vectors=normalized_value_vectors,
                        query_frame=int(frame_index),
                        winner_candidate=int(winner_candidate),
                        loser_candidate=int(loser_candidate),
                        target_frames=candidate_target_frames,
                        save_file=storyboard_save,
                        title=(
                            f"Winner-versus-loser coupling storyboard ({mode_title}) at step={int(step)}, "
                            f"layer={int(layer)}, query frame={int(frame_index)}"
                        ),
                        value_name="normalized_coupling",
                        query_video_frame_label=int(frame_index),
                        target_video_frame_labels=tuple(int(x) for x in candidate_target_video_labels),
                    )
                    if plot_path:
                        plot_paths.append(plot_path)
    return plot_paths


def _trajectory_consensus_render_self_attention_plot_task(task: Tuple) -> Any:
    """Render one non-observation self-attention plot task."""
    plot_kind = str(task[0])
    if plot_kind == "evolution":
        _, rows, feature_names, axis_mode, fixed_value, save_file, title = task
        return _plot_wan21_t2v_self_attention_feature_evolution_panel(
            rows=rows,
            feature_names=feature_names,
            axis_mode=str(axis_mode),
            fixed_value=int(fixed_value),
            save_file=str(save_file),
            title=str(title),
        )
    if plot_kind == "signed_offset":
        _, signed_offset_rows, selected_observations, save_file, title = task
        return _plot_wan21_t2v_self_attention_signed_offset_panel(
            signed_offset_rows=signed_offset_rows,
            selected_observations=selected_observations,
            save_file=str(save_file),
            title=str(title),
        )
    if plot_kind == "competition":
        _, rows, axis_mode, fixed_value, normalized, save_file, title = task
        return _plot_wan21_t2v_self_attention_competition_curves(
            rows=rows,
            axis_mode=str(axis_mode),
            fixed_value=int(fixed_value),
            normalized=bool(normalized),
            save_file=str(save_file),
            title=str(title),
        )
    if plot_kind == "precedence_bar":
        _, rows, save_file, title, x_key, y_key, x_label, y_label = task
        return _plot_wan21_t2v_trajectory_consensus_bar(
            rows=rows,
            save_file=str(save_file),
            x_key=str(x_key),
            y_key=str(y_key),
            title=str(title),
            x_label=str(x_label),
            y_label=str(y_label),
        )
    if plot_kind == "heatmap":
        (
            _,
            matrix_rows,
            save_file,
            title,
            row_key,
            col_key,
            value_key,
            row_label,
            col_label,
        ) = task
        return _plot_wan21_t2v_trajectory_consensus_heatmap(
            matrix_rows=matrix_rows,
            save_file=str(save_file),
            title=str(title),
            row_key=str(row_key),
            col_key=str(col_key),
            value_key=str(value_key),
            row_label=str(row_label),
            col_label=str(col_label),
        )
    if plot_kind == "scatter":
        _, candidate_feature_rows, feature_name, target_key, target_label, save_file, title, selected_step = task
        return _plot_wan21_t2v_self_attention_feature_target_scatter(
            candidate_feature_rows=candidate_feature_rows,
            feature_name=str(feature_name),
            target_key=str(target_key),
            target_label=str(target_label),
            save_file=str(save_file),
            title=str(title),
            selected_step=(None if selected_step is None else int(selected_step)),
        )
    if plot_kind == "scatter_family":
        (
            _,
            feature_values_np,
            target_values_np,
            role_codes_np,
            step_values_np,
            feature_name,
            target_label,
            overall_save_file,
            overall_title,
            by_step_jobs,
            skip_existing_plots,
        ) = task
        plot_paths: List[str] = []
        feature_values_np = np.asarray(feature_values_np, dtype=np.float32)
        target_values_np = np.asarray(target_values_np, dtype=np.float32)
        role_codes_np = np.asarray(role_codes_np, dtype=np.int8)
        step_values_np = np.asarray(step_values_np, dtype=np.int16)
        if _maybe_skip_wan21_t2v_existing_plot(str(overall_save_file), bool(skip_existing_plots)):
            plot_paths.append(str(overall_save_file))
        else:
            plot_path = _plot_wan21_t2v_self_attention_feature_target_scatter_points(
                feature_values=feature_values_np,
                target_values=target_values_np,
                role_codes=role_codes_np,
                feature_name=str(feature_name),
                target_label=str(target_label),
                save_file=str(overall_save_file),
                title=str(overall_title),
            )
            if plot_path:
                plot_paths.append(plot_path)
        for step_value, save_file, title in by_step_jobs:
            if _maybe_skip_wan21_t2v_existing_plot(str(save_file), bool(skip_existing_plots)):
                plot_paths.append(str(save_file))
                continue
            step_mask = step_values_np == int(step_value)
            if not np.any(step_mask):
                continue
            plot_path = _plot_wan21_t2v_self_attention_feature_target_scatter_points(
                feature_values=feature_values_np[step_mask],
                target_values=target_values_np[step_mask],
                role_codes=role_codes_np[step_mask],
                feature_name=str(feature_name),
                target_label=str(target_label),
                save_file=str(save_file),
                title=str(title),
            )
            if plot_path:
                plot_paths.append(plot_path)
        return plot_paths
    raise ValueError(f"Unsupported self-attention plot task kind: {plot_kind}")


def _render_wan21_t2v_self_attention_coupling_plots(
    output_dir: str,
    frame_num: int,
    pairwise_rows: Sequence[Dict[str, object]],
    candidate_feature_rows: Sequence[Dict[str, object]],
    feature_summary_rows: Sequence[Dict[str, object]],
    temporal_precedence_rows: Sequence[Dict[str, object]],
    candidate_region_cache: Dict[Tuple[int, int], Dict[str, object]],
    skip_existing_plots: bool,
    num_workers: int = 0,
) -> List[str]:
    """Render the three-layer plot set for the self-attention coupling stage."""
    plot_paths: List[str] = []
    plots_root_dir = os.path.join(output_dir, "trajectory_consensus_self_attention_plots")
    layer1_dir = os.path.join(plots_root_dir, "layer1_mechanistic")
    layer2_dir = os.path.join(plots_root_dir, "layer2_trends")
    layer3_dir = os.path.join(plots_root_dir, "layer3_navigation")
    layer1_overlay_dir = os.path.join(layer1_dir, "candidate_score_overlays")
    layer1_storyboard_dir = os.path.join(layer1_dir, "coupling_storyboard")
    layer1_evolution_dir = os.path.join(layer1_dir, "feature_evolution")
    layer2_signed_offset_dir = os.path.join(layer2_dir, "signed_offset")
    layer2_competition_dir = os.path.join(layer2_dir, "competition")
    layer2_precedence_dir = os.path.join(layer2_dir, "temporal_precedence")
    heatmaps_dir = os.path.join(layer3_dir, "heatmaps")
    heatmap_gap_dir = os.path.join(heatmaps_dir, "winner_loser_gap")
    heatmap_auroc_dir = os.path.join(heatmaps_dir, "auroc")
    scatter_dir = os.path.join(layer3_dir, "scatter")
    scatter_anchor_dir = os.path.join(scatter_dir, "feature_vs_anchor_iou")
    scatter_anchor_dist_dir = os.path.join(scatter_dir, "feature_vs_anchor_dist")
    for directory in [
        layer1_dir,
        layer2_dir,
        layer3_dir,
        layer1_overlay_dir,
        layer1_storyboard_dir,
        layer1_evolution_dir,
        layer2_signed_offset_dir,
        layer2_competition_dir,
        layer2_precedence_dir,
        heatmaps_dir,
        heatmap_gap_dir,
        heatmap_auroc_dir,
        scatter_dir,
        scatter_anchor_dir,
        scatter_anchor_dist_dir,
    ]:
        _ensure_dir(directory)

    features = sorted({str(row["feature"]) for row in feature_summary_rows})
    gap_rows = _build_wan21_t2v_self_attention_feature_gap_rows(
        candidate_feature_rows,
        feature_names=features,
    )
    step_layer_feature_rows: Dict[Tuple[int, int], List[Dict[str, object]]] = defaultdict(list)
    for row in candidate_feature_rows:
        step_layer_feature_rows[(int(row["step"]), int(row["layer"]))].append(row)
    step_layer_pairwise_rows: Dict[Tuple[int, int], List[Dict[str, object]]] = defaultdict(list)
    for row in pairwise_rows:
        step_layer_pairwise_rows[(int(row["step"]), int(row["layer"]))].append(row)
    step_layer_gap_rows: Dict[Tuple[int, int], List[Dict[str, object]]] = defaultdict(list)
    for row in gap_rows:
        step_layer_gap_rows[(int(row["step"]), int(row["layer"]))].append(row)

    overlay_features = [
        "proposal_pi",
        "proposal_vote_share",
        "local_compatibility",
        "global_compatibility",
        "local_chainability",
        "global_chainability",
        "local_mutual_consistency",
        "global_mutual_consistency",
        "local_head_agreement",
        "global_head_agreement",
        "local_soft_head_vote_share",
        "global_soft_head_vote_share",
        "local_soft_head_agreement",
        "global_soft_head_agreement",
    ]
    overlay_anchor_modes = [
        "anchor_iou",
        "anchor_distance",
    ]
    storyboard_modes = [
        "fixed_local_global",
        "next6",
        "around5",
    ]
    sorted_step_layer_keys = sorted(step_layer_feature_rows.keys())
    observation_plot_tasks: List[Tuple] = []
    for step, layer in sorted_step_layer_keys:
        label_payload = candidate_region_cache.get((int(step), int(layer)))
        feature_rows_for_obs = step_layer_feature_rows.get((int(step), int(layer)), [])
        pairwise_rows_for_obs = step_layer_pairwise_rows.get((int(step), int(layer)), [])
        if label_payload is None or not feature_rows_for_obs:
            continue
        label_map_fhw = label_payload["label_map_fhw"]
        frame_count = int(label_map_fhw.shape[0])
        overlay_frame_indices, overlay_video_frame_labels = _resolve_wan21_t2v_trajectory_consensus_viz_frames(
            attention_frame_count=int(frame_count),
            video_frame_count=int(frame_num),
            num_frames=10,
        )
        label_map_np = np.ascontiguousarray(
            label_map_fhw.detach().cpu().numpy().astype(np.int16, copy=False)
        )
        observation_plot_tasks.append(
            (
                int(step),
                int(layer),
                label_map_np,
                feature_rows_for_obs,
                pairwise_rows_for_obs,
                tuple(str(feature_name) for feature_name in overlay_features),
                tuple(str(anchor_mode) for anchor_mode in overlay_anchor_modes),
                tuple(int(frame_index) for frame_index in overlay_frame_indices),
                tuple(int(frame_label) for frame_label in overlay_video_frame_labels),
                tuple(str(mode) for mode in storyboard_modes),
                str(layer1_overlay_dir),
                str(layer1_storyboard_dir),
                bool(skip_existing_plots),
            )
        )

    observation_progress_bar = None
    if observation_plot_tasks:
        try:
            from tqdm import tqdm
            observation_progress_bar = tqdm(
                total=int(len(observation_plot_tasks)),
                desc="trajectory consensus self-attention observation plots",
                unit="obs",
                leave=True,
            )
        except Exception:
            observation_progress_bar = None
    try:
        effective_num_workers = _resolve_wan21_t2v_num_workers(
            requested_num_workers=int(num_workers),
            task_count=int(len(observation_plot_tasks)),
        )
        for task_plot_paths in _iter_wan21_t2v_parallel_results(
            tasks=observation_plot_tasks,
            worker_fn=_trajectory_consensus_render_self_attention_observation_plots_task,
            num_workers=int(effective_num_workers),
        ):
            plot_paths.extend([path for path in task_plot_paths if path])
            if observation_progress_bar is not None:
                observation_progress_bar.update(1)
    finally:
        if observation_progress_bar is not None:
            observation_progress_bar.close()

    evolution_features = [
        "proposal_pi",
        "proposal_vote_share",
        "local_avg_covered_mass",
        "global_avg_covered_mass",
        "local_entropy",
        "global_entropy",
        "local_dominant_link_ratio",
        "global_dominant_link_ratio",
        "local_link_margin",
        "global_link_margin",
        "local_compatibility",
        "global_compatibility",
        "local_chainability",
        "global_chainability",
        "local_incoming_support",
        "global_incoming_support",
        "local_incoming_preference_share",
        "global_incoming_preference_share",
        "local_incoming_vote_share",
        "global_incoming_vote_share",
        "local_mutual_consistency",
        "global_mutual_consistency",
        "local_head_agreement",
        "global_head_agreement",
        "local_soft_head_vote_share",
        "global_soft_head_vote_share",
        "local_soft_head_agreement",
        "global_soft_head_agreement",
    ]
    stepwise_gap_rows = _build_wan21_t2v_self_attention_stepwise_gap_rows(gap_rows)
    layerwise_gap_rows = _build_wan21_t2v_self_attention_layerwise_gap_rows(gap_rows)
    all_layers = sorted({int(row["layer"]) for row in stepwise_gap_rows})
    all_steps = sorted({int(row["step"]) for row in layerwise_gap_rows})
    misc_plot_tasks: List[Tuple] = []
    for layer in all_layers:
        evolution_save = os.path.join(
            layer1_evolution_dir,
            "by_layer",
            f"layer_{int(layer):02d}",
            "winner_loser_gap_vs_step.pdf",
        )
        if _maybe_skip_wan21_t2v_existing_plot(evolution_save, skip_existing_plots):
            plot_paths.append(evolution_save)
        else:
            misc_plot_tasks.append(
                (
                    "evolution",
                    stepwise_gap_rows,
                    tuple(evolution_features),
                    "step",
                    int(layer),
                    evolution_save,
                    f"Winner-minus-loser gaps versus step at layer={int(layer)}",
                )
            )
    for step in all_steps:
        evolution_save = os.path.join(
            layer1_evolution_dir,
            "by_step",
            f"step_{int(step):03d}",
            "winner_loser_gap_vs_layer.pdf",
        )
        if _maybe_skip_wan21_t2v_existing_plot(evolution_save, skip_existing_plots):
            plot_paths.append(evolution_save)
        else:
            misc_plot_tasks.append(
                (
                    "evolution",
                    layerwise_gap_rows,
                    tuple(evolution_features),
                    "layer",
                    int(step),
                    evolution_save,
                    f"Winner-minus-loser gaps versus layer at step={int(step)}",
                )
            )

    signed_offset_rows = _build_wan21_t2v_self_attention_signed_offset_rows(pairwise_rows)
    rows_by_step: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    rows_by_layer: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for step, layer in sorted_step_layer_keys:
        rows_by_step[int(step)].append((int(step), int(layer)))
        rows_by_layer[int(layer)].append((int(step), int(layer)))
    for step, observations in sorted(rows_by_step.items()):
        signed_offset_save = os.path.join(
            layer2_signed_offset_dir,
            "by_step",
            f"step_{int(step):03d}",
            "signed_offset_planning_curves.pdf",
        )
        if _maybe_skip_wan21_t2v_existing_plot(signed_offset_save, skip_existing_plots):
            plot_paths.append(signed_offset_save)
        else:
            misc_plot_tasks.append(
                (
                    "signed_offset",
                    signed_offset_rows,
                    tuple(sorted(observations, key=lambda item: int(item[1]))),
                    signed_offset_save,
                    f"Signed-offset self-attention planning curves at step={int(step)}",
                )
            )
    for layer, observations in sorted(rows_by_layer.items()):
        signed_offset_save = os.path.join(
            layer2_signed_offset_dir,
            "by_layer",
            f"layer_{int(layer):02d}",
            "signed_offset_planning_curves.pdf",
        )
        if _maybe_skip_wan21_t2v_existing_plot(signed_offset_save, skip_existing_plots):
            plot_paths.append(signed_offset_save)
        else:
            misc_plot_tasks.append(
                (
                    "signed_offset",
                    signed_offset_rows,
                    tuple(sorted(observations, key=lambda item: int(item[0]))),
                    signed_offset_save,
                    f"Signed-offset self-attention planning curves at layer={int(layer)}",
                )
            )

    for layer in all_layers:
        for normalized, value_tag in [(False, "raw"), (True, "normalized")]:
            competition_save = os.path.join(
                layer2_competition_dir,
                "by_layer",
                f"layer_{int(layer):02d}",
                value_tag,
                "cross_attention_vs_self_attention_competition.pdf",
            )
            if _maybe_skip_wan21_t2v_existing_plot(competition_save, skip_existing_plots):
                plot_paths.append(competition_save)
            else:
                misc_plot_tasks.append(
                    (
                        "competition",
                        stepwise_gap_rows,
                        "step",
                        int(layer),
                        bool(normalized),
                        competition_save,
                        (
                            f"Cross-attention proposal versus self-attention coordination "
                            f"at layer={int(layer)} ({value_tag})"
                        ),
                    )
                )
    for step in all_steps:
        for normalized, value_tag in [(False, "raw"), (True, "normalized")]:
            competition_save = os.path.join(
                layer2_competition_dir,
                "by_step",
                f"step_{int(step):03d}",
                value_tag,
                "cross_attention_vs_self_attention_competition.pdf",
            )
            if _maybe_skip_wan21_t2v_existing_plot(competition_save, skip_existing_plots):
                plot_paths.append(competition_save)
            else:
                misc_plot_tasks.append(
                    (
                        "competition",
                        layerwise_gap_rows,
                        "layer",
                        int(step),
                        bool(normalized),
                        competition_save,
                        (
                            f"Cross-attention proposal versus self-attention coordination "
                            f"at step={int(step)} ({value_tag})"
                        ),
                    )
                )

    precedence_summary_rows: List[Dict[str, object]] = []
    rows_by_feature: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in temporal_precedence_rows:
        rows_by_feature[str(row["feature"])].append(row)
    for feature_name, rows in sorted(rows_by_feature.items()):
        precedence_values = [
            float(row["precedence_observation_index"])
            for row in rows
            if int(row["precedence_observation_index"]) >= 0
        ]
        precedence_summary_rows.append({
            "feature": str(feature_name),
            "mean_precedence_index": float(_mean_wan21_t2v_finite(precedence_values)),
        })
    precedence_save = os.path.join(layer2_precedence_dir, "temporal_precedence_summary.pdf")
    if precedence_summary_rows:
        if _maybe_skip_wan21_t2v_existing_plot(precedence_save, skip_existing_plots):
            plot_paths.append(precedence_save)
        else:
            misc_plot_tasks.append(
                (
                    "precedence_bar",
                    precedence_summary_rows,
                    precedence_save,
                    "Temporal precedence summary (lower means earlier stable winner-loser separation)",
                    "feature",
                    "mean_precedence_index",
                    "feature",
                    "mean earliest stable separation order (lower earlier)",
                )
            )

    for feature_name in features:
        feature_rows = [row for row in feature_summary_rows if str(row["feature"]) == str(feature_name)]
        if not feature_rows:
            continue
        gap_save = os.path.join(heatmap_gap_dir, f"{feature_name}_winner_loser_gap.pdf")
        if _maybe_skip_wan21_t2v_existing_plot(gap_save, skip_existing_plots):
            plot_paths.append(gap_save)
        else:
            misc_plot_tasks.append(
                (
                    "heatmap",
                    feature_rows,
                    gap_save,
                    f"{feature_name} winner-minus-loser gap",
                    "step",
                    "layer",
                    "winner_loser_gap",
                    "step",
                    "layer",
                )
            )
        auroc_save = os.path.join(heatmap_auroc_dir, f"{feature_name}_auroc.pdf")
        if _maybe_skip_wan21_t2v_existing_plot(auroc_save, skip_existing_plots):
            plot_paths.append(auroc_save)
        else:
            misc_plot_tasks.append(
                (
                    "heatmap",
                    feature_rows,
                    auroc_save,
                    f"{feature_name} AUROC",
                    "step",
                    "layer",
                    "auroc",
                    "step",
                    "layer",
                )
            )

    scatter_features = [
        "proposal_pi",
        "proposal_vote_share",
        "local_compatibility",
        "global_compatibility",
        "local_chainability",
        "global_chainability",
        "local_incoming_support",
        "global_incoming_support",
        "local_incoming_preference_share",
        "global_incoming_preference_share",
        "local_incoming_vote_share",
        "global_incoming_vote_share",
        "local_mutual_consistency",
        "global_mutual_consistency",
        "local_head_agreement",
        "global_head_agreement",
        "local_soft_head_vote_share",
        "global_soft_head_vote_share",
        "local_soft_head_agreement",
        "global_soft_head_agreement",
        "local_link_margin",
        "global_link_margin",
        "local_dominant_link_ratio",
        "global_dominant_link_ratio",
        "local_entropy",
        "global_entropy",
    ]
    all_scatter_steps = sorted({int(row["step"]) for row in candidate_feature_rows})
    scatter_targets = [
        ("anchor_iou", "anchor IoU", scatter_anchor_dir, "anchor_iou"),
        ("anchor_distance", "anchor-distance score", scatter_anchor_dist_dir, "anchor_dist"),
    ]
    scatter_filter_specs = [
        (1, "", ""),
        (2, "_candidate_count_ge_2", " | candidate_count>=2"),
    ]
    for feature_name in scatter_features:
        for target_key, target_label, target_root_dir, target_tag in scatter_targets:
            feature_scatter_dir = os.path.join(target_root_dir, str(feature_name))
            overall_dir = os.path.join(feature_scatter_dir, "overall")
            by_step_dir = os.path.join(feature_scatter_dir, "by_step")
            _ensure_dir(overall_dir)
            _ensure_dir(by_step_dir)
            for min_candidate_count, file_suffix, title_suffix in scatter_filter_specs:
                scatter_payload = _build_wan21_t2v_self_attention_scatter_family_payload(
                    candidate_feature_rows=candidate_feature_rows,
                    feature_name=str(feature_name),
                    target_key=str(target_key),
                    min_candidate_count=int(min_candidate_count),
                )
                if scatter_payload is None:
                    continue
                overall_save = os.path.join(
                    overall_dir,
                    f"{feature_name}_vs_{target_tag}{file_suffix}.pdf",
                )
                overall_exists = _maybe_skip_wan21_t2v_existing_plot(overall_save, skip_existing_plots)
                overall_title = (
                    f"{_trajectory_consensus_self_attention_feature_display_name(feature_name)} "
                    f"versus {target_label}{title_suffix}"
                )
                if overall_exists:
                    plot_paths.append(overall_save)
                by_step_jobs: List[Tuple[int, str, str]] = []
                for step in all_scatter_steps:
                    step_scatter_save = os.path.join(
                        by_step_dir,
                        f"step_{int(step):03d}",
                        f"{feature_name}_vs_{target_tag}_step_{int(step):03d}{file_suffix}.pdf",
                    )
                    if _maybe_skip_wan21_t2v_existing_plot(step_scatter_save, skip_existing_plots):
                        plot_paths.append(step_scatter_save)
                        continue
                    by_step_jobs.append(
                        (
                            int(step),
                            step_scatter_save,
                            (
                                f"{_trajectory_consensus_self_attention_feature_display_name(feature_name)} "
                                f"versus {target_label} at step={int(step)}{title_suffix}"
                            ),
                        )
                    )
                if (not overall_exists) or by_step_jobs:
                    misc_plot_tasks.append(
                        (
                            "scatter_family",
                            scatter_payload[0],
                            scatter_payload[1],
                            scatter_payload[2],
                            scatter_payload[3],
                            feature_name,
                            target_label,
                            overall_save,
                            overall_title,
                            tuple(by_step_jobs),
                            bool(skip_existing_plots),
                        )
                    )

    misc_progress_bar = None
    if misc_plot_tasks:
        try:
            from tqdm import tqdm
            misc_progress_bar = tqdm(
                total=int(len(misc_plot_tasks)),
                desc="trajectory consensus self-attention summary plots",
                unit="plot",
                leave=True,
            )
        except Exception:
            misc_progress_bar = None
    try:
        effective_num_workers = _resolve_wan21_t2v_num_workers(
            requested_num_workers=int(num_workers),
            task_count=int(len(misc_plot_tasks)),
        )
        for plot_result in _iter_wan21_t2v_parallel_results(
            tasks=misc_plot_tasks,
            worker_fn=_trajectory_consensus_render_self_attention_plot_task,
            num_workers=int(effective_num_workers),
        ):
            if isinstance(plot_result, (list, tuple)):
                plot_paths.extend([path for path in plot_result if path])
            elif plot_result:
                plot_paths.append(plot_result)
            if misc_progress_bar is not None:
                misc_progress_bar.update(1)
    finally:
        if misc_progress_bar is not None:
            misc_progress_bar.close()
    return [path for path in plot_paths if path]


def run_wan21_t2v_trajectory_consensus_dynamics(
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
    parallel_cfg: Optional[Wan21T2VParallelConfig] = None,
    target_object_words: Sequence[str] = tuple(),
    target_verb_words: Sequence[str] = tuple(),
    reuse_cross_attention_dir: Optional[str] = None,
    reuse_head_trajectory_dynamics_dir: Optional[str] = None,
    reuse_head_evolution_dir: Optional[str] = None,
    trajectory_consensus_stages: Sequence[str] = ("candidate_consensus", "head_contribution"),
    trajectory_consensus_steps: Sequence[int] = tuple(),
    trajectory_consensus_layers: Sequence[int] = tuple(),
    trajectory_consensus_cross_heads: Optional[Sequence[str]] = tuple(),
    trajectory_consensus_self_heads: Optional[Sequence[str]] = tuple(),
    trajectory_consensus_modules: Sequence[str] = ("cross", "self"),
    trajectory_consensus_branch: str = "cond",
    trajectory_consensus_reference_distance_metrics: Sequence[str] = ("center_l2",),
    trajectory_consensus_alignment_summary_steps: int = 10,
    trajectory_consensus_scatter_outlier_heads: Sequence[str] = tuple(),
    trajectory_consensus_scatter_outlier_cross_heads: Sequence[str] = tuple(),
    trajectory_consensus_scatter_outlier_self_heads: Sequence[str] = tuple(),
    trajectory_consensus_candidate_base_quantile: float = 0.85,
    trajectory_consensus_candidate_split_quantiles: Sequence[float] = (0.92, 0.95, 0.97),
    trajectory_consensus_candidate_smooth_radius: int = 1,
    trajectory_consensus_candidate_stable_peak_min_levels: int = 2,
    trajectory_consensus_candidate_peak_merge_distance: float = 2.0,
    trajectory_consensus_candidate_preprocess_winsorize_quantile: float = 0.995,
    trajectory_consensus_candidate_preprocess_despike_quantile: float = 0.98,
    trajectory_consensus_candidate_min_component_area: int = 4,
    trajectory_consensus_candidate_viz_num_frames: int = 8,
    trajectory_consensus_candidate_enable_per_head: bool = True,
    trajectory_consensus_do_ablation: bool = True,
    trajectory_consensus_contribution_method: str = "exact_ablation",
    trajectory_consensus_ablate_position: str = "pre_o",
    trajectory_consensus_do_direct_proxy: bool = True,
    trajectory_consensus_compute_direct_projection: Optional[bool] = None,
    trajectory_consensus_object_mask_reference_step: int = 50,
    trajectory_consensus_object_mask_reference_layer: int = 27,
    trajectory_consensus_taylor_object_only: bool = True,
    trajectory_consensus_taylor_num_latent_frames: int = 10,
    trajectory_consensus_taylor_metric_scope: str = "obj",
    trajectory_consensus_taylor_patching_metric: str = "v_sum",
    trajectory_consensus_taylor_ablation_mode: str = "zero_ablation",
    trajectory_consensus_taylor_use_gradient_checkpointing: bool = True,
    trajectory_consensus_filter_heads: bool = False,
    trajectory_consensus_filter_step: int = 1,
    trajectory_consensus_filter_convergence_speed_rule: str = "gt_-inf",
    trajectory_consensus_filter_contribution_rule: str = "lt_inf",
    trajectory_consensus_filter_contribution_metric: str = "",
    trajectory_consensus_filter_module: str = "cross",
    trajectory_consensus_sa_anchor_step: int = 49,
    trajectory_consensus_sa_anchor_layer: int = 27,
    trajectory_consensus_sa_covered_mass_min: float = 0.0,
    trajectory_consensus_sa_precedence_persistence: int = 2,
    trajectory_consensus_plot_only_from_csv: bool = False,
    trajectory_consensus_seed_influence_mode: str = "seed_sensitivity",
    trajectory_consensus_seed_influence_seeds: Sequence[int] = tuple(),
    trajectory_consensus_seed_sensitivity_zr_metric: str = "global_mutual_consistency",
    trajectory_consensus_seed_sensitivity_steps: Sequence[int] = tuple(),
    trajectory_consensus_anchor_frame_steps: Sequence[int] = tuple(),
    trajectory_consensus_seed_influence_anchor_topk: int = 2,
    trajectory_consensus_seed_influence_arrow_topk: int = 2,
    trajectory_consensus_skip_existing_plots: bool = True,
    trajectory_consensus_num_workers: int = 0,
) -> Dict[str, object]:
    """Run the trajectory-consensus-dynamics experiment."""
    _ensure_dir(output_dir)
    stages = [str(stage).strip().lower() for stage in trajectory_consensus_stages if str(stage).strip()]
    stages = list(dict.fromkeys(stages))
    allowed_stages = {"candidate_consensus", "head_contribution", "self_attention_coupling", "seed_influence"}
    bad_stages = [stage for stage in stages if stage not in allowed_stages]
    if bad_stages:
        raise ValueError(f"Unsupported trajectory_consensus stages: {bad_stages}")
    if not stages:
        raise ValueError("trajectory_consensus_stages must be non-empty.")
    if "seed_influence" in stages and len(stages) > 1:
        raise ValueError(
            "seed_influence should be run as a standalone trajectory_consensus stage. "
            "Please set trajectory_consensus_stages to `seed_influence` only."
        )
    if stages == ["seed_influence"]:
        from .trajectory_consensus_seed_influence import (
            run_wan21_t2v_trajectory_consensus_seed_influence,
        )
        return run_wan21_t2v_trajectory_consensus_seed_influence(
            wan21_root=wan21_root,
            ckpt_dir=ckpt_dir,
            output_dir=output_dir,
            prompt=prompt,
            size=size,
            task=task,
            frame_num=frame_num,
            shift=shift,
            sample_solver=sample_solver,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            device_id=device_id,
            offload_model=offload_model,
            parallel_cfg=parallel_cfg,
            target_object_words=target_object_words,
            target_verb_words=target_verb_words,
            reuse_cross_attention_dir=reuse_cross_attention_dir,
            trajectory_consensus_steps=trajectory_consensus_steps,
            trajectory_consensus_seed_sensitivity_steps=trajectory_consensus_seed_sensitivity_steps,
            trajectory_consensus_anchor_frame_steps=trajectory_consensus_anchor_frame_steps,
            trajectory_consensus_layers=trajectory_consensus_layers,
            trajectory_consensus_self_heads=trajectory_consensus_self_heads,
            trajectory_consensus_branch=trajectory_consensus_branch,
            trajectory_consensus_candidate_base_quantile=trajectory_consensus_candidate_base_quantile,
            trajectory_consensus_candidate_split_quantiles=trajectory_consensus_candidate_split_quantiles,
            trajectory_consensus_candidate_smooth_radius=trajectory_consensus_candidate_smooth_radius,
            trajectory_consensus_candidate_stable_peak_min_levels=trajectory_consensus_candidate_stable_peak_min_levels,
            trajectory_consensus_candidate_peak_merge_distance=trajectory_consensus_candidate_peak_merge_distance,
            trajectory_consensus_candidate_preprocess_winsorize_quantile=trajectory_consensus_candidate_preprocess_winsorize_quantile,
            trajectory_consensus_candidate_preprocess_despike_quantile=trajectory_consensus_candidate_preprocess_despike_quantile,
            trajectory_consensus_candidate_min_component_area=trajectory_consensus_candidate_min_component_area,
            trajectory_consensus_object_mask_reference_step=trajectory_consensus_object_mask_reference_step,
            trajectory_consensus_object_mask_reference_layer=trajectory_consensus_object_mask_reference_layer,
            trajectory_consensus_sa_anchor_step=trajectory_consensus_sa_anchor_step,
            trajectory_consensus_sa_anchor_layer=trajectory_consensus_sa_anchor_layer,
            trajectory_consensus_sa_covered_mass_min=trajectory_consensus_sa_covered_mass_min,
            trajectory_consensus_plot_only_from_csv=trajectory_consensus_plot_only_from_csv,
            trajectory_consensus_seed_influence_mode=trajectory_consensus_seed_influence_mode,
            trajectory_consensus_seed_influence_seeds=trajectory_consensus_seed_influence_seeds,
            trajectory_consensus_seed_sensitivity_zr_metric=trajectory_consensus_seed_sensitivity_zr_metric,
            trajectory_consensus_seed_influence_anchor_topk=trajectory_consensus_seed_influence_anchor_topk,
            trajectory_consensus_seed_influence_arrow_topk=trajectory_consensus_seed_influence_arrow_topk,
            trajectory_consensus_skip_existing_plots=trajectory_consensus_skip_existing_plots,
            trajectory_consensus_num_workers=trajectory_consensus_num_workers,
        )
    do_ablation = bool(trajectory_consensus_do_ablation)
    contribution_method = str(trajectory_consensus_contribution_method).strip().lower()
    if contribution_method not in {"exact_ablation", "taylor_approx"}:
        raise ValueError(
            "trajectory_consensus_contribution_method must be `exact_ablation` or `taylor_approx`."
        )
    ablate_position = str(trajectory_consensus_ablate_position).strip().lower()
    if ablate_position not in {"pre_o", "post_o"}:
        raise ValueError(
            "trajectory_consensus_ablate_position must be `pre_o` or `post_o`."
        )
    do_direct_proxy = bool(trajectory_consensus_do_direct_proxy)
    if trajectory_consensus_compute_direct_projection is not None:
        do_direct_proxy = bool(trajectory_consensus_compute_direct_projection)
    reference_distance_metrics = [
        str(metric_name).strip()
        for metric_name in trajectory_consensus_reference_distance_metrics
        if str(metric_name).strip()
    ]
    if not reference_distance_metrics:
        reference_distance_metrics = ["center_l2"]
    alignment_summary_steps = max(1, int(trajectory_consensus_alignment_summary_steps))
    legacy_scatter_outlier_heads = [
        str(head_tag).strip()
        for head_tag in trajectory_consensus_scatter_outlier_heads
        if str(head_tag).strip()
    ]
    scatter_outlier_cross_heads = [
        str(head_tag).strip()
        for head_tag in trajectory_consensus_scatter_outlier_cross_heads
        if str(head_tag).strip()
    ]
    scatter_outlier_self_heads = [
        str(head_tag).strip()
        for head_tag in trajectory_consensus_scatter_outlier_self_heads
        if str(head_tag).strip()
    ]
    if legacy_scatter_outlier_heads:
        if not scatter_outlier_cross_heads:
            scatter_outlier_cross_heads = list(legacy_scatter_outlier_heads)
        if not scatter_outlier_self_heads:
            scatter_outlier_self_heads = list(legacy_scatter_outlier_heads)
    scatter_outlier_heads_by_module = {
        "cross": list(scatter_outlier_cross_heads),
        "self": list(scatter_outlier_self_heads),
    }
    contribution_branch = str(trajectory_consensus_branch).strip().lower()
    taylor_object_only = bool(trajectory_consensus_taylor_object_only)
    taylor_num_latent_frames = int(trajectory_consensus_taylor_num_latent_frames)
    taylor_metric_scope = str(trajectory_consensus_taylor_metric_scope).strip().lower()
    taylor_patching_metric = str(trajectory_consensus_taylor_patching_metric).strip().lower()
    taylor_ablation_mode = str(trajectory_consensus_taylor_ablation_mode).strip().lower()
    taylor_use_gradient_checkpointing = bool(trajectory_consensus_taylor_use_gradient_checkpointing)
    if taylor_num_latent_frames == 0:
        raise ValueError("trajectory_consensus_taylor_num_latent_frames must be positive or -1.")
    if taylor_num_latent_frames < -1:
        raise ValueError("trajectory_consensus_taylor_num_latent_frames must be positive or -1.")
    if taylor_metric_scope not in {"obj", "global"}:
        raise ValueError("trajectory_consensus_taylor_metric_scope must be `obj` or `global`.")
    if taylor_patching_metric not in {"v_sum", "ref_dot", "sem_obj"}:
        raise ValueError("trajectory_consensus_taylor_patching_metric must be `v_sum`, `ref_dot`, or `sem_obj`.")
    if taylor_patching_metric == "sem_obj" and str(contribution_branch).strip().lower() != "cond":
        raise ValueError("trajectory_consensus_taylor_patching_metric='sem_obj' currently requires trajectory_consensus_branch='cond'.")
    if taylor_patching_metric == "sem_obj" and taylor_metric_scope != "obj":
        raise ValueError("trajectory_consensus_taylor_patching_metric='sem_obj' currently requires trajectory_consensus_taylor_metric_scope='obj'.")
    if taylor_ablation_mode not in {"zero_ablation", "mean_ablation"}:
        raise ValueError("trajectory_consensus_taylor_ablation_mode must be `zero_ablation` or `mean_ablation`.")
    if int(trajectory_consensus_sa_precedence_persistence) <= 0:
        raise ValueError("trajectory_consensus_sa_precedence_persistence must be positive.")
    filter_heads = bool(trajectory_consensus_filter_heads)
    filter_step = int(trajectory_consensus_filter_step)
    filter_convergence_speed_rule = _parse_wan21_t2v_filter_rule(
        str(trajectory_consensus_filter_convergence_speed_rule),
        default_direction="gt",
        rule_name="trajectory_consensus_filter_convergence_speed_rule",
    )
    filter_contribution_rule = _parse_wan21_t2v_filter_rule(
        str(trajectory_consensus_filter_contribution_rule),
        default_direction="lt",
        rule_name="trajectory_consensus_filter_contribution_rule",
    )
    filter_contribution_metric = str(trajectory_consensus_filter_contribution_metric).strip()
    filter_module = str(trajectory_consensus_filter_module).strip().lower()
    if filter_module not in {"cross", "self"}:
        raise ValueError("trajectory_consensus_filter_module must be `cross` or `self`.")

    head_contribution_base_dir = os.path.join(output_dir, "trajectory_consensus_head_contribution")
    head_contribution_output_dirs: Dict[str, str] = {}
    if do_ablation:
        ablate_position_tag = str(ablate_position)
        if contribution_method == "taylor_approx":
            taylor_region_tag = "obj" if taylor_object_only else "global"
            taylor_scope_tag = f"ablate_at_{taylor_region_tag}_{int(taylor_num_latent_frames)}frames"
            patching_metric_tag = f"patching_metric_{str(taylor_patching_metric)}_{str(taylor_metric_scope)}"
            ablation_mode_tag = f"baseline_{str(taylor_ablation_mode)}"
            method_dir_name = (
                f"taylor_approx-{ablate_position_tag}-"
                f"{taylor_scope_tag}-{patching_metric_tag}-{ablation_mode_tag}"
            )
        else:
            method_dir_name = f"{str(contribution_method)}-{ablate_position_tag}"
        head_contribution_output_dirs[str(contribution_method)] = os.path.join(
            head_contribution_base_dir,
            method_dir_name,
        )
    if do_direct_proxy:
        head_contribution_output_dirs["direct_proxy"] = os.path.join(
            head_contribution_base_dir,
            "direct_proxy",
        )

    mean_maps: Dict[Tuple[int, int, str], torch.Tensor] = {}
    loaded_map_path = ""
    object_words_in_maps: List[str] = []

    need_cross_attention_maps = (
        (not bool(trajectory_consensus_plot_only_from_csv))
        or ("candidate_consensus" in stages and trajectory_consensus_cross_heads is not None)
    )
    can_load_cross_attention_maps = bool(reuse_cross_attention_dir) and bool(target_object_words)
    if need_cross_attention_maps and (not can_load_cross_attention_maps):
        raise ValueError(
            "reuse_cross_attention_dir and target_object_words are required for this "
            "trajectory_consensus_dynamics run."
        )

    if can_load_cross_attention_maps:
        mean_maps, loaded_map_path = _load_wan21_t2v_cross_attention_mean_maps_from_disk(reuse_cross_attention_dir)
        words_in_maps = sorted({str(key[2]) for key in mean_maps.keys()})
        _load_wan21_t2v_cross_attention_token_meta(
            output_dir=reuse_cross_attention_dir,
            words_in_maps=words_in_maps,
            target_object_words=target_object_words,
            target_verb_words=target_verb_words,
        )
        object_words_in_maps = [str(word) for word in target_object_words if str(word) in words_in_maps]
        if need_cross_attention_maps and (not object_words_in_maps):
            raise ValueError(
                "None of target_object_words are present in reused cross-attention maps. "
                f"requested={list(target_object_words)} available={words_in_maps}"
            )

    if mean_maps:
        selected_steps, selected_layers = _resolve_wan21_t2v_steps_and_layers_from_maps(
            mean_maps=mean_maps,
            requested_steps=trajectory_consensus_steps,
            requested_layers=trajectory_consensus_layers,
        )
    else:
        selected_steps = _dedup_wan21_t2v_int_list(trajectory_consensus_steps) if trajectory_consensus_steps else []
        selected_layers = _dedup_wan21_t2v_int_list(trajectory_consensus_layers) if trajectory_consensus_layers else []

    num_cross_heads_per_layer: Dict[int, int] = {}
    if mean_maps and selected_steps and selected_layers:
        exemplar_step = int(selected_steps[0])
        for layer in selected_layers:
            exemplar_cross_maps = _mean_wan21_t2v_head_maps_for_words(
                mean_maps=mean_maps,
                step=exemplar_step,
                layer=int(layer),
                words=object_words_in_maps,
            )
            if exemplar_cross_maps is not None:
                num_cross_heads_per_layer[int(layer)] = int(exemplar_cross_maps.shape[0])

    selected_cross_head_specs = _resolve_wan21_t2v_selected_head_specs_from_layer_counts(
        explicit_head_specs=trajectory_consensus_cross_heads,
        num_heads_per_layer=num_cross_heads_per_layer,
    )
    selected_self_head_specs: List[Tuple[int, int]] = []
    summary: Dict[str, object] = {
        "experiment": "wan21_t2v_trajectory_consensus_dynamics",
        "loaded_map_path": loaded_map_path,
        "stages": list(stages),
        "selected_steps": list(selected_steps),
        "selected_layers": list(selected_layers),
        "selected_cross_head_specs": [f"L{layer}H{head}" for layer, head in selected_cross_head_specs],
        "selected_self_head_specs": [],
        "trajectory_consensus_do_ablation": bool(do_ablation),
        "trajectory_consensus_do_direct_proxy": bool(do_direct_proxy),
        "trajectory_consensus_reference_distance_metrics": list(reference_distance_metrics),
        "trajectory_consensus_alignment_summary_steps": int(alignment_summary_steps),
        "trajectory_consensus_scatter_outlier_heads": list(legacy_scatter_outlier_heads),
        "trajectory_consensus_scatter_outlier_cross_heads": list(scatter_outlier_cross_heads),
        "trajectory_consensus_scatter_outlier_self_heads": list(scatter_outlier_self_heads),
        "trajectory_consensus_head_contribution_output_dirs": dict(head_contribution_output_dirs),
        "object_words_in_maps": list(object_words_in_maps),
        "trajectory_consensus_contribution_method": str(contribution_method),
        "trajectory_consensus_ablate_position": str(ablate_position),
        "trajectory_consensus_taylor_object_only": bool(taylor_object_only),
        "trajectory_consensus_taylor_num_latent_frames": int(taylor_num_latent_frames),
        "trajectory_consensus_taylor_metric_scope": str(taylor_metric_scope),
        "trajectory_consensus_taylor_patching_metric": str(taylor_patching_metric),
        "trajectory_consensus_taylor_ablation_mode": str(taylor_ablation_mode),
        "trajectory_consensus_taylor_use_gradient_checkpointing": bool(taylor_use_gradient_checkpointing),
        "trajectory_consensus_filter_heads": bool(filter_heads),
        "trajectory_consensus_filter_step": int(filter_step),
        "trajectory_consensus_filter_convergence_speed_rule": str(filter_convergence_speed_rule["text"]),
        "trajectory_consensus_filter_contribution_rule": str(filter_contribution_rule["text"]),
        "trajectory_consensus_filter_contribution_metric": str(filter_contribution_metric),
        "trajectory_consensus_filter_module": str(filter_module),
        "trajectory_consensus_sa_anchor_step": int(trajectory_consensus_sa_anchor_step),
        "trajectory_consensus_sa_anchor_layer": int(trajectory_consensus_sa_anchor_layer),
        "trajectory_consensus_sa_covered_mass_min": float(trajectory_consensus_sa_covered_mass_min),
        "trajectory_consensus_sa_precedence_persistence": int(trajectory_consensus_sa_precedence_persistence),
        "trajectory_consensus_plot_only_from_csv": bool(trajectory_consensus_plot_only_from_csv),
        "trajectory_consensus_skip_existing_plots": bool(trajectory_consensus_skip_existing_plots),
        "trajectory_consensus_num_workers": int(trajectory_consensus_num_workers),
        "trajectory_consensus_candidate_enable_per_head": bool(trajectory_consensus_candidate_enable_per_head),
    }

    candidate_region_rows: List[Dict[str, object]] = []
    candidate_weight_rows: List[Dict[str, object]] = []
    winner_gap_rows: List[Dict[str, object]] = []
    candidate_region_cache: Dict[Tuple[int, int], Dict[str, object]] = {}
    per_head_candidate_region_rows: List[Dict[str, object]] = []
    per_head_candidate_region_cache: Dict[Tuple[int, int, int], Dict[str, object]] = {}
    self_attention_coupling_pairwise_rows: List[Dict[str, object]] = []
    self_attention_coupling_candidate_feature_rows: List[Dict[str, object]] = []
    self_attention_coupling_feature_summary_rows: List[Dict[str, object]] = []
    self_attention_coupling_temporal_precedence_rows: List[Dict[str, object]] = []
    plot_paths: List[str] = []

    if "candidate_consensus" in stages:
        candidate_regions_pt_path = os.path.join(output_dir, "trajectory_consensus_candidate_regions.pt")
        candidate_regions_csv_path = os.path.join(output_dir, "trajectory_consensus_candidate_regions.csv")
        candidate_weights_csv_path = os.path.join(output_dir, "trajectory_consensus_candidate_weights.csv")
        winner_gap_csv_path = os.path.join(output_dir, "trajectory_consensus_winner_gap.csv")
        per_head_candidate_regions_pt_path = os.path.join(
            output_dir,
            "trajectory_consensus_candidate_regions_per_head.pt",
        )
        per_head_candidate_regions_csv_path = os.path.join(
            output_dir,
            "trajectory_consensus_candidate_regions_per_head.csv",
        )
        reference_object_boxes: Optional[List[Dict[str, float]]] = None
        reference_head_maps = _mean_wan21_t2v_head_maps_for_words(
            mean_maps=mean_maps,
            step=int(trajectory_consensus_object_mask_reference_step),
            layer=int(trajectory_consensus_object_mask_reference_layer),
            words=object_words_in_maps,
        )
        if reference_head_maps is not None:
            reference_head_mean_map = reference_head_maps.mean(dim=0)
            reference_preprocessed_head_mean_map, _ = _preprocess_wan21_t2v_attention_map_fhw(
                map_fhw=reference_head_mean_map,
                winsorize_quantile=float(trajectory_consensus_candidate_preprocess_winsorize_quantile),
                despike_quantile=float(trajectory_consensus_candidate_preprocess_despike_quantile),
                min_component_area=int(trajectory_consensus_candidate_min_component_area),
            )
            reference_object_boxes = _build_wan21_t2v_reference_object_boxes(
                reference_map_fhw=reference_preprocessed_head_mean_map,
                center_mode="geometric_center",
                traj_power=1.5,
                traj_quantile=0.95,
                support_radius_mode="adaptive_area",
                support_radius_fixed=2.0,
                support_radius_alpha=1.0,
                support_radius_min=1.0,
                support_radius_max_ratio=0.25,
            )

        if bool(trajectory_consensus_plot_only_from_csv):
            if not os.path.exists(candidate_regions_pt_path):
                raise FileNotFoundError(
                    f"trajectory_consensus_plot_only_from_csv=True but missing cached candidate regions: {candidate_regions_pt_path}"
                )
            loaded_candidate_cache = _load_wan21_t2v_torch_cache(candidate_regions_pt_path)
            candidate_region_cache = {
                (int(step), int(layer)): {
                    "label_map_fhw": (
                        torch.from_numpy(np.asarray(candidate_payload["label_map_fhw_np"])).to(torch.int64)
                        if "label_map_fhw_np" in candidate_payload
                        else candidate_payload["label_map_fhw"].detach().cpu().to(torch.int64)
                    )
                }
                for (step, layer), candidate_payload in loaded_candidate_cache.items()
            }
            candidate_region_rows = _load_wan21_t2v_csv_rows(candidate_regions_csv_path)
            candidate_weight_rows = _load_wan21_t2v_csv_rows(candidate_weights_csv_path)
            winner_gap_rows = _load_wan21_t2v_csv_rows(winner_gap_csv_path)
            if bool(trajectory_consensus_candidate_enable_per_head) and os.path.exists(per_head_candidate_regions_pt_path):
                loaded_per_head_cache = _load_wan21_t2v_torch_cache(per_head_candidate_regions_pt_path)
                per_head_candidate_region_cache = {
                    (int(step), int(layer), int(head)): {
                        "label_map_fhw": (
                            torch.from_numpy(np.asarray(candidate_payload["label_map_fhw_np"])).to(torch.int64)
                            if "label_map_fhw_np" in candidate_payload
                            else candidate_payload["label_map_fhw"].detach().cpu().to(torch.int64)
                        )
                    }
                    for (step, layer, head), candidate_payload in loaded_per_head_cache.items()
                }
            if bool(trajectory_consensus_candidate_enable_per_head) and os.path.exists(per_head_candidate_regions_csv_path):
                per_head_candidate_region_rows = _load_wan21_t2v_csv_rows(per_head_candidate_regions_csv_path)
        else:
            extraction_tasks = []
            for step in selected_steps:
                for layer in selected_layers:
                    headmean_map = _mean_wan21_t2v_headmean_map_for_words(
                        mean_maps=mean_maps,
                        step=int(step),
                        layer=int(layer),
                        words=object_words_in_maps,
                    )
                    if headmean_map is None:
                        continue
                    # IMPORTANT:
                    # `avg_pool2d` inside `_smooth_wan21_t2v_map_fhw` can hang when executed
                    # inside forked process-pool workers in this environment. We therefore
                    # apply the optional smoothing on the parent process and pass
                    # `smooth_radius=0` to the child worker.
                    if int(trajectory_consensus_candidate_smooth_radius) > 0:
                        worker_map_fhw = _smooth_wan21_t2v_map_fhw(
                            headmean_map,
                            smooth_radius=int(trajectory_consensus_candidate_smooth_radius),
                        )
                        worker_smooth_radius = 0
                    else:
                        worker_map_fhw = headmean_map
                        worker_smooth_radius = 0
                    extraction_tasks.append(
                        (
                            int(step),
                            int(layer),
                            worker_map_fhw,
                            float(trajectory_consensus_candidate_base_quantile),
                            tuple(float(x) for x in trajectory_consensus_candidate_split_quantiles),
                            int(trajectory_consensus_candidate_min_component_area),
                            int(worker_smooth_radius),
                            int(trajectory_consensus_candidate_stable_peak_min_levels),
                            float(trajectory_consensus_candidate_peak_merge_distance),
                            float(trajectory_consensus_candidate_preprocess_winsorize_quantile),
                            float(trajectory_consensus_candidate_preprocess_despike_quantile),
                            int(trajectory_consensus_candidate_min_component_area),
                        )
                    )

            effective_num_workers = _resolve_wan21_t2v_num_workers(
                requested_num_workers=int(trajectory_consensus_num_workers),
                task_count=int(len(extraction_tasks)),
            )
            extraction_progress_bar = None
            if extraction_tasks:
                try:
                    from tqdm import tqdm
                    extraction_progress_bar = tqdm(
                        total=int(len(extraction_tasks)),
                        desc="trajectory consensus candidate extraction",
                        unit="item",
                        leave=True,
                    )
                except Exception:
                    extraction_progress_bar = None

            try:
                for step, layer, candidate_data in _iter_wan21_t2v_parallel_results(
                    tasks=extraction_tasks,
                    worker_fn=_trajectory_consensus_extract_candidate_task,
                    num_workers=int(effective_num_workers),
                ):
                    head_maps = _mean_wan21_t2v_head_maps_for_words(
                        mean_maps=mean_maps,
                        step=int(step),
                        layer=int(layer),
                        words=object_words_in_maps,
                    )
                    if head_maps is None:
                        if extraction_progress_bar is not None:
                            extraction_progress_bar.update(1)
                        continue
                    candidate_data = dict(candidate_data)
                    candidate_data["label_map_fhw"] = torch.from_numpy(
                        np.asarray(candidate_data.pop("label_map_fhw_np"))
                    ).to(torch.int64)
                    if reference_object_boxes is not None:
                        merged_label_map_fhw, merged_frame_metadata = _merge_wan21_t2v_candidate_regions_by_reference_box(
                            label_map_fhw=candidate_data["label_map_fhw"],
                            frame_metadata=candidate_data["frame_metadata"],
                            reference_boxes=reference_object_boxes,
                        )
                        candidate_data["label_map_fhw"] = merged_label_map_fhw
                        candidate_data["frame_metadata"] = merged_frame_metadata
                    candidate_region_cache[(int(step), int(layer))] = candidate_data
                    label_map_fhw = candidate_data["label_map_fhw"]
                    frame_metadata = candidate_data["frame_metadata"]
                    frame_count = int(label_map_fhw.shape[0])
                    for frame_index in range(frame_count):
                        frame_candidates = frame_metadata[frame_index]
                        for candidate_row in frame_candidates:
                            candidate_region_rows.append({
                                "step": int(step),
                                "layer": int(layer),
                                "frame": int(frame_index),
                                "candidate_index": int(candidate_row["candidate_index"]),
                                "area": int(candidate_row["area"]),
                                "mass": float(candidate_row.get("mass", float("nan"))),
                                "density": float(candidate_row.get("density", float("nan"))),
                                "bbox_height": float(candidate_row.get("bbox_height", float("nan"))),
                                "bbox_width": float(candidate_row.get("bbox_width", float("nan"))),
                                "bbox_y_min": float(candidate_row.get("bbox_y_min", float("nan"))),
                                "bbox_y_max": float(candidate_row.get("bbox_y_max", float("nan"))),
                                "bbox_x_min": float(candidate_row.get("bbox_x_min", float("nan"))),
                                "bbox_x_max": float(candidate_row.get("bbox_x_max", float("nan"))),
                                "peak_y": float(candidate_row["peak_y"]),
                                "peak_x": float(candidate_row["peak_x"]),
                                "centroid_y": float(candidate_row["centroid_y"]),
                                "centroid_x": float(candidate_row["centroid_x"]),
                                "seed_y": float(candidate_row.get("seed_y", float("nan"))),
                                "seed_x": float(candidate_row.get("seed_x", float("nan"))),
                                "seed_score": float(candidate_row.get("seed_score", float("nan"))),
                                "support_count": int(candidate_row.get("support_count", 0)),
                                "support_level": float(candidate_row.get("support_level", float("nan"))),
                                "candidate_count_in_frame": int(len(frame_candidates)),
                            })

                    per_head_weights: Dict[int, List[List[float]]] = {}
                    for head_index in range(int(head_maps.shape[0])):
                        probability_map = _normalize_wan21_t2v_attention_map_per_frame(head_maps[head_index])
                        per_head_weights[int(head_index)] = _compute_wan21_t2v_candidate_weights_for_head_map(
                            probability_map_fhw=probability_map,
                            label_map_fhw=label_map_fhw,
                        )
                        for frame_index, frame_weights in enumerate(per_head_weights[int(head_index)]):
                            for candidate_offset, weight_value in enumerate(frame_weights):
                                candidate_weight_rows.append({
                                    "step": int(step),
                                    "layer": int(layer),
                                    "head": int(head_index),
                                    "head_tag": f"L{int(layer)}H{int(head_index)}",
                                    "frame": int(frame_index),
                                    "candidate_index": int(candidate_offset + 1),
                                    "candidate_weight": float(weight_value),
                                })

                    num_heads = int(head_maps.shape[0])
                    for frame_index in range(frame_count):
                        candidate_count = int(label_map_fhw[frame_index].max().item())
                        layer_mean_weights = []
                        for candidate_index in range(candidate_count):
                            head_values = [
                                per_head_weights[head_index][frame_index][candidate_index]
                                for head_index in range(num_heads)
                                if candidate_index < len(per_head_weights[head_index][frame_index])
                            ]
                            layer_mean_weights.append(float(sum(head_values) / max(1, len(head_values))))
                        if not layer_mean_weights:
                            continue
                        sorted_weights = sorted(layer_mean_weights, reverse=True)
                        top1 = float(sorted_weights[0])
                        top2 = float(sorted_weights[1]) if len(sorted_weights) >= 2 else 0.0
                        probability = torch.tensor(layer_mean_weights, dtype=torch.float32)
                        probability = probability / probability.sum().clamp_min(1e-12)
                        candidate_entropy = float(
                            -(probability * probability.clamp_min(1e-12).log()).sum().item()
                        )
                        winner_gap_rows.append({
                            "step": int(step),
                            "layer": int(layer),
                            "frame": int(frame_index),
                            "candidate_count": int(candidate_count),
                            "winner_gap": float(top1 - top2),
                            "winner_weight": float(top1),
                            "runner_up_weight": float(top2),
                            "candidate_entropy": float(candidate_entropy),
                        })

                    if extraction_progress_bar is not None:
                        extraction_progress_bar.update(1)
            finally:
                if extraction_progress_bar is not None:
                    extraction_progress_bar.close()

            candidate_region_cache_to_save = {
                (int(step), int(layer)): {
                    "label_map_fhw": candidate_data["label_map_fhw"].detach().cpu().to(torch.int16),
                }
                for (step, layer), candidate_data in candidate_region_cache.items()
            }
            torch.save(candidate_region_cache_to_save, candidate_regions_pt_path)
            _save_csv(candidate_regions_csv_path, candidate_region_rows)
            _save_csv(candidate_weights_csv_path, candidate_weight_rows)
            _save_csv(winner_gap_csv_path, winner_gap_rows)

        if bool(trajectory_consensus_candidate_enable_per_head):
            per_head_specs_by_layer: Dict[int, List[int]] = defaultdict(list)
            for layer_index, head_index in selected_cross_head_specs:
                if int(layer_index) not in selected_layers:
                    continue
                per_head_specs_by_layer[int(layer_index)].append(int(head_index))

            per_head_extraction_tasks = []
            per_head_extraction_result_count = 0
            for step in selected_steps:
                for layer_index in selected_layers:
                    selected_head_indices = sorted(
                        {
                            int(head_index)
                            for head_index in per_head_specs_by_layer.get(int(layer_index), [])
                            if (int(step), int(layer_index), int(head_index)) not in per_head_candidate_region_cache
                        }
                    )
                    if not selected_head_indices:
                        continue
                    head_maps = _mean_wan21_t2v_head_maps_for_words(
                        mean_maps=mean_maps,
                        step=int(step),
                        layer=int(layer_index),
                        words=object_words_in_maps,
                    )
                    if head_maps is None:
                        continue
                    head_payloads = []
                    for head_index in selected_head_indices:
                        if int(head_index) >= int(head_maps.shape[0]):
                            continue
                        # IMPORTANT:
                        # Keep the per-head path consistent with head-mean extraction.
                        # `avg_pool2d` inside `_smooth_wan21_t2v_map_fhw` can hang in
                        # forked workers in this environment, so optional smoothing must
                        # happen on the parent process before we dispatch worker tasks.
                        if int(trajectory_consensus_candidate_smooth_radius) > 0:
                            worker_map_fhw = _smooth_wan21_t2v_map_fhw(
                                head_maps[int(head_index)],
                                smooth_radius=int(trajectory_consensus_candidate_smooth_radius),
                            )
                            worker_smooth_radius = 0
                        else:
                            worker_map_fhw = head_maps[int(head_index)]
                            worker_smooth_radius = 0
                        head_payloads.append((int(head_index), worker_map_fhw))
                    if not head_payloads:
                        continue
                    per_head_extraction_result_count += int(len(head_payloads))
                    per_head_extraction_tasks.append(
                        (
                            int(step),
                            int(layer_index),
                            tuple(head_payloads),
                            float(trajectory_consensus_candidate_base_quantile),
                            tuple(float(x) for x in trajectory_consensus_candidate_split_quantiles),
                            int(trajectory_consensus_candidate_min_component_area),
                            int(worker_smooth_radius),
                            int(trajectory_consensus_candidate_stable_peak_min_levels),
                            float(trajectory_consensus_candidate_peak_merge_distance),
                            float(trajectory_consensus_candidate_preprocess_winsorize_quantile),
                            float(trajectory_consensus_candidate_preprocess_despike_quantile),
                            int(trajectory_consensus_candidate_min_component_area),
                        )
                    )

            per_head_extraction_progress_bar = None
            if per_head_extraction_tasks:
                try:
                    from tqdm import tqdm
                    per_head_extraction_progress_bar = tqdm(
                        total=int(per_head_extraction_result_count),
                        desc="trajectory consensus per-head candidate extraction",
                        unit="head",
                        leave=True,
                    )
                except Exception:
                    per_head_extraction_progress_bar = None
            try:
                effective_num_workers = _resolve_wan21_t2v_num_workers(
                    requested_num_workers=int(trajectory_consensus_num_workers),
                    task_count=int(len(per_head_extraction_tasks)),
                )
                for step, layer, head_results in _iter_wan21_t2v_parallel_results(
                    tasks=per_head_extraction_tasks,
                    worker_fn=_trajectory_consensus_extract_layer_head_candidates_task,
                    num_workers=int(effective_num_workers),
                ):
                    for head, candidate_data in head_results:
                        candidate_data = dict(candidate_data)
                        label_map_fhw = torch.from_numpy(
                            np.asarray(candidate_data.pop("label_map_fhw_np"))
                        ).to(torch.int64)
                        frame_metadata = candidate_data["frame_metadata"]
                        if reference_object_boxes is not None:
                            label_map_fhw, frame_metadata = _merge_wan21_t2v_candidate_regions_by_reference_box(
                                label_map_fhw=label_map_fhw,
                                frame_metadata=frame_metadata,
                                reference_boxes=reference_object_boxes,
                            )
                        per_head_candidate_region_cache[(int(step), int(layer), int(head))] = {
                            "label_map_fhw": label_map_fhw,
                        }
                        for frame_index, frame_candidates in enumerate(frame_metadata):
                            for candidate_row in frame_candidates:
                                per_head_candidate_region_rows.append({
                                    "step": int(step),
                                    "layer": int(layer),
                                    "head": int(head),
                                    "head_tag": f"L{int(layer)}H{int(head)}",
                                    "frame": int(frame_index),
                                    "candidate_index": int(candidate_row["candidate_index"]),
                                    "area": int(candidate_row["area"]),
                                    "mass": float(candidate_row.get("mass", float("nan"))),
                                    "density": float(candidate_row.get("density", float("nan"))),
                                    "bbox_height": float(candidate_row.get("bbox_height", float("nan"))),
                                    "bbox_width": float(candidate_row.get("bbox_width", float("nan"))),
                                    "peak_y": float(candidate_row["peak_y"]),
                                    "peak_x": float(candidate_row["peak_x"]),
                                    "centroid_y": float(candidate_row["centroid_y"]),
                                    "centroid_x": float(candidate_row["centroid_x"]),
                                    "seed_y": float(candidate_row.get("seed_y", float("nan"))),
                                    "seed_x": float(candidate_row.get("seed_x", float("nan"))),
                                    "seed_score": float(candidate_row.get("seed_score", float("nan"))),
                                    "support_count": int(candidate_row.get("support_count", 0)),
                                    "support_level": float(candidate_row.get("support_level", float("nan"))),
                                    "candidate_count_in_frame": int(len(frame_candidates)),
                                })
                        if per_head_extraction_progress_bar is not None:
                            per_head_extraction_progress_bar.update(1)
            finally:
                if per_head_extraction_progress_bar is not None:
                    per_head_extraction_progress_bar.close()

            if per_head_candidate_region_rows:
                per_head_candidate_region_rows = sorted(
                    per_head_candidate_region_rows,
                    key=lambda row: (
                        int(row["step"]),
                        int(row["layer"]),
                        int(row["head"]),
                        int(row["frame"]),
                        int(row["candidate_index"]),
                    ),
                )
                _save_csv(per_head_candidate_regions_csv_path, per_head_candidate_region_rows)
            if per_head_candidate_region_cache:
                per_head_candidate_region_cache_to_save = {
                    (int(step), int(layer), int(head)): {
                        "label_map_fhw": candidate_data["label_map_fhw"].detach().cpu().to(torch.int16),
                    }
                    for (step, layer, head), candidate_data in per_head_candidate_region_cache.items()
                }
                torch.save(per_head_candidate_region_cache_to_save, per_head_candidate_regions_pt_path)

        plot_paths.extend(
            _render_wan21_t2v_candidate_consensus_plots(
                output_dir,
                winner_gap_rows,
                skip_existing_plots=bool(trajectory_consensus_skip_existing_plots),
            )
        )

        candidate_viz_tasks = []
        head_specific_candidate_viz_tasks = []
        for step in selected_steps:
            for layer_index in selected_layers:
                candidate_data = candidate_region_cache.get((int(step), int(layer_index)))
                if candidate_data is None:
                    continue

                headmean_map = _mean_wan21_t2v_headmean_map_for_words(
                    mean_maps=mean_maps,
                    step=int(step),
                    layer=int(layer_index),
                    words=object_words_in_maps,
                )
                if headmean_map is not None:
                    attention_frame_indices, video_frame_labels = _resolve_wan21_t2v_trajectory_consensus_viz_frames(
                        attention_frame_count=int(headmean_map.shape[0]),
                        video_frame_count=int(frame_num),
                        num_frames=int(trajectory_consensus_candidate_viz_num_frames),
                    )
                    save_file = os.path.join(
                        output_dir,
                        "trajectory_consensus_candidate_region_viz",
                        f"step_{int(step):03d}",
                        f"layer_{int(layer_index):02d}_head_mean.pdf",
                    )
                    contour_save_file = os.path.join(
                        output_dir,
                        "trajectory_consensus_candidate_region_viz",
                        f"step_{int(step):03d}",
                        f"layer_{int(layer_index):02d}_head_mean_contour.pdf",
                    )
                    skip_standard = _maybe_skip_wan21_t2v_existing_plot(
                        save_file,
                        bool(trajectory_consensus_skip_existing_plots),
                    )
                    skip_contour = _maybe_skip_wan21_t2v_existing_plot(
                        contour_save_file,
                        bool(trajectory_consensus_skip_existing_plots),
                    )
                    if skip_standard:
                        plot_paths.append(save_file)
                    if skip_contour:
                        plot_paths.append(contour_save_file)
                    if (not skip_standard) or (not skip_contour):
                        raw_map_np, label_map_np = _pack_wan21_t2v_candidate_viz_arrays(
                            raw_map_fhw=headmean_map,
                            label_map_fhw=candidate_data["label_map_fhw"],
                        )
                    if not skip_standard:
                        candidate_viz_tasks.append(
                            (
                                raw_map_np,
                                label_map_np,
                                save_file,
                                f"Candidate regions | step={int(step)} layer={int(layer_index)} head=mean",
                                tuple(int(frame_index) for frame_index in attention_frame_indices),
                                tuple(int(frame_label) for frame_label in video_frame_labels),
                                False,
                                "magma",
                            )
                        )
                    if not skip_contour:
                        candidate_viz_tasks.append(
                            (
                                raw_map_np,
                                label_map_np,
                                contour_save_file,
                                (
                                    f"Candidate regions with contours | step={int(step)} "
                                    f"layer={int(layer_index)} head=mean"
                                ),
                                tuple(int(frame_index) for frame_index in attention_frame_indices),
                                tuple(int(frame_label) for frame_label in video_frame_labels),
                                True,
                                "magma",
                            )
                        )

            if bool(trajectory_consensus_candidate_enable_per_head) and selected_cross_head_specs:
                for layer_index, head_index in selected_cross_head_specs:
                    if int(layer_index) not in selected_layers:
                        continue
                    head_maps = _mean_wan21_t2v_head_maps_for_words(
                        mean_maps=mean_maps,
                        step=int(step),
                        layer=int(layer_index),
                        words=object_words_in_maps,
                    )
                    if head_maps is None or int(head_index) >= int(head_maps.shape[0]):
                        continue
                    attention_frame_indices, video_frame_labels = _resolve_wan21_t2v_trajectory_consensus_viz_frames(
                        attention_frame_count=int(head_maps.shape[1]),
                        video_frame_count=int(frame_num),
                        num_frames=int(trajectory_consensus_candidate_viz_num_frames),
                    )
                    save_file = os.path.join(
                        output_dir,
                        "trajectory_consensus_candidate_region_viz",
                        f"step_{int(step):03d}",
                        f"layer_{int(layer_index):02d}_head_{int(head_index):02d}.pdf",
                    )
                    contour_save_file = os.path.join(
                        output_dir,
                        "trajectory_consensus_candidate_region_viz",
                        f"step_{int(step):03d}",
                        f"layer_{int(layer_index):02d}_head_{int(head_index):02d}_contour.pdf",
                    )
                    skip_standard = _maybe_skip_wan21_t2v_existing_plot(
                        save_file,
                        bool(trajectory_consensus_skip_existing_plots),
                    )
                    skip_contour = _maybe_skip_wan21_t2v_existing_plot(
                        contour_save_file,
                        bool(trajectory_consensus_skip_existing_plots),
                    )
                    if skip_standard:
                        plot_paths.append(save_file)
                    if skip_contour:
                        plot_paths.append(contour_save_file)
                    if skip_standard and skip_contour:
                        continue
                    head_candidate_data = per_head_candidate_region_cache.get(
                        (int(step), int(layer_index), int(head_index))
                    )
                    if head_candidate_data is None:
                        continue
                    raw_map_np, label_map_np = _pack_wan21_t2v_candidate_viz_arrays(
                        raw_map_fhw=head_maps[int(head_index)],
                        label_map_fhw=head_candidate_data["label_map_fhw"],
                    )
                    if not skip_standard:
                        head_specific_candidate_viz_tasks.append(
                            (
                                raw_map_np,
                                label_map_np,
                                save_file,
                                (
                                    f"Candidate regions | step={int(step)} "
                                    f"layer={int(layer_index)} head={int(head_index)}"
                                ),
                                tuple(int(frame_index) for frame_index in attention_frame_indices),
                                tuple(int(frame_label) for frame_label in video_frame_labels),
                                False,
                                "magma",
                            )
                        )
                    if not skip_contour:
                        head_specific_candidate_viz_tasks.append(
                            (
                                raw_map_np,
                                label_map_np,
                                contour_save_file,
                                (
                                    f"Candidate regions with contours | step={int(step)} "
                                    f"layer={int(layer_index)} head={int(head_index)}"
                                ),
                                tuple(int(frame_index) for frame_index in attention_frame_indices),
                                tuple(int(frame_label) for frame_label in video_frame_labels),
                                True,
                                "magma",
                            )
                        )
        viz_progress_bar = None
        if candidate_viz_tasks:
            try:
                from tqdm import tqdm
                viz_progress_bar = tqdm(
                    total=int(len(candidate_viz_tasks)),
                    desc="trajectory consensus candidate viz",
                    unit="plot",
                    leave=True,
                )
            except Exception:
                viz_progress_bar = None
        try:
            effective_num_workers = _resolve_wan21_t2v_num_workers(
                requested_num_workers=int(trajectory_consensus_num_workers),
                task_count=int(len(candidate_viz_tasks)),
            )
            for plot_path in _iter_wan21_t2v_parallel_results(
                tasks=candidate_viz_tasks,
                worker_fn=_trajectory_consensus_render_candidate_viz_task,
                num_workers=int(effective_num_workers),
            ):
                if plot_path:
                    plot_paths.append(plot_path)
                if viz_progress_bar is not None:
                    viz_progress_bar.update(1)
        finally:
            if viz_progress_bar is not None:
                viz_progress_bar.close()

        if bool(trajectory_consensus_candidate_enable_per_head):
            head_viz_progress_bar = None
            if head_specific_candidate_viz_tasks:
                try:
                    from tqdm import tqdm
                    head_viz_progress_bar = tqdm(
                        total=int(len(head_specific_candidate_viz_tasks)),
                        desc="trajectory consensus per-head candidate viz",
                        unit="plot",
                        leave=True,
                    )
                except Exception:
                    head_viz_progress_bar = None
            try:
                effective_num_workers = _resolve_wan21_t2v_num_workers(
                    requested_num_workers=int(trajectory_consensus_num_workers),
                    task_count=int(len(head_specific_candidate_viz_tasks)),
                )
                for plot_path in _iter_wan21_t2v_parallel_results(
                    tasks=head_specific_candidate_viz_tasks,
                    worker_fn=_trajectory_consensus_render_head_specific_candidate_viz_task,
                    num_workers=int(effective_num_workers),
                ):
                    if plot_path:
                        plot_paths.append(plot_path)
                    if head_viz_progress_bar is not None:
                        head_viz_progress_bar.update(1)
            finally:
                if head_viz_progress_bar is not None:
                    head_viz_progress_bar.close()

    head_contribution_rows_by_method: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    scatter_rows_by_method: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    if "self_attention_coupling" in stages:
        pairwise_csv_path = os.path.join(output_dir, "trajectory_consensus_self_attention_coupling_pairwise.csv")
        candidate_feature_csv_path = os.path.join(
            output_dir,
            "trajectory_consensus_self_attention_coupling_candidate_features.csv",
        )
        feature_summary_csv_path = os.path.join(
            output_dir,
            "trajectory_consensus_self_attention_coupling_feature_summary.csv",
        )
        temporal_precedence_csv_path = os.path.join(
            output_dir,
            "trajectory_consensus_self_attention_coupling_temporal_precedence.csv",
        )

        if not candidate_region_cache:
            (
                candidate_region_cache,
                candidate_region_rows,
                candidate_weight_rows,
                winner_gap_rows,
            ) = _load_wan21_t2v_trajectory_consensus_candidate_cache(output_dir)

        if bool(trajectory_consensus_plot_only_from_csv):
            if not os.path.exists(candidate_feature_csv_path):
                raise FileNotFoundError(
                    "trajectory_consensus_plot_only_from_csv=True but missing cached self-attention "
                    f"candidate features CSV: {candidate_feature_csv_path}"
                )
            self_attention_coupling_pairwise_rows = (
                _load_wan21_t2v_csv_rows(pairwise_csv_path) if os.path.exists(pairwise_csv_path) else []
            )
            self_attention_coupling_candidate_feature_rows = _load_wan21_t2v_csv_rows(candidate_feature_csv_path)
            self_attention_coupling_feature_summary_rows = _summarize_wan21_t2v_self_attention_candidate_features(
                self_attention_coupling_candidate_feature_rows
            )
            self_attention_coupling_temporal_precedence_rows = _summarize_wan21_t2v_self_attention_temporal_precedence(
                self_attention_coupling_candidate_feature_rows,
                persistence_window=int(trajectory_consensus_sa_precedence_persistence),
            )
            plot_paths.extend(
                _render_wan21_t2v_self_attention_coupling_plots(
                    output_dir=output_dir,
                    frame_num=int(frame_num),
                    pairwise_rows=self_attention_coupling_pairwise_rows,
                    candidate_feature_rows=self_attention_coupling_candidate_feature_rows,
                    feature_summary_rows=self_attention_coupling_feature_summary_rows,
                    temporal_precedence_rows=self_attention_coupling_temporal_precedence_rows,
                    candidate_region_cache=candidate_region_cache,
                    skip_existing_plots=bool(trajectory_consensus_skip_existing_plots),
                    num_workers=int(trajectory_consensus_num_workers),
                )
            )
        else:
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
            del cfg
            offload_model = _resolve_wan21_t2v_offload_model(runtime, offload_model)
            target_model = _unwrap_wan21_t2v_dit_model_for_runtime_patch(pipeline.model)
            num_self_heads_per_layer = {
                int(layer): int(target_model.blocks[int(layer)].self_attn.num_heads)
                for layer in selected_layers
            }
            selected_self_head_specs = _resolve_wan21_t2v_selected_head_specs_from_layer_counts(
                explicit_head_specs=trajectory_consensus_self_heads,
                num_heads_per_layer=num_self_heads_per_layer,
            )
            if not selected_self_head_specs:
                raise ValueError(
                    "self_attention_coupling requires at least one self-attention head. "
                    "Use `trajectory_consensus_self_heads=\"\"` for all heads in selected layers."
                )
            summary["selected_self_head_specs"] = [f"L{layer}H{head}" for layer, head in selected_self_head_specs]
            head_indices_by_layer: Dict[int, Tuple[int, ...]] = {}
            for layer_index in sorted(set(int(layer) for layer, _ in selected_self_head_specs)):
                head_indices_by_layer[int(layer_index)] = tuple(
                    sorted(
                        int(head)
                        for layer, head in selected_self_head_specs
                        if int(layer) == int(layer_index)
                    )
                )

            selected_label_maps_by_step_layer = {
                (int(step), int(layer)): candidate_payload["label_map_fhw"].detach().cpu().to(torch.int64)
                for (step, layer), candidate_payload in candidate_region_cache.items()
                if int(step) in selected_steps and int(layer) in selected_layers
            }
            if not selected_label_maps_by_step_layer:
                raise ValueError(
                    "No candidate-region caches are available for the selected steps/layers. "
                    "Run candidate_consensus first or adjust trajectory_consensus_steps/layers."
                )

            coupling_branch = str(trajectory_consensus_branch).strip().lower()
            if coupling_branch not in {"cond", "uncond"}:
                raise ValueError(
                    "trajectory_consensus_branch must be `cond` or `uncond` for self_attention_coupling."
                )

            patch_cfg = Wan21T2VPatchBundleConfig(
                rope=Wan21T2VRopePatchConfig(enabled=True, mode="full"),
                probe=Wan21T2VAttentionProbeConfig(
                    enabled=True,
                    probe_steps=tuple(int(step) for step in selected_steps),
                    probe_branch=str(coupling_branch),
                    collect_dt_histograms=False,
                    collect_maas_maps=False,
                    collect_distribution=False,
                    collect_candidate_coupling=True,
                    candidate_coupling_layers=tuple(int(layer) for layer in selected_layers),
                    candidate_coupling_label_maps_by_step_layer=selected_label_maps_by_step_layer,
                    candidate_coupling_head_indices_by_layer=head_indices_by_layer,
                    stop_after_last_probe_step=True,
                ),
                causal=Wan21T2VCausalAttentionConfig(enabled=False),
            )

            _, state = _run_wan21_t2v_once_with_patch(
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
            if runtime.rank != 0:
                return None

            self_attention_coupling_pairwise_rows = state.export_candidate_coupling_rows()
            _save_csv(pairwise_csv_path, self_attention_coupling_pairwise_rows)

            anchor_payload = _build_wan21_t2v_anchor_union_payload(
                candidate_region_cache=candidate_region_cache,
                anchor_step=int(trajectory_consensus_sa_anchor_step),
                anchor_layer=int(trajectory_consensus_sa_anchor_layer),
            )
            pairwise_rows_by_step_layer: Dict[Tuple[int, int], List[Dict[str, object]]] = defaultdict(list)
            for row in self_attention_coupling_pairwise_rows:
                pairwise_rows_by_step_layer[(int(row["step"]), int(row["layer"]))].append(row)

            proposal_rows_by_step_layer: Dict[Tuple[int, int], List[Dict[str, object]]] = defaultdict(list)
            for row in candidate_weight_rows:
                proposal_rows_by_step_layer[(int(row["step"]), int(row["layer"]))].append(row)

            feature_tasks = []
            for step_layer_key, pairwise_group_rows in sorted(pairwise_rows_by_step_layer.items()):
                if not pairwise_group_rows:
                    continue
                label_payload = candidate_region_cache.get(step_layer_key)
                if label_payload is None:
                    continue
                feature_tasks.append(
                    (
                        int(step_layer_key[0]),
                        int(step_layer_key[1]),
                        pairwise_group_rows,
                        np.ascontiguousarray(
                            label_payload["label_map_fhw"].detach().cpu().numpy().astype(np.int16, copy=False)
                        ),
                        np.ascontiguousarray(
                            anchor_payload["mask_fhw"].detach().cpu().numpy().astype(np.bool_, copy=False)
                        ),
                        tuple(anchor_payload["centers"]),
                        proposal_rows_by_step_layer.get(step_layer_key, []),
                        float(trajectory_consensus_sa_covered_mass_min),
                    )
                )

            feature_progress_bar = None
            if feature_tasks:
                try:
                    from tqdm import tqdm
                    feature_progress_bar = tqdm(
                        total=int(len(feature_tasks)),
                        desc="trajectory consensus self-attention features",
                        unit="item",
                        leave=True,
                    )
                except Exception:
                    feature_progress_bar = None
            try:
                effective_num_workers = _resolve_wan21_t2v_num_workers(
                    requested_num_workers=int(trajectory_consensus_num_workers),
                    task_count=int(len(feature_tasks)),
                )
                for _, _, feature_rows in _iter_wan21_t2v_parallel_results(
                    tasks=feature_tasks,
                    worker_fn=_trajectory_consensus_compute_candidate_feature_task,
                    num_workers=int(effective_num_workers),
                ):
                    self_attention_coupling_candidate_feature_rows.extend(feature_rows)
                    if feature_progress_bar is not None:
                        feature_progress_bar.update(1)
            finally:
                if feature_progress_bar is not None:
                    feature_progress_bar.close()

            self_attention_coupling_candidate_feature_rows = sorted(
                self_attention_coupling_candidate_feature_rows,
                key=lambda row: (
                    int(row["step"]),
                    int(row["layer"]),
                    int(row["frame"]),
                    int(row["candidate_index"]),
                ),
            )
            _save_csv(candidate_feature_csv_path, self_attention_coupling_candidate_feature_rows)

            self_attention_coupling_feature_summary_rows = _summarize_wan21_t2v_self_attention_candidate_features(
                self_attention_coupling_candidate_feature_rows
            )
            _save_csv(feature_summary_csv_path, self_attention_coupling_feature_summary_rows)

            self_attention_coupling_temporal_precedence_rows = _summarize_wan21_t2v_self_attention_temporal_precedence(
                self_attention_coupling_candidate_feature_rows,
                persistence_window=int(trajectory_consensus_sa_precedence_persistence),
            )
            _save_csv(temporal_precedence_csv_path, self_attention_coupling_temporal_precedence_rows)

            plot_paths.extend(
                _render_wan21_t2v_self_attention_coupling_plots(
                    output_dir=output_dir,
                    frame_num=int(frame_num),
                    pairwise_rows=self_attention_coupling_pairwise_rows,
                    candidate_feature_rows=self_attention_coupling_candidate_feature_rows,
                    feature_summary_rows=self_attention_coupling_feature_summary_rows,
                    temporal_precedence_rows=self_attention_coupling_temporal_precedence_rows,
                    candidate_region_cache=candidate_region_cache,
                    skip_existing_plots=bool(trajectory_consensus_skip_existing_plots),
                    num_workers=int(trajectory_consensus_num_workers),
                )
            )

    if "head_contribution" in stages:
        contribution_branch = str(trajectory_consensus_branch).strip().lower()
        if contribution_branch not in {"cond", "uncond"}:
            raise ValueError(
                "trajectory_consensus_branch must be `cond` or `uncond` for head_contribution."
            )

        early_alignment_summaries = _load_wan21_t2v_reference_distance_summaries(
            reuse_head_trajectory_dynamics_dir=reuse_head_trajectory_dynamics_dir,
            reuse_head_evolution_dir=reuse_head_evolution_dir,
            distance_metrics=reference_distance_metrics,
            selected_steps=selected_steps,
            alignment_summary_steps=alignment_summary_steps,
        )
        legacy_head_contribution_csv_path = os.path.join(output_dir, "trajectory_consensus_head_contribution.csv")

        if bool(trajectory_consensus_plot_only_from_csv):
            for analysis_method, method_output_dir in sorted(head_contribution_output_dirs.items()):
                method_csv_path = os.path.join(method_output_dir, "trajectory_consensus_head_contribution.csv")
                head_contribution_csv_read_path = method_csv_path
                use_legacy_head_contribution_csv = False
                if (not os.path.exists(head_contribution_csv_read_path)) and os.path.exists(legacy_head_contribution_csv_path):
                    head_contribution_csv_read_path = legacy_head_contribution_csv_path
                    use_legacy_head_contribution_csv = True
                if not os.path.exists(head_contribution_csv_read_path):
                    raise FileNotFoundError(
                        "trajectory_consensus_plot_only_from_csv=True but missing cached "
                        f"head contribution CSV: {method_csv_path}"
                    )
                method_rows = _load_wan21_t2v_csv_rows(head_contribution_csv_read_path)
                if use_legacy_head_contribution_csv:
                    method_rows = [
                        row for row in method_rows
                        if str(row.get("analysis_method", "")).strip().lower() == str(analysis_method)
                    ]
                for row in method_rows:
                    _apply_wan21_t2v_abs_to_ablation_contribution_row(row)
                    head_contribution_rows_by_method[str(analysis_method)].append(row)
                    layer = int(row["layer"])
                    head = int(row["head"])
                    alignment_key = (int(layer), int(head))
                    if str(row.get("module", "")).strip().lower() == "cross":
                        loaded_method = str(row.get("analysis_method", "")).strip().lower()
                        if loaded_method == "exact_ablation":
                            metric_names = ("cos_obj", "cos_full", "ablate_cos_obj", "ablate_cos_full")
                        elif loaded_method == "taylor_approx":
                            metric_names = ("contribution",)
                        elif loaded_method == "direct_proxy":
                            metric_names = ("proj_dot_obj", "proj_dot_full", "proj_cos_obj", "proj_cos_full")
                        else:
                            metric_names = tuple()
                        for alignment_metric_name, early_alignment_summary in early_alignment_summaries.items():
                            if alignment_key not in early_alignment_summary:
                                continue
                            for metric_name in metric_names:
                                _append_wan21_t2v_alignment_scatter_row(
                                    scatter_rows_by_method[str(analysis_method)],
                                    alignment_metric_name=str(alignment_metric_name),
                                    analysis_method=loaded_method,
                                    module_name=str(row["module"]),
                                    branch_name=str(row["branch"]),
                                    metric_name=str(metric_name),
                                    head_tag=str(row["head_tag"]),
                                    step=int(row["step"]),
                                    metric_value=row.get(metric_name, ""),
                                    alignment_summary=early_alignment_summary[alignment_key],
                                )
                if filter_heads:
                    filtered_head_path = _export_wan21_t2v_filtered_heads(
                        method_output_dir,
                        head_contribution_rows_by_method[str(analysis_method)],
                        analysis_method=str(analysis_method),
                        module_name=str(filter_module),
                        branch_name=str(contribution_branch),
                        reuse_head_evolution_dir=reuse_head_evolution_dir,
                        alignment_summary_steps=int(alignment_summary_steps),
                        filter_step=int(filter_step),
                        convergence_speed_rule=str(filter_convergence_speed_rule["text"]),
                        contribution_rule=str(filter_contribution_rule["text"]),
                        contribution_metric_name=str(filter_contribution_metric),
                    )
                    if filtered_head_path:
                        plot_paths.append(filtered_head_path)
                plot_paths.extend(
                    _render_wan21_t2v_head_contribution_plots(
                        method_output_dir,
                        head_contribution_rows_by_method[str(analysis_method)],
                        scatter_rows_by_method[str(analysis_method)],
                        scatter_outlier_heads_by_module=scatter_outlier_heads_by_module,
                        alignment_summary_steps=int(alignment_summary_steps),
                        skip_existing_plots=bool(trajectory_consensus_skip_existing_plots),
                    )
                )
        else:
            reference_support = _build_wan21_t2v_self_attention_distribution_reference_support(
                reuse_cross_attention_dir=reuse_cross_attention_dir,
                target_object_words=target_object_words,
                target_verb_words=target_verb_words,
                reference_step=int(trajectory_consensus_object_mask_reference_step),
                reference_layer=int(trajectory_consensus_object_mask_reference_layer),
                center_mode="geometric_center",
                center_power=1.5,
                center_quantile=0.8,
                preprocess_winsorize_quantile=0.995,
                preprocess_despike_quantile=0.98,
                preprocess_min_component_area=2,
                support_radius_mode="adaptive_area",
                support_radius_fixed=2.0,
                support_radius_alpha=1.5,
                support_radius_min=1.0,
                support_radius_max_ratio=0.25,
            )

            parallel_cfg = parallel_cfg or Wan21T2VParallelConfig()
            runtime = _init_wan21_t2v_runtime(parallel_cfg, explicit_device_id=device_id)
            seed = _broadcast_seed_if_needed(seed, runtime)
            pipeline, _ = _build_wan21_t2v_pipeline(
                wan21_root=wan21_root,
                ckpt_dir=ckpt_dir,
                task=task,
                runtime=runtime,
                parallel_cfg=parallel_cfg,
            )
            offload_model = _resolve_wan21_t2v_offload_model(runtime, offload_model)
            target_model = _unwrap_wan21_t2v_dit_model_for_runtime_patch(pipeline.model)
            num_self_heads_per_layer = {
                int(layer): int(target_model.blocks[int(layer)].self_attn.num_heads)
                for layer in selected_layers
            }
            selected_self_head_specs = _resolve_wan21_t2v_selected_head_specs_from_layer_counts(
                explicit_head_specs=trajectory_consensus_self_heads,
                num_heads_per_layer=num_self_heads_per_layer,
            )
            object_mask_vpred = _upsample_wan21_t2v_token_mask_to_vpred(
                token_mask_fhw=reference_support["support_mask_fhw"],
                patch_size=tuple(int(x) for x in target_model.patch_size),
            )
            object_mask_vpred = object_mask_vpred.float().cpu()
            object_token_mask_fhw = reference_support["support_mask_fhw"].detach().float().cpu()
            latent_frame_count = int(object_token_mask_fhw.shape[0])
            selected_latent_frame_indices = (
                list(range(latent_frame_count))
                if int(taylor_num_latent_frames) < 0
                else _resolve_wan21_t2v_candidate_viz_frame_indices(
                    frame_count=latent_frame_count,
                    num_frames=int(taylor_num_latent_frames),
                )
            )
            taylor_token_mask_fhw = _subset_wan21_t2v_token_mask_by_frames(
                token_mask_fhw=object_token_mask_fhw,
                selected_frame_indices=selected_latent_frame_indices,
            )
            taylor_object_mask_vpred = _upsample_wan21_t2v_token_mask_to_vpred(
                token_mask_fhw=taylor_token_mask_fhw,
                patch_size=tuple(int(x) for x in target_model.patch_size),
            ).float().cpu()
            taylor_token_indices = _token_grid_mask_to_wan21_t2v_sequence_indices(taylor_token_mask_fhw)

            clean_cache: Dict[Tuple[int, str], torch.Tensor] = {}
            selected_modules = [str(module).strip().lower() for module in trajectory_consensus_modules if str(module).strip()]
            selected_modules = list(dict.fromkeys(selected_modules))
            for module_name in selected_modules:
                if module_name not in {"cross", "self"}:
                    raise ValueError(f"Unsupported trajectory_consensus module: {module_name}")
            module_to_specs = {
                "cross": [(layer, head) for layer, head in selected_cross_head_specs if int(layer) in selected_layers],
                "self": [(layer, head) for layer, head in selected_self_head_specs if int(layer) in selected_layers],
            }
            summary["selected_self_head_specs"] = [f"L{layer}H{head}" for layer, head in selected_self_head_specs]
            selected_targets: Dict[int, Tuple[str, ...]] = {}
            for module_name in selected_modules:
                specs_in_scope = module_to_specs.get(module_name, [])
                if not specs_in_scope:
                    continue
                for layer_index in sorted(set(int(layer) for layer, _ in specs_in_scope)):
                    selected_targets.setdefault(int(layer_index), [])
                    if str(module_name) not in selected_targets[int(layer_index)]:
                        selected_targets[int(layer_index)].append(str(module_name))
            selected_targets = {
                int(layer): tuple(sorted(modules))
                for layer, modules in selected_targets.items()
                if modules
            }
            step_latent_cache: Dict[int, Dict[str, Any]] = {}
            if selected_targets and (do_ablation or do_direct_proxy):
                step_latent_cache = _run_wan21_t2v_collect_target_step_latents(
                    pipeline=pipeline,
                    prompt=prompt,
                    size=size,
                    frame_num=frame_num,
                    shift=shift,
                    sample_solver=sample_solver,
                    sampling_steps=sampling_steps,
                    guide_scale=guide_scale,
                    seed=seed,
                    target_steps=selected_steps,
                    offload_model=offload_model,
                )

            num_row_methods = int(do_ablation) + int(do_direct_proxy)
            total_head_tasks = int(
                sum(len(selected_steps) * len(module_to_specs.get(module_name, [])) for module_name in selected_modules)
                * max(0, num_row_methods)
            )
            contribution_progress_bar = None
            if total_head_tasks > 0:
                try:
                    from tqdm import tqdm
                    contribution_progress_bar = tqdm(
                        total=int(total_head_tasks),
                        desc="trajectory consensus head contribution",
                        unit="head",
                        leave=True,
                    )
                except Exception:
                    contribution_progress_bar = None
            try:
                for step in selected_steps:
                    step_latent_state = None
                    branch_context = None
                    if int(step) in step_latent_cache:
                        step_latent_state = step_latent_cache[int(step)]
                        branch_context = (
                            step_latent_state["context"]
                            if str(contribution_branch) == "cond"
                            else step_latent_state["context_null"]
                        )

                    for module_name in selected_modules:
                        specs_in_scope = module_to_specs.get(module_name, [])
                        if not specs_in_scope:
                            continue
                        module_selected_targets = {
                            int(layer): (str(module_name),)
                            for layer in sorted(set(int(layer) for layer, _ in specs_in_scope))
                        }
                        taylor_metrics_by_layer_module: Dict[Tuple[int, str], Dict[str, torch.Tensor]] = {}
                        all_proxy_metrics: Dict[Tuple[int, str], Dict[str, torch.Tensor]] = {}
                        if (
                            do_ablation
                            and contribution_method == "taylor_approx"
                            and module_selected_targets
                        ):
                            if step_latent_state is None or branch_context is None:
                                raise RuntimeError(f"Missing cached step latent for Taylor attribution at step={step}.")
                            global_attribution_state = _run_wan21_t2v_global_attribution_clean_forward(
                                pipeline=pipeline,
                                latent_input=step_latent_state["latent_input"],
                                timestep_value=step_latent_state["timestep"],
                                seq_len=int(step_latent_state["seq_len"]),
                                context=branch_context,
                                branch=str(contribution_branch),
                                target_step=1,
                                selected_targets=module_selected_targets,
                                attribution_position=str(ablate_position),
                                token_indices=taylor_token_indices if taylor_object_only else None,
                                use_gradient_checkpointing=bool(taylor_use_gradient_checkpointing),
                                offload_model=offload_model,
                            )
                            if global_attribution_state.captured_clean_vpred is None:
                                raise RuntimeError(
                                    f"Failed to capture global attribution clean v_pred for step={step}, module={module_name}."
                                )
                            clean_vpred_device = global_attribution_state.captured_clean_vpred
                            clean_uncond_vpred_cpu = None
                            if str(taylor_patching_metric) == "sem_obj":
                                uncond_key = (int(step), "uncond")
                                if uncond_key not in clean_cache:
                                    clean_cache[uncond_key] = _run_wan21_t2v_local_clean_vpred(
                                        pipeline=pipeline,
                                        latent_input=step_latent_state["latent_input"],
                                        timestep_value=step_latent_state["timestep"],
                                        seq_len=int(step_latent_state["seq_len"]),
                                        context=step_latent_state["context_null"],
                                    )
                                clean_uncond_vpred_cpu = clean_cache[uncond_key]
                            target_model.zero_grad(set_to_none=True)
                            scalar_metric = _build_wan21_t2v_taylor_scalar_metric(
                                clean_vpred=clean_vpred_device,
                                metric_scope=str(taylor_metric_scope),
                                patching_metric=str(taylor_patching_metric),
                                object_mask_fhw=object_mask_vpred,
                                clean_uncond_vpred=clean_uncond_vpred_cpu,
                            )
                            scalar_metric.backward(retain_graph=False)
                            for key, head_writes in global_attribution_state.captured_head_writes.items():
                                grad_obj = global_attribution_state.captured_head_writes_grad_obj.get(key, None)
                                if grad_obj is None:
                                    raise RuntimeError(
                                        "Missing global attribution gradient for "
                                        f"step={step}, layer={key[0]}, module={key[1]}."
                                    )
                                taylor_metrics_by_layer_module[key] = _compute_wan21_t2v_taylor_contribution_metrics(
                                    head_writes=head_writes,
                                    head_writes_grad=grad_obj,
                                    ablation_mode=str(taylor_ablation_mode),
                                    full_token_head_mean=global_attribution_state.captured_head_write_means.get(key),
                                )
                            global_attribution_state.captured_clean_vpred = None
                            global_attribution_state.captured_head_writes.clear()
                            global_attribution_state.captured_head_write_means.clear()
                            global_attribution_state.captured_head_writes_grad_obj.clear()
                            del scalar_metric
                            del clean_vpred_device
                            del global_attribution_state
                            target_model.zero_grad(set_to_none=True)
                            gc.collect()
                            if offload_model:
                                pipeline.model.cpu()
                                torch.cuda.empty_cache()
                        if do_direct_proxy and module_selected_targets:
                            if step_latent_state is None or branch_context is None:
                                raise RuntimeError(f"Missing cached step latent for direct proxy at step={step}.")
                            global_proxy_state = _run_wan21_t2v_global_direct_proxy_clean_forward(
                                pipeline=pipeline,
                                latent_input=step_latent_state["latent_input"],
                                timestep_value=step_latent_state["timestep"],
                                seq_len=int(step_latent_state["seq_len"]),
                                context=branch_context,
                                branch=str(contribution_branch),
                                target_step=1,
                                selected_targets=module_selected_targets,
                                offload_model=offload_model,
                            )
                            if (
                                global_proxy_state.captured_clean_vpred is None
                                or global_proxy_state.captured_head_e is None
                                or global_proxy_state.captured_grid_sizes is None
                            ):
                                raise RuntimeError(
                                    f"Failed to capture global direct proxy state for step={step}, module={module_name}."
                                )
                            all_proxy_metrics = _compute_wan21_t2v_global_direct_proxy_metrics(
                                pipeline=pipeline,
                                clean_vpred=global_proxy_state.captured_clean_vpred,
                                post_o_head_writes=global_proxy_state.captured_post_o_head_writes,
                                head_e=global_proxy_state.captured_head_e,
                                grid_sizes=global_proxy_state.captured_grid_sizes,
                                object_mask_fhw=object_mask_vpred,
                            )
                        layers_in_scope = sorted(set(int(layer) for layer, _ in specs_in_scope))
                        for layer in layers_in_scope:
                            layer_head_specs = [(int(l), int(h)) for l, h in specs_in_scope if int(l) == int(layer)]
                            if not layer_head_specs:
                                continue
                            clean_key = (int(step), str(contribution_branch))
                            if (
                                do_ablation
                                and contribution_method == "exact_ablation"
                                and clean_key not in clean_cache
                            ):
                                if step_latent_state is None or branch_context is None:
                                    raise RuntimeError(f"Missing cached step latent for exact ablation at step={step}.")
                                clean_cache[clean_key] = _run_wan21_t2v_local_clean_vpred(
                                    pipeline=pipeline,
                                    latent_input=step_latent_state["latent_input"],
                                    timestep_value=step_latent_state["timestep"],
                                    seq_len=int(step_latent_state["seq_len"]),
                                    context=branch_context,
                                )

                            for _, head in layer_head_specs:
                                head = int(head)
                                alignment_key = (int(layer), int(head))
                                if do_ablation and contribution_method == "exact_ablation":
                                    if step_latent_state is None or branch_context is None:
                                        raise RuntimeError(f"Missing cached step latent for exact ablation at step={step}.")
                                    clean_vpred = clean_cache[(int(step), str(contribution_branch))]
                                    ablate_state = _run_wan21_t2v_local_contribution_forward(
                                        pipeline=pipeline,
                                        latent_input=step_latent_state["latent_input"],
                                        timestep_value=step_latent_state["timestep"],
                                        seq_len=int(step_latent_state["seq_len"]),
                                        context=branch_context,
                                        branch=str(contribution_branch),
                                        target_layer=int(layer),
                                        target_head=int(head),
                                        target_module=str(module_name),
                                        ablate_position=str(ablate_position),
                                        ablate_head=True,
                                        capture_selected_head_write=False,
                                    )
                                    if ablate_state.captured_vpred is None:
                                        raise RuntimeError(
                                            f"Failed to capture ablated v_pred for step={step}, layer={layer}, head={head}, module={module_name}."
                                        )
                                    ablated_vpred = ablate_state.captured_vpred
                                    delta_vpred = clean_vpred - ablated_vpred
                                    metric_values = _compute_wan21_t2v_contribution_metrics(
                                        delta_vpred=delta_vpred,
                                        clean_vpred=clean_vpred,
                                        object_mask_fhw=object_mask_vpred,
                                        ablated_vpred=ablated_vpred,
                                    )
                                    row = {
                                        "step": int(step),
                                        "layer": int(layer),
                                        "head": int(head),
                                        "head_tag": f"L{int(layer)}H{int(head)}",
                                        "module": str(module_name),
                                        "branch": str(contribution_branch),
                                        "analysis_method": "exact_ablation",
                                        "contribution_method": str(contribution_method),
                                        "contribution": "",
                                        "cos_full": float(metric_values["cos_full"]),
                                        "dot_full": float(metric_values["dot_full"]),
                                        "cos_obj": float(metric_values["cos_obj"]),
                                        "dot_obj": float(metric_values["dot_obj"]),
                                        "ablate_cos_full": float(metric_values["ablate_cos_full"]),
                                        "ablate_dot_full": float(metric_values["ablate_dot_full"]),
                                        "ablate_cos_obj": float(metric_values["ablate_cos_obj"]),
                                        "ablate_dot_obj": float(metric_values["ablate_dot_obj"]),
                                        "proj_cos_full": "",
                                        "proj_dot_full": "",
                                        "proj_cos_obj": "",
                                        "proj_dot_obj": "",
                                        "proj_share_full": "",
                                        "proj_share_obj": "",
                                    }
                                    _apply_wan21_t2v_abs_to_ablation_contribution_row(row)
                                    if str(module_name) == "cross":
                                        for alignment_metric_name, early_alignment_summary in early_alignment_summaries.items():
                                            if alignment_key not in early_alignment_summary:
                                                continue
                                            for metric_name in ("cos_obj", "cos_full", "ablate_cos_obj", "ablate_cos_full"):
                                                _append_wan21_t2v_alignment_scatter_row(
                                                    scatter_rows_by_method["exact_ablation"],
                                                    alignment_metric_name=str(alignment_metric_name),
                                                    analysis_method="exact_ablation",
                                                    module_name=str(module_name),
                                                    branch_name=str(contribution_branch),
                                                    metric_name=str(metric_name),
                                                    head_tag=str(row["head_tag"]),
                                                    step=int(step),
                                                    metric_value=row[metric_name],
                                                    alignment_summary=early_alignment_summary[alignment_key],
                                                )
                                    head_contribution_rows_by_method["exact_ablation"].append(row)
                                    if contribution_progress_bar is not None:
                                        contribution_progress_bar.update(1)

                                if do_ablation and contribution_method == "taylor_approx":
                                    attribution_metrics = taylor_metrics_by_layer_module.get((int(layer), str(module_name)), None)
                                    if attribution_metrics is None:
                                        raise RuntimeError(
                                            f"Missing global attribution metrics for step={step}, layer={layer}, module={module_name}."
                                        )
                                    contribution_value = float(attribution_metrics["contribution"][int(head)].item())
                                    row = {
                                        "step": int(step),
                                        "layer": int(layer),
                                        "head": int(head),
                                        "head_tag": f"L{int(layer)}H{int(head)}",
                                        "module": str(module_name),
                                        "branch": str(contribution_branch),
                                        "analysis_method": "taylor_approx",
                                        "contribution_method": str(contribution_method),
                                        "contribution": contribution_value,
                                        "cos_full": "",
                                        "dot_full": "",
                                        "cos_obj": "",
                                        "dot_obj": "",
                                        "ablate_cos_full": "",
                                        "ablate_dot_full": "",
                                        "ablate_cos_obj": "",
                                        "ablate_dot_obj": "",
                                        "proj_cos_full": "",
                                        "proj_dot_full": "",
                                        "proj_cos_obj": "",
                                        "proj_dot_obj": "",
                                        "proj_share_full": "",
                                        "proj_share_obj": "",
                                    }
                                    _apply_wan21_t2v_abs_to_ablation_contribution_row(row)
                                    if str(module_name) == "cross":
                                        for alignment_metric_name, early_alignment_summary in early_alignment_summaries.items():
                                            if alignment_key not in early_alignment_summary:
                                                continue
                                            _append_wan21_t2v_alignment_scatter_row(
                                                scatter_rows_by_method["taylor_approx"],
                                                alignment_metric_name=str(alignment_metric_name),
                                                analysis_method="taylor_approx",
                                                module_name=str(module_name),
                                                branch_name=str(contribution_branch),
                                                metric_name="contribution",
                                                head_tag=str(row["head_tag"]),
                                                step=int(step),
                                                metric_value=contribution_value,
                                                alignment_summary=early_alignment_summary[alignment_key],
                                            )
                                    head_contribution_rows_by_method["taylor_approx"].append(row)
                                    if contribution_progress_bar is not None:
                                        contribution_progress_bar.update(1)

                                if do_direct_proxy:
                                    proxy_metrics = all_proxy_metrics.get((int(layer), str(module_name)), None)
                                    if proxy_metrics is None:
                                        raise RuntimeError(
                                            f"Missing direct projection cache for step={step}, layer={layer}, module={module_name}."
                                        )
                                    row = {
                                        "step": int(step),
                                        "layer": int(layer),
                                        "head": int(head),
                                        "head_tag": f"L{int(layer)}H{int(head)}",
                                        "module": str(module_name),
                                        "branch": str(contribution_branch),
                                        "analysis_method": "direct_proxy",
                                        "contribution_method": "",
                                        "contribution": "",
                                        "cos_full": "",
                                        "dot_full": "",
                                        "cos_obj": "",
                                        "dot_obj": "",
                                        "ablate_cos_full": "",
                                        "ablate_dot_full": "",
                                        "ablate_cos_obj": "",
                                        "ablate_dot_obj": "",
                                        "proj_cos_full": float(proxy_metrics["proj_cos_full"][int(head)].item()),
                                        "proj_dot_full": float(proxy_metrics["proj_dot_full"][int(head)].item()),
                                        "proj_cos_obj": float(proxy_metrics["proj_cos_obj"][int(head)].item()),
                                        "proj_dot_obj": float(proxy_metrics["proj_dot_obj"][int(head)].item()),
                                        "proj_share_full": "",
                                        "proj_share_obj": "",
                                    }
                                    if str(module_name) == "cross":
                                        for alignment_metric_name, early_alignment_summary in early_alignment_summaries.items():
                                            if alignment_key not in early_alignment_summary:
                                                continue
                                            for metric_name in ("proj_dot_obj", "proj_dot_full", "proj_cos_obj", "proj_cos_full"):
                                                _append_wan21_t2v_alignment_scatter_row(
                                                    scatter_rows_by_method["direct_proxy"],
                                                    alignment_metric_name=str(alignment_metric_name),
                                                    analysis_method="direct_proxy",
                                                    module_name=str(module_name),
                                                    branch_name=str(contribution_branch),
                                                    metric_name=str(metric_name),
                                                    head_tag=str(row["head_tag"]),
                                                    step=int(step),
                                                    metric_value=row[metric_name],
                                                    alignment_summary=early_alignment_summary[alignment_key],
                                                )
                                    head_contribution_rows_by_method["direct_proxy"].append(row)
                                    if contribution_progress_bar is not None:
                                        contribution_progress_bar.update(1)
            finally:
                if contribution_progress_bar is not None:
                    contribution_progress_bar.close()

            share_groups = defaultdict(list)
            for row in head_contribution_rows_by_method.get("direct_proxy", []):
                share_groups[(int(row["step"]), int(row["layer"]), str(row["module"]), str(row["branch"]))].append(row)
            for _, rows in share_groups.items():
                positive_full_sum = sum(max(0.0, float(row["proj_dot_full"])) for row in rows if row["proj_dot_full"] != "")
                positive_obj_sum = sum(max(0.0, float(row["proj_dot_obj"])) for row in rows if row["proj_dot_obj"] != "")
                for row in rows:
                    if row["proj_dot_full"] != "":
                        row["proj_share_full"] = float(max(0.0, float(row["proj_dot_full"])) / max(positive_full_sum, 1e-12))
                    if row["proj_dot_obj"] != "":
                        row["proj_share_obj"] = float(max(0.0, float(row["proj_dot_obj"])) / max(positive_obj_sum, 1e-12))

            for analysis_method, method_rows in sorted(head_contribution_rows_by_method.items()):
                method_output_dir = head_contribution_output_dirs.get(str(analysis_method))
                if not method_output_dir:
                    continue
                _ensure_dir(method_output_dir)
                _save_csv(
                    os.path.join(method_output_dir, "trajectory_consensus_head_contribution.csv"),
                    method_rows,
                )

        for analysis_method, method_rows in sorted(head_contribution_rows_by_method.items()):
            method_output_dir = head_contribution_output_dirs.get(str(analysis_method))
            if not method_output_dir:
                continue
            if filter_heads:
                filtered_head_path = _export_wan21_t2v_filtered_heads(
                    method_output_dir,
                    method_rows,
                    analysis_method=str(analysis_method),
                    module_name=str(filter_module),
                    branch_name=str(contribution_branch),
                    reuse_head_evolution_dir=reuse_head_evolution_dir,
                    alignment_summary_steps=int(alignment_summary_steps),
                    filter_step=int(filter_step),
                    convergence_speed_rule=str(filter_convergence_speed_rule["text"]),
                    contribution_rule=str(filter_contribution_rule["text"]),
                    contribution_metric_name=str(filter_contribution_metric),
                )
                if filtered_head_path:
                    plot_paths.append(filtered_head_path)
            plot_paths.extend(
                _render_wan21_t2v_head_contribution_plots(
                    method_output_dir,
                    method_rows,
                    scatter_rows_by_method.get(str(analysis_method), []),
                    scatter_outlier_heads_by_module=scatter_outlier_heads_by_module,
                    alignment_summary_steps=int(alignment_summary_steps),
                    skip_existing_plots=bool(trajectory_consensus_skip_existing_plots),
                )
            )

    summary.update({
        "num_candidate_region_rows": int(len(candidate_region_rows)),
        "num_candidate_weight_rows": int(len(candidate_weight_rows)),
        "num_winner_gap_rows": int(len(winner_gap_rows)),
        "num_head_contribution_rows": int(sum(len(rows) for rows in head_contribution_rows_by_method.values())),
        "num_self_attention_coupling_pairwise_rows": int(len(self_attention_coupling_pairwise_rows)),
        "num_self_attention_coupling_candidate_feature_rows": int(len(self_attention_coupling_candidate_feature_rows)),
        "num_self_attention_coupling_feature_summary_rows": int(len(self_attention_coupling_feature_summary_rows)),
        "num_self_attention_coupling_temporal_precedence_rows": int(len(self_attention_coupling_temporal_precedence_rows)),
        "plot_paths": list(plot_paths),
    })

    method_summaries: Dict[str, str] = {}
    for analysis_method, method_rows in sorted(head_contribution_rows_by_method.items()):
        method_output_dir = head_contribution_output_dirs.get(str(analysis_method))
        if not method_output_dir:
            continue
        method_plot_paths = [path for path in plot_paths if str(path).startswith(method_output_dir)]
        method_summary_path = os.path.join(method_output_dir, "trajectory_consensus_summary.json")
        previous_method_summary: Dict[str, object] = {}
        if os.path.exists(method_summary_path):
            try:
                with open(method_summary_path, "r", encoding="utf-8") as handle:
                    loaded_summary = json.load(handle)
                if isinstance(loaded_summary, dict):
                    previous_method_summary = loaded_summary
            except Exception:
                previous_method_summary = {}

        method_summary = dict(summary)
        method_summary.update({
            "analysis_method": str(analysis_method),
            "trajectory_consensus_do_ablation": bool(analysis_method in {"exact_ablation", "taylor_approx"}),
            "trajectory_consensus_do_direct_proxy": bool(analysis_method == "direct_proxy"),
            "trajectory_consensus_contribution_method": str(analysis_method),
            "head_contribution_output_dir": str(method_output_dir),
            "head_contribution_plot_paths": list(method_plot_paths),
            "num_head_contribution_rows": int(len(method_rows)) if method_rows else int(previous_method_summary.get("num_head_contribution_rows", 0)),
            "num_candidate_region_rows": int(previous_method_summary.get("num_candidate_region_rows", summary.get("num_candidate_region_rows", 0))),
            "num_candidate_weight_rows": int(previous_method_summary.get("num_candidate_weight_rows", summary.get("num_candidate_weight_rows", 0))),
            "num_winner_gap_rows": int(previous_method_summary.get("num_winner_gap_rows", summary.get("num_winner_gap_rows", 0))),
        })
        if previous_method_summary.get("stages"):
            method_summary["stages"] = list(dict.fromkeys(
                [str(stage) for stage in previous_method_summary.get("stages", [])]
                + [str(stage) for stage in summary.get("stages", [])]
            ))
        _ensure_dir(method_output_dir)
        _save_json(method_summary_path, method_summary)
        method_summaries[str(analysis_method)] = method_summary_path

    summary["method_summaries"] = method_summaries
    return summary
