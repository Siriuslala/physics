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
    _js_wan21_t2v_distance_per_frame,
    _load_wan21_t2v_cross_attention_mean_maps_from_disk,
    _load_wan21_t2v_cross_attention_token_meta,
    _mean_wan21_t2v_head_maps_for_words,
    _normalize_wan21_t2v_attention_map_per_frame,
    _parse_wan21_t2v_layer_head_specs,
    _save_csv,
    _save_json,
    _soft_center_wan21_t2v_attention_map,
    _trajectory_distance_wan21_t2v_soft_centers,
    _wasserstein_approx_wan21_t2v_distance_per_frame,
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

def _plot_wan21_t2v_head_trajectory_soft_centers(
    center_rows: Sequence[Dict[str, object]],
    save_file: str,
    title: str,
):
    """Plot soft-center trajectories for selected heads on a token-grid plane."""
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
        all_y.append(float(row["soft_center_y"]))
        all_x.append(float(row["soft_center_x"]))

    fig, axis = plt.subplots(1, 1, figsize=(7.2, 6.0))
    color_map = plt.get_cmap("tab20", max(1, len(grouped)))
    for color_index, head_tag in enumerate(sorted(grouped.keys())):
        rows_for_head = sorted(grouped[head_tag], key=lambda row: int(row["frame"]))
        ys = [float(row["soft_center_y"]) for row in rows_for_head]
        xs = [float(row["soft_center_x"]) for row in rows_for_head]
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
        requested_distance_metrics = ["js", "wasserstein"]
    requested_distance_metrics = list(dict.fromkeys(requested_distance_metrics))
    for metric_name in requested_distance_metrics:
        if metric_name not in {"js", "wasserstein"}:
            raise ValueError(
                "head_trajectory_dynamics_distance_metrics must be chosen from {'js', 'wasserstein'}, "
                f"got `{metric_name}`."
            )

    parsed_heads = _parse_wan21_t2v_layer_head_specs(head_trajectory_dynamics_heads)
    requested_head_set = set(parsed_heads)

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
    final_reference_soft_center = _soft_center_wan21_t2v_attention_map(reference_probability_map)

    head_map_records = []
    pairwise_rows = []
    consensus_rows = []
    attractor_rows = []
    final_distance_rows = []
    soft_center_rows = []

    probability_maps_by_step_layer_head: Dict[Tuple[int, int, int], torch.Tensor] = {}
    soft_centers_by_step_layer_head: Dict[Tuple[int, int, int], torch.Tensor] = {}

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
                soft_center = _soft_center_wan21_t2v_attention_map(probability_map)
                key = (int(step_index), int(layer_index), int(head_index))
                probability_maps_by_step_layer_head[key] = probability_map
                soft_centers_by_step_layer_head[key] = soft_center
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
                final_distance_rows.append(
                    {
                        "step": int(step_index),
                        "layer": int(layer_index),
                        "head": int(head_index),
                        "head_tag": f"L{int(layer_index)}H{int(head_index)}",
                        "distance_to_final_soft_center": _trajectory_distance_wan21_t2v_soft_centers(
                            soft_center,
                            final_reference_soft_center,
                        ),
                    }
                )
                for frame_index in range(int(soft_center.size(0))):
                    soft_center_rows.append(
                        {
                            "step": int(step_index),
                            "layer": int(layer_index),
                            "head": int(head_index),
                            "head_tag": f"L{int(layer_index)}H{int(head_index)}",
                            "frame": int(frame_index),
                            "soft_center_y": float(soft_center[frame_index, 0].item()),
                            "soft_center_x": float(soft_center[frame_index, 1].item()),
                        }
                    )

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
                if "wasserstein" in requested_distance_metrics:
                    wasserstein_distance = _wasserstein_approx_wan21_t2v_distance_per_frame(prob_i, prob_j).mean().item()
                    row["wasserstein_distance"] = float(wasserstein_distance)
                    metric_to_pairwise_values["wasserstein"].append(float(wasserstein_distance))
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
    adjacent_step_pairs = list(zip(sorted_resolved_steps[:-1], sorted_resolved_steps[1:]))
    for step_index, next_step in adjacent_step_pairs:
        for layer_index in available_layers:
            head_indices = per_step_layer_heads.get((int(step_index), int(layer_index)), [])
            next_head_indices = per_step_layer_heads.get((int(next_step), int(layer_index)), [])
            if not head_indices or not next_head_indices:
                continue
            next_head_indices_set = set(next_head_indices)
            for leader_head in head_indices:
                if int(leader_head) not in next_head_indices_set:
                    continue
                leader_traj = soft_centers_by_step_layer_head[(int(step_index), int(layer_index), int(leader_head))]
                deltas = []
                for follower_head in head_indices:
                    if int(follower_head) == int(leader_head):
                        continue
                    if int(follower_head) not in next_head_indices_set:
                        continue
                    follower_current = soft_centers_by_step_layer_head[(int(step_index), int(layer_index), int(follower_head))]
                    follower_next = soft_centers_by_step_layer_head[(int(next_step), int(layer_index), int(follower_head))]
                    current_distance = _trajectory_distance_wan21_t2v_soft_centers(follower_current, leader_traj)
                    next_distance = _trajectory_distance_wan21_t2v_soft_centers(follower_next, leader_traj)
                    deltas.append(float(current_distance - next_distance))
                if not deltas:
                    continue
                attractor_rows.append(
                    {
                        "step": int(step_index),
                        "next_step": int(next_step),
                        "layer": int(layer_index),
                        "head": int(leader_head),
                        "head_tag": f"L{int(layer_index)}H{int(leader_head)}",
                        "attractor_score_mean": float(sum(deltas) / len(deltas)),
                        "attractor_score_max": float(max(deltas)),
                        "attractor_score_min": float(min(deltas)),
                        "num_followers": int(len(deltas)),
                    }
                )

    _save_csv(os.path.join(output_dir, "head_trajectory_dynamics_head_maps.csv"), head_map_records)
    _save_csv(os.path.join(output_dir, "head_trajectory_dynamics_pairwise.csv"), pairwise_rows)
    _save_csv(os.path.join(output_dir, "head_trajectory_dynamics_consensus.csv"), consensus_rows)
    _save_csv(os.path.join(output_dir, "head_trajectory_dynamics_attractor.csv"), attractor_rows)
    _save_csv(os.path.join(output_dir, "head_trajectory_dynamics_final_distance.csv"), final_distance_rows)
    _save_csv(os.path.join(output_dir, "head_trajectory_dynamics_soft_centers.csv"), soft_center_rows)

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

    for layer_index in available_layers:
        layer_attractor_rows = [row for row in attractor_rows if int(row["layer"]) == int(layer_index)]
        if not layer_attractor_rows:
            continue
        by_head: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for row in layer_attractor_rows:
            by_head[str(row["head_tag"])].append(row)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axis = plt.subplots(1, 1, figsize=(8.2, 5.0))
        all_values = []
        head_tags = sorted(by_head.keys())
        color_map = plt.get_cmap("tab20", max(1, len(head_tags)))
        for color_index, head_tag in enumerate(head_tags):
            head_rows = sorted(by_head[head_tag], key=lambda row: int(row["step"]))
            x_steps = [int(row["step"]) for row in head_rows]
            y_values = [float(row["attractor_score_mean"]) for row in head_rows]
            all_values.extend(y_values)
            axis.plot(
                x_steps,
                y_values,
                linewidth=1.4,
                alpha=0.92,
                color=color_map(color_index),
                label=head_tag,
            )
        axis.set_title(f"Head Attractor Score | layer={int(layer_index)}")
        axis.set_xlabel("diffusion step")
        axis.set_ylabel("attractor score")
        axis.grid(alpha=0.22, linestyle="--")
        if len(head_tags) <= 20:
            axis.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        plot_path = os.path.join(
            plots_dir,
            "attractor_curves",
            f"attractor_layer_{int(layer_index):02d}.pdf",
        )
        _ensure_dir(os.path.dirname(plot_path))
        fig.savefig(plot_path, format="pdf")
        plt.close(fig)
        plot_paths.append(plot_path)

    selected_soft_center_groups = defaultdict(list)
    for row in soft_center_rows:
        step_index = int(row["step"])
        layer_index = int(row["layer"])
        if int(step_index) not in set(resolved_steps[: min(len(resolved_steps), 6)]):
            continue
        selected_soft_center_groups[(step_index, layer_index)].append(row)
    for (step_index, layer_index), rows_for_plot in sorted(selected_soft_center_groups.items()):
        plot_path = _plot_wan21_t2v_head_trajectory_soft_centers(
            center_rows=rows_for_plot,
            save_file=os.path.join(
                plots_dir,
                "soft_center_spaghetti",
                f"soft_centers_step_{int(step_index):03d}_layer_{int(layer_index):02d}.pdf",
            ),
            title=f"Head Soft Trajectories | step={int(step_index)} layer={int(layer_index)}",
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
        "num_head_records": int(len(head_map_records)),
        "num_pairwise_rows": int(len(pairwise_rows)),
        "num_consensus_rows": int(len(consensus_rows)),
        "num_attractor_rows": int(len(attractor_rows)),
        "num_final_distance_rows": int(len(final_distance_rows)),
        "num_soft_center_rows": int(len(soft_center_rows)),
        "plot_paths": plot_paths,
        "head_maps_csv": os.path.join(output_dir, "head_trajectory_dynamics_head_maps.csv"),
        "pairwise_csv": os.path.join(output_dir, "head_trajectory_dynamics_pairwise.csv"),
        "consensus_csv": os.path.join(output_dir, "head_trajectory_dynamics_consensus.csv"),
        "attractor_csv": os.path.join(output_dir, "head_trajectory_dynamics_attractor.csv"),
        "final_distance_csv": os.path.join(output_dir, "head_trajectory_dynamics_final_distance.csv"),
        "soft_centers_csv": os.path.join(output_dir, "head_trajectory_dynamics_soft_centers.csv"),
    }
    _save_json(os.path.join(output_dir, "head_trajectory_dynamics_summary.json"), summary)
    if dist.is_initialized():
        dist.barrier()
    return summary
