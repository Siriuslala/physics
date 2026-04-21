"""Wan2.1-T2V experiment: head_evolution.

Main entry:
- run_wan21_t2v_head_evolution

This module analyzes how cross-attention head patterns evolve over denoising
steps, including entropy, trajectory support quality, concentrated-region score,
and head categorization. Shared runtime and map helpers come from utils.py;
head-evolution-specific metrics and plots stay local here.
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
    _compute_wan21_t2v_spatial_entropy_stats,
    _dedup_wan21_t2v_int_list,
    _ensure_dir,
    _load_wan21_t2v_cross_attention_mean_maps_from_disk,
    _load_wan21_t2v_cross_attention_token_meta,
    _mean_wan21_t2v_head_maps_for_words,
    _resolve_wan21_t2v_viz_frame_indices,
    _save_csv,
    _save_json,
)

def _extract_wan21_t2v_reference_peak_and_centroid_trajectory(
    map_fhw: torch.Tensor,
    power: float = 1.5,
    quantile: float = 0.8,
) -> Dict[str, List]:
    """Extract per-frame peak/centroid/geometric centers and component areas from one [F, H, W] map.

    The component is defined as the connected region (8-neighborhood) that contains
    the frame peak after quantile thresholding.
    """
    if map_fhw.dim() != 3:
        raise ValueError(f"Expected [F, H, W], got shape={tuple(map_fhw.shape)}")

    frame_count, token_grid_height, token_grid_width = map_fhw.shape
    threshold_quantile = max(0.0, min(1.0, float(quantile)))
    sharpen_power = max(1e-6, float(power))

    peak_centers: List[Tuple[float, float]] = []
    centroid_centers: List[Tuple[float, float]] = []
    geometric_centers: List[Tuple[float, float]] = []
    component_areas: List[int] = []

    # 8-neighborhood offsets.
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for frame_idx in range(frame_count):
        attention_frame = map_fhw[frame_idx].detach().float().clamp_min(0.0)
        flattened_attention = attention_frame.reshape(-1)
        peak_flat_index = int(flattened_attention.argmax().item())
        peak_y = int(peak_flat_index // token_grid_width)
        peak_x = int(peak_flat_index % token_grid_width)

        if threshold_quantile > 0.0:
            quantile_value = float(torch.quantile(flattened_attention, threshold_quantile).item())
            threshold_mask = attention_frame >= quantile_value
        else:
            threshold_mask = torch.ones_like(attention_frame, dtype=torch.bool)

        # Ensure the peak location is always preserved in the threshold mask.
        if not bool(threshold_mask[peak_y, peak_x].item()):
            threshold_mask = threshold_mask.clone()
            threshold_mask[peak_y, peak_x] = True

        visited_mask = torch.zeros_like(threshold_mask, dtype=torch.bool)
        component_mask = torch.zeros_like(threshold_mask, dtype=torch.bool)
        search_queue = deque([(peak_y, peak_x)])
        visited_mask[peak_y, peak_x] = True

        while search_queue:
            current_y, current_x = search_queue.popleft()
            if not bool(threshold_mask[current_y, current_x].item()):
                continue
            component_mask[current_y, current_x] = True
            for delta_y, delta_x in neighbors:
                next_y = current_y + delta_y
                next_x = current_x + delta_x
                if next_y < 0 or next_y >= token_grid_height or next_x < 0 or next_x >= token_grid_width:
                    continue
                if bool(visited_mask[next_y, next_x].item()):
                    continue
                visited_mask[next_y, next_x] = True
                search_queue.append((next_y, next_x))

        component_y_indices, component_x_indices = component_mask.nonzero(as_tuple=True)
        if component_y_indices.numel() == 0:
            peak_centers.append((float(peak_y), float(peak_x)))
            centroid_centers.append((float(peak_y), float(peak_x)))
            geometric_centers.append((float(peak_y), float(peak_x)))
            component_areas.append(1)
            continue

        component_areas.append(int(component_y_indices.numel()))
        peak_centers.append((float(peak_y), float(peak_x)))
        geometric_center_y = float(component_y_indices.float().mean().item())
        geometric_center_x = float(component_x_indices.float().mean().item())
        geometric_centers.append((geometric_center_y, geometric_center_x))

        component_weights = attention_frame[component_y_indices, component_x_indices].pow(sharpen_power)
        weight_sum = float(component_weights.sum().item())
        if weight_sum <= 1e-12:
            centroid_centers.append((float(peak_y), float(peak_x)))
            continue

        centroid_y = float((component_y_indices.float() * component_weights).sum().item() / weight_sum)
        centroid_x = float((component_x_indices.float() * component_weights).sum().item() / weight_sum)
        centroid_centers.append((centroid_y, centroid_x))

    return {
        "peak_centers": peak_centers,
        "centroid_centers": centroid_centers,
        "geometric_centers": geometric_centers,
        "component_areas": component_areas,
        "frame_count": int(frame_count),
        "token_grid_height": int(token_grid_height),
        "token_grid_width": int(token_grid_width),
    }

def _build_wan21_t2v_trajectory_support_mask_from_centers(
    center_trajectory: Sequence[Tuple[float, float]],
    component_areas: Sequence[int],
    token_grid_height: int,
    token_grid_width: int,
    support_radius_mode: str = "adaptive_area",
    support_radius_fixed: float = 2.0,
    support_radius_alpha: float = 1.5,
    support_radius_min: float = 1.0,
    support_radius_max_ratio: float = 0.25,
) -> Tuple[torch.Tensor, List[float]]:
    """Build frame-wise circular support masks around trajectory centers.

    Returns:
        support_mask_fhw: Float mask [F, H, W], values in {0, 1}.
        support_radius_per_frame: Radius (token-grid unit) for each frame.
    """
    frame_count = len(center_trajectory)
    if frame_count <= 0:
        raise ValueError("center_trajectory is empty.")
    if len(component_areas) != frame_count:
        raise ValueError(
            "component_areas must have the same length as center_trajectory, "
            f"got {len(component_areas)} vs {frame_count}."
        )

    support_radius_mode = str(support_radius_mode).strip().lower()
    if support_radius_mode not in {"fixed", "adaptive_area"}:
        raise ValueError(
            "support_radius_mode must be one of {'fixed', 'adaptive_area'}, "
            f"got: {support_radius_mode}"
        )

    minimum_radius = max(1e-6, float(support_radius_min))
    maximum_radius = max(
        minimum_radius,
        float(support_radius_max_ratio) * float(max(1, min(int(token_grid_height), int(token_grid_width)))),
    )
    fixed_radius = float(max(minimum_radius, min(maximum_radius, float(support_radius_fixed))))

    grid_y = torch.arange(int(token_grid_height), dtype=torch.float32).view(int(token_grid_height), 1)
    grid_x = torch.arange(int(token_grid_width), dtype=torch.float32).view(1, int(token_grid_width))

    support_mask_fhw = torch.zeros((frame_count, int(token_grid_height), int(token_grid_width)), dtype=torch.float32)
    support_radius_per_frame: List[float] = []

    for frame_idx, (center_y, center_x) in enumerate(center_trajectory):
        if support_radius_mode == "fixed":
            support_radius = fixed_radius
        else:
            # Equivalent-circle radius from connected-component area, then scaled by alpha.
            component_area = max(1.0, float(component_areas[frame_idx]))
            equivalent_circle_radius = math.sqrt(component_area / math.pi)
            support_radius = float(support_radius_alpha) * equivalent_circle_radius
            support_radius = float(max(minimum_radius, min(maximum_radius, support_radius)))

        support_radius_per_frame.append(float(support_radius))
        squared_distance = (grid_y - float(center_y)).pow(2) + (grid_x - float(center_x)).pow(2)
        support_mask_fhw[frame_idx] = (squared_distance <= (support_radius ** 2)).float()

    return support_mask_fhw, support_radius_per_frame

def _extract_wan21_t2v_connected_components(
    binary_mask_hw: torch.Tensor,
) -> List[List[Tuple[int, int]]]:
    """Extract 8-neighborhood connected components from one binary [H, W] mask."""
    if binary_mask_hw.dim() != 2:
        raise ValueError(f"Expected [H, W], got shape={tuple(binary_mask_hw.shape)}")

    token_grid_height, token_grid_width = binary_mask_hw.shape
    visited_mask = torch.zeros_like(binary_mask_hw, dtype=torch.bool)
    components: List[List[Tuple[int, int]]] = []
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

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

def _preprocess_wan21_t2v_attention_map_fhw(
    map_fhw: torch.Tensor,
    winsorize_quantile: float = 0.995,
    despike_quantile: float = 0.98,
    min_component_area: int = 2,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Preprocess one [F, H, W] attention map with winsorization and spatial despiking."""
    if map_fhw.dim() != 3:
        raise ValueError(f"Expected [F, H, W], got shape={tuple(map_fhw.shape)}")

    attention_map = map_fhw.detach().float().clamp_min(0.0)
    frame_count, token_grid_height, token_grid_width = attention_map.shape

    q_winsor = max(0.0, min(1.0, float(winsorize_quantile)))
    q_despike = max(0.0, min(1.0, float(despike_quantile)))
    area_threshold = max(1, int(min_component_area))

    cleaned_map = attention_map.clone()
    total_despiked_points = 0
    total_positive_points = 0
    winsor_thresholds: List[float] = []

    for frame_index in range(int(frame_count)):
        frame_map = cleaned_map[frame_index]
        flattened_frame = frame_map.reshape(-1)

        if q_winsor < 1.0:
            winsor_threshold = float(torch.quantile(flattened_frame, q_winsor).item())
            winsor_thresholds.append(winsor_threshold)
            frame_map = frame_map.clamp_max(winsor_threshold)
        else:
            winsor_thresholds.append(float(flattened_frame.max().item()))

        if q_despike <= 0.0:
            cleaned_map[frame_index] = frame_map
            continue

        despike_threshold = float(torch.quantile(frame_map.reshape(-1), q_despike).item())
        high_value_mask = frame_map >= despike_threshold
        positive_point_count = int(high_value_mask.sum().item())
        total_positive_points += positive_point_count
        if positive_point_count <= 0:
            cleaned_map[frame_index] = frame_map
            continue

        # Keep only large connected high-value components; remove tiny isolated spikes.
        kept_mask = torch.zeros_like(high_value_mask, dtype=torch.bool)
        components = _extract_wan21_t2v_connected_components(high_value_mask)
        for component in components:
            if len(component) >= area_threshold:
                for point_y, point_x in component:
                    kept_mask[point_y, point_x] = True

        removed_mask = high_value_mask & (~kept_mask)
        removed_count = int(removed_mask.sum().item())
        total_despiked_points += removed_count
        if removed_count > 0:
            frame_map = frame_map.masked_fill(removed_mask, 0.0)

        cleaned_map[frame_index] = frame_map

    preprocess_stats = {
        "winsorize_quantile": float(q_winsor),
        "despike_quantile": float(q_despike),
        "min_component_area": int(area_threshold),
        "winsor_threshold_mean": float(sum(winsor_thresholds) / max(1, len(winsor_thresholds))),
        "despiked_points_total": int(total_despiked_points),
        "high_value_points_total": int(total_positive_points),
        "despike_removed_ratio": float(
            float(total_despiked_points) / float(max(1, total_positive_points))
        ),
    }
    return cleaned_map, preprocess_stats

def _compute_wan21_t2v_concentrated_region_score_stats(
    probability_map_fhw: torch.Tensor,
    top_ratio: float = 0.05,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Compute concentrated-region score from one probability-like [F, H, W] map."""
    if probability_map_fhw.dim() != 3:
        raise ValueError(f"Expected [F, H, W], got shape={tuple(probability_map_fhw.shape)}")

    probability_map = probability_map_fhw.detach().float().clamp_min(0.0)
    frame_count, token_grid_height, token_grid_width = probability_map.shape
    spatial_size = int(token_grid_height * token_grid_width)
    top_ratio = max(1e-6, min(1.0, float(top_ratio)))
    top_k = max(1, int(math.ceil(float(spatial_size) * top_ratio)))

    frame_scores: List[float] = []
    frame_intensity_scores: List[float] = []
    frame_connectivity_scores: List[float] = []
    frame_compactness_scores: List[float] = []
    diagonal_length = math.sqrt(float((token_grid_height - 1) ** 2 + (token_grid_width - 1) ** 2))
    diagonal_length = max(diagonal_length, 1e-6)

    for frame_index in range(int(frame_count)):
        frame_map = probability_map[frame_index]
        flattened_frame = frame_map.reshape(-1)
        total_mass = float(flattened_frame.sum().item())
        if total_mass <= float(eps):
            frame_scores.append(0.0)
            frame_intensity_scores.append(0.0)
            frame_connectivity_scores.append(0.0)
            frame_compactness_scores.append(0.0)
            continue

        frame_probabilities = flattened_frame / max(total_mass, float(eps))
        top_values, top_indices = torch.topk(frame_probabilities, k=min(top_k, int(frame_probabilities.numel())))
        top_mass = float(top_values.sum().item())

        random_baseline = float(top_values.numel()) / float(max(1, spatial_size))
        if random_baseline >= 1.0:
            intensity_score = 0.0
        else:
            intensity_score = (top_mass - random_baseline) / max(1e-12, (1.0 - random_baseline))
        intensity_score = float(max(0.0, min(1.0, intensity_score)))

        top_mask = torch.zeros((int(token_grid_height), int(token_grid_width)), dtype=torch.bool)
        for flat_index in top_indices.tolist():
            point_y = int(flat_index // int(token_grid_width))
            point_x = int(flat_index % int(token_grid_width))
            top_mask[point_y, point_x] = True

        components = _extract_wan21_t2v_connected_components(top_mask)
        if not components:
            frame_scores.append(0.0)
            frame_intensity_scores.append(float(intensity_score))
            frame_connectivity_scores.append(0.0)
            frame_compactness_scores.append(0.0)
            continue

        component_masses: List[float] = []
        for component in components:
            component_mass = 0.0
            for point_y, point_x in component:
                component_mass += float(frame_probabilities[point_y * int(token_grid_width) + point_x].item())
            component_masses.append(float(component_mass))
        largest_component_index = int(torch.tensor(component_masses, dtype=torch.float32).argmax().item())
        largest_component = components[largest_component_index]
        largest_component_mass = float(component_masses[largest_component_index])
        connectivity_score = float(largest_component_mass / max(top_mass, float(eps)))
        connectivity_score = float(max(0.0, min(1.0, connectivity_score)))

        component_weights = []
        component_y_coords = []
        component_x_coords = []
        for point_y, point_x in largest_component:
            component_weights.append(float(frame_probabilities[point_y * int(token_grid_width) + point_x].item()))
            component_y_coords.append(float(point_y))
            component_x_coords.append(float(point_x))
        weight_sum = float(sum(component_weights))
        if weight_sum <= float(eps):
            compactness_score = 0.0
        else:
            center_y = sum(weight * coord for weight, coord in zip(component_weights, component_y_coords)) / weight_sum
            center_x = sum(weight * coord for weight, coord in zip(component_weights, component_x_coords)) / weight_sum
            mean_squared_radius = sum(
                weight * ((coord_y - center_y) ** 2 + (coord_x - center_x) ** 2)
                for weight, coord_y, coord_x in zip(component_weights, component_y_coords, component_x_coords)
            ) / weight_sum
            normalized_radius = math.sqrt(max(0.0, mean_squared_radius)) / float(diagonal_length)
            compactness_score = float(max(0.0, min(1.0, 1.0 - normalized_radius)))

        frame_score = float(intensity_score * connectivity_score * compactness_score)
        frame_scores.append(frame_score)
        frame_intensity_scores.append(float(intensity_score))
        frame_connectivity_scores.append(float(connectivity_score))
        frame_compactness_scores.append(float(compactness_score))

    score_tensor = torch.tensor(frame_scores, dtype=torch.float32)
    intensity_tensor = torch.tensor(frame_intensity_scores, dtype=torch.float32)
    connectivity_tensor = torch.tensor(frame_connectivity_scores, dtype=torch.float32)
    compactness_tensor = torch.tensor(frame_compactness_scores, dtype=torch.float32)

    return {
        "concentrated_region_score": float(score_tensor.mean().item()),
        "concentrated_region_score_std": float(score_tensor.std(unbiased=False).item()),
        "concentrated_region_intensity": float(intensity_tensor.mean().item()),
        "concentrated_region_connectivity": float(connectivity_tensor.mean().item()),
        "concentrated_region_compactness": float(compactness_tensor.mean().item()),
        "concentrated_region_top_ratio": float(top_ratio),
        "concentrated_region_top_k": int(top_k),
    }

def _compute_wan21_t2v_trajectory_support_quality_stats(
    map_fhw: torch.Tensor,
    support_mask_fhw: torch.Tensor,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Compute trajectory support quality from one [F, H, W] map and one [F, H, W] support mask."""
    if map_fhw.dim() != 3:
        raise ValueError(f"Expected map_fhw [F, H, W], got shape={tuple(map_fhw.shape)}")
    if support_mask_fhw.dim() != 3:
        raise ValueError(
            f"Expected support_mask_fhw [F, H, W], got shape={tuple(support_mask_fhw.shape)}"
        )
    if tuple(map_fhw.shape) != tuple(support_mask_fhw.shape):
        raise ValueError(
            "map_fhw and support_mask_fhw must have the same shape, "
            f"got {tuple(map_fhw.shape)} vs {tuple(support_mask_fhw.shape)}"
        )

    attention_map = map_fhw.detach().float().clamp_min(0.0)
    support_mask = support_mask_fhw.detach().float().clamp(0.0, 1.0)

    frame_count, token_grid_height, token_grid_width = attention_map.shape

    # Frame-wise normalization over H*W.
    frame_flat = attention_map.reshape(frame_count, token_grid_height * token_grid_width)
    frame_denominator = frame_flat.sum(dim=1, keepdim=True).clamp_min(float(eps))
    frame_probabilities = frame_flat / frame_denominator
    support_flat = support_mask.reshape(frame_count, token_grid_height * token_grid_width)
    frame_support_quality = (frame_probabilities * support_flat).sum(dim=1)

    # Video-wise normalization over F*H*W.
    video_flat_attention = attention_map.reshape(-1)
    video_probabilities = video_flat_attention / video_flat_attention.sum().clamp_min(float(eps))
    video_flat_support = support_mask.reshape(-1)
    video_support_quality = float((video_probabilities * video_flat_support).sum().item())

    support_quality_frame = float(frame_support_quality.mean().item())
    support_quality_frame_std = float(frame_support_quality.std(unbiased=False).item())
    support_quality_frame_min = float(frame_support_quality.min().item())
    support_quality_frame_max = float(frame_support_quality.max().item())
    support_quality_video = float(video_support_quality)

    return {
        "support_quality_frame": support_quality_frame,
        "support_quality_frame_std": support_quality_frame_std,
        "support_quality_frame_min": support_quality_frame_min,
        "support_quality_frame_max": support_quality_frame_max,
        "support_quality_video": support_quality_video,
        "support_mask_coverage_ratio": float(support_mask.mean().item()),
    }

def _compute_wan21_t2v_head_evolution_metrics(
    map_fhw: torch.Tensor,
    support_mask_fhw: torch.Tensor,
    apply_preprocess_on_metrics: bool = True,
    preprocess_winsorize_quantile: float = 0.995,
    preprocess_despike_quantile: float = 0.98,
    preprocess_min_component_area: int = 2,
    concentrated_region_top_ratio: float = 0.05,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Compute all head_evolution metrics for one [F, H, W] map."""
    if bool(apply_preprocess_on_metrics):
        map_for_metrics_fhw, preprocess_stats = _preprocess_wan21_t2v_attention_map_fhw(
            map_fhw=map_fhw,
            winsorize_quantile=float(preprocess_winsorize_quantile),
            despike_quantile=float(preprocess_despike_quantile),
            min_component_area=int(preprocess_min_component_area),
        )
        preprocess_stats["apply_preprocess_on_metrics"] = 1
    else:
        map_for_metrics_fhw = map_fhw.detach().float().clamp_min(0.0)
        preprocess_stats = {
            "apply_preprocess_on_metrics": 0,
            "winsorize_quantile": float(preprocess_winsorize_quantile),
            "despike_quantile": float(preprocess_despike_quantile),
            "min_component_area": int(max(1, int(preprocess_min_component_area))),
            "winsor_threshold_mean": 0.0,
            "despiked_points_total": 0,
            "high_value_points_total": 0,
            "despike_removed_ratio": 0.0,
        }

    entropy_stats = _compute_wan21_t2v_spatial_entropy_stats(map_fhw=map_for_metrics_fhw, eps=eps)
    support_quality_stats = _compute_wan21_t2v_trajectory_support_quality_stats(
        map_fhw=map_for_metrics_fhw,
        support_mask_fhw=support_mask_fhw,
        eps=eps,
    )
    if bool(apply_preprocess_on_metrics):
        concentrated_region_stats = _compute_wan21_t2v_concentrated_region_score_stats(
            probability_map_fhw=map_for_metrics_fhw,
            top_ratio=float(concentrated_region_top_ratio),
            eps=eps,
        )
        concentrated_region_stats["concentrated_region_enabled"] = 1
    else:
        # When preprocessing is disabled, skip all outlier-related region logic (top-k + connectivity).
        concentrated_region_stats = {
            "concentrated_region_score": 0.0,
            "concentrated_region_score_std": 0.0,
            "concentrated_region_intensity": 0.0,
            "concentrated_region_connectivity": 0.0,
            "concentrated_region_compactness": 0.0,
            "concentrated_region_top_ratio": float(concentrated_region_top_ratio),
            "concentrated_region_top_k": 0,
            "concentrated_region_enabled": 0,
        }

    entropy_frame = float(entropy_stats["entropy_norm_mean"])
    entropy_video = float(entropy_stats["entropy_video_norm"])
    frame_entropy_complement = 1.0 - entropy_frame
    video_entropy_complement = 1.0 - entropy_video
    frame_support_quality = float(support_quality_stats["support_quality_frame"])
    video_support_quality = float(support_quality_stats["support_quality_video"])

    # Aggregate score is only a convenience readout; all raw metrics are exported independently.
    planning_aggregate_frame = 0.5 * frame_entropy_complement + 0.5 * frame_support_quality
    planning_aggregate_video = 0.5 * video_entropy_complement + 0.5 * video_support_quality

    merged_metrics = {
        "entropy_frame": float(entropy_frame),
        "entropy_video": float(entropy_video),
        "support_quality_frame": float(frame_support_quality),
        "support_quality_video": float(video_support_quality),
    }
    merged_metrics.update(concentrated_region_stats)
    merged_metrics.update(preprocess_stats)
    merged_metrics["apply_preprocess_on_metrics"] = int(bool(apply_preprocess_on_metrics))
    merged_metrics.update(
        {
            "planning_aggregate_frame": float(planning_aggregate_frame),
            "planning_aggregate_video": float(planning_aggregate_video),
            "entropy_complement_frame": float(frame_entropy_complement),
            "entropy_complement_video": float(video_entropy_complement),
        }
    )
    return merged_metrics

def _build_wan21_t2v_head_evolution_row(
    step: int,
    layer: int,
    head,
    token: str,
    source: str,
    metrics: Dict[str, float],
    center_mode: str,
    support_radius_mode: str,
) -> Dict[str, object]:
    """Build one row for head_evolution CSV exports."""
    row = {
        "step": int(step),
        "layer": int(layer),
        "head": str(head),
        "token": str(token),
        "source": str(source),
        "reference_center_mode": str(center_mode),
        "support_radius_mode": str(support_radius_mode),
    }
    row.update(metrics)
    return row

def _set_wan21_t2v_axis_ylim_from_values(
    axis,
    values: Sequence[float],
    clamp_to_unit_interval: bool = True,
):
    """Apply stable y-axis range from value list."""
    if not values:
        return
    y_min = float(min(values))
    y_max = float(max(values))

    if y_max - y_min < 1e-8:
        padding = 0.05
    else:
        padding = max(0.015, 0.08 * (y_max - y_min))
    lower = y_min - padding
    upper = y_max + padding

    if clamp_to_unit_interval:
        lower = max(0.0, lower)
        upper = min(1.02, upper)
        if upper - lower < 0.06:
            center = 0.5 * (upper + lower)
            lower = max(0.0, center - 0.03)
            upper = min(1.02, center + 0.03)

    axis.set_ylim(lower, upper)

def _plot_wan21_t2v_head_evolution_stepwise_metric(
    rows: Sequence[Dict[str, object]],
    save_file: str,
    metric_key: str,
    y_label: str,
    stepwise_layer: int,
):
    """Plot one step-wise metric curve on object-mean rows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    object_rows = [row for row in rows if str(row.get("token", "")) == "__object_mean__"]
    object_rows = [row for row in object_rows if metric_key in row]
    if not object_rows:
        return ""

    object_rows = sorted(object_rows, key=lambda row: int(row["step"]))
    x_steps = [int(row["step"]) for row in object_rows]
    y_values = [float(row[metric_key]) for row in object_rows]

    fig, axis = plt.subplots(1, 1, figsize=(7.8, 4.8))
    axis.plot(x_steps, y_values, marker="o", linewidth=1.8, color="#1f77b4")
    axis.set_title(
        f"Head Evolution Step-Wise: {metric_key} (layer={int(stepwise_layer)}, head-mean map)"
    )
    axis.set_xlabel("diffusion step")
    axis.set_ylabel(y_label)
    _set_wan21_t2v_axis_ylim_from_values(axis, y_values, clamp_to_unit_interval=True)
    axis.grid(alpha=0.22, linestyle="--")
    fig.tight_layout()

    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file

def _plot_wan21_t2v_head_evolution_layerwise_metric_for_step(
    rows: Sequence[Dict[str, object]],
    save_file: str,
    metric_key: str,
    y_label: str,
    step: int,
):
    """Plot one layer-wise metric curve for a fixed step on object-mean rows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    object_rows = [row for row in rows if str(row.get("token", "")) == "__object_mean__"]
    object_rows = [row for row in object_rows if int(row.get("step", -1)) == int(step)]
    object_rows = [row for row in object_rows if metric_key in row]
    if not object_rows:
        return ""

    object_rows = sorted(object_rows, key=lambda row: int(row["layer"]))
    x_layers = [int(row["layer"]) for row in object_rows]
    y_values = [float(row[metric_key]) for row in object_rows]

    fig, axis = plt.subplots(1, 1, figsize=(8.0, 4.8))
    axis.plot(x_layers, y_values, marker="o", linewidth=1.6, color="#0ea5e9")
    axis.set_title(f"Head Evolution Layer-Wise: {metric_key} (step={int(step)}, head-mean map)")
    axis.set_xlabel("layer")
    axis.set_ylabel(y_label)
    _set_wan21_t2v_axis_ylim_from_values(axis, y_values, clamp_to_unit_interval=True)
    axis.grid(alpha=0.22, linestyle="--")
    fig.tight_layout()

    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file

def _plot_wan21_t2v_head_evolution_headwise_metric_for_layer(
    rows: Sequence[Dict[str, object]],
    save_file: str,
    metric_key: str,
    y_label: str,
    layer: int,
):
    """Plot one head-wise metric figure for a fixed layer across diffusion steps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    object_rows = [row for row in rows if str(row.get("token", "")) == "__object_mean__"]
    object_rows = [row for row in object_rows if int(row.get("layer", -1)) == int(layer)]
    object_rows = [row for row in object_rows if metric_key in row]
    if not object_rows:
        return ""

    rows_by_head: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for row in object_rows:
        rows_by_head[int(row["head"])].append(row)

    if not rows_by_head:
        return ""

    fig, axis = plt.subplots(1, 1, figsize=(8.2, 5.0))
    head_indices = sorted(rows_by_head.keys())
    num_heads = len(head_indices)
    if num_heads <= 20:
        # Use a discrete qualitative palette to avoid repeated default cycle colors.
        color_map = plt.get_cmap("tab20", num_heads)
    else:
        # Fall back to a high-cardinality palette when the head count is large.
        color_map = plt.get_cmap("gist_rainbow", num_heads)

    all_values: List[float] = []
    for color_index, head_index in enumerate(head_indices):
        head_rows = sorted(rows_by_head[head_index], key=lambda row: int(row["step"]))
        x_steps = [int(row["step"]) for row in head_rows]
        y_values = [float(row[metric_key]) for row in head_rows]
        all_values.extend(y_values)
        axis.plot(
            x_steps,
            y_values,
            linewidth=1.2,
            alpha=0.9,
            color=color_map(color_index),
            label=f"h{head_index:02d}",
        )

    axis.set_title(f"Head Evolution Head-Wise: {metric_key} (layer={int(layer)})")
    axis.set_xlabel("diffusion step")
    axis.set_ylabel(y_label)
    _set_wan21_t2v_axis_ylim_from_values(axis, all_values, clamp_to_unit_interval=True)
    axis.grid(alpha=0.22, linestyle="--")
    if len(rows_by_head) <= 20:
        axis.legend(fontsize=7, ncol=2)
    fig.tight_layout()

    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file

def _save_wan21_t2v_reference_radius_overlay_pdf(
    reference_map_fhw: torch.Tensor,
    center_trajectory: Sequence[Tuple[float, float]],
    support_radius_per_frame: Sequence[float],
    attention_frame_indices: Sequence[int],
    frame_labels: Optional[Sequence[int]],
    save_file: str,
    title: str,
):
    """Save timeline-style reference maps with center point and radius-circle overlays."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    if reference_map_fhw.dim() != 3:
        raise ValueError(
            f"Expected reference_map_fhw [F, H, W], got shape={tuple(reference_map_fhw.shape)}"
        )
    if not attention_frame_indices:
        return

    frame_count, token_grid_height, token_grid_width = reference_map_fhw.shape
    valid_indices = [
        int(frame_index)
        for frame_index in attention_frame_indices
        if 0 <= int(frame_index) < int(frame_count)
    ]
    if not valid_indices:
        return

    if len(center_trajectory) < frame_count or len(support_radius_per_frame) < frame_count:
        raise ValueError(
            "center_trajectory and support_radius_per_frame must cover all frames in reference_map_fhw."
        )

    figure = plt.figure(figsize=(2.35 * len(valid_indices), 3.1))
    grid_spec = figure.add_gridspec(1, len(valid_indices), wspace=0.02, hspace=0.0)
    axes = [figure.add_subplot(grid_spec[0, panel_index]) for panel_index in range(len(valid_indices))]

    for panel_index, frame_index in enumerate(valid_indices):
        axis = axes[panel_index]
        background = reference_map_fhw[frame_index].detach().float().cpu().numpy()
        axis.imshow(background, cmap="magma", alpha=0.92)

        center_y, center_x = center_trajectory[frame_index]
        circle_radius = float(support_radius_per_frame[frame_index])
        circle_patch = Circle(
            (float(center_x), float(center_y)),
            radius=float(circle_radius),
            fill=False,
            linewidth=1.8,
            linestyle="-",
            edgecolor="#9be15d",
            alpha=0.95,
        )
        axis.add_patch(circle_patch)
        axis.scatter(
            [float(center_x)],
            [float(center_y)],
            s=42,
            c=["#ff4d4f"],
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
        )

        if frame_labels and panel_index < len(frame_labels):
            axis.set_title(f"frame={int(frame_labels[panel_index])}", fontsize=9)
        else:
            axis.set_title(f"frame={int(frame_index)}", fontsize=9)
        axis.set_xlim(-0.5, token_grid_width - 0.5)
        axis.set_ylim(token_grid_height - 0.5, -0.5)
        axis.axis("off")

    figure.suptitle(title, fontsize=10, y=0.975)
    figure.subplots_adjust(left=0.005, right=0.995, top=0.9, bottom=0.01, wspace=0.02, hspace=0.0)
    _ensure_dir(os.path.dirname(save_file))
    figure.savefig(save_file, format="pdf")
    plt.close(figure)

def _safe_wan21_t2v_pearson_correlation(
    values_x: Sequence[float],
    values_y: Sequence[float],
    eps: float = 1e-12,
) -> float:
    """Compute Pearson correlation robustly; return 0 when variance is near zero."""
    if len(values_x) != len(values_y) or len(values_x) < 2:
        return 0.0

    tensor_x = torch.tensor(values_x, dtype=torch.float32)
    tensor_y = torch.tensor(values_y, dtype=torch.float32)
    centered_x = tensor_x - tensor_x.mean()
    centered_y = tensor_y - tensor_y.mean()
    denominator = centered_x.pow(2).sum().sqrt() * centered_y.pow(2).sum().sqrt()
    denominator_value = float(denominator.item())
    if denominator_value <= float(eps):
        return 0.0
    correlation = float((centered_x * centered_y).sum().item() / denominator_value)
    return max(-1.0, min(1.0, correlation))

def _quantile_wan21_t2v(values: Sequence[float], quantile: float, fallback: float = 0.0) -> float:
    """Compute scalar quantile with a fallback for empty inputs."""
    if not values:
        return float(fallback)
    tensor_values = torch.tensor([float(value) for value in values], dtype=torch.float32)
    quantile_value = float(torch.quantile(tensor_values, float(quantile)).item())
    return quantile_value

def _compute_wan21_t2v_head_evolution_head_scores(
    headwise_rows: Sequence[Dict[str, object]],
    early_step_end: int,
    score_quantile: float,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    """Compute per-head planning/readout/modulation scores and head categories."""
    if not headwise_rows:
        return [], {}

    # Layer-step mean support curve serves as the state baseline for SCS and MSS.
    layer_step_to_support_values: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    for row in headwise_rows:
        layer_index = int(row["layer"])
        step_index = int(row["step"])
        support_quality_video = float(row["support_quality_video"])
        layer_step_to_support_values[(layer_index, step_index)].append(support_quality_video)

    layer_step_mean_support: Dict[Tuple[int, int], float] = {}
    for key, values in layer_step_to_support_values.items():
        layer_step_mean_support[key] = float(sum(values) / max(1, len(values)))

    rows_by_head: Dict[Tuple[int, int], List[Dict[str, object]]] = defaultdict(list)
    for row in headwise_rows:
        rows_by_head[(int(row["layer"]), int(row["head"]))].append(row)

    score_rows: List[Dict[str, object]] = []
    for (layer_index, head_index), rows_for_head in sorted(rows_by_head.items()):
        sorted_rows = sorted(rows_for_head, key=lambda row: int(row["step"]))
        step_indices = [int(row["step"]) for row in sorted_rows]
        support_curve = [float(row["support_quality_video"]) for row in sorted_rows]
        entropy_curve = [float(row["entropy_video"]) for row in sorted_rows]

        early_indices = [idx for idx, step in enumerate(step_indices) if step <= int(early_step_end)]
        late_indices = [idx for idx, step in enumerate(step_indices) if step > int(early_step_end)]
        if not early_indices:
            early_indices = [0]
        if not late_indices:
            late_indices = [len(step_indices) - 1]

        early_support_mean = float(sum(support_curve[idx] for idx in early_indices) / len(early_indices))
        late_support_mean = float(sum(support_curve[idx] for idx in late_indices) / len(late_indices))
        early_entropy_complement_mean = float(
            sum((1.0 - entropy_curve[idx]) for idx in early_indices) / len(early_indices)
        )
        late_entropy_complement_mean = float(
            sum((1.0 - entropy_curve[idx]) for idx in late_indices) / len(late_indices)
        )

        early_planning_score = early_support_mean
        readout_score = late_entropy_complement_mean
        late_gain = late_support_mean - early_support_mean
        planning_score = 0.5 * early_planning_score + 0.5 * readout_score

        state_baseline_curve = [
            float(layer_step_mean_support[(layer_index, step_index)])
            for step_index in step_indices
        ]
        state_coupling_score = _safe_wan21_t2v_pearson_correlation(
            values_x=support_curve,
            values_y=state_baseline_curve,
        )

        head_tensor = torch.tensor(support_curve, dtype=torch.float32)
        baseline_tensor = torch.tensor(state_baseline_curve, dtype=torch.float32)
        residual_tensor = head_tensor - baseline_tensor
        head_variance = float(head_tensor.var(unbiased=False).item())
        residual_variance = float(residual_tensor.var(unbiased=False).item())
        modulation_sensitivity_score = float(residual_variance / max(head_variance, 1e-12))

        score_rows.append(
            {
                "layer": int(layer_index),
                "head": int(head_index),
                "num_steps": int(len(step_indices)),
                "early_planning_score": float(early_planning_score),
                "readout_score": float(readout_score),
                "planning_score": float(planning_score),
                "late_gain": float(late_gain),
                "state_coupling_score": float(state_coupling_score),
                "modulation_sensitivity_score": float(modulation_sensitivity_score),
                "early_support_quality_video_mean": float(early_support_mean),
                "late_support_quality_video_mean": float(late_support_mean),
                "early_entropy_complement_video_mean": float(early_entropy_complement_mean),
                "late_entropy_complement_video_mean": float(late_entropy_complement_mean),
            }
        )

    if not score_rows:
        return [], {}

    # Data-driven thresholds (quantiles) for lightweight rule-based categorization.
    early_planning_values = [float(row["early_planning_score"]) for row in score_rows]
    late_gain_values = [float(row["late_gain"]) for row in score_rows]
    state_coupling_values = [float(row["state_coupling_score"]) for row in score_rows]
    modulation_sensitivity_values = [float(row["modulation_sensitivity_score"]) for row in score_rows]

    early_planning_threshold = _quantile_wan21_t2v(early_planning_values, quantile=score_quantile)
    late_gain_high_threshold = _quantile_wan21_t2v(late_gain_values, quantile=score_quantile)
    late_gain_median = _quantile_wan21_t2v(late_gain_values, quantile=0.5)
    state_coupling_threshold = _quantile_wan21_t2v(state_coupling_values, quantile=score_quantile)
    modulation_sensitivity_threshold = _quantile_wan21_t2v(
        modulation_sensitivity_values,
        quantile=score_quantile,
    )

    for row in score_rows:
        early_planning_score = float(row["early_planning_score"])
        late_gain = float(row["late_gain"])
        state_coupling_score = float(row["state_coupling_score"])
        modulation_sensitivity_score = float(row["modulation_sensitivity_score"])

        # Category rules:
        # - planning_head: strong early support, not dominated by late-only gain.
        # - state_driven: mostly follows denoising-state trend (high late gain + high coupling).
        # - modulation_driven: strongly step-selective residual behavior (high MSS).
        # - mixed_or_uncertain: none of the above confidently.
        if (
            early_planning_score >= early_planning_threshold
            and late_gain <= late_gain_median
            and modulation_sensitivity_score <= modulation_sensitivity_threshold
        ):
            head_category = "planning_head"
        elif (
            late_gain >= late_gain_high_threshold
            and state_coupling_score >= state_coupling_threshold
            and modulation_sensitivity_score < modulation_sensitivity_threshold
        ):
            head_category = "state_driven"
        elif modulation_sensitivity_score >= modulation_sensitivity_threshold:
            head_category = "modulation_driven"
        else:
            head_category = "mixed_or_uncertain"

        row["head_category"] = head_category

    threshold_summary = {
        "score_quantile": float(score_quantile),
        "early_planning_threshold": float(early_planning_threshold),
        "late_gain_high_threshold": float(late_gain_high_threshold),
        "late_gain_median": float(late_gain_median),
        "state_coupling_threshold": float(state_coupling_threshold),
        "modulation_sensitivity_threshold": float(modulation_sensitivity_threshold),
    }
    return score_rows, threshold_summary

def run_wan21_t2v_head_evolution(
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
    head_evolution_steps: Sequence[int] = tuple(),
    head_evolution_layerwise_steps: Sequence[int] = tuple(),
    head_evolution_head_layer: int = -2,
    head_evolution_stepwise_layer: int = 27,
    head_evolution_reference_step: int = 50,
    head_evolution_reference_layer: int = 27,
    head_evolution_center_mode: str = "centroid",
    head_evolution_support_radius_mode: str = "adaptive_area",
    head_evolution_support_radius_fixed: float = 2.0,
    head_evolution_support_radius_alpha: float = 1.5,
    head_evolution_support_radius_min: float = 1.0,
    head_evolution_support_radius_max_ratio: float = 0.25,
    head_evolution_traj_power: float = 1.5,
    head_evolution_traj_quantile: float = 0.8,
    head_evolution_reference_viz_num_frames: int = 10,
    head_evolution_save_reference_radius_overlay: bool = True,
    head_evolution_early_step_end: int = 5,
    head_evolution_score_quantile: float = 0.7,
    head_evolution_apply_preprocess_on_metrics: bool = True,
    head_evolution_preprocess_winsorize_quantile: float = 0.995,
    head_evolution_preprocess_despike_quantile: float = 0.98,
    head_evolution_preprocess_min_component_area: int = 2,
    head_evolution_concentrated_region_top_ratio: float = 0.05,
    entropy_eps: float = 1e-12,
    reuse_cross_attention_dir: Optional[str] = None,
    parallel_cfg: Optional[Wan21T2VParallelConfig] = None,
):
    """Run offline head-evolution analysis from saved cross-attention maps."""
    del wan21_root, ckpt_dir, task, frame_num, size, shift, sample_solver, sampling_steps, guide_scale
    del seed, device_id, offload_model, parallel_cfg, prompt

    if dist.is_initialized() and dist.get_rank() != 0:
        dist.barrier()
        return None

    object_words = [str(word).strip() for word in target_object_words if str(word).strip()]
    if not object_words:
        raise ValueError("head_evolution requires non-empty target_object_words.")
    object_words = list(dict.fromkeys(object_words))

    verb_words = [str(word).strip() for word in target_verb_words if str(word).strip()]
    verb_words = list(dict.fromkeys(verb_words))

    if not reuse_cross_attention_dir or (not str(reuse_cross_attention_dir).strip()):
        raise ValueError(
            "head_evolution is an offline analysis and requires --reuse_cross_attention_dir "
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

    object_words_in_maps = [word for word in object_words if word in set(words_in_maps)]
    if not object_words_in_maps:
        object_words_in_maps = [word for word in words_in_maps if word_to_type.get(word) == "object"]
    if not object_words_in_maps:
        raise ValueError(
            "None of target_object_words found in reused cross-attention maps and no fallback object token type found. "
            f"target_object_words={object_words}, words_in_maps={words_in_maps[:50]}"
        )

    if head_evolution_steps:
        resolved_steps = _dedup_wan21_t2v_int_list(head_evolution_steps)
    else:
        resolved_steps = list(available_steps)
    missing_steps = [step for step in resolved_steps if step not in set(available_steps)]
    if missing_steps:
        raise ValueError(
            "Some head_evolution_steps are not present in reused maps: "
            f"{missing_steps}; available={available_steps}"
        )

    if head_evolution_layerwise_steps:
        resolved_layerwise_steps = _dedup_wan21_t2v_int_list(head_evolution_layerwise_steps)
    else:
        resolved_layerwise_steps = list(resolved_steps)
    missing_layerwise_steps = [step for step in resolved_layerwise_steps if step not in set(available_steps)]
    if missing_layerwise_steps:
        raise ValueError(
            "Some head_evolution_layerwise_steps are not present in reused maps: "
            f"{missing_layerwise_steps}; available={available_steps}"
        )

    num_layers = max(available_layers) + 1
    last_layer_index = num_layers - 1

    stepwise_layer_raw = int(head_evolution_stepwise_layer)
    if stepwise_layer_raw == -1:
        stepwise_layer_index = last_layer_index
    elif stepwise_layer_raw >= 0:
        stepwise_layer_index = int(stepwise_layer_raw)
    else:
        raise ValueError(
            "head_evolution_stepwise_layer must be -1 or a non-negative layer index, "
            f"got {stepwise_layer_raw}."
        )
    if stepwise_layer_index >= num_layers:
        raise ValueError(
            f"head_evolution_stepwise_layer={stepwise_layer_raw} resolved to invalid layer "
            f"{stepwise_layer_index} (num_layers={num_layers})."
        )

    head_layer_raw = int(head_evolution_head_layer)
    if head_layer_raw == -2:
        head_layer_indices = list(range(num_layers))
    elif head_layer_raw == -1:
        head_layer_indices = [last_layer_index]
    elif head_layer_raw >= 0:
        head_layer_indices = [head_layer_raw]
    else:
        raise ValueError(
            "head_evolution_head_layer must be -2 (all), -1 (last), or non-negative layer index, "
            f"got {head_layer_raw}."
        )
    if any(layer_index < 0 or layer_index >= num_layers for layer_index in head_layer_indices):
        raise ValueError(
            f"head_evolution_head_layer={head_layer_raw} resolved to invalid layers "
            f"{head_layer_indices} (num_layers={num_layers})."
        )

    reference_step = int(head_evolution_reference_step)
    reference_layer = int(head_evolution_reference_layer)
    if reference_step not in set(available_steps):
        raise ValueError(
            f"head_evolution_reference_step={reference_step} not found in maps. available_steps={available_steps}"
        )
    if reference_layer not in set(available_layers):
        raise ValueError(
            f"head_evolution_reference_layer={reference_layer} not found in maps. available_layers={available_layers}"
        )

    reference_head_maps = _mean_wan21_t2v_head_maps_for_words(
        mean_maps=mean_maps,
        step=reference_step,
        layer=reference_layer,
        words=object_words_in_maps,
    )
    if reference_head_maps is None:
        raise ValueError(
            "Cannot construct reference object map for support mask. "
            f"step={reference_step}, layer={reference_layer}, object_words={object_words_in_maps}"
        )

    reference_head_mean_map = reference_head_maps.mean(dim=0)
    reference_preprocessed_head_mean_map, reference_preprocess_stats = _preprocess_wan21_t2v_attention_map_fhw(
        map_fhw=reference_head_mean_map,
        winsorize_quantile=float(head_evolution_preprocess_winsorize_quantile),
        despike_quantile=float(head_evolution_preprocess_despike_quantile),
        min_component_area=int(head_evolution_preprocess_min_component_area),
    )
    reference_trajectory_data = _extract_wan21_t2v_reference_peak_and_centroid_trajectory(
        map_fhw=reference_preprocessed_head_mean_map,
        power=head_evolution_traj_power,
        quantile=head_evolution_traj_quantile,
    )

    reference_center_mode = str(head_evolution_center_mode).strip().lower()
    if reference_center_mode not in {"peak", "centroid", "geometric_center"}:
        raise ValueError(
            "head_evolution_center_mode must be one of "
            "{'peak', 'centroid', 'geometric_center'}, "
            f"got: {head_evolution_center_mode}"
        )
    if reference_center_mode == "peak":
        reference_center_trajectory = reference_trajectory_data["peak_centers"]
    elif reference_center_mode == "centroid":
        reference_center_trajectory = reference_trajectory_data["centroid_centers"]
    else:
        reference_center_trajectory = reference_trajectory_data["geometric_centers"]

    token_grid_height = int(reference_head_mean_map.size(1))
    token_grid_width = int(reference_head_mean_map.size(2))
    support_mask_fhw, support_radius_per_frame = _build_wan21_t2v_trajectory_support_mask_from_centers(
        center_trajectory=reference_center_trajectory,
        component_areas=reference_trajectory_data["component_areas"],
        token_grid_height=token_grid_height,
        token_grid_width=token_grid_width,
        support_radius_mode=head_evolution_support_radius_mode,
        support_radius_fixed=head_evolution_support_radius_fixed,
        support_radius_alpha=head_evolution_support_radius_alpha,
        support_radius_min=head_evolution_support_radius_min,
        support_radius_max_ratio=head_evolution_support_radius_max_ratio,
    )

    # Optional overlay figure to verify whether radius r is visually reasonable before scoring.
    reference_radius_overlay_path = ""
    if bool(head_evolution_save_reference_radius_overlay):
        reference_attention_indices, reference_frame_labels = _resolve_wan21_t2v_viz_frame_indices(
            attention_frame_count=int(reference_head_mean_map.size(0)),
            video_frame_count=int(reference_head_mean_map.size(0)),
            num_frames=int(head_evolution_reference_viz_num_frames),
            explicit_indices=None,
        )
        reference_radius_overlay_path = os.path.join(output_dir, "head_evolution_reference_radius_overlay.pdf")
        _save_wan21_t2v_reference_radius_overlay_pdf(
            reference_map_fhw=reference_head_mean_map,
            center_trajectory=reference_center_trajectory,
            support_radius_per_frame=support_radius_per_frame,
            attention_frame_indices=reference_attention_indices,
            frame_labels=reference_frame_labels,
            save_file=reference_radius_overlay_path,
            title=(
                f"Reference Radius Overlay (step={reference_step}, layer={reference_layer}, "
                f"center_mode={reference_center_mode})"
            ),
        )

    reference_rows: List[Dict[str, object]] = []
    for frame_index, (
        (peak_y, peak_x),
        (centroid_y, centroid_x),
        (geometric_center_y, geometric_center_x),
        component_area,
        support_radius,
    ) in enumerate(
        zip(
            reference_trajectory_data["peak_centers"],
            reference_trajectory_data["centroid_centers"],
            reference_trajectory_data["geometric_centers"],
            reference_trajectory_data["component_areas"],
            support_radius_per_frame,
        )
    ):
        reference_rows.append(
            {
                "frame": int(frame_index),
                "peak_y": float(peak_y),
                "peak_x": float(peak_x),
                "centroid_y": float(centroid_y),
                "centroid_x": float(centroid_x),
                "geometric_center_y": float(geometric_center_y),
                "geometric_center_x": float(geometric_center_x),
                "component_area": int(component_area),
                "support_radius": float(support_radius),
            }
        )

    stepwise_rows: List[Dict[str, object]] = []
    layerwise_rows: List[Dict[str, object]] = []
    headwise_rows: List[Dict[str, object]] = []

    object_words_in_maps_set = set(object_words_in_maps)
    step_layer_head_count: Dict[Tuple[int, int], int] = {}
    for step_index, layer_index, token_name in mean_maps.keys():
        if token_name not in object_words_in_maps_set:
            continue
        tensor = mean_maps.get((int(step_index), int(layer_index), str(token_name)))
        if tensor is None:
            continue
        key = (int(step_index), int(layer_index))
        head_count = int(tensor.size(0))
        previous_count = step_layer_head_count.get(key, 0)
        if head_count > previous_count:
            step_layer_head_count[key] = head_count

    stepwise_total_tasks = sum(
        1 for step_index in resolved_steps if step_layer_head_count.get((int(step_index), int(stepwise_layer_index)), 0) > 0
    )
    headwise_total_tasks = sum(
        step_layer_head_count.get((int(step_index), int(layer_index)), 0)
        for step_index in resolved_steps
        for layer_index in head_layer_indices
    )
    layerwise_total_tasks = sum(
        1
        for step_index in resolved_layerwise_steps
        for layer_index in available_layers
        if step_layer_head_count.get((int(step_index), int(layer_index)), 0) > 0
    )
    total_metric_tasks = int(stepwise_total_tasks + headwise_total_tasks + layerwise_total_tasks)

    metric_progress_bar = None
    if total_metric_tasks > 0:
        try:
            from tqdm import tqdm
            metric_progress_bar = tqdm(
                total=total_metric_tasks,
                desc="head_evolution metrics",
                unit="map",
                leave=True,
            )
        except Exception:
            metric_progress_bar = None

    try:
        for step_index in resolved_steps:
            object_head_maps = _mean_wan21_t2v_head_maps_for_words(
                mean_maps=mean_maps,
                step=step_index,
                layer=stepwise_layer_index,
                words=object_words_in_maps,
            )
            if object_head_maps is not None:
                object_head_mean_map = object_head_maps.mean(dim=0)
                stepwise_metrics = _compute_wan21_t2v_head_evolution_metrics(
                    map_fhw=object_head_mean_map,
                    support_mask_fhw=support_mask_fhw,
                    apply_preprocess_on_metrics=bool(head_evolution_apply_preprocess_on_metrics),
                    preprocess_winsorize_quantile=float(head_evolution_preprocess_winsorize_quantile),
                    preprocess_despike_quantile=float(head_evolution_preprocess_despike_quantile),
                    preprocess_min_component_area=int(head_evolution_preprocess_min_component_area),
                    concentrated_region_top_ratio=float(head_evolution_concentrated_region_top_ratio),
                    eps=entropy_eps,
                )
                stepwise_rows.append(
                    _build_wan21_t2v_head_evolution_row(
                        step=step_index,
                        layer=stepwise_layer_index,
                        head="mean",
                        token="__object_mean__",
                        source="stepwise_selected_layer_head_mean_map",
                        metrics=stepwise_metrics,
                        center_mode=reference_center_mode,
                        support_radius_mode=head_evolution_support_radius_mode,
                    )
                )
                if metric_progress_bar is not None:
                    metric_progress_bar.update(1)

            for layer_index in head_layer_indices:
                head_maps_for_layer = _mean_wan21_t2v_head_maps_for_words(
                    mean_maps=mean_maps,
                    step=step_index,
                    layer=layer_index,
                    words=object_words_in_maps,
                )
                if head_maps_for_layer is None:
                    continue
                for head_index in range(int(head_maps_for_layer.size(0))):
                    map_fhw = head_maps_for_layer[head_index]
                    head_metrics = _compute_wan21_t2v_head_evolution_metrics(
                        map_fhw=map_fhw,
                        support_mask_fhw=support_mask_fhw,
                        apply_preprocess_on_metrics=bool(head_evolution_apply_preprocess_on_metrics),
                        preprocess_winsorize_quantile=float(head_evolution_preprocess_winsorize_quantile),
                        preprocess_despike_quantile=float(head_evolution_preprocess_despike_quantile),
                        preprocess_min_component_area=int(head_evolution_preprocess_min_component_area),
                        concentrated_region_top_ratio=float(head_evolution_concentrated_region_top_ratio),
                        eps=entropy_eps,
                    )
                    headwise_rows.append(
                        _build_wan21_t2v_head_evolution_row(
                            step=step_index,
                            layer=layer_index,
                            head=head_index,
                            token="__object_mean__",
                            source="headwise_selected_layers",
                            metrics=head_metrics,
                            center_mode=reference_center_mode,
                            support_radius_mode=head_evolution_support_radius_mode,
                        )
                    )
                    if metric_progress_bar is not None:
                        metric_progress_bar.update(1)

        for step_index in resolved_layerwise_steps:
            for layer_index in available_layers:
                object_head_maps = _mean_wan21_t2v_head_maps_for_words(
                    mean_maps=mean_maps,
                    step=step_index,
                    layer=layer_index,
                    words=object_words_in_maps,
                )
                if object_head_maps is None:
                    continue
                object_head_mean_map = object_head_maps.mean(dim=0)
                layerwise_metrics = _compute_wan21_t2v_head_evolution_metrics(
                    map_fhw=object_head_mean_map,
                    support_mask_fhw=support_mask_fhw,
                    apply_preprocess_on_metrics=bool(head_evolution_apply_preprocess_on_metrics),
                    preprocess_winsorize_quantile=float(head_evolution_preprocess_winsorize_quantile),
                    preprocess_despike_quantile=float(head_evolution_preprocess_despike_quantile),
                    preprocess_min_component_area=int(head_evolution_preprocess_min_component_area),
                    concentrated_region_top_ratio=float(head_evolution_concentrated_region_top_ratio),
                    eps=entropy_eps,
                )
                layerwise_rows.append(
                    _build_wan21_t2v_head_evolution_row(
                        step=step_index,
                        layer=layer_index,
                        head="mean",
                        token="__object_mean__",
                        source="layerwise_head_mean_map",
                        metrics=layerwise_metrics,
                        center_mode=reference_center_mode,
                        support_radius_mode=head_evolution_support_radius_mode,
                    )
                )
                if metric_progress_bar is not None:
                    metric_progress_bar.update(1)
    finally:
        if metric_progress_bar is not None:
            metric_progress_bar.close()

    stepwise_rows = sorted(stepwise_rows, key=lambda row: int(row["step"]))
    layerwise_rows = sorted(layerwise_rows, key=lambda row: (int(row["step"]), int(row["layer"])))
    headwise_rows = sorted(
        headwise_rows,
        key=lambda row: (int(row["layer"]), int(row["head"]), int(row["step"])),
    )

    score_rows, score_threshold_summary = _compute_wan21_t2v_head_evolution_head_scores(
        headwise_rows=headwise_rows,
        early_step_end=int(head_evolution_early_step_end),
        score_quantile=float(head_evolution_score_quantile),
    )
    score_rows = sorted(score_rows, key=lambda row: (int(row["layer"]), int(row["head"])))

    _save_csv(os.path.join(output_dir, "head_evolution_reference_trajectory.csv"), reference_rows)
    _save_csv(os.path.join(output_dir, "head_evolution_stepwise.csv"), stepwise_rows)
    _save_csv(os.path.join(output_dir, "head_evolution_layerwise.csv"), layerwise_rows)
    _save_csv(os.path.join(output_dir, "head_evolution_headwise.csv"), headwise_rows)
    _save_csv(os.path.join(output_dir, "head_evolution_head_scores.csv"), score_rows)

    torch.save(reference_head_mean_map, os.path.join(output_dir, "head_evolution_reference_head_mean_map.pt"))
    torch.save(
        reference_preprocessed_head_mean_map,
        os.path.join(output_dir, "head_evolution_reference_head_mean_map_preprocessed.pt"),
    )
    torch.save(support_mask_fhw, os.path.join(output_dir, "head_evolution_support_mask_fhw.pt"))

    metric_specs = [
        ("entropy_frame", "frame-level normalized entropy"),
        ("entropy_video", "video-level normalized entropy"),
        ("support_quality_frame", "frame-level trajectory support quality"),
        ("support_quality_video", "video-level trajectory support quality"),
        ("planning_aggregate_video", "planning aggregate score"),
    ]
    if bool(head_evolution_apply_preprocess_on_metrics):
        metric_specs.append(("concentrated_region_score", "concentrated-region score"))

    plot_paths_stepwise: List[str] = []
    plot_paths_layerwise: List[str] = []
    plot_paths_headwise: List[str] = []

    stepwise_plot_dir = os.path.join(output_dir, "head_evolution_stepwise_plots")
    layerwise_plot_dir = os.path.join(output_dir, "head_evolution_layerwise_plots")
    headwise_plot_dir = os.path.join(output_dir, "head_evolution_headwise_plots")

    for metric_key, y_label in metric_specs:
        stepwise_plot_path = _plot_wan21_t2v_head_evolution_stepwise_metric(
            rows=stepwise_rows,
            save_file=os.path.join(stepwise_plot_dir, f"stepwise_{metric_key}.pdf"),
            metric_key=metric_key,
            y_label=y_label,
            stepwise_layer=stepwise_layer_index,
        )
        if stepwise_plot_path:
            plot_paths_stepwise.append(stepwise_plot_path)

        for step_index in resolved_layerwise_steps:
            layerwise_plot_path = _plot_wan21_t2v_head_evolution_layerwise_metric_for_step(
                rows=layerwise_rows,
                save_file=os.path.join(
                    layerwise_plot_dir,
                    metric_key,
                    f"layerwise_step_{int(step_index):03d}_{metric_key}.pdf",
                ),
                metric_key=metric_key,
                y_label=y_label,
                step=int(step_index),
            )
            if layerwise_plot_path:
                plot_paths_layerwise.append(layerwise_plot_path)

        plotted_head_layers = sorted({int(row["layer"]) for row in headwise_rows})
        for layer_index in plotted_head_layers:
            headwise_plot_path = _plot_wan21_t2v_head_evolution_headwise_metric_for_layer(
                rows=headwise_rows,
                save_file=os.path.join(
                    headwise_plot_dir,
                    metric_key,
                    f"headwise_layer_{int(layer_index):02d}_{metric_key}.pdf",
                ),
                metric_key=metric_key,
                y_label=y_label,
                layer=int(layer_index),
            )
            if headwise_plot_path:
                plot_paths_headwise.append(headwise_plot_path)

    category_counts = defaultdict(int)
    for row in score_rows:
        category_counts[str(row.get("head_category", "mixed_or_uncertain"))] += 1

    summary = {
        "experiment": "wan21_t2v_head_evolution",
        "target_object_words": list(object_words),
        "target_verb_words": list(verb_words),
        "prompt_tokens": prompt_tokens,
        "token_positions": word_to_positions,
        "token_types": word_to_type,
        "object_words_in_maps": object_words_in_maps,
        "reuse_cross_attention_dir": cross_attention_dir,
        "loaded_maps_source": loaded_maps_source,
        "available_steps": [int(step) for step in available_steps],
        "available_layers": [int(layer) for layer in available_layers],
        "head_evolution_steps": [int(step) for step in resolved_steps],
        "head_evolution_layerwise_steps": [int(step) for step in resolved_layerwise_steps],
        "head_evolution_stepwise_layer": int(stepwise_layer_raw),
        "head_evolution_stepwise_layer_resolved": int(stepwise_layer_index),
        "head_evolution_head_layer": int(head_layer_raw),
        "head_evolution_head_layers_resolved": [int(layer) for layer in head_layer_indices],
        "head_evolution_reference_step": int(reference_step),
        "head_evolution_reference_layer": int(reference_layer),
        "head_evolution_center_mode": reference_center_mode,
        "head_evolution_support_radius_mode": str(head_evolution_support_radius_mode),
        "head_evolution_support_radius_fixed": float(head_evolution_support_radius_fixed),
        "head_evolution_support_radius_alpha": float(head_evolution_support_radius_alpha),
        "head_evolution_support_radius_min": float(head_evolution_support_radius_min),
        "head_evolution_support_radius_max_ratio": float(head_evolution_support_radius_max_ratio),
        "head_evolution_traj_power": float(head_evolution_traj_power),
        "head_evolution_traj_quantile": float(head_evolution_traj_quantile),
        "head_evolution_reference_viz_num_frames": int(head_evolution_reference_viz_num_frames),
        "head_evolution_save_reference_radius_overlay": bool(head_evolution_save_reference_radius_overlay),
        "reference_radius_overlay_path": reference_radius_overlay_path,
        "head_evolution_early_step_end": int(head_evolution_early_step_end),
        "head_evolution_score_quantile": float(head_evolution_score_quantile),
        "head_evolution_apply_preprocess_on_metrics": bool(head_evolution_apply_preprocess_on_metrics),
        "head_evolution_preprocess_winsorize_quantile": float(head_evolution_preprocess_winsorize_quantile),
        "head_evolution_preprocess_despike_quantile": float(head_evolution_preprocess_despike_quantile),
        "head_evolution_preprocess_min_component_area": int(head_evolution_preprocess_min_component_area),
        "head_evolution_concentrated_region_top_ratio": float(head_evolution_concentrated_region_top_ratio),
        "reference_preprocess_stats": reference_preprocess_stats,
        "head_score_thresholds": score_threshold_summary,
        "head_category_counts": {key: int(value) for key, value in sorted(category_counts.items())},
        "num_layers": int(num_layers),
        "stepwise_rows": int(len(stepwise_rows)),
        "layerwise_rows": int(len(layerwise_rows)),
        "headwise_rows": int(len(headwise_rows)),
        "head_score_rows": int(len(score_rows)),
        "plot_paths_stepwise": plot_paths_stepwise,
        "plot_paths_layerwise": plot_paths_layerwise,
        "plot_paths_headwise": plot_paths_headwise,
        "reference_csv": os.path.join(output_dir, "head_evolution_reference_trajectory.csv"),
        "stepwise_csv": os.path.join(output_dir, "head_evolution_stepwise.csv"),
        "layerwise_csv": os.path.join(output_dir, "head_evolution_layerwise.csv"),
        "headwise_csv": os.path.join(output_dir, "head_evolution_headwise.csv"),
        "head_scores_csv": os.path.join(output_dir, "head_evolution_head_scores.csv"),
        "reference_head_mean_map_path": os.path.join(output_dir, "head_evolution_reference_head_mean_map.pt"),
        "reference_head_mean_map_preprocessed_path": os.path.join(
            output_dir,
            "head_evolution_reference_head_mean_map_preprocessed.pt",
        ),
        "support_mask_path": os.path.join(output_dir, "head_evolution_support_mask_fhw.pt"),
        "entropy_eps": float(entropy_eps),
    }

    _save_json(os.path.join(output_dir, "head_evolution_summary.json"), summary)
    if dist.is_initialized():
        dist.barrier()
    return summary
