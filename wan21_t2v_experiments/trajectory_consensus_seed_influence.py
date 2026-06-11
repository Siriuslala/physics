"""Wan2.1-T2V follow-up: seed influence on trajectory consensus.

This module implements the follow-up analysis documented after
`self_attention_coupling` in `trajectory_consensus_dynamics.md`.

Two independent modes are supported:
1. `seed_sensitivity`: run a seed ensemble and measure how strongly the early
   winner-minus-loser preference fluctuates across seeds.
2. `anchor_frame`: reuse saved `candidate_consensus` and
   `self_attention_coupling` caches to identify frame-level proposal anchors and
   routing anchors.

The module is intentionally conservative:
- it reuses the same candidate partition and candidate-level metrics already
  used by `trajectory_consensus_dynamics`;
- it does not introduce a new external trajectory decoder;
- for the first implementation, the seed-sensitivity metric is based on
  winner-minus-loser gaps of global mutual consistency.
"""

import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist

from .utils import (
    Wan21T2VCrossAttentionVizState,
    Wan21T2VParallelConfig,
    _broadcast_seed_if_needed,
    _build_wan21_t2v_pipeline,
    _dedup_wan21_t2v_int_list,
    _ensure_dir,
    _generate_wan21_t2v_video,
    _init_wan21_t2v_runtime,
    _install_wan21_t2v_cross_attention_viz_patch,
    _iter_wan21_t2v_parallel_results,
    _load_wan21_t2v_cross_attention_mean_maps_from_disk,
    _load_wan21_t2v_cross_attention_token_meta,
    _load_wan21_t2v_csv_rows,
    _locate_wan21_t2v_prompt_words,
    _mean_wan21_t2v_head_maps_for_words,
    _mean_wan21_t2v_headmean_map_for_words,
    _normalize_wan21_t2v_attention_map_per_frame,
    _resolve_wan21_t2v_offload_model,
    _resolve_wan21_t2v_num_workers,
    _run_wan21_t2v_once_with_patch,
    _save_csv,
    _save_json,
    _unwrap_wan21_t2v_dit_model_for_runtime_patch,
)
from .wan21_t2v_experiment_patch import (
    Wan21T2VAttentionProbeConfig,
    Wan21T2VCausalAttentionConfig,
    Wan21T2VPatchBundleConfig,
    Wan21T2VRopePatchConfig,
)


def _seed_influence_import_tc_helpers():
    """Import trajectory-consensus internals lazily to avoid circular imports."""
    from .trajectory_consensus_dynamics import (
        _build_wan21_t2v_anchor_union_payload,
        _build_wan21_t2v_self_attention_pairwise_layer_value_vectors,
        _build_wan21_t2v_reference_object_boxes,
        _compute_wan21_t2v_candidate_weights_for_head_map,
        _extract_wan21_t2v_candidate_regions_for_map,
        _load_wan21_t2v_trajectory_consensus_candidate_cache,
        _maybe_skip_wan21_t2v_existing_plot,
        _merge_wan21_t2v_candidate_regions_by_reference_box,
        _plot_wan21_t2v_trajectory_consensus_heatmap,
        _preprocess_wan21_t2v_attention_map_fhw,
        _resolve_wan21_t2v_selected_head_specs_from_layer_counts,
        _resolve_wan21_t2v_steps_and_layers_from_maps,
        _safe_wan21_t2v_float,
        _sample_wan21_t2v_evenly_spaced_indices,
        _smooth_wan21_t2v_map_fhw,
        _summarize_wan21_t2v_self_attention_candidate_features,
        _summarize_wan21_t2v_self_attention_temporal_precedence,
        _trajectory_consensus_compute_candidate_feature_task,
        _trajectory_consensus_self_attention_feature_display_name,
    )
    return {
        "anchor_union_payload": _build_wan21_t2v_anchor_union_payload,
        "pairwise_value_vectors": _build_wan21_t2v_self_attention_pairwise_layer_value_vectors,
        "reference_object_boxes": _build_wan21_t2v_reference_object_boxes,
        "candidate_weights_for_head_map": _compute_wan21_t2v_candidate_weights_for_head_map,
        "candidate_regions_for_map": _extract_wan21_t2v_candidate_regions_for_map,
        "load_candidate_cache": _load_wan21_t2v_trajectory_consensus_candidate_cache,
        "maybe_skip_existing_plot": _maybe_skip_wan21_t2v_existing_plot,
        "merge_candidate_regions_by_reference_box": _merge_wan21_t2v_candidate_regions_by_reference_box,
        "plot_heatmap": _plot_wan21_t2v_trajectory_consensus_heatmap,
        "preprocess_attention_map": _preprocess_wan21_t2v_attention_map_fhw,
        "resolve_selected_head_specs": _resolve_wan21_t2v_selected_head_specs_from_layer_counts,
        "resolve_steps_and_layers": _resolve_wan21_t2v_steps_and_layers_from_maps,
        "safe_float": _safe_wan21_t2v_float,
        "sample_evenly_spaced_indices": _sample_wan21_t2v_evenly_spaced_indices,
        "smooth_map": _smooth_wan21_t2v_map_fhw,
        "summarize_candidate_features": _summarize_wan21_t2v_self_attention_candidate_features,
        "summarize_temporal_precedence": _summarize_wan21_t2v_self_attention_temporal_precedence,
        "candidate_feature_task": _trajectory_consensus_compute_candidate_feature_task,
        "feature_display_name": _trajectory_consensus_self_attention_feature_display_name,
    }


def _seed_influence_seed_dir(output_dir: str, seed: int) -> str:
    return os.path.join(output_dir, "trajectory_consensus_seed_influence", f"seed_{int(seed):06d}")


def _seed_influence_sensitivity_root(output_dir: str) -> str:
    return os.path.join(output_dir, "trajectory_consensus_seed_influence")


def _seed_influence_anchor_root(output_dir: str) -> str:
    return os.path.join(output_dir, "trajectory_consensus_anchor_frames")


def _seed_influence_progress(
    iterable: Iterable,
    desc: str,
    total: Optional[int] = None,
    enabled: bool = True,
):
    """Wrap an iterable with a tqdm progress bar when available."""
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=str(desc), total=total, leave=False)
    except Exception:
        return iterable


def _seed_influence_metric_display_name(metric_name: str) -> str:
    tc = _seed_influence_import_tc_helpers()
    if str(metric_name) == "candidate_entropy":
        return "candidate entropy"
    if str(metric_name) == "winner_gap":
        return "winner gap"
    if str(metric_name) == "spatial_map_entropy":
        return "spatial map entropy"
    return str(tc["feature_display_name"](metric_name))


def _seed_influence_finite_mean(values: Sequence[float]) -> float:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return float("nan")
    return float(sum(finite_values) / len(finite_values))


def _seed_influence_finite_sum(values: Sequence[float]) -> float:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return float("nan")
    return float(sum(finite_values))


def _seed_influence_top1_top2_gap(values: Sequence[float]) -> float:
    finite_values = sorted(
        [float(value) for value in values if math.isfinite(float(value))],
        reverse=True,
    )
    if len(finite_values) < 2:
        return float("nan")
    return float(finite_values[0] - finite_values[1])


def _seed_influence_zr_metric_name_to_field(metric_name: str) -> str:
    metric_key = str(metric_name).strip().lower()
    mapping = {
        "global_mutual_consistency": "global_mutual_consistency",
        "local_mutual_consistency": "local_mutual_consistency",
        "global_incoming_support": "global_incoming_support",
        "local_incoming_support": "local_incoming_support",
        "global_incoming_preference_share": "global_incoming_preference_share",
        "local_incoming_preference_share": "local_incoming_preference_share",
        "global_incoming_vote_share": "global_incoming_vote_share",
        "local_incoming_vote_share": "local_incoming_vote_share",
        "proposal_pi": "proposal_pi",
        "proposal_vote_share": "proposal_vote_share",
    }
    if metric_key not in mapping:
        raise ValueError(f"Unsupported seed-sensitivity z_r metric: {metric_name}")
    return mapping[metric_key]


def _seed_influence_compute_spatial_entropy(map_fhw: torch.Tensor) -> List[float]:
    """Compute per-frame spatial entropy of a normalized head-mean attention map."""
    probability_map = _normalize_wan21_t2v_attention_map_per_frame(map_fhw)
    frame_entropies: List[float] = []
    for frame_index in range(int(probability_map.shape[0])):
        frame_prob = probability_map[int(frame_index)].reshape(-1).float()
        frame_prob = frame_prob / frame_prob.sum().clamp_min(1e-12)
        entropy = float(-(frame_prob * frame_prob.clamp_min(1e-12).log()).sum().item())
        frame_entropies.append(float(entropy))
    return frame_entropies


def _seed_influence_save_candidate_cache_pt(
    candidate_region_cache: Dict[Tuple[int, int], Dict[str, object]],
    save_path: str,
) -> None:
    payload = {}
    for (step, layer), candidate_payload in candidate_region_cache.items():
        payload[(int(step), int(layer))] = {
            "label_map_fhw_np": candidate_payload["label_map_fhw"].detach().cpu().numpy().astype(
                np.int16,
                copy=False,
            ),
            "frame_metadata": candidate_payload.get("frame_metadata", []),
        }
    torch.save(payload, save_path)


def _seed_influence_build_candidate_consensus_from_mean_maps(
    mean_maps: Dict[Tuple[int, int, str], torch.Tensor],
    object_words_in_maps: Sequence[str],
    selected_steps: Sequence[int],
    selected_layers: Sequence[int],
    reference_step: int,
    reference_layer: int,
    candidate_base_quantile: float,
    candidate_split_quantiles: Sequence[float],
    candidate_smooth_radius: int,
    candidate_stable_peak_min_levels: int,
    candidate_peak_merge_distance: float,
    candidate_preprocess_winsorize_quantile: float,
    candidate_preprocess_despike_quantile: float,
    candidate_min_component_area: int,
) -> Tuple[
    Dict[Tuple[int, int], Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
]:
    """Reproduce the cache-only part of `candidate_consensus` without plotting."""
    tc = _seed_influence_import_tc_helpers()
    candidate_region_cache: Dict[Tuple[int, int], Dict[str, object]] = {}
    candidate_region_rows: List[Dict[str, object]] = []
    candidate_weight_rows: List[Dict[str, object]] = []
    winner_gap_rows: List[Dict[str, object]] = []

    reference_object_boxes = None
    reference_head_maps = _mean_wan21_t2v_head_maps_for_words(
        mean_maps=mean_maps,
        step=int(reference_step),
        layer=int(reference_layer),
        words=object_words_in_maps,
    )
    if reference_head_maps is not None:
        reference_head_mean_map = reference_head_maps.mean(dim=0)
        reference_preprocessed_head_mean_map, _ = tc["preprocess_attention_map"](
            map_fhw=reference_head_mean_map,
            winsorize_quantile=float(candidate_preprocess_winsorize_quantile),
            despike_quantile=float(candidate_preprocess_despike_quantile),
            min_component_area=int(candidate_min_component_area),
        )
        reference_object_boxes = tc["reference_object_boxes"](
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
            if int(candidate_smooth_radius) > 0:
                worker_map_fhw = tc["smooth_map"](
                    headmean_map,
                    smooth_radius=int(candidate_smooth_radius),
                )
                worker_smooth_radius = 0
            else:
                worker_map_fhw = headmean_map
                worker_smooth_radius = 0
            candidate_data = tc["candidate_regions_for_map"](
                map_fhw=worker_map_fhw,
                base_quantile=float(candidate_base_quantile),
                split_quantiles=tuple(float(x) for x in candidate_split_quantiles),
                min_component_area=int(candidate_min_component_area),
                smooth_radius=int(worker_smooth_radius),
                stable_peak_min_levels=int(candidate_stable_peak_min_levels),
                peak_merge_distance=float(candidate_peak_merge_distance),
            )
            if reference_object_boxes is not None:
                merged_label_map_fhw, merged_frame_metadata = tc["merge_candidate_regions_by_reference_box"](
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
                        "peak_y": float(candidate_row.get("peak_y", float("nan"))),
                        "peak_x": float(candidate_row.get("peak_x", float("nan"))),
                        "centroid_y": float(candidate_row.get("centroid_y", float("nan"))),
                        "centroid_x": float(candidate_row.get("centroid_x", float("nan"))),
                        "seed_y": float(candidate_row.get("seed_y", float("nan"))),
                        "seed_x": float(candidate_row.get("seed_x", float("nan"))),
                        "seed_score": float(candidate_row.get("seed_score", float("nan"))),
                        "support_count": int(candidate_row.get("support_count", 0)),
                        "support_level": float(candidate_row.get("support_level", float("nan"))),
                        "candidate_count_in_frame": int(len(frame_candidates)),
                    })

            head_maps = _mean_wan21_t2v_head_maps_for_words(
                mean_maps=mean_maps,
                step=int(step),
                layer=int(layer),
                words=object_words_in_maps,
            )
            if head_maps is None:
                continue
            per_head_weights: Dict[int, List[List[float]]] = {}
            for head_index in range(int(head_maps.shape[0])):
                probability_map = _normalize_wan21_t2v_attention_map_per_frame(head_maps[int(head_index)])
                per_head_weights[int(head_index)] = tc["candidate_weights_for_head_map"](
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
                candidate_count = int(label_map_fhw[int(frame_index)].max().item())
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
    return candidate_region_cache, candidate_region_rows, candidate_weight_rows, winner_gap_rows


def _seed_influence_collect_cross_attention_mean_maps_for_seed(
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
    target_object_words: Sequence[str],
    target_verb_words: Sequence[str],
    collect_steps: Sequence[int],
    layers_to_collect: Sequence[int],
) -> Tuple[Dict[Tuple[int, int, str], torch.Tensor], Dict[str, List[int]], List[str]]:
    """Collect cross-attention mean maps for one seed with one loaded pipeline."""
    word_to_positions, _word_to_type, _prompt_tokens = _locate_wan21_t2v_prompt_words(
        text_encoder=pipeline.text_encoder,
        prompt=prompt,
        target_object_words=target_object_words,
        target_verb_words=target_verb_words,
    )
    object_words_in_prompt = [
        str(word) for word in target_object_words
        if str(word) in word_to_positions
    ]
    if not object_words_in_prompt:
        raise ValueError("No target_object_words found in prompt tokenization for seed influence.")

    state = Wan21T2VCrossAttentionVizState(
        token_positions=word_to_positions,
        collect_steps=tuple(int(x) for x in collect_steps),
        num_layers=len(pipeline.model.blocks),
        num_heads=pipeline.model.num_heads,
        chunk_size=1024,
        layers_to_collect=tuple(int(x) for x in layers_to_collect) if layers_to_collect else None,
    )
    handle = _install_wan21_t2v_cross_attention_viz_patch(pipeline.model, state)
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
            seed=int(seed),
            offload_model=offload_model,
        )
    finally:
        handle.restore()
    if dist.is_initialized():
        dist.barrier()
    return state.export_mean_maps(), word_to_positions, object_words_in_prompt


def _seed_influence_collect_self_attention_features_for_seed(
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
    selected_steps: Sequence[int],
    selected_layers: Sequence[int],
    candidate_region_cache: Dict[Tuple[int, int], Dict[str, object]],
    candidate_weight_rows: Sequence[Dict[str, object]],
    trajectory_consensus_branch: str,
    trajectory_consensus_self_heads: Optional[Sequence[str]],
    trajectory_consensus_sa_anchor_step: int,
    trajectory_consensus_sa_anchor_layer: int,
    trajectory_consensus_sa_covered_mass_min: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    """Collect self-attention coupling rows and candidate features for one seed."""
    tc = _seed_influence_import_tc_helpers()
    coupling_branch = str(trajectory_consensus_branch).strip().lower()
    if coupling_branch not in {"cond", "uncond"}:
        raise ValueError("trajectory_consensus_seed_influence currently supports branch `cond` or `uncond` only.")

    target_model = _unwrap_wan21_t2v_dit_model_for_runtime_patch(pipeline.model)
    num_self_heads_per_layer = {
        int(layer): int(target_model.blocks[int(layer)].self_attn.num_heads)
        for layer in selected_layers
    }
    selected_self_head_specs = tc["resolve_selected_head_specs"](
        explicit_head_specs=trajectory_consensus_self_heads,
        num_heads_per_layer=num_self_heads_per_layer,
    )
    if not selected_self_head_specs:
        raise ValueError(
            "seed influence self-attention coupling requires at least one self-attention head. "
            "Use an empty string for all heads in selected layers."
        )
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
        seed=int(seed),
        offload_model=offload_model,
    )
    if dist.is_initialized():
        dist.barrier()
    pairwise_rows = state.export_candidate_coupling_rows()
    anchor_payload = tc["anchor_union_payload"](
        candidate_region_cache=candidate_region_cache,
        anchor_step=int(trajectory_consensus_sa_anchor_step),
        anchor_layer=int(trajectory_consensus_sa_anchor_layer),
    )
    pairwise_rows_by_step_layer: Dict[Tuple[int, int], List[Dict[str, object]]] = defaultdict(list)
    for row in pairwise_rows:
        pairwise_rows_by_step_layer[(int(row["step"]), int(row["layer"]))].append(row)
    proposal_rows_by_step_layer: Dict[Tuple[int, int], List[Dict[str, object]]] = defaultdict(list)
    for row in candidate_weight_rows:
        proposal_rows_by_step_layer[(int(row["step"]), int(row["layer"]))].append(row)

    candidate_feature_rows: List[Dict[str, object]] = []
    for step_layer_key, pairwise_group_rows in sorted(pairwise_rows_by_step_layer.items()):
        label_payload = candidate_region_cache.get(step_layer_key)
        if label_payload is None:
            continue
        _, _, feature_rows = tc["candidate_feature_task"](
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
        candidate_feature_rows.extend(feature_rows)
    candidate_feature_rows = sorted(
        candidate_feature_rows,
        key=lambda row: (
            int(row["step"]),
            int(row["layer"]),
            int(row["frame"]),
            int(row["candidate_index"]),
        ),
    )
    feature_summary_rows = tc["summarize_candidate_features"](candidate_feature_rows)
    temporal_precedence_rows = tc["summarize_temporal_precedence"](
        candidate_feature_rows,
        persistence_window=2,
    )
    return pairwise_rows, candidate_feature_rows, feature_summary_rows, temporal_precedence_rows


def _seed_influence_compute_z_rows(
    candidate_feature_rows: Sequence[Dict[str, object]],
    metric_field: str,
    seed: int,
) -> List[Dict[str, object]]:
    safe_float = _seed_influence_import_tc_helpers()["safe_float"]
    rows_by_step_layer_frame: Dict[Tuple[int, int, int], List[Dict[str, object]]] = defaultdict(list)
    for row in candidate_feature_rows:
        rows_by_step_layer_frame[(int(row["step"]), int(row["layer"]), int(row["frame"]))].append(row)
    out_rows: List[Dict[str, object]] = []
    for (step, layer, frame_index), group_rows in sorted(rows_by_step_layer_frame.items()):
        winner_rows = [row for row in group_rows if int(row.get("is_winner_aligned", 0)) == 1]
        loser_rows = [row for row in group_rows if int(row.get("is_strongest_loser", 0)) == 1]
        if not winner_rows or not loser_rows:
            continue
        winner_value = safe_float(winner_rows[0].get(metric_field, float("nan")))
        loser_value = safe_float(loser_rows[0].get(metric_field, float("nan")))
        z_value = float(winner_value - loser_value) if math.isfinite(winner_value) and math.isfinite(loser_value) else float("nan")
        out_rows.append({
            "seed": int(seed),
            "step": int(step),
            "layer": int(layer),
            "frame": int(frame_index),
            "metric": str(metric_field),
            "winner_value": float(winner_value),
            "loser_value": float(loser_value),
            "z_value": float(z_value),
            "winner_candidate": int(winner_rows[0].get("candidate_index", -1)),
            "loser_candidate": int(loser_rows[0].get("candidate_index", -1)),
        })
    return out_rows


def _seed_influence_aggregate_seed_sensitivity(
    z_rows: Sequence[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    grouped_values: Dict[Tuple[int, int, int], List[Tuple[int, float]]] = defaultdict(list)
    for row in z_rows:
        z_value = float(row["z_value"])
        if not math.isfinite(z_value):
            continue
        grouped_values[(int(row["step"]), int(row["layer"]), int(row["frame"]))].append(
            (int(row["seed"]), float(z_value))
        )

    framewise_rows: List[Dict[str, object]] = []
    standardized_by_observation_seed: Dict[Tuple[int, int, int], Dict[int, float]] = defaultdict(dict)
    z_by_observation_seed: Dict[Tuple[int, int, int], Dict[int, float]] = defaultdict(dict)
    for (step, layer, frame_index), values in sorted(grouped_values.items()):
        z_values = np.asarray([value for _, value in values], dtype=np.float64)
        seeds = [int(seed) for seed, _ in values]
        if z_values.size <= 0:
            continue
        mean_value = float(z_values.mean())
        sigma_value = float(z_values.std(ddof=1)) if z_values.size >= 2 else 0.0
        eta_value = float(mean_value / max(1e-8, sigma_value))
        flip_probability = float((z_values <= 0.0).mean())
        for seed, z_value in values:
            standardized_by_observation_seed[(int(step), int(layer), int(frame_index))][int(seed)] = float(
                (float(z_value) - mean_value) / max(1e-8, sigma_value)
            ) if sigma_value > 1e-8 else 0.0
            z_by_observation_seed[(int(step), int(layer), int(frame_index))][int(seed)] = float(z_value)
        framewise_rows.append({
            "step": int(step),
            "layer": int(layer),
            "frame": int(frame_index),
            "num_seeds": int(z_values.size),
            "z_mean": float(mean_value),
            "z_std": float(sigma_value),
            "eta": float(eta_value),
            "flip_probability": float(flip_probability),
        })

    seed_to_standardized_values: Dict[int, List[float]] = defaultdict(list)
    seed_to_flip_values: Dict[int, List[float]] = defaultdict(list)
    for observation_key, standardized_by_seed in standardized_by_observation_seed.items():
        del observation_key
        for seed, standardized_value in standardized_by_seed.items():
            seed_to_standardized_values[int(seed)].append(float(standardized_value))
    for observation_key, z_by_seed in z_by_observation_seed.items():
        del observation_key
        for seed, z_value in z_by_seed.items():
            seed_to_flip_values[int(seed)].append(1.0 if float(z_value) <= 0.0 else 0.0)

    seedwise_rows: List[Dict[str, object]] = []
    for seed in sorted(set(list(seed_to_standardized_values.keys()) + list(seed_to_flip_values.keys()))):
        standardized_values = seed_to_standardized_values.get(int(seed), [])
        flip_values = seed_to_flip_values.get(int(seed), [])
        b_value = float(sum(standardized_values) / len(standardized_values)) if standardized_values else float("nan")
        f_value = float(sum(flip_values) / len(flip_values)) if flip_values else float("nan")
        seedwise_rows.append({
            "seed": int(seed),
            "B_r": float(b_value),
            "F_r": float(f_value),
            "num_observations": int(len(flip_values)),
        })

    summary_rows: List[Dict[str, object]] = []
    grouped_step_layer: Dict[Tuple[int, int], List[Dict[str, object]]] = defaultdict(list)
    for row in framewise_rows:
        grouped_step_layer[(int(row["step"]), int(row["layer"]))].append(row)
    for (step, layer), group_rows in sorted(grouped_step_layer.items()):
        eta_values = [float(row["eta"]) for row in group_rows if math.isfinite(float(row["eta"]))]
        flip_values = [float(row["flip_probability"]) for row in group_rows if math.isfinite(float(row["flip_probability"]))]
        summary_rows.append({
            "step": int(step),
            "layer": int(layer),
            "mean_eta": float(sum(eta_values) / len(eta_values)) if eta_values else float("nan"),
            "mean_flip_probability": float(sum(flip_values) / len(flip_values)) if flip_values else float("nan"),
            "num_frames": int(len(group_rows)),
        })
    return framewise_rows, seedwise_rows, summary_rows


def _seed_influence_plot_seed_heatmap_rows(
    rows: Sequence[Dict[str, object]],
    value_key: str,
    save_file: str,
    title: str,
    skip_existing_plots: bool = False,
) -> str:
    tc = _seed_influence_import_tc_helpers()
    if tc["maybe_skip_existing_plot"](save_file, bool(skip_existing_plots)):
        return save_file
    return tc["plot_heatmap"](
        matrix_rows=rows,
        save_file=save_file,
        title=title,
        row_key="step",
        col_key="layer",
        value_key=value_key,
        row_label="step",
        col_label="layer",
    )


def _seed_influence_plot_step_frame_heatmap(
    framewise_rows: Sequence[Dict[str, object]],
    selected_step: int,
    value_key: str,
    save_file: str,
    title: str,
    skip_existing_plots: bool = False,
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tc = _seed_influence_import_tc_helpers()
    if tc["maybe_skip_existing_plot"](save_file, bool(skip_existing_plots)):
        return save_file
    step_rows = [row for row in framewise_rows if int(row["step"]) == int(selected_step)]
    if not step_rows:
        return ""
    layers = sorted({int(row["layer"]) for row in step_rows})
    frames = sorted({int(row["frame"]) for row in step_rows})
    layer_to_index = {value: idx for idx, value in enumerate(layers)}
    frame_to_index = {value: idx for idx, value in enumerate(frames)}
    matrix = np.full((len(layers), len(frames)), np.nan, dtype=np.float32)
    for row in step_rows:
        value = float(row.get(value_key, float("nan")))
        matrix[layer_to_index[int(row["layer"])], frame_to_index[int(row["frame"])]] = value
    fig, axis = plt.subplots(1, 1, figsize=(max(7.0, 0.25 * len(frames)), max(5.0, 0.3 * len(layers))))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis")
    axis.set_title(title)
    axis.set_xlabel("frame")
    axis.set_ylabel("layer")
    axis.set_xticks(list(range(len(frames))))
    axis.set_xticklabels([str(frame) for frame in frames], rotation=45, ha="right", fontsize=7)
    axis.set_yticks(list(range(len(layers))))
    axis.set_yticklabels([str(layer) for layer in layers], fontsize=7)
    fig.colorbar(image, ax=axis, shrink=0.82)
    fig.tight_layout()
    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file


def _seed_influence_plot_standardized_deviation_canvas(
    z_rows: Sequence[Dict[str, object]],
    framewise_rows: Sequence[Dict[str, object]],
    save_file: str,
    title: str,
    skip_existing_plots: bool = False,
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tc = _seed_influence_import_tc_helpers()
    if tc["maybe_skip_existing_plot"](save_file, bool(skip_existing_plots)):
        return save_file
    if not z_rows or not framewise_rows:
        return ""
    step = int(framewise_rows[0]["step"])
    layer = int(framewise_rows[0]["layer"])
    selected_frames = sorted({int(row["frame"]) for row in framewise_rows})
    per_frame_stats = {
        int(row["frame"]): (
            float(row["z_mean"]),
            float(row["z_std"]),
        )
        for row in framewise_rows
    }
    selected_z_rows = [
        row for row in z_rows
        if int(row["step"]) == int(step) and int(row["layer"]) == int(layer) and int(row["frame"]) in per_frame_stats
    ]
    if not selected_z_rows:
        return ""
    seeds = sorted({int(row["seed"]) for row in selected_z_rows})
    seed_to_index = {seed: idx for idx, seed in enumerate(seeds)}
    frame_to_index = {frame: idx for idx, frame in enumerate(selected_frames)}
    matrix = np.full((len(seeds), len(selected_frames)), np.nan, dtype=np.float32)
    for row in selected_z_rows:
        frame_index = int(row["frame"])
        mean_value, std_value = per_frame_stats.get(frame_index, (float("nan"), float("nan")))
        if not math.isfinite(mean_value) or not math.isfinite(std_value):
            standardized = float("nan")
        elif std_value <= 1e-8:
            standardized = 0.0
        else:
            standardized = float((float(row["z_value"]) - mean_value) / std_value)
        matrix[seed_to_index[int(row["seed"])], frame_to_index[frame_index]] = standardized
    fig, axis = plt.subplots(1, 1, figsize=(max(7.0, 0.5 * len(selected_frames)), max(5.2, 0.18 * len(seeds))))
    image = axis.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    axis.set_title(title)
    axis.set_xlabel("frame")
    axis.set_ylabel("seed")
    axis.set_xticks(list(range(len(selected_frames))))
    axis.set_xticklabels([str(frame) for frame in selected_frames], rotation=45, ha="right", fontsize=8)
    y_tick_indices = list(range(0, len(seeds), max(1, len(seeds) // 12)))
    axis.set_yticks(y_tick_indices)
    axis.set_yticklabels([str(seeds[idx]) for idx in y_tick_indices], fontsize=8)
    fig.colorbar(image, ax=axis, shrink=0.82)
    fig.tight_layout()
    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file


def _seed_influence_plot_anchor_overlay_scatter_points(
    feature_values: np.ndarray,
    target_values: np.ndarray,
    role_codes: np.ndarray,
    anchor_flags: np.ndarray,
    feature_name: str,
    target_label: str,
    save_file: str,
    title: str,
    skip_existing_plots: bool = False,
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tc = _seed_influence_import_tc_helpers()
    if tc["maybe_skip_existing_plot"](save_file, bool(skip_existing_plots)):
        return save_file
    feature_values = np.asarray(feature_values, dtype=np.float64)
    target_values = np.asarray(target_values, dtype=np.float64)
    role_codes = np.asarray(role_codes, dtype=np.int8)
    anchor_flags = np.asarray(anchor_flags, dtype=np.bool_)
    if (
        feature_values.size == 0
        or target_values.size == 0
        or role_codes.size == 0
        or anchor_flags.size == 0
    ):
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
        axis.scatter(
            feature_values[role_mask],
            target_values[role_mask],
            s=18 if role_code == 0 else 24,
            alpha=0.72,
            color=role_color,
            edgecolors="none",
            label=role_name,
        )
    if np.any(anchor_flags):
        axis.scatter(
            feature_values[anchor_flags],
            target_values[anchor_flags],
            s=92,
            facecolors="none",
            edgecolors="#7dd3fc",
            linewidths=1.9,
            label="anchor frame",
            zorder=4,
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
            alpha=0.88,
            label="linear fit",
        )
    axis.set_title(title)
    axis.set_xlabel(_seed_influence_metric_display_name(feature_name))
    axis.set_ylabel(str(target_label))
    axis.grid(alpha=0.22, linestyle="--")
    axis.legend(fontsize=8)
    fig.tight_layout()
    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file


def _seed_influence_build_anchor_frame_highlight_payload(
    rows: Sequence[Dict[str, object]],
    metric_name: str,
    topk: int,
    higher_is_better: bool,
) -> Dict[Tuple[int, int, int], bool]:
    tc = _seed_influence_import_tc_helpers()
    safe_float = tc["safe_float"]
    if not rows:
        return {}
    grouped_rows: Dict[Tuple[int, int], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped_rows[(int(row["step"]), int(row["layer"]))].append(row)
    highlight_map: Dict[Tuple[int, int, int], bool] = {}
    for (step, layer), group_rows in sorted(grouped_rows.items()):
        del layer
        finite_group_rows = [
            row for row in group_rows
            if math.isfinite(safe_float(row.get(metric_name, float("nan"))))
        ]
        ranked_rows = sorted(
            finite_group_rows,
            key=lambda row: safe_float(row.get(metric_name, float("nan"))),
            reverse=bool(higher_is_better),
        )
        anchor_frame_set = {
            int(row["frame"])
            for row in ranked_rows[: max(1, int(topk))]
        }
        for row in finite_group_rows:
            highlight_map[(int(step), int(row["layer"]), int(row["frame"]))] = bool(
                int(row["frame"]) in anchor_frame_set
            )
    return highlight_map


def _seed_influence_plot_anchor_metric_heatmaps_by_layer(
    rows: Sequence[Dict[str, object]],
    metric_name: str,
    save_root_dir: str,
    title_prefix: str,
    skip_existing_plots: bool = False,
) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tc = _seed_influence_import_tc_helpers()
    safe_float = tc["safe_float"]
    if not rows:
        return []
    layer_values = sorted({int(row["layer"]) for row in rows})
    plot_paths: List[str] = []
    for layer in _seed_influence_progress(layer_values, desc=f"anchor heatmap:{metric_name}", total=len(layer_values)):
        save_file = os.path.join(str(save_root_dir), f"layer_{int(layer):02d}.pdf")
        if tc["maybe_skip_existing_plot"](save_file, bool(skip_existing_plots)):
            plot_paths.append(save_file)
            continue
        layer_rows = [row for row in rows if int(row["layer"]) == int(layer)]
        if not layer_rows:
            continue
        steps = sorted({int(row["step"]) for row in layer_rows})
        frames = sorted({int(row["frame"]) for row in layer_rows})
        step_to_index = {value: idx for idx, value in enumerate(steps)}
        frame_to_index = {value: idx for idx, value in enumerate(frames)}
        matrix = np.full((len(steps), len(frames)), np.nan, dtype=np.float32)
        for row in layer_rows:
            matrix[step_to_index[int(row["step"])], frame_to_index[int(row["frame"])]] = safe_float(
                row.get(metric_name, float("nan"))
            )
        fig, axis = plt.subplots(
            1,
            1,
            figsize=(max(8.2, 0.26 * len(frames)), max(4.8, 0.24 * len(steps))),
        )
        image = axis.imshow(matrix, aspect="auto", cmap="viridis")
        axis.set_title(f"{title_prefix} | layer={int(layer)}")
        axis.set_xlabel("frame")
        axis.set_ylabel("step")
        axis.set_xticks(list(range(len(frames))))
        axis.set_xticklabels([str(frame) for frame in frames], rotation=45, ha="right", fontsize=7)
        axis.set_yticks(list(range(len(steps))))
        axis.set_yticklabels([str(step) for step in steps], fontsize=7)
        axis.grid(False)
        fig.colorbar(image, ax=axis, shrink=0.82)
        fig.tight_layout()
        _ensure_dir(os.path.dirname(save_file))
        fig.savefig(save_file, format="pdf")
        plt.close(fig)
        plot_paths.append(save_file)
    return plot_paths


def _seed_influence_build_candidate_scatter_payload_with_anchor_overlay(
    candidate_feature_rows: Sequence[Dict[str, object]],
    feature_name: str,
    target_key: str,
    anchor_frame_map: Dict[Tuple[int, int, int], bool],
    min_candidate_count: int = 1,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    tc = _seed_influence_import_tc_helpers()
    safe_float = tc["safe_float"]
    feature_values: List[float] = []
    target_values: List[float] = []
    role_codes: List[int] = []
    anchor_flags: List[bool] = []
    step_values: List[int] = []
    for row in candidate_feature_rows:
        if int(row.get("candidate_count", 0)) < max(1, int(min_candidate_count)):
            continue
        feature_value = safe_float(row.get(feature_name, float("nan")))
        target_value = safe_float(row.get(target_key, float("nan")))
        if not math.isfinite(feature_value) or not math.isfinite(target_value):
            continue
        if int(row.get("is_winner_aligned", 0)) == 1:
            role_code = 2
        elif int(row.get("is_strongest_loser", 0)) == 1:
            role_code = 1
        else:
            role_code = 0
        step = int(row["step"])
        layer = int(row["layer"])
        frame = int(row["frame"])
        feature_values.append(float(feature_value))
        target_values.append(float(target_value))
        role_codes.append(int(role_code))
        anchor_flags.append(
            bool(anchor_frame_map.get((int(step), int(layer), int(frame)), False))
            and int(role_code) == 2
        )
        step_values.append(int(step))
    if not feature_values:
        return None
    return (
        np.asarray(feature_values, dtype=np.float32),
        np.asarray(target_values, dtype=np.float32),
        np.asarray(role_codes, dtype=np.int8),
        np.asarray(anchor_flags, dtype=np.bool_),
        np.asarray(step_values, dtype=np.int16),
    )


def _seed_influence_constrain_routing_arrow_layout(
    sampled_frames: Sequence[int],
    selected_edges: Sequence[Dict[str, object]],
    frame_panel_width_in: float,
    frame_panel_height_in: float,
    frame_gap_in: float,
    side_margin_in: float,
) -> Dict[str, object]:
    """Plan routing-arrow geometry under explicit formatting constraints.

    The planner enforces the routing-arrow conventions requested by the user:
    - frames stay on a single horizontal row with fixed panel geometry;
    - each arrow starts and ends at the exact midpoint of the chosen frame edge;
    - rightward arrows always use the top edge, leftward arrows always use the
      bottom edge;
    - arc height depends only on frame distance |a-b|, so equal-span arrows
      share the same curvature magnitude;
    - larger frame distance always yields a higher top arc or a lower bottom
      arc, which keeps long-range arrows outside short-range arrows;
    - labels are anchored directly at the arc apex.
    """
    sampled_frames = [int(frame) for frame in sampled_frames]
    frame_to_rank = {int(frame): int(rank) for rank, frame in enumerate(sampled_frames)}
    frame_bounds_in: Dict[int, Tuple[float, float, float, float]] = {}
    for rank, frame in enumerate(sampled_frames):
        left_in = float(side_margin_in + float(rank) * (frame_panel_width_in + frame_gap_in))
        frame_bounds_in[int(frame)] = (left_in, 0.0, float(frame_panel_width_in), float(frame_panel_height_in))
    total_width_in = (
        2.0 * float(side_margin_in)
        + float(len(sampled_frames)) * float(frame_panel_width_in)
        + float(max(0, len(sampled_frames) - 1)) * float(frame_gap_in)
    )
    edge_specs: List[Dict[str, object]] = []
    ordered_edges = sorted(
        selected_edges,
        key=lambda row: (
            -abs(frame_to_rank[int(row["target_frame"])] - frame_to_rank[int(row["source_frame"])]),
            -float(row["routing_strength"]),
            int(row["source_frame"]),
            int(row["target_frame"]),
        ),
    )
    for row in ordered_edges:
        source_frame = int(row["source_frame"])
        target_frame = int(row["target_frame"])
        source_rank = int(frame_to_rank[source_frame])
        target_rank = int(frame_to_rank[target_frame])
        frame_distance = abs(int(target_rank) - int(source_rank))
        chosen_side = "top" if int(target_rank) > int(source_rank) else "bottom"
        edge_specs.append({
            "source_frame": int(source_frame),
            "target_frame": int(target_frame),
            "source_rank": int(source_rank),
            "target_rank": int(target_rank),
            "side": str(chosen_side),
            "frame_distance": int(frame_distance),
            "routing_strength": float(row["routing_strength"]),
        })

    arrow_layouts: List[Dict[str, object]] = []
    top_extent_in = 0.0
    bottom_extent_in = 0.0
    for spec in edge_specs:
        source_frame = int(spec["source_frame"])
        target_frame = int(spec["target_frame"])
        source_left, source_bottom, source_width, source_height = frame_bounds_in[int(source_frame)]
        target_left, target_bottom, target_width, target_height = frame_bounds_in[int(target_frame)]
        x0_in = float(source_left + 0.5 * source_width)
        x1_in = float(target_left + 0.5 * target_width)
        frame_distance = int(spec["frame_distance"])
        amp_in = float(0.13 + 0.135 * float(frame_distance))
        side = str(spec["side"])
        if side == "top":
            y0_in = float(source_bottom + source_height)
            y1_in = float(target_bottom + target_height)
            control_x_in = float(0.5 * (x0_in + x1_in))
            control_y_in = float(y0_in + amp_in)
            label_x_in = float(0.5 * (x0_in + x1_in))
            label_y_in = float(control_y_in)
            top_extent_in = max(top_extent_in, float((control_y_in + 0.18) - source_height))
        else:
            y0_in = float(source_bottom)
            y1_in = float(target_bottom)
            control_x_in = float(0.5 * (x0_in + x1_in))
            control_y_in = float(y0_in - amp_in)
            label_x_in = float(0.5 * (x0_in + x1_in))
            label_y_in = float(control_y_in)
            bottom_extent_in = max(bottom_extent_in, float((-control_y_in) + 0.18))
        arrow_layouts.append({
            "source_frame": int(source_frame),
            "target_frame": int(target_frame),
            "side": str(side),
            "frame_distance": int(frame_distance),
            "routing_strength": float(spec["routing_strength"]),
            "x0_in": float(x0_in),
            "y0_in": float(y0_in),
            "x1_in": float(x1_in),
            "y1_in": float(y1_in),
            "control_x_in": float(control_x_in),
            "control_y_in": float(control_y_in),
            "label_x_in": float(label_x_in),
            "label_y_in": float(label_y_in),
        })
    top_extent_in = max(top_extent_in, 0.18)
    bottom_extent_in = max(bottom_extent_in, 0.18)
    return {
        "frame_bounds_in": frame_bounds_in,
        "arrow_layouts": arrow_layouts,
        "frame_panel_height_in": float(frame_panel_height_in),
        "top_extent_in": float(top_extent_in),
        "bottom_extent_in": float(bottom_extent_in),
        "total_width_in": float(total_width_in),
    }


def _seed_influence_plot_anchor_arrow_panel(
    routing_rows: Sequence[Dict[str, object]],
    save_file: str,
    title: str,
    topk: int,
    token_grid_shape: Optional[Tuple[int, int]] = None,
    highlighted_frames: Optional[Sequence[int]] = None,
    skip_existing_plots: bool = False,
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.path import Path

    tc = _seed_influence_import_tc_helpers()
    if tc["maybe_skip_existing_plot"](save_file, bool(skip_existing_plots)):
        return save_file
    if not routing_rows:
        return ""
    frames = sorted({int(row["source_frame"]) for row in routing_rows} | {int(row["target_frame"]) for row in routing_rows})
    sampled_frames = sorted({int(row["source_frame"]) for row in routing_rows if int(row.get("is_sampled_frame", 0)) == 1})
    if not sampled_frames:
        sampled_frames = frames
    routing_by_source: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for row in routing_rows:
        routing_by_source[int(row["source_frame"])].append(row)
    selected_edges: List[Dict[str, object]] = []
    for source_frame in sampled_frames:
        source_rows = sorted(
            routing_by_source.get(int(source_frame), []),
            key=lambda row: float(row["routing_strength"]),
            reverse=True,
        )
        selected_edges.extend(source_rows[: max(1, int(topk))])
    if not selected_edges:
        return ""

    token_grid_height = int(token_grid_shape[0]) if token_grid_shape is not None else 14
    token_grid_width = int(token_grid_shape[1]) if token_grid_shape is not None else 22
    aspect_ratio = float(token_grid_width) / max(1.0, float(token_grid_height))
    num_frames = max(1, int(len(sampled_frames)))
    frame_panel_width_in = 1.72
    frame_gap_in = 0.05
    side_margin_in = 0.34
    frame_panel_height_in = float(frame_panel_width_in / max(1e-6, aspect_ratio))

    constrained_layout = _seed_influence_constrain_routing_arrow_layout(
        sampled_frames=tuple(int(frame) for frame in sampled_frames),
        selected_edges=selected_edges,
        frame_panel_width_in=float(frame_panel_width_in),
        frame_panel_height_in=float(frame_panel_height_in),
        frame_gap_in=float(frame_gap_in),
        side_margin_in=float(side_margin_in),
    )

    strengths = [float(row["routing_strength"]) for row in selected_edges]
    strength_min = float(min(strengths))
    strength_max = float(max(strengths))
    highlighted_frame_set = {int(frame) for frame in highlighted_frames} if highlighted_frames is not None else set()

    def _normalize_strength(value: float) -> float:
        if strength_max - strength_min <= 1e-8:
            return 0.6
        return 0.25 + 0.95 * ((float(value) - strength_min) / (strength_max - strength_min))

    horizontal_padding_in = 0.06
    vertical_padding_in = 0.28
    fig_width_in = float(constrained_layout["total_width_in"] + 2.0 * horizontal_padding_in)
    fig_height_in = float(
        constrained_layout["top_extent_in"]
        + constrained_layout["frame_panel_height_in"]
        + constrained_layout["bottom_extent_in"]
        + 2.0 * vertical_padding_in
    )
    fig = plt.figure(figsize=(fig_width_in, fig_height_in))
    overlay_axis = fig.add_axes([0.0, 0.0, 1.0, 1.0], zorder=5)
    overlay_axis.set_xlim(0.0, 1.0)
    overlay_axis.set_ylim(0.0, 1.0)
    overlay_axis.axis("off")

    y_shift_in = float(vertical_padding_in + constrained_layout["bottom_extent_in"])
    x_shift_in = float(horizontal_padding_in)
    blank_frame = np.ones((token_grid_height, token_grid_width), dtype=np.float32)
    frame_bounds_norm: Dict[int, Tuple[float, float, float, float]] = {}
    for frame in sampled_frames:
        left_in, bottom_in, width_in, height_in = constrained_layout["frame_bounds_in"][int(frame)]
        panel_left = float((left_in + x_shift_in) / fig_width_in)
        panel_bottom = float((bottom_in + y_shift_in) / fig_height_in)
        panel_width = float(width_in / fig_width_in)
        panel_height = float(height_in / fig_height_in)
        frame_axis = fig.add_axes([panel_left, panel_bottom, panel_width, panel_height], zorder=2)
        frame_axis.imshow(blank_frame, cmap="gray", vmin=0.0, vmax=1.0, alpha=0.12)
        if int(frame) in highlighted_frame_set:
            frame_axis.set_facecolor("#fef3c7")
            frame_axis.imshow(
                np.ones_like(blank_frame, dtype=np.float32),
                cmap="YlOrBr",
                vmin=0.0,
                vmax=1.0,
                alpha=0.18,
            )
        frame_axis.text(
            0.5,
            0.5,
            f"frame {int(frame)}",
            transform=frame_axis.transAxes,
            ha="center",
            va="center",
            fontsize=16,
            color="#111827",
        )
        frame_axis.set_xticks([])
        frame_axis.set_yticks([])
        for spine in frame_axis.spines.values():
            spine.set_linewidth(1.0)
            spine.set_edgecolor("#64748b")
        frame_bounds_norm[int(frame)] = (panel_left, panel_bottom, panel_width, panel_height)

    for arrow_layout in constrained_layout["arrow_layouts"]:
        x0 = float((arrow_layout["x0_in"] + x_shift_in) / fig_width_in)
        x1 = float((arrow_layout["x1_in"] + x_shift_in) / fig_width_in)
        y0 = float((arrow_layout["y0_in"] + y_shift_in) / fig_height_in)
        y1 = float((arrow_layout["y1_in"] + y_shift_in) / fig_height_in)
        control_x = float((arrow_layout["control_x_in"] + x_shift_in) / fig_width_in)
        control_y = float((arrow_layout["control_y_in"] + y_shift_in) / fig_height_in)
        bezier_path = Path(
            [(x0, y0), (control_x, control_y), (x1, y1)],
            [Path.MOVETO, Path.CURVE3, Path.CURVE3],
        )
        arrow = FancyArrowPatch(
            path=bezier_path,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.2 + 3.2 * _normalize_strength(float(arrow_layout["routing_strength"])),
            color="#16a34a",
            alpha=0.82,
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=6,
            clip_on=False,
            transform=overlay_axis.transAxes,
        )
        overlay_axis.add_patch(arrow)
        label_x = float((arrow_layout["label_x_in"] + x_shift_in) / fig_width_in)
        label_y = float((arrow_layout["label_y_in"] + y_shift_in) / fig_height_in)
        overlay_axis.text(
            label_x,
            label_y,
            f"-> {int(arrow_layout['target_frame'])}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#166534",
            transform=overlay_axis.transAxes,
            zorder=7,
            clip_on=False,
        )

    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file


def _seed_influence_plot_routing_score_bar_chart(
    frame_scores: Dict[int, float],
    save_file: str,
    title: str,
    highlighted_frames: Optional[Sequence[int]] = None,
    skip_existing_plots: bool = False,
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tc = _seed_influence_import_tc_helpers()
    if tc["maybe_skip_existing_plot"](save_file, bool(skip_existing_plots)):
        return save_file
    finite_items = [
        (int(frame), float(score))
        for frame, score in frame_scores.items()
        if math.isfinite(float(score))
    ]
    if not finite_items:
        return ""
    ranked_items = sorted(finite_items, key=lambda item: (-float(item[1]), int(item[0])))
    highlight_set = {int(frame) for frame in highlighted_frames} if highlighted_frames is not None else set()
    xs = np.arange(len(ranked_items), dtype=np.int32)
    heights = [float(score) for _, score in ranked_items]
    labels = [str(frame) for frame, _ in ranked_items]
    colors = ["#fcd34d" if int(frame) in highlight_set else "#60a5fa" for frame, _ in ranked_items]

    fig_width = max(7.2, 0.52 * len(ranked_items))
    fig, axis = plt.subplots(1, 1, figsize=(fig_width, 4.2))
    axis.bar(xs, heights, color=colors, edgecolor="#334155", linewidth=0.8)
    axis.set_title(title, fontsize=11)
    axis.set_xlabel("frame idx (sorted by score)")
    axis.set_ylabel("routing score")
    axis.set_xticks(xs)
    axis.set_xticklabels(labels, rotation=0, fontsize=8)
    axis.grid(axis="y", alpha=0.22, linestyle="--")
    fig.tight_layout()
    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file


def _seed_influence_plot_step_anchor_top1_frequency(
    rows: Sequence[Dict[str, object]],
    save_file: str,
    title: str,
    value_key: str = "mean_top1_frequency",
    y_label: str = "mean top-1 frequency across layers",
    skip_existing_plots: bool = False,
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tc = _seed_influence_import_tc_helpers()
    if tc["maybe_skip_existing_plot"](save_file, bool(skip_existing_plots)):
        return save_file
    if not rows:
        return ""
    ordered_rows = sorted(rows, key=lambda row: int(row["frame"]))
    frames = [int(row["frame"]) for row in ordered_rows]
    values = [float(row[value_key]) for row in ordered_rows]
    xs = np.arange(len(frames), dtype=np.int32)
    fig, axis = plt.subplots(1, 1, figsize=(8.0, 4.2))
    axis.bar(xs, values, color="#34d399", edgecolor="#065f46", linewidth=0.8)
    axis.set_title(title, fontsize=11)
    axis.set_xlabel("frame idx")
    axis.set_ylabel(str(y_label))
    axis.set_xticks(xs)
    axis.set_xticklabels([str(frame) for frame in frames], fontsize=8)
    axis.grid(axis="y", alpha=0.22, linestyle="--")
    fig.tight_layout()
    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file


def _seed_influence_plot_step_anchor_mean_score(
    rows: Sequence[Dict[str, object]],
    save_file: str,
    title: str,
    skip_existing_plots: bool = False,
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tc = _seed_influence_import_tc_helpers()
    if tc["maybe_skip_existing_plot"](save_file, bool(skip_existing_plots)):
        return save_file
    if not rows:
        return ""
    ordered_rows = sorted(rows, key=lambda row: int(row["frame"]))
    frames = [int(row["frame"]) for row in ordered_rows]
    values = [float(row["mean_score"]) for row in ordered_rows]
    xs = np.arange(len(frames), dtype=np.int32)
    fig, axis = plt.subplots(1, 1, figsize=(8.0, 4.2))
    axis.bar(xs, values, color="#818cf8", edgecolor="#3730a3", linewidth=0.8)
    axis.set_title(title, fontsize=11)
    axis.set_xlabel("frame idx")
    axis.set_ylabel("mean routing score across layers")
    axis.set_xticks(xs)
    axis.set_xticklabels([str(frame) for frame in frames], fontsize=8)
    axis.grid(axis="y", alpha=0.22, linestyle="--")
    fig.tight_layout()
    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file


def _seed_influence_render_anchor_plot_task(task: Tuple):
    plot_kind = str(task[0])
    if plot_kind == "layer_heatmap":
        _, rows, metric_name, save_root_dir, title_prefix, skip_existing_plots = task
        return _seed_influence_plot_anchor_metric_heatmaps_by_layer(
            rows=rows,
            metric_name=str(metric_name),
            save_root_dir=str(save_root_dir),
            title_prefix=str(title_prefix),
            skip_existing_plots=bool(skip_existing_plots),
        )
    if plot_kind == "scatter_overlay":
        (
            _,
            feature_values_np,
            target_values_np,
            role_codes_np,
            anchor_flags_np,
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
        anchor_flags_np = np.asarray(anchor_flags_np, dtype=np.bool_)
        step_values_np = np.asarray(step_values_np, dtype=np.int16)
        plot_path = _seed_influence_plot_anchor_overlay_scatter_points(
            feature_values=feature_values_np,
            target_values=target_values_np,
            role_codes=role_codes_np,
            anchor_flags=anchor_flags_np,
            feature_name=str(feature_name),
            target_label=str(target_label),
            save_file=str(overall_save_file),
            title=str(overall_title),
            skip_existing_plots=bool(skip_existing_plots),
        )
        if plot_path:
            plot_paths.append(plot_path)
        for step_value, save_file, title in by_step_jobs:
            step_mask = step_values_np == int(step_value)
            if not np.any(step_mask):
                continue
            plot_path = _seed_influence_plot_anchor_overlay_scatter_points(
                feature_values=feature_values_np[step_mask],
                target_values=target_values_np[step_mask],
                role_codes=role_codes_np[step_mask],
                anchor_flags=anchor_flags_np[step_mask],
                feature_name=str(feature_name),
                target_label=str(target_label),
                save_file=str(save_file),
                title=str(title),
                skip_existing_plots=bool(skip_existing_plots),
            )
            if plot_path:
                plot_paths.append(plot_path)
        return plot_paths
    raise ValueError(f"Unknown anchor plot task kind: {plot_kind}")


def _seed_influence_seed_shards(seed_list: Sequence[int], world_size: int, rank: int) -> Tuple[List[int], List[bool]]:
    """Split seed list into equal-length shards so every rank performs the same number of forwards."""
    if int(world_size) <= 1:
        return [int(seed) for seed in seed_list], [False for _ in seed_list]
    seeds = [int(seed) for seed in seed_list]
    remainder = len(seeds) % int(world_size)
    padded = list(seeds)
    padding_flags = [False for _ in padded]
    if remainder != 0 and padded:
        pad_count = int(world_size) - int(remainder)
        for _ in range(pad_count):
            padded.append(int(padded[-1]))
            padding_flags.append(True)
    local_seeds = []
    local_padding = []
    for index, seed in enumerate(padded):
        if int(index % int(world_size)) != int(rank):
            continue
        local_seeds.append(int(seed))
        local_padding.append(bool(padding_flags[index]))
    return local_seeds, local_padding


def _run_seed_sensitivity_mode(
    wan21_root: str,
    ckpt_dir: str,
    output_dir: str,
    prompt: str,
    size: Tuple[int, int],
    task: str,
    frame_num: int,
    shift: float,
    sample_solver: str,
    sampling_steps: int,
    guide_scale: float,
    device_id: Optional[int],
    offload_model: bool,
    parallel_cfg: Optional[Wan21T2VParallelConfig],
    target_object_words: Sequence[str],
    target_verb_words: Sequence[str],
    trajectory_consensus_steps: Sequence[int],
    trajectory_consensus_seed_sensitivity_steps: Sequence[int],
    trajectory_consensus_layers: Sequence[int],
    trajectory_consensus_self_heads: Optional[Sequence[str]],
    trajectory_consensus_branch: str,
    trajectory_consensus_candidate_base_quantile: float,
    trajectory_consensus_candidate_split_quantiles: Sequence[float],
    trajectory_consensus_candidate_smooth_radius: int,
    trajectory_consensus_candidate_stable_peak_min_levels: int,
    trajectory_consensus_candidate_peak_merge_distance: float,
    trajectory_consensus_candidate_preprocess_winsorize_quantile: float,
    trajectory_consensus_candidate_preprocess_despike_quantile: float,
    trajectory_consensus_candidate_min_component_area: int,
    trajectory_consensus_object_mask_reference_step: int,
    trajectory_consensus_object_mask_reference_layer: int,
    trajectory_consensus_sa_anchor_step: int,
    trajectory_consensus_sa_anchor_layer: int,
    trajectory_consensus_sa_covered_mass_min: float,
    trajectory_consensus_plot_only_from_csv: bool,
    trajectory_consensus_seed_influence_seeds: Sequence[int],
    trajectory_consensus_seed_sensitivity_zr_metric: str,
    skip_existing_plots: bool = False,
) -> Optional[Dict[str, object]]:
    """Run the seed-sensitivity branch of seed influence."""
    tc = _seed_influence_import_tc_helpers()
    metric_field = _seed_influence_zr_metric_name_to_field(trajectory_consensus_seed_sensitivity_zr_metric)
    sensitivity_root = _seed_influence_sensitivity_root(output_dir)
    _ensure_dir(sensitivity_root)
    framewise_csv_path = os.path.join(sensitivity_root, "trajectory_consensus_seed_sensitivity_framewise.csv")
    seedwise_csv_path = os.path.join(sensitivity_root, "trajectory_consensus_seed_sensitivity_seedwise.csv")
    summary_csv_path = os.path.join(sensitivity_root, "trajectory_consensus_seed_sensitivity_summary.csv")
    z_rows_csv_path = os.path.join(sensitivity_root, "trajectory_consensus_seed_sensitivity_z_rows.csv")
    plot_root = os.path.join(sensitivity_root, "trajectory_consensus_seed_sensitivity_plots")
    _ensure_dir(plot_root)

    if bool(trajectory_consensus_plot_only_from_csv):
        if not os.path.exists(framewise_csv_path):
            raise FileNotFoundError(
                "trajectory_consensus_plot_only_from_csv=True but missing seed influence framewise CSV: "
                f"{framewise_csv_path}"
            )
        framewise_rows = _load_wan21_t2v_csv_rows(framewise_csv_path)
        seedwise_rows = _load_wan21_t2v_csv_rows(seedwise_csv_path) if os.path.exists(seedwise_csv_path) else []
        summary_rows = _load_wan21_t2v_csv_rows(summary_csv_path) if os.path.exists(summary_csv_path) else []
        z_rows = _load_wan21_t2v_csv_rows(z_rows_csv_path) if os.path.exists(z_rows_csv_path) else []
    else:
        if not trajectory_consensus_seed_influence_seeds:
            raise ValueError("trajectory_consensus_seed_influence_seeds must be non-empty for seed_sensitivity mode.")
        parallel_cfg = parallel_cfg or Wan21T2VParallelConfig()
        runtime = _init_wan21_t2v_runtime(parallel_cfg, explicit_device_id=device_id)
        if runtime.world_size > 1 and (parallel_cfg.t5_fsdp or parallel_cfg.dit_fsdp or runtime.use_usp):
            raise RuntimeError(
                "seed_sensitivity currently supports either single-GPU execution or multi-GPU seed sharding with "
                "independent full-model replicas only. Disable USP/FSDP for this mode."
            )
        pipeline, cfg = _build_wan21_t2v_pipeline(
            wan21_root=wan21_root,
            ckpt_dir=ckpt_dir,
            task=task,
            runtime=runtime,
            parallel_cfg=parallel_cfg,
        )
        del cfg
        offload_model = _resolve_wan21_t2v_offload_model(runtime, offload_model)

        local_seeds, local_padding_flags = _seed_influence_seed_shards(
            seed_list=trajectory_consensus_seed_influence_seeds,
            world_size=runtime.world_size,
            rank=runtime.rank,
        )
        selected_steps = (
            _dedup_wan21_t2v_int_list(trajectory_consensus_seed_sensitivity_steps)
            if trajectory_consensus_seed_sensitivity_steps
            else (
                _dedup_wan21_t2v_int_list(trajectory_consensus_steps)
                if trajectory_consensus_steps else [1, 2, 3, 4, 5]
            )
        )
        selected_layers = _dedup_wan21_t2v_int_list(trajectory_consensus_layers) if trajectory_consensus_layers else list(range(len(pipeline.model.blocks)))
        collect_steps = sorted(
            set(int(step) for step in selected_steps)
            | {int(trajectory_consensus_object_mask_reference_step)}
        )
        collect_layers = sorted(
            set(int(layer) for layer in selected_layers)
            | {int(trajectory_consensus_object_mask_reference_layer)}
        )

        local_payloads: List[Dict[str, object]] = []
        seed_loop = zip(local_seeds, local_padding_flags)
        seed_loop = _seed_influence_progress(
            seed_loop,
            desc="seed_sensitivity: seeds",
            total=len(local_seeds),
            enabled=bool(runtime.rank == 0),
        )
        for local_seed, is_padding_seed in seed_loop:
            seed_dir = _seed_influence_seed_dir(output_dir, int(local_seed))
            _ensure_dir(seed_dir)
            mean_maps, word_to_positions, object_words_in_prompt = _seed_influence_collect_cross_attention_mean_maps_for_seed(
                pipeline=pipeline,
                prompt=prompt,
                size=size,
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                seed=int(local_seed),
                offload_model=offload_model,
                target_object_words=target_object_words,
                target_verb_words=target_verb_words,
                collect_steps=collect_steps,
                layers_to_collect=collect_layers,
            )
            torch.save(mean_maps, os.path.join(seed_dir, "seed_influence_cross_attention_mean_maps.pt"))
            _save_json(
                os.path.join(seed_dir, "seed_influence_cross_attention_token_meta.json"),
                {
                    "token_positions": word_to_positions,
                    "target_object_words": list(target_object_words),
                    "target_verb_words": list(target_verb_words),
                },
            )
            object_words_in_maps = [str(word) for word in object_words_in_prompt if str(word) in {str(key[2]) for key in mean_maps.keys()}]
            resolved_steps, resolved_layers = tc["resolve_steps_and_layers"](
                mean_maps=mean_maps,
                requested_steps=selected_steps,
                requested_layers=selected_layers,
            )
            candidate_region_cache, candidate_region_rows, candidate_weight_rows, winner_gap_rows = _seed_influence_build_candidate_consensus_from_mean_maps(
                mean_maps=mean_maps,
                object_words_in_maps=object_words_in_maps,
                selected_steps=resolved_steps,
                selected_layers=resolved_layers,
                reference_step=int(trajectory_consensus_object_mask_reference_step),
                reference_layer=int(trajectory_consensus_object_mask_reference_layer),
                candidate_base_quantile=float(trajectory_consensus_candidate_base_quantile),
                candidate_split_quantiles=trajectory_consensus_candidate_split_quantiles,
                candidate_smooth_radius=int(trajectory_consensus_candidate_smooth_radius),
                candidate_stable_peak_min_levels=int(trajectory_consensus_candidate_stable_peak_min_levels),
                candidate_peak_merge_distance=float(trajectory_consensus_candidate_peak_merge_distance),
                candidate_preprocess_winsorize_quantile=float(trajectory_consensus_candidate_preprocess_winsorize_quantile),
                candidate_preprocess_despike_quantile=float(trajectory_consensus_candidate_preprocess_despike_quantile),
                candidate_min_component_area=int(trajectory_consensus_candidate_min_component_area),
            )
            _seed_influence_save_candidate_cache_pt(
                candidate_region_cache=candidate_region_cache,
                save_path=os.path.join(seed_dir, "trajectory_consensus_candidate_regions.pt"),
            )
            _save_csv(os.path.join(seed_dir, "trajectory_consensus_candidate_regions.csv"), candidate_region_rows)
            _save_csv(os.path.join(seed_dir, "trajectory_consensus_candidate_weights.csv"), candidate_weight_rows)
            _save_csv(os.path.join(seed_dir, "trajectory_consensus_winner_gap.csv"), winner_gap_rows)

            pairwise_rows, candidate_feature_rows, feature_summary_rows, temporal_precedence_rows = _seed_influence_collect_self_attention_features_for_seed(
                pipeline=pipeline,
                prompt=prompt,
                size=size,
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                seed=int(local_seed),
                offload_model=offload_model,
                selected_steps=resolved_steps,
                selected_layers=resolved_layers,
                candidate_region_cache=candidate_region_cache,
                candidate_weight_rows=candidate_weight_rows,
                trajectory_consensus_branch=trajectory_consensus_branch,
                trajectory_consensus_self_heads=trajectory_consensus_self_heads,
                trajectory_consensus_sa_anchor_step=int(trajectory_consensus_sa_anchor_step),
                trajectory_consensus_sa_anchor_layer=int(trajectory_consensus_sa_anchor_layer),
                trajectory_consensus_sa_covered_mass_min=float(trajectory_consensus_sa_covered_mass_min),
            )
            _save_csv(os.path.join(seed_dir, "trajectory_consensus_self_attention_coupling_pairwise.csv"), pairwise_rows)
            _save_csv(os.path.join(seed_dir, "trajectory_consensus_self_attention_coupling_candidate_features.csv"), candidate_feature_rows)
            _save_csv(os.path.join(seed_dir, "trajectory_consensus_self_attention_coupling_feature_summary.csv"), feature_summary_rows)
            _save_csv(os.path.join(seed_dir, "trajectory_consensus_self_attention_coupling_temporal_precedence.csv"), temporal_precedence_rows)
            z_rows = _seed_influence_compute_z_rows(
                candidate_feature_rows=candidate_feature_rows,
                metric_field=str(metric_field),
                seed=int(local_seed),
            )
            local_payloads.append({
                "seed": int(local_seed),
                "is_padding_seed": bool(is_padding_seed),
                "z_rows": z_rows,
            })

        if dist.is_initialized():
            gathered_payloads: List[Optional[List[Dict[str, object]]]] = [None for _ in range(runtime.world_size)]
            dist.all_gather_object(gathered_payloads, local_payloads)
            if runtime.rank != 0:
                return None
            merged_payloads: List[Dict[str, object]] = []
            for payload_list in gathered_payloads:
                if not payload_list:
                    continue
                merged_payloads.extend(payload_list)
        else:
            merged_payloads = local_payloads

        valid_seed_payloads = [payload for payload in merged_payloads if not bool(payload.get("is_padding_seed", False))]
        z_rows = []
        for payload in valid_seed_payloads:
            z_rows.extend(payload.get("z_rows", []))
        framewise_rows, seedwise_rows, summary_rows = _seed_influence_aggregate_seed_sensitivity(z_rows)
        _save_csv(z_rows_csv_path, z_rows)
        _save_csv(framewise_csv_path, framewise_rows)
        _save_csv(seedwise_csv_path, seedwise_rows)
        _save_csv(summary_csv_path, summary_rows)

    plot_paths: List[str] = []
    plot_paths.append(
        _seed_influence_plot_seed_heatmap_rows(
            rows=summary_rows,
            value_key="mean_eta",
            save_file=os.path.join(plot_root, "mean_eta_step_layer_heatmap.pdf"),
            title="Mean standardized competition margin (eta)",
            skip_existing_plots=bool(skip_existing_plots),
        )
    )
    plot_paths.append(
        _seed_influence_plot_seed_heatmap_rows(
            rows=summary_rows,
            value_key="mean_flip_probability",
            save_file=os.path.join(plot_root, "mean_flip_probability_step_layer_heatmap.pdf"),
            title="Mean flip probability",
            skip_existing_plots=bool(skip_existing_plots),
        )
    )
    selected_steps = sorted({int(row["step"]) for row in framewise_rows})
    for step in _seed_influence_progress(selected_steps, desc="seed_sensitivity: frame heatmaps", total=len(selected_steps)):
        plot_paths.append(
            _seed_influence_plot_step_frame_heatmap(
                framewise_rows=framewise_rows,
                selected_step=int(step),
                value_key="eta",
                save_file=os.path.join(plot_root, "frame_level_eta", f"step_{int(step):03d}.pdf"),
                title=f"Frame-level eta at step={int(step)}",
                skip_existing_plots=bool(skip_existing_plots),
            )
        )
        plot_paths.append(
            _seed_influence_plot_step_frame_heatmap(
                framewise_rows=framewise_rows,
                selected_step=int(step),
                value_key="flip_probability",
                save_file=os.path.join(plot_root, "frame_level_flip_probability", f"step_{int(step):03d}.pdf"),
                title=f"Frame-level flip probability at step={int(step)}",
                skip_existing_plots=bool(skip_existing_plots),
            )
        )

    z_rows = _load_wan21_t2v_csv_rows(z_rows_csv_path) if os.path.exists(z_rows_csv_path) else []
    summary_rows_for_canvas = sorted(
        summary_rows,
        key=lambda row: (
            float(row["mean_eta"]) if str(row.get("mean_eta", "")) != "" else float("inf"),
            int(row["step"]),
            int(row["layer"]),
        ),
    )
    for canvas_row in summary_rows_for_canvas[: min(3, len(summary_rows_for_canvas))]:
        step = int(canvas_row["step"])
        layer = int(canvas_row["layer"])
        frame_rows = [
            row for row in framewise_rows
            if int(row["step"]) == int(step) and int(row["layer"]) == int(layer)
        ]
        plot_paths.append(
            _seed_influence_plot_standardized_deviation_canvas(
                z_rows=[
                    row for row in z_rows
                    if int(row["step"]) == int(step) and int(row["layer"]) == int(layer)
                ],
                framewise_rows=frame_rows,
                save_file=os.path.join(
                    plot_root,
                    "seed_deviation_canvas",
                    f"step_{int(step):03d}_layer_{int(layer):02d}.pdf",
                ),
                title=f"Standardized seed deviations | step={int(step)} layer={int(layer)}",
                skip_existing_plots=bool(skip_existing_plots),
            )
        )

    plot_paths = [path for path in plot_paths if path]
    summary = {
        "mode": "seed_sensitivity",
        "z_r_metric_field": str(metric_field),
        "num_framewise_rows": int(len(framewise_rows)),
        "num_seedwise_rows": int(len(seedwise_rows)),
        "num_summary_rows": int(len(summary_rows)),
        "plot_paths": plot_paths,
        "framewise_csv_path": framewise_csv_path,
        "seedwise_csv_path": seedwise_csv_path,
        "summary_csv_path": summary_csv_path,
        "z_rows_csv_path": z_rows_csv_path,
    }
    _save_json(os.path.join(sensitivity_root, "trajectory_consensus_seed_sensitivity_summary.json"), summary)
    return summary


def _run_anchor_frame_mode(
    output_dir: str,
    frame_num: int,
    reuse_cross_attention_dir: Optional[str],
    target_object_words: Sequence[str],
    target_verb_words: Sequence[str],
    trajectory_consensus_steps: Sequence[int],
    trajectory_consensus_anchor_frame_steps: Sequence[int],
    trajectory_consensus_layers: Sequence[int],
    trajectory_consensus_seed_influence_anchor_topk: int,
    trajectory_consensus_seed_influence_arrow_topk: int,
    num_workers: int = 0,
    skip_existing_plots: bool = False,
) -> Dict[str, object]:
    """Run the frame-level anchor analysis from saved caches only."""
    tc = _seed_influence_import_tc_helpers()
    safe_float = tc["safe_float"]
    maybe_skip_existing_plot = tc["maybe_skip_existing_plot"]
    anchor_root = _seed_influence_anchor_root(output_dir)
    _ensure_dir(anchor_root)
    framewise_csv_path = os.path.join(anchor_root, "trajectory_consensus_anchor_frames.csv")
    summary_json_path = os.path.join(anchor_root, "trajectory_consensus_anchor_summary.json")

    candidate_region_cache, candidate_region_rows, candidate_weight_rows, winner_gap_rows = tc["load_candidate_cache"](output_dir)
    del candidate_region_rows, candidate_weight_rows
    candidate_feature_csv_path = os.path.join(output_dir, "trajectory_consensus_self_attention_coupling_candidate_features.csv")
    pairwise_csv_path = os.path.join(output_dir, "trajectory_consensus_self_attention_coupling_pairwise.csv")
    if not os.path.exists(candidate_feature_csv_path):
        raise FileNotFoundError(
            "anchor_frame mode requires existing self_attention_coupling candidate features: "
            f"{candidate_feature_csv_path}"
        )
    candidate_feature_rows = _load_wan21_t2v_csv_rows(candidate_feature_csv_path)
    pairwise_rows = _load_wan21_t2v_csv_rows(pairwise_csv_path) if os.path.exists(pairwise_csv_path) else []
    mean_maps, loaded_map_path = _load_wan21_t2v_cross_attention_mean_maps_from_disk(reuse_cross_attention_dir)
    words_in_maps = sorted({str(key[2]) for key in mean_maps.keys()})
    _load_wan21_t2v_cross_attention_token_meta(
        output_dir=reuse_cross_attention_dir,
        words_in_maps=words_in_maps,
        target_object_words=target_object_words,
        target_verb_words=target_verb_words,
    )
    object_words_in_maps = [str(word) for word in target_object_words if str(word) in words_in_maps]

    selected_steps = (
        _dedup_wan21_t2v_int_list(trajectory_consensus_anchor_frame_steps)
        if trajectory_consensus_anchor_frame_steps
        else (
            _dedup_wan21_t2v_int_list(trajectory_consensus_steps)
            if trajectory_consensus_steps else sorted({int(row["step"]) for row in candidate_feature_rows})
        )
    )
    selected_layers = _dedup_wan21_t2v_int_list(trajectory_consensus_layers) if trajectory_consensus_layers else sorted({int(row["layer"]) for row in candidate_feature_rows})

    winner_gap_by_key = {
        (int(row["step"]), int(row["layer"]), int(row["frame"])): row
        for row in winner_gap_rows
    }
    rows_by_step_layer_frame: Dict[Tuple[int, int, int], List[Dict[str, object]]] = defaultdict(list)
    for row in candidate_feature_rows:
        step = int(row["step"])
        layer = int(row["layer"])
        frame_index = int(row["frame"])
        if step not in selected_steps or layer not in selected_layers:
            continue
        rows_by_step_layer_frame[(step, layer, frame_index)].append(row)

    spatial_entropy_by_key: Dict[Tuple[int, int, int], float] = {}
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
            for frame_index, entropy_value in enumerate(_seed_influence_compute_spatial_entropy(headmean_map)):
                spatial_entropy_by_key[(int(step), int(layer), int(frame_index))] = float(entropy_value)

    framewise_rows: List[Dict[str, object]] = []
    for (step, layer, frame_index), group_rows in sorted(rows_by_step_layer_frame.items()):
        winner_gap_row = winner_gap_by_key.get((int(step), int(layer), int(frame_index)), {})
        spatial_entropy = float(spatial_entropy_by_key.get((int(step), int(layer), int(frame_index)), float("nan")))
        leader_row = max(
            group_rows,
            key=lambda row: (
                -1e9 if not math.isfinite(safe_float(row.get("proposal_pi", float("nan"))))
                else safe_float(row.get("proposal_pi", float("nan")))
            ),
        )
        local_mc_values = [
            safe_float(row.get("local_mutual_consistency", float("nan")))
            for row in group_rows
            if math.isfinite(safe_float(row.get("local_mutual_consistency", float("nan"))))
        ]
        global_mc_values = [
            safe_float(row.get("global_mutual_consistency", float("nan")))
            for row in group_rows
            if math.isfinite(safe_float(row.get("global_mutual_consistency", float("nan"))))
        ]
        local_incoming_support_values = [
            safe_float(row.get("local_incoming_support", float("nan")))
            for row in group_rows
            if math.isfinite(safe_float(row.get("local_incoming_support", float("nan"))))
        ]
        global_incoming_support_values = [
            safe_float(row.get("global_incoming_support", float("nan")))
            for row in group_rows
            if math.isfinite(safe_float(row.get("global_incoming_support", float("nan"))))
        ]
        local_incoming_preference_values = [
            safe_float(row.get("local_incoming_preference_share", float("nan")))
            for row in group_rows
            if math.isfinite(safe_float(row.get("local_incoming_preference_share", float("nan"))))
        ]
        global_incoming_preference_values = [
            safe_float(row.get("global_incoming_preference_share", float("nan")))
            for row in group_rows
            if math.isfinite(safe_float(row.get("global_incoming_preference_share", float("nan"))))
        ]
        local_incoming_vote_values = [
            safe_float(row.get("local_incoming_vote_share", float("nan")))
            for row in group_rows
            if math.isfinite(safe_float(row.get("local_incoming_vote_share", float("nan"))))
        ]
        global_incoming_vote_values = [
            safe_float(row.get("global_incoming_vote_share", float("nan")))
            for row in group_rows
            if math.isfinite(safe_float(row.get("global_incoming_vote_share", float("nan"))))
        ]
        local_mc_sorted = sorted(local_mc_values, reverse=True)
        global_mc_sorted = sorted(global_mc_values, reverse=True)
        local_incoming_support_sorted = sorted(local_incoming_support_values, reverse=True)
        global_incoming_support_sorted = sorted(global_incoming_support_values, reverse=True)
        local_incoming_preference_sorted = sorted(local_incoming_preference_values, reverse=True)
        global_incoming_preference_sorted = sorted(global_incoming_preference_values, reverse=True)
        local_incoming_vote_sorted = sorted(local_incoming_vote_values, reverse=True)
        global_incoming_vote_sorted = sorted(global_incoming_vote_values, reverse=True)
        framewise_rows.append({
            "step": int(step),
            "layer": int(layer),
            "frame": int(frame_index),
            "candidate_count": int(len(group_rows)),
            "candidate_entropy": float(safe_float(winner_gap_row.get("candidate_entropy", float("nan")))),
            "winner_gap": float(safe_float(winner_gap_row.get("winner_gap", float("nan")))),
            "spatial_map_entropy": float(spatial_entropy),
            "leader_candidate_index": int(leader_row.get("candidate_index", -1)),
            "leader_anchor_iou": float(safe_float(leader_row.get("anchor_iou", float("nan")))),
            "leader_anchor_distance": float(safe_float(leader_row.get("anchor_distance", float("nan")))),
            "leader_local_mutual_consistency": float(safe_float(leader_row.get("local_mutual_consistency", float("nan")))),
            "leader_global_mutual_consistency": float(safe_float(leader_row.get("global_mutual_consistency", float("nan")))),
            "leader_local_incoming_support": float(safe_float(leader_row.get("local_incoming_support", float("nan")))),
            "leader_global_incoming_support": float(safe_float(leader_row.get("global_incoming_support", float("nan")))),
            "leader_local_incoming_preference_share": float(safe_float(leader_row.get("local_incoming_preference_share", float("nan")))),
            "leader_global_incoming_preference_share": float(safe_float(leader_row.get("global_incoming_preference_share", float("nan")))),
            "leader_local_incoming_vote_share": float(safe_float(leader_row.get("local_incoming_vote_share", float("nan")))),
            "leader_global_incoming_vote_share": float(safe_float(leader_row.get("global_incoming_vote_share", float("nan")))),
            "frame_local_mutual_consistency_sum": float(_seed_influence_finite_sum(local_mc_values)),
            "frame_global_mutual_consistency_sum": float(_seed_influence_finite_sum(global_mc_values)),
            "frame_local_mutual_consistency_mean": float(_seed_influence_finite_mean(local_mc_values)),
            "frame_global_mutual_consistency_mean": float(_seed_influence_finite_mean(global_mc_values)),
            "frame_local_mutual_consistency_max": float(local_mc_sorted[0]) if local_mc_sorted else float("nan"),
            "frame_global_mutual_consistency_max": float(global_mc_sorted[0]) if global_mc_sorted else float("nan"),
            "frame_local_mutual_consistency_gap": float(_seed_influence_top1_top2_gap(local_mc_sorted)),
            "frame_global_mutual_consistency_gap": float(_seed_influence_top1_top2_gap(global_mc_sorted)),
            "frame_local_incoming_support_sum": float(_seed_influence_finite_sum(local_incoming_support_values)),
            "frame_global_incoming_support_sum": float(_seed_influence_finite_sum(global_incoming_support_values)),
            "frame_local_incoming_support_mean": float(_seed_influence_finite_mean(local_incoming_support_values)),
            "frame_global_incoming_support_mean": float(_seed_influence_finite_mean(global_incoming_support_values)),
            "frame_local_incoming_support_max": float(local_incoming_support_sorted[0]) if local_incoming_support_sorted else float("nan"),
            "frame_global_incoming_support_max": float(global_incoming_support_sorted[0]) if global_incoming_support_sorted else float("nan"),
            "frame_local_incoming_support_gap": float(_seed_influence_top1_top2_gap(local_incoming_support_sorted)),
            "frame_global_incoming_support_gap": float(_seed_influence_top1_top2_gap(global_incoming_support_sorted)),
            "frame_local_incoming_preference_share_sum": float(_seed_influence_finite_sum(local_incoming_preference_values)),
            "frame_global_incoming_preference_share_sum": float(_seed_influence_finite_sum(global_incoming_preference_values)),
            "frame_local_incoming_preference_share_mean": float(_seed_influence_finite_mean(local_incoming_preference_values)),
            "frame_global_incoming_preference_share_mean": float(_seed_influence_finite_mean(global_incoming_preference_values)),
            "frame_local_incoming_preference_share_max": float(local_incoming_preference_sorted[0]) if local_incoming_preference_sorted else float("nan"),
            "frame_global_incoming_preference_share_max": float(global_incoming_preference_sorted[0]) if global_incoming_preference_sorted else float("nan"),
            "frame_local_incoming_preference_share_gap": float(_seed_influence_top1_top2_gap(local_incoming_preference_sorted)),
            "frame_global_incoming_preference_share_gap": float(_seed_influence_top1_top2_gap(global_incoming_preference_sorted)),
            "frame_local_incoming_vote_share_sum": float(_seed_influence_finite_sum(local_incoming_vote_values)),
            "frame_global_incoming_vote_share_sum": float(_seed_influence_finite_sum(global_incoming_vote_values)),
            "frame_local_incoming_vote_share_mean": float(_seed_influence_finite_mean(local_incoming_vote_values)),
            "frame_global_incoming_vote_share_mean": float(_seed_influence_finite_mean(global_incoming_vote_values)),
            "frame_local_incoming_vote_share_max": float(local_incoming_vote_sorted[0]) if local_incoming_vote_sorted else float("nan"),
            "frame_global_incoming_vote_share_max": float(global_incoming_vote_sorted[0]) if global_incoming_vote_sorted else float("nan"),
            "frame_local_incoming_vote_share_gap": float(_seed_influence_top1_top2_gap(local_incoming_vote_sorted)),
            "frame_global_incoming_vote_share_gap": float(_seed_influence_top1_top2_gap(global_incoming_vote_sorted)),
        })

    _save_csv(framewise_csv_path, framewise_rows)

    plot_paths: List[str] = []
    metric_specs = [
        ("candidate_entropy", False),
        ("winner_gap", True),
        ("spatial_map_entropy", False),
        ("leader_local_mutual_consistency", True),
        ("leader_global_mutual_consistency", True),
        ("leader_local_incoming_support", True),
        ("leader_global_incoming_support", True),
        ("leader_local_incoming_preference_share", True),
        ("leader_global_incoming_preference_share", True),
        ("leader_local_incoming_vote_share", True),
        ("leader_global_incoming_vote_share", True),
        ("frame_local_mutual_consistency_sum", True),
        ("frame_global_mutual_consistency_sum", True),
        ("frame_local_mutual_consistency_mean", True),
        ("frame_global_mutual_consistency_mean", True),
        ("frame_local_mutual_consistency_max", True),
        ("frame_global_mutual_consistency_max", True),
        ("frame_local_mutual_consistency_gap", True),
        ("frame_global_mutual_consistency_gap", True),
        ("frame_local_incoming_support_sum", True),
        ("frame_global_incoming_support_sum", True),
        ("frame_local_incoming_support_mean", True),
        ("frame_global_incoming_support_mean", True),
        ("frame_local_incoming_support_max", True),
        ("frame_global_incoming_support_max", True),
        ("frame_local_incoming_support_gap", True),
        ("frame_global_incoming_support_gap", True),
        ("frame_local_incoming_preference_share_sum", True),
        ("frame_global_incoming_preference_share_sum", True),
        ("frame_local_incoming_preference_share_mean", True),
        ("frame_global_incoming_preference_share_mean", True),
        ("frame_local_incoming_preference_share_max", True),
        ("frame_global_incoming_preference_share_max", True),
        ("frame_local_incoming_preference_share_gap", True),
        ("frame_global_incoming_preference_share_gap", True),
        ("frame_local_incoming_vote_share_sum", True),
        ("frame_global_incoming_vote_share_sum", True),
        ("frame_local_incoming_vote_share_mean", True),
        ("frame_global_incoming_vote_share_mean", True),
        ("frame_local_incoming_vote_share_max", True),
        ("frame_global_incoming_vote_share_max", True),
        ("frame_local_incoming_vote_share_gap", True),
        ("frame_global_incoming_vote_share_gap", True),
    ]

    scatter_feature_names = [
        "local_mutual_consistency",
        "global_mutual_consistency",
        "local_incoming_support",
        "global_incoming_support",
        "local_incoming_preference_share",
        "global_incoming_preference_share",
        "local_incoming_vote_share",
        "global_incoming_vote_share",
    ]
    all_scatter_steps = sorted({int(row["step"]) for row in candidate_feature_rows})

    anchor_plot_tasks: List[Tuple] = []
    for metric_name, higher_is_better in metric_specs:
        heatmap_save_root = os.path.join(anchor_root, "metric_heatmaps", str(metric_name))
        anchor_plot_tasks.append(
            (
                "layer_heatmap",
                framewise_rows,
                metric_name,
                heatmap_save_root,
                f"Anchor-frame heatmap | {_seed_influence_metric_display_name(metric_name)}",
                bool(skip_existing_plots),
            )
        )
        anchor_frame_map = _seed_influence_build_anchor_frame_highlight_payload(
            rows=framewise_rows,
            metric_name=metric_name,
            topk=int(trajectory_consensus_seed_influence_anchor_topk),
            higher_is_better=bool(higher_is_better),
        )
        for scatter_feature_name in scatter_feature_names:
            scatter_payload = _seed_influence_build_candidate_scatter_payload_with_anchor_overlay(
                candidate_feature_rows=candidate_feature_rows,
                feature_name=str(scatter_feature_name),
                target_key="anchor_distance",
                anchor_frame_map=anchor_frame_map,
                min_candidate_count=1,
            )
            if scatter_payload is None:
                continue
            metric_scatter_dir = os.path.join(anchor_root, "scatter", str(metric_name), str(scatter_feature_name))
            overall_dir = os.path.join(metric_scatter_dir, "overall")
            by_step_dir = os.path.join(metric_scatter_dir, "by_step")
            _ensure_dir(overall_dir)
            _ensure_dir(by_step_dir)
            overall_save = os.path.join(overall_dir, f"{scatter_feature_name}_vs_anchor_dist.pdf")
            by_step_jobs = []
            for step in all_scatter_steps:
                step_save = os.path.join(
                    by_step_dir,
                    f"step_{int(step):03d}",
                    f"{scatter_feature_name}_vs_anchor_dist_step_{int(step):03d}.pdf",
                )
                by_step_jobs.append(
                    (
                        int(step),
                        step_save,
                        f"{_seed_influence_metric_display_name(scatter_feature_name)} versus anchor-distance score at step={int(step)}",
                    )
                )
            if bool(skip_existing_plots):
                overall_exists = maybe_skip_existing_plot(overall_save, True)
                step_exists = bool(by_step_jobs) and all(
                    maybe_skip_existing_plot(str(step_save), True)
                    for _, step_save, _ in by_step_jobs
                )
                if overall_exists and step_exists:
                    plot_paths.append(overall_save)
                    plot_paths.extend([str(step_save) for _, step_save, _ in by_step_jobs])
                    continue
            anchor_plot_tasks.append(
                (
                    "scatter_overlay",
                    scatter_payload[0],
                    scatter_payload[1],
                    scatter_payload[2],
                    scatter_payload[3],
                    scatter_payload[4],
                    str(scatter_feature_name),
                    "anchor-distance score",
                    overall_save,
                    f"{_seed_influence_metric_display_name(scatter_feature_name)} versus anchor-distance score",
                    tuple(by_step_jobs),
                    bool(skip_existing_plots),
                )
            )

    metric_progress_bar = None
    if anchor_plot_tasks:
        try:
            from tqdm import tqdm
            metric_progress_bar = tqdm(
                total=int(len(anchor_plot_tasks)),
                desc="anchor_frame: metrics",
                unit="plot",
                leave=True,
            )
        except Exception:
            metric_progress_bar = None
    try:
        effective_num_workers = _resolve_wan21_t2v_num_workers(
            requested_num_workers=int(num_workers),
            task_count=int(len(anchor_plot_tasks)),
        )
        for plot_result in _iter_wan21_t2v_parallel_results(
            tasks=anchor_plot_tasks,
            worker_fn=_seed_influence_render_anchor_plot_task,
            num_workers=int(effective_num_workers),
        ):
            if isinstance(plot_result, (list, tuple)):
                plot_paths.extend([path for path in plot_result if path])
            elif plot_result:
                plot_paths.append(plot_result)
            if metric_progress_bar is not None:
                metric_progress_bar.update(1)
    finally:
        if metric_progress_bar is not None:
            metric_progress_bar.close()

    pairwise_rows_by_step_layer: Dict[Tuple[int, int], List[Dict[str, object]]] = defaultdict(list)
    for row in pairwise_rows:
        pairwise_rows_by_step_layer[(int(row["step"]), int(row["layer"]))].append(row)
    feature_rows_by_step_layer_frame: Dict[Tuple[int, int, int], List[Dict[str, object]]] = defaultdict(list)
    for row in candidate_feature_rows:
        feature_rows_by_step_layer_frame[(int(row["step"]), int(row["layer"]), int(row["frame"]))].append(row)
    token_grid_shape_by_step_layer: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for (step, layer), candidate_payload in candidate_region_cache.items():
        label_map_fhw = torch.as_tensor(candidate_payload["label_map_fhw"])
        if int(label_map_fhw.ndim) != 3:
            continue
        token_grid_shape_by_step_layer[(int(step), int(layer))] = (
            int(label_map_fhw.shape[1]),
            int(label_map_fhw.shape[2]),
        )

    routing_summary_rows: List[Dict[str, object]] = []
    step_mode_frame_scores: Dict[Tuple[str, int], Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    step_mode_top1_counts: Dict[Tuple[str, int], Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    step_mode_top2_counts: Dict[Tuple[str, int], Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    arrow_jobs = [
        (str(mode_name), step_layer_key, pairwise_group_rows)
        for mode_name in ["leader", "avg"]
        for step_layer_key, pairwise_group_rows in sorted(pairwise_rows_by_step_layer.items())
    ]
    for mode_name, (step, layer), pairwise_group_rows in _seed_influence_progress(
        arrow_jobs,
        desc="anchor_frame: routing arrows",
        total=len(arrow_jobs),
    ):
            if int(step) not in selected_steps or int(layer) not in selected_layers or not pairwise_group_rows:
                continue
            raw_vectors = tc["pairwise_value_vectors"](
                pairwise_rows=pairwise_group_rows,
                value_key="raw_coupling",
            )
            all_frames = sorted({int(row["frame"]) for row in framewise_rows if int(row["step"]) == int(step) and int(row["layer"]) == int(layer)})
            full_frame_scores: Dict[int, float] = {}
            for target_frame in all_frames:
                per_source_scores: List[float] = []
                for source_frame in all_frames:
                    if int(source_frame) == int(target_frame):
                        continue
                    frame_candidate_rows = feature_rows_by_step_layer_frame.get((int(step), int(layer), int(source_frame)), [])
                    if not frame_candidate_rows:
                        continue
                    leader_candidate = max(
                        frame_candidate_rows,
                        key=lambda row: safe_float(row.get("proposal_pi", float("nan"))),
                    ).get("candidate_index", -1)
                    candidate_indices = [int(row["candidate_index"]) for row in frame_candidate_rows]
                    source_score = float("nan")
                    if str(mode_name) == "leader":
                        vector = raw_vectors.get((int(source_frame), int(leader_candidate), int(target_frame)))
                        source_score = float(np.asarray(vector, dtype=np.float64).sum()) if vector is not None else float("nan")
                    else:
                        per_candidate_values = []
                        for candidate_index in candidate_indices:
                            vector = raw_vectors.get((int(source_frame), int(candidate_index), int(target_frame)))
                            if vector is None:
                                continue
                            per_candidate_values.append(float(np.asarray(vector, dtype=np.float64).sum()))
                        source_score = (
                            float(sum(per_candidate_values) / len(per_candidate_values))
                            if per_candidate_values else float("nan")
                        )
                    if math.isfinite(source_score):
                        per_source_scores.append(float(source_score))
                full_frame_scores[int(target_frame)] = (
                    float(sum(per_source_scores) / len(per_source_scores))
                    if per_source_scores else float("nan")
                )
            ranked_full_frames = [
                int(frame)
                for frame, score in sorted(
                    full_frame_scores.items(),
                    key=lambda item: (
                        -float(item[1]) if math.isfinite(float(item[1])) else float("inf"),
                        int(item[0]),
                    ),
                )
                if math.isfinite(float(score))
            ]
            anchor_topk_frames = ranked_full_frames[: max(1, int(trajectory_consensus_seed_influence_anchor_topk))]
            if ranked_full_frames:
                step_mode_top1_counts[(str(mode_name), int(step))][int(ranked_full_frames[0])] += 1
            for topk_frame in anchor_topk_frames:
                step_mode_top2_counts[(str(mode_name), int(step))][int(topk_frame)] += 1
            for frame, score in full_frame_scores.items():
                if math.isfinite(float(score)):
                    step_mode_frame_scores[(str(mode_name), int(step))][int(frame)].append(float(score))
                routing_summary_rows.append({
                    "mode": str(mode_name),
                    "step": int(step),
                    "layer": int(layer),
                    "frame": int(frame),
                    "routing_score": float(score),
                    "is_top1_anchor": int(bool(ranked_full_frames) and int(frame) == int(ranked_full_frames[0])),
                    "is_topk_anchor": int(int(frame) in {int(x) for x in anchor_topk_frames}),
                })
            sampled_frames = tc["sample_evenly_spaced_indices"](len(all_frames), max_count=10)
            sampled_frame_ids = [int(all_frames[index]) for index in sampled_frames]
            routing_rows: List[Dict[str, object]] = []
            for source_frame in sampled_frame_ids:
                frame_candidate_rows = feature_rows_by_step_layer_frame.get((int(step), int(layer), int(source_frame)), [])
                if not frame_candidate_rows:
                    continue
                leader_candidate = max(
                    frame_candidate_rows,
                    key=lambda row: safe_float(row.get("proposal_pi", float("nan"))),
                ).get("candidate_index", -1)
                candidate_indices = [int(row["candidate_index"]) for row in frame_candidate_rows]
                for target_frame in sampled_frame_ids:
                    if int(target_frame) == int(source_frame):
                        continue
                    routing_strength = float("nan")
                    if str(mode_name) == "leader":
                        vector = raw_vectors.get((int(source_frame), int(leader_candidate), int(target_frame)))
                        routing_strength = float(np.asarray(vector, dtype=np.float64).sum()) if vector is not None else float("nan")
                    else:
                        per_candidate_values = []
                        for candidate_index in candidate_indices:
                            vector = raw_vectors.get((int(source_frame), int(candidate_index), int(target_frame)))
                            if vector is None:
                                continue
                            per_candidate_values.append(float(np.asarray(vector, dtype=np.float64).sum()))
                        routing_strength = float(sum(per_candidate_values) / len(per_candidate_values)) if per_candidate_values else float("nan")
                    if not math.isfinite(routing_strength):
                        continue
                    routing_rows.append({
                        "step": int(step),
                        "layer": int(layer),
                        "source_frame": int(source_frame),
                        "target_frame": int(target_frame),
                        "routing_strength": float(routing_strength),
                        "is_sampled_frame": 1,
                    })
            plot_paths.append(
                _seed_influence_plot_anchor_arrow_panel(
                    routing_rows=routing_rows,
                    save_file=os.path.join(
                        anchor_root,
                        "routing_arrows",
                        str(mode_name),
                        f"step_{int(step):03d}",
                        f"layer_{int(layer):02d}.pdf",
                    ),
                    title=f"Frame-to-frame routing | mode={str(mode_name)} | step={int(step)} layer={int(layer)}",
                    topk=int(trajectory_consensus_seed_influence_arrow_topk),
                    token_grid_shape=token_grid_shape_by_step_layer.get((int(step), int(layer))),
                    highlighted_frames=tuple(int(frame) for frame in anchor_topk_frames if int(frame) in sampled_frame_ids),
                    skip_existing_plots=bool(skip_existing_plots),
                )
            )
            plot_paths.append(
                _seed_influence_plot_routing_score_bar_chart(
                    frame_scores=full_frame_scores,
                    save_file=os.path.join(
                        anchor_root,
                        "routing_arrows",
                        str(mode_name),
                        f"step_{int(step):03d}",
                        f"layer_{int(layer):02d}_anchor_score_distribution.pdf",
                    ),
                    title=f"Anchor-frame score distribution | mode={str(mode_name)} | step={int(step)} layer={int(layer)}",
                    highlighted_frames=tuple(int(frame) for frame in anchor_topk_frames),
                    skip_existing_plots=bool(skip_existing_plots),
                )
            )

    routing_summary_csv_path = os.path.join(anchor_root, "trajectory_consensus_anchor_routing_scores.csv")
    _save_csv(routing_summary_csv_path, routing_summary_rows)
    for mode_name in ["leader", "avg"]:
        for step in selected_steps:
            mean_score_rows: List[Dict[str, object]] = []
            top1_frequency_rows: List[Dict[str, object]] = []
            top2_frequency_rows: List[Dict[str, object]] = []
            frame_score_map = step_mode_frame_scores.get((str(mode_name), int(step)), {})
            top1_count_map = step_mode_top1_counts.get((str(mode_name), int(step)), {})
            top2_count_map = step_mode_top2_counts.get((str(mode_name), int(step)), {})
            denominator = max(1, len(selected_layers))
            all_frames = sorted(
                {
                    int(row["frame"])
                    for row in framewise_rows
                    if int(row["step"]) == int(step)
                }
            )
            for frame in all_frames:
                frame_scores = frame_score_map.get(int(frame), [])
                mean_score_rows.append({
                    "frame": int(frame),
                    "mean_score": (
                        float(sum(frame_scores) / len(frame_scores))
                        if frame_scores else float("nan")
                    ),
                })
                top1_frequency_rows.append({
                    "frame": int(frame),
                    "mean_top1_frequency": float(top1_count_map.get(int(frame), 0) / float(denominator)),
                })
                top2_frequency_rows.append({
                    "frame": int(frame),
                    "mean_top2_frequency": float(top2_count_map.get(int(frame), 0) / float(denominator)),
                })
            plot_paths.append(
                _seed_influence_plot_step_anchor_top1_frequency(
                    rows=top1_frequency_rows,
                    save_file=os.path.join(
                        anchor_root,
                        "routing_arrows",
                        str(mode_name),
                        f"step_{int(step):03d}",
                        "top1_anchor_frequency_across_layers.pdf",
                    ),
                    title=f"Top-1 anchor frequency across layers | mode={str(mode_name)} | step={int(step)}",
                    value_key="mean_top1_frequency",
                    y_label="mean top-1 frequency across layers",
                    skip_existing_plots=bool(skip_existing_plots),
                )
            )
            plot_paths.append(
                _seed_influence_plot_step_anchor_top1_frequency(
                    rows=top2_frequency_rows,
                    save_file=os.path.join(
                        anchor_root,
                        "routing_arrows",
                        str(mode_name),
                        f"step_{int(step):03d}",
                        "top1or2_anchor_frequency_across_layers.pdf",
                    ),
                    title=f"Top-1-or-2 anchor frequency across layers | mode={str(mode_name)} | step={int(step)}",
                    value_key="mean_top2_frequency",
                    y_label="mean top-1-or-2 frequency across layers",
                    skip_existing_plots=bool(skip_existing_plots),
                )
            )
            plot_paths.append(
                _seed_influence_plot_step_anchor_mean_score(
                    rows=mean_score_rows,
                    save_file=os.path.join(
                        anchor_root,
                        "routing_arrows",
                        str(mode_name),
                        f"step_{int(step):03d}",
                        "mean_anchor_score_across_layers.pdf",
                    ),
                    title=f"Mean anchor score across layers | mode={str(mode_name)} | step={int(step)}",
                    skip_existing_plots=bool(skip_existing_plots),
                )
            )

    plot_paths = [path for path in plot_paths if path]
    summary = {
        "mode": "anchor_frame",
        "loaded_map_path": loaded_map_path,
        "framewise_csv_path": framewise_csv_path,
        "routing_summary_csv_path": routing_summary_csv_path,
        "num_framewise_rows": int(len(framewise_rows)),
        "plot_paths": plot_paths,
    }
    _save_json(summary_json_path, summary)
    return summary


def run_wan21_t2v_trajectory_consensus_seed_influence(
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
    device_id: Optional[int] = None,
    offload_model: bool = True,
    parallel_cfg: Optional[Wan21T2VParallelConfig] = None,
    target_object_words: Sequence[str] = tuple(),
    target_verb_words: Sequence[str] = tuple(),
    reuse_cross_attention_dir: Optional[str] = None,
    trajectory_consensus_steps: Sequence[int] = tuple(),
    trajectory_consensus_seed_sensitivity_steps: Sequence[int] = tuple(),
    trajectory_consensus_anchor_frame_steps: Sequence[int] = tuple(),
    trajectory_consensus_layers: Sequence[int] = tuple(),
    trajectory_consensus_self_heads: Optional[Sequence[str]] = tuple(),
    trajectory_consensus_branch: str = "cond",
    trajectory_consensus_candidate_base_quantile: float = 0.85,
    trajectory_consensus_candidate_split_quantiles: Sequence[float] = (0.92, 0.95, 0.97),
    trajectory_consensus_candidate_smooth_radius: int = 1,
    trajectory_consensus_candidate_stable_peak_min_levels: int = 2,
    trajectory_consensus_candidate_peak_merge_distance: float = 2.0,
    trajectory_consensus_candidate_preprocess_winsorize_quantile: float = 0.995,
    trajectory_consensus_candidate_preprocess_despike_quantile: float = 0.98,
    trajectory_consensus_candidate_min_component_area: int = 4,
    trajectory_consensus_object_mask_reference_step: int = 50,
    trajectory_consensus_object_mask_reference_layer: int = 27,
    trajectory_consensus_sa_anchor_step: int = 49,
    trajectory_consensus_sa_anchor_layer: int = 27,
    trajectory_consensus_sa_covered_mass_min: float = 0.0,
    trajectory_consensus_plot_only_from_csv: bool = False,
    trajectory_consensus_seed_influence_mode: str = "seed_sensitivity",
    trajectory_consensus_seed_influence_seeds: Sequence[int] = tuple(),
    trajectory_consensus_seed_sensitivity_zr_metric: str = "global_mutual_consistency",
    trajectory_consensus_seed_influence_anchor_topk: int = 2,
    trajectory_consensus_seed_influence_arrow_topk: int = 2,
    trajectory_consensus_num_workers: int = 0,
    trajectory_consensus_skip_existing_plots: bool = True,
) -> Optional[Dict[str, object]]:
    """Run the seed influence follow-up analysis."""
    mode = str(trajectory_consensus_seed_influence_mode).strip().lower()
    if mode == "seed_sensitivity":
        return _run_seed_sensitivity_mode(
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
            trajectory_consensus_steps=trajectory_consensus_steps,
            trajectory_consensus_seed_sensitivity_steps=trajectory_consensus_seed_sensitivity_steps,
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
            trajectory_consensus_seed_influence_seeds=trajectory_consensus_seed_influence_seeds,
            trajectory_consensus_seed_sensitivity_zr_metric=trajectory_consensus_seed_sensitivity_zr_metric,
            skip_existing_plots=bool(trajectory_consensus_skip_existing_plots),
        )
    if mode == "anchor_frame":
        if not reuse_cross_attention_dir:
            raise ValueError("anchor_frame mode requires reuse_cross_attention_dir for spatial-map entropy.")
        return _run_anchor_frame_mode(
            output_dir=output_dir,
            frame_num=frame_num,
            reuse_cross_attention_dir=reuse_cross_attention_dir,
            target_object_words=target_object_words,
            target_verb_words=target_verb_words,
            trajectory_consensus_steps=trajectory_consensus_steps,
            trajectory_consensus_anchor_frame_steps=trajectory_consensus_anchor_frame_steps,
            trajectory_consensus_layers=trajectory_consensus_layers,
            trajectory_consensus_seed_influence_anchor_topk=trajectory_consensus_seed_influence_anchor_topk,
            trajectory_consensus_seed_influence_arrow_topk=trajectory_consensus_seed_influence_arrow_topk,
            num_workers=trajectory_consensus_num_workers,
            skip_existing_plots=bool(trajectory_consensus_skip_existing_plots),
        )
    raise ValueError(
        "trajectory_consensus_seed_influence_mode must be `seed_sensitivity` or `anchor_frame`."
    )
