"""Wan2.1-T2V experiment: seed_to_trajectory_predictability.

Main entry:
- run_wan21_t2v_seed_to_trajectory_predictability

This module tests whether a selected reference trajectory is predictable from
initial noise and/or early cross-attention trajectories. Shared runtime and
cross-attention collection helpers come from utils.py; probe-specific feature
construction and fitting stay local here.
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
    Wan21T2VCrossAttentionVizState,
    Wan21T2VParallelConfig,
    _build_wan21_t2v_pipeline,
    _dedup_wan21_t2v_int_list,
    _ensure_dir,
    _extract_wan21_t2v_attention_region_center_trajectory,
    _generate_wan21_t2v_video_with_initial_noise,
    _init_wan21_t2v_runtime,
    _install_wan21_t2v_cross_attention_viz_patch,
    _locate_wan21_t2v_prompt_words,
    _mean_wan21_t2v_head_maps_for_words,
    _resample_wan21_t2v_trajectory,
    _resolve_wan21_t2v_offload_model,
    _save_csv,
    _save_json,
    _save_wan21_t2v_video,
)

def _flatten_wan21_t2v_trajectory_xy(
    trajectory: Sequence[Tuple[float, float]],
) -> List[float]:
    """Flatten a `(y, x)` trajectory into `[y0, x0, y1, x1, ...]`."""
    out: List[float] = []
    for y, x in trajectory:
        out.append(float(y))
        out.append(float(x))
    return out

def _ridge_fit_wan21_t2v(
    x: torch.Tensor,
    y: torch.Tensor,
    ridge_alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fit a small closed-form ridge regressor with intercept.

    Args:
        x: Feature matrix `[N, D_in]`.
        y: Target matrix `[N, D_out]`.
        ridge_alpha: Non-negative L2 regularization coefficient. The intercept
            column is not regularized.

    Returns:
        `(weight, prediction)` where `weight` has shape `[D_in + 1, D_out]`
        and `prediction` has shape `[N, D_out]`.
    """
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError("ridge fit expects x=[N,D] and y=[N,K].")
    if x.size(0) != y.size(0):
        raise ValueError("ridge fit expects matching batch dimension.")
    ones = torch.ones((x.size(0), 1), dtype=x.dtype)
    design = torch.cat([x, ones], dim=1)
    gram = design.T @ design
    reg = torch.eye(gram.size(0), dtype=x.dtype) * float(max(0.0, ridge_alpha))
    reg[-1, -1] = 0.0
    weight = torch.linalg.solve(gram + reg, design.T @ y)
    pred = design @ weight
    return weight, pred

def _noise_features_wan21_t2v(
    noise_cfhw: torch.Tensor,
    max_flat_features: int = 4096,
) -> List[float]:
    """Build a compact deterministic feature vector from initial latent noise.

    Args:
        noise_cfhw: Initial Gaussian latent noise `[C, F, H, W]`.
        max_flat_features: Maximum number of uniformly sampled raw-noise entries.

    Returns:
        A list containing global statistics, per-channel statistics,
        per-latent-frame statistics, and a uniform subsample of raw values.
    """
    noise = noise_cfhw.detach().float().cpu()
    if noise.dim() != 4:
        raise ValueError(f"Expected noise shape [C,F,H,W], got {tuple(noise.shape)}")

    features: List[float] = []
    features.extend(
        [
            float(noise.mean().item()),
            float(noise.std(unbiased=False).item()),
            float(noise.abs().mean().item()),
            float(noise.square().mean().item()),
            float(noise.min().item()),
            float(noise.max().item()),
        ]
    )

    channel_mean = noise.mean(dim=(1, 2, 3))
    channel_std = noise.std(dim=(1, 2, 3), unbiased=False)
    frame_mean = noise.mean(dim=(0, 2, 3))
    frame_std = noise.std(dim=(0, 2, 3), unbiased=False)
    features.extend(float(x.item()) for x in channel_mean)
    features.extend(float(x.item()) for x in channel_std)
    features.extend(float(x.item()) for x in frame_mean)
    features.extend(float(x.item()) for x in frame_std)

    flat = noise.flatten()
    if flat.numel() > 0:
        take = min(int(max_flat_features), int(flat.numel()))
        if take > 0:
            indices = torch.linspace(0, flat.numel() - 1, steps=take).round().long()
            features.extend(float(x.item()) for x in flat.index_select(0, indices))
    return features

def _attention_trajectory_features_wan21_t2v(
    mean_maps: Dict[Tuple[int, int, str], torch.Tensor],
    object_words: Sequence[str],
    steps: Sequence[int],
    layer: int,
    head: str,
    num_points: int,
) -> List[float]:
    """Extract flattened trajectory features from saved object-token attention maps.

    Args:
        mean_maps: Cross-attention maps keyed by `(step, layer, token)` and
            shaped `[heads, F, H, W]`.
        object_words: Object words to aggregate.
        steps: 1-based diffusion steps to include.
        layer: DiT layer index.
        head: `mean` or an integer head id string.
        num_points: Number of trajectory points after interpolation. When
            `num_points <= 0`, keep the original latent-frame trajectory length.

    Returns:
        Flattened trajectory vector. Each requested step contributes one
        resampled trajectory when available.
    """
    features: List[float] = []
    for step in steps:
        object_head_maps = _mean_wan21_t2v_head_maps_for_words(
            mean_maps=mean_maps,
            step=int(step),
            layer=int(layer),
            words=object_words,
        )
        if object_head_maps is None:
            continue
        if str(head) == "mean":
            map_fhw = object_head_maps.mean(dim=0)
        else:
            head_index = int(head)
            if head_index < 0 or head_index >= int(object_head_maps.size(0)):
                continue
            map_fhw = object_head_maps[head_index]
        trajectory = _extract_wan21_t2v_attention_region_center_trajectory(map_fhw)
        resolved_num_points = int(num_points) if int(num_points) > 0 else len(trajectory)
        resampled = _resample_wan21_t2v_trajectory(trajectory, num_points=resolved_num_points)
        features.extend(_flatten_wan21_t2v_trajectory_xy(resampled))
    return features

def _target_trajectory_from_attention_maps_wan21_t2v(
    mean_maps: Dict[Tuple[int, int, str], torch.Tensor],
    object_words: Sequence[str],
    step: int,
    layer: int,
    head: str,
    num_points: int,
) -> List[Tuple[float, float]]:
    """Extract the reference trajectory from object-token attention maps."""
    object_head_maps = _mean_wan21_t2v_head_maps_for_words(
        mean_maps=mean_maps,
        step=int(step),
        layer=int(layer),
        words=object_words,
    )
    if object_head_maps is None:
        return []
    if str(head) == "mean":
        map_fhw = object_head_maps.mean(dim=0)
    else:
        head_index = int(head)
        if head_index < 0 or head_index >= int(object_head_maps.size(0)):
            return []
        map_fhw = object_head_maps[head_index]
    trajectory = _extract_wan21_t2v_attention_region_center_trajectory(map_fhw)
    resolved_num_points = int(num_points) if int(num_points) > 0 else len(trajectory)
    return _resample_wan21_t2v_trajectory(trajectory, num_points=resolved_num_points)

def _fit_predict_leave_one_out_wan21_t2v(
    variant_name: str,
    seed_to_feature: Dict[int, List[float]],
    seed_to_target: Dict[int, List[float]],
    ridge_alpha: float,
) -> Tuple[List[Dict[str, object]], Dict[int, List[Tuple[float, float]]]]:
    """Run leave-one-seed-out ridge prediction for one feature variant."""
    usable_seeds = sorted(seed for seed in seed_to_feature.keys() if seed in seed_to_target)
    rows: List[Dict[str, object]] = []
    prediction_by_seed: Dict[int, List[Tuple[float, float]]] = {}
    if len(usable_seeds) < 3:
        rows.append(
            {
                "variant": variant_name,
                "heldout_seed": -1,
                "num_train_seeds": max(0, len(usable_seeds) - 1),
                "feature_dim": len(seed_to_feature[usable_seeds[0]]) if usable_seeds else 0,
                "mean_point_error": "",
                "normalized_mean_point_error": "",
                "status": "need_at_least_3_usable_seeds",
            }
        )
        return rows, prediction_by_seed

    feature_lengths = {len(seed_to_feature[seed]) for seed in usable_seeds}
    target_lengths = {len(seed_to_target[seed]) for seed in usable_seeds}
    if len(feature_lengths) != 1 or len(target_lengths) != 1:
        rows.append(
            {
                "variant": variant_name,
                "heldout_seed": -1,
                "num_train_seeds": len(usable_seeds),
                "feature_dim": -1,
                "mean_point_error": "",
                "normalized_mean_point_error": "",
                "status": f"inconsistent_feature_or_target_dims feature={sorted(feature_lengths)} target={sorted(target_lengths)}",
            }
        )
        return rows, prediction_by_seed

    feature_matrix = torch.tensor([seed_to_feature[seed] for seed in usable_seeds], dtype=torch.float64)
    target_matrix = torch.tensor([seed_to_target[seed] for seed in usable_seeds], dtype=torch.float64)

    for held_seed in usable_seeds:
        held_index = usable_seeds.index(held_seed)
        train_indices = [idx for idx in range(len(usable_seeds)) if idx != held_index]
        x_train = feature_matrix[train_indices]
        y_train = target_matrix[train_indices]
        x_test = feature_matrix[held_index:held_index + 1]
        y_test = target_matrix[held_index:held_index + 1]

        weight, _ = _ridge_fit_wan21_t2v(x_train, y_train, ridge_alpha)
        test_design = torch.cat([x_test, torch.ones((1, 1), dtype=x_test.dtype)], dim=1)
        pred = (test_design @ weight).float().reshape(-1, 2)
        target = y_test.float().reshape(-1, 2)
        per_point_error = (pred - target).pow(2).sum(dim=-1).sqrt()
        mean_error = float(per_point_error.mean().item())
        target_step = target[1:] - target[:-1]
        target_mean_step = float(target_step.pow(2).sum(dim=-1).sqrt().mean().item()) if target.size(0) > 1 else 0.0
        normalized_error = float(mean_error / max(target_mean_step, 1e-6))
        prediction_by_seed[int(held_seed)] = [(float(p[0].item()), float(p[1].item())) for p in pred]
        rows.append(
            {
                "variant": variant_name,
                "heldout_seed": int(held_seed),
                "num_train_seeds": int(len(train_indices)),
                "feature_dim": int(feature_matrix.size(1)),
                "mean_point_error": mean_error,
                "normalized_mean_point_error": normalized_error,
                "status": "ok",
            }
        )
    return rows, prediction_by_seed

def run_wan21_t2v_seed_to_trajectory_predictability(
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
    seed_to_trajectory_early_steps: Sequence[int] = (1, 2, 3, 4, 5, 6),
    seed_to_trajectory_reference_step: int = 50,
    seed_to_trajectory_reference_layer: int = 27,
    seed_to_trajectory_head: str = "mean",
    seed_to_trajectory_num_points: int = 0,
    seed_to_trajectory_ridge_alpha: float = 1e-3,
    num_viz_frames: int = 5,
    layers_to_collect: Optional[Sequence[int]] = None,
    chunk_size: int = 1024,
    trajectory_style: str = "glow_arrow",
    trajectory_smooth_radius: int = 2,
    trajectory_power: float = 1.5,
    trajectory_quantile: float = 0.8,
    trajectory_arrow_stride: int = 4,
    parallel_cfg: Optional[Wan21T2VParallelConfig] = None,
):
    """Test whether reference object trajectory is predictable from noise and/or early attention.

    This implements four planned variants:
    1. `noise_only`: compact features from the initial Gaussian latent `z_T`.
    2. `step1_attention`: object-token attention trajectory at diffusion step 1.
    3. `steps1_to_k_attention`: object-token attention trajectories over the configured early steps.
    4. `noise_plus_attention`: concatenation of `z_T` features and early attention features.

    Unlike `cross_attention_token_viz`, this experiment does not reuse saved
    attention-map directories by default. It runs generation once per seed,
    captures `z_T`, captures early/reference cross-attention maps in the same run,
    then performs leave-one-seed-out ridge prediction of the reference trajectory.
    """
    del num_viz_frames, trajectory_style
    del trajectory_smooth_radius, trajectory_power, trajectory_quantile, trajectory_arrow_stride
    del seed

    object_words = [str(word).strip() for word in target_object_words if str(word).strip()]
    if not object_words:
        raise ValueError("seed_to_trajectory_predictability requires non-empty target_object_words.")
    if not seed_list:
        raise ValueError("seed_list is empty.")

    early_steps = _dedup_wan21_t2v_int_list(seed_to_trajectory_early_steps)
    if not early_steps:
        raise ValueError("seed_to_trajectory_early_steps must be non-empty.")
    reference_step = int(seed_to_trajectory_reference_step)
    # `step1_attention` is always one of the four built-in variants, so step 1
    # must be collected even if `seed_to_trajectory_early_steps` does not
    # explicitly include it.
    collect_steps = sorted(set([1] + early_steps + [reference_step]))
    reference_layer = int(seed_to_trajectory_reference_layer)
    layers_for_collection = [reference_layer] if layers_to_collect is None else _dedup_wan21_t2v_int_list(layers_to_collect)
    if reference_layer not in set(layers_for_collection):
        layers_for_collection.append(reference_layer)

    parallel_cfg = parallel_cfg or Wan21T2VParallelConfig()
    runtime = _init_wan21_t2v_runtime(parallel_cfg, explicit_device_id=device_id)
    if runtime.use_usp:
        raise RuntimeError("seed_to_trajectory_predictability currently requires use_usp=False.")

    pipeline, cfg = _build_wan21_t2v_pipeline(
        wan21_root=wan21_root,
        ckpt_dir=ckpt_dir,
        task=task,
        runtime=runtime,
        parallel_cfg=parallel_cfg,
    )
    offload_model = _resolve_wan21_t2v_offload_model(runtime, offload_model)
    word_to_positions, word_to_type, prompt_tokens = _locate_wan21_t2v_prompt_words(
        text_encoder=pipeline.text_encoder,
        prompt=prompt,
        target_object_words=object_words,
        target_verb_words=target_verb_words,
    )
    object_words_in_prompt = [word for word in object_words if word in word_to_positions]
    if not object_words_in_prompt:
        raise ValueError("No target_object_words found in prompt tokenization.")

    if runtime.rank == 0:
        _ensure_dir(output_dir)

    seed_run_rows: List[Dict[str, object]] = []
    target_rows: List[Dict[str, object]] = []
    feature_rows: List[Dict[str, object]] = []
    seed_to_target_vector: Dict[int, List[float]] = {}
    seed_to_target_trajectory: Dict[int, List[Tuple[float, float]]] = {}
    variant_to_seed_features: Dict[str, Dict[int, List[float]]] = {
        "noise_only": {},
        "step1_attention": {},
        "steps1_to_k_attention": {},
        "noise_plus_attention": {},
    }

    for run_seed in seed_list:
        run_seed = int(run_seed)
        state = Wan21T2VCrossAttentionVizState(
            token_positions=word_to_positions,
            collect_steps=collect_steps,
            num_layers=len(pipeline.model.blocks),
            num_heads=pipeline.model.num_heads,
            chunk_size=chunk_size,
            layers_to_collect=layers_for_collection,
        )
        handle = _install_wan21_t2v_cross_attention_viz_patch(pipeline.model, state)
        try:
            video, initial_noise = _generate_wan21_t2v_video_with_initial_noise(
                pipeline=pipeline,
                prompt=prompt,
                size=size,
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                seed=run_seed,
                offload_model=offload_model,
            )
        finally:
            handle.restore()

        if dist.is_initialized():
            dist.barrier()
        if runtime.rank != 0:
            continue

        seed_dir = os.path.join(output_dir, f"seed_{run_seed:06d}")
        _ensure_dir(seed_dir)
        video_path = os.path.join(seed_dir, f"wan21_t2v_seed_to_trajectory_seed_{run_seed:06d}.mp4")
        if video is not None:
            _save_wan21_t2v_video(video, video_path, fps=cfg.sample_fps)

        mean_maps = state.export_mean_maps()
        maps_path = os.path.join(seed_dir, "seed_to_trajectory_cross_attention_maps.pt")
        noise_path = os.path.join(seed_dir, "initial_noise.pt")
        torch.save(mean_maps, maps_path)
        torch.save(initial_noise, noise_path)

        target_trajectory = _target_trajectory_from_attention_maps_wan21_t2v(
            mean_maps=mean_maps,
            object_words=object_words_in_prompt,
            step=reference_step,
            layer=reference_layer,
            head=seed_to_trajectory_head,
            num_points=seed_to_trajectory_num_points,
        )
        if not target_trajectory:
            seed_run_rows.append(
                {
                    "seed": run_seed,
                    "seed_output_dir": seed_dir,
                    "maps_path": maps_path,
                    "noise_path": noise_path,
                    "video_path": video_path if video is not None else "",
                    "status": "missing_final_target_trajectory",
                }
            )
            continue

        noise_feature = _noise_features_wan21_t2v(initial_noise)
        step1_feature = _attention_trajectory_features_wan21_t2v(
            mean_maps=mean_maps,
            object_words=object_words_in_prompt,
            steps=[1],
            layer=reference_layer,
            head=seed_to_trajectory_head,
            num_points=seed_to_trajectory_num_points,
        )
        early_attention_feature = _attention_trajectory_features_wan21_t2v(
            mean_maps=mean_maps,
            object_words=object_words_in_prompt,
            steps=early_steps,
            layer=reference_layer,
            head=seed_to_trajectory_head,
            num_points=seed_to_trajectory_num_points,
        )

        target_vector = _flatten_wan21_t2v_trajectory_xy(target_trajectory)
        seed_to_target_vector[run_seed] = target_vector
        seed_to_target_trajectory[run_seed] = target_trajectory
        variant_to_seed_features["noise_only"][run_seed] = noise_feature
        if step1_feature:
            variant_to_seed_features["step1_attention"][run_seed] = step1_feature
        if early_attention_feature:
            variant_to_seed_features["steps1_to_k_attention"][run_seed] = early_attention_feature
            variant_to_seed_features["noise_plus_attention"][run_seed] = noise_feature + early_attention_feature

        seed_run_rows.append(
            {
                "seed": run_seed,
                "seed_output_dir": seed_dir,
                "maps_path": maps_path,
                "noise_path": noise_path,
                "video_path": video_path if video is not None else "",
                "status": "ok",
            }
        )
        target_rows.append(
            {
                "seed": run_seed,
                "target_dim": len(target_vector),
                "reference_step": reference_step,
                "reference_layer": reference_layer,
                "head": str(seed_to_trajectory_head),
            }
        )
        for variant_name, feature_by_seed in variant_to_seed_features.items():
            if run_seed in feature_by_seed:
                feature_rows.append(
                    {
                        "seed": run_seed,
                        "variant": variant_name,
                        "feature_dim": len(feature_by_seed[run_seed]),
                    }
                )

    if dist.is_initialized():
        dist.barrier()
    if runtime.rank != 0:
        return None

    _save_csv(os.path.join(output_dir, "seed_to_trajectory_seed_runs.csv"), seed_run_rows)
    _save_csv(os.path.join(output_dir, "seed_to_trajectory_features.csv"), feature_rows)
    _save_csv(os.path.join(output_dir, "seed_to_trajectory_targets.csv"), target_rows)

    prediction_rows: List[Dict[str, object]] = []
    variant_prediction_by_seed: Dict[str, Dict[int, List[Tuple[float, float]]]] = {}
    for variant_name, feature_by_seed in variant_to_seed_features.items():
        rows, prediction_by_seed = _fit_predict_leave_one_out_wan21_t2v(
            variant_name=variant_name,
            seed_to_feature=feature_by_seed,
            seed_to_target=seed_to_target_vector,
            ridge_alpha=seed_to_trajectory_ridge_alpha,
        )
        prediction_rows.extend(rows)
        variant_prediction_by_seed[variant_name] = prediction_by_seed
    _save_csv(os.path.join(output_dir, "seed_to_trajectory_predictions.csv"), prediction_rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = os.path.join(output_dir, "seed_to_trajectory_plots")
    _ensure_dir(plots_dir)
    plot_paths: List[str] = []
    for variant_name, prediction_by_seed in sorted(variant_prediction_by_seed.items()):
        variant_dir = os.path.join(plots_dir, variant_name)
        _ensure_dir(variant_dir)
        for held_seed, predicted_traj in sorted(prediction_by_seed.items()):
            target_traj = seed_to_target_trajectory.get(int(held_seed), [])
            if not target_traj:
                continue
            fig, ax = plt.subplots(1, 1, figsize=(6.6, 5.4))
            ax.plot(
                [p[1] for p in target_traj],
                [p[0] for p in target_traj],
                color="#111827",
                linewidth=2.0,
                label="target reference trajectory",
            )
            ax.plot(
                [p[1] for p in predicted_traj],
                [p[0] for p in predicted_traj],
                color="#ef4444",
                linewidth=2.0,
                linestyle="--",
                label=f"predicted: {variant_name}",
            )
            ax.set_title(f"{variant_name} | heldout seed={int(held_seed)}")
            ax.set_xlabel("token-x")
            ax.set_ylabel("token-y")
            ax.invert_yaxis()
            ax.grid(alpha=0.2, linestyle="--")
            ax.legend(fontsize=8)
            fig.tight_layout()
            plot_path = os.path.join(variant_dir, f"heldout_seed_{int(held_seed):06d}.pdf")
            fig.savefig(plot_path, format="pdf")
            plt.close(fig)
            plot_paths.append(plot_path)

    summary = {
        "experiment": "wan21_t2v_seed_to_trajectory_predictability",
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "token_positions": word_to_positions,
        "token_types": word_to_type,
        "target_object_words": list(target_object_words),
        "target_verb_words": list(target_verb_words),
        "seed_list": [int(seed_value) for seed_value in seed_list],
        "usable_seeds": sorted(int(seed_value) for seed_value in seed_to_target_vector.keys()),
        "early_steps": [int(step) for step in early_steps],
        "collected_steps": [int(step) for step in collect_steps],
        "reference_step": int(reference_step),
        "reference_layer": int(reference_layer),
        "head": str(seed_to_trajectory_head),
        "num_points": int(seed_to_trajectory_num_points),
        "use_original_trajectory_length": bool(int(seed_to_trajectory_num_points) <= 0),
        "ridge_alpha": float(seed_to_trajectory_ridge_alpha),
        "variants": sorted(variant_to_seed_features.keys()),
        "predictions_csv": os.path.join(output_dir, "seed_to_trajectory_predictions.csv"),
        "plot_paths": plot_paths,
    }
    _save_json(os.path.join(output_dir, "seed_to_trajectory_predictability_summary.json"), summary)
    return summary
