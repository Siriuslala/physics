"""Wan2.1-T2V experiment: trajectory_entropy.

Main entry:
- run_wan21_t2v_trajectory_entropy

This module computes step-wise, layer-wise, and head-wise cross-attention
trajectory entropy. Shared runtime, cross-attention collection, and map helpers
come from utils.py; entropy-specific row builders and plots stay local here.
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
    _compute_wan21_t2v_spatial_entropy_stats,
    _dedup_wan21_t2v_int_list,
    _ensure_dir,
    _generate_wan21_t2v_video,
    _init_wan21_t2v_runtime,
    _install_wan21_t2v_cross_attention_viz_patch,
    _load_wan21_t2v_cross_attention_mean_maps_from_disk,
    _load_wan21_t2v_cross_attention_token_meta,
    _locate_wan21_t2v_prompt_words,
    _mean_wan21_t2v_headmean_map_for_words,
    _mean_wan21_t2v_head_maps_for_words,
    _resolve_wan21_t2v_offload_model,
    _save_csv,
    _save_json,
    _save_wan21_t2v_video,
    _unwrap_wan21_t2v_dit_model_for_runtime_patch,
    Wan21T2VCrossAttentionVizState,
)

def _resolve_wan21_t2v_trajectory_entropy_steps(
    sampling_steps: int,
    steps: Sequence[int],
) -> List[int]:
    """Resolve trajectory-entropy probe steps.

    If `steps` is empty, this defaults to all diffusion steps: [1..sampling_steps].
    """
    if steps:
        resolved = _dedup_wan21_t2v_int_list(steps)
    else:
        resolved = list(range(1, int(sampling_steps) + 1))
    if not resolved:
        raise ValueError("Resolved trajectory entropy steps are empty.")
    if any(s < 1 or s > int(sampling_steps) for s in resolved):
        raise ValueError(
            f"trajectory entropy steps must be in [1, {sampling_steps}], got: {resolved}"
        )
    return resolved

def _resolve_wan21_t2v_trajectory_entropy_layerwise_steps(
    sampling_steps: int,
    entropy_steps: Sequence[int],
    layerwise_steps: Sequence[int],
) -> List[int]:
    """Resolve layer-wise entropy steps.

    If `layerwise_steps` is empty, default to all trajectory-entropy steps.
    """
    if layerwise_steps:
        resolved = _dedup_wan21_t2v_int_list(layerwise_steps)
    else:
        if not entropy_steps:
            raise ValueError("entropy_steps must be non-empty when layerwise_steps is empty.")
        resolved = _dedup_wan21_t2v_int_list(entropy_steps)
    if any(s < 1 or s > int(sampling_steps) for s in resolved):
        raise ValueError(
            f"trajectory entropy layer-wise steps must be in [1, {sampling_steps}], got: {resolved}"
        )
    return resolved

def _collect_wan21_t2v_cross_attention_maps_once(
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
    token_positions: Dict[str, List[int]],
    collect_steps: Sequence[int],
    layers_to_collect: Optional[Sequence[int]],
    stream_flush_per_step: bool = False,
    on_step_maps=None,
):
    """Run one generation pass and collect mean cross-attention maps.

    Returns:
        video: generated video tensor.
        mean_maps: Dict[(step, layer, token)] -> Tensor[num_heads, F, H, W]
    """
    dit_target = _unwrap_wan21_t2v_dit_model_for_runtime_patch(pipeline.model)
    if not hasattr(dit_target, "blocks"):
        raise RuntimeError("Invalid DiT model for cross-attention map collection.")
    num_layers = len(dit_target.blocks)
    num_heads = int(getattr(dit_target, "num_heads", 0))
    if num_heads <= 0:
        raise RuntimeError("Invalid DiT model: missing num_heads.")

    state = Wan21T2VCrossAttentionVizState(
        token_positions=token_positions,
        collect_steps=collect_steps,
        num_layers=num_layers,
        num_heads=num_heads,
        chunk_size=1024,
        layers_to_collect=layers_to_collect,
    )
    def _on_step_complete(step_id: int):
        if not stream_flush_per_step:
            return
        step_maps = state.export_step_mean_maps(int(step_id), clear_after_export=True)
        if on_step_maps is not None and step_maps:
            on_step_maps(int(step_id), step_maps)

    handle = _install_wan21_t2v_cross_attention_viz_patch(
        pipeline.model,
        state,
        on_step_complete=_on_step_complete if stream_flush_per_step else None,
    )
    try:
        video = _generate_wan21_t2v_video(
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
    finally:
        handle.restore()
    if stream_flush_per_step:
        final_step_maps = state.export_step_mean_maps(int(state.current_step), clear_after_export=True)
        if on_step_maps is not None and final_step_maps:
            on_step_maps(int(state.current_step), final_step_maps)
        return video, {}
    return video, state.export_mean_maps()

def _build_wan21_t2v_entropy_row(
    step: int,
    layer: int,
    head,
    token: str,
    token_type: str,
    source: str,
    map_fhw: torch.Tensor,
    eps: float = 1e-12,
) -> Dict[str, object]:
    stats = _compute_wan21_t2v_spatial_entropy_stats(map_fhw, eps=eps)
    row = {
        "step": int(step),
        "layer": int(layer),
        "head": str(head),
        "token": str(token),
        "token_type": str(token_type),
        "source": str(source),
    }
    row.update(stats)
    return row

def _mean_wan21_t2v_entropy_stats(stats_list: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Average same-schema entropy stats (typically across heads)."""
    if not stats_list:
        raise ValueError("stats_list is empty.")

    keep_int_keys = {"frame_count", "token_grid_h", "token_grid_w", "spatial_size"}
    out: Dict[str, float] = {}
    for key in stats_list[0].keys():
        vals = [s[key] for s in stats_list if key in s]
        if not vals:
            continue
        if key in keep_int_keys:
            out[key] = int(vals[0])
        else:
            out[key] = float(sum(float(v) for v in vals) / float(len(vals)))
    return out

def _build_wan21_t2v_entropy_row_from_stats(
    step: int,
    layer: int,
    head,
    token: str,
    token_type: str,
    source: str,
    stats: Dict[str, float],
) -> Dict[str, object]:
    row = {
        "step": int(step),
        "layer": int(layer),
        "head": str(head),
        "token": str(token),
        "token_type": str(token_type),
        "source": str(source),
    }
    row.update(stats)
    return row

def _build_wan21_t2v_mean_over_heads_entropy_row(
    step: int,
    layer: int,
    token: str,
    token_type: str,
    source: str,
    map_hfhw: torch.Tensor,
    eps: float = 1e-12,
) -> Dict[str, object]:
    """Build one row by computing entropy per head first, then averaging over heads."""
    if map_hfhw.dim() != 4:
        raise ValueError(f"Expected [num_heads, F, H, W], got shape={tuple(map_hfhw.shape)}")
    stats_per_head = [
        _compute_wan21_t2v_spatial_entropy_stats(map_hfhw[h], eps=eps)
        for h in range(int(map_hfhw.size(0)))
    ]
    mean_stats = _mean_wan21_t2v_entropy_stats(stats_per_head)
    return _build_wan21_t2v_entropy_row_from_stats(
        step=step,
        layer=layer,
        head="mean",
        token=token,
        token_type=token_type,
        source=source,
        stats=mean_stats,
    )

def _plot_wan21_t2v_trajectory_entropy_stepwise(
    rows: Sequence[Dict[str, object]],
    save_file: str,
    metric_key: str = "entropy_norm_mean",
    ylabel: str = "normalized entropy",
    title_suffix: str = "Frame-Wise",
    stepwise_layer: Optional[int] = None,
):
    """Plot step-wise trajectory entropy on object-mean rows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obj_rows = [r for r in rows if str(r.get("token", "")) == "__object_mean__"]
    if not obj_rows:
        return ""
    obj_rows = [r for r in obj_rows if metric_key in r]
    if not obj_rows:
        return ""
    obj_rows = sorted(obj_rows, key=lambda r: int(r["step"]))
    xs = [int(r["step"]) for r in obj_rows]
    ys = [float(r[metric_key]) for r in obj_rows]

    fig, ax = plt.subplots(1, 1, figsize=(7.8, 4.8))
    ax.plot(xs, ys, marker="o", linewidth=1.8, color="#1f77b4")
    layer_tag = f"layer={int(stepwise_layer)}" if stepwise_layer is not None else "selected layer"
    ax.set_title(f"Trajectory Entropy vs Diffusion Step ({layer_tag}, Mean Over Heads, {title_suffix})")
    ax.set_xlabel("diffusion step")
    ax.set_ylabel(ylabel)
    y_min = min(ys)
    y_max = max(ys)
    if y_max - y_min < 1e-8:
        pad = 0.05
    else:
        pad = max(0.015, 0.08 * (y_max - y_min))
    lo = max(0.0, y_min - pad)
    hi = min(1.02, y_max + pad)
    if hi - lo < 0.06:
        c = 0.5 * (hi + lo)
        lo = max(0.0, c - 0.03)
        hi = min(1.02, c + 0.03)
    ax.set_ylim(lo, hi)
    ax.grid(alpha=0.22, linestyle="--")
    fig.tight_layout()
    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file

def _plot_wan21_t2v_trajectory_entropy_layerwise(
    rows: Sequence[Dict[str, object]],
    save_file: str,
):
    """Plot layer-wise normalized trajectory entropy (line per selected step)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obj_rows = [r for r in rows if str(r.get("token", "")) == "__object_mean__"]
    if not obj_rows:
        return ""
    by_step = defaultdict(list)
    for row in obj_rows:
        by_step[int(row["step"])].append(row)
    if not by_step:
        return ""

    sorted_steps = sorted(by_step.keys())
    num_steps = len(sorted_steps)

    fig_w = 10.2 if num_steps > 14 else 8.2
    fig_h = 5.4 if num_steps > 24 else 5.0
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    cmap = plt.get_cmap("turbo")
    label_steps = set(sorted_steps)
    max_legend_entries = 14
    if num_steps > max_legend_entries:
        stride = max(1, int(math.ceil(num_steps / float(max_legend_entries))))
        label_steps = set(sorted_steps[::stride])
        label_steps.add(sorted_steps[-1])

    all_ys: List[float] = []
    denom = max(1, num_steps - 1)
    for idx, step in enumerate(sorted_steps):
        layer_rows = sorted(by_step[step], key=lambda r: int(r["layer"]))
        xs = [int(r["layer"]) for r in layer_rows]
        ys = [float(r["entropy_norm_mean"]) for r in layer_rows]
        all_ys.extend(ys)
        label = f"step={step}" if step in label_steps else None
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=1.2 if num_steps > 20 else 1.5,
            markersize=2.6 if num_steps > 20 else 3.2,
            color=cmap(float(idx) / float(denom)),
            alpha=0.9,
            label=label,
        )
    ax.set_title("Layer-wise Trajectory Entropy (Mean Over Heads)")
    ax.set_xlabel("layer")
    ax.set_ylabel("normalized entropy")
    if all_ys:
        y_min = min(all_ys)
        y_max = max(all_ys)
        if y_max - y_min < 1e-8:
            pad = 0.05
        else:
            pad = max(0.02, 0.08 * (y_max - y_min))
        lo = max(0.0, y_min - pad)
        hi = min(1.02, y_max + pad)
        if hi - lo < 0.08:
            c = 0.5 * (hi + lo)
            lo = max(0.0, c - 0.04)
            hi = min(1.02, c + 0.04)
        ax.set_ylim(lo, hi)
    ax.grid(alpha=0.22, linestyle="--")
    handles, labels = ax.get_legend_handles_labels()
    legend_outside = False
    if handles:
        if num_steps > 10:
            ax.legend(
                handles,
                labels,
                fontsize=7,
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                frameon=False,
                title="steps",
                title_fontsize=8,
            )
            legend_outside = True
        else:
            ax.legend(fontsize=8, ncol=2)
    if legend_outside:
        fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    else:
        fig.tight_layout()
    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file

def _plot_wan21_t2v_trajectory_entropy_headwise(
    rows: Sequence[Dict[str, object]],
    save_file: str,
    head_layer: int,
):
    """Plot head-wise normalized entropy curves across diffusion steps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obj_rows = [r for r in rows if str(r.get("token", "")) == "__object_mean__"]
    if not obj_rows:
        return ""
    by_head = defaultdict(list)
    for row in obj_rows:
        by_head[int(row["head"])].append(row)
    if not by_head:
        return ""

    fig, ax = plt.subplots(1, 1, figsize=(8.2, 5.0))
    all_ys: List[float] = []
    for head in sorted(by_head.keys()):
        head_rows = sorted(by_head[head], key=lambda r: int(r["step"]))
        xs = [int(r["step"]) for r in head_rows]
        ys = [float(r["entropy_norm_mean"]) for r in head_rows]
        all_ys.extend(ys)
        ax.plot(xs, ys, linewidth=1.2, alpha=0.85, label=f"h{head:02d}")
    ax.set_title(f"Head-wise Entropy vs Diffusion Step (layer={head_layer})")
    ax.set_xlabel("diffusion step")
    ax.set_ylabel("normalized entropy")
    if all_ys:
        y_min = min(all_ys)
        y_max = max(all_ys)
        if y_max - y_min < 1e-8:
            pad = 0.05
        else:
            pad = max(0.02, 0.08 * (y_max - y_min))
        lo = max(0.0, y_min - pad)
        hi = min(1.02, y_max + pad)
        if hi - lo < 0.08:
            c = 0.5 * (hi + lo)
            lo = max(0.0, c - 0.04)
            hi = min(1.02, c + 0.04)
        ax.set_ylim(lo, hi)
    ax.grid(alpha=0.22, linestyle="--")
    if len(by_head) <= 20:
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file

def run_wan21_t2v_trajectory_entropy(
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
    trajectory_entropy_steps: Sequence[int] = tuple(),
    trajectory_entropy_layerwise_steps: Sequence[int] = tuple(),
    trajectory_entropy_head_layer: int = -1,
    trajectory_entropy_stepwise_layer: int = -1,
    save_video: bool = True,
    entropy_eps: float = 1e-12,
    reuse_cross_attention_dir: Optional[str] = None,
    parallel_cfg: Optional[Wan21T2VParallelConfig] = None,
):
    """Run trajectory-entropy analysis on cross-attention maps.

    This experiment exports:
    1) step-wise entropy curve (selected layer, mean over per-head entropy),
    2) layer-wise entropy curve (selected steps, mean over per-head entropy),
    3) head-wise entropy curve (selected layer, each head across steps).
    """
    object_words = [str(w).strip() for w in target_object_words if str(w).strip()]
    if not object_words:
        raise ValueError("trajectory_entropy requires non-empty target_object_words.")
    object_words = list(dict.fromkeys(object_words))
    verb_words = [str(w).strip() for w in target_verb_words if str(w).strip()]
    verb_words = list(dict.fromkeys(verb_words))

    parallel_cfg = parallel_cfg or Wan21T2VParallelConfig()
    runtime = _init_wan21_t2v_runtime(parallel_cfg, explicit_device_id=device_id)
    seed = _broadcast_seed_if_needed(seed, runtime)

    cross_attention_reused = bool(reuse_cross_attention_dir and str(reuse_cross_attention_dir).strip())
    cross_attention_dir = (
        os.path.abspath(str(reuse_cross_attention_dir))
        if cross_attention_reused
        else ""
    )
    loaded_maps_source = ""
    mean_maps_from_disk: Dict[Tuple[int, int, str], torch.Tensor] = {}
    pipeline = None
    cfg = None

    if cross_attention_reused:
        raw_maps, loaded_maps_source = _load_wan21_t2v_cross_attention_mean_maps_from_disk(
            output_dir=cross_attention_dir,
            draw_attention_maps_path="",
        )
        if not raw_maps:
            raise ValueError(
                f"No valid cross-attention maps found under reuse_cross_attention_dir={cross_attention_dir}."
            )

        for key, value in raw_maps.items():
            if not isinstance(key, (tuple, list)) or len(key) != 3:
                continue
            step = int(key[0])
            layer = int(key[1])
            word = str(key[2])
            mean_maps_from_disk[(step, layer, word)] = value
        if not mean_maps_from_disk:
            raise ValueError(
                f"Loaded maps exist but no valid (step, layer, token) keys found in: {loaded_maps_source}"
            )

        available_steps = sorted({int(k[0]) for k in mean_maps_from_disk.keys()})
        if trajectory_entropy_steps:
            entropy_steps = _dedup_wan21_t2v_int_list(trajectory_entropy_steps)
        else:
            entropy_steps = list(available_steps)
        if not entropy_steps:
            raise ValueError("No trajectory entropy steps resolved from reused cross-attention maps.")
        missing_entropy_steps = [s for s in entropy_steps if s not in set(available_steps)]
        if missing_entropy_steps:
            raise ValueError(
                "Some trajectory_entropy_steps are not present in reused maps: "
                f"{missing_entropy_steps}; available={available_steps}"
            )

        if trajectory_entropy_layerwise_steps:
            layerwise_steps = _dedup_wan21_t2v_int_list(trajectory_entropy_layerwise_steps)
        else:
            layerwise_steps = list(entropy_steps)
        missing_layerwise_steps = [s for s in layerwise_steps if s not in set(available_steps)]
        if missing_layerwise_steps:
            raise ValueError(
                "Some trajectory_entropy_layerwise_steps are not present in reused maps: "
                f"{missing_layerwise_steps}; available={available_steps}"
            )

        words_in_maps = sorted(set(str(k[2]) for k in mean_maps_from_disk.keys()))
        word_to_positions, word_to_type, prompt_tokens = _load_wan21_t2v_cross_attention_token_meta(
            output_dir=cross_attention_dir,
            words_in_maps=words_in_maps,
            target_object_words=object_words,
            target_verb_words=verb_words,
        )
        object_words_in_prompt = [w for w in object_words if w in set(words_in_maps)]
        if not object_words_in_prompt:
            raise ValueError(
                "None of target_object_words found in reused cross-attention maps. "
                f"target_object_words={object_words}, words_in_maps={words_in_maps[:50]}"
            )

        num_layers = max(int(k[1]) for k in mean_maps_from_disk.keys()) + 1
    else:
        entropy_steps = _resolve_wan21_t2v_trajectory_entropy_steps(
            sampling_steps=int(sampling_steps),
            steps=trajectory_entropy_steps,
        )
        layerwise_steps = _resolve_wan21_t2v_trajectory_entropy_layerwise_steps(
            sampling_steps=int(sampling_steps),
            entropy_steps=entropy_steps,
            layerwise_steps=trajectory_entropy_layerwise_steps,
        )

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
            target_verb_words=verb_words,
        )
        object_words_in_prompt = [w for w in object_words if w in word_to_positions]
        if not object_words_in_prompt:
            raise ValueError("No target_object_words found in prompt tokenization.")

        dit_target = _unwrap_wan21_t2v_dit_model_for_runtime_patch(pipeline.model)
        num_layers = len(dit_target.blocks)
        if num_layers <= 0:
            raise RuntimeError("Invalid DiT model: no blocks found.")

    last_layer_idx = num_layers - 1
    stepwise_layer_raw = int(trajectory_entropy_stepwise_layer)
    if stepwise_layer_raw == -1:
        stepwise_layer_idx = last_layer_idx
    elif stepwise_layer_raw >= 0:
        stepwise_layer_idx = stepwise_layer_raw
    else:
        raise ValueError(
            "trajectory_entropy_stepwise_layer must be -1 or a non-negative layer index, "
            f"got: {stepwise_layer_raw}"
        )
    if stepwise_layer_idx >= num_layers:
        raise ValueError(
            f"trajectory_entropy_stepwise_layer={stepwise_layer_raw} resolved to invalid layer "
            f"{stepwise_layer_idx} (num_layers={num_layers})."
        )

    head_layer_raw = int(trajectory_entropy_head_layer)
    if head_layer_raw == -2:
        head_layer_indices = list(range(num_layers))
    elif head_layer_raw < 0:
        head_layer_indices = [last_layer_idx]
    else:
        head_layer_indices = [head_layer_raw]
    if any(idx < 0 or idx >= num_layers for idx in head_layer_indices):
        raise ValueError(
            f"trajectory_entropy_head_layer={head_layer_raw} resolved to invalid layers "
            f"{head_layer_indices} (num_layers={num_layers})."
        )

    stepwise_rows: List[Dict[str, object]] = []
    layerwise_rows: List[Dict[str, object]] = []
    headwise_rows: List[Dict[str, object]] = []
    object_word_set = set(object_words_in_prompt)

    def _accumulate_step_head_rows(step_id: int, step_maps: Dict[Tuple[int, int, str], torch.Tensor]):
        for word in object_words_in_prompt:
            key = (int(step_id), int(stepwise_layer_idx), str(word))
            maps = step_maps.get(key)
            if maps is None:
                continue
            stepwise_rows.append(
                _build_wan21_t2v_mean_over_heads_entropy_row(
                    step=step_id,
                    layer=stepwise_layer_idx,
                    token=word,
                    token_type=word_to_type.get(word, "object"),
                    source="stepwise_selected_layer_mean_head_entropy",
                    map_hfhw=maps.float(),
                    eps=entropy_eps,
                )
            )

        obj_head_maps = _mean_wan21_t2v_head_maps_for_words(
            mean_maps=step_maps,
            step=step_id,
            layer=stepwise_layer_idx,
            words=object_words_in_prompt,
        )
        if obj_head_maps is not None:
            stepwise_rows.append(
                _build_wan21_t2v_mean_over_heads_entropy_row(
                    step=step_id,
                    layer=stepwise_layer_idx,
                    token="__object_mean__",
                    token_type="object",
                    source="stepwise_selected_layer_mean_head_entropy",
                    map_hfhw=obj_head_maps,
                    eps=entropy_eps,
                )
            )

        for head_layer_idx in head_layer_indices:
            obj_head_maps = _mean_wan21_t2v_head_maps_for_words(
                mean_maps=step_maps,
                step=step_id,
                layer=head_layer_idx,
                words=object_words_in_prompt,
            )
            if obj_head_maps is None:
                continue
            for head_idx in range(int(obj_head_maps.size(0))):
                headwise_rows.append(
                    _build_wan21_t2v_entropy_row(
                        step=step_id,
                        layer=head_layer_idx,
                        head=head_idx,
                        token="__object_mean__",
                        token_type="object",
                        source="headwise_selected_layer",
                        map_fhw=obj_head_maps[head_idx],
                        eps=entropy_eps,
                    )
                )

    def _accumulate_layerwise_rows(step_id: int, step_maps: Dict[Tuple[int, int, str], torch.Tensor]):
        layers = sorted(
            {
                int(layer)
                for (s, layer, word) in step_maps.keys()
                if int(s) == int(step_id) and str(word) in object_word_set
            }
        )
        for layer in layers:
            for word in object_words_in_prompt:
                key = (int(step_id), int(layer), str(word))
                maps = step_maps.get(key)
                if maps is None:
                    continue
                layerwise_rows.append(
                    _build_wan21_t2v_mean_over_heads_entropy_row(
                        step=step_id,
                        layer=layer,
                        token=word,
                        token_type=word_to_type.get(word, "object"),
                        source="layerwise_mean_head_entropy",
                        map_hfhw=maps.float(),
                        eps=entropy_eps,
                    )
                )

            obj_head_maps = _mean_wan21_t2v_head_maps_for_words(
                mean_maps=step_maps,
                step=step_id,
                layer=layer,
                words=object_words_in_prompt,
            )
            if obj_head_maps is not None:
                layerwise_rows.append(
                    _build_wan21_t2v_mean_over_heads_entropy_row(
                        step=step_id,
                        layer=layer,
                        token="__object_mean__",
                        token_type="object",
                        source="layerwise_mean_head_entropy",
                        map_hfhw=obj_head_maps,
                        eps=entropy_eps,
                    )
                )

    step_head_layers = sorted(set([stepwise_layer_idx] + [int(x) for x in head_layer_indices]))
    video = None

    if cross_attention_reused:
        for step_id in entropy_steps:
            step_maps = {
                key: value
                for key, value in mean_maps_from_disk.items()
                if int(key[0]) == int(step_id)
            }
            if not step_maps:
                continue
            _accumulate_step_head_rows(int(step_id), step_maps)

        for step_id in layerwise_steps:
            step_maps = {
                key: value
                for key, value in mean_maps_from_disk.items()
                if int(key[0]) == int(step_id)
            }
            if not step_maps:
                continue
            _accumulate_layerwise_rows(int(step_id), step_maps)
    else:
        same_step_set = set(int(s) for s in entropy_steps) == set(int(s) for s in layerwise_steps)

        if same_step_set:
            # Default/typical path: one pass collects full-layer maps once, then
            # derives step-wise, layer-wise, and head-wise entropy together.
            def _accumulate_all(step_id: int, step_maps: Dict[Tuple[int, int, str], torch.Tensor]):
                _accumulate_step_head_rows(step_id, step_maps)
                _accumulate_layerwise_rows(step_id, step_maps)

            video, _ = _collect_wan21_t2v_cross_attention_maps_once(
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
                token_positions=word_to_positions,
                collect_steps=entropy_steps,
                layers_to_collect=None,
                stream_flush_per_step=True,
                on_step_maps=_accumulate_all,
            )
        else:
            # Optimized two-pass path when layer-wise is requested on a subset of steps.
            # Pass-1: only layers needed for step/head-wise across entropy_steps.
            video, _ = _collect_wan21_t2v_cross_attention_maps_once(
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
                token_positions=word_to_positions,
                collect_steps=entropy_steps,
                layers_to_collect=step_head_layers,
                stream_flush_per_step=True,
                on_step_maps=_accumulate_step_head_rows,
            )

            # Pass-2: all layers on selected layer-wise steps only.
            _, _ = _collect_wan21_t2v_cross_attention_maps_once(
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
                token_positions=word_to_positions,
                collect_steps=layerwise_steps,
                layers_to_collect=None,
                stream_flush_per_step=True,
                on_step_maps=_accumulate_layerwise_rows,
            )

    if dist.is_initialized():
        dist.barrier()

    if runtime.rank == 0:
        _ensure_dir(output_dir)

        video_path = ""
        if save_video and (not cross_attention_reused) and (video is not None):
            video_path = os.path.join(output_dir, f"wan21_t2v_trajectory_entropy_seed_{seed}.mp4")
            _save_wan21_t2v_video(video, video_path, fps=cfg.sample_fps)

        # Keep row ordering stable for downstream analysis.
        stepwise_rows = sorted(stepwise_rows, key=lambda r: (int(r["step"]), str(r["token"]), str(r["head"])))
        layerwise_rows = sorted(layerwise_rows, key=lambda r: (int(r["step"]), int(r["layer"]), str(r["token"])))
        headwise_rows = sorted(headwise_rows, key=lambda r: (int(r["step"]), int(r["layer"]), int(r["head"])))

        stepwise_csv = os.path.join(output_dir, "trajectory_entropy_stepwise.csv")
        layerwise_csv = os.path.join(output_dir, "trajectory_entropy_layerwise.csv")
        headwise_csv = os.path.join(output_dir, "trajectory_entropy_headwise.csv")
        _save_csv(stepwise_csv, stepwise_rows)
        _save_csv(layerwise_csv, layerwise_rows)
        _save_csv(headwise_csv, headwise_rows)

        stepwise_plot_framewise_path = _plot_wan21_t2v_trajectory_entropy_stepwise(
            rows=stepwise_rows,
            save_file=os.path.join(output_dir, "trajectory_entropy_stepwise_object_mean.pdf"),
            metric_key="entropy_norm_mean",
            ylabel="frame-wise normalized entropy",
            title_suffix="Frame-Wise Normalization",
            stepwise_layer=stepwise_layer_idx,
        )
        stepwise_plot_videowise_path = _plot_wan21_t2v_trajectory_entropy_stepwise(
            rows=stepwise_rows,
            save_file=os.path.join(output_dir, "trajectory_entropy_stepwise_object_mean_videowise.pdf"),
            metric_key="entropy_video_norm",
            ylabel="video-wise normalized entropy",
            title_suffix="Video-Wise Normalization (F*H*W)",
            stepwise_layer=stepwise_layer_idx,
        )
        layerwise_plot_path = _plot_wan21_t2v_trajectory_entropy_layerwise(
            rows=layerwise_rows,
            save_file=os.path.join(output_dir, "trajectory_entropy_layerwise_object_mean.pdf"),
        )
        headwise_plot_paths: List[str] = []
        plotted_head_layers = sorted({int(r["layer"]) for r in headwise_rows}) if headwise_rows else []
        for layer_idx in plotted_head_layers:
            layer_rows = [r for r in headwise_rows if int(r["layer"]) == int(layer_idx)]
            plot_path = _plot_wan21_t2v_trajectory_entropy_headwise(
                rows=layer_rows,
                save_file=os.path.join(output_dir, f"trajectory_entropy_headwise_layer_{layer_idx:02d}.pdf"),
                head_layer=int(layer_idx),
            )
            if plot_path:
                headwise_plot_paths.append(plot_path)
        headwise_plot_path = headwise_plot_paths[0] if len(headwise_plot_paths) == 1 else ""

        summary = {
            "experiment": "wan21_t2v_trajectory_entropy",
            "prompt": prompt,
            "target_object_words": list(object_words),
            "target_verb_words": list(verb_words),
            "prompt_tokens": prompt_tokens,
            "token_positions": word_to_positions,
            "token_types": word_to_type,
            "object_words_in_prompt": object_words_in_prompt,
            "trajectory_entropy_steps": [int(s) for s in entropy_steps],
            "trajectory_entropy_layerwise_steps": [int(s) for s in layerwise_steps],
            "trajectory_entropy_stepwise_layer": int(stepwise_layer_raw),
            "trajectory_entropy_stepwise_layer_resolved": int(stepwise_layer_idx),
            "trajectory_entropy_head_layer": int(head_layer_raw),
            "trajectory_entropy_head_layers_resolved": [int(x) for x in head_layer_indices],
            "last_layer_idx": int(last_layer_idx),
            "sampling_steps": int(sampling_steps),
            "step_head_layers_collected": [int(x) for x in step_head_layers],
            "save_video": bool(save_video),
            "video_path": video_path,
            "cross_attention_reused": bool(cross_attention_reused),
            "reuse_cross_attention_dir": cross_attention_dir if cross_attention_reused else "",
            "loaded_maps_source": loaded_maps_source,
            "stepwise_csv": stepwise_csv,
            "layerwise_csv": layerwise_csv,
            "headwise_csv": headwise_csv,
            "stepwise_plot_path": stepwise_plot_framewise_path,
            "stepwise_plot_framewise_path": stepwise_plot_framewise_path,
            "stepwise_plot_videowise_path": stepwise_plot_videowise_path,
            "layerwise_plot_path": layerwise_plot_path,
            "headwise_plot_path": headwise_plot_path,
            "headwise_plot_paths": headwise_plot_paths,
            "stepwise_rows": len(stepwise_rows),
            "layerwise_rows": len(layerwise_rows),
            "headwise_rows": len(headwise_rows),
            "entropy_eps": float(entropy_eps),
            "num_layers": int(num_layers),
            "task": task,
            "frame_num": int(frame_num),
            "size": f"{size[0]}x{size[1]}",
            "seed": int(seed),
            "sample_solver": sample_solver,
            "shift": float(shift),
            "guide_scale": float(guide_scale),
            "runtime_device_id": int(runtime.device_id),
            "runtime_local_rank": int(runtime.local_rank),
            "runtime_world_size": int(runtime.world_size),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        }
        _save_json(os.path.join(output_dir, "trajectory_entropy_summary.json"), summary)
        return summary
    return None
