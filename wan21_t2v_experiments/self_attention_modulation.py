"""Wan2.1-T2V experiment: self_attention_modulation.

This experiment profiles the three self-attention modulation tensors `e0`,
`e1`, and `e2` over diffusion steps and DiT layers. For all three tensors it
records generic magnitude/sign statistics. For `e2`, which gates the
self-attention branch write into the residual stream, it additionally records
RMS statistics for the raw and gated self-attention outputs.
"""

import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from .utils import (
    Wan21T2VParallelConfig,
    _broadcast_seed_if_needed,
    _build_wan21_t2v_pipeline,
    _ensure_dir,
    _init_wan21_t2v_runtime,
    _resolve_wan21_t2v_offload_model,
    _run_wan21_t2v_once_with_patch,
    _save_csv,
    _save_json,
    _save_wan21_t2v_video,
)
from .wan21_t2v_experiment_patch import (
    Wan21T2VAttentionProbeConfig,
    Wan21T2VCausalAttentionConfig,
    Wan21T2VPatchBundleConfig,
    Wan21T2VRopePatchConfig,
)


def _split_wan21_t2v_layers_into_buckets(
    layers: Sequence[int],
) -> List[Tuple[str, List[int]]]:
    """Split selected layers into three count-balanced buckets."""
    unique_layers = sorted(set(int(layer_id) for layer_id in layers))
    if not unique_layers:
        return []

    bucket_sizes = [len(unique_layers) // 3] * 3
    for bucket_index in range(len(unique_layers) % 3):
        bucket_sizes[bucket_index] += 1

    buckets: List[Tuple[str, List[int]]] = []
    cursor = 0
    bucket_names = ["shallow", "middle", "deep"]
    for bucket_name, bucket_size in zip(bucket_names, bucket_sizes):
        bucket_layers = unique_layers[cursor:cursor + bucket_size]
        cursor += bucket_size
        if bucket_layers:
            buckets.append((bucket_name, bucket_layers))
    return buckets


def _plot_wan21_t2v_self_attention_modulation_heatmap(
    rows: Sequence[Dict[str, object]],
    value_key: str,
    save_file: str,
    title: str,
):
    """Plot one step x layer heatmap from modulation CSV rows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return ""

    grouped_values: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    steps = sorted(set(int(row["step"]) for row in rows))
    layers = sorted(set(int(row["layer"]) for row in rows))
    for row in rows:
        grouped_values[(int(row["layer"]), int(row["step"]))].append(float(row[value_key]))

    heatmap = torch.zeros((len(layers), len(steps)), dtype=torch.float32)
    for layer_index, layer_id in enumerate(layers):
        for step_index, step_id in enumerate(steps):
            values = grouped_values.get((int(layer_id), int(step_id)), [])
            heatmap[layer_index, step_index] = float(sum(values) / len(values)) if values else 0.0

    fig_width = max(7.0, 0.45 * len(steps))
    fig_height = max(5.6, 0.26 * len(layers))
    fig, axis = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    image = axis.imshow(heatmap.numpy(), cmap="magma", aspect="auto", origin="lower")
    axis.set_title(title)
    axis.set_xlabel("diffusion step")
    axis.set_ylabel("DiT layer")
    axis.set_xticks(list(range(len(steps))))
    axis.set_xticklabels([str(step_id) for step_id in steps], rotation=45, ha="right", fontsize=8)
    axis.set_yticks(list(range(len(layers))))
    axis.set_yticklabels([str(layer_id) for layer_id in layers], fontsize=8)
    fig.colorbar(image, ax=axis, shrink=0.84)
    fig.tight_layout()

    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file


def _plot_wan21_t2v_self_attention_modulation_layer_step_curves(
    rows: Sequence[Dict[str, object]],
    value_key: str,
    save_file: str,
    title: str,
):
    """Plot one step curve per layer for one modulation metric."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    if not rows:
        return ""

    grouped_values: Dict[int, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    steps = sorted(set(int(row["step"]) for row in rows))
    layers = sorted(set(int(row["layer"]) for row in rows))
    for row in rows:
        grouped_values[int(row["layer"])][int(row["step"])].append(float(row[value_key]))

    fig, axis = plt.subplots(1, 1, figsize=(10.2, 5.6))
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=float(min(layers)), vmax=float(max(layers)))
    for color_index, layer_id in enumerate(layers):
        y_values = []
        for step_id in steps:
            values = grouped_values[int(layer_id)].get(int(step_id), [])
            y_values.append(float(sum(values) / len(values)) if values else 0.0)
        axis.plot(
            steps,
            y_values,
            linewidth=1.2,
            color=cmap(norm(float(layer_id))),
            label=f"L{layer_id}",
        )

    axis.set_title(title)
    axis.set_xlabel("diffusion step")
    axis.set_ylabel(value_key)
    axis.grid(alpha=0.22, linestyle="--")
    if len(layers) <= 16:
        axis.legend(fontsize=7, ncol=2)
    scalar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=axis, shrink=0.92, pad=0.02)
    colorbar.set_label("DiT layer")
    if len(layers) <= 32:
        colorbar.set_ticks([float(layer_id) for layer_id in layers])
        colorbar.set_ticklabels([str(layer_id) for layer_id in layers])
        colorbar.ax.tick_params(labelsize=7)
    fig.tight_layout()

    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file


def _plot_wan21_t2v_self_attention_modulation_single_layer_curve(
    rows: Sequence[Dict[str, object]],
    value_key: str,
    layer_id: int,
    save_file: str,
    title: str,
):
    """Plot one diffusion-step curve for one specific layer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return ""

    layer_rows = [row for row in rows if int(row["layer"]) == int(layer_id)]
    if not layer_rows:
        return ""

    grouped_values: Dict[int, List[float]] = defaultdict(list)
    steps = sorted(set(int(row["step"]) for row in layer_rows))
    for row in layer_rows:
        grouped_values[int(row["step"])].append(float(row[value_key]))

    y_values = []
    for step_id in steps:
        values = grouped_values.get(int(step_id), [])
        y_values.append(float(sum(values) / len(values)) if values else 0.0)

    fig, axis = plt.subplots(1, 1, figsize=(7.4, 4.8))
    axis.plot(
        steps,
        y_values,
        linewidth=1.9,
        marker="o",
        markersize=3.8,
        color="#0f766e",
    )
    axis.set_title(title)
    axis.set_xlabel("diffusion step")
    axis.set_ylabel(value_key)
    axis.grid(alpha=0.22, linestyle="--")
    fig.tight_layout()

    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file


def _plot_wan21_t2v_self_attention_modulation_step_curves(
    rows: Sequence[Dict[str, object]],
    value_key: str,
    save_file: str,
    title: str,
):
    """Plot step-wise modulation curves averaged over all layers and layer buckets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return ""

    steps = sorted(set(int(row["step"]) for row in rows))
    layers = sorted(set(int(row["layer"]) for row in rows))
    if not layers or not steps:
        return ""

    layer_buckets = _split_wan21_t2v_layers_into_buckets(layers)
    layer_to_bucket_name: Dict[int, str] = {}
    bucket_display_names: Dict[str, str] = {"all": "all"}
    for bucket_name, bucket_layers in layer_buckets:
        for layer_id in bucket_layers:
            layer_to_bucket_name[int(layer_id)] = str(bucket_name)
        if len(bucket_layers) == 1:
            bucket_display_names[str(bucket_name)] = f"{bucket_name}(L{int(bucket_layers[0])})"
        else:
            bucket_display_names[str(bucket_name)] = (
                f"{bucket_name}(L{int(bucket_layers[0])}-L{int(bucket_layers[-1])})"
            )

    bucket_to_step_values: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        step_id = int(row["step"])
        layer_id = int(row["layer"])
        value = float(row[value_key])
        bucket_to_step_values["all"][step_id].append(value)
        bucket_name = layer_to_bucket_name.get(int(layer_id), None)
        if bucket_name is not None:
            bucket_to_step_values[bucket_name][step_id].append(value)

    fig, axis = plt.subplots(1, 1, figsize=(8.6, 5.2))
    bucket_order = ["all", "shallow", "middle", "deep"]
    colors = {
        "all": "#0f766e",
        "shallow": "#2563eb",
        "middle": "#d97706",
        "deep": "#dc2626",
    }
    for bucket_name in bucket_order:
        if bucket_name not in bucket_to_step_values:
            continue
        y_values = []
        for step_id in steps:
            values = bucket_to_step_values[bucket_name].get(int(step_id), [])
            y_values.append(float(sum(values) / len(values)) if values else 0.0)
        axis.plot(
            steps,
            y_values,
            linewidth=1.9,
            marker="o",
            markersize=3.6,
            color=colors.get(bucket_name, None),
            label=bucket_display_names.get(bucket_name, bucket_name),
        )

    axis.set_title(title)
    axis.set_xlabel("diffusion step")
    axis.set_ylabel(value_key)
    axis.grid(alpha=0.22, linestyle="--")
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file


def _render_wan21_t2v_self_attention_modulation_plots(
    rows: Sequence[Dict[str, object]],
    output_dir: str,
) -> Dict[str, str]:
    """Render modulation heatmaps plus step curves grouped by modulation tensor."""
    plot_root = os.path.join(output_dir, "self_attention_modulation_plots")
    _ensure_dir(plot_root)

    metric_specs = [
        ("gate_mean", "Mean", True),
        ("gate_abs_mean", "Absolute Mean", True),
        ("gate_rms", "RMS", True),
        ("gate_positive_fraction", "Positive Fraction", True),
        ("gate_negative_fraction", "Negative Fraction", True),
        ("gate_max_abs", "Max Absolute Value", True),
        ("sa_output_rms", "SA Output RMS", False),
        ("gated_sa_output_rms", "Gated SA Output RMS", False),
        ("gated_to_raw_rms_ratio", "Gated / Raw SA Output RMS Ratio", False),
    ]

    plot_paths: Dict[str, str] = {}
    modulation_names = sorted(set(str(row["modulation_name"]) for row in rows))
    for modulation_name in modulation_names:
        modulation_rows = [row for row in rows if str(row["modulation_name"]) == str(modulation_name)]
        modulation_dir = os.path.join(plot_root, str(modulation_name))
        _ensure_dir(modulation_dir)
        for value_key, title, allow_for_all_modulations in metric_specs:
            if (not allow_for_all_modulations) and str(modulation_name) != "e2":
                continue
            metric_dir = os.path.join(modulation_dir, value_key)
            _ensure_dir(metric_dir)
            heatmap_path = _plot_wan21_t2v_self_attention_modulation_heatmap(
                rows=modulation_rows,
                value_key=value_key,
                save_file=os.path.join(metric_dir, "heatmap.pdf"),
                title=f"{str(modulation_name).upper()} {title}",
            )
            plot_paths[f"{modulation_name}/{value_key}/heatmap"] = heatmap_path
            step_curve_path = _plot_wan21_t2v_self_attention_modulation_step_curves(
                rows=modulation_rows,
                value_key=value_key,
                save_file=os.path.join(metric_dir, "step_curves_bucketed.pdf"),
                title=f"{str(modulation_name).upper()} {title} vs Diffusion Step",
            )
            plot_paths[f"{modulation_name}/{value_key}/step_curves_bucketed"] = step_curve_path
            layer_curve_path = _plot_wan21_t2v_self_attention_modulation_layer_step_curves(
                rows=modulation_rows,
                value_key=value_key,
                save_file=os.path.join(metric_dir, "step_curves_per_layer.pdf"),
                title=f"{str(modulation_name).upper()} {title} vs Diffusion Step | Per Layer",
            )
            plot_paths[f"{modulation_name}/{value_key}/step_curves_per_layer"] = layer_curve_path
            single_layer_dir = os.path.join(metric_dir, "per_layer")
            _ensure_dir(single_layer_dir)
            for row_layer_id in sorted(set(int(row["layer"]) for row in modulation_rows)):
                single_layer_path = _plot_wan21_t2v_self_attention_modulation_single_layer_curve(
                    rows=modulation_rows,
                    value_key=value_key,
                    layer_id=int(row_layer_id),
                    save_file=os.path.join(single_layer_dir, f"layer_{int(row_layer_id):02d}.pdf"),
                    title=f"{str(modulation_name).upper()} {title} vs Diffusion Step | Layer {int(row_layer_id)}",
                )
                plot_paths[
                    f"{modulation_name}/{value_key}/per_layer/layer_{int(row_layer_id):02d}"
                ] = single_layer_path
    return plot_paths


def run_wan21_t2v_self_attention_modulation(
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
    self_attention_modulation_steps: Sequence[int] = (),
    self_attention_modulation_layers: Sequence[int] = (),
    self_attention_modulation_branch: str = "cond",
    self_attention_modulation_stop_after_last_probe_step: bool = False,
    save_video: bool = True,
    parallel_cfg: Optional[Wan21T2VParallelConfig] = None,
):
    """Run step/layer profiling for the self-attention modulation tensors."""
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
    offload_model = _resolve_wan21_t2v_offload_model(runtime, offload_model)

    if self_attention_modulation_steps:
        resolved_steps = sorted(set(int(step) for step in self_attention_modulation_steps))
    else:
        resolved_steps = list(range(1, int(sampling_steps) + 1))
    if not resolved_steps:
        raise ValueError("self_attention_modulation resolved to an empty step list.")

    if self_attention_modulation_layers:
        resolved_layers = sorted(set(int(layer) for layer in self_attention_modulation_layers))
    else:
        resolved_layers = list(range(int(cfg.num_layers)))

    patch_cfg = Wan21T2VPatchBundleConfig(
        rope=Wan21T2VRopePatchConfig(enabled=True, mode="full"),
        probe=Wan21T2VAttentionProbeConfig(
            enabled=True,
            probe_steps=tuple(resolved_steps),
            probe_branch=str(self_attention_modulation_branch),
            collect_dt_histograms=False,
            collect_maas_maps=False,
            collect_distribution=False,
            collect_self_attn_modulation=True,
            modulation_layers=tuple(resolved_layers),
            stop_after_last_probe_step=bool(self_attention_modulation_stop_after_last_probe_step),
        ),
        causal=Wan21T2VCausalAttentionConfig(enabled=False),
    )

    video, state = _run_wan21_t2v_once_with_patch(
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
        return {
            "experiment": "wan21_t2v_self_attention_modulation",
            "seed": int(seed),
            "steps_resolved": [int(step) for step in resolved_steps],
            "layers_resolved": [int(layer) for layer in resolved_layers],
        }

    _ensure_dir(output_dir)
    if bool(save_video) and video is not None:
        video_path = os.path.join(output_dir, f"wan21_t2v_self_attention_modulation_seed_{seed}.mp4")
        _save_wan21_t2v_video(video, video_path, fps=cfg.sample_fps)

    rows = state.export_self_attn_modulation_rows()
    rows_path = os.path.join(output_dir, "self_attention_modulation_rows.csv")
    _save_csv(rows_path, rows)

    plot_paths = _render_wan21_t2v_self_attention_modulation_plots(
        rows=rows,
        output_dir=output_dir,
    )

    summary = {
        "experiment": "wan21_t2v_self_attention_modulation",
        "task": str(task),
        "prompt": str(prompt),
        "size": [int(size[0]), int(size[1])],
        "frame_num": int(frame_num),
        "sampling_steps": int(sampling_steps),
        "shift": float(shift),
        "guide_scale": float(guide_scale),
        "seed": int(seed),
        "self_attention_modulation_steps_input": [int(step) for step in self_attention_modulation_steps],
        "self_attention_modulation_steps_resolved": [int(step) for step in resolved_steps],
        "self_attention_modulation_layers_input": [int(layer) for layer in self_attention_modulation_layers],
        "self_attention_modulation_layers_resolved": [int(layer) for layer in resolved_layers],
        "self_attention_modulation_branch": str(self_attention_modulation_branch),
        "self_attention_modulation_stop_after_last_probe_step": bool(self_attention_modulation_stop_after_last_probe_step),
        "early_stop_triggered": bool(state.early_stop_triggered),
        "early_stop_completed_step": int(state.early_stop_completed_step),
        "early_stop_reason": str(state.early_stop_reason),
        "num_rows": int(len(rows)),
        "rows_path": rows_path,
        "plot_paths": plot_paths,
    }
    _save_json(os.path.join(output_dir, "self_attention_modulation_summary.json"), summary)
    return summary
