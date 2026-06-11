"""Wan2.1-T2V experiment: self_attention_modulation.

This experiment profiles the three self-attention modulation tensors `e0`,
`e1`, and `e2` over diffusion steps and DiT layers. For all three tensors it
records generic magnitude/sign statistics. For `e2`, which gates the
self-attention branch write into the residual stream, it additionally records
RMS statistics for the raw and gated self-attention outputs.
"""

import gc
import os
import csv
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
    _unwrap_wan21_t2v_dit_model_for_runtime_patch,
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


def _plot_wan21_t2v_self_attention_modulation_head_heatmap(
    rows: Sequence[Dict[str, object]],
    value_key: str,
    layer_id: int,
    save_file: str,
    title: str,
):
    """Plot one step x head heatmap for one selected layer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layer_rows = [row for row in rows if int(row["layer"]) == int(layer_id)]
    if not layer_rows:
        return ""

    grouped_values: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    steps = sorted(set(int(row["step"]) for row in layer_rows))
    heads = sorted(set(int(row["head"]) for row in layer_rows))
    for row in layer_rows:
        grouped_values[(int(row["head"]), int(row["step"]))].append(float(row[value_key]))

    heatmap = torch.zeros((len(heads), len(steps)), dtype=torch.float32)
    for head_index, head_id in enumerate(heads):
        for step_index, step_id in enumerate(steps):
            values = grouped_values.get((int(head_id), int(step_id)), [])
            heatmap[head_index, step_index] = float(sum(values) / len(values)) if values else 0.0

    fig_width = max(7.0, 0.42 * len(steps))
    fig_height = max(4.8, 0.34 * len(heads))
    fig, axis = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    image = axis.imshow(heatmap.numpy(), cmap="magma", aspect="auto", origin="lower")
    axis.set_title(title)
    axis.set_xlabel("diffusion step")
    axis.set_ylabel("SA head")
    axis.set_xticks(list(range(len(steps))))
    axis.set_xticklabels([str(step_id) for step_id in steps], rotation=45, ha="right", fontsize=8)
    axis.set_yticks(list(range(len(heads))))
    axis.set_yticklabels([str(head_id) for head_id in heads], fontsize=8)
    fig.colorbar(image, ax=axis, shrink=0.86)
    fig.tight_layout()

    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file


def _plot_wan21_t2v_self_attention_modulation_head_step_curves(
    rows: Sequence[Dict[str, object]],
    value_key: str,
    layer_id: int,
    save_file: str,
    title: str,
):
    """Overlay one diffusion-step curve per head for one selected layer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    layer_rows = [row for row in rows if int(row["layer"]) == int(layer_id)]
    if not layer_rows:
        return ""

    grouped_values: Dict[int, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    steps = sorted(set(int(row["step"]) for row in layer_rows))
    heads = sorted(set(int(row["head"]) for row in layer_rows))
    for row in layer_rows:
        grouped_values[int(row["head"])][int(row["step"])].append(float(row[value_key]))

    fig, axis = plt.subplots(1, 1, figsize=(9.6, 5.2))
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=float(min(heads)), vmax=float(max(heads)))
    for head_id in heads:
        y_values = []
        for step_id in steps:
            values = grouped_values[int(head_id)].get(int(step_id), [])
            y_values.append(float(sum(values) / len(values)) if values else 0.0)
        axis.plot(
            steps,
            y_values,
            linewidth=1.4,
            color=cmap(norm(float(head_id))),
            label=f"H{int(head_id)}",
        )

    axis.set_title(title)
    axis.set_xlabel("diffusion step")
    axis.set_ylabel(value_key)
    axis.grid(alpha=0.22, linestyle="--")
    if len(heads) <= 12:
        axis.legend(fontsize=7, ncol=2)
    scalar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=axis, shrink=0.92, pad=0.02)
    colorbar.set_label("SA head")
    colorbar.set_ticks([float(head_id) for head_id in heads])
    colorbar.set_ticklabels([str(head_id) for head_id in heads])
    colorbar.ax.tick_params(labelsize=7)
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


def _render_wan21_t2v_self_attention_modulation_head_plots(
    rows: Sequence[Dict[str, object]],
    output_dir: str,
) -> Dict[str, str]:
    """Render per-head self-attention write plots for the `e2` branch."""
    plot_root = os.path.join(output_dir, "self_attention_modulation_per_head_plots")
    _ensure_dir(plot_root)

    metric_specs = [
        ("sa_head_write_rms", "Per-Head SA Write RMS"),
        ("gated_sa_head_write_rms", "Per-Head Gated SA Write RMS"),
        ("gated_to_raw_sa_head_write_rms_ratio", "Per-Head Gated / Raw SA Write RMS Ratio"),
    ]

    plot_paths: Dict[str, str] = {}
    layers = sorted(set(int(row["layer"]) for row in rows))
    render_jobs = [
        (str(value_key), str(title), int(layer_id))
        for value_key, title in metric_specs
        for layer_id in layers
    ]
    if not render_jobs:
        return plot_paths

    from tqdm import tqdm

    progress_bar = tqdm(
        render_jobs,
        desc="Rendering self_attention_modulation per-head plots",
        leave=False,
    )
    for value_key, title, layer_id in progress_bar:
        metric_dir = os.path.join(plot_root, value_key)
        _ensure_dir(metric_dir)
        heatmap_path = _plot_wan21_t2v_self_attention_modulation_head_heatmap(
            rows=rows,
            value_key=value_key,
            layer_id=int(layer_id),
            save_file=os.path.join(metric_dir, f"layer_{int(layer_id):02d}_heatmap.pdf"),
            title=f"{title} | Layer {int(layer_id)}",
        )
        plot_paths[f"{value_key}/layer_{int(layer_id):02d}/heatmap"] = heatmap_path
        curve_path = _plot_wan21_t2v_self_attention_modulation_head_step_curves(
            rows=rows,
            value_key=value_key,
            layer_id=int(layer_id),
            save_file=os.path.join(metric_dir, f"layer_{int(layer_id):02d}_step_curves.pdf"),
            title=f"{title} vs Diffusion Step | Layer {int(layer_id)}",
        )
        plot_paths[f"{value_key}/layer_{int(layer_id):02d}/step_curves"] = curve_path
    progress_bar.close()
    return plot_paths


def _render_wan21_t2v_self_attention_modulation_decomposition_plots(
    rows: Sequence[Dict[str, object]],
    output_dir: str,
) -> Dict[str, str]:
    """Render heatmaps for SA decomposition RMS terms."""
    plot_root = os.path.join(output_dir, "self_attention_modulation_decomposition_plots")
    _ensure_dir(plot_root)

    metric_specs = [
        ("x_hat_rms", "X Hat RMS"),
        ("v_rms", "V RMS"),
        ("attn_out_pre_o_rms", "Attention Output Pre-O RMS"),
    ]
    plot_paths: Dict[str, str] = {}
    for value_key, title in metric_specs:
        save_path = _plot_wan21_t2v_self_attention_modulation_heatmap(
            rows=rows,
            value_key=value_key,
            save_file=os.path.join(plot_root, f"{value_key}_heatmap.pdf"),
            title=title,
        )
        plot_paths[f"{value_key}/heatmap"] = save_path
    return plot_paths


def _compute_wan21_t2v_self_attention_weight_norm_rows(target_model) -> List[Dict[str, object]]:
    """Return per-layer norm summaries for self-attention `W_V` and `W_O`."""
    rows: List[Dict[str, object]] = []
    for layer_idx, block in enumerate(target_model.blocks):
        w_v = block.self_attn.v.weight.detach().float()
        w_o = block.self_attn.o.weight.detach().float()
        rows.append(
            {
                "layer": int(layer_idx),
                "w_v_fro_norm": float(torch.linalg.norm(w_v).item()),
                "w_o_fro_norm": float(torch.linalg.norm(w_o).item()),
                "w_v_spectral_proxy": float(torch.linalg.vector_norm(w_v, ord=2, dim=1).max().item()),
                "w_o_spectral_proxy": float(torch.linalg.vector_norm(w_o, ord=2, dim=1).max().item()),
            }
        )
    return rows


def _plot_wan21_t2v_self_attention_weight_norms(
    rows: Sequence[Dict[str, object]],
    save_file: str,
) -> str:
    """Plot layer-wise `W_V / W_O` norm comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return ""

    layers = [int(row["layer"]) for row in rows]
    w_v = [float(row["w_v_fro_norm"]) for row in rows]
    w_o = [float(row["w_o_fro_norm"]) for row in rows]

    fig, axis = plt.subplots(1, 1, figsize=(8.6, 5.0))
    axis.plot(layers, w_v, linewidth=1.8, marker="o", markersize=3.6, color="#2563eb", label="W_V Frobenius norm")
    axis.plot(layers, w_o, linewidth=1.8, marker="o", markersize=3.6, color="#dc2626", label="W_O Frobenius norm")
    axis.set_title("Self-Attention W_V / W_O Norms")
    axis.set_xlabel("DiT layer")
    axis.set_ylabel("matrix Frobenius norm")
    axis.grid(alpha=0.22, linestyle="--")
    axis.legend(fontsize=8)
    fig.tight_layout()

    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file


def _parse_wan21_t2v_channel_profile_targets(spec: str) -> List[Tuple[int, int]]:
    """Parse `step:layer` pairs from a CSV string such as `1:0,1:28,25:0`."""
    text = str(spec).strip()
    if not text:
        return []
    pairs: List[Tuple[int, int]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        step_text, layer_text = item.split(":")
        pairs.append((int(step_text), int(layer_text)))
    return pairs


def _load_wan21_t2v_csv_rows(path: str) -> List[Dict[str, str]]:
    """Load a CSV file into a list of row dictionaries."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _plot_wan21_t2v_channel_profile_bars(
    profile: torch.Tensor,
    step: int,
    layer: int,
    save_dir: str,
    topk: int,
) -> Dict[str, str]:
    """Render three channel-profile bar charts plus top-k annotated variants."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import transforms

    metric_specs = [
        (0, "sa_channel_energy", "SA Channel Energy"),
        (1, "gate_channel_rms", "Gate Channel RMS"),
        (2, "gated_channel_energy", "Gated SA Channel Energy"),
    ]
    channel_count = int(profile.size(1))
    channels = list(range(channel_count))
    out_paths: Dict[str, str] = {}
    tick_fontsize = 20
    annotation_fontsize = tick_fontsize
    x_min = -0.5
    x_max = float(channel_count) - 0.5
    text_gap = 1.0
    base_bar_color = "#2563eb"
    annotated_bar_color = "#16a34a"

    for metric_index, metric_key, title in metric_specs:
        values = profile[metric_index].detach().cpu().float()
        base_path = os.path.join(save_dir, f"{metric_key}.pdf")
        fig, axis = plt.subplots(1, 1, figsize=(max(12.0, 0.04 * channel_count), 4.6))
        axis.bar(channels, values.numpy(), width=1.0, color=base_bar_color)
        axis.set_xlim(x_min, x_max)
        axis.set_title(f"{title} | Step {int(step)} Layer {int(layer)}")
        axis.set_xlabel("channel")
        axis.set_ylabel(metric_key)
        axis.grid(alpha=0.18, linestyle="--", axis="y")
        axis.tick_params(axis="both", labelsize=tick_fontsize)
        fig.tight_layout()
        _ensure_dir(os.path.dirname(base_path))
        fig.savefig(base_path, format="pdf")
        plt.close(fig)
        out_paths[metric_key] = base_path

        annotated_path = os.path.join(save_dir, f"{metric_key}_topk_annotated.pdf")
        fig, axis = plt.subplots(1, 1, figsize=(max(12.0, 0.04 * channel_count), 4.8))
        top_indices = torch.topk(values, k=min(int(topk), int(values.numel()))).indices.tolist()
        top_index_set = {int(channel_index) for channel_index in top_indices}
        bar_colors = [
            annotated_bar_color if int(channel_index) in top_index_set else base_bar_color
            for channel_index in channels
        ]
        axis.bar(channels, values.numpy(), width=1.0, color=bar_colors)
        axis.set_xlim(x_min, x_max)
        for channel_index in top_indices:
            value = float(values[int(channel_index)].item())
            label = f"({int(channel_index)}, {value:.4g})"
            text_artist = axis.text(
                float(channel_index) + text_gap,
                value,
                label,
                rotation=0,
                fontsize=annotation_fontsize,
                ha="left",
                va="top",
            )
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            axis_bbox = axis.get_window_extent(renderer=renderer)
            use_axes_x = False
            use_axes_y = False
            text_bbox = text_artist.get_window_extent(renderer=renderer)
            if text_bbox.x1 > axis_bbox.x1:
                use_axes_x = True
            if text_bbox.y0 < axis_bbox.y0:
                use_axes_y = True
            if use_axes_x or use_axes_y:
                text_artist.set_transform(
                    transforms.blended_transform_factory(
                        axis.transAxes if use_axes_x else axis.transData,
                        axis.transAxes if use_axes_y else axis.transData,
                    )
                )
                if use_axes_x:
                    text_artist.set_x(1.0)
                    text_artist.set_ha("right")
                if use_axes_y:
                    text_artist.set_y(0.0)
                    text_artist.set_va("bottom")
        axis.set_title(f"{title} | Step {int(step)} Layer {int(layer)} | Top-{min(int(topk), int(values.numel()))} annotated")
        axis.set_xlabel("channel")
        axis.set_ylabel(metric_key)
        axis.grid(alpha=0.18, linestyle="--", axis="y")
        axis.tick_params(axis="both", labelsize=tick_fontsize)
        fig.tight_layout()
        fig.savefig(annotated_path, format="pdf")
        plt.close(fig)
        out_paths[f"{metric_key}_topk_annotated"] = annotated_path

    return out_paths


def _render_wan21_t2v_channel_profile_plots(
    profiles: Dict[Tuple[int, int], torch.Tensor],
    output_dir: str,
    target_pairs: Sequence[Tuple[int, int]],
    topk: int,
) -> Dict[str, str]:
    """Render channel-profile bar charts.

    If `target_pairs` is empty, render all collected `(step, layer)` pairs.
    Otherwise, render only the explicitly requested subset.
    """
    plot_root = os.path.join(output_dir, "self_attention_modulation_channel_profiles")
    _ensure_dir(plot_root)
    plot_paths: Dict[str, str] = {}
    resolved_pairs = list(target_pairs) if target_pairs else sorted(profiles.keys())
    if not resolved_pairs:
        return plot_paths

    from tqdm import tqdm

    progress_bar = tqdm(
        resolved_pairs,
        desc="Rendering self_attention_modulation channel profiles",
        leave=False,
    )
    for step, layer in progress_bar:
        key = (int(step), int(layer))
        if key not in profiles:
            continue
        save_dir = os.path.join(plot_root, f"step_{int(step):03d}", f"layer_{int(layer):02d}")
        layer_paths = _plot_wan21_t2v_channel_profile_bars(
            profile=profiles[key],
            step=int(step),
            layer=int(layer),
            save_dir=save_dir,
            topk=int(topk),
        )
        for metric_key, path in layer_paths.items():
            plot_paths[f"step_{int(step):03d}/layer_{int(layer):02d}/{metric_key}"] = path
    progress_bar.close()
    return plot_paths


def _render_wan21_t2v_self_attention_modulation_saved_outputs(
    output_dir: str,
    target_pairs: Sequence[Tuple[int, int]],
    topk: int,
) -> Dict[str, object]:
    """Redraw all self_attention_modulation plots from previously saved CSV/PT artifacts."""
    rows_path = os.path.join(output_dir, "self_attention_modulation_rows.csv")
    head_rows_path = os.path.join(output_dir, "self_attention_modulation_head_rows.csv")
    decomposition_rows_path = os.path.join(output_dir, "self_attention_modulation_decomposition_rows.csv")
    channel_profiles_path = os.path.join(output_dir, "self_attention_modulation_channel_profiles.pt")
    weight_norm_rows_path = os.path.join(output_dir, "self_attention_modulation_weight_norms.csv")

    rows = _load_wan21_t2v_csv_rows(rows_path)
    head_rows = _load_wan21_t2v_csv_rows(head_rows_path)
    decomposition_rows = _load_wan21_t2v_csv_rows(decomposition_rows_path)
    channel_profiles = (
        torch.load(channel_profiles_path, map_location="cpu", weights_only=False)
        if os.path.exists(channel_profiles_path)
        else {}
    )
    weight_norm_rows = _load_wan21_t2v_csv_rows(weight_norm_rows_path)

    plot_paths = _render_wan21_t2v_self_attention_modulation_plots(
        rows=rows,
        output_dir=output_dir,
    )
    head_plot_paths = _render_wan21_t2v_self_attention_modulation_head_plots(
        rows=head_rows,
        output_dir=output_dir,
    )
    decomposition_plot_paths = _render_wan21_t2v_self_attention_modulation_decomposition_plots(
        rows=decomposition_rows,
        output_dir=output_dir,
    )
    weight_norm_plot_path = _plot_wan21_t2v_self_attention_weight_norms(
        rows=weight_norm_rows,
        save_file=os.path.join(output_dir, "self_attention_modulation_weight_norms.pdf"),
    )
    channel_profile_plot_paths = _render_wan21_t2v_channel_profile_plots(
        profiles=channel_profiles,
        output_dir=output_dir,
        target_pairs=target_pairs,
        topk=int(topk),
    )

    return {
        "rows_path": rows_path,
        "head_rows_path": head_rows_path,
        "decomposition_rows_path": decomposition_rows_path,
        "channel_profiles_path": channel_profiles_path,
        "weight_norm_rows_path": weight_norm_rows_path,
        "plot_paths": plot_paths,
        "head_plot_paths": head_plot_paths,
        "decomposition_plot_paths": decomposition_plot_paths,
        "weight_norm_plot_path": weight_norm_plot_path,
        "channel_profile_plot_paths": channel_profile_plot_paths,
        "num_rows": int(len(rows)),
        "num_head_rows": int(len(head_rows)),
        "num_decomposition_rows": int(len(decomposition_rows)),
    }


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
    self_attention_modulation_channel_profile_targets: Sequence[Tuple[int, int]] = (),
    self_attention_modulation_channel_profile_topk: int = 12,
    self_attention_modulation_plot_only_from_saved: bool = False,
    save_video: bool = True,
    parallel_cfg: Optional[Wan21T2VParallelConfig] = None,
):
    """Run step/layer profiling for the self-attention modulation tensors."""
    parallel_cfg = parallel_cfg or Wan21T2VParallelConfig()
    if bool(self_attention_modulation_plot_only_from_saved):
        runtime = _init_wan21_t2v_runtime(parallel_cfg, explicit_device_id=device_id)
        if dist.is_initialized():
            dist.barrier()
        if runtime.rank != 0:
            return {
                "experiment": "wan21_t2v_self_attention_modulation",
                "self_attention_modulation_plot_only_from_saved": True,
                "rank": int(runtime.rank),
                "world_size": int(runtime.world_size),
            }
        _ensure_dir(output_dir)
        rendered = _render_wan21_t2v_self_attention_modulation_saved_outputs(
            output_dir=output_dir,
            target_pairs=self_attention_modulation_channel_profile_targets,
            topk=int(self_attention_modulation_channel_profile_topk),
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
            "self_attention_modulation_layers_input": [int(layer) for layer in self_attention_modulation_layers],
            "self_attention_modulation_branch": str(self_attention_modulation_branch),
            "self_attention_modulation_stop_after_last_probe_step": bool(self_attention_modulation_stop_after_last_probe_step),
            "self_attention_modulation_channel_profile_targets": [
                [int(step), int(layer)] for step, layer in self_attention_modulation_channel_profile_targets
            ],
            "self_attention_modulation_channel_profile_topk": int(self_attention_modulation_channel_profile_topk),
            "self_attention_modulation_plot_only_from_saved": True,
            **rendered,
        }
        _save_json(os.path.join(output_dir, "self_attention_modulation_summary.json"), summary)
        return summary

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
    target_model = _unwrap_wan21_t2v_dit_model_for_runtime_patch(pipeline.model)

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
    head_rows = state.export_self_attn_modulation_head_rows()
    head_rows_path = os.path.join(output_dir, "self_attention_modulation_head_rows.csv")
    _save_csv(head_rows_path, head_rows)
    decomposition_rows = state.export_self_attn_modulation_decomposition_rows()
    decomposition_rows_path = os.path.join(output_dir, "self_attention_modulation_decomposition_rows.csv")
    _save_csv(decomposition_rows_path, decomposition_rows)
    channel_profiles = state.export_self_attn_modulation_channel_profiles()
    channel_profiles_path = os.path.join(output_dir, "self_attention_modulation_channel_profiles.pt")
    torch.save(channel_profiles, channel_profiles_path)
    weight_norm_rows = _compute_wan21_t2v_self_attention_weight_norm_rows(target_model)
    weight_norm_rows_path = os.path.join(output_dir, "self_attention_modulation_weight_norms.csv")
    _save_csv(weight_norm_rows_path, weight_norm_rows)
    early_stop_triggered = bool(state.early_stop_triggered)
    early_stop_completed_step = int(state.early_stop_completed_step)
    early_stop_reason = str(state.early_stop_reason)

    if video is not None:
        del video
        video = None
    del target_model
    target_model = None
    del pipeline
    pipeline = None
    del state
    state = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    plot_paths = _render_wan21_t2v_self_attention_modulation_plots(
        rows=rows,
        output_dir=output_dir,
    )
    head_plot_paths = _render_wan21_t2v_self_attention_modulation_head_plots(
        rows=head_rows,
        output_dir=output_dir,
    )
    decomposition_plot_paths = _render_wan21_t2v_self_attention_modulation_decomposition_plots(
        rows=decomposition_rows,
        output_dir=output_dir,
    )
    weight_norm_plot_path = _plot_wan21_t2v_self_attention_weight_norms(
        rows=weight_norm_rows,
        save_file=os.path.join(output_dir, "self_attention_modulation_weight_norms.pdf"),
    )
    channel_profile_plot_paths = _render_wan21_t2v_channel_profile_plots(
        profiles=channel_profiles,
        output_dir=output_dir,
        target_pairs=self_attention_modulation_channel_profile_targets,
        topk=int(self_attention_modulation_channel_profile_topk),
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
        "self_attention_modulation_channel_profile_targets": [
            [int(step), int(layer)] for step, layer in self_attention_modulation_channel_profile_targets
        ],
        "self_attention_modulation_channel_profile_topk": int(self_attention_modulation_channel_profile_topk),
        "self_attention_modulation_plot_only_from_saved": False,
        "early_stop_triggered": early_stop_triggered,
        "early_stop_completed_step": early_stop_completed_step,
        "early_stop_reason": early_stop_reason,
        "num_rows": int(len(rows)),
        "rows_path": rows_path,
        "num_head_rows": int(len(head_rows)),
        "head_rows_path": head_rows_path,
        "num_decomposition_rows": int(len(decomposition_rows)),
        "decomposition_rows_path": decomposition_rows_path,
        "channel_profiles_path": channel_profiles_path,
        "weight_norm_rows_path": weight_norm_rows_path,
        "plot_paths": plot_paths,
        "head_plot_paths": head_plot_paths,
        "decomposition_plot_paths": decomposition_plot_paths,
        "weight_norm_plot_path": weight_norm_plot_path,
        "channel_profile_plot_paths": channel_profile_plot_paths,
    }
    _save_json(os.path.join(output_dir, "self_attention_modulation_summary.json"), summary)
    return summary
