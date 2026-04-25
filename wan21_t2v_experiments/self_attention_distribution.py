"""Wan2.1-T2V experiment: self_attention_distribution.

Main entry:
- run_wan21_t2v_self_attention_distribution

This experiment probes self-attention distributions in two complementary ways:
1) object-region queries: tokens inside a reference object support region
2) global queries: uniformly sampled tokens over the whole token grid

It reuses the existing q/k probe stack from wan21_t2v_experiment_patch.py and
the reference-trajectory construction used by head_evolution.py.
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
    _load_wan21_t2v_cross_attention_mean_maps_from_disk,
    _load_wan21_t2v_cross_attention_token_meta,
    _mean_wan21_t2v_headmean_map_for_words,
    _resolve_wan21_t2v_offload_model,
    _run_wan21_t2v_once_with_patch,
    _save_csv,
    _save_json,
    _save_wan21_t2v_video,
    _init_wan21_t2v_runtime,
)
from .head_evolution import (
    _build_wan21_t2v_trajectory_support_mask_from_centers,
    _extract_wan21_t2v_reference_peak_and_centroid_trajectory,
)
from .wan21_t2v_experiment_patch import (
    Wan21T2VAttentionProbeConfig,
    Wan21T2VCausalAttentionConfig,
    Wan21T2VPatchBundleConfig,
    Wan21T2VRopePatchConfig,
)


def _build_wan21_t2v_self_attention_distribution_reference_support(
    reuse_cross_attention_dir: str,
    target_object_words: Sequence[str],
    target_verb_words: Sequence[str],
    reference_step: int,
    reference_layer: int,
    center_mode: str,
    center_power: float,
    center_quantile: float,
    support_radius_mode: str,
    support_radius_fixed: float,
    support_radius_alpha: float,
    support_radius_min: float,
    support_radius_max_ratio: float,
) -> Dict[str, object]:
    """Build a frame-wise object support mask from reused cross-attention maps."""
    mean_maps, loaded_map_path = _load_wan21_t2v_cross_attention_mean_maps_from_disk(reuse_cross_attention_dir)
    words_in_maps = sorted({str(key[2]) for key in mean_maps.keys()})
    _load_wan21_t2v_cross_attention_token_meta(
        output_dir=reuse_cross_attention_dir,
        words_in_maps=words_in_maps,
        target_object_words=target_object_words,
        target_verb_words=target_verb_words,
    )
    object_words_in_maps = [str(word) for word in target_object_words if str(word) in words_in_maps]
    if not object_words_in_maps:
        raise ValueError(
            "None of target_object_words are present in reused cross-attention maps. "
            f"requested={list(target_object_words)} available={words_in_maps}"
        )

    reference_map = _mean_wan21_t2v_headmean_map_for_words(
        mean_maps=mean_maps,
        step=int(reference_step),
        layer=int(reference_layer),
        words=object_words_in_maps,
    )
    if reference_map is None:
        raise ValueError(
            "Cannot build reference support mask because the requested reference map is missing. "
            f"step={int(reference_step)} layer={int(reference_layer)} words={object_words_in_maps}"
        )

    reference_trajectory_data = _extract_wan21_t2v_reference_peak_and_centroid_trajectory(
        map_fhw=reference_map,
        power=float(center_power),
        quantile=float(center_quantile),
    )
    center_mode_name = str(center_mode).strip().lower()
    if center_mode_name == "peak":
        center_trajectory = reference_trajectory_data["peak_centers"]
    elif center_mode_name == "centroid":
        center_trajectory = reference_trajectory_data["centroid_centers"]
    elif center_mode_name == "geometric_center":
        center_trajectory = reference_trajectory_data["geometric_centers"]
    else:
        raise ValueError(
            "self_attention_distribution_reference_center_mode must be one of "
            "{'peak', 'centroid', 'geometric_center'}."
        )

    support_mask_fhw, support_radius_per_frame = _build_wan21_t2v_trajectory_support_mask_from_centers(
        center_trajectory=center_trajectory,
        component_areas=reference_trajectory_data["component_areas"],
        token_grid_height=int(reference_trajectory_data["token_grid_height"]),
        token_grid_width=int(reference_trajectory_data["token_grid_width"]),
        support_radius_mode=str(support_radius_mode),
        support_radius_fixed=float(support_radius_fixed),
        support_radius_alpha=float(support_radius_alpha),
        support_radius_min=float(support_radius_min),
        support_radius_max_ratio=float(support_radius_max_ratio),
    )

    return {
        "reference_map_path": loaded_map_path,
        "object_words_in_maps": object_words_in_maps,
        "reference_map": reference_map,
        "center_trajectory": center_trajectory,
        "support_mask_fhw": support_mask_fhw,
        "support_radius_per_frame": support_radius_per_frame,
    }


def _plot_wan21_t2v_self_attention_distribution_object_heatmap(
    rows: Sequence[Dict[str, object]],
    save_file: str,
    title: str,
    value_key: str,
):
    """Plot head-mean query-frame x key-frame heatmap for object-region statistics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return ""

    grouped_values: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    query_frames = sorted(set(int(row["query_frame"]) for row in rows))
    key_frames = sorted(set(int(row["key_frame"]) for row in rows))
    for row in rows:
        grouped_values[(int(row["query_frame"]), int(row["key_frame"]))].append(float(row[value_key]))

    heatmap = torch.zeros((len(query_frames), len(key_frames)), dtype=torch.float32)
    for query_index, query_frame in enumerate(query_frames):
        for key_index, key_frame in enumerate(key_frames):
            values = grouped_values.get((int(query_frame), int(key_frame)), [])
            heatmap[query_index, key_index] = float(sum(values) / len(values)) if values else 0.0

    fig_width = max(6.6, 0.28 * len(key_frames))
    fig_height = max(5.4, 0.28 * len(query_frames))
    fig, axis = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    image = axis.imshow(heatmap.numpy(), cmap="magma", aspect="auto")
    axis.set_title(title)
    axis.set_xlabel("key frame")
    axis.set_ylabel("query frame")
    axis.set_xticks(list(range(len(key_frames))))
    axis.set_xticklabels([str(v) for v in key_frames], rotation=45, ha="right", fontsize=8)
    axis.set_yticks(list(range(len(query_frames))))
    axis.set_yticklabels([str(v) for v in query_frames], fontsize=8)
    fig.colorbar(image, ax=axis, shrink=0.82)
    fig.tight_layout()

    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file


def _plot_wan21_t2v_self_attention_distribution_object_dt_curves(
    rows: Sequence[Dict[str, object]],
    save_file: str,
    title: str,
):
    """Plot head-mean object/non-object/frame mass curves against signed dt."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return ""

    grouped: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        dt_value = int(row["dt"])
        grouped[dt_value]["frame_mass"].append(float(row["frame_mass"]))
        grouped[dt_value]["object_mass"].append(float(row["object_mass"]))
        grouped[dt_value]["nonobject_mass"].append(float(row["nonobject_mass"]))
        grouped[dt_value]["object_fraction"].append(float(row["object_fraction"]))

    dt_values = sorted(grouped.keys())
    frame_mass = [sum(grouped[dt]["frame_mass"]) / len(grouped[dt]["frame_mass"]) for dt in dt_values]
    object_mass = [sum(grouped[dt]["object_mass"]) / len(grouped[dt]["object_mass"]) for dt in dt_values]
    nonobject_mass = [sum(grouped[dt]["nonobject_mass"]) / len(grouped[dt]["nonobject_mass"]) for dt in dt_values]
    object_fraction = [sum(grouped[dt]["object_fraction"]) / len(grouped[dt]["object_fraction"]) for dt in dt_values]

    fig, axis_left = plt.subplots(1, 1, figsize=(8.4, 5.0))
    axis_left.plot(dt_values, frame_mass, linewidth=1.8, color="#334155", label="frame_mass")
    axis_left.plot(dt_values, object_mass, linewidth=1.8, color="#dc2626", label="object_mass")
    axis_left.plot(dt_values, nonobject_mass, linewidth=1.8, color="#2563eb", label="nonobject_mass")
    axis_left.set_xlabel("signed dt")
    axis_left.set_ylabel("attention mass")
    axis_left.grid(alpha=0.22, linestyle="--")

    axis_right = axis_left.twinx()
    axis_right.plot(dt_values, object_fraction, linewidth=1.8, color="#16a34a", linestyle="--", label="object_fraction")
    axis_right.set_ylabel("object fraction")
    axis_right.set_ylim(0.0, 1.0)

    lines_left, labels_left = axis_left.get_legend_handles_labels()
    lines_right, labels_right = axis_right.get_legend_handles_labels()
    axis_left.legend(lines_left + lines_right, labels_left + labels_right, fontsize=8, ncol=2)
    axis_left.set_title(title)
    fig.tight_layout()

    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file


def _plot_wan21_t2v_self_attention_distribution_global_dt_curves(
    rows: Sequence[Dict[str, object]],
    save_file: str,
    title: str,
):
    """Plot head-mean global self-attention mass curves against signed dt for each query bucket."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return ""

    bucket_to_grouped: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        bucket_to_grouped[str(row["query_bucket"])][int(row["dt"])].append(float(row["attention_mass"]))

    fig, axis = plt.subplots(1, 1, figsize=(8.4, 5.0))
    bucket_order = [bucket for bucket in ("all", "early", "middle", "late") if bucket in bucket_to_grouped]
    colors = {"all": "#0f766e", "early": "#dc2626", "middle": "#2563eb", "late": "#7c3aed"}
    for bucket_name in bucket_order:
        dt_values = sorted(bucket_to_grouped[bucket_name].keys())
        mean_values = [
            sum(bucket_to_grouped[bucket_name][dt_value]) / len(bucket_to_grouped[bucket_name][dt_value])
            for dt_value in dt_values
        ]
        axis.plot(
            dt_values,
            mean_values,
            linewidth=1.8,
            color=colors.get(bucket_name, None),
            label=bucket_name,
        )

    axis.set_title(title)
    axis.set_xlabel("signed dt")
    axis.set_ylabel("attention mass")
    axis.grid(alpha=0.22, linestyle="--")
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file


def run_wan21_t2v_self_attention_distribution(
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
    target_object_words: Sequence[str] = tuple(),
    target_verb_words: Sequence[str] = tuple(),
    reuse_cross_attention_dir: str = "",
    self_attention_distribution_steps: Sequence[int] = (1, 2, 3),
    self_attention_distribution_layers: Sequence[int] = tuple(),
    self_attention_distribution_branch: str = "cond",
    self_attention_distribution_reference_step: int = 50,
    self_attention_distribution_reference_layer: int = 27,
    self_attention_distribution_reference_center_mode: str = "geometric_center",
    self_attention_distribution_reference_center_power: float = 1.5,
    self_attention_distribution_reference_center_quantile: float = 0.8,
    self_attention_distribution_support_radius_mode: str = "adaptive_area",
    self_attention_distribution_support_radius_fixed: float = 2.0,
    self_attention_distribution_support_radius_alpha: float = 1.5,
    self_attention_distribution_support_radius_min: float = 1.0,
    self_attention_distribution_support_radius_max_ratio: float = 0.25,
    self_attention_distribution_query_frame_count: int = 8,
    self_attention_distribution_global_query_tokens_per_frame: int = 64,
    self_attention_distribution_object_query_token_limit_per_frame: int = 0,
    save_video: bool = True,
    parallel_cfg: Optional[Wan21T2VParallelConfig] = None,
):
    """Run object-region and global self-attention distribution analysis."""
    if not reuse_cross_attention_dir:
        raise ValueError("self_attention_distribution requires reuse_cross_attention_dir.")
    if not target_object_words:
        raise ValueError("self_attention_distribution requires target_object_words.")

    reference_support = _build_wan21_t2v_self_attention_distribution_reference_support(
        reuse_cross_attention_dir=reuse_cross_attention_dir,
        target_object_words=target_object_words,
        target_verb_words=target_verb_words,
        reference_step=int(self_attention_distribution_reference_step),
        reference_layer=int(self_attention_distribution_reference_layer),
        center_mode=str(self_attention_distribution_reference_center_mode),
        center_power=float(self_attention_distribution_reference_center_power),
        center_quantile=float(self_attention_distribution_reference_center_quantile),
        support_radius_mode=str(self_attention_distribution_support_radius_mode),
        support_radius_fixed=float(self_attention_distribution_support_radius_fixed),
        support_radius_alpha=float(self_attention_distribution_support_radius_alpha),
        support_radius_min=float(self_attention_distribution_support_radius_min),
        support_radius_max_ratio=float(self_attention_distribution_support_radius_max_ratio),
    )

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

    patch_cfg = Wan21T2VPatchBundleConfig(
        rope=Wan21T2VRopePatchConfig(enabled=True, mode="full"),
        probe=Wan21T2VAttentionProbeConfig(
            enabled=True,
            probe_steps=tuple(self_attention_distribution_steps),
            probe_branch=str(self_attention_distribution_branch),
            collect_dt_histograms=False,
            collect_maas_maps=False,
            collect_distribution=True,
            distribution_layers=tuple(int(layer) for layer in self_attention_distribution_layers),
            distribution_query_frame_count=int(self_attention_distribution_query_frame_count),
            distribution_global_query_tokens_per_frame=int(self_attention_distribution_global_query_tokens_per_frame),
            distribution_object_query_token_limit_per_frame=int(self_attention_distribution_object_query_token_limit_per_frame),
            distribution_object_support_mask=reference_support["support_mask_fhw"],
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
        return None

    _ensure_dir(output_dir)
    video_path = ""
    if bool(save_video):
        video_path = os.path.join(output_dir, f"wan21_t2v_self_attention_distribution_seed_{seed}.mp4")
        _save_wan21_t2v_video(video, video_path, fps=cfg.sample_fps)

    exported_rows = state.export_distribution_rows()
    object_rows = exported_rows["object_rows"]
    object_dt_rows = exported_rows["object_dt_rows"]
    global_dt_rows = exported_rows["global_dt_rows"]

    object_rows_path = os.path.join(output_dir, "self_attention_distribution_object_rows.csv")
    object_dt_rows_path = os.path.join(output_dir, "self_attention_distribution_object_dt_rows.csv")
    global_dt_rows_path = os.path.join(output_dir, "self_attention_distribution_global_dt_rows.csv")
    _save_csv(object_rows_path, object_rows)
    _save_csv(object_dt_rows_path, object_dt_rows)
    _save_csv(global_dt_rows_path, global_dt_rows)

    reference_rows = []
    center_trajectory = reference_support["center_trajectory"]
    support_radius_per_frame = reference_support["support_radius_per_frame"]
    support_mask_fhw = reference_support["support_mask_fhw"]
    for frame_index, (center_y, center_x) in enumerate(center_trajectory):
        reference_rows.append(
            {
                "frame": int(frame_index),
                "center_y": float(center_y),
                "center_x": float(center_x),
                "support_radius": float(support_radius_per_frame[frame_index]),
                "support_area": float(support_mask_fhw[frame_index].sum().item()),
            }
        )
    reference_rows_path = os.path.join(output_dir, "self_attention_distribution_reference_support.csv")
    _save_csv(reference_rows_path, reference_rows)

    plot_paths: List[str] = []
    plots_dir = os.path.join(output_dir, "self_attention_distribution_plots")
    object_rows_by_step_layer: Dict[Tuple[int, int], List[Dict[str, object]]] = defaultdict(list)
    object_dt_rows_by_step_layer: Dict[Tuple[int, int], List[Dict[str, object]]] = defaultdict(list)
    global_dt_rows_by_step_layer: Dict[Tuple[int, int], List[Dict[str, object]]] = defaultdict(list)
    for row in object_rows:
        object_rows_by_step_layer[(int(row["step"]), int(row["layer"]))].append(row)
    for row in object_dt_rows:
        object_dt_rows_by_step_layer[(int(row["step"]), int(row["layer"]))].append(row)
    for row in global_dt_rows:
        global_dt_rows_by_step_layer[(int(row["step"]), int(row["layer"]))].append(row)

    all_step_layers = sorted(
        set(object_rows_by_step_layer.keys()) | set(object_dt_rows_by_step_layer.keys()) | set(global_dt_rows_by_step_layer.keys())
    )
    for step_index, layer_index in all_step_layers:
        object_heatmap_rows = object_rows_by_step_layer.get((int(step_index), int(layer_index)), [])
        if object_heatmap_rows:
            for value_key in ("object_fraction", "object_mass"):
                plot_path = _plot_wan21_t2v_self_attention_distribution_object_heatmap(
                    rows=object_heatmap_rows,
                    save_file=os.path.join(
                        plots_dir,
                        f"step_{int(step_index):03d}",
                        f"layer_{int(layer_index):02d}",
                        f"object_query_key_heatmap_{value_key}.pdf",
                    ),
                    title=(
                        f"Object-Region Self-Attention Heatmap ({value_key}) | "
                        f"step={int(step_index)} layer={int(layer_index)}"
                    ),
                    value_key=value_key,
                )
                if plot_path:
                    plot_paths.append(plot_path)

        object_dt_plot_rows = object_dt_rows_by_step_layer.get((int(step_index), int(layer_index)), [])
        if object_dt_plot_rows:
            plot_path = _plot_wan21_t2v_self_attention_distribution_object_dt_curves(
                rows=object_dt_plot_rows,
                save_file=os.path.join(
                    plots_dir,
                    f"step_{int(step_index):03d}",
                    f"layer_{int(layer_index):02d}",
                    "object_dt_curves.pdf",
                ),
                title=f"Object-Region Self-Attention vs Signed dt | step={int(step_index)} layer={int(layer_index)}",
            )
            if plot_path:
                plot_paths.append(plot_path)

        global_dt_plot_rows = global_dt_rows_by_step_layer.get((int(step_index), int(layer_index)), [])
        if global_dt_plot_rows:
            plot_path = _plot_wan21_t2v_self_attention_distribution_global_dt_curves(
                rows=global_dt_plot_rows,
                save_file=os.path.join(
                    plots_dir,
                    f"step_{int(step_index):03d}",
                    f"layer_{int(layer_index):02d}",
                    "global_dt_curves.pdf",
                ),
                title=f"Global Self-Attention vs Signed dt | step={int(step_index)} layer={int(layer_index)}",
            )
            if plot_path:
                plot_paths.append(plot_path)

    summary = {
        "experiment": "wan21_t2v_self_attention_distribution",
        "prompt": prompt,
        "video_path": video_path,
        "object_rows_csv": object_rows_path,
        "object_dt_rows_csv": object_dt_rows_path,
        "global_dt_rows_csv": global_dt_rows_path,
        "reference_support_csv": reference_rows_path,
        "reference_map_path": reference_support["reference_map_path"],
        "target_object_words": list(target_object_words),
        "target_verb_words": list(target_verb_words),
        "self_attention_distribution_steps": list(self_attention_distribution_steps),
        "self_attention_distribution_layers": list(self_attention_distribution_layers),
        "self_attention_distribution_branch": str(self_attention_distribution_branch),
        "self_attention_distribution_reference_step": int(self_attention_distribution_reference_step),
        "self_attention_distribution_reference_layer": int(self_attention_distribution_reference_layer),
        "self_attention_distribution_reference_center_mode": str(self_attention_distribution_reference_center_mode),
        "self_attention_distribution_query_frame_count": int(self_attention_distribution_query_frame_count),
        "self_attention_distribution_global_query_tokens_per_frame": int(self_attention_distribution_global_query_tokens_per_frame),
        "self_attention_distribution_object_query_token_limit_per_frame": int(self_attention_distribution_object_query_token_limit_per_frame),
        "num_object_rows": int(len(object_rows)),
        "num_object_dt_rows": int(len(object_dt_rows)),
        "num_global_dt_rows": int(len(global_dt_rows)),
        "plot_paths": plot_paths,
    }
    _save_json(os.path.join(output_dir, "self_attention_distribution_summary.json"), summary)
    return summary
