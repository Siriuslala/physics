"""Wan2.1-T2V experiment: self_attention_viz.

Main entry:
- run_wan21_t2v_self_attention_viz

This experiment visualizes object-guided self-attention maps by reusing a
reference object region extracted from cross-attention head-mean maps. It keeps
Wan's original diffusion forward path unchanged: the experiment only reads the
patched q/k tensors and computes a side-channel sampled probe over selected
object-region query tokens.
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from .head_evolution import (
    _build_wan21_t2v_trajectory_support_mask_from_centers,
    _extract_wan21_t2v_reference_peak_and_centroid_trajectory,
    _preprocess_wan21_t2v_attention_map_fhw,
)
from .utils import (
    Wan21T2VParallelConfig,
    _broadcast_seed_if_needed,
    _build_wan21_t2v_pipeline,
    _ensure_dir,
    _init_wan21_t2v_runtime,
    _iter_wan21_t2v_parallel_results,
    _load_wan21_t2v_cross_attention_mean_maps_from_disk,
    _load_wan21_t2v_cross_attention_token_meta,
    _mean_wan21_t2v_headmean_map_for_words,
    _resolve_wan21_t2v_num_workers,
    _resolve_wan21_t2v_offload_model,
    _resolve_wan21_t2v_viz_frame_indices,
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


def _resolve_wan21_t2v_self_attention_viz_num_workers(
    requested_num_workers: int,
    task_count: int,
) -> int:
    """Resolve the CPU worker count used to render self-attention visualization PDFs."""
    return _resolve_wan21_t2v_num_workers(
        requested_num_workers=int(requested_num_workers),
        task_count=int(task_count),
    )


def _load_wan21_t2v_self_attention_viz_maps_from_disk(
    output_dir: str,
    draw_self_attention_maps_path: str = "",
) -> Tuple[Dict[Tuple[int, int, int], torch.Tensor], str]:
    """Load stored self_attention_viz maps from one `.pt` file."""
    map_path = str(draw_self_attention_maps_path).strip() if draw_self_attention_maps_path else ""
    if map_path:
        map_path = os.path.abspath(map_path)
    else:
        map_path = os.path.join(output_dir, "self_attention_viz_maps.pt")

    if not os.path.exists(map_path):
        raise FileNotFoundError(f"Cannot find self_attention_viz maps: {map_path}")

    loaded = torch.load(map_path, map_location="cpu")
    if not isinstance(loaded, dict):
        raise ValueError(f"Invalid self_attention_viz map file format: {map_path}")
    return loaded, map_path


def _build_wan21_t2v_self_attention_viz_reference_support(
    reuse_cross_attention_dir: str,
    target_object_words: Sequence[str],
    target_verb_words: Sequence[str],
    reference_step: int,
    reference_layer: int,
    center_mode: str,
    center_power: float,
    center_quantile: float,
    preprocess_winsorize_quantile: float,
    preprocess_despike_quantile: float,
    preprocess_min_component_area: int,
    support_radius_mode: str,
    support_radius_fixed: float,
    support_radius_alpha: float,
    support_radius_min: float,
    support_radius_max_ratio: float,
) -> Dict[str, object]:
    """Build the frame-wise object support mask reused by self_attention_viz."""
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
            "Cannot build self_attention_viz reference support because the requested reference map is missing. "
            f"step={int(reference_step)} layer={int(reference_layer)} words={object_words_in_maps}"
        )

    reference_preprocessed_map, reference_preprocess_stats = _preprocess_wan21_t2v_attention_map_fhw(
        map_fhw=reference_map,
        winsorize_quantile=float(preprocess_winsorize_quantile),
        despike_quantile=float(preprocess_despike_quantile),
        min_component_area=int(preprocess_min_component_area),
    )
    reference_trajectory_data = _extract_wan21_t2v_reference_peak_and_centroid_trajectory(
        map_fhw=reference_preprocessed_map,
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
            "self_attention_viz_reference_center_mode must be one of "
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
        "reference_preprocessed_map": reference_preprocessed_map,
        "reference_preprocess_stats": reference_preprocess_stats,
        "center_trajectory": center_trajectory,
        "support_mask_fhw": support_mask_fhw,
        "support_radius_per_frame": support_radius_per_frame,
    }


def _save_wan21_t2v_self_attention_viz_pdf(
    map_fhw: torch.Tensor,
    frame_indices: Sequence[int],
    frame_labels: Sequence[int],
    save_file: str,
    title: str,
    share_color_scale: bool,
):
    """Save one self-attention timeline PDF with `viridis` colormap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid_indices = [int(frame_idx) for frame_idx in frame_indices if 0 <= int(frame_idx) < int(map_fhw.size(0))]
    if not valid_indices:
        return ""

    shared_vmin = None
    shared_vmax = None
    if bool(share_color_scale):
        stacked_frames = torch.stack([map_fhw[frame_idx].detach().float().cpu() for frame_idx in valid_indices], dim=0)
        shared_vmin = float(stacked_frames.min().item())
        shared_vmax = float(stacked_frames.max().item())
        if abs(shared_vmax - shared_vmin) < 1e-12:
            shared_vmax = shared_vmin + 1e-12

    num_panels = len(valid_indices)
    fig = plt.figure(figsize=(2.25 * num_panels, 2.8))
    gs = fig.add_gridspec(1, num_panels, wspace=0.01, hspace=0.0)
    axes = [fig.add_subplot(gs[0, panel_idx]) for panel_idx in range(num_panels)]

    token_grid_height = int(map_fhw.size(-2))
    token_grid_width = int(map_fhw.size(-1))
    for panel_idx, frame_idx in enumerate(valid_indices):
        axis = axes[panel_idx]
        axis.imshow(
            map_fhw[frame_idx].detach().float().cpu().numpy(),
            cmap="viridis",
            alpha=0.95,
            vmin=shared_vmin,
            vmax=shared_vmax,
        )
        label = int(frame_labels[panel_idx]) if panel_idx < len(frame_labels) else int(frame_idx)
        axis.set_title(f"frame={label}", fontsize=9)
        axis.set_xlim(-0.5, token_grid_width - 0.5)
        axis.set_ylim(token_grid_height - 0.5, -0.5)
        axis.axis("off")

    fig.suptitle(title, fontsize=10, y=0.97)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.88, bottom=0.01, wspace=0.01, hspace=0.0)
    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file


def _materialize_wan21_t2v_self_attention_viz_map_task(task: Dict[str, object]) -> Dict[str, object]:
    """Render one self-attention visualization bundle and return one index row."""
    step = int(task["step"])
    layer = int(task["layer"])
    head = task["head"]
    head_name = str(task["head_name"])
    query_token_frame = int(task["query_token_frame"])
    query_video_frame = int(task["query_video_frame"])
    map_fhw: torch.Tensor = task["map_fhw"]  # type: ignore[assignment]
    save_attention_pdfs = bool(task["save_attention_pdfs"])
    skip_existing_pdfs = bool(task["skip_existing_pdfs"])
    attention_pdf_share_color_scale = bool(task["attention_pdf_share_color_scale"])

    layer_dir = os.path.join(
        str(task["visualization_output_dir"]),
        f"step_{step:03d}",
        f"layer_{layer:02d}",
        f"query_frame_{query_video_frame:03d}",
    )
    _ensure_dir(layer_dir)

    attention_frame_indices, video_frame_labels = _resolve_wan21_t2v_viz_frame_indices(
        attention_frame_count=int(map_fhw.size(0)),
        video_frame_count=int(task["frame_num"]),
        num_frames=int(task["num_viz_frames"]),
        explicit_indices=task["viz_frame_indices"],
    )

    row: Dict[str, object] = {
        "step": int(step),
        "layer": int(layer),
        "head": head,
        "query_token_frame": int(query_token_frame),
        "query_video_frame": int(query_video_frame),
        "frame_indices": video_frame_labels,
        "attention_frame_indices": attention_frame_indices,
        "pdf_path": "",
    }

    if save_attention_pdfs:
        pdf_path = os.path.join(layer_dir, f"head_{head_name}.pdf")
        if (not skip_existing_pdfs) or (not os.path.exists(pdf_path)):
            _save_wan21_t2v_self_attention_viz_pdf(
                map_fhw=map_fhw,
                frame_indices=attention_frame_indices,
                frame_labels=video_frame_labels,
                save_file=pdf_path,
                title=(
                    f"step={step} layer={layer} head={head_name} "
                    f"query_video_frame={query_video_frame}"
                ),
                share_color_scale=attention_pdf_share_color_scale,
            )
        row["pdf_path"] = pdf_path

    return row


def _build_wan21_t2v_self_attention_viz_task_rows(
    attention_maps: Dict[Tuple[int, int, int], torch.Tensor],
    map_rows_by_key: Dict[Tuple[int, int, int], Dict[str, object]],
    video_frame_count: int,
    frame_num: int,
    num_viz_frames: int,
    viz_frame_indices: Optional[Sequence[int]],
    save_attention_pdfs: bool,
    skip_existing_pdfs: bool,
    attention_pdf_share_color_scale: bool,
    visualization_output_dir: str,
) -> List[Dict[str, object]]:
    """Build materialization tasks from saved self-attention visualization maps."""
    tasks: List[Dict[str, object]] = []
    for (step, layer, query_token_frame), map_hfhw in sorted(attention_maps.items(), key=lambda item: item[0]):
        row_meta = map_rows_by_key.get((int(step), int(layer), int(query_token_frame)), {})
        query_video_frame = int(row_meta.get("query_video_frame", 0))
        if query_video_frame <= 0:
            if int(video_frame_count) <= 1 or int(map_hfhw.size(1)) <= 1:
                query_video_frame = int(query_token_frame) + 1
            else:
                query_video_frame = round(
                    float(int(query_token_frame)) * float(int(video_frame_count) - 1) / float(int(map_hfhw.size(1)) - 1)
                ) + 1
        per_head_count = int(map_hfhw.size(0))

        for head_index in range(per_head_count):
            tasks.append(
                {
                    "step": int(step),
                    "layer": int(layer),
                    "head": int(head_index),
                    "head_name": f"{int(head_index):02d}",
                    "query_token_frame": int(query_token_frame),
                    "query_video_frame": int(query_video_frame),
                    "map_fhw": map_hfhw[head_index],
                    "frame_num": int(frame_num),
                    "num_viz_frames": int(num_viz_frames),
                    "viz_frame_indices": None if not viz_frame_indices else [int(i) for i in viz_frame_indices],
                    "save_attention_pdfs": bool(save_attention_pdfs),
                    "skip_existing_pdfs": bool(skip_existing_pdfs),
                    "attention_pdf_share_color_scale": bool(attention_pdf_share_color_scale),
                    "visualization_output_dir": str(visualization_output_dir),
                }
            )

        tasks.append(
            {
                "step": int(step),
                "layer": int(layer),
                "head": "mean",
                "head_name": "mean",
                "query_token_frame": int(query_token_frame),
                "query_video_frame": int(query_video_frame),
                "map_fhw": map_hfhw.mean(dim=0),
                "frame_num": int(frame_num),
                "num_viz_frames": int(num_viz_frames),
                "viz_frame_indices": None if not viz_frame_indices else [int(i) for i in viz_frame_indices],
                "save_attention_pdfs": bool(save_attention_pdfs),
                "skip_existing_pdfs": bool(skip_existing_pdfs),
                "attention_pdf_share_color_scale": bool(attention_pdf_share_color_scale),
                "visualization_output_dir": str(visualization_output_dir),
            }
        )
    return tasks


def _render_wan21_t2v_self_attention_viz_tasks(
    tasks: Sequence[Dict[str, object]],
    num_workers: int,
) -> List[Dict[str, object]]:
    """Render self_attention_viz tasks with tqdm progress and optional multi-process execution."""
    if not tasks:
        return []

    from tqdm import tqdm

    results_by_key: Dict[Tuple[int, int, str, int], Dict[str, object]] = {}
    progress_bar = tqdm(
        total=int(len(tasks)),
        desc="self_attention_viz plotting",
        unit="pdf",
        leave=True,
    )
    try:
        for row in _iter_wan21_t2v_parallel_results(
            tasks=tasks,
            worker_fn=_materialize_wan21_t2v_self_attention_viz_map_task,
            num_workers=int(num_workers),
        ):
            sort_head = int(row["head"]) if row["head"] != "mean" else 10**9
            sort_key = (
                int(row["step"]),
                int(row["layer"]),
                int(row["query_video_frame"]),
                int(sort_head),
            )
            results_by_key[sort_key] = row
            progress_bar.update(1)
    finally:
        progress_bar.close()

    return [results_by_key[key] for key in sorted(results_by_key.keys())]


def run_wan21_t2v_self_attention_viz(
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
    self_attention_viz_steps: Sequence[int] = (1, 2, 3),
    self_attention_viz_layers: Sequence[int] = tuple(),
    self_attention_viz_branch: str = "cond",
    self_attention_viz_reference_step: int = 50,
    self_attention_viz_reference_layer: int = 27,
    self_attention_viz_reference_center_mode: str = "geometric_center",
    self_attention_viz_reference_center_power: float = 1.5,
    self_attention_viz_reference_center_quantile: float = 0.8,
    self_attention_viz_reference_preprocess_winsorize_quantile: float = 0.995,
    self_attention_viz_reference_preprocess_despike_quantile: float = 0.98,
    self_attention_viz_reference_preprocess_min_component_area: int = 2,
    self_attention_viz_support_radius_mode: str = "adaptive_area",
    self_attention_viz_support_radius_fixed: float = 2.0,
    self_attention_viz_support_radius_alpha: float = 1.5,
    self_attention_viz_support_radius_min: float = 1.0,
    self_attention_viz_support_radius_max_ratio: float = 0.25,
    self_attention_viz_query_video_frame_indices: Sequence[int] = (1, 33, 41, 81),
    self_attention_viz_object_query_token_limit_per_frame: int = 64,
    self_attention_viz_num_viz_frames: int = 10,
    self_attention_viz_viz_frame_indices: Optional[Sequence[int]] = None,
    self_attention_viz_save_attention_pdfs: bool = True,
    self_attention_viz_attention_pdf_share_color_scale: bool = False,
    self_attention_viz_skip_existing_pdfs: bool = True,
    self_attention_viz_stop_after_last_probe_step: bool = False,
    draw_self_attention_maps_only: bool = False,
    draw_self_attention_maps_path: str = "",
    self_attention_viz_visualization_output_dir: str = "",
    self_attention_viz_num_workers: int = 0,
    save_video: bool = True,
    parallel_cfg: Optional[Wan21T2VParallelConfig] = None,
):
    """Run object-guided self-attention visualization with side-channel q/k probing."""
    if not reuse_cross_attention_dir:
        raise ValueError("self_attention_viz requires reuse_cross_attention_dir.")
    if not target_object_words:
        raise ValueError("self_attention_viz requires target_object_words.")

    if self_attention_viz_steps:
        resolved_steps = sorted(set(int(step) for step in self_attention_viz_steps))
    else:
        resolved_steps = list(range(1, int(sampling_steps) + 1))
    if not resolved_steps:
        raise ValueError("self_attention_viz resolved to an empty step list.")

    if self_attention_viz_layers:
        resolved_layers = sorted(set(int(layer) for layer in self_attention_viz_layers))
    else:
        resolved_layers = []

    reference_support = _build_wan21_t2v_self_attention_viz_reference_support(
        reuse_cross_attention_dir=reuse_cross_attention_dir,
        target_object_words=target_object_words,
        target_verb_words=target_verb_words,
        reference_step=int(self_attention_viz_reference_step),
        reference_layer=int(self_attention_viz_reference_layer),
        center_mode=str(self_attention_viz_reference_center_mode),
        center_power=float(self_attention_viz_reference_center_power),
        center_quantile=float(self_attention_viz_reference_center_quantile),
        preprocess_winsorize_quantile=float(self_attention_viz_reference_preprocess_winsorize_quantile),
        preprocess_despike_quantile=float(self_attention_viz_reference_preprocess_despike_quantile),
        preprocess_min_component_area=int(self_attention_viz_reference_preprocess_min_component_area),
        support_radius_mode=str(self_attention_viz_support_radius_mode),
        support_radius_fixed=float(self_attention_viz_support_radius_fixed),
        support_radius_alpha=float(self_attention_viz_support_radius_alpha),
        support_radius_min=float(self_attention_viz_support_radius_min),
        support_radius_max_ratio=float(self_attention_viz_support_radius_max_ratio),
    )

    parallel_cfg = parallel_cfg or Wan21T2VParallelConfig()
    runtime = _init_wan21_t2v_runtime(parallel_cfg, explicit_device_id=device_id)
    seed = _broadcast_seed_if_needed(seed, runtime)

    pipeline = None
    cfg = None
    state = None
    video = None
    loaded_attention_map_path = ""
    attention_maps: Dict[Tuple[int, int, int], torch.Tensor]

    if not draw_self_attention_maps_only:
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
                probe_steps=tuple(resolved_steps),
                probe_branch=str(self_attention_viz_branch),
                collect_dt_histograms=False,
                collect_maas_maps=False,
                collect_distribution=False,
                collect_self_attention_viz=True,
                self_attention_viz_layers=tuple(int(layer_id) for layer_id in resolved_layers),
                self_attention_viz_query_video_frame_indices=tuple(
                    int(frame_id) for frame_id in self_attention_viz_query_video_frame_indices
                ),
                self_attention_viz_video_frame_count=int(frame_num),
                self_attention_viz_object_query_token_limit_per_frame=int(
                    self_attention_viz_object_query_token_limit_per_frame
                ),
                self_attention_viz_object_support_mask=reference_support["support_mask_fhw"],
                stop_after_last_probe_step=bool(self_attention_viz_stop_after_last_probe_step),
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
        attention_maps = state.export_self_attention_viz_maps()
    else:
        attention_maps, loaded_attention_map_path = _load_wan21_t2v_self_attention_viz_maps_from_disk(
            output_dir=output_dir,
            draw_self_attention_maps_path=draw_self_attention_maps_path,
        )

    if dist.is_initialized():
        dist.barrier()

    if runtime.rank != 0:
        return None

    _ensure_dir(output_dir)
    visualization_output_dir = (
        str(self_attention_viz_visualization_output_dir).strip()
        if str(self_attention_viz_visualization_output_dir).strip()
        else os.path.join(output_dir, "self_attention_viz")
    )
    _ensure_dir(visualization_output_dir)

    early_stop_triggered = bool(getattr(state, "early_stop_triggered", False)) if state is not None else False
    video_path = ""
    if (not draw_self_attention_maps_only) and bool(save_video) and (not early_stop_triggered) and video is not None:
        video_path = os.path.join(output_dir, f"wan21_t2v_self_attention_viz_seed_{seed}.mp4")
        _save_wan21_t2v_video(video, video_path, fps=cfg.sample_fps)

    attention_map_path = os.path.join(output_dir, "self_attention_viz_maps.pt")
    if not draw_self_attention_maps_only:
        torch.save(attention_maps, attention_map_path)
    else:
        attention_map_path = loaded_attention_map_path or attention_map_path

    map_rows = state.export_self_attention_viz_rows() if state is not None else []
    map_rows_by_key = {
        (int(row["step"]), int(row["layer"]), int(row["query_token_frame"])): row
        for row in map_rows
    }

    tasks = _build_wan21_t2v_self_attention_viz_task_rows(
        attention_maps=attention_maps,
        map_rows_by_key=map_rows_by_key,
        video_frame_count=int(frame_num),
        frame_num=int(frame_num),
        num_viz_frames=int(self_attention_viz_num_viz_frames),
        viz_frame_indices=self_attention_viz_viz_frame_indices,
        save_attention_pdfs=bool(self_attention_viz_save_attention_pdfs),
        skip_existing_pdfs=bool(self_attention_viz_skip_existing_pdfs),
        attention_pdf_share_color_scale=bool(self_attention_viz_attention_pdf_share_color_scale),
        visualization_output_dir=str(visualization_output_dir),
    )
    num_workers = _resolve_wan21_t2v_self_attention_viz_num_workers(
        requested_num_workers=int(self_attention_viz_num_workers),
        task_count=int(len(tasks)),
    )
    index_rows = _render_wan21_t2v_self_attention_viz_tasks(
        tasks=tasks,
        num_workers=int(num_workers),
    )

    index_csv_path = os.path.join(output_dir, "self_attention_viz_index.csv")
    _save_csv(index_csv_path, index_rows)

    reference_rows = []
    for frame_index, (center_y, center_x) in enumerate(reference_support["center_trajectory"]):
        reference_rows.append(
            {
                "frame": int(frame_index),
                "center_y": float(center_y),
                "center_x": float(center_x),
                "support_radius": float(reference_support["support_radius_per_frame"][frame_index]),
                "support_area": float(reference_support["support_mask_fhw"][frame_index].sum().item()),
            }
        )
    reference_rows_path = os.path.join(output_dir, "self_attention_viz_reference_support.csv")
    if not draw_self_attention_maps_only:
        _save_csv(reference_rows_path, reference_rows)

    summary = {
        "experiment": "wan21_t2v_self_attention_viz",
        "prompt": prompt,
        "video_path": video_path,
        "attention_map_path": attention_map_path,
        "index_csv": index_csv_path,
        "reference_support_csv": reference_rows_path,
        "reference_map_path": reference_support["reference_map_path"],
        "target_object_words": list(target_object_words),
        "target_verb_words": list(target_verb_words),
        "self_attention_viz_steps_input": [int(step) for step in self_attention_viz_steps],
        "self_attention_viz_steps_resolved": [int(step) for step in resolved_steps],
        "self_attention_viz_layers_input": [int(layer) for layer in self_attention_viz_layers],
        "self_attention_viz_layers_resolved": [int(layer) for layer in resolved_layers],
        "self_attention_viz_branch": str(self_attention_viz_branch),
        "self_attention_viz_reference_step": int(self_attention_viz_reference_step),
        "self_attention_viz_reference_layer": int(self_attention_viz_reference_layer),
        "self_attention_viz_reference_center_mode": str(self_attention_viz_reference_center_mode),
        "self_attention_viz_reference_center_power": float(self_attention_viz_reference_center_power),
        "self_attention_viz_reference_center_quantile": float(self_attention_viz_reference_center_quantile),
        "self_attention_viz_reference_preprocess_winsorize_quantile": float(
            self_attention_viz_reference_preprocess_winsorize_quantile
        ),
        "self_attention_viz_reference_preprocess_despike_quantile": float(
            self_attention_viz_reference_preprocess_despike_quantile
        ),
        "self_attention_viz_reference_preprocess_min_component_area": int(
            self_attention_viz_reference_preprocess_min_component_area
        ),
        "self_attention_viz_support_radius_mode": str(self_attention_viz_support_radius_mode),
        "self_attention_viz_support_radius_fixed": float(self_attention_viz_support_radius_fixed),
        "self_attention_viz_support_radius_alpha": float(self_attention_viz_support_radius_alpha),
        "self_attention_viz_support_radius_min": float(self_attention_viz_support_radius_min),
        "self_attention_viz_support_radius_max_ratio": float(self_attention_viz_support_radius_max_ratio),
        "self_attention_viz_query_video_frame_indices": [
            int(frame_id) for frame_id in self_attention_viz_query_video_frame_indices
        ],
        "self_attention_viz_object_query_token_limit_per_frame": int(
            self_attention_viz_object_query_token_limit_per_frame
        ),
        "self_attention_viz_num_viz_frames": int(self_attention_viz_num_viz_frames),
        "self_attention_viz_viz_frame_indices": (
            None if self_attention_viz_viz_frame_indices is None
            else [int(frame_id) for frame_id in self_attention_viz_viz_frame_indices]
        ),
        "self_attention_viz_save_attention_pdfs": bool(self_attention_viz_save_attention_pdfs),
        "self_attention_viz_attention_pdf_share_color_scale": bool(
            self_attention_viz_attention_pdf_share_color_scale
        ),
        "self_attention_viz_skip_existing_pdfs": bool(self_attention_viz_skip_existing_pdfs),
        "self_attention_viz_stop_after_last_probe_step": bool(self_attention_viz_stop_after_last_probe_step),
        "draw_self_attention_maps_only": bool(draw_self_attention_maps_only),
        "draw_self_attention_maps_path": str(draw_self_attention_maps_path),
        "self_attention_viz_visualization_output_dir": str(visualization_output_dir),
        "self_attention_viz_num_workers": int(self_attention_viz_num_workers),
        "reference_preprocess_stats": reference_support["reference_preprocess_stats"],
        "early_stop_triggered": bool(early_stop_triggered),
        "early_stop_completed_step": int(getattr(state, "early_stop_completed_step", 0)) if state is not None else 0,
        "early_stop_reason": str(getattr(state, "early_stop_reason", "")) if state is not None else "",
        "num_map_groups": int(len(attention_maps)),
        "num_index_rows": int(len(index_rows)),
    }
    _save_json(os.path.join(output_dir, "self_attention_viz_summary.json"), summary)
    return summary
