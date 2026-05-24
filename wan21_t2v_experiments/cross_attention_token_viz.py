"""Wan2.1-T2V experiment: cross_attention_token_viz.

Main entry:
- run_wan21_t2v_cross_attention_token_viz

This module is the main cross-attention collection and visualization workflow.
It depends on the shared cross-attention visualization core in utils.py and
contains the experiment-specific orchestration logic for streaming maps,
rendering PDFs, and exporting object-token trajectories.
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
    _broadcast_seed_if_needed,
    _build_wan21_t2v_pipeline,
    _ensure_dir,
    _extract_wan21_t2v_attention_region_center_trajectory,
    _generate_wan21_t2v_video,
    _init_wan21_t2v_runtime,
    _install_wan21_t2v_cross_attention_viz_patch,
    _iter_wan21_t2v_parallel_results,
    _load_wan21_t2v_cross_attention_mean_maps_from_disk,
    _load_wan21_t2v_cross_attention_token_meta,
    _locate_wan21_t2v_prompt_words,
    _normalize_wan21_t2v_attention_map_per_frame,
    _resolve_wan21_t2v_offload_model,
    _resolve_wan21_t2v_num_workers,
    _resolve_wan21_t2v_viz_frame_indices,
    _sanitize_wan21_t2v_token_name,
    _save_csv,
    _save_json,
    _save_wan21_t2v_cross_attention_pdf,
    _save_wan21_t2v_token_trajectory_pdf,
    _save_wan21_t2v_token_trajectory_timeline_pdf,
    _save_wan21_t2v_video,
    _smooth_wan21_t2v_trajectory,
    _subsample_wan21_t2v_trajectory,
    _trajectory_stats_wan21_t2v,
)

def _resolve_wan21_t2v_cross_attention_token_viz_num_workers(
    requested_num_workers: int,
    task_count: int,
) -> int:
    """Resolve the CPU worker count used to render cross-attention PDFs and trajectories."""
    return _resolve_wan21_t2v_num_workers(
        requested_num_workers=int(requested_num_workers),
        task_count=int(task_count),
    )


def _materialize_wan21_t2v_cross_attention_token_viz_task(task: Dict[str, object]) -> Dict[str, object]:
    """Render one cross-attention head/mean bundle and return its materialized rows."""
    step = int(task["step"])
    layer = int(task["layer"])
    word = str(task["word"])
    head = task["head"]
    head_name = str(task["head_name"])
    head_sort_index = int(task["head_sort_index"])
    token_type = str(task["token_type"])
    token_positions = list(task["token_positions"])
    map_fhw: torch.Tensor = task["map_fhw"]  # type: ignore[assignment]
    frame_num = int(task["frame_num"])
    num_viz_frames = int(task["num_viz_frames"])
    viz_frame_indices = task["viz_frame_indices"]
    trajectory_enable = bool(task["trajectory_enable"])
    trajectory_style = str(task["trajectory_style"])
    trajectory_num_frames = int(task["trajectory_num_frames"])
    trajectory_smooth_radius = int(task["trajectory_smooth_radius"])
    trajectory_power = float(task["trajectory_power"])
    trajectory_quantile = float(task["trajectory_quantile"])
    trajectory_arrow_stride = int(task["trajectory_arrow_stride"])
    trajectory_timeline_num_frames = int(task["trajectory_timeline_num_frames"])
    trajectory_include_head_mean = bool(task["trajectory_include_head_mean"])
    attention_pdf_per_frame_normalize = bool(task["attention_pdf_per_frame_normalize"])
    attention_pdf_share_color_scale = bool(task["attention_pdf_share_color_scale"])
    skip_existing_pdfs = bool(task["skip_existing_pdfs"])
    plot_attention_now = bool(task["plot_attention_now"])
    plot_trajectory_now = bool(task["plot_trajectory_now"])
    plot_trajectory_timeline_now = bool(task["plot_trajectory_timeline_now"])

    token_dir = os.path.join(
        str(task["visualization_output_dir"]),
        f"timestep_{step:03d}",
        f"token_{token_type}_{_sanitize_wan21_t2v_token_name(word)}",
    )
    _ensure_dir(token_dir)

    attention_frame_indices, video_frame_labels = _resolve_wan21_t2v_viz_frame_indices(
        attention_frame_count=int(map_fhw.size(0)),
        video_frame_count=frame_num,
        num_frames=num_viz_frames,
        explicit_indices=viz_frame_indices,
    )

    row: Dict[str, object] = {
        "step": step,
        "layer": layer,
        "head": head,
        "token": word,
        "token_type": token_type,
        "token_positions": token_positions,
        "pdf_path": "",
        "frame_indices": video_frame_labels,
        "attention_frame_indices": attention_frame_indices,
    }

    if plot_attention_now:
        map_pdf_path = os.path.join(token_dir, f"layer_{layer:02d}_head_{head_name}.pdf")
        if (not skip_existing_pdfs) or (not os.path.exists(map_pdf_path)):
            _save_wan21_t2v_cross_attention_pdf(
                map_hfhw=(
                    _normalize_wan21_t2v_attention_map_per_frame(map_fhw)
                    if attention_pdf_per_frame_normalize
                    else map_fhw
                ),
                frame_indices=attention_frame_indices,
                frame_labels=video_frame_labels,
                save_file=map_pdf_path,
                title=f"step={step} layer={layer} head={head_name} token={word}",
                share_color_scale=attention_pdf_share_color_scale,
            )
        row["pdf_path"] = map_pdf_path

    trajectory_rows: List[Dict[str, object]] = []
    trajectory_path_length = None
    if trajectory_enable and token_type == "object" and (head != "mean" or trajectory_include_head_mean):
        trajectory_raw = _extract_wan21_t2v_attention_region_center_trajectory(
            map_fhw=map_fhw,
            power=trajectory_power,
            quantile=trajectory_quantile,
        )
        # Export trajectory CSV from raw per-frame centers so downstream stages use
        # frame-local centers unaffected by temporal smoothing.
        export_frame_indices, export_points = _subsample_wan21_t2v_trajectory(
            trajectory_raw,
            num_points=0,
        )
        trajectory = _smooth_wan21_t2v_trajectory(trajectory_raw, radius=trajectory_smooth_radius)
        trajectory_frame_indices, trajectory_points = _subsample_wan21_t2v_trajectory(
            trajectory,
            num_points=trajectory_num_frames,
        )
        timeline_attention_indices, timeline_video_labels = _resolve_wan21_t2v_viz_frame_indices(
            attention_frame_count=len(trajectory),
            video_frame_count=frame_num,
            num_frames=trajectory_timeline_num_frames,
            explicit_indices=None,
        )

        trajectory_pdf_path = os.path.join(token_dir, f"layer_{layer:02d}_head_{head_name}_trajectory.pdf")
        trajectory_timeline_pdf_path = os.path.join(
            token_dir,
            f"layer_{layer:02d}_head_{head_name}_trajectory_timeline.pdf",
        )
        if plot_trajectory_now:
            if (not skip_existing_pdfs) or (not os.path.exists(trajectory_pdf_path)):
                _save_wan21_t2v_token_trajectory_pdf(
                    trajectory=trajectory_points,
                    frame_indices=trajectory_frame_indices,
                    mean_map_hw=map_fhw.mean(dim=0),
                    save_file=trajectory_pdf_path,
                    title=f"step={step} layer={layer} head={head_name} token={word}",
                    style=trajectory_style,
                    arrow_stride=trajectory_arrow_stride,
                )
            row["trajectory_pdf_path"] = trajectory_pdf_path
        else:
            row["trajectory_pdf_path"] = ""

        if plot_trajectory_timeline_now:
            if (not skip_existing_pdfs) or (not os.path.exists(trajectory_timeline_pdf_path)):
                _save_wan21_t2v_token_trajectory_timeline_pdf(
                    map_fhw=map_fhw,
                    trajectory=trajectory_raw,
                    attention_frame_indices=timeline_attention_indices,
                    frame_labels=timeline_video_labels,
                    save_file=trajectory_timeline_pdf_path,
                    title=f"step={step} layer={layer} head={head_name} token={word}",
                )
            row["trajectory_timeline_pdf_path"] = trajectory_timeline_pdf_path
        else:
            row["trajectory_timeline_pdf_path"] = ""

        trajectory_stats = _trajectory_stats_wan21_t2v(trajectory_points)
        row.update(
            {
                "trajectory_path_length": trajectory_stats["path_length"],
                "trajectory_net_displacement": trajectory_stats["net_displacement"],
                "trajectory_mean_step_displacement": trajectory_stats["mean_step_displacement"],
            }
        )
        for frame_idx, (y, x) in zip(export_frame_indices, export_points):
            trajectory_rows.append(
                {
                    "step": step,
                    "layer": layer,
                    "head": head,
                    "token": word,
                    "token_type": token_type,
                    "frame": int(frame_idx),
                    "y": float(y),
                    "x": float(x),
                    "trajectory_pdf_path": row["trajectory_pdf_path"],
                }
            )
        trajectory_path_length = float(trajectory_stats["path_length"])

    return {
        "sort_key": (step, layer, word, head_sort_index),
        "summary_key": (step, layer, word),
        "row": row,
        "trajectory_rows": trajectory_rows,
        "trajectory_path_length": trajectory_path_length,
        "pdf_units": int(task["pdf_units"]),
    }

def run_wan21_t2v_cross_attention_token_viz(
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
    device_id: Optional[int] = None,
    offload_model: bool = True,
    collect_steps: Sequence[int] = (1, 2, 3),
    num_viz_frames: int = 5,
    viz_frame_indices: Optional[Sequence[int]] = None,
    layers_to_collect: Optional[Sequence[int]] = None,
    chunk_size: int = 1024,
    trajectory_enable: bool = True,
    trajectory_style: str = "glow_arrow",  # glow, arrow, glow_arrow
    trajectory_num_frames: int = 0,
    trajectory_smooth_radius: int = 2,
    trajectory_power: float = 1.5,
    trajectory_quantile: float = 0.8,
    trajectory_arrow_stride: int = 4,
    trajectory_include_head_mean: bool = True,
    save_attention_pdfs: bool = True,
    attention_pdf_per_frame_normalize: bool = False,
    attention_pdf_share_color_scale: bool = False,
    skip_existing_pdfs: bool = True,
    save_trajectory_pdfs: bool = True,
    save_trajectory_timeline_pdfs: bool = True,
    trajectory_timeline_num_frames: int = 10,
    save_video: bool = True,
    stream_flush_per_step: bool = False,
    plot_during_sampling: bool = False,
    stream_maps_dirname: str = "cross_attention_maps_stream",
    draw_attention_map_only: bool = False,
    draw_attention_maps_path: str = "",
    visualization_output_dir: str = "",
    cross_attention_token_viz_num_workers: int = 0,
    parallel_cfg: Optional[Wan21T2VParallelConfig] = None,
):
    """Visualize video-to-text cross-attention maps for user-selected prompt words.

    Output structure:
    - output_dir/timestep_xxx/token_<word>/layer_XX_head_YY.pdf

    CPU parallelism:
    - cross_attention_token_viz_num_workers: number of CPU worker processes used
      to render PDFs and extract per-head trajectories after map collection.
      A non-positive value means use os.cpu_count().
    """
    parallel_cfg = parallel_cfg or Wan21T2VParallelConfig()
    runtime = _init_wan21_t2v_runtime(parallel_cfg, explicit_device_id=device_id)
    seed = _broadcast_seed_if_needed(seed, runtime)
    visualization_output_dir = (
        os.path.abspath(str(visualization_output_dir))
        if str(visualization_output_dir).strip()
        else output_dir
    )
    artifact_output_dir = (
        visualization_output_dir
        if draw_attention_map_only and (os.path.abspath(visualization_output_dir) != os.path.abspath(output_dir))
        else output_dir
    )

    if runtime.use_usp:
        raise RuntimeError("Cross-attention token visualization currently requires use_usp=False.")

    pipeline = None
    cfg = None
    prompt_tokens: List[str] = []
    word_to_positions: Dict[str, List[int]] = {}
    word_to_type: Dict[str, str] = {}
    mean_maps_from_disk: Optional[Dict[Tuple[int, int, str], torch.Tensor]] = None
    loaded_maps_source = ""

    if draw_attention_map_only:
        mean_maps_from_disk, loaded_maps_source = _load_wan21_t2v_cross_attention_mean_maps_from_disk(
            output_dir=output_dir,
            draw_attention_maps_path=draw_attention_maps_path,
        )
        words_in_maps = sorted(set(str(k[2]) for k in mean_maps_from_disk.keys()))
        word_to_positions, word_to_type, prompt_tokens = _load_wan21_t2v_cross_attention_token_meta(
            output_dir=output_dir,
            words_in_maps=words_in_maps,
            target_object_words=target_object_words,
            target_verb_words=target_verb_words,
        )
    else:
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
            target_object_words=target_object_words,
            target_verb_words=target_verb_words,
        )

    if trajectory_enable and not any(v == "object" for v in word_to_type.values()):
        raise ValueError(
            "trajectory_enable=True requires object tokens. "
            "Provide --target_object_words, or ensure summary has token_types for draw-only mode."
        )
    if plot_during_sampling and not stream_flush_per_step and runtime.rank == 0:
        print(
            "[wan21_t2v_experiments] plot_during_sampling=True requires stream_flush_per_step=True. "
            "Falling back to end-of-run plotting."
        )

    rows: List[Dict] = []
    trajectory_rows: List[Dict] = []
    trajectory_summary_acc = defaultdict(list)
    stream_map_files: List[str] = []
    stream_maps_dir = os.path.join(output_dir, stream_maps_dirname)

    if runtime.rank == 0 and stream_flush_per_step:
        _ensure_dir(output_dir)
        _ensure_dir(stream_maps_dir)
    if runtime.rank == 0:
        _ensure_dir(visualization_output_dir)
        _ensure_dir(artifact_output_dir)

    def _accumulate_from_mean_maps(
        mean_maps: Dict[Tuple[int, int, str], torch.Tensor],
        allow_plot: bool,
    ):
        plot_attention_now = bool(allow_plot and save_attention_pdfs)
        plot_trajectory_now = bool(allow_plot and save_trajectory_pdfs)
        plot_trajectory_timeline_now = bool(allow_plot and save_trajectory_timeline_pdfs)

        tasks: List[Dict[str, object]] = []
        total_pdf_tasks = 0
        for (step, layer, word), maps in sorted(mean_maps.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
            heads = int(maps.size(0))
            token_type = word_to_type.get(word, "unknown")
            token_positions = word_to_positions.get(word, [])
            for head in range(heads):
                pdf_units = 0
                if plot_attention_now:
                    pdf_units += 1
                if trajectory_enable and token_type == "object":
                    if plot_trajectory_now:
                        pdf_units += 1
                    if plot_trajectory_timeline_now:
                        pdf_units += 1
                tasks.append(
                    {
                        "step": int(step),
                        "layer": int(layer),
                        "word": str(word),
                        "head": int(head),
                        "head_name": f"{int(head):02d}",
                        "head_sort_index": int(head),
                        "token_type": token_type,
                        "token_positions": list(token_positions),
                        "map_fhw": maps[head],
                        "frame_num": int(frame_num),
                        "num_viz_frames": int(num_viz_frames),
                        "viz_frame_indices": None
                        if not viz_frame_indices
                        else [int(i) for i in viz_frame_indices],
                        "trajectory_enable": bool(trajectory_enable),
                        "trajectory_style": trajectory_style,
                        "trajectory_num_frames": int(trajectory_num_frames),
                        "trajectory_smooth_radius": int(trajectory_smooth_radius),
                        "trajectory_power": float(trajectory_power),
                        "trajectory_quantile": float(trajectory_quantile),
                        "trajectory_arrow_stride": int(trajectory_arrow_stride),
                        "trajectory_timeline_num_frames": int(trajectory_timeline_num_frames),
                        "trajectory_include_head_mean": bool(trajectory_include_head_mean),
                        "attention_pdf_per_frame_normalize": bool(attention_pdf_per_frame_normalize),
                        "attention_pdf_share_color_scale": bool(attention_pdf_share_color_scale),
                        "skip_existing_pdfs": bool(skip_existing_pdfs),
                        "plot_attention_now": bool(plot_attention_now),
                        "plot_trajectory_now": bool(plot_trajectory_now),
                        "plot_trajectory_timeline_now": bool(plot_trajectory_timeline_now),
                        "visualization_output_dir": str(visualization_output_dir),
                        "pdf_units": int(pdf_units),
                    }
                )
                total_pdf_tasks += pdf_units

            if heads > 1:
                pdf_units = 0
                if plot_attention_now:
                    pdf_units += 1
                if trajectory_enable and token_type == "object" and trajectory_include_head_mean:
                    if plot_trajectory_now:
                        pdf_units += 1
                    if plot_trajectory_timeline_now:
                        pdf_units += 1
                tasks.append(
                    {
                        "step": int(step),
                        "layer": int(layer),
                        "word": str(word),
                        "head": "mean",
                        "head_name": "mean",
                        "head_sort_index": int(heads),
                        "token_type": token_type,
                        "token_positions": list(token_positions),
                        "map_fhw": maps.mean(dim=0),
                        "frame_num": int(frame_num),
                        "num_viz_frames": int(num_viz_frames),
                        "viz_frame_indices": None
                        if not viz_frame_indices
                        else [int(i) for i in viz_frame_indices],
                        "trajectory_enable": bool(trajectory_enable),
                        "trajectory_style": trajectory_style,
                        "trajectory_num_frames": int(trajectory_num_frames),
                        "trajectory_smooth_radius": int(trajectory_smooth_radius),
                        "trajectory_power": float(trajectory_power),
                        "trajectory_quantile": float(trajectory_quantile),
                        "trajectory_arrow_stride": int(trajectory_arrow_stride),
                        "trajectory_timeline_num_frames": int(trajectory_timeline_num_frames),
                        "trajectory_include_head_mean": bool(trajectory_include_head_mean),
                        "attention_pdf_per_frame_normalize": bool(attention_pdf_per_frame_normalize),
                        "attention_pdf_share_color_scale": bool(attention_pdf_share_color_scale),
                        "skip_existing_pdfs": bool(skip_existing_pdfs),
                        "plot_attention_now": bool(plot_attention_now),
                        "plot_trajectory_now": bool(plot_trajectory_now),
                        "plot_trajectory_timeline_now": bool(plot_trajectory_timeline_now),
                        "visualization_output_dir": str(visualization_output_dir),
                        "pdf_units": int(pdf_units),
                    }
                )
                total_pdf_tasks += pdf_units

        pbar = None
        if total_pdf_tasks > 0:
            from tqdm import tqdm
            pbar = tqdm(
                total=total_pdf_tasks,
                desc="cross_attention_token_viz plotting",
                unit="pdf",
                leave=True,
            )

        try:
            if tasks:
                num_workers = _resolve_wan21_t2v_cross_attention_token_viz_num_workers(
                    requested_num_workers=int(cross_attention_token_viz_num_workers),
                    task_count=int(len(tasks)),
                )
                task_results_by_key: Dict[Tuple[int, int, str, int], Dict[str, object]] = {}
                for result in _iter_wan21_t2v_parallel_results(
                    tasks=tasks,
                    worker_fn=_materialize_wan21_t2v_cross_attention_token_viz_task,
                    num_workers=int(num_workers),
                ):
                    task_results_by_key[result["sort_key"]] = result
                    if pbar is not None:
                        pbar.update(int(result["pdf_units"]))

                for sort_key in sorted(task_results_by_key.keys()):
                    result = task_results_by_key[sort_key]
                    rows.append(result["row"])
                    trajectory_rows.extend(result["trajectory_rows"])
                    trajectory_path_length = result["trajectory_path_length"]
                    if trajectory_path_length is not None:
                        trajectory_summary_acc[result["summary_key"]].append(float(trajectory_path_length))
        finally:
            if pbar is not None:
                pbar.close()

    def _save_step_maps_to_stream(step_maps: Dict[Tuple[int, int, str], torch.Tensor]) -> Optional[str]:
        if runtime.rank != 0 or not step_maps:
            return None
        step_id = int(next(iter(step_maps.keys()))[0])
        step_file = os.path.join(stream_maps_dir, f"cross_attention_maps_step_{step_id:03d}.pt")
        torch.save(step_maps, step_file)
        stream_map_files.append(step_file)
        return step_file

    state = None
    video = None
    if not draw_attention_map_only:
        state = Wan21T2VCrossAttentionVizState(
            token_positions=word_to_positions,
            collect_steps=collect_steps,
            num_layers=len(pipeline.model.blocks),
            num_heads=pipeline.model.num_heads,
            chunk_size=chunk_size,
            layers_to_collect=layers_to_collect,
        )

        def _on_step_complete(completed_step: int):
            if not stream_flush_per_step:
                return
            step_maps = state.export_step_mean_maps(completed_step, clear_after_export=True)
            _save_step_maps_to_stream(step_maps)
            if runtime.rank == 0 and plot_during_sampling and step_maps:
                _accumulate_from_mean_maps(step_maps, allow_plot=True)

        handle = _install_wan21_t2v_cross_attention_viz_patch(
            pipeline.model,
            state,
            on_step_complete=_on_step_complete,
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
            final_step_maps = state.export_step_mean_maps(state.current_step, clear_after_export=True)
            _save_step_maps_to_stream(final_step_maps)
            if runtime.rank == 0 and plot_during_sampling and final_step_maps:
                _accumulate_from_mean_maps(final_step_maps, allow_plot=True)

    if dist.is_initialized():
        dist.barrier()

    if runtime.rank == 0:
        _ensure_dir(artifact_output_dir)

        video_path = ""
        if not draw_attention_map_only:
            video_path = os.path.join(artifact_output_dir, f"wan21_t2v_cross_attention_token_viz_seed_{seed}.mp4")
            if save_video:
                _save_wan21_t2v_video(video, video_path, fps=cfg.sample_fps)
            else:
                video_path = ""

            if stream_flush_per_step:
                if not plot_during_sampling:
                    for step_file in sorted(stream_map_files):
                        step_maps = torch.load(step_file, map_location="cpu")
                        _accumulate_from_mean_maps(step_maps, allow_plot=True)
            else:
                mean_maps = state.export_mean_maps()
                torch.save(mean_maps, os.path.join(output_dir, "cross_attention_maps.pt"))
                _accumulate_from_mean_maps(mean_maps, allow_plot=True)
        else:
            _accumulate_from_mean_maps(mean_maps_from_disk, allow_plot=True)

        _save_csv(os.path.join(artifact_output_dir, "cross_attention_token_viz_index.csv"), rows)
        if trajectory_enable:
            _save_csv(os.path.join(artifact_output_dir, "cross_attention_token_trajectory.csv"), trajectory_rows)

        trajectory_summary_rows = []
        for (step, layer, word), vals in sorted(trajectory_summary_acc.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
            if not vals:
                continue
            v = torch.tensor(vals, dtype=torch.float32)
            trajectory_summary_rows.append(
                {
                    "step": step,
                    "layer": layer,
                    "token": word,
                    "num_trajectories": int(v.numel()),
                    "path_length_mean": float(v.mean().item()),
                    "path_length_std": float(v.std(unbiased=False).item()),
                }
            )
        if trajectory_enable:
            _save_csv(os.path.join(artifact_output_dir, "cross_attention_token_trajectory_summary.csv"), trajectory_summary_rows)

        summary = {
            "experiment": "wan21_t2v_cross_attention_token_viz",
            "prompt": prompt,
            "frame_num": int(frame_num),
            "prompt_tokens": prompt_tokens,
            "token_positions": word_to_positions,
            "token_types": word_to_type,
            "target_object_words": list(target_object_words),
            "target_verb_words": list(target_verb_words),
            "collect_steps": list(collect_steps),
            "num_viz_frames": int(num_viz_frames),
            "viz_frame_indices": None if not viz_frame_indices else [int(i) for i in viz_frame_indices],
            "layers_to_collect": None if layers_to_collect is None else list(layers_to_collect),
            "chunk_size": int(chunk_size),
            "video_path": video_path,
            "num_maps": len(rows),
            "trajectory_enable": bool(trajectory_enable),
            "trajectory_style": trajectory_style,
            "trajectory_num_frames": int(trajectory_num_frames),
            "trajectory_smooth_radius": int(trajectory_smooth_radius),
            "trajectory_power": float(trajectory_power),
            "trajectory_quantile": float(trajectory_quantile),
            "trajectory_arrow_stride": int(trajectory_arrow_stride),
            "trajectory_include_head_mean": bool(trajectory_include_head_mean),
            "trajectory_csv_point_mode": "raw_region_center_per_frame",
            "save_trajectory_timeline_pdfs": bool(save_trajectory_timeline_pdfs),
            "trajectory_timeline_num_frames": int(trajectory_timeline_num_frames),
            "trajectory_num_points_rows": len(trajectory_rows),
            "save_attention_pdfs": bool(save_attention_pdfs),
            "attention_pdf_per_frame_normalize": bool(attention_pdf_per_frame_normalize),
            "attention_pdf_share_color_scale": bool(attention_pdf_share_color_scale),
            "skip_existing_pdfs": bool(skip_existing_pdfs),
            "save_trajectory_pdfs": bool(save_trajectory_pdfs),
            "save_video": bool(save_video),
            "draw_attention_map_only": bool(draw_attention_map_only),
            "draw_attention_maps_path": str(draw_attention_maps_path),
            "loaded_maps_source": loaded_maps_source,
            "visualization_output_dir": visualization_output_dir,
            "artifact_output_dir": artifact_output_dir,
            "cross_attention_token_viz_num_workers": int(cross_attention_token_viz_num_workers),
            "stream_flush_per_step": bool(stream_flush_per_step),
            "plot_during_sampling": bool(plot_during_sampling),
            "stream_maps_dir": stream_maps_dir if stream_flush_per_step else "",
            "stream_map_files_count": len(stream_map_files),
        }
        if stream_flush_per_step and not draw_attention_map_only:
            _save_json(
                os.path.join(output_dir, "cross_attention_stream_index.json"),
                {"stream_maps_dir": stream_maps_dir, "step_map_files": sorted(stream_map_files)},
            )
        _save_json(os.path.join(artifact_output_dir, "cross_attention_token_viz_summary.json"), summary)
        return summary
    return None
