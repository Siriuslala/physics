"""Simple RoPE decay visualization for Wan2.1 T2V."""

import os
from typing import Optional, Tuple

import torch

from .utils import _ensure_dir, _save_json


def _wan21_t2v_rope_angular_frequencies(real_dim: int, theta: float = 10000.0) -> torch.Tensor:
    """Return RoPE angular frequencies used by Wan's rope_params helper."""
    if int(real_dim) <= 0 or int(real_dim) % 2 != 0:
        raise ValueError(f"RoPE real_dim must be a positive even integer, got {real_dim}.")
    return 1.0 / torch.pow(
        torch.tensor(float(theta), dtype=torch.float64),
        torch.arange(0, int(real_dim), 2, dtype=torch.float64) / float(real_dim),
    )


def _wan21_t2v_mean_rope_cosine_kernel(delta: int, angular_frequencies: torch.Tensor) -> float:
    """Return the mean cosine kernel over one RoPE frequency bank."""
    if int(delta) == 0:
        return 1.0
    return float(torch.cos(float(delta) * angular_frequencies).mean().item())


def _wan21_t2v_full_rope_kernel(
    delta_f: int,
    delta_h: int,
    delta_w: int,
    temporal_frequencies: torch.Tensor,
    height_frequencies: torch.Tensor,
    width_frequencies: torch.Tensor,
) -> float:
    """Return the mean RoPE-only cosine kernel over Wan's full head subspace."""
    temporal_term = torch.cos(float(delta_f) * temporal_frequencies)
    height_term = torch.cos(float(delta_h) * height_frequencies)
    width_term = torch.cos(float(delta_w) * width_frequencies)
    return float(torch.cat([temporal_term, height_term, width_term], dim=0).mean().item())


def _plot_wan21_t2v_rope_decay_curve(
    x_values,
    series,
    save_file: str,
    title: str,
    x_label: str,
    y_label: str,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(1, 1, figsize=(8.2, 4.8))
    color_cycle = ["#2563eb", "#dc2626", "#16a34a", "#d97706"]
    for color_index, (label, y_values) in enumerate(series):
        axis.plot(
            x_values,
            y_values,
            linewidth=1.9,
            color=color_cycle[color_index % len(color_cycle)],
            label=label,
        )
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(alpha=0.22, linestyle="--")
    if len(series) > 1:
        axis.legend(fontsize=8)
    fig.tight_layout()
    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)
    return save_file


def run_wan21_t2v_rope_decay_curve(
    wan21_root: str,
    ckpt_dir: str,
    output_dir: str,
    prompt: str,
    size: Tuple[int, int],
    task: str = "t2v-1.3B",
    frame_num: int = 81,
    shift: float = 8.0,
    sample_solver: str = "unipc",
    sampling_steps: int = 50,
    guide_scale: float = 12.0,
    seed: int = 0,
    device_id: Optional[int] = None,
    offload_model: bool = True,
    parallel_cfg=None,
):
    """Visualize Wan2.1 RoPE decay without loading the model."""
    del wan21_root, ckpt_dir, prompt, shift, sample_solver, sampling_steps, guide_scale, seed, device_id, offload_model, parallel_cfg

    from projects.Wan2_1 import wan

    cfg = wan.configs.WAN_CONFIGS[str(task)]
    head_dim = int(cfg.dim) // int(cfg.num_heads)
    temporal_real_dim = int(head_dim - 4 * (head_dim // 6))
    spatial_real_dim = int(2 * (head_dim // 6))

    temporal_frequencies = _wan21_t2v_rope_angular_frequencies(temporal_real_dim)
    height_frequencies = _wan21_t2v_rope_angular_frequencies(spatial_real_dim)
    width_frequencies = _wan21_t2v_rope_angular_frequencies(spatial_real_dim)

    latent_frames = (int(frame_num) - 1) // int(cfg.vae_stride[0]) + 1
    latent_height = int(size[1]) // int(cfg.vae_stride[1])
    latent_width = int(size[0]) // int(cfg.vae_stride[2])
    token_grid_height = latent_height // int(cfg.patch_size[1])
    token_grid_width = latent_width // int(cfg.patch_size[2])
    sequence_token_count = latent_frames * token_grid_height * token_grid_width

    frame_distances = list(range(latent_frames))
    frame_full_kernel = [
        _wan21_t2v_full_rope_kernel(
            delta_f=delta_frame,
            delta_h=0,
            delta_w=0,
            temporal_frequencies=temporal_frequencies,
            height_frequencies=height_frequencies,
            width_frequencies=width_frequencies,
        )
        for delta_frame in frame_distances
    ]
    frame_temporal_only_kernel = [
        _wan21_t2v_mean_rope_cosine_kernel(delta_frame, temporal_frequencies)
        for delta_frame in frame_distances
    ]

    token_distances = list(range(sequence_token_count))
    token_full_kernel = []
    spatial_area = token_grid_height * token_grid_width
    for token_delta in token_distances:
        delta_f = int(token_delta) // int(spatial_area)
        rem = int(token_delta) % int(spatial_area)
        delta_h = int(rem) // int(token_grid_width)
        delta_w = int(rem) % int(token_grid_width)
        token_full_kernel.append(
            _wan21_t2v_full_rope_kernel(
                delta_f=delta_f,
                delta_h=delta_h,
                delta_w=delta_w,
                temporal_frequencies=temporal_frequencies,
                height_frequencies=height_frequencies,
                width_frequencies=width_frequencies,
            )
        )

    _ensure_dir(output_dir)
    frame_plot_path = _plot_wan21_t2v_rope_decay_curve(
        x_values=frame_distances,
        series=[
            ("full_head_same_spatial", frame_full_kernel),
            ("temporal_axis_only", frame_temporal_only_kernel),
        ],
        save_file=os.path.join(output_dir, "rope_decay_curve_frame_level.pdf"),
        title=f"Wan2.1 RoPE Decay vs Frame Distance | {task} | size={size[0]}x{size[1]}",
        x_label="relative frame distance in latent token frames",
        y_label="mean RoPE cosine kernel",
    )
    token_plot_path = _plot_wan21_t2v_rope_decay_curve(
        x_values=token_distances,
        series=[("flattened_token_anchor_curve", token_full_kernel)],
        save_file=os.path.join(output_dir, "rope_decay_curve_token_level.pdf"),
        title=f"Wan2.1 RoPE Decay vs Token Distance | {task} | size={size[0]}x{size[1]}",
        x_label="relative distance in flattened video tokens",
        y_label="mean RoPE cosine kernel",
    )

    summary = {
        "experiment": "wan21_t2v_rope_decay_curve",
        "task": str(task),
        "frame_num": int(frame_num),
        "size": [int(size[0]), int(size[1])],
        "vae_stride": [int(v) for v in cfg.vae_stride],
        "patch_size": [int(v) for v in cfg.patch_size],
        "num_heads": int(cfg.num_heads),
        "head_dim": int(head_dim),
        "temporal_real_dim": int(temporal_real_dim),
        "spatial_real_dim_per_axis": int(spatial_real_dim),
        "latent_frames": int(latent_frames),
        "token_grid_height": int(token_grid_height),
        "token_grid_width": int(token_grid_width),
        "sequence_token_count": int(sequence_token_count),
        "frame_plot_path": frame_plot_path,
        "token_plot_path": token_plot_path,
    }
    _save_json(os.path.join(output_dir, "rope_decay_curve_summary.json"), summary)
    return summary
