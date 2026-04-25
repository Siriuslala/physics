"""Shared utilities for Wan2.1-T2V experiment modules.

This module centralizes the helper code that is reused by multiple experiments.
It intentionally excludes experiment-specific analysis logic so that each
experiment module can stay focused on its own method.

Included shared groups:
- Runtime / pipeline / IO helpers
- DT / motion-alignment shared helpers
- Generic patch / parse helpers
- Step-window shared patch core
- Cross-attention visualization shared core
- Shared map / trajectory analysis helpers

These shared helpers are consumed by the per-experiment modules created during
this refactor. The public CLI and bash entrypoints remain unchanged because the
compatibility facade still re-exports the original names.
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
from pathlib import Path
from types import MethodType
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F
from dotenv import load_dotenv

from .wan21_t2v_experiment_patch import (
    Wan21T2VAttentionProbeConfig,
    Wan21T2VCausalAttentionConfig,
    Wan21T2VPatchBundleConfig,
    Wan21T2VRopePatchConfig,
    install_wan21_t2v_dit_patch_stack,
)

load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
root_dir = Path(os.getenv('ROOT_DIR', Path(__file__).parent))
data_dir = Path(os.getenv('DATA_DIR'))
work_dir = Path(os.getenv('WORK_DIR'))

sys.path.append(root_dir.as_posix())

from projects.Wan2_1 import wan
from projects.Wan2_1.wan.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from projects.Wan2_1.wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from projects.Wan2_1.wan.utils.utils import cache_video

# ============================================================================
# Runtime / Pipeline / IO
# ============================================================================

@dataclass
class Wan21T2VParallelConfig:
    """Parallel and model-loading options aligned with Wan2.1 generate.py."""

    t5_fsdp: bool = False
    dit_fsdp: bool = False
    t5_cpu: bool = False

    use_usp: bool = False
    ulysses_size: int = 1
    ring_size: int = 1

@dataclass
class Wan21T2VRuntimeContext:
    """Runtime context inferred from environment variables."""

    rank: int
    world_size: int
    local_rank: int
    device_id: int
    use_usp: bool

def ensure_wan21_t2v_repo_on_path(wan21_root: str):
    """Append Wan2.1 repository root to `sys.path` if needed."""
    if wan21_root not in sys.path:
        sys.path.insert(0, wan21_root)

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _save_csv(path: str, rows: List[Dict]):
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def _init_wan21_t2v_runtime(parallel_cfg: Wan21T2VParallelConfig, explicit_device_id: Optional[int] = None):
    """Initialize distributed/runtime environment similarly to Wan2.1 generate.py."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if world_size <= 1 and (parallel_cfg.t5_fsdp or parallel_cfg.dit_fsdp):
        raise RuntimeError("t5_fsdp/dit_fsdp requires distributed launch (torchrun).")

    if world_size > 1:
        # In torchrun mode, always bind model shards to local rank device.
        device_id = local_rank
    else:
        # In single-process mode, ignore stale distributed env values.
        rank = 0
        local_rank = 0
        world_size = 1
        device_id = 0 if explicit_device_id is None else explicit_device_id

    # Always bind this process to resolved CUDA device.
    torch.cuda.set_device(device_id)

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )

    use_usp = parallel_cfg.use_usp or parallel_cfg.ulysses_size > 1 or parallel_cfg.ring_size > 1
    if use_usp:
        if not dist.is_initialized():
            raise RuntimeError("USP requires distributed initialization. Please run with torchrun.")
        if parallel_cfg.ulysses_size * parallel_cfg.ring_size != world_size:
            raise ValueError("ulysses_size * ring_size must equal WORLD_SIZE when USP is enabled.")

        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )

        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=parallel_cfg.ring_size,
            ulysses_degree=parallel_cfg.ulysses_size,
        )

    return Wan21T2VRuntimeContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device_id=device_id,
        use_usp=use_usp,
    )

def _broadcast_seed_if_needed(seed: int, runtime: Wan21T2VRuntimeContext) -> int:
    if dist.is_initialized():
        payload = [seed] if runtime.rank == 0 else [None]
        dist.broadcast_object_list(payload, src=0)
        return int(payload[0])
    return seed

def _build_wan21_t2v_pipeline(
    wan21_root: str,
    ckpt_dir: str,
    task: str,
    runtime: Wan21T2VRuntimeContext,
    parallel_cfg: Wan21T2VParallelConfig,
):
    """Create Wan2.1 T2V pipeline with configurable parallel options."""
    ensure_wan21_t2v_repo_on_path(wan21_root)

    cfg = wan.configs.WAN_CONFIGS[task]
    if parallel_cfg.ulysses_size > 1 and (cfg.num_heads % parallel_cfg.ulysses_size != 0):
        raise ValueError(
            f"cfg.num_heads={cfg.num_heads} must be divisible by ulysses_size={parallel_cfg.ulysses_size}."
        )

    pipeline = wan.WanT2V(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        device_id=runtime.device_id,
        rank=runtime.rank,
        t5_fsdp=parallel_cfg.t5_fsdp,
        dit_fsdp=parallel_cfg.dit_fsdp,
        use_usp=runtime.use_usp,
        t5_cpu=parallel_cfg.t5_cpu,
    )
    return pipeline, cfg

def _generate_wan21_t2v_video(
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
):
    """Run Wan2.1 T2V generation once."""
    return pipeline.generate(
        input_prompt=prompt,
        size=size,
        frame_num=frame_num,
        shift=shift,
        sample_solver=sample_solver,
        sampling_steps=sampling_steps,
        guide_scale=guide_scale,
        seed=seed,
        offload_model=offload_model,
        step_callback=None,
    )

def _generate_wan21_t2v_video_with_initial_noise(
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
):
    """Run Wan2.1 T2V generation once and return the initial latent noise.

    This mirrors `WanT2V.generate` but exposes the sampled Gaussian latent
    `z_T`. It is intentionally local to the experiment module so Wan source
    files remain untouched.

    Returns:
        `(video, initial_noise)` where `initial_noise` has shape
        `[C, F_latent, H_latent, W_latent]` on CPU and `video` is Wan's decoded
        output on rank 0.
    """
    target_shape = (
        pipeline.vae.model.z_dim,
        (int(frame_num) - 1) // pipeline.vae_stride[0] + 1,
        int(size[1]) // pipeline.vae_stride[1],
        int(size[0]) // pipeline.vae_stride[2],
    )
    seq_len = math.ceil(
        (target_shape[2] * target_shape[3])
        / (pipeline.patch_size[1] * pipeline.patch_size[2])
        * target_shape[1]
        / pipeline.sp_size
    ) * pipeline.sp_size

    n_prompt = pipeline.sample_neg_prompt
    seed = int(seed) if int(seed) >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=pipeline.device)
    seed_g.manual_seed(seed)

    if not pipeline.t5_cpu:
        pipeline.text_encoder.model.to(pipeline.device)
        context = pipeline.text_encoder([prompt], pipeline.device)
        context_null = pipeline.text_encoder([n_prompt], pipeline.device)
        if offload_model:
            pipeline.text_encoder.model.cpu()
    else:
        context = pipeline.text_encoder([prompt], torch.device("cpu"))
        context_null = pipeline.text_encoder([n_prompt], torch.device("cpu"))
        context = [t.to(pipeline.device) for t in context]
        context_null = [t.to(pipeline.device) for t in context_null]

    noise = [
        torch.randn(
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device=pipeline.device,
            generator=seed_g,
        )
    ]
    initial_noise = noise[0].detach().cpu().clone()

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(pipeline.model, "no_sync", noop_no_sync)
    videos = None
    latents = noise

    with amp.autocast(dtype=pipeline.param_dtype), torch.no_grad(), no_sync():
        if sample_solver == "unipc":
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=pipeline.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sample_scheduler.set_timesteps(sampling_steps, device=pipeline.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == "dpm++":
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=pipeline.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=pipeline.device,
                sigmas=sampling_sigmas,
            )
        else:
            raise NotImplementedError("Unsupported solver.")

        arg_c = {"context": context, "seq_len": seq_len}
        arg_null = {"context": context_null, "seq_len": seq_len}

        from tqdm import tqdm

        for t in tqdm(timesteps):
            latent_model_input = latents
            timestep = torch.stack([t])

            pipeline.model.to(pipeline.device)
            noise_pred_cond = pipeline.model(latent_model_input, t=timestep, **arg_c)[0]
            noise_pred_uncond = pipeline.model(latent_model_input, t=timestep, **arg_null)[0]
            noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0),
                t,
                latents[0].unsqueeze(0),
                return_dict=False,
                generator=seed_g,
            )[0]
            latents = [temp_x0.squeeze(0)]

        x0 = latents
        if offload_model:
            pipeline.model.cpu()
            torch.cuda.empty_cache()
        if pipeline.rank == 0:
            videos = pipeline.vae.decode(x0)

    del noise, latents, sample_scheduler
    if offload_model:
        gc.collect()
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    return (videos[0] if pipeline.rank == 0 else None), initial_noise

def _encode_wan21_t2v_text_context_once(
    pipeline,
    text: str,
    offload_model: bool,
):
    """Encode one text prompt into Wan T2V context format (List[Tensor])."""
    if not pipeline.t5_cpu:
        pipeline.text_encoder.model.to(pipeline.device)
        context = pipeline.text_encoder([text], pipeline.device)
        if offload_model:
            pipeline.text_encoder.model.cpu()
    else:
        context = pipeline.text_encoder([text], torch.device("cpu"))
        context = [t.to(pipeline.device) for t in context]
    return context

def _resolve_wan21_t2v_offload_model(runtime: Wan21T2VRuntimeContext, offload_model: bool) -> bool:
    """Normalize offload behavior for distributed inference.

    Wan official generate.py defaults to offload_model=False when world_size>1.
    We follow that behavior for stability in multi-GPU runs.
    """
    if runtime.world_size > 1 and offload_model:
        if runtime.rank == 0:
            print(
                "[wan21_t2v_experiments] offload_model=True with distributed inference can cause instability. "
                "Force offload_model=False for parity with Wan generate.py."
            )
        return False
    return offload_model

def _save_wan21_t2v_video(video_tensor: torch.Tensor, save_path: str, fps: int):
    """Save generated video tensor using Wan utilities."""

    _ensure_dir(os.path.dirname(save_path))
    cache_video(
        tensor=video_tensor[None],
        save_file=save_path,
        fps=fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )

# ============================================================================
# DT / Motion Alignment Shared Helpers
# ============================================================================

def _extract_wan21_t2v_motion_centroid_trajectory(video_tensor: torch.Tensor) -> List[Tuple[float, float]]:
    """Compute simple motion-centroid trajectory from generated video.

    This is a lightweight proxy trajectory suitable for MAAS analysis.
    """
    x = video_tensor.detach().float().cpu()
    _, t, _, w = x.shape

    gray = x.mean(dim=0)
    trajectory = []

    for i in range(t):
        prev_i = max(0, i - 1)
        diff = (gray[i] - gray[prev_i]).abs()
        threshold = torch.quantile(diff.flatten(), 0.95)
        mask = diff >= threshold

        if mask.any():
            ys, xs = torch.nonzero(mask, as_tuple=True)
            vals = diff[ys, xs].clamp_min(1e-6)
            y = float((ys.float() * vals).sum().item() / vals.sum().item())
            x_ = float((xs.float() * vals).sum().item() / vals.sum().item())
        else:
            flat_idx = int(diff.view(-1).argmax().item())
            y = float(flat_idx // w)
            x_ = float(flat_idx % w)

        trajectory.append((y, x_))

    return trajectory

def _project_wan21_t2v_pixel_traj_to_tokens(
    pixel_traj: Sequence[Tuple[float, float]],
    video_h: int,
    video_w: int,
    token_f: int,
    token_h: int,
    token_w: int,
    temporal_stride: int = 4,
) -> Dict[int, Tuple[int, int]]:
    """Project pixel-space trajectory into token grid indices."""
    out = {}
    t_total = len(pixel_traj)

    for tf in range(token_f):
        frame_idx = min(t_total - 1, tf * temporal_stride)
        y_px, x_px = pixel_traj[frame_idx]

        y_tok = int(round((y_px / max(1, video_h - 1)) * (token_h - 1)))
        x_tok = int(round((x_px / max(1, video_w - 1)) * (token_w - 1)))

        y_tok = max(0, min(token_h - 1, y_tok))
        x_tok = max(0, min(token_w - 1, x_tok))
        out[tf] = (y_tok, x_tok)

    return out

def _summarize_wan21_t2v_dt_hist(dt_tensors: Dict[str, torch.Tensor]) -> Dict:
    """Create lightweight summary statistics from dt-hist tensors."""
    summary = {}
    for key, tensor in dt_tensors.items():
        if not key.startswith("dt_hist_step_"):
            continue

        bins = tensor.size(-1)
        center = tensor[..., 0].mean().item()
        near = tensor[..., : min(3, bins)].sum(dim=-1).mean().item()
        far_start = min(5, bins)
        far = tensor[..., far_start:].sum(dim=-1).mean().item() if far_start < bins else 0.0

        summary[key] = {
            "center_mass_mean": float(center),
            "near_mass_mean": float(near),
            "far_mass_mean": float(far),
            "dt_bins": int(bins),
        }
    return summary

def _export_wan21_t2v_dt_profile_visualizations(
    dt_tensors: Dict[str, torch.Tensor],
    output_dir: str,
    file_prefix: str,
) -> Dict[str, object]:
    """Export readable plots from dt histogram tensors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    step_items: List[Tuple[int, torch.Tensor]] = []
    for key, tensor in dt_tensors.items():
        if key.startswith("dt_hist_step_"):
            step_id = int(key.rsplit("_", 1)[-1])
            step_items.append((step_id, tensor.detach().float().cpu()))
    step_items = sorted(step_items, key=lambda x: x[0])
    if not step_items:
        return {"global_curve_pdf": "", "layer_heatmap_dir": "", "layer_heatmap_files": []}

    _ensure_dir(output_dir)

    # 1) Global P(|dt|): averaged across layers and heads.
    global_pdf = os.path.join(output_dir, f"{file_prefix}_global_dt_curve.pdf")
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8))
    for step_id, hist_lhb in step_items:
        hist_b = hist_lhb.mean(dim=(0, 1))  # [B]
        hist_b = hist_b / hist_b.sum().clamp_min(1e-8)
        xs = torch.arange(hist_b.numel())
        ax.plot(xs.numpy(), hist_b.numpy(), marker="o", linewidth=1.6, markersize=3.6, label=f"step={step_id}")
    ax.set_title("Global Temporal Offset Profile P(|dt|)")
    ax.set_xlabel("|dt| in token-frame units")
    ax.set_ylabel("probability mass")
    ax.grid(alpha=0.22, linestyle="--")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(global_pdf, format="pdf")
    plt.close(fig)

    # 2) Layer-wise heatmaps per step: each row is one layer.
    heatmap_dir = os.path.join(output_dir, f"{file_prefix}_layer_heatmaps")
    _ensure_dir(heatmap_dir)
    heatmap_files: List[str] = []
    for step_id, hist_lhb in step_items:
        layer_hist = hist_lhb.mean(dim=1)  # [L, B], head-averaged
        layer_hist = layer_hist / layer_hist.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        save_path = os.path.join(heatmap_dir, f"step_{step_id:03d}.pdf")
        fig, ax = plt.subplots(1, 1, figsize=(8.2, 5.2))
        im = ax.imshow(layer_hist.numpy(), aspect="auto", cmap="viridis")
        ax.set_title(f"Layer-wise P(|dt|), step={step_id}")
        ax.set_xlabel("|dt| in token-frame units")
        ax.set_ylabel("layer index")
        cbar = fig.colorbar(im, ax=ax, shrink=0.88)
        cbar.set_label("probability")
        fig.tight_layout()
        fig.savefig(save_path, format="pdf")
        plt.close(fig)
        heatmap_files.append(save_path)

    return {
        "global_curve_pdf": global_pdf,
        "layer_heatmap_dir": heatmap_dir,
        "layer_heatmap_files": heatmap_files,
    }

def _run_wan21_t2v_once_with_patch(
    pipeline,
    patch_cfg: Wan21T2VPatchBundleConfig,
    prompt: str,
    size: Tuple[int, int],
    frame_num: int,
    shift: float,
    sample_solver: str,
    sampling_steps: int,
    guide_scale: float,
    seed: int,
    offload_model: bool,
):
    """Apply patch stack, run one generation, then restore patches."""
    handle = install_wan21_t2v_dit_patch_stack(pipeline.model, patch_cfg)
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

    return video, handle.state

def _save_wan21_t2v_motion_alignment_map_pdf(
    attention_map_hw: torch.Tensor,
    target_y: int,
    target_x: int,
    radius: int,
    save_file: str,
    title: str,
):
    """Save one motion-alignment heatmap with target neighborhood overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = attention_map_hw.shape
    y0 = max(0, target_y - radius)
    y1 = min(h, target_y + radius + 1)
    x0 = max(0, target_x - radius)
    x1 = min(w, target_x + radius + 1)

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 5.3))
    im = ax.imshow(attention_map_hw.detach().cpu().numpy(), cmap="viridis")
    ax.scatter([target_x], [target_y], c=["#ff3b30"], s=45, marker="x", linewidths=1.5)
    rect = plt.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, edgecolor="#ff9500", linewidth=1.6)
    ax.add_patch(rect)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("token-x")
    ax.set_ylabel("token-y")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)

def _export_wan21_t2v_motion_alignment_visualizations(
    state,
    motion_alignment_rows: Sequence[Dict],
    token_trajectory: Dict[int, Tuple[int, int]],
    radius: int,
    output_dir: str,
    video_frame_count: Optional[int] = None,
):
    """Export motion-aligned attention visualizations and summary curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not token_trajectory:
        return
    _ensure_dir(output_dir)
    token_frame_count = int(state.maas_grid_size[0]) if state.maas_grid_size is not None else None

    # 1) Per-map head-mean visualizations.
    map_root = os.path.join(output_dir, "next_frame_attention_maps")
    for (step, layer, query_frame), map_sum in sorted(state.maas_maps_sum.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        target = token_trajectory.get(query_frame + 1)
        if target is None:
            continue
        target_y, target_x = target
        count = max(1, state.maas_maps_count[(step, layer, query_frame)])
        mean_map = (map_sum / count).mean(dim=0)
        query_video_frame = _map_wan21_t2v_token_frame_to_video_frame_label(
            token_frame_idx=query_frame,
            token_frame_count=token_frame_count,
            video_frame_count=video_frame_count,
        )
        target_video_frame = _map_wan21_t2v_token_frame_to_video_frame_label(
            token_frame_idx=query_frame + 1,
            token_frame_count=token_frame_count,
            video_frame_count=video_frame_count,
        )
        save_path = os.path.join(
            map_root,
            f"step_{step:03d}",
            f"layer_{layer:02d}",
            f"query_frame_{query_video_frame:03d}_to_{target_video_frame:03d}.pdf",
        )
        _save_wan21_t2v_motion_alignment_map_pdf(
            attention_map_hw=mean_map,
            target_y=target_y,
            target_x=target_x,
            radius=radius,
            save_file=save_path,
            title=(
                f"step={step} layer={layer} "
                f"query_frame={query_video_frame} -> target_frame={target_video_frame} "
                f"(token {query_frame}->{query_frame + 1})"
            ),
        )

    if not motion_alignment_rows:
        return

    # 2) MAAS ratio vs layer (line per diffusion step).
    by_step_layer = defaultdict(list)
    for row in motion_alignment_rows:
        by_step_layer[(int(row["step"]), int(row["layer"]))].append(float(row["maas_local_ratio"]))

    layer_values = sorted({k[1] for k in by_step_layer.keys()})
    step_values = sorted({k[0] for k in by_step_layer.keys()})
    if layer_values and step_values:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.4))
        for step in step_values:
            ys = []
            for layer in layer_values:
                vals = by_step_layer.get((step, layer), [])
                ys.append(float(torch.tensor(vals).mean().item()) if vals else float("nan"))
            ax.plot(layer_values, ys, marker="o", linewidth=1.6, label=f"step={step}")
        ax.set_title("Motion-Aligned Attention Ratio vs Layer")
        ax.set_xlabel("layer")
        ax.set_ylabel("mean local ratio")
        ax.grid(alpha=0.2, linestyle="--")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "motion_alignment_ratio_vs_layer.pdf"), format="pdf")
        plt.close(fig)

    # 3) MAAS ratio vs query frame (line per diffusion step).
    by_step_query = defaultdict(list)
    for row in motion_alignment_rows:
        query_frame_key = int(row.get("query_frame_video", row["query_frame"]))
        by_step_query[(int(row["step"]), query_frame_key)].append(float(row["maas_local_ratio"]))
    query_values = sorted({k[1] for k in by_step_query.keys()})
    if query_values and step_values:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.4))
        for step in step_values:
            ys = []
            for query_frame in query_values:
                vals = by_step_query.get((step, query_frame), [])
                ys.append(float(torch.tensor(vals).mean().item()) if vals else float("nan"))
            ax.plot(query_values, ys, marker="o", linewidth=1.6, label=f"step={step}")
        ax.set_title("Motion-Aligned Attention Ratio vs Query Frame (Video Index)")
        ax.set_xlabel("query_frame (video index, 1-based)")
        ax.set_ylabel("mean local ratio")
        ax.grid(alpha=0.2, linestyle="--")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "motion_alignment_ratio_vs_query_frame.pdf"), format="pdf")
        plt.close(fig)

# ============================================================================
# Generic Patch / Parse Helpers
# ============================================================================

def _dedup_wan21_t2v_int_list(values: Sequence[int]) -> List[int]:
    """Deduplicate an integer sequence while preserving input order."""
    out: List[int] = []
    seen = set()
    for v in values:
        vv = int(v)
        if vv in seen:
            continue
        seen.add(vv)
        out.append(vv)
    return out

def _parse_wan21_t2v_layer_head_specs(head_specs: Sequence[str]) -> List[Tuple[int, int]]:
    """Parse head specs into `(layer_idx, head_idx)` tuples.

    Canonical format is `LxHy` (e.g. `L29H7`).
    For convenience, `(x,y)` is also accepted.
    """
    out: List[Tuple[int, int]] = []
    seen = set()
    for raw in head_specs:
        s = str(raw).strip()
        if not s:
            continue

        m = re.fullmatch(r"[Ll](\d+)[Hh](\d+)", s)
        if m is None:
            m = re.fullmatch(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", s)
        if m is None:
            raise ValueError(
                f"Invalid head spec: `{s}`. Use `LxHy` (recommended), e.g. `L29H7`."
            )

        layer_idx = int(m.group(1))
        head_idx = int(m.group(2))
        if layer_idx < 0 or head_idx < 0:
            raise ValueError(f"Head spec must be non-negative, got `{s}`.")

        key = (layer_idx, head_idx)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out

def _resolve_wan21_t2v_branch_from_forward_call_index(
    forward_call_index_in_step: int,
) -> str:
    """Return CFG branch name for Wan2.1 T2V DiT calls.

    Wan's official T2V sampler calls the DiT twice per diffusion step: first
    with the conditional prompt context and then with the unconditional context.

    Args:
        forward_call_index_in_step: Zero-based DiT forward count within the
            current diffusion timestep.

    Returns:
        `cond`, `uncond`, or `extra` for unexpected additional calls.
    """
    if int(forward_call_index_in_step) == 0:
        return "cond"
    if int(forward_call_index_in_step) == 1:
        return "uncond"
    return "extra"

def _wan21_t2v_branch_matches(
    requested_branch: str,
    forward_call_index_in_step: int,
) -> bool:
    """Check whether a DiT forward call matches a requested CFG branch."""
    branch = str(requested_branch).strip().lower()
    current = _resolve_wan21_t2v_branch_from_forward_call_index(forward_call_index_in_step)
    if branch == "both":
        return current in {"cond", "uncond"}
    if branch in {"cond", "uncond"}:
        return current == branch
    raise ValueError(f"Unknown branch selector: {requested_branch}")

def _resolve_wan21_t2v_steps(
    explicit_steps: Sequence[int],
    sampling_steps: int,
) -> List[int]:
    """Resolve optional 1-based diffusion step list."""
    if explicit_steps:
        steps = _dedup_wan21_t2v_int_list(explicit_steps)
    else:
        steps = list(range(1, int(sampling_steps) + 1))
    bad = [step for step in steps if int(step) < 1 or int(step) > int(sampling_steps)]
    if bad:
        raise ValueError(f"Steps must lie in [1, {int(sampling_steps)}], got {bad}.")
    return [int(step) for step in steps]

def _unwrap_wan21_t2v_dit_model_for_runtime_patch(model):
    """Return inner DiT module that owns `blocks` and `forward`."""
    if hasattr(model, "module") and hasattr(model.module, "blocks"):
        return model.module
    return model

# ============================================================================
# Step-Window Shared Patch Core
# ============================================================================

class Wan21T2VStepWindowAblationState:
    """Runtime state for Wan2.1-T2V step-window ablations."""

    def __init__(
        self,
        cross_attn_remove_start_step: Optional[int] = None,
        cross_attn_step_scope: str = "from_step",
        ffn_remove_step: Optional[int] = None,
        ffn_step_scope: str = "single_step",
        ffn_remove_layers: Optional[Sequence[int]] = None,
        reuse_removed_cond_for_uncond: bool = False,
    ):
        self.cross_attn_remove_start_step = (
            int(cross_attn_remove_start_step)
            if cross_attn_remove_start_step is not None
            else None
        )
        if cross_attn_step_scope not in {"single_step", "from_step"}:
            raise ValueError(f"Invalid cross_attn_step_scope: {cross_attn_step_scope}")
        self.cross_attn_step_scope = cross_attn_step_scope

        self.ffn_remove_step = int(ffn_remove_step) if ffn_remove_step is not None else None
        if ffn_step_scope not in {"single_step", "from_step"}:
            raise ValueError(f"Invalid ffn_step_scope: {ffn_step_scope}")
        self.ffn_step_scope = ffn_step_scope

        self.ffn_remove_layers = (
            None if ffn_remove_layers is None else set(int(x) for x in ffn_remove_layers)
        )
        self.reuse_removed_cond_for_uncond = bool(reuse_removed_cond_for_uncond)

        self.current_step = 0
        self.current_timestep_value = None
        self.forward_call_index_in_step = 0
        self.total_forward_calls = 0
        self.cross_attn_removed_calls = 0
        self.ffn_removed_calls = 0
        self.uncond_reused_calls = 0
        self.cached_removed_cond_output = None

    def on_forward_start(self, t_tensor):
        """Track diffusion step id and intra-step forward-call index."""
        t_value = float(t_tensor.flatten()[0].item()) if t_tensor is not None else None
        self.total_forward_calls += 1

        if self.current_timestep_value is None or t_value != self.current_timestep_value:
            self.current_step += 1
            self.current_timestep_value = t_value
            self.forward_call_index_in_step = 0
            self.cached_removed_cond_output = None
        else:
            self.forward_call_index_in_step += 1

    def _match_step_scope(self, target_step: Optional[int], scope: str) -> bool:
        if target_step is None:
            return False
        if scope == "single_step":
            return self.current_step == int(target_step)
        return self.current_step >= int(target_step)

    def should_remove_cross_attn(self) -> bool:
        return self._match_step_scope(
            target_step=self.cross_attn_remove_start_step,
            scope=self.cross_attn_step_scope,
        )

    def should_remove_ffn(self, layer_idx: int) -> bool:
        if not self._match_step_scope(
            target_step=self.ffn_remove_step,
            scope=self.ffn_step_scope,
        ):
            return False
        if self.ffn_remove_layers is None:
            return True
        return int(layer_idx) in self.ffn_remove_layers

class Wan21T2VStepWindowAblationPatchHandle:
    """Restore handle for temporary step-window ablation patches."""

    def __init__(self, target_model, state, original_forward, original_block_forwards):
        self.target_model = target_model
        self.state = state
        self.original_forward = original_forward
        self.original_block_forwards = original_block_forwards

    def restore(self):
        self.target_model.forward = self.original_forward
        for idx, block in enumerate(self.target_model.blocks):
            block.forward = self.original_block_forwards[idx]

def _install_wan21_t2v_step_window_ablation_patch(
    model,
    cross_attn_remove_start_step: Optional[int] = None,
    cross_attn_step_scope: str = "from_step",
    ffn_remove_step: Optional[int] = None,
    ffn_step_scope: str = "single_step",
    ffn_remove_layers: Optional[Sequence[int]] = None,
    reuse_removed_cond_for_uncond: bool = False,
) -> Wan21T2VStepWindowAblationPatchHandle:
    """Install runtime patch for cross-attn/ffn step-window ablations."""
    target = _unwrap_wan21_t2v_dit_model_for_runtime_patch(model)
    if not hasattr(target, "blocks"):
        raise RuntimeError("Invalid DiT model: missing blocks.")

    state = Wan21T2VStepWindowAblationState(
        cross_attn_remove_start_step=cross_attn_remove_start_step,
        cross_attn_step_scope=cross_attn_step_scope,
        ffn_remove_step=ffn_remove_step,
        ffn_step_scope=ffn_step_scope,
        ffn_remove_layers=ffn_remove_layers,
        reuse_removed_cond_for_uncond=reuse_removed_cond_for_uncond,
    )

    original_forward = target.forward

    def patched_dit_forward(this, *args, **kwargs):
        t = kwargs.get("t", None)
        if t is None and len(args) > 1:
            t = args[1]
        state.on_forward_start(t)

        # Optional speed optimization:
        # when cross-attn is removed for this step, cond/uncond branches become
        # identical; reuse cond output for uncond call.
        if (
            state.reuse_removed_cond_for_uncond
            and state.should_remove_cross_attn()
            and state.forward_call_index_in_step == 1
            and state.cached_removed_cond_output is not None
        ):
            state.uncond_reused_calls += 1
            return state.cached_removed_cond_output

        out = original_forward(*args, **kwargs)
        if (
            state.reuse_removed_cond_for_uncond
            and state.should_remove_cross_attn()
            and state.forward_call_index_in_step == 0
        ):
            state.cached_removed_cond_output = out
        return out

    target.forward = MethodType(patched_dit_forward, target)

    original_block_forwards = []

    def build_patched_block_forward(layer_idx: int):
        def patched_block_forward(
            self,
            x,
            e,
            seq_lens,
            grid_sizes,
            freqs,
            context,
            context_lens,
        ):
            assert e.dtype == torch.float32
            with amp.autocast(dtype=torch.float32):
                e = (self.modulation + e).chunk(6, dim=1)
            assert e[0].dtype == torch.float32

            # self-attention
            y = self.self_attn(
                self.norm1(x).float() * (1 + e[1]) + e[0],
                seq_lens,
                grid_sizes,
                freqs,
            )
            with amp.autocast(dtype=torch.float32):
                x = x + y * e[2]

            # cross-attention
            if state.should_remove_cross_attn():
                state.cross_attn_removed_calls += 1
            else:
                x = x + self.cross_attn(self.norm3(x), context, context_lens)

            # ffn
            if state.should_remove_ffn(layer_idx):
                state.ffn_removed_calls += 1
            else:
                y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
                with amp.autocast(dtype=torch.float32):
                    x = x + y * e[5]

            return x

        return patched_block_forward

    for layer_idx, block in enumerate(target.blocks):
        original_block_forwards.append(block.forward)
        block.forward = MethodType(build_patched_block_forward(layer_idx), block)

    return Wan21T2VStepWindowAblationPatchHandle(
        target_model=target,
        state=state,
        original_forward=original_forward,
        original_block_forwards=original_block_forwards,
    )

# ============================================================================
# Cross-Attention Visualization Shared Core
# ============================================================================

class Wan21T2VCrossAttentionVizState:
    """Runtime collector for token-targeted cross-attention visualization."""

    def __init__(
        self,
        token_positions: Dict[str, List[int]],
        collect_steps: Sequence[int],
        num_layers: int,
        num_heads: int,
        chunk_size: int = 1024,
        layers_to_collect: Optional[Sequence[int]] = None,
    ):
        self.token_positions = {k: sorted(set(v)) for k, v in token_positions.items()}
        self.collect_steps = set(int(s) for s in collect_steps)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.chunk_size = int(chunk_size)
        self.layers_to_collect = set(layers_to_collect) if layers_to_collect else None

        self.current_step = 0
        self.current_timestep_value = None
        self.forward_call_index_in_step = 0

        self.layer_meta: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.maps_sum: Dict[Tuple[int, int, str], torch.Tensor] = {}
        self.maps_count: Dict[Tuple[int, int, str], int] = {}

    def on_forward_start(self, t_tensor):
        """Update diffusion step id from timestep tensor.

        Returns:
            Optional[int]: completed diffusion step id when stepping to next t.
        """
        t_value = float(t_tensor.flatten()[0].item()) if t_tensor is not None else None
        completed_step = None
        if self.current_timestep_value is None or t_value != self.current_timestep_value:
            if self.current_timestep_value is not None:
                completed_step = self.current_step
            self.current_step += 1
            self.current_timestep_value = t_value
            self.forward_call_index_in_step = 0
        else:
            self.forward_call_index_in_step += 1

        # Clear per-step layer cache.
        self.layer_meta.clear()
        return completed_step

    def should_collect(self, layer_idx: int) -> bool:
        if self.current_step not in self.collect_steps:
            return False
        if self.forward_call_index_in_step != 0:
            return False
        if self.layers_to_collect is not None and layer_idx not in self.layers_to_collect:
            return False
        return True

    def set_layer_meta(self, layer_idx: int, seq_lens: torch.Tensor, grid_sizes: torch.Tensor):
        """Cache seq_lens/grid_sizes for subsequent cross-attn call in the same block."""
        self.layer_meta[layer_idx] = (seq_lens, grid_sizes)

    def collect_cross_attention(
        self,
        layer_idx: int,
        cross_attn_module,
        x: torch.Tensor,
        context: torch.Tensor,
    ):
        """Collect token-targeted cross-attention mass over video tokens."""
        if not self.should_collect(layer_idx):
            return
        if layer_idx not in self.layer_meta:
            return

        seq_lens, grid_sizes = self.layer_meta[layer_idx]
        bsz = x.size(0)
        n_heads = cross_attn_module.num_heads
        head_dim = cross_attn_module.head_dim

        q = cross_attn_module.norm_q(cross_attn_module.q(x)).view(bsz, -1, n_heads, head_dim)
        k = cross_attn_module.norm_k(cross_attn_module.k(context)).view(bsz, -1, n_heads, head_dim)
        scale = 1.0 / (head_dim ** 0.5)

        word_idx_tensors = {
            word: torch.tensor(indices, device=q.device, dtype=torch.long)
            for word, indices in self.token_positions.items()
        }

        for b in range(bsz):
            seq_len = int(seq_lens[b].item())
            if seq_len <= 0:
                continue

            f, h, w = [int(v) for v in grid_sizes[b].tolist()]
            valid_len = min(seq_len, f * h * w, q.size(1))
            if valid_len <= 0:
                continue

            q_i = q[b, :valid_len]
            k_i = k[b]
            lk = k_i.size(0)

            # Ignore token indices out of context length.
            valid_word_indices: Dict[str, torch.Tensor] = {}
            for word, idxs in word_idx_tensors.items():
                keep = idxs[idxs < lk]
                if keep.numel() > 0:
                    valid_word_indices[word] = keep
            if not valid_word_indices:
                continue

            word_scores = {
                word: torch.zeros((n_heads, valid_len), dtype=torch.float32, device=q.device)
                for word in valid_word_indices
            }

            for q_start in range(0, valid_len, self.chunk_size):
                q_end = min(valid_len, q_start + self.chunk_size)
                q_chunk = q_i[q_start:q_end].float()  # [Qc, H, D]
                logits = torch.einsum("qhd,khd->hqk", q_chunk, k_i.float()) * scale  # [H, Qc, Lk]
                log_denom = torch.logsumexp(logits, dim=-1)  # [H, Qc]

                for word, idxs in valid_word_indices.items():
                    sel_logits = logits.index_select(dim=-1, index=idxs)  # [H, Qc, |word_subtokens|]
                    # Word-level probability = sum over all matched subtoken positions.
                    log_word = torch.logsumexp(sel_logits, dim=-1) - log_denom
                    word_scores[word][:, q_start:q_end] = torch.exp(log_word)

            for word, score_hl in word_scores.items():
                score_hfhw = score_hl.reshape(n_heads, f, h, w).detach().cpu()
                key = (self.current_step, layer_idx, word)
                if key not in self.maps_sum:
                    self.maps_sum[key] = score_hfhw.double()
                    self.maps_count[key] = 1
                else:
                    self.maps_sum[key] += score_hfhw.double()
                    self.maps_count[key] += 1

    def export_mean_maps(self) -> Dict[Tuple[int, int, str], torch.Tensor]:
        """Export averaged maps with shape [num_heads, F, H, W]."""
        out = {}
        for key, tensor_sum in self.maps_sum.items():
            cnt = max(1, self.maps_count[key])
            out[key] = (tensor_sum / cnt).float()
        return out

    def export_step_mean_maps(self, step: int, clear_after_export: bool = True) -> Dict[Tuple[int, int, str], torch.Tensor]:
        """Export averaged maps for one diffusion step."""
        out = {}
        keys = [k for k in self.maps_sum.keys() if k[0] == int(step)]
        for key in keys:
            cnt = max(1, self.maps_count[key])
            out[key] = (self.maps_sum[key] / cnt).float()
        if clear_after_export:
            for key in keys:
                self.maps_sum.pop(key, None)
                self.maps_count.pop(key, None)
        return out

class Wan21T2VCrossAttentionVizPatchHandle:
    """Restore handle for cross-attention visualization patch."""

    def __init__(self, target_model, original_forward, original_self_forwards, original_cross_forwards):
        self.target_model = target_model
        self.original_forward = original_forward
        self.original_self_forwards = original_self_forwards
        self.original_cross_forwards = original_cross_forwards

    def restore(self):
        self.target_model.forward = self.original_forward
        for idx, block in enumerate(self.target_model.blocks):
            block.self_attn.forward = self.original_self_forwards[idx]
            block.cross_attn.forward = self.original_cross_forwards[idx]

def _install_wan21_t2v_cross_attention_viz_patch(
    model,
    state: Wan21T2VCrossAttentionVizState,
    on_step_complete=None,
):
    """Install temporary hooks to collect cross-attention maps without changing model outputs."""
    target = model.module if (hasattr(model, "module") and hasattr(model.module, "blocks")) else model
    if not hasattr(target, "blocks"):
        raise RuntimeError("Invalid DiT model for cross-attention visualization patch.")

    original_forward = target.forward

    def patched_dit_forward(this, *args, **kwargs):
        t = kwargs.get("t", None)
        if t is None and len(args) > 1:
            t = args[1]
        completed_step = state.on_forward_start(t)
        if completed_step is not None and on_step_complete is not None:
            on_step_complete(int(completed_step))
        return original_forward(*args, **kwargs)

    target.forward = MethodType(patched_dit_forward, target)

    original_self_forwards = []
    original_cross_forwards = []

    for layer_idx, block in enumerate(target.blocks):
        original_self = block.self_attn.forward
        original_cross = block.cross_attn.forward
        original_self_forwards.append(original_self)
        original_cross_forwards.append(original_cross)

        def build_patched_self(layer_id: int, orig_self_fn):
            def patched_self(self, x, seq_lens, grid_sizes, freqs):
                state.set_layer_meta(layer_id, seq_lens, grid_sizes)
                return orig_self_fn(x, seq_lens, grid_sizes, freqs)
            return patched_self

        def build_patched_cross(layer_id: int, orig_cross_fn):
            def patched_cross(self, x, context, context_lens):
                state.collect_cross_attention(layer_id, self, x, context)
                return orig_cross_fn(x, context, context_lens)
            return patched_cross

        block.self_attn.forward = MethodType(build_patched_self(layer_idx, original_self), block.self_attn)
        block.cross_attn.forward = MethodType(build_patched_cross(layer_idx, original_cross), block.cross_attn)

    return Wan21T2VCrossAttentionVizPatchHandle(
        target_model=target,
        original_forward=original_forward,
        original_self_forwards=original_self_forwards,
        original_cross_forwards=original_cross_forwards,
    )

def _sanitize_wan21_t2v_token_name(token_name: str) -> str:
    out = re.sub(r"[^0-9A-Za-z._-]+", "_", token_name).strip("_")
    return out if out else "token"

def _uniform_wan21_t2v_frame_indices(frame_count: int, num_frames: int) -> List[int]:
    if frame_count <= 0:
        return []
    n = min(max(1, int(num_frames)), frame_count)
    if n == 1:
        return [0]
    return torch.linspace(0, frame_count - 1, steps=n).round().long().tolist()

def _resolve_wan21_t2v_viz_frame_indices(
    attention_frame_count: int,
    video_frame_count: int,
    num_frames: int,
    explicit_indices: Optional[Sequence[int]] = None,
) -> Tuple[List[int], List[int]]:
    """Resolve visualization frames from video-frame space and map to attention-frame space.

    Returns:
        attention_indices: indices for indexing attention map time dimension.
        video_indices_1based: displayed frame ids in video frame numbering (1-based).
    """
    if attention_frame_count <= 0 or video_frame_count <= 0:
        return [], []

    def _project_video_to_attention(video_idx0: int) -> int:
        if video_frame_count <= 1 or attention_frame_count <= 1:
            return 0
        idx = round(float(video_idx0) * float(attention_frame_count - 1) / float(video_frame_count - 1))
        return max(0, min(attention_frame_count - 1, int(idx)))

    video_indices_0based: List[int]
    if explicit_indices:
        raw = [int(i) for i in explicit_indices]
        if not raw:
            raise ValueError("viz_frame_indices is empty.")

        # If all values lie in [1, video_frame_count], interpret as 1-based.
        if all(1 <= i <= video_frame_count for i in raw):
            video_indices_0based = sorted(set(i - 1 for i in raw))
        else:
            video_indices_0based = sorted(set(i for i in raw if 0 <= i < video_frame_count))

        if not video_indices_0based:
            raise ValueError(
                f"viz_frame_indices has no valid index in [0, {max(0, video_frame_count - 1)}] "
                f"or [1, {video_frame_count}]. "
                f"got={list(explicit_indices)}"
            )
    else:
        n = min(max(1, int(num_frames)), video_frame_count)
        if n == 1:
            video_indices_0based = [0]
        else:
            # Match user's expected rule:
            # interval=(F_video-1)//(n-1), include first and last frame.
            interval = max(1, (video_frame_count - 1) // (n - 1))
            video_indices_0based = [i * interval for i in range(n - 1)]
            video_indices_0based.append(video_frame_count - 1)

    attention_indices = [_project_video_to_attention(i) for i in video_indices_0based]
    video_indices_1based = [i + 1 for i in video_indices_0based]
    return attention_indices, video_indices_1based

def _map_wan21_t2v_token_frame_to_video_frame_label(
    token_frame_idx: int,
    token_frame_count: Optional[int],
    video_frame_count: Optional[int],
) -> int:
    """Map token-frame index to 1-based real video-frame label."""
    tf = int(token_frame_idx)
    if token_frame_count is None or video_frame_count is None:
        return tf + 1

    tf_count = int(token_frame_count)
    vf_count = int(video_frame_count)
    if tf_count <= 1 or vf_count <= 1:
        return tf + 1

    tf = max(0, min(tf_count - 1, tf))
    video_idx0 = round(float(tf) * float(vf_count - 1) / float(tf_count - 1))
    video_idx0 = max(0, min(vf_count - 1, int(video_idx0)))
    return video_idx0 + 1

def _locate_wan21_t2v_prompt_words(
    text_encoder,
    prompt: str,
    target_object_words: Sequence[str],
    target_verb_words: Sequence[str],
):
    """Locate user words in prompt tokenization and return positions with token types."""
    if not target_object_words and not target_verb_words:
        raise ValueError("At least one of target_object_words/target_verb_words must be non-empty.")

    tokenizer = text_encoder.tokenizer.tokenizer
    encoded = tokenizer(
        prompt,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=text_encoder.text_len,
        return_tensors="pt",
    )
    prompt_ids = encoded["input_ids"][0].tolist()
    prompt_tokens = tokenizer.convert_ids_to_tokens(prompt_ids)

    word_to_positions: Dict[str, List[int]] = {}
    word_to_type: Dict[str, str] = {}

    def _match_words(words: Sequence[str], token_type: str):
        for word in words:
            w = word.strip()
            if not w:
                continue

            word_ids = tokenizer(w, add_special_tokens=False)["input_ids"]
            if len(word_ids) == 0:
                raise ValueError(f"Word `{w}` becomes empty after tokenization.")

            positions = []
            span_len = len(word_ids)
            for i in range(0, len(prompt_ids) - span_len + 1):
                if prompt_ids[i:i + span_len] == word_ids:
                    positions.extend(range(i, i + span_len))

            positions = sorted(set(positions))
            if not positions:
                raise ValueError(
                    f"Word `{w}` not found in prompt tokenization. "
                    f"Prompt tokens: {prompt_tokens}"
                )
            word_to_positions[w] = positions
            word_to_type[w] = token_type

    _match_words(target_object_words, "object")
    _match_words(target_verb_words, "verb")

    if not word_to_positions:
        raise ValueError("No valid target words after parsing.")

    return word_to_positions, word_to_type, prompt_tokens

def _save_wan21_t2v_timeline_map_pdf(
    map_fhw: torch.Tensor,
    attention_frame_indices: Sequence[int],
    frame_labels: Optional[Sequence[int]],
    save_file: str,
    title: str,
    trajectory: Optional[Sequence[Tuple[float, float]]] = None,
    draw_points: bool = False,
    draw_arrows: bool = False,
    share_color_scale: bool = False,
):
    """Save one-row timeline map panels with optional trajectory points/arrows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if map_fhw.dim() != 3 or len(attention_frame_indices) == 0:
        return

    f, h, w = map_fhw.shape
    valid_indices = [int(i) for i in attention_frame_indices if 0 <= int(i) < f]
    if not valid_indices:
        return

    points: List[Tuple[float, float]] = []
    if draw_points or draw_arrows:
        if trajectory is None or len(trajectory) == 0:
            return
        valid_indices = [i for i in valid_indices if i < len(trajectory)]
        if not valid_indices:
            return
        points = [(float(trajectory[i][0]), float(trajectory[i][1])) for i in valid_indices]

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
    axes = [fig.add_subplot(gs[0, i]) for i in range(num_panels)]

    panel_points: List[Tuple[float, float]] = []
    for i, frame_idx in enumerate(valid_indices):
        ax = axes[i]
        bg = map_fhw[frame_idx].detach().float().cpu().numpy()
        ax.imshow(
            bg,
            cmap="magma",
            alpha=0.9,
            vmin=shared_vmin,
            vmax=shared_vmax,
        )

        if draw_points:
            y, x = points[i]
            marker_color = "#ff4d4f" if i == (num_panels - 1) else "#9be15d"
            ax.scatter([x], [y], s=42, c=[marker_color], edgecolors="white", linewidths=0.8, zorder=3)
            panel_points.append((float(x), float(y)))

        if frame_labels and i < len(frame_labels):
            ax.set_title(f"frame={int(frame_labels[i])}", fontsize=9)
        else:
            ax.set_title(f"frame={frame_idx}", fontsize=9)
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.axis("off")

    if draw_arrows and len(panel_points) >= 2:
        from matplotlib.patches import ConnectionPatch
        for i in range(len(panel_points) - 1):
            x0, y0 = panel_points[i]
            x1, y1 = panel_points[i + 1]
            rad = 0.18 if (i % 2 == 0) else -0.18
            arrow = ConnectionPatch(
                xyA=(x0, y0),
                xyB=(x1, y1),
                coordsA="data",
                coordsB="data",
                axesA=axes[i],
                axesB=axes[i + 1],
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=12.0,
                shrinkA=0.0,
                shrinkB=0.0,
                linewidth=1.8,
                color="#bfefff",
                alpha=0.95,
                zorder=10,
            )
            axes[i + 1].add_artist(arrow)

    fig.suptitle(title, fontsize=10, y=0.97)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.88, bottom=0.01, wspace=0.01, hspace=0.0)
    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)

def _save_wan21_t2v_cross_attention_pdf(
    map_hfhw: torch.Tensor,
    frame_indices: Sequence[int],
    frame_labels: Optional[Sequence[int]],
    save_file: str,
    title: str,
    share_color_scale: bool = False,
):
    """Save a single head's timeline-style cross-attention map as a PDF."""
    _save_wan21_t2v_timeline_map_pdf(
        map_fhw=map_hfhw,
        attention_frame_indices=frame_indices,
        frame_labels=frame_labels,
        save_file=save_file,
        title=title,
        trajectory=None,
        draw_points=False,
        draw_arrows=False,
        share_color_scale=share_color_scale,
    )

def _extract_wan21_t2v_attention_trajectory(
    map_fhw: torch.Tensor,
    power: float = 1.5,
    quantile: float = 0.8,
) -> List[Tuple[float, float]]:
    """Extract frame-wise attention trajectory by weighted centroid.

    Args:
        map_fhw: Tensor with shape [F, H, W], non-negative attention mass.
        power: Exponent for peak sharpening (`w <- w^power`).
        quantile: Keep top-q region per frame before centroid.
    """
    if map_fhw.dim() != 3:
        raise ValueError(f"Expected [F, H, W], got shape={tuple(map_fhw.shape)}")

    f, h, w = map_fhw.shape
    y_grid = torch.arange(h, dtype=torch.float32).view(h, 1)
    x_grid = torch.arange(w, dtype=torch.float32).view(1, w)
    traj: List[Tuple[float, float]] = []

    q = max(0.0, min(1.0, float(quantile)))
    p = max(1e-6, float(power))

    for frame in range(f):
        attn = map_fhw[frame].detach().float().clamp_min(0.0)

        if q > 0.0:
            thresh = torch.quantile(attn.flatten(), q)
            attn = torch.where(attn >= thresh, attn, torch.zeros_like(attn))

        if p != 1.0:
            attn = attn.pow(p)

        denom = float(attn.sum().item())
        if denom <= 1e-12:
            flat_idx = int(map_fhw[frame].reshape(-1).argmax().item())
            y = float(flat_idx // w)
            x = float(flat_idx % w)
        else:
            y = float((attn * y_grid).sum().item() / denom)
            x = float((attn * x_grid).sum().item() / denom)

        traj.append((y, x))

    return traj

def _extract_wan21_t2v_attention_region_center_trajectory(
    map_fhw: torch.Tensor,
    power: float = 1.5,
    quantile: float = 0.8,
) -> List[Tuple[float, float]]:
    """Extract per-frame center from dominant high-attention region.

    Region definition:
    1) threshold by quantile,
    2) select connected component containing frame argmax,
    3) compute weighted centroid inside that component.
    """
    if map_fhw.dim() != 3:
        raise ValueError(f"Expected [F, H, W], got shape={tuple(map_fhw.shape)}")

    f, h, w = map_fhw.shape
    q = max(0.0, min(1.0, float(quantile)))
    p = max(1e-6, float(power))
    out: List[Tuple[float, float]] = []

    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for frame_idx in range(f):
        attn = map_fhw[frame_idx].detach().float().clamp_min(0.0)
        flat = attn.reshape(-1)
        peak_idx = int(flat.argmax().item())
        peak_y = peak_idx // w
        peak_x = peak_idx % w

        if q > 0.0:
            thresh = float(torch.quantile(flat, q).item())
            mask = attn >= thresh
        else:
            mask = torch.ones_like(attn, dtype=torch.bool)

        # Ensure peak is in mask.
        if not bool(mask[peak_y, peak_x].item()):
            mask = mask.clone()
            mask[peak_y, peak_x] = True

        # Flood-fill the connected component that contains the peak.
        visited = torch.zeros_like(mask, dtype=torch.bool)
        component = torch.zeros_like(mask, dtype=torch.bool)
        q_nodes = deque([(peak_y, peak_x)])
        visited[peak_y, peak_x] = True

        while q_nodes:
            cy, cx = q_nodes.popleft()
            if not bool(mask[cy, cx].item()):
                continue
            component[cy, cx] = True
            for dy, dx in neighbors:
                ny = cy + dy
                nx = cx + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if bool(visited[ny, nx].item()):
                    continue
                visited[ny, nx] = True
                q_nodes.append((ny, nx))

        ys, xs = component.nonzero(as_tuple=True)
        if ys.numel() == 0:
            out.append((float(peak_y), float(peak_x)))
            continue

        weights = attn[ys, xs].pow(p)
        denom = float(weights.sum().item())
        if denom <= 1e-12:
            out.append((float(peak_y), float(peak_x)))
            continue
        y = float((ys.float() * weights).sum().item() / denom)
        x = float((xs.float() * weights).sum().item() / denom)
        out.append((y, x))

    return out

def _smooth_wan21_t2v_trajectory(
    trajectory: Sequence[Tuple[float, float]],
    radius: int = 2,
) -> List[Tuple[float, float]]:
    """Apply moving-average smoothing to trajectory points."""
    if radius <= 0 or len(trajectory) <= 1:
        return [(float(y), float(x)) for y, x in trajectory]

    n = len(trajectory)
    ys = torch.tensor([p[0] for p in trajectory], dtype=torch.float32)
    xs = torch.tensor([p[1] for p in trajectory], dtype=torch.float32)
    out: List[Tuple[float, float]] = []

    for i in range(n):
        l = max(0, i - radius)
        r = min(n, i + radius + 1)
        out.append((float(ys[l:r].mean().item()), float(xs[l:r].mean().item())))

    return out

def _subsample_wan21_t2v_trajectory(
    trajectory: Sequence[Tuple[float, float]],
    num_points: int,
) -> Tuple[List[int], List[Tuple[float, float]]]:
    """Uniformly subsample trajectory points. `num_points<=0` means keep all."""
    n = len(trajectory)
    if n == 0:
        return [], []
    if num_points <= 0 or num_points >= n:
        idx = list(range(n))
        return idx, [(float(y), float(x)) for y, x in trajectory]
    idx = torch.linspace(0, n - 1, steps=num_points).round().long().tolist()
    idx = sorted(set(int(i) for i in idx))
    return idx, [(float(trajectory[i][0]), float(trajectory[i][1])) for i in idx]

def _trajectory_stats_wan21_t2v(trajectory: Sequence[Tuple[float, float]]) -> Dict[str, float]:
    """Compute path statistics for one token-attention trajectory."""
    if len(trajectory) <= 1:
        return {"path_length": 0.0, "net_displacement": 0.0, "mean_step_displacement": 0.0}

    step_dist = []
    for i in range(len(trajectory) - 1):
        y0, x0 = trajectory[i]
        y1, x1 = trajectory[i + 1]
        step_dist.append(((y1 - y0) ** 2 + (x1 - x0) ** 2) ** 0.5)

    y0, x0 = trajectory[0]
    y1, x1 = trajectory[-1]
    net = ((y1 - y0) ** 2 + (x1 - x0) ** 2) ** 0.5
    return {
        "path_length": float(sum(step_dist)),
        "net_displacement": float(net),
        "mean_step_displacement": float(sum(step_dist) / max(1, len(step_dist))),
    }

def _save_wan21_t2v_token_trajectory_pdf(
    trajectory: Sequence[Tuple[float, float]],
    frame_indices: Sequence[int],
    mean_map_hw: torch.Tensor,
    save_file: str,
    title: str,
    style: str = "glow_arrow",
    arrow_stride: int = 4,
):
    """Save trajectory figure with optional glow trail and arrow direction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(trajectory) == 0:
        return

    h, w = mean_map_hw.shape
    y = [p[0] for p in trajectory]
    x = [p[1] for p in trajectory]

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 6.0))
    bg = mean_map_hw.detach().float().cpu().numpy()
    ax.imshow(bg, cmap="magma", alpha=0.5)

    if style in {"glow", "glow_arrow", "arrow_glow"}:
        # Draw wide transparent lines first to create a glow effect.
        for lw, alpha in [(10.0, 0.06), (7.0, 0.11), (5.0, 0.18)]:
            ax.plot(x, y, color="#4ee6ff", linewidth=lw, alpha=alpha, solid_capstyle="round")

    # Draw color-progressed trajectory to encode temporal order.
    if len(trajectory) > 1:
        colors = plt.cm.turbo(torch.linspace(0.1, 0.95, steps=len(trajectory) - 1).cpu().numpy())
        for i in range(len(trajectory) - 1):
            ax.plot(
                [x[i], x[i + 1]],
                [y[i], y[i + 1]],
                color=colors[i],
                linewidth=2.2,
                alpha=0.98,
                solid_capstyle="round",
            )

    use_arrow = style in {"arrow", "glow_arrow", "arrow_glow"}
    step = max(1, int(arrow_stride))
    if use_arrow and len(trajectory) > 1:
        for i in range(0, len(trajectory) - 1, step):
            dx = x[i + 1] - x[i]
            dy = y[i + 1] - y[i]
            if abs(dx) + abs(dy) < 1e-6:
                continue
            ax.arrow(
                x[i],
                y[i],
                dx,
                dy,
                width=0.02,
                head_width=0.45,
                head_length=0.55,
                color="#d9f0ff",
                alpha=0.82,
                length_includes_head=True,
            )

    ax.scatter([x[0]], [y[0]], s=50, c=["#33cc66"], marker="o", edgecolors="white", linewidths=0.8, label="start")
    ax.scatter([x[-1]], [y[-1]], s=58, c=["#ff4d4f"], marker="o", edgecolors="white", linewidths=0.8, label="end")

    if frame_indices:
        frame_text = f"frames={frame_indices[0]}..{frame_indices[-1]}, n={len(frame_indices)}"
    else:
        frame_text = "frames=none"
    ax.set_title(f"{title}\n{frame_text}", fontsize=10)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_xlabel("token-x")
    ax.set_ylabel("token-y")
    ax.grid(alpha=0.15, linestyle="--", linewidth=0.6)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.75)
    fig.tight_layout()

    _ensure_dir(os.path.dirname(save_file))
    fig.savefig(save_file, format="pdf")
    plt.close(fig)

def _catmull_rom_wan21_t2v_curve(
    points: Sequence[Tuple[float, float]],
    samples_per_segment: int = 20,
) -> List[Tuple[float, float]]:
    """Build a smooth Catmull-Rom curve through 2D points."""
    if len(points) <= 1:
        return [(float(y), float(x)) for y, x in points]

    pts = [(float(y), float(x)) for y, x in points]
    padded = [pts[0]] + pts + [pts[-1]]
    out: List[Tuple[float, float]] = []
    seg_samples = max(2, int(samples_per_segment))

    for i in range(1, len(padded) - 2):
        p0 = torch.tensor(padded[i - 1], dtype=torch.float32)
        p1 = torch.tensor(padded[i], dtype=torch.float32)
        p2 = torch.tensor(padded[i + 1], dtype=torch.float32)
        p3 = torch.tensor(padded[i + 2], dtype=torch.float32)
        for j in range(seg_samples):
            t = float(j) / float(seg_samples)
            t2 = t * t
            t3 = t2 * t
            point = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            out.append((float(point[0].item()), float(point[1].item())))
    out.append(pts[-1])
    return out

def _save_wan21_t2v_token_trajectory_timeline_pdf(
    map_fhw: torch.Tensor,
    trajectory: Sequence[Tuple[float, float]],
    attention_frame_indices: Sequence[int],
    frame_labels: Optional[Sequence[int]],
    save_file: str,
    title: str,
):
    """Save one-row per-frame timeline and connect adjacent frames with curved arrows."""
    _save_wan21_t2v_timeline_map_pdf(
        map_fhw=map_fhw,
        attention_frame_indices=attention_frame_indices,
        frame_labels=frame_labels,
        save_file=save_file,
        title=title,
        trajectory=trajectory,
        draw_points=True,
        draw_arrows=True,
    )

def _load_wan21_t2v_cross_attention_mean_maps_from_disk(
    output_dir: str,
    draw_attention_maps_path: str = "",
) -> Tuple[Dict[Tuple[int, int, str], torch.Tensor], str]:
    """Load stored cross-attention mean maps from one .pt or stream step files."""
    map_path = str(draw_attention_maps_path).strip() if draw_attention_maps_path else ""
    if map_path:
        map_path = os.path.abspath(map_path)
    else:
        map_path = os.path.join(output_dir, "cross_attention_maps.pt")

    if os.path.exists(map_path):
        loaded = torch.load(map_path, map_location="cpu")
        if not isinstance(loaded, dict):
            raise ValueError(f"Invalid map file format: {map_path}")
        return loaded, map_path

    stream_index_path = os.path.join(output_dir, "cross_attention_stream_index.json")
    if not os.path.exists(stream_index_path):
        raise FileNotFoundError(
            f"Cannot find attention maps: neither {map_path} nor {stream_index_path} exists."
        )

    with open(stream_index_path, "r", encoding="utf-8") as f:
        stream_index = json.load(f)
    step_files = stream_index.get("step_map_files", [])
    if not step_files:
        raise ValueError(f"Stream index has no step files: {stream_index_path}")

    merged: Dict[Tuple[int, int, str], torch.Tensor] = {}
    for step_file in step_files:
        sf = os.path.abspath(step_file)
        if not os.path.exists(sf):
            continue
        part = torch.load(sf, map_location="cpu")
        if not isinstance(part, dict):
            continue
        for key, value in part.items():
            merged[key] = value

    if not merged:
        raise ValueError(f"No valid step maps loaded from stream files in: {stream_index_path}")
    return merged, stream_index_path

def _load_wan21_t2v_cross_attention_token_meta(
    output_dir: str,
    words_in_maps: Sequence[str],
    target_object_words: Sequence[str],
    target_verb_words: Sequence[str],
) -> Tuple[Dict[str, List[int]], Dict[str, str], List[str]]:
    """Load token metadata for draw-only mode, with robust fallback."""
    summary_path = os.path.join(output_dir, "cross_attention_token_viz_summary.json")
    word_to_positions: Dict[str, List[int]] = {}
    word_to_type: Dict[str, str] = {}
    prompt_tokens: List[str] = []

    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        token_positions = summary.get("token_positions", {})
        token_types = summary.get("token_types", {})
        if isinstance(token_positions, dict):
            for k, v in token_positions.items():
                if isinstance(v, list):
                    word_to_positions[str(k)] = [int(x) for x in v]
        if isinstance(token_types, dict):
            word_to_type = {str(k): str(v) for k, v in token_types.items()}
        prompt_tokens = [str(x) for x in summary.get("prompt_tokens", [])]

    object_set = set(str(w).strip() for w in target_object_words if str(w).strip())
    verb_set = set(str(w).strip() for w in target_verb_words if str(w).strip())
    for word in words_in_maps:
        if word not in word_to_positions:
            word_to_positions[word] = []
        if word in word_to_type:
            continue
        if word in object_set:
            word_to_type[word] = "object"
        elif word in verb_set:
            word_to_type[word] = "verb"
        else:
            word_to_type[word] = "unknown"

    return word_to_positions, word_to_type, prompt_tokens

# ============================================================================
# Shared Map / Trajectory Analysis Helpers
# ============================================================================

def _compute_wan21_t2v_spatial_entropy_stats(
    map_fhw: torch.Tensor,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Compute entropy stats from one [F, H, W] map.

    Returns both:
    - frame-wise spatial entropy stats (normalize each frame over H*W, then average over frames),
    - video-wise spatiotemporal entropy stats (normalize once over full F*H*W support).
    """
    if map_fhw.dim() != 3:
        raise ValueError(f"Expected [F, H, W], got shape={tuple(map_fhw.shape)}")

    attn = map_fhw.detach().float().clamp_min(0.0)
    f, h, w = attn.shape
    flat = attn.reshape(f, h * w)
    denom = flat.sum(dim=1, keepdim=True).clamp_min(float(eps))
    probs = flat / denom
    log_probs = probs.clamp_min(float(eps)).log()
    entropy = -(probs * log_probs).sum(dim=1)
    top1 = probs.max(dim=1).values

    spatial_size = int(h * w)
    if spatial_size > 1:
        entropy_norm = entropy / math.log(float(spatial_size))
    else:
        entropy_norm = torch.zeros_like(entropy)
    support_ratio = entropy.exp() / float(max(spatial_size, 1))

    # Video-wise entropy on the full spatiotemporal support [F*H*W].
    flat_video = attn.reshape(-1)
    video_mass = flat_video.sum().clamp_min(float(eps))
    probs_video = flat_video / video_mass
    entropy_video = -(probs_video * probs_video.clamp_min(float(eps)).log()).sum()
    video_size = int(f * h * w)
    if video_size > 1:
        entropy_video_norm = entropy_video / math.log(float(video_size))
    else:
        entropy_video_norm = torch.tensor(0.0, dtype=entropy_video.dtype)
    top1_video = probs_video.max()
    support_ratio_video = entropy_video.exp() / float(max(video_size, 1))

    return {
        "entropy_mean": float(entropy.mean().item()),
        "entropy_std": float(entropy.std(unbiased=False).item()),
        "entropy_min": float(entropy.min().item()),
        "entropy_max": float(entropy.max().item()),
        "entropy_norm_mean": float(entropy_norm.mean().item()),
        "entropy_norm_std": float(entropy_norm.std(unbiased=False).item()),
        "entropy_norm_min": float(entropy_norm.min().item()),
        "entropy_norm_max": float(entropy_norm.max().item()),
        "top1_prob_mean": float(top1.mean().item()),
        "top1_prob_std": float(top1.std(unbiased=False).item()),
        "top1_prob_min": float(top1.min().item()),
        "top1_prob_max": float(top1.max().item()),
        "effective_support_ratio_mean": float(support_ratio.mean().item()),
        "effective_support_ratio_std": float(support_ratio.std(unbiased=False).item()),
        "effective_support_ratio_min": float(support_ratio.min().item()),
        "effective_support_ratio_max": float(support_ratio.max().item()),
        "entropy_video": float(entropy_video.item()),
        "entropy_video_norm": float(entropy_video_norm.item()),
        "top1_prob_video": float(top1_video.item()),
        "effective_support_ratio_video": float(support_ratio_video.item()),
        "frame_count": int(f),
        "token_grid_h": int(h),
        "token_grid_w": int(w),
        "spatial_size": int(spatial_size),
    }

def _mean_wan21_t2v_headmean_map_for_words(
    mean_maps: Dict[Tuple[int, int, str], torch.Tensor],
    step: int,
    layer: int,
    words: Sequence[str],
) -> Optional[torch.Tensor]:
    """Return mean [F, H, W] map across words after head averaging."""
    maps: List[torch.Tensor] = []
    for word in words:
        key = (int(step), int(layer), str(word))
        tensor = mean_maps.get(key)
        if tensor is None:
            continue
        maps.append(tensor.float().mean(dim=0))
    if not maps:
        return None
    return torch.stack(maps, dim=0).mean(dim=0)

def _mean_wan21_t2v_head_maps_for_words(
    mean_maps: Dict[Tuple[int, int, str], torch.Tensor],
    step: int,
    layer: int,
    words: Sequence[str],
) -> Optional[torch.Tensor]:
    """Return mean [num_heads, F, H, W] map across words."""
    maps: List[torch.Tensor] = []
    for word in words:
        key = (int(step), int(layer), str(word))
        tensor = mean_maps.get(key)
        if tensor is None:
            continue
        maps.append(tensor.float())
    if not maps:
        return None
    return torch.stack(maps, dim=0).mean(dim=0)

def _load_wan21_t2v_csv_rows(csv_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

def _resample_wan21_t2v_trajectory(
    trajectory: Sequence[Tuple[float, float]],
    num_points: int,
) -> List[Tuple[float, float]]:
    """Resample trajectory to a fixed number of points using linear interpolation."""
    if not trajectory:
        return []
    if len(trajectory) == 1:
        p = (float(trajectory[0][0]), float(trajectory[0][1]))
        return [p for _ in range(max(1, num_points))]

    n = len(trajectory)
    m = max(2, int(num_points))
    src = torch.linspace(0.0, 1.0, steps=n)
    dst = torch.linspace(0.0, 1.0, steps=m)
    ys = torch.tensor([p[0] for p in trajectory], dtype=torch.float32)
    xs = torch.tensor([p[1] for p in trajectory], dtype=torch.float32)

    out = []
    for t in dst:
        idx = int(torch.searchsorted(src, t, right=False).item())
        if idx <= 0:
            out.append((float(ys[0].item()), float(xs[0].item())))
            continue
        if idx >= n:
            out.append((float(ys[-1].item()), float(xs[-1].item())))
            continue
        t0 = float(src[idx - 1].item())
        t1 = float(src[idx].item())
        alpha = 0.0 if t1 == t0 else (float(t.item()) - t0) / (t1 - t0)
        y = float((1 - alpha) * ys[idx - 1].item() + alpha * ys[idx].item())
        x = float((1 - alpha) * xs[idx - 1].item() + alpha * xs[idx].item())
        out.append((y, x))
    return out

def _dtw_wan21_t2v_trajectory_distance(
    traj_a: Sequence[Tuple[float, float]],
    traj_b: Sequence[Tuple[float, float]],
) -> float:
    """Compute DTW distance between two 2D trajectories."""
    na = len(traj_a)
    nb = len(traj_b)
    if na == 0 or nb == 0:
        return float("nan")

    dp = [[float("inf")] * (nb + 1) for _ in range(na + 1)]
    dp[0][0] = 0.0
    for i in range(1, na + 1):
        ay, ax = traj_a[i - 1]
        for j in range(1, nb + 1):
            by, bx = traj_b[j - 1]
            cost = ((ay - by) ** 2 + (ax - bx) ** 2) ** 0.5
            dp[i][j] = cost + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return float(dp[na][nb])

def _normalize_wan21_t2v_attention_map_per_frame(
    map_fhw: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Normalize one `[F, H, W]` attention map over spatial dimensions for each frame."""
    if map_fhw.dim() != 3:
        raise ValueError(f"Expected [F, H, W], got shape={tuple(map_fhw.shape)}")

    attn = map_fhw.detach().float().clamp_min(0.0)
    frame_count, token_grid_height, token_grid_width = attn.shape
    flat = attn.reshape(frame_count, token_grid_height * token_grid_width)
    denom = flat.sum(dim=1, keepdim=True).clamp_min(float(eps))
    normalized = flat / denom
    return normalized.reshape(frame_count, token_grid_height, token_grid_width)

def _soft_center_wan21_t2v_attention_map(
    probability_map_fhw: torch.Tensor,
) -> torch.Tensor:
    """Compute soft center trajectory from one normalized `[F, H, W]` map.

    Returns:
        Tensor: shape `[F, 2]`, each row is `(y, x)`.
    """
    if probability_map_fhw.dim() != 3:
        raise ValueError(f"Expected [F, H, W], got shape={tuple(probability_map_fhw.shape)}")

    probability_map = probability_map_fhw.detach().float()
    frame_count, token_grid_height, token_grid_width = probability_map.shape
    y_coords = torch.arange(token_grid_height, dtype=torch.float32).view(1, token_grid_height, 1)
    x_coords = torch.arange(token_grid_width, dtype=torch.float32).view(1, 1, token_grid_width)

    center_y = (probability_map * y_coords).sum(dim=(1, 2))
    center_x = (probability_map * x_coords).sum(dim=(1, 2))
    return torch.stack([center_y, center_x], dim=-1)

def _js_wan21_t2v_distance_per_frame(
    probability_map_a_fhw: torch.Tensor,
    probability_map_b_fhw: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute frame-wise Jensen-Shannon distance between two `[F, H, W]` distributions."""
    if tuple(probability_map_a_fhw.shape) != tuple(probability_map_b_fhw.shape):
        raise ValueError(
            "Expected same shapes for JS distance, "
            f"got {tuple(probability_map_a_fhw.shape)} vs {tuple(probability_map_b_fhw.shape)}"
        )

    flat_a = probability_map_a_fhw.reshape(probability_map_a_fhw.size(0), -1).clamp_min(float(eps))
    flat_b = probability_map_b_fhw.reshape(probability_map_b_fhw.size(0), -1).clamp_min(float(eps))
    mixture = 0.5 * (flat_a + flat_b)
    kl_a = (flat_a * (flat_a.log() - mixture.log())).sum(dim=1)
    kl_b = (flat_b * (flat_b.log() - mixture.log())).sum(dim=1)
    js_divergence = 0.5 * (kl_a + kl_b)
    return js_divergence.clamp_min(0.0).sqrt()

def _wasserstein_approx_wan21_t2v_distance_per_frame(
    probability_map_a_fhw: torch.Tensor,
    probability_map_b_fhw: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Approximate frame-wise Wasserstein distance via soft-center displacement."""
    del eps
    if tuple(probability_map_a_fhw.shape) != tuple(probability_map_b_fhw.shape):
        raise ValueError(
            "Expected same shapes for Wasserstein approximation, "
            f"got {tuple(probability_map_a_fhw.shape)} vs {tuple(probability_map_b_fhw.shape)}"
        )

    center_a = _soft_center_wan21_t2v_attention_map(probability_map_a_fhw)
    center_b = _soft_center_wan21_t2v_attention_map(probability_map_b_fhw)
    return (center_a - center_b).pow(2).sum(dim=-1).sqrt()

def _trajectory_distance_wan21_t2v_soft_centers(
    center_traj_a: torch.Tensor,
    center_traj_b: torch.Tensor,
) -> float:
    """Compute mean per-frame L2 distance between two soft-center trajectories `[F, 2]`."""
    if tuple(center_traj_a.shape) != tuple(center_traj_b.shape):
        raise ValueError(
            "Soft-center trajectory shapes must match, "
            f"got {tuple(center_traj_a.shape)} vs {tuple(center_traj_b.shape)}"
        )
    return float((center_traj_a - center_traj_b).pow(2).sum(dim=-1).sqrt().mean().item())
