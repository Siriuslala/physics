"""Experiment runners for Wan2.1-T2V with runtime monkey patches.

The runners in this file are model-specific (`wan21_t2v`) and do not modify
Wan2.1 source files. They support both single-GPU and torchrun multi-GPU modes.
"""

import csv
import json
import math
import os
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from types import MethodType
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.cuda.amp as amp
import torch.distributed as dist

from .wan21_t2v_experiment_patch import (
    Wan21T2VAttentionProbeConfig,
    Wan21T2VCausalAttentionConfig,
    Wan21T2VPatchBundleConfig,
    Wan21T2VRopePatchConfig,
    install_wan21_t2v_dit_patch_stack,
)

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
root_dir = Path(os.getenv('ROOT_DIR', Path(__file__).parent))
data_dir = Path(os.getenv('DATA_DIR'))
work_dir = Path(os.getenv('WORK_DIR'))

sys.path.append(root_dir.as_posix())

from projects.Wan2_1 import wan
from projects.Wan2_1.wan.utils.utils import cache_video

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


def _unwrap_wan21_t2v_dit_model_for_runtime_patch(model):
    """Return inner DiT module that owns `blocks` and `forward`."""
    if hasattr(model, "module") and hasattr(model.module, "blocks"):
        return model.module
    return model


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


def run_wan21_t2v_rope_axis_ablation(
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
    rope_modes: Sequence[str] = ("full", "no_f", "no_h", "no_w", "only_f", "only_hw"),
    parallel_cfg: Optional[Wan21T2VParallelConfig] = None,
):
    """Run RoPE axis ablation for Wan2.1-T2V."""
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

    rows = []
    for mode in rope_modes:
        if mode == "full":
            # Strict baseline: do not install any monkey patch in full mode.
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
        else:
            patch_cfg = Wan21T2VPatchBundleConfig(
                rope=Wan21T2VRopePatchConfig(enabled=True, mode=mode),
                probe=Wan21T2VAttentionProbeConfig(enabled=False),
                causal=Wan21T2VCausalAttentionConfig(enabled=False),
            )

            video, _ = _run_wan21_t2v_once_with_patch(
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

        if runtime.rank == 0:
            video_path = os.path.join(output_dir, f"wan21_{task}-rope_axis-mode_{mode}-seed_{seed}-shift_{shift}-scale_{guide_scale}-frame_{frame_num}.mp4")
            _save_wan21_t2v_video(video, video_path, fps=cfg.sample_fps)
            rows.append({
                "rope_mode": mode,
                "video_path": video_path,
                "seed": seed,
                "frame_num": frame_num,
                "size": f"{size[0]}x{size[1]}",
                "sampling_steps": sampling_steps,
                "guide_scale": guide_scale,
                "shift": shift,
                "sample_solver": sample_solver,
                "task": task,
            })

    if dist.is_initialized():
        dist.barrier()

    if runtime.rank == 0:
        _ensure_dir(output_dir)
        meta = {
            "experiment": "wan21_t2v_rope_axis_ablation",
            "prompt": prompt,
            "rows": rows,
        }
        _save_json(os.path.join(output_dir, "rope_axis_ablation_summary.json"), meta)
        _save_csv(os.path.join(output_dir, "rope_axis_ablation_summary.csv"), rows)
        return meta
    
    return None


def run_wan21_t2v_attention_dt_profile(
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
    probe_steps: Sequence[int] = (1, 2, 3),
    query_frame_count: int = 8,
    query_mode: str = "center",
    probe_branch: str = "uncond",
    object_token_trajectory: Optional[Dict[int, Tuple[float, float]]] = None,
    parallel_cfg: Optional[Wan21T2VParallelConfig] = None,
):
    """Run early-step attention dt profiling (P(|dt|))."""
    if query_mode == "object_guided" and not object_token_trajectory:
        raise ValueError("query_mode=object_guided requires object_token_trajectory.")

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
            probe_steps=tuple(probe_steps),
            query_frame_count=query_frame_count,
            query_mode=query_mode,
            probe_branch=probe_branch,
            use_abs_dt=True,
            object_token_trajectory=object_token_trajectory,
            collect_maas_maps=False,
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

    if runtime.rank == 0:
        _ensure_dir(output_dir)

        video_path = os.path.join(output_dir, f"wan21_t2v_attention_dt_profile_seed_{seed}.mp4")
        _save_wan21_t2v_video(video, video_path, fps=cfg.sample_fps)

        dt_tensors = state.export_dt_histograms()
        dt_path = os.path.join(output_dir, "attention_dt_histograms.pt")
        torch.save(dt_tensors, dt_path)
        dt_viz = _export_wan21_t2v_dt_profile_visualizations(
            dt_tensors=dt_tensors,
            output_dir=output_dir,
            file_prefix="attention_dt_profile",
        )

        summary = {
            "experiment": "wan21_t2v_attention_dt_profile",
            "prompt": prompt,
            "video_path": video_path,
            "dt_hist_path": dt_path,
            "dt_hist_summary": _summarize_wan21_t2v_dt_hist(dt_tensors),
            "dt_visualizations": dt_viz,
            "probe_steps": list(probe_steps),
            "query_frame_count": query_frame_count,
            "query_mode": query_mode,
            "probe_branch": probe_branch,
        }
        _save_json(os.path.join(output_dir, "attention_dt_profile_summary.json"), summary)
        return summary
    return None


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


def run_wan21_t2v_motion_aligned_attention(
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
    probe_steps: Sequence[int] = (1, 2, 3),
    query_frame_count: int = 8,
    query_mode: str = "center",
    probe_branch: str = "uncond",
    maas_layers: Sequence[int] = (0, 10, 20, 30, 39),
    maas_radius: int = 1,
    motion_target_source: str = "motion_centroid",  # motion_centroid or object_token_trajectory
    object_token_trajectory: Optional[Dict[int, Tuple[float, float]]] = None,
    export_motion_alignment_visualizations: bool = True,
    parallel_cfg: Optional[Wan21T2VParallelConfig] = None,
):
    """Run MAAS analysis from early-step attention and motion targets."""
    if query_mode == "object_guided" and not object_token_trajectory:
        raise ValueError("query_mode=object_guided requires object_token_trajectory.")

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
            probe_steps=tuple(probe_steps),
            query_frame_count=query_frame_count,
            query_mode=query_mode,
            probe_branch=probe_branch,
            use_abs_dt=True,
            object_token_trajectory=object_token_trajectory,
            collect_maas_maps=True,
            maas_steps=tuple(probe_steps),
            maas_layers=tuple(maas_layers),
            maas_radius=maas_radius,
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

    if runtime.rank == 0:
        _ensure_dir(output_dir)

        video_path = os.path.join(output_dir, f"wan21_t2v_motion_aligned_attention_seed_{seed}.mp4")
        _save_wan21_t2v_video(video, video_path, fps=cfg.sample_fps)

        token_motion_trajectory: Dict[int, Tuple[int, int]] = {}
        if motion_target_source == "object_token_trajectory":
            if not object_token_trajectory:
                raise ValueError("motion_target_source=object_token_trajectory requires object_token_trajectory.")
            if state.maas_grid_size is not None:
                _, token_h, token_w = state.maas_grid_size
                for frame, (y, x) in object_token_trajectory.items():
                    yy = max(0, min(token_h - 1, int(round(float(y)))))
                    xx = max(0, min(token_w - 1, int(round(float(x)))))
                    token_motion_trajectory[int(frame)] = (yy, xx)
            object_rows = [
                {"frame": int(frame), "y": float(y), "x": float(x)}
                for frame, (y, x) in sorted(token_motion_trajectory.items())
            ]
            _save_csv(os.path.join(output_dir, "object_token_trajectory.csv"), object_rows)
        elif motion_target_source == "motion_centroid":
            pixel_trajectory = _extract_wan21_t2v_motion_centroid_trajectory(video)
            pixel_rows = [{"frame": i, "y": float(y), "x": float(x)} for i, (y, x) in enumerate(pixel_trajectory)]
            _save_csv(os.path.join(output_dir, "motion_centroid_trajectory.csv"), pixel_rows)
            if state.maas_grid_size is not None:
                token_f, token_h, token_w = state.maas_grid_size
                _, _, video_h, video_w = video.shape
                token_motion_trajectory = _project_wan21_t2v_pixel_traj_to_tokens(
                    pixel_traj=pixel_trajectory,
                    video_h=video_h,
                    video_w=video_w,
                    token_f=token_f,
                    token_h=token_h,
                    token_w=token_w,
                    temporal_stride=4,
                )
        else:
            raise ValueError(f"Unknown motion_target_source: {motion_target_source}")

        if state.maas_grid_size is None:
            motion_alignment_rows = []
            motion_alignment_summary = {
                "maas_mean_local_ratio": 0.0,
                "maas_std_local_ratio": 0.0,
                "maas_mean_local_mass": 0.0,
                "maas_std_local_mass": 0.0,
                "maas_num_rows": 0,
            }
        else:
            motion_alignment_summary, motion_alignment_rows = state.compute_maas(
                token_motion_trajectory,
                radius=maas_radius,
            )

        if state.maas_grid_size is not None:
            token_frame_count = int(state.maas_grid_size[0])
            for row in motion_alignment_rows:
                query_frame_token = int(row.get("query_frame", 0))
                target_frame_token = query_frame_token + 1
                row["query_frame_token"] = query_frame_token
                row["target_frame_token"] = target_frame_token
                row["query_frame_video"] = _map_wan21_t2v_token_frame_to_video_frame_label(
                    token_frame_idx=query_frame_token,
                    token_frame_count=token_frame_count,
                    video_frame_count=frame_num,
                )
                row["target_frame_video"] = _map_wan21_t2v_token_frame_to_video_frame_label(
                    token_frame_idx=target_frame_token,
                    token_frame_count=token_frame_count,
                    video_frame_count=frame_num,
                )

        _save_csv(os.path.join(output_dir, "motion_aligned_attention_rows.csv"), motion_alignment_rows)
        if export_motion_alignment_visualizations:
            _export_wan21_t2v_motion_alignment_visualizations(
                state=state,
                motion_alignment_rows=motion_alignment_rows,
                token_trajectory=token_motion_trajectory,
                radius=maas_radius,
                output_dir=os.path.join(output_dir, "motion_alignment_visualizations"),
                video_frame_count=frame_num,
            )

        dt_tensors = state.export_dt_histograms()
        dt_path = os.path.join(output_dir, "motion_aligned_attention_dt_histograms.pt")
        torch.save(dt_tensors, dt_path)
        dt_viz = _export_wan21_t2v_dt_profile_visualizations(
            dt_tensors=dt_tensors,
            output_dir=output_dir,
            file_prefix="motion_aligned_attention_dt_profile",
        )

        summary = {
            "experiment": "wan21_t2v_motion_aligned_attention",
            "prompt": prompt,
            "video_path": video_path,
            "maas_summary": motion_alignment_summary,
            "dt_hist_path": dt_path,
            "dt_visualizations": dt_viz,
            "probe_steps": list(probe_steps),
            "query_frame_count": query_frame_count,
            "query_mode": query_mode,
            "probe_branch": probe_branch,
            "maas_layers": list(maas_layers),
            "maas_radius": maas_radius,
            "motion_target_source": motion_target_source,
            "maas_token_frame_count": int(state.maas_grid_size[0]) if state.maas_grid_size is not None else None,
            "video_frame_count": int(frame_num),
            "export_motion_alignment_visualizations": bool(export_motion_alignment_visualizations),
        }
        _save_json(os.path.join(output_dir, "motion_aligned_attention_summary.json"), summary)
        return summary
    return None


def run_wan21_t2v_causal_schedule(
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
    causal_first_n_steps: int = 3,
    causal_mode: str = "flat",  # flat or temporal
    attention_backend: str = "auto",  # auto / flash / torch_sdpa
    run_baseline: bool = True,
    collect_dt_profile: bool = True,
    probe_steps: Sequence[int] = (1, 2, 3),
    query_frame_count: int = 8,
    parallel_cfg: Optional[Wan21T2VParallelConfig] = None,
):
    """Run causal schedule experiment with configurable causal mode/backend.

    This function supports both:
    - flat token causal,
    - strict temporal causal.

    It also supports non-flash backend via `attention_backend=torch_sdpa`.
    """
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
    if runtime.rank == 0 and causal_mode == "temporal" and attention_backend == "torch_sdpa":
        print(
            "[wan21_t2v_experiments] causal_mode=temporal with attention_backend=torch_sdpa "
            "is memory-heavy. If you hit OOM, try attention_backend=auto."
        )

    variants: List[Tuple[str, bool, int, str]] = []
    if run_baseline:
        variants.append(("bidirectional", False, 0, "none"))
    variants.append((f"causal_{causal_mode}_{causal_first_n_steps}", True, causal_first_n_steps, causal_mode))

    rows = []
    for name, enabled, k, mode in variants:
        patch_cfg = Wan21T2VPatchBundleConfig(
            rope=Wan21T2VRopePatchConfig(enabled=True, mode="full"),
            probe=Wan21T2VAttentionProbeConfig(
                enabled=collect_dt_profile,
                probe_steps=tuple(probe_steps),
                query_frame_count=query_frame_count,
                use_abs_dt=True,
                collect_maas_maps=False,
            ),
            causal=Wan21T2VCausalAttentionConfig(
                enabled=enabled,
                causal_first_n_steps=k,
                mode=mode,
                backend=attention_backend,
            ),
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

        if runtime.rank == 0:
            video_path = os.path.join(output_dir, f"wan21_t2v_causal_schedule_{name}_seed_{seed}.mp4")
            _save_wan21_t2v_video(video, video_path, fps=cfg.sample_fps)

            dt_path = None
            dt_summary = None
            if collect_dt_profile:
                dt_tensors = state.export_dt_histograms()
                dt_path = os.path.join(output_dir, f"wan21_t2v_causal_schedule_{name}_dt_histograms.pt")
                torch.save(dt_tensors, dt_path)
                dt_summary = _summarize_wan21_t2v_dt_hist(dt_tensors)

            rows.append(
                {
                    "variant": name,
                    "causal_enabled": enabled,
                    "causal_first_n_steps": k,
                    "causal_mode": mode,
                    "attention_backend": attention_backend,
                    "video_path": video_path,
                    "dt_hist_path": dt_path,
                    "dt_hist_summary": dt_summary,
                }
            )

    if dist.is_initialized():
        dist.barrier()

    if runtime.rank == 0:
        _ensure_dir(output_dir)
        summary = {
            "experiment": "wan21_t2v_causal_schedule",
            "prompt": prompt,
            "rows": rows,
            "probe_steps": list(probe_steps),
            "query_frame_count": query_frame_count,
            "runtime_device_id": int(runtime.device_id),
            "runtime_local_rank": int(runtime.local_rank),
            "runtime_world_size": int(runtime.world_size),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        }
        _save_json(os.path.join(output_dir, "causal_schedule_summary.json"), summary)
        return summary
    return None


def run_wan21_t2v_step_window_cross_attn_off(
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
    condition_remove_start_steps: Sequence[int] = (9999,),
    condition_remove_scope: str = "from_step",  # single_step or from_step
    reuse_removed_cond_for_uncond: bool = False,
    parallel_cfg: Optional[Wan21T2VParallelConfig] = None,
):
    """Run Wan2.1-T2V step-window cross-attention removal experiment.

    For each `condition_remove_start_step = s`, this removes all cross-attn
    residual injections from diffusion step `s` and onwards.
    """
    if condition_remove_scope not in {"single_step", "from_step"}:
        raise ValueError("condition_remove_scope must be one of: single_step, from_step")
    if not condition_remove_start_steps:
        raise ValueError("condition_remove_start_steps must be non-empty.")
    start_steps = _dedup_wan21_t2v_int_list(condition_remove_start_steps)
    if any(s < 1 for s in start_steps):
        raise ValueError("condition_remove_start_steps must be >= 1.")

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

    rows = []
    for start_step in start_steps:
        handle = _install_wan21_t2v_step_window_ablation_patch(
            model=pipeline.model,
            cross_attn_remove_start_step=int(start_step),
            cross_attn_step_scope=condition_remove_scope,
            ffn_remove_step=None,
            ffn_step_scope="single_step",
            ffn_remove_layers=None,
            reuse_removed_cond_for_uncond=reuse_removed_cond_for_uncond,
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
            state = handle.state
        finally:
            handle.restore()

        if runtime.rank == 0:
            video_path = os.path.join(
                output_dir,
                f"wan21_t2v_step_window_cross_attn_off_start_{start_step:03d}_seed_{seed}.mp4",
            )
            _save_wan21_t2v_video(video, video_path, fps=cfg.sample_fps)
            rows.append(
                {
                    "condition_remove_start_step": int(start_step),
                    "video_path": video_path,
                    "cross_attn_removed_calls": int(state.cross_attn_removed_calls),
                    "ffn_removed_calls": int(state.ffn_removed_calls),
                    "uncond_reused_calls": int(state.uncond_reused_calls),
                    "total_forward_calls": int(state.total_forward_calls),
                    "sampling_steps": int(sampling_steps),
                    "seed": int(seed),
                    "task": task,
                    "sample_solver": sample_solver,
                    "shift": float(shift),
                    "guide_scale": float(guide_scale),
                    "frame_num": int(frame_num),
                    "size": f"{size[0]}x{size[1]}",
                }
            )

    if dist.is_initialized():
        dist.barrier()

    if runtime.rank == 0:
        _ensure_dir(output_dir)
        summary = {
            "experiment": "wan21_t2v_step_window_cross_attn_off",
            "prompt": prompt,
            "rows": rows,
            "condition_remove_start_steps": [int(s) for s in start_steps],
            "condition_remove_scope": condition_remove_scope,
            "reuse_removed_cond_for_uncond": bool(reuse_removed_cond_for_uncond),
            "runtime_device_id": int(runtime.device_id),
            "runtime_local_rank": int(runtime.local_rank),
            "runtime_world_size": int(runtime.world_size),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        }
        _save_json(os.path.join(output_dir, "step_window_cross_attn_off_summary.json"), summary)
        _save_csv(os.path.join(output_dir, "step_window_cross_attn_off_summary.csv"), rows)
        return summary
    return None


def run_wan21_t2v_step_window_ffn_off(
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
    timestep_idx_to_remove_ffn: Sequence[int] = (1,),
    layer_idx_to_remove_ffn: Optional[Sequence[int]] = None,
    ffn_remove_scope: str = "single_step",  # single_step or from_step
    parallel_cfg: Optional[Wan21T2VParallelConfig] = None,
):
    """Run Wan2.1-T2V step-window FFN ablation experiment.

    For each target step `s`, FFN residual updates are removed only when
    `current_step == s`, on all layers or selected layer indices.
    """
    if ffn_remove_scope not in {"single_step", "from_step"}:
        raise ValueError("ffn_remove_scope must be one of: single_step, from_step")
    if not timestep_idx_to_remove_ffn:
        raise ValueError("timestep_idx_to_remove_ffn must be non-empty.")
    remove_steps = _dedup_wan21_t2v_int_list(timestep_idx_to_remove_ffn)
    if any(s < 1 for s in remove_steps):
        raise ValueError("timestep_idx_to_remove_ffn must be >= 1.")

    layer_indices = None
    if layer_idx_to_remove_ffn is not None and len(layer_idx_to_remove_ffn) > 0:
        layer_indices = _dedup_wan21_t2v_int_list(layer_idx_to_remove_ffn)
        if any(i < 0 for i in layer_indices):
            raise ValueError("layer_idx_to_remove_ffn must be >= 0.")

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

    dit_target = _unwrap_wan21_t2v_dit_model_for_runtime_patch(pipeline.model)
    num_layers = len(dit_target.blocks)
    if layer_indices is not None:
        out_of_range = [i for i in layer_indices if i >= num_layers]
        if out_of_range:
            raise ValueError(
                f"layer_idx_to_remove_ffn out of range: {out_of_range}, num_layers={num_layers}"
            )

    rows = []
    for remove_step in remove_steps:
        handle = _install_wan21_t2v_step_window_ablation_patch(
            model=pipeline.model,
            cross_attn_remove_start_step=None,
            cross_attn_step_scope="from_step",
            ffn_remove_step=int(remove_step),
            ffn_step_scope=ffn_remove_scope,
            ffn_remove_layers=layer_indices,
            reuse_removed_cond_for_uncond=False,
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
            state = handle.state
        finally:
            handle.restore()

        if runtime.rank == 0:
            layer_tag = (
                "all"
                if layer_indices is None
                else "_".join(f"{idx:02d}" for idx in layer_indices)
            )
            video_path = os.path.join(
                output_dir,
                f"wan21_t2v_step_window_ffn_off_step_{remove_step:03d}_layers_{layer_tag}_seed_{seed}.mp4",
            )
            _save_wan21_t2v_video(video, video_path, fps=cfg.sample_fps)
            rows.append(
                {
                    "timestep_idx_to_remove_ffn": int(remove_step),
                    "layer_idx_to_remove_ffn": "all" if layer_indices is None else ",".join(str(i) for i in layer_indices),
                    "video_path": video_path,
                    "cross_attn_removed_calls": int(state.cross_attn_removed_calls),
                    "ffn_removed_calls": int(state.ffn_removed_calls),
                    "uncond_reused_calls": int(state.uncond_reused_calls),
                    "total_forward_calls": int(state.total_forward_calls),
                    "sampling_steps": int(sampling_steps),
                    "seed": int(seed),
                    "task": task,
                    "sample_solver": sample_solver,
                    "shift": float(shift),
                    "guide_scale": float(guide_scale),
                    "frame_num": int(frame_num),
                    "size": f"{size[0]}x{size[1]}",
                }
            )

    if dist.is_initialized():
        dist.barrier()

    if runtime.rank == 0:
        _ensure_dir(output_dir)
        summary = {
            "experiment": "wan21_t2v_step_window_ffn_off",
            "prompt": prompt,
            "rows": rows,
            "timestep_idx_to_remove_ffn": [int(s) for s in remove_steps],
            "layer_idx_to_remove_ffn": None if layer_indices is None else [int(i) for i in layer_indices],
            "ffn_remove_scope": ffn_remove_scope,
            "num_layers": int(num_layers),
            "runtime_device_id": int(runtime.device_id),
            "runtime_local_rank": int(runtime.local_rank),
            "runtime_world_size": int(runtime.world_size),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        }
        _save_json(os.path.join(output_dir, "step_window_ffn_off_summary.json"), summary)
        _save_csv(os.path.join(output_dir, "step_window_ffn_off_summary.csv"), rows)
        return summary
    return None


class Wan21T2VPromptReplaceState:
    """Runtime state for step-window prompt replacement in cond branch."""

    def __init__(
        self,
        replace_step: int,
        replace_scope: str,
        replace_cond_only: bool = True,
    ):
        self.replace_step = int(replace_step)
        if replace_scope not in {"single_step", "from_step"}:
            raise ValueError(f"Invalid replace_scope: {replace_scope}")
        self.replace_scope = replace_scope
        self.replace_cond_only = bool(replace_cond_only)

        self.current_step = 0
        self.current_timestep_value = None
        self.forward_call_index_in_step = 0
        self.total_forward_calls = 0
        self.replaced_context_calls = 0

    def on_forward_start(self, t_tensor):
        t_value = float(t_tensor.flatten()[0].item()) if t_tensor is not None else None
        self.total_forward_calls += 1
        if self.current_timestep_value is None or t_value != self.current_timestep_value:
            self.current_step += 1
            self.current_timestep_value = t_value
            self.forward_call_index_in_step = 0
        else:
            self.forward_call_index_in_step += 1

    def _match_scope(self) -> bool:
        if self.replace_scope == "single_step":
            return self.current_step == self.replace_step
        return self.current_step >= self.replace_step

    def should_replace_context(self) -> bool:
        if not self._match_scope():
            return False
        if self.replace_cond_only:
            # Wan generation order per step: cond forward first, then uncond.
            return self.forward_call_index_in_step == 0
        return True


class Wan21T2VPromptReplacePatchHandle:
    """Restore handle for prompt replacement patch."""

    def __init__(self, target_model, state, original_forward):
        self.target_model = target_model
        self.state = state
        self.original_forward = original_forward

    def restore(self):
        self.target_model.forward = self.original_forward


def _install_wan21_t2v_prompt_replace_patch(
    model,
    replacement_context,
    replace_step: int,
    replace_scope: str,
    replace_cond_only: bool = True,
) -> Wan21T2VPromptReplacePatchHandle:
    """Install runtime patch to replace cond prompt context at selected steps."""
    target = _unwrap_wan21_t2v_dit_model_for_runtime_patch(model)
    if not hasattr(target, "forward"):
        raise RuntimeError("Invalid DiT model for prompt replacement patch.")

    state = Wan21T2VPromptReplaceState(
        replace_step=replace_step,
        replace_scope=replace_scope,
        replace_cond_only=replace_cond_only,
    )
    original_forward = target.forward

    def patched_dit_forward(this, *args, **kwargs):
        t = kwargs.get("t", None)
        if t is None and len(args) > 1:
            t = args[1]
        state.on_forward_start(t)

        if state.should_replace_context():
            state.replaced_context_calls += 1
            if "context" in kwargs:
                kwargs = dict(kwargs)
                kwargs["context"] = replacement_context
                return original_forward(*args, **kwargs)
            if len(args) >= 3:
                args_list = list(args)
                args_list[2] = replacement_context
                return original_forward(*tuple(args_list), **kwargs)

        return original_forward(*args, **kwargs)

    target.forward = MethodType(patched_dit_forward, target)
    return Wan21T2VPromptReplacePatchHandle(
        target_model=target,
        state=state,
        original_forward=original_forward,
    )


def run_wan21_t2v_step_window_prompt_replace(
    wan21_root: str,
    ckpt_dir: str,
    output_dir: str,
    prompt: str,
    replacement_prompt: str,
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
    prompt_replace_steps: Sequence[int] = (1,),
    prompt_replace_scope: str = "from_step",  # single_step or from_step
    replace_cond_only: bool = True,
    parallel_cfg: Optional[Wan21T2VParallelConfig] = None,
):
    """Run Wan2.1-T2V step-window prompt replacement experiment."""
    if not replacement_prompt.strip():
        raise ValueError("replacement_prompt must be non-empty.")
    if prompt_replace_scope not in {"single_step", "from_step"}:
        raise ValueError("prompt_replace_scope must be one of: single_step, from_step")
    if not prompt_replace_steps:
        raise ValueError("prompt_replace_steps must be non-empty.")
    replace_steps = _dedup_wan21_t2v_int_list(prompt_replace_steps)
    if any(s < 1 for s in replace_steps):
        raise ValueError("prompt_replace_steps must be >= 1.")

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

    replacement_context = _encode_wan21_t2v_text_context_once(
        pipeline=pipeline,
        text=replacement_prompt,
        offload_model=offload_model,
    )

    rows = []
    for replace_step in replace_steps:
        handle = _install_wan21_t2v_prompt_replace_patch(
            model=pipeline.model,
            replacement_context=replacement_context,
            replace_step=int(replace_step),
            replace_scope=prompt_replace_scope,
            replace_cond_only=replace_cond_only,
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
            state = handle.state
        finally:
            handle.restore()

        if runtime.rank == 0:
            video_path = os.path.join(
                output_dir,
                f"wan21_t2v_step_window_prompt_replace_step_{replace_step:03d}_scope_{prompt_replace_scope}_seed_{seed}.mp4",
            )
            _save_wan21_t2v_video(video, video_path, fps=cfg.sample_fps)
            rows.append(
                {
                    "prompt_replace_step": int(replace_step),
                    "prompt_replace_scope": prompt_replace_scope,
                    "replace_cond_only": bool(replace_cond_only),
                    "video_path": video_path,
                    "replaced_context_calls": int(state.replaced_context_calls),
                    "total_forward_calls": int(state.total_forward_calls),
                    "sampling_steps": int(sampling_steps),
                    "seed": int(seed),
                    "task": task,
                    "sample_solver": sample_solver,
                    "shift": float(shift),
                    "guide_scale": float(guide_scale),
                    "frame_num": int(frame_num),
                    "size": f"{size[0]}x{size[1]}",
                }
            )

    if dist.is_initialized():
        dist.barrier()

    if runtime.rank == 0:
        _ensure_dir(output_dir)
        summary = {
            "experiment": "wan21_t2v_step_window_prompt_replace",
            "prompt": prompt,
            "replacement_prompt": replacement_prompt,
            "rows": rows,
            "prompt_replace_steps": [int(s) for s in replace_steps],
            "prompt_replace_scope": prompt_replace_scope,
            "replace_cond_only": bool(replace_cond_only),
            "runtime_device_id": int(runtime.device_id),
            "runtime_local_rank": int(runtime.local_rank),
            "runtime_world_size": int(runtime.world_size),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        }
        _save_json(os.path.join(output_dir, "step_window_prompt_replace_summary.json"), summary)
        _save_csv(os.path.join(output_dir, "step_window_prompt_replace_summary.csv"), rows)
        return summary
    return None


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

    num_panels = len(valid_indices)
    fig = plt.figure(figsize=(2.25 * num_panels, 2.8))
    gs = fig.add_gridspec(1, num_panels, wspace=0.01, hspace=0.0)
    axes = [fig.add_subplot(gs[0, i]) for i in range(num_panels)]

    panel_points: List[Tuple[float, float]] = []
    for i, frame_idx in enumerate(valid_indices):
        ax = axes[i]
        bg = map_fhw[frame_idx].detach().float().cpu().numpy()
        ax.imshow(bg, cmap="magma", alpha=0.9)

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
    parallel_cfg: Optional[Wan21T2VParallelConfig] = None,
):
    """Visualize video-to-text cross-attention maps for user-selected prompt words.

    Output structure:
    - output_dir/timestep_xxx/token_<word>/layer_XX_head_YY.pdf
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

        for (step, layer, word), maps in sorted(mean_maps.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
            _, f, _, _ = maps.shape
            attention_frame_indices, video_frame_labels = _resolve_wan21_t2v_viz_frame_indices(
                attention_frame_count=f,
                video_frame_count=frame_num,
                num_frames=num_viz_frames,
                explicit_indices=viz_frame_indices,
            )
            token_type = word_to_type.get(word, "unknown")
            token_dir = os.path.join(
                visualization_output_dir,
                f"timestep_{step:03d}",
                f"token_{token_type}_{_sanitize_wan21_t2v_token_name(word)}",
            )
            _ensure_dir(token_dir)

            for head in range(maps.size(0)):
                map_pdf_path = os.path.join(token_dir, f"layer_{layer:02d}_head_{head:02d}.pdf")
                if plot_attention_now:
                    _save_wan21_t2v_cross_attention_pdf(
                        map_hfhw=maps[head],
                        frame_indices=attention_frame_indices,
                        frame_labels=video_frame_labels,
                        save_file=map_pdf_path,
                        title=f"step={step} layer={layer} head={head} token={word}",
                    )
                rows.append(
                    {
                        "step": step,
                        "layer": layer,
                        "head": head,
                        "token": word,
                        "token_type": token_type,
                        "token_positions": word_to_positions.get(word, []),
                        "pdf_path": map_pdf_path if plot_attention_now else "",
                        "frame_indices": video_frame_labels,
                        "attention_frame_indices": attention_frame_indices,
                    }
                )

                if trajectory_enable and token_type == "object":
                    trajectory_raw = _extract_wan21_t2v_attention_region_center_trajectory(
                        map_fhw=maps[head],
                        power=trajectory_power,
                        quantile=trajectory_quantile,
                    )
                    # Export trajectory CSV from raw per-frame centers so downstream
                    # stages use frame-local centers unaffected by temporal smoothing.
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

                    trajectory_pdf_path = os.path.join(token_dir, f"layer_{layer:02d}_head_{head:02d}_trajectory.pdf")
                    trajectory_timeline_pdf_path = os.path.join(
                        token_dir,
                        f"layer_{layer:02d}_head_{head:02d}_trajectory_timeline.pdf",
                    )
                    if plot_trajectory_now:
                        _save_wan21_t2v_token_trajectory_pdf(
                            trajectory=trajectory_points,
                            frame_indices=trajectory_frame_indices,
                            mean_map_hw=maps[head].mean(dim=0),
                            save_file=trajectory_pdf_path,
                            title=f"step={step} layer={layer} head={head} token={word}",
                            style=trajectory_style,
                            arrow_stride=trajectory_arrow_stride,
                        )
                    if plot_trajectory_timeline_now:
                        _save_wan21_t2v_token_trajectory_timeline_pdf(
                            map_fhw=maps[head],
                            trajectory=trajectory_raw,
                            attention_frame_indices=timeline_attention_indices,
                            frame_labels=timeline_video_labels,
                            save_file=trajectory_timeline_pdf_path,
                            title=f"step={step} layer={layer} head={head} token={word}",
                        )

                    trajectory_stats = _trajectory_stats_wan21_t2v(trajectory_points)
                    rows[-1]["trajectory_pdf_path"] = trajectory_pdf_path if plot_trajectory_now else ""
                    rows[-1]["trajectory_timeline_pdf_path"] = (
                        trajectory_timeline_pdf_path if plot_trajectory_timeline_now else ""
                    )
                    rows[-1].update(
                        {
                            "trajectory_path_length": trajectory_stats["path_length"],
                            "trajectory_net_displacement": trajectory_stats["net_displacement"],
                            "trajectory_mean_step_displacement": trajectory_stats["mean_step_displacement"],
                        }
                    )
                    trajectory_summary_acc[(step, layer, word)].append(trajectory_stats["path_length"])

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
                                "trajectory_pdf_path": trajectory_pdf_path if plot_trajectory_now else "",
                            }
                        )

            if maps.size(0) > 1:
                mean_map = maps.mean(dim=0)
                mean_map_pdf_path = os.path.join(token_dir, f"layer_{layer:02d}_head_mean.pdf")
                if plot_attention_now:
                    _save_wan21_t2v_cross_attention_pdf(
                        map_hfhw=mean_map,
                        frame_indices=attention_frame_indices,
                        frame_labels=video_frame_labels,
                        save_file=mean_map_pdf_path,
                        title=f"step={step} layer={layer} head=mean token={word}",
                    )

                mean_row = {
                    "step": step,
                    "layer": layer,
                    "head": "mean",
                    "token": word,
                    "token_type": token_type,
                    "token_positions": word_to_positions.get(word, []),
                    "pdf_path": mean_map_pdf_path if plot_attention_now else "",
                    "frame_indices": video_frame_labels,
                    "attention_frame_indices": attention_frame_indices,
                }

                if trajectory_enable and token_type == "object" and trajectory_include_head_mean:
                    mean_trajectory_raw = _extract_wan21_t2v_attention_region_center_trajectory(
                        map_fhw=mean_map,
                        power=trajectory_power,
                        quantile=trajectory_quantile,
                    )
                    export_mean_indices, export_mean_points = _subsample_wan21_t2v_trajectory(
                        mean_trajectory_raw,
                        num_points=0,
                    )
                    mean_trajectory = _smooth_wan21_t2v_trajectory(
                        mean_trajectory_raw,
                        radius=trajectory_smooth_radius,
                    )
                    mean_frame_indices, mean_points = _subsample_wan21_t2v_trajectory(
                        mean_trajectory,
                        num_points=trajectory_num_frames,
                    )
                    mean_timeline_attention_indices, mean_timeline_video_labels = _resolve_wan21_t2v_viz_frame_indices(
                        attention_frame_count=len(mean_trajectory_raw),
                        video_frame_count=frame_num,
                        num_frames=trajectory_timeline_num_frames,
                        explicit_indices=None,
                    )
                    mean_trajectory_pdf_path = os.path.join(token_dir, f"layer_{layer:02d}_head_mean_trajectory.pdf")
                    mean_trajectory_timeline_pdf_path = os.path.join(
                        token_dir,
                        f"layer_{layer:02d}_head_mean_trajectory_timeline.pdf",
                    )
                    if plot_trajectory_now:
                        _save_wan21_t2v_token_trajectory_pdf(
                            trajectory=mean_points,
                            frame_indices=mean_frame_indices,
                            mean_map_hw=mean_map.mean(dim=0),
                            save_file=mean_trajectory_pdf_path,
                            title=f"step={step} layer={layer} head=mean token={word}",
                            style=trajectory_style,
                            arrow_stride=trajectory_arrow_stride,
                        )
                    if plot_trajectory_timeline_now:
                        _save_wan21_t2v_token_trajectory_timeline_pdf(
                            map_fhw=mean_map,
                            trajectory=mean_trajectory_raw,
                            attention_frame_indices=mean_timeline_attention_indices,
                            frame_labels=mean_timeline_video_labels,
                            save_file=mean_trajectory_timeline_pdf_path,
                            title=f"step={step} layer={layer} head=mean token={word}",
                        )
                    mean_stats = _trajectory_stats_wan21_t2v(mean_points)
                    mean_row.update(
                        {
                            "trajectory_pdf_path": mean_trajectory_pdf_path if plot_trajectory_now else "",
                            "trajectory_timeline_pdf_path": (
                                mean_trajectory_timeline_pdf_path if plot_trajectory_timeline_now else ""
                            ),
                            "trajectory_path_length": mean_stats["path_length"],
                            "trajectory_net_displacement": mean_stats["net_displacement"],
                            "trajectory_mean_step_displacement": mean_stats["mean_step_displacement"],
                        }
                    )
                    for frame_idx, (y, x) in zip(export_mean_indices, export_mean_points):
                        trajectory_rows.append(
                            {
                                "step": step,
                                "layer": layer,
                                "head": "mean",
                                "token": word,
                                "token_type": token_type,
                                "frame": int(frame_idx),
                                "y": float(y),
                                "x": float(x),
                                "trajectory_pdf_path": mean_trajectory_pdf_path if plot_trajectory_now else "",
                            }
                        )
                    trajectory_summary_acc[(step, layer, word)].append(mean_stats["path_length"])

                rows.append(mean_row)

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
            "save_trajectory_pdfs": bool(save_trajectory_pdfs),
            "save_video": bool(save_video),
            "draw_attention_map_only": bool(draw_attention_map_only),
            "draw_attention_maps_path": str(draw_attention_maps_path),
            "loaded_maps_source": loaded_maps_source,
            "visualization_output_dir": visualization_output_dir,
            "artifact_output_dir": artifact_output_dir,
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


def _load_wan21_t2v_csv_rows(csv_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _build_wan21_t2v_object_token_trajectory_from_csv(
    trajectory_csv_path: str,
    target_object_words: Sequence[str],
    step: Optional[int] = None,
    layer: Optional[int] = None,
    head: str = "mean",
) -> Tuple[Dict[int, Tuple[float, float]], Dict[str, object]]:
    """Load and aggregate object-token trajectory from cross-attention trajectory CSV."""
    rows = _load_wan21_t2v_csv_rows(trajectory_csv_path)
    if not rows:
        raise ValueError(f"Trajectory CSV is empty or missing: {trajectory_csv_path}")

    object_words = set(w.strip() for w in target_object_words if w.strip())
    filtered = []
    for row in rows:
        token_type = row.get("token_type", "")
        token_name = row.get("token", "")
        if token_type != "object":
            continue
        if object_words and token_name not in object_words:
            continue
        if str(row.get("head", "")) != str(head):
            continue
        filtered.append(row)
    if not filtered:
        raise ValueError(f"No object-token trajectory rows found for head={head}.")

    step_values = sorted(set(int(row["step"]) for row in filtered))
    if step is not None:
        selected_step = int(step)
        if selected_step not in step_values:
            raise ValueError(
                f"Requested object trajectory step={selected_step} is unavailable. "
                f"available_steps={step_values}"
            )
    else:
        # Default to the latest diffusion step (usually sharpest object localization).
        selected_step = step_values[-1]
    filtered = [row for row in filtered if int(row["step"]) == selected_step]
    if not filtered:
        raise ValueError(f"No rows found for selected step={selected_step}.")

    layer_values = sorted(set(int(row["layer"]) for row in filtered))
    if layer is not None:
        selected_layer = int(layer)
        if selected_layer not in layer_values:
            raise ValueError(
                f"Requested object trajectory layer={selected_layer} is unavailable at step={selected_step}. "
                f"available_layers={layer_values}"
            )
    else:
        # Prefer layer 27 if available (empirically stable), else use the deepest available layer.
        selected_layer = 27 if 27 in layer_values else layer_values[-1]
    filtered = [row for row in filtered if int(row["layer"]) == selected_layer]
    if not filtered:
        raise ValueError(f"No rows found for selected layer={selected_layer}.")

    by_frame = defaultdict(list)
    for row in filtered:
        frame = int(row["frame"])
        y = float(row["y"])
        x = float(row["x"])
        by_frame[frame].append((y, x))

    trajectory = {}
    for frame, points in by_frame.items():
        ys = [p[0] for p in points]
        xs = [p[1] for p in points]
        trajectory[int(frame)] = (float(sum(ys) / len(ys)), float(sum(xs) / len(xs)))

    if not trajectory:
        raise ValueError("Failed to build object-token trajectory from CSV.")

    metadata = {
        "selected_step": selected_step,
        "selected_layer": selected_layer,
        "selected_head": str(head),
        "selection_policy": "latest_step + prefer_layer_27_else_last_layer",
        "available_steps": step_values,
        "available_layers_at_selected_step": layer_values,
        "num_frames": len(trajectory),
    }
    return dict(sorted(trajectory.items())), metadata


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


def run_wan21_t2v_token_trajectory_seed_stability(
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
    collect_steps: Sequence[int] = (1, 2, 3),
    num_viz_frames: int = 5,
    layers_to_collect: Optional[Sequence[int]] = None,
    chunk_size: int = 1024,
    stability_num_points: int = 41,
    stability_head: str = "mean",
    trajectory_style: str = "glow_arrow",
    trajectory_smooth_radius: int = 2,
    trajectory_power: float = 1.5,
    trajectory_quantile: float = 0.8,
    trajectory_arrow_stride: int = 4,
    parallel_cfg: Optional[Wan21T2VParallelConfig] = None,
):
    """Run multi-seed trajectory stability analysis based on object-token cross-attention trajectories."""
    if not target_object_words:
        raise ValueError("token_trajectory_seed_stability requires non-empty target_object_words.")
    if not seed_list:
        raise ValueError("seed_list is empty.")

    _ensure_dir(output_dir)
    seed_rows = []
    trajectories_by_key = defaultdict(dict)  # (step, layer, token, head) -> {seed: [(y, x), ...]}

    for seed in seed_list:
        seed_output_dir = os.path.join(output_dir, f"seed_{int(seed):06d}")
        summary = run_wan21_t2v_cross_attention_token_viz(
            wan21_root=wan21_root,
            ckpt_dir=ckpt_dir,
            output_dir=seed_output_dir,
            prompt=prompt,
            size=size,
            target_object_words=target_object_words,
            target_verb_words=target_verb_words,
            task=task,
            frame_num=frame_num,
            shift=shift,
            sample_solver=sample_solver,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            seed=int(seed),
            device_id=device_id,
            offload_model=offload_model,
            collect_steps=collect_steps,
            num_viz_frames=num_viz_frames,
            layers_to_collect=layers_to_collect,
            chunk_size=chunk_size,
            trajectory_enable=True,
            trajectory_style=trajectory_style,
            trajectory_num_frames=0,
            trajectory_smooth_radius=trajectory_smooth_radius,
            trajectory_power=trajectory_power,
            trajectory_quantile=trajectory_quantile,
            trajectory_arrow_stride=trajectory_arrow_stride,
            trajectory_include_head_mean=True,
            save_attention_pdfs=False,
            save_trajectory_pdfs=False,
            save_video=False,
            parallel_cfg=parallel_cfg,
        )
        seed_rows.append({"seed": int(seed), "seed_output_dir": seed_output_dir, "summary_path": os.path.join(seed_output_dir, "cross_attention_token_viz_summary.json")})

        if summary is None:
            continue

        trajectory_csv_path = os.path.join(seed_output_dir, "cross_attention_token_trajectory.csv")
        trajectory_rows = _load_wan21_t2v_csv_rows(trajectory_csv_path)
        grouped = defaultdict(dict)  # key -> frame -> (y, x)
        for row in trajectory_rows:
            if row.get("token_type") != "object":
                continue
            if str(row.get("head", "")) != str(stability_head):
                continue
            key = (
                int(row["step"]),
                int(row["layer"]),
                row["token"],
                str(row["head"]),
            )
            grouped[key][int(row["frame"])] = (float(row["y"]), float(row["x"]))

        for key, frame_to_point in grouped.items():
            ordered = [frame_to_point[f] for f in sorted(frame_to_point.keys())]
            if ordered:
                trajectories_by_key[key][int(seed)] = ordered

    if dist.is_initialized():
        dist.barrier()

    # This function delegates heavy generation to sub-runs and aggregates only on rank 0.
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        return None

    _save_csv(os.path.join(output_dir, "seed_run_index.csv"), seed_rows)

    stability_rows = []
    plots_dir = os.path.join(output_dir, "stability_plots")
    _ensure_dir(plots_dir)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for key in sorted(trajectories_by_key.keys()):
        seed_to_traj = trajectories_by_key[key]
        seeds = sorted(seed_to_traj.keys())
        if len(seeds) < 2:
            continue

        resampled = [
            _resample_wan21_t2v_trajectory(seed_to_traj[seed], num_points=stability_num_points)
            for seed in seeds
        ]
        data = torch.tensor(resampled, dtype=torch.float32)  # [S, T, 2]
        mean_traj = data.mean(dim=0)  # [T, 2]
        residual = (data - mean_traj.unsqueeze(0)).pow(2).sum(dim=-1).sqrt()  # [S, T]
        pointwise_dispersion = float(residual.mean().item())

        path_lengths = []
        for seq in resampled:
            length = 0.0
            for i in range(len(seq) - 1):
                y0, x0 = seq[i]
                y1, x1 = seq[i + 1]
                length += ((y1 - y0) ** 2 + (x1 - x0) ** 2) ** 0.5
            path_lengths.append(length)
        path_len_t = torch.tensor(path_lengths, dtype=torch.float32)
        mean_path = float(path_len_t.mean().item())
        std_path = float(path_len_t.std(unbiased=False).item())
        path_cv = float(std_path / max(mean_path, 1e-6))

        velocity = data[:, 1:, :] - data[:, :-1, :]
        speed = velocity.pow(2).sum(dim=-1).sqrt().clamp_min(1e-8)
        unit_velocity = velocity / speed.unsqueeze(-1)
        direction_consistency = float(unit_velocity.mean(dim=0).pow(2).sum(dim=-1).sqrt().mean().item())

        dtw_values = []
        dtw_matrix = torch.zeros((len(seeds), len(seeds)), dtype=torch.float32)
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                d = _dtw_wan21_t2v_trajectory_distance(resampled[i], resampled[j])
                dtw_values.append(d)
                dtw_matrix[i, j] = float(d)
                dtw_matrix[j, i] = float(d)
        mean_dtw = float(torch.tensor(dtw_values).mean().item()) if dtw_values else 0.0

        step, layer, token, head = key
        stability_rows.append(
            {
                "step": int(step),
                "layer": int(layer),
                "token": token,
                "head": head,
                "num_seeds": len(seeds),
                "pointwise_dispersion": pointwise_dispersion,
                "path_length_mean": mean_path,
                "path_length_std": std_path,
                "path_length_cv": path_cv,
                "direction_consistency": direction_consistency,
                "pairwise_dtw_mean": mean_dtw,
            }
        )

        key_tag = f"step_{step:03d}_layer_{layer:02d}_{_sanitize_wan21_t2v_token_name(token)}_{head}"
        fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.6))
        for idx, seed in enumerate(seeds):
            seq = resampled[idx]
            ys = [p[0] for p in seq]
            xs = [p[1] for p in seq]
            ax.plot(xs, ys, alpha=0.28, linewidth=1.0, label=f"seed={seed}" if idx < 8 else None)
        mean_y = mean_traj[:, 0].tolist()
        mean_x = mean_traj[:, 1].tolist()
        ax.plot(mean_x, mean_y, color="#ff3b30", linewidth=2.2, label="mean")
        ax.set_title(f"Trajectory Stability: step={step} layer={layer} token={token} head={head}")
        ax.set_xlabel("token-x")
        ax.set_ylabel("token-y")
        ax.grid(alpha=0.2, linestyle="--")
        if len(seeds) <= 8:
            ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{key_tag}_trajectory_overlay.pdf"), format="pdf")
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(5.8, 5.1))
        im = ax.imshow(dtw_matrix.numpy(), cmap="magma")
        ax.set_title(f"Pairwise DTW: step={step} layer={layer} token={token} head={head}")
        ax.set_xlabel("seed index")
        ax.set_ylabel("seed index")
        fig.colorbar(im, ax=ax, shrink=0.82)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{key_tag}_dtw_matrix.pdf"), format="pdf")
        plt.close(fig)

    _save_csv(os.path.join(output_dir, "seed_stability_summary.csv"), stability_rows)
    summary = {
        "experiment": "wan21_t2v_token_trajectory_seed_stability",
        "prompt": prompt,
        "target_object_words": list(target_object_words),
        "target_verb_words": list(target_verb_words),
        "seed_list": [int(s) for s in seed_list],
        "stability_head": stability_head,
        "stability_num_points": int(stability_num_points),
        "num_summary_rows": len(stability_rows),
    }
    _save_json(os.path.join(output_dir, "seed_stability_summary.json"), summary)
    return summary


def run_wan21_t2v_joint_attention_suite(
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
    layers_to_collect: Optional[Sequence[int]] = None,
    query_frame_count: int = 8,
    maas_layers: Sequence[int] = (0, 10, 20, 30, 39),
    maas_radius: int = 1,
    probe_branch: str = "cond",
    motion_target_source: str = "object_token_trajectory",  # object_token_trajectory or motion_centroid
    object_trajectory_step: Optional[int] = None,
    object_trajectory_layer: Optional[int] = None,
    object_trajectory_head: str = "mean",
    reuse_cross_attention_dir: Optional[str] = None,
    parallel_cfg: Optional[Wan21T2VParallelConfig] = None,
):
    """Run a unified 3-stage suite: cross-attn -> object-guided dt profile -> motion-aligned attention."""
    _ensure_dir(output_dir)

    cross_attention_reused = bool(reuse_cross_attention_dir and str(reuse_cross_attention_dir).strip())
    cross_attention_dir = (
        os.path.abspath(str(reuse_cross_attention_dir))
        if cross_attention_reused
        else os.path.join(output_dir, "cross_attention_token_viz")
    )
    dt_profile_dir = os.path.join(output_dir, "object_guided_attention_dt_profile")
    motion_alignment_dir = os.path.join(output_dir, "motion_aligned_attention")

    cross_attention_summary = None
    if cross_attention_reused:
        trajectory_csv_path = os.path.join(cross_attention_dir, "cross_attention_token_trajectory.csv")
        if not os.path.exists(trajectory_csv_path):
            raise FileNotFoundError(
                f"reuse_cross_attention_dir is set but missing trajectory CSV: {trajectory_csv_path}"
            )
        summary_path = os.path.join(cross_attention_dir, "cross_attention_token_viz_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                cross_attention_summary = json.load(f)
        else:
            cross_attention_summary = {
                "experiment": "wan21_t2v_cross_attention_token_viz",
                "reused_cross_attention_dir": cross_attention_dir,
                "cross_attention_token_trajectory_csv": trajectory_csv_path,
            }
    else:
        cross_attention_summary = run_wan21_t2v_cross_attention_token_viz(
            wan21_root=wan21_root,
            ckpt_dir=ckpt_dir,
            output_dir=cross_attention_dir,
            prompt=prompt,
            size=size,
            target_object_words=target_object_words,
            target_verb_words=target_verb_words,
            task=task,
            frame_num=frame_num,
            shift=shift,
            sample_solver=sample_solver,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            seed=seed,
            device_id=device_id,
            offload_model=offload_model,
            collect_steps=collect_steps,
            num_viz_frames=5,
            layers_to_collect=layers_to_collect,
            chunk_size=1024,
            trajectory_enable=True,
            trajectory_style="glow_arrow",
            trajectory_num_frames=0,
            trajectory_smooth_radius=2,
            trajectory_power=1.5,
            trajectory_quantile=0.8,
            trajectory_arrow_stride=4,
            trajectory_include_head_mean=True,
            save_attention_pdfs=True,
            save_trajectory_pdfs=True,
            save_video=True,
            parallel_cfg=parallel_cfg,
        )

    rank = dist.get_rank() if dist.is_initialized() else 0
    object_token_trajectory = None
    object_trajectory_metadata = None
    if rank == 0:
        csv_point_mode = ""
        if isinstance(cross_attention_summary, dict):
            csv_point_mode = str(cross_attention_summary.get("trajectory_csv_point_mode", "")).strip()
        if csv_point_mode and csv_point_mode != "raw_region_center_per_frame":
            raise ValueError(
                "cross_attention trajectory CSV was not exported from raw per-frame centers. "
                "Please rerun cross_attention_token_viz with current code to regenerate "
                "cross_attention_token_trajectory.csv."
            )

        trajectory_csv_path = os.path.join(cross_attention_dir, "cross_attention_token_trajectory.csv")
        object_token_trajectory, object_trajectory_metadata = _build_wan21_t2v_object_token_trajectory_from_csv(
            trajectory_csv_path=trajectory_csv_path,
            target_object_words=target_object_words,
            step=object_trajectory_step,
            layer=object_trajectory_layer,
            head=object_trajectory_head,
        )
        object_trajectory_metadata["trajectory_source_csv"] = trajectory_csv_path
        object_trajectory_metadata["trajectory_source_dir"] = cross_attention_dir
        object_trajectory_metadata["cross_attention_reused"] = cross_attention_reused
        object_trajectory_metadata["trajectory_csv_point_mode"] = (
            csv_point_mode if csv_point_mode else "unknown(no-summary)"
        )
    if dist.is_initialized():
        payload = [object_token_trajectory]
        dist.broadcast_object_list(payload, src=0)
        object_token_trajectory = payload[0]

    dt_profile_summary = run_wan21_t2v_attention_dt_profile(
        wan21_root=wan21_root,
        ckpt_dir=ckpt_dir,
        output_dir=dt_profile_dir,
        prompt=prompt,
        size=size,
        task=task,
        frame_num=frame_num,
        shift=shift,
        sample_solver=sample_solver,
        sampling_steps=sampling_steps,
        guide_scale=guide_scale,
        seed=seed,
        device_id=device_id,
        offload_model=offload_model,
        probe_steps=collect_steps,
        query_frame_count=query_frame_count,
        query_mode="object_guided",
        probe_branch=probe_branch,
        object_token_trajectory=object_token_trajectory,
        parallel_cfg=parallel_cfg,
    )

    motion_alignment_summary = run_wan21_t2v_motion_aligned_attention(
        wan21_root=wan21_root,
        ckpt_dir=ckpt_dir,
        output_dir=motion_alignment_dir,
        prompt=prompt,
        size=size,
        task=task,
        frame_num=frame_num,
        shift=shift,
        sample_solver=sample_solver,
        sampling_steps=sampling_steps,
        guide_scale=guide_scale,
        seed=seed,
        device_id=device_id,
        offload_model=offload_model,
        probe_steps=collect_steps,
        query_frame_count=query_frame_count,
        query_mode="object_guided",
        probe_branch=probe_branch,
        maas_layers=maas_layers,
        maas_radius=maas_radius,
        motion_target_source=motion_target_source,
        object_token_trajectory=object_token_trajectory,
        export_motion_alignment_visualizations=True,
        parallel_cfg=parallel_cfg,
    )

    if rank == 0:
        joint_summary = {
            "experiment": "wan21_t2v_joint_attention_suite",
            "prompt": prompt,
            "target_object_words": list(target_object_words),
            "target_verb_words": list(target_verb_words),
            "collect_steps": list(collect_steps),
            "probe_branch": probe_branch,
            "motion_target_source": motion_target_source,
            "object_trajectory_request": {
                "step": object_trajectory_step,
                "layer": object_trajectory_layer,
                "head": object_trajectory_head,
            },
            "cross_attention_reused": cross_attention_reused,
            "reuse_cross_attention_dir": cross_attention_dir if cross_attention_reused else "",
            "object_trajectory_metadata": object_trajectory_metadata,
            "reports": {
                "cross_attention_token_viz": os.path.join(cross_attention_dir, "cross_attention_token_viz_summary.json"),
                "object_guided_attention_dt_profile": os.path.join(dt_profile_dir, "attention_dt_profile_summary.json"),
                "motion_aligned_attention": os.path.join(motion_alignment_dir, "motion_aligned_attention_summary.json"),
            },
            "sub_summaries": {
                "cross_attention_token_viz": cross_attention_summary,
                "object_guided_attention_dt_profile": dt_profile_summary,
                "motion_aligned_attention": motion_alignment_summary,
            },
        }
        _save_json(os.path.join(output_dir, "joint_attention_suite_summary.json"), joint_summary)
        return joint_summary
    return None
