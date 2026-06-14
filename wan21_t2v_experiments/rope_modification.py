"""Wan2.1-T2V experiment: rope_modification.

Main entry:
- run_wan21_t2v_rope_modification

This experiment studies RoPE modification for Wan2.1 T2V with three modes:
1. `manual`: training-free axis-wise lambda scaling.
2. `spatial_temporal_reweight`: training-free post-RoPE temporal/spatial
   channel reweighting before self-attention.
3. `timestep_conditioned`: a trainable scale learner with `global` and
   `head_aware` sub-modes, attached by monkey patching so its parameters can
   later join a training pipeline and appear in `state_dict`.
"""

import os
from typing import Optional, Sequence, Tuple

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
    _save_wan21_t2v_video,
)
from .wan21_t2v_experiment_patch import (
    Wan21T2VAttentionProbeConfig,
    Wan21T2VCausalAttentionConfig,
    Wan21T2VPatchBundleConfig,
    Wan21T2VRopePatchConfig,
    Wan21T2VSemanticResidualConfig,
)


def _format_rope_modification_steps_tag(steps: Sequence[int]) -> str:
    if not steps:
        return "all_steps"
    return "steps_" + "-".join(str(int(v)) for v in steps)


def _format_value_list_tag(values: Sequence[int]) -> str:
    if not values:
        return "all"
    return "-".join(str(int(v)) for v in values)


def _build_rope_modification_video_filename(
    rope_modification_mode: str,
    rope_modification_lambda_f: float,
    rope_modification_lambda_h: float,
    rope_modification_lambda_w: float,
    rope_modification_steps: Sequence[int],
    rope_modification_spatial_temporal_reweight_alpha: float,
    rope_modification_timestep_conditioned_resolution: str,
    rope_modification_semantic_residual_enabled: bool,
    rope_modification_semantic_residual_alpha: float,
    rope_modification_semantic_residual_steps: Sequence[int],
    rope_modification_semantic_residual_timestep_conditioned: bool,
    rope_modification_semantic_residual_timestep_conditioned_resolution: str,
) -> str:
    tags = [f"lambda_modify_mode_{rope_modification_mode}"]

    if rope_modification_mode == "manual":
        tags.extend(
            [
                f"lambdaf_{rope_modification_lambda_f}",
                f"lambdah_{rope_modification_lambda_h}",
                f"lambdaw_{rope_modification_lambda_w}",
                f"lambda_steps_{_format_value_list_tag(rope_modification_steps)}",
            ]
        )
    elif rope_modification_mode == "spatial_temporal_reweight":
        tags.extend(
            [
                f"spatial_temporal_reweight_alpha_{rope_modification_spatial_temporal_reweight_alpha}",
                f"spatial_temporal_reweight_steps_{_format_value_list_tag(rope_modification_steps)}",
            ]
        )
    elif rope_modification_mode == "timestep_conditioned":
        tags.append(
            f"lambda_timestep_condition_mode_{rope_modification_timestep_conditioned_resolution}"
        )
    else:
        raise ValueError(
            "rope_modification_mode must be one of "
            "{'manual', 'spatial_temporal_reweight', 'timestep_conditioned'}, "
            f"got {rope_modification_mode!r}."
        )

    if rope_modification_semantic_residual_enabled:
        tags.extend(
            [
                "use_semantic_SA",
                f"semantic_alpha_{rope_modification_semantic_residual_alpha}",
                f"semantic_steps_{_format_value_list_tag(rope_modification_semantic_residual_steps)}",
            ]
        )
        if rope_modification_semantic_residual_timestep_conditioned:
            tags.append(
                "semantic_timestep_condition_mode_"
                f"{rope_modification_semantic_residual_timestep_conditioned_resolution}"
            )

    return "-".join(tags) + ".mp4"


def run_wan21_t2v_rope_modification(
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
    rope_modification_mode: str = "manual",
    rope_modification_lambda_f: float = 1.0,
    rope_modification_lambda_h: float = 1.0,
    rope_modification_lambda_w: float = 1.0,
    rope_modification_steps: Sequence[int] = (),
    rope_modification_spatial_temporal_reweight_alpha: float = 0.5,
    rope_modification_timestep_conditioned_resolution: str = "global",
    rope_modification_timestep_conditioned_hidden_dim: int = 128,
    rope_modification_timestep_conditioned_checkpoint: str = "",
    rope_modification_semantic_residual_enabled: bool = False,
    rope_modification_semantic_residual_alpha: float = 0.0,
    rope_modification_semantic_residual_steps: Sequence[int] = (),
    rope_modification_semantic_residual_query_chunk_size: int = 64,
    rope_modification_semantic_residual_timestep_conditioned: bool = False,
    rope_modification_semantic_residual_timestep_conditioned_resolution: str = "global",
    rope_modification_semantic_residual_timestep_conditioned_hidden_dim: int = 128,
    rope_modification_semantic_residual_timestep_conditioned_checkpoint: str = "",
    parallel_cfg: Optional[Wan21T2VParallelConfig] = None,
):
    """Run RoPE modification with manual, reweight, or timestep-conditioned scaling."""
    parallel_cfg = parallel_cfg or Wan21T2VParallelConfig()
    runtime = _init_wan21_t2v_runtime(parallel_cfg, explicit_device_id=device_id)
    seed = _broadcast_seed_if_needed(seed, runtime)

    if rope_modification_mode not in {"manual", "spatial_temporal_reweight", "timestep_conditioned"}:
        raise ValueError(
            "rope_modification_mode must be one of "
            "{'manual', 'spatial_temporal_reweight', 'timestep_conditioned'}, "
            f"got {rope_modification_mode!r}."
        )
    if not (0.0 <= float(rope_modification_spatial_temporal_reweight_alpha) <= 1.0):
        raise ValueError(
            "rope_modification_spatial_temporal_reweight_alpha must lie in [0, 1], "
            f"got {rope_modification_spatial_temporal_reweight_alpha}."
        )
    if rope_modification_timestep_conditioned_resolution not in {"global", "head_aware"}:
        raise ValueError(
            "rope_modification_timestep_conditioned_resolution must be one of "
            "{'global', 'head_aware'}, "
            f"got {rope_modification_timestep_conditioned_resolution!r}."
        )
    if rope_modification_semantic_residual_timestep_conditioned_resolution not in {"global", "head_aware"}:
        raise ValueError(
            "rope_modification_semantic_residual_timestep_conditioned_resolution must be one of "
            "{'global', 'head_aware'}, "
            f"got {rope_modification_semantic_residual_timestep_conditioned_resolution!r}."
        )

    pipeline, cfg = _build_wan21_t2v_pipeline(
        wan21_root=wan21_root,
        ckpt_dir=ckpt_dir,
        task=task,
        runtime=runtime,
        parallel_cfg=parallel_cfg,
    )
    offload_model = _resolve_wan21_t2v_offload_model(runtime, offload_model)

    patch_cfg = Wan21T2VPatchBundleConfig(
        rope=Wan21T2VRopePatchConfig(
            enabled=True,
            mode="full",
            lambda_f=float(rope_modification_lambda_f),
            lambda_h=float(rope_modification_lambda_h),
            lambda_w=float(rope_modification_lambda_w),
            apply_steps=tuple(int(v) for v in rope_modification_steps),
            scale_mode=str(rope_modification_mode),
            spatial_temporal_reweight_alpha=float(rope_modification_spatial_temporal_reweight_alpha),
            timestep_conditioned_resolution=str(rope_modification_timestep_conditioned_resolution),
            timestep_conditioned_hidden_dim=int(rope_modification_timestep_conditioned_hidden_dim),
            timestep_conditioned_checkpoint=str(rope_modification_timestep_conditioned_checkpoint).strip(),
        ),
        semantic=Wan21T2VSemanticResidualConfig(
            enabled=bool(rope_modification_semantic_residual_enabled),
            alpha=float(rope_modification_semantic_residual_alpha),
            apply_steps=tuple(int(v) for v in rope_modification_semantic_residual_steps),
            query_chunk_size=int(rope_modification_semantic_residual_query_chunk_size),
            timestep_conditioned=bool(rope_modification_semantic_residual_timestep_conditioned),
            timestep_conditioned_resolution=str(rope_modification_semantic_residual_timestep_conditioned_resolution),
            timestep_conditioned_hidden_dim=int(rope_modification_semantic_residual_timestep_conditioned_hidden_dim),
            timestep_conditioned_checkpoint=str(rope_modification_semantic_residual_timestep_conditioned_checkpoint).strip(),
        ),
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

    if dist.is_initialized():
        dist.barrier()

    if runtime.rank != 0:
        return None

    _ensure_dir(output_dir)
    video_path = os.path.join(
        output_dir,
        _build_rope_modification_video_filename(
            rope_modification_mode=rope_modification_mode,
            rope_modification_lambda_f=rope_modification_lambda_f,
            rope_modification_lambda_h=rope_modification_lambda_h,
            rope_modification_lambda_w=rope_modification_lambda_w,
            rope_modification_steps=rope_modification_steps,
            rope_modification_spatial_temporal_reweight_alpha=rope_modification_spatial_temporal_reweight_alpha,
            rope_modification_timestep_conditioned_resolution=rope_modification_timestep_conditioned_resolution,
            rope_modification_semantic_residual_enabled=rope_modification_semantic_residual_enabled,
            rope_modification_semantic_residual_alpha=rope_modification_semantic_residual_alpha,
            rope_modification_semantic_residual_steps=rope_modification_semantic_residual_steps,
            rope_modification_semantic_residual_timestep_conditioned=rope_modification_semantic_residual_timestep_conditioned,
            rope_modification_semantic_residual_timestep_conditioned_resolution=rope_modification_semantic_residual_timestep_conditioned_resolution,
        ),
    )
    _save_wan21_t2v_video(video, video_path, fps=cfg.sample_fps)

    return {"video_path": video_path}
