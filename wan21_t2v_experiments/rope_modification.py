"""Wan2.1-T2V experiment: rope_modification.

Main entry:
- run_wan21_t2v_rope_modification

This experiment studies RoPE modification for Wan2.1 T2V with two modes:
1. `manual`: training-free axis-wise lambda scaling.
2. `step_conditioned`: a timestep-conditioned scaling head whose parameters
   are attached to the model through monkey patching so they can later join
   a training pipeline and appear in `state_dict`.
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


def _format_rope_modification_steps_tag(steps: Sequence[int]) -> str:
    if not steps:
        return "all_steps"
    return "steps_" + "-".join(str(int(v)) for v in steps)


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
    rope_modification_step_conditioned_hidden_dim: int = 128,
    rope_modification_step_conditioned_checkpoint: str = "",
    parallel_cfg: Optional[Wan21T2VParallelConfig] = None,
):
    """Run RoPE modification with manual or step-conditioned scaling."""
    parallel_cfg = parallel_cfg or Wan21T2VParallelConfig()
    runtime = _init_wan21_t2v_runtime(parallel_cfg, explicit_device_id=device_id)
    seed = _broadcast_seed_if_needed(seed, runtime)

    if rope_modification_mode not in {"manual", "step_conditioned"}:
        raise ValueError(
            "rope_modification_mode must be one of {'manual', 'step_conditioned'}, "
            f"got {rope_modification_mode!r}."
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
            step_conditioned=(rope_modification_mode == "step_conditioned"),
            step_conditioned_hidden_dim=int(rope_modification_step_conditioned_hidden_dim),
            step_conditioned_checkpoint=str(rope_modification_step_conditioned_checkpoint).strip(),
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
    steps_tag = _format_rope_modification_steps_tag(rope_modification_steps)
    video_path = os.path.join(
        output_dir,
        (
            f"wan21_{task}-rope_modification-mode_{rope_modification_mode}"
            f"-lambdaf_{rope_modification_lambda_f}"
            f"-lambdah_{rope_modification_lambda_h}"
            f"-lambdaw_{rope_modification_lambda_w}"
            f"-{steps_tag}-seed_{seed}-shift_{shift}-guide_{guide_scale}-frame_{frame_num}.mp4"
        ),
    )
    _save_wan21_t2v_video(video, video_path, fps=cfg.sample_fps)

    row = {
        "mode": str(rope_modification_mode),
        "lambda_f": float(rope_modification_lambda_f),
        "lambda_h": float(rope_modification_lambda_h),
        "lambda_w": float(rope_modification_lambda_w),
        "apply_steps": ",".join(str(int(v)) for v in rope_modification_steps),
        "step_conditioned_hidden_dim": int(rope_modification_step_conditioned_hidden_dim),
        "step_conditioned_checkpoint": str(rope_modification_step_conditioned_checkpoint).strip(),
        "video_path": video_path,
        "seed": int(seed),
        "frame_num": int(frame_num),
        "size": f"{size[0]}x{size[1]}",
        "sampling_steps": int(sampling_steps),
        "guide_scale": float(guide_scale),
        "shift": float(shift),
        "sample_solver": str(sample_solver),
        "task": str(task),
    }
    summary = {
        "experiment": "wan21_t2v_rope_modification",
        "prompt": str(prompt),
        "rows": [row],
    }
    _save_json(os.path.join(output_dir, "rope_modification_summary.json"), summary)
    _save_csv(os.path.join(output_dir, "rope_modification_summary.csv"), [row])
    return summary
