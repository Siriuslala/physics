"""Wan2.1-T2V experiment: step_window_prompt_replace.

Main entries:
- Wan21T2VPromptReplaceState
- Wan21T2VPromptReplacePatchHandle
- _install_wan21_t2v_prompt_replace_patch
- run_wan21_t2v_step_window_prompt_replace

This module replaces the prompt during selected denoising windows to test when
trajectory semantics become locked in. Shared runtime helpers come from utils.py.
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
    Wan21T2VParallelConfig,
    _broadcast_seed_if_needed,
    _build_wan21_t2v_pipeline,
    _dedup_wan21_t2v_int_list,
    _encode_wan21_t2v_text_context_once,
    _ensure_dir,
    _generate_wan21_t2v_video,
    _init_wan21_t2v_runtime,
    _resolve_wan21_t2v_offload_model,
    _save_csv,
    _save_json,
    _save_wan21_t2v_video,
    _unwrap_wan21_t2v_dit_model_for_runtime_patch,
)

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
