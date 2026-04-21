"""Wan2.1-T2V experiment: event_token_value.

Main entries:
- Wan21T2VEventTokenValueState
- _install_wan21_t2v_event_token_value_patch
- run_wan21_t2v_event_token_value

This module collects token-specific value-side cross-attention contributions.
It reuses the shared runtime helpers and prompt/token visualization utilities
from utils.py.
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
    Wan21T2VCrossAttentionVizPatchHandle,
    Wan21T2VParallelConfig,
    _broadcast_seed_if_needed,
    _build_wan21_t2v_pipeline,
    _dedup_wan21_t2v_int_list,
    _ensure_dir,
    _generate_wan21_t2v_video,
    _init_wan21_t2v_runtime,
    _locate_wan21_t2v_prompt_words,
    _resolve_wan21_t2v_branch_from_forward_call_index,
    _resolve_wan21_t2v_offload_model,
    _resolve_wan21_t2v_steps,
    _resolve_wan21_t2v_viz_frame_indices,
    _sanitize_wan21_t2v_token_name,
    _save_csv,
    _save_json,
    _save_wan21_t2v_cross_attention_pdf,
    _save_wan21_t2v_video,
    _wan21_t2v_branch_matches,
)

class Wan21T2VEventTokenValueState:
    """Runtime collector for event-token cross-attention value contributions."""

    def __init__(
        self,
        token_positions: Dict[str, List[int]],
        token_types: Dict[str, str],
        collect_steps: Sequence[int],
        num_layers: int,
        num_heads: int,
        chunk_size: int = 512,
        layers_to_collect: Optional[Sequence[int]] = None,
        branch: str = "cond",
    ):
        self.token_positions = {k: sorted(set(v)) for k, v in token_positions.items()}
        self.token_types = dict(token_types)
        self.collect_steps = set(int(step) for step in collect_steps)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.chunk_size = int(chunk_size)
        self.layers_to_collect = set(int(layer) for layer in layers_to_collect) if layers_to_collect else None
        self.branch = str(branch).strip().lower()
        if self.branch not in {"cond", "uncond", "both"}:
            raise ValueError("branch must be one of cond/uncond/both.")

        self.current_step = 0
        self.current_timestep_value = None
        self.forward_call_index_in_step = 0
        self.layer_meta: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        self.value_maps_sum: Dict[Tuple[int, int, str, str], torch.Tensor] = {}
        self.value_maps_count: Dict[Tuple[int, int, str, str], int] = defaultdict(int)
        self.attention_maps_sum: Dict[Tuple[int, int, str], torch.Tensor] = {}
        self.attention_maps_count: Dict[Tuple[int, int, str], int] = defaultdict(int)
        self.value_summary_rows: List[Dict[str, object]] = []

    def on_forward_start(self, t_tensor):
        """Update diffusion step id and CFG branch index."""
        t_value = float(t_tensor.flatten()[0].item()) if t_tensor is not None else None
        if self.current_timestep_value is None or t_value != self.current_timestep_value:
            self.current_step += 1
            self.current_timestep_value = t_value
            self.forward_call_index_in_step = 0
        else:
            self.forward_call_index_in_step += 1
        self.layer_meta.clear()

    def should_collect(self, layer_idx: int) -> bool:
        """Return True if this cross-attention call should be collected."""
        if self.current_step not in self.collect_steps:
            return False
        if self.layers_to_collect is not None and int(layer_idx) not in self.layers_to_collect:
            return False
        if not _wan21_t2v_branch_matches(self.branch, self.forward_call_index_in_step):
            return False
        return True

    def set_layer_meta(self, layer_idx: int, seq_lens: torch.Tensor, grid_sizes: torch.Tensor):
        """Cache video sequence metadata from the preceding self-attention call."""
        self.layer_meta[int(layer_idx)] = (seq_lens, grid_sizes)

    def collect_event_token_value(
        self,
        layer_idx: int,
        cross_attn_module,
        x: torch.Tensor,
        context: torch.Tensor,
    ):
        """Collect per-token value contribution maps for selected prompt tokens.

        For a selected word made of token positions `P`, the per-head pre-output
        contribution at video query `q` is:
        `sum_{p in P} softmax(q k^T)_p * v_p`.
        The stored projected map multiplies the head slice by the corresponding
        slice of `cross_attn.o.weight`, without assigning the output bias to any
        individual token.
        """
        if not self.should_collect(layer_idx):
            return
        if int(layer_idx) not in self.layer_meta:
            return

        seq_lens, grid_sizes = self.layer_meta[int(layer_idx)]
        bsz = x.size(0)
        num_heads = cross_attn_module.num_heads
        head_dim = cross_attn_module.head_dim
        q = cross_attn_module.norm_q(cross_attn_module.q(x)).view(bsz, -1, num_heads, head_dim)
        k = cross_attn_module.norm_k(cross_attn_module.k(context)).view(bsz, -1, num_heads, head_dim)
        v = cross_attn_module.v(context).view(bsz, -1, num_heads, head_dim)
        scale = 1.0 / (head_dim ** 0.5)

        output_weight = cross_attn_module.o.weight.detach().float()
        head_weight_slices = [
            output_weight[:, head_index * head_dim:(head_index + 1) * head_dim]
            for head_index in range(num_heads)
        ]
        word_idx_tensors = {
            word: torch.tensor(indices, device=q.device, dtype=torch.long)
            for word, indices in self.token_positions.items()
        }

        for batch_index in range(bsz):
            seq_len = int(seq_lens[batch_index].item())
            if seq_len <= 0:
                continue
            frame_count, token_grid_height, token_grid_width = [
                int(vv) for vv in grid_sizes[batch_index].tolist()
            ]
            valid_len = min(seq_len, frame_count * token_grid_height * token_grid_width, q.size(1))
            if valid_len <= 0:
                continue

            q_i = q[batch_index, :valid_len]
            k_i = k[batch_index]
            v_i = v[batch_index]
            context_len = k_i.size(0)

            valid_word_indices: Dict[str, torch.Tensor] = {}
            for word, idxs in word_idx_tensors.items():
                keep = idxs[idxs < context_len]
                if keep.numel() > 0:
                    valid_word_indices[word] = keep
            if not valid_word_indices:
                continue

            attention_scores = {
                word: torch.zeros((num_heads, valid_len), dtype=torch.float32, device=q.device)
                for word in valid_word_indices
            }
            pre_output_norm = {
                word: torch.zeros((num_heads, valid_len), dtype=torch.float32, device=q.device)
                for word in valid_word_indices
            }
            projected_norm = {
                word: torch.zeros((num_heads, valid_len), dtype=torch.float32, device=q.device)
                for word in valid_word_indices
            }

            for query_start in range(0, valid_len, self.chunk_size):
                query_end = min(valid_len, query_start + self.chunk_size)
                q_chunk = q_i[query_start:query_end].float()
                logits = torch.einsum("qhd,khd->hqk", q_chunk, k_i.float()) * scale
                probs = torch.softmax(logits, dim=-1)
                for word, idxs in valid_word_indices.items():
                    selected_probs = probs.index_select(dim=-1, index=idxs)
                    selected_values = v_i.index_select(dim=0, index=idxs).float()
                    contribution_hqd = torch.einsum("hqm,mhd->hqd", selected_probs, selected_values)
                    attention_scores[word][:, query_start:query_end] = selected_probs.sum(dim=-1)
                    pre_output_norm[word][:, query_start:query_end] = contribution_hqd.pow(2).sum(dim=-1).sqrt()
                    for head_index in range(num_heads):
                        projected = contribution_hqd[head_index] @ head_weight_slices[head_index].to(q.device).T
                        projected_norm[word][head_index, query_start:query_end] = projected.pow(2).sum(dim=-1).sqrt()

            branch = _resolve_wan21_t2v_branch_from_forward_call_index(self.forward_call_index_in_step)
            for word in valid_word_indices.keys():
                attention_map = attention_scores[word].reshape(num_heads, frame_count, token_grid_height, token_grid_width)
                attention_key = (int(self.current_step), int(layer_idx), word)
                if attention_key not in self.attention_maps_sum:
                    self.attention_maps_sum[attention_key] = attention_map.detach().cpu().double()
                else:
                    self.attention_maps_sum[attention_key] += attention_map.detach().cpu().double()
                self.attention_maps_count[attention_key] += 1

                for value_kind, source in (
                    ("pre_output_norm", pre_output_norm[word]),
                    ("projected_norm", projected_norm[word]),
                ):
                    value_map = source.reshape(num_heads, frame_count, token_grid_height, token_grid_width)
                    value_key = (int(self.current_step), int(layer_idx), word, value_kind)
                    if value_key not in self.value_maps_sum:
                        self.value_maps_sum[value_key] = value_map.detach().cpu().double()
                    else:
                        self.value_maps_sum[value_key] += value_map.detach().cpu().double()
                    self.value_maps_count[value_key] += 1

                token_type = self.token_types.get(word, "target")
                attn_mean = attention_scores[word].mean(dim=1)
                pre_norm_mean = pre_output_norm[word].mean(dim=1)
                projected_norm_mean = projected_norm[word].mean(dim=1)
                for head_index in range(num_heads):
                    self.value_summary_rows.append(
                        {
                            "step": int(self.current_step),
                            "layer": int(layer_idx),
                            "head": int(head_index),
                            "head_tag": f"L{int(layer_idx)}H{int(head_index)}",
                            "token": word,
                            "token_type": token_type,
                            "branch": branch,
                            "attention_mean": float(attn_mean[head_index].item()),
                            "pre_output_value_norm_mean": float(pre_norm_mean[head_index].item()),
                            "projected_value_norm_mean": float(projected_norm_mean[head_index].item()),
                        }
                    )

    def export_mean_value_maps(self) -> Dict[Tuple[int, int, str, str], torch.Tensor]:
        """Return averaged value maps with shape `[num_heads, F, H, W]`."""
        out = {}
        for key, value_sum in self.value_maps_sum.items():
            count = max(1, int(self.value_maps_count[key]))
            out[key] = (value_sum / count).float()
        return out

    def export_mean_attention_maps(self) -> Dict[Tuple[int, int, str], torch.Tensor]:
        """Return averaged token-attention maps with shape `[num_heads, F, H, W]`."""
        out = {}
        for key, value_sum in self.attention_maps_sum.items():
            count = max(1, int(self.attention_maps_count[key]))
            out[key] = (value_sum / count).float()
        return out

def _install_wan21_t2v_event_token_value_patch(
    model,
    state: Wan21T2VEventTokenValueState,
) -> Wan21T2VCrossAttentionVizPatchHandle:
    """Install hooks that collect event-token value contributions."""
    target = model.module if (hasattr(model, "module") and hasattr(model.module, "blocks")) else model
    if not hasattr(target, "blocks"):
        raise RuntimeError("Invalid DiT model for event-token value patch.")

    original_forward = target.forward

    def patched_dit_forward(this, *args, **kwargs):
        t = kwargs.get("t", None)
        if t is None and len(args) > 1:
            t = args[1]
        state.on_forward_start(t)
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
                state.collect_event_token_value(layer_id, self, x, context)
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

def run_wan21_t2v_event_token_value(
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
    event_token_value_words: Sequence[str] = tuple(),
    event_token_value_steps: Sequence[int] = (1, 2, 3, 4, 5, 6),
    event_token_value_layers: Sequence[int] = tuple(),
    event_token_value_branch: str = "cond",
    event_token_value_chunk_size: int = 512,
    event_token_value_num_viz_frames: int = 10,
    parallel_cfg: Optional[Wan21T2VParallelConfig] = None,
):
    """Collect and visualize selected text tokens' cross-attention value contribution.

    This experiment reuses `_locate_wan21_t2v_prompt_words` for token matching.
    `event_token_value_words` are passed through the existing verb-word path for
    token lookup, then relabeled as `event` in the output metadata.

    Outputs:
        - `event_token_value_summary.csv`: per step/layer/head/token contribution norms.
        - `event_token_value_maps.pt`: projected and pre-output contribution maps.
        - `event_token_attention_maps.pt`: matching attention probability maps.
        - timeline PDFs for head-mean projected value maps.
    """
    target_object_words = [str(word).strip() for word in target_object_words if str(word).strip()]
    target_verb_words = [str(word).strip() for word in target_verb_words if str(word).strip()]
    event_words = [str(word).strip() for word in event_token_value_words if str(word).strip()]
    locator_verb_words = list(dict.fromkeys(target_verb_words + event_words))
    if not target_object_words and not locator_verb_words:
        raise ValueError("event_token_value requires at least one target word.")

    parallel_cfg = parallel_cfg or Wan21T2VParallelConfig()
    runtime = _init_wan21_t2v_runtime(parallel_cfg, explicit_device_id=device_id)
    seed = _broadcast_seed_if_needed(seed, runtime)
    if runtime.use_usp:
        raise RuntimeError("event_token_value currently requires use_usp=False.")

    collect_steps = _resolve_wan21_t2v_steps(event_token_value_steps, sampling_steps)
    layers_to_collect = _dedup_wan21_t2v_int_list(event_token_value_layers)

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
        target_verb_words=locator_verb_words,
    )
    for word in event_words:
        if word in word_to_type:
            word_to_type[word] = "event"

    state = Wan21T2VEventTokenValueState(
        token_positions=word_to_positions,
        token_types=word_to_type,
        collect_steps=collect_steps,
        num_layers=len(pipeline.model.blocks),
        num_heads=pipeline.model.num_heads,
        chunk_size=event_token_value_chunk_size,
        layers_to_collect=layers_to_collect if layers_to_collect else None,
        branch=event_token_value_branch,
    )
    handle = _install_wan21_t2v_event_token_value_patch(pipeline.model, state)

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

    if dist.is_initialized():
        dist.barrier()
    if runtime.rank != 0:
        return None

    _ensure_dir(output_dir)
    video_path = os.path.join(output_dir, f"wan21_t2v_event_token_value_seed_{int(seed)}.mp4")
    _save_wan21_t2v_video(video, video_path, fps=cfg.sample_fps)
    value_maps = state.export_mean_value_maps()
    attention_maps = state.export_mean_attention_maps()
    torch.save(value_maps, os.path.join(output_dir, "event_token_value_maps.pt"))
    torch.save(attention_maps, os.path.join(output_dir, "event_token_attention_maps.pt"))
    _save_csv(os.path.join(output_dir, "event_token_value_summary.csv"), state.value_summary_rows)

    plots_dir = os.path.join(output_dir, "event_token_value_maps")
    plot_paths = []
    for (step, layer, token, value_kind), maps in sorted(value_maps.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3])):
        if value_kind != "projected_norm":
            continue
        head_mean_map = maps.mean(dim=0)
        attention_frame_indices, video_frame_labels = _resolve_wan21_t2v_viz_frame_indices(
            attention_frame_count=int(head_mean_map.size(0)),
            video_frame_count=frame_num,
            num_frames=event_token_value_num_viz_frames,
            explicit_indices=None,
        )
        token_type = word_to_type.get(token, "target")
        save_path = os.path.join(
            plots_dir,
            f"timestep_{int(step):03d}",
            f"token_{token_type}_{_sanitize_wan21_t2v_token_name(token)}",
            f"layer_{int(layer):02d}_head_mean_projected_value_norm.pdf",
        )
        _save_wan21_t2v_cross_attention_pdf(
            map_hfhw=head_mean_map,
            frame_indices=attention_frame_indices,
            frame_labels=video_frame_labels,
            save_file=save_path,
            title=f"event-token projected value norm | step={step} layer={layer} token={token}",
        )
        plot_paths.append(save_path)

    summary = {
        "experiment": "wan21_t2v_event_token_value",
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "target_object_words": list(target_object_words),
        "target_verb_words": list(target_verb_words),
        "event_token_value_words": list(event_words),
        "token_positions": word_to_positions,
        "token_types": word_to_type,
        "collect_steps": [int(step) for step in collect_steps],
        "layers_to_collect": [int(layer) for layer in layers_to_collect],
        "branch": str(event_token_value_branch),
        "chunk_size": int(event_token_value_chunk_size),
        "video_path": video_path,
        "value_maps_path": os.path.join(output_dir, "event_token_value_maps.pt"),
        "attention_maps_path": os.path.join(output_dir, "event_token_attention_maps.pt"),
        "summary_csv": os.path.join(output_dir, "event_token_value_summary.csv"),
        "plot_paths": plot_paths,
    }
    _save_json(os.path.join(output_dir, "event_token_value_summary.json"), summary)
    return summary
