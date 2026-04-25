"""Monkey patch utilities for Wan2.1-T2V DiT analysis experiments.

This module provides composable patch components for:
1) 3D RoPE axis intervention,
2) early-step attention probing (P(|dt|), MAAS maps),
3) causal attention schedule with two modes:
   - flat token causal,
   - strict temporal causal (frame-wise).

All patches are applied at runtime and can be fully restored.
"""

import math
from dataclasses import dataclass, field
from types import MethodType
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

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


@dataclass
class Wan21T2VRopePatchConfig:
    """Configuration for 3D RoPE axis intervention."""

    enabled: bool = True
    mode: str = "full"  # full, no_f, no_h, no_w, only_f, only_hw
    f_scale: float = 1.0
    h_scale: float = 1.0
    w_scale: float = 1.0


@dataclass
class Wan21T2VAttentionProbeConfig:
    """Configuration for attention probing.

    `enabled=True` records early-step attention statistics:
    - P(|dt|): time-offset attention histogram
    - optional MAAS maps: next-frame attention mass maps for selected layers
    """

    enabled: bool = False
    probe_steps: Tuple[int, ...] = (1, 2, 3)
    query_frame_count: int = 8
    query_mode: str = "center"  # center, multi_anchor, object_guided
    probe_branch: str = "uncond"  # uncond, cond, both
    use_abs_dt: bool = True
    object_token_trajectory: Optional[Dict[int, Tuple[float, float]]] = None
    collect_dt_histograms: bool = True

    collect_maas_maps: bool = False
    maas_steps: Tuple[int, ...] = (1, 2, 3)
    maas_layers: Tuple[int, ...] = tuple()
    maas_radius: int = 1

    collect_distribution: bool = False
    distribution_layers: Tuple[int, ...] = tuple()
    distribution_query_frame_count: int = 8
    distribution_global_query_tokens_per_frame: int = 64
    distribution_object_query_token_limit_per_frame: int = 0
    distribution_object_support_mask: Optional[torch.Tensor] = None


@dataclass
class Wan21T2VCausalAttentionConfig:
    """Configuration for causal attention schedule.

    mode:
    - none: no causal restriction
    - flat: token-index causal (upper-triangular on flattened sequence)
    - temporal: strict frame-wise causal (frame_k <= frame_q)

    backend:
    - auto: use flash attention when possible, otherwise torch SDPA
    - flash: force flash attention (unsupported for temporal mode)
    - torch_sdpa: always use torch SDPA
    """

    enabled: bool = False
    causal_first_n_steps: int = 0
    mode: str = "flat"  # none, flat, temporal
    backend: str = "auto"  # auto, flash, torch_sdpa


@dataclass
class Wan21T2VPatchBundleConfig:
    """Composable bundle of patch configurations."""

    rope: Wan21T2VRopePatchConfig = field(default_factory=Wan21T2VRopePatchConfig)
    probe: Wan21T2VAttentionProbeConfig = field(default_factory=Wan21T2VAttentionProbeConfig)
    causal: Wan21T2VCausalAttentionConfig = field(default_factory=Wan21T2VCausalAttentionConfig)


class Wan21T2VProbeState:
    """Runtime state and accumulators for early-step probes."""

    def __init__(self, config: Wan21T2VPatchBundleConfig, num_layers: int, num_heads: int):
        self.config = config
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.current_step = 0
        self.current_timestep_value = None
        self.forward_call_index_in_step = 0
        self.collect_this_forward = False

        self.cached_query_indices: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

        self.dt_hist_sum: Dict[int, torch.Tensor] = {}
        self.dt_hist_count: Dict[int, torch.Tensor] = {}
        self.dt_bins: Optional[int] = None

        self.maas_maps_sum: Dict[Tuple[int, int, int], torch.Tensor] = {}
        self.maas_maps_count: Dict[Tuple[int, int, int], int] = {}
        self.maas_grid_size: Optional[Tuple[int, int, int]] = None

        self.distribution_object_sum: Dict[Tuple[int, int, int], torch.Tensor] = {}
        self.distribution_object_count: Dict[Tuple[int, int, int], int] = {}
        self.distribution_object_dt_sum: Dict[Tuple[int, int], torch.Tensor] = {}
        self.distribution_object_dt_count: Dict[Tuple[int, int], int] = {}
        self.distribution_global_dt_sum: Dict[Tuple[int, int, str], torch.Tensor] = {}
        self.distribution_global_dt_count: Dict[Tuple[int, int, str], int] = {}
        self.distribution_grid_size: Optional[Tuple[int, int, int]] = None

    def on_forward_start(self, t_tensor):
        """Track diffusion step based on timestep tensor and CFG forward index."""
        t_value = float(t_tensor.flatten()[0].item()) if t_tensor is not None else None

        if self.current_timestep_value is None or t_value != self.current_timestep_value:
            self.current_step += 1
            self.current_timestep_value = t_value
            self.forward_call_index_in_step = 0
        else:
            self.forward_call_index_in_step += 1

        if not self.config.probe.enabled or self.current_step not in self.config.probe.probe_steps:
            self.collect_this_forward = False
            return

        branch = self.config.probe.probe_branch
        if branch == "uncond":
            self.collect_this_forward = (self.forward_call_index_in_step == 1)
        elif branch == "cond":
            self.collect_this_forward = (self.forward_call_index_in_step == 0)
        elif branch == "both":
            self.collect_this_forward = True
        else:
            raise ValueError(f"Unknown probe_branch: {branch}")

    def should_collect_layer(self, layer_idx: int) -> bool:
        if not self.collect_this_forward:
            return False
        if not self.config.probe.enabled:
            return False
        return True

    def should_apply_causal(self) -> bool:
        if not self.config.causal.enabled:
            return False
        return self.current_step <= self.config.causal.causal_first_n_steps

    def _resolve_object_guided_point(self, frame_id: int, h: int, w: int) -> Tuple[int, int]:
        traj = self.config.probe.object_token_trajectory or {}
        if not traj:
            return h // 2, w // 2

        if frame_id in traj:
            y, x = traj[frame_id]
        else:
            keys = sorted(traj.keys())
            nearest = min(keys, key=lambda k: abs(k - frame_id))
            y, x = traj[nearest]

        y_i = int(round(float(y)))
        x_i = int(round(float(x)))
        y_i = max(0, min(h - 1, y_i))
        x_i = max(0, min(w - 1, x_i))
        return y_i, x_i

    def _get_query_indices(self, f: int, h: int, w: int, device: torch.device):
        mode = self.config.probe.query_mode
        frame_count = min(max(1, self.config.probe.query_frame_count), f)
        frame_ids = torch.linspace(0, f - 1, steps=frame_count, device=device).round().long().unique(sorted=True)

        if mode in {"center", "multi_anchor"}:
            key = (mode, f, h, w, frame_count, device)
            if key in self.cached_query_indices:
                return self.cached_query_indices[key]

        query_indices = []
        query_frames = []

        if mode == "center":
            y = h // 2
            x = w // 2
            for qf in frame_ids.tolist():
                query_indices.append(qf * (h * w) + y * w + x)
                query_frames.append(qf)
        elif mode == "multi_anchor":
            anchors = [
                (h // 2, w // 2),
                (h // 4, w // 4),
                (h // 4, (3 * w) // 4),
                ((3 * h) // 4, w // 4),
                ((3 * h) // 4, (3 * w) // 4),
            ]
            anchors = sorted(set((max(0, min(h - 1, y)), max(0, min(w - 1, x))) for y, x in anchors))
            for qf in frame_ids.tolist():
                for y, x in anchors:
                    query_indices.append(qf * (h * w) + y * w + x)
                    query_frames.append(qf)
        elif mode == "object_guided":
            for qf in frame_ids.tolist():
                y, x = self._resolve_object_guided_point(qf, h, w)
                query_indices.append(qf * (h * w) + y * w + x)
                query_frames.append(qf)
        else:
            raise ValueError(f"Unknown query_mode: {mode}")

        query_indices_t = torch.tensor(query_indices, device=device, dtype=torch.long)
        query_frames_t = torch.tensor(query_frames, device=device, dtype=torch.long)

        if mode in {"center", "multi_anchor"}:
            self.cached_query_indices[(mode, f, h, w, frame_count, device)] = (query_indices_t, query_frames_t)
        return query_indices_t, query_frames_t

    def _get_distribution_query_frames(self, f: int, device: torch.device) -> torch.Tensor:
        """Return evenly spaced token-frame ids used by self-attention distribution probes."""
        frame_count = min(max(1, self.config.probe.distribution_query_frame_count), f)
        return torch.linspace(0, f - 1, steps=frame_count, device=device).round().long().unique(sorted=True)

    def _sample_evenly_from_indices(self, indices: torch.Tensor, limit: int) -> torch.Tensor:
        """Subsample 1D indices deterministically with near-uniform spacing."""
        if int(limit) <= 0 or int(indices.numel()) <= int(limit):
            return indices
        sample_positions = torch.linspace(
            0,
            int(indices.numel()) - 1,
            steps=int(limit),
            device=indices.device,
        ).round().long()
        return indices[sample_positions]

    def _distribution_query_bucket(self, frame_id: int, frame_count: int) -> str:
        """Bucket query frames into coarse temporal regions to expose edge effects."""
        if frame_count <= 1:
            return "middle"
        normalized_position = float(frame_id) / float(max(1, frame_count - 1))
        if normalized_position < (1.0 / 3.0):
            return "early"
        if normalized_position > (2.0 / 3.0):
            return "late"
        return "middle"

    def _get_distribution_object_query_indices(
        self,
        f: int,
        h: int,
        w: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return query-token indices inside the reference object support mask."""
        support_mask = self.config.probe.distribution_object_support_mask
        if support_mask is None:
            return (
                torch.empty(0, device=device, dtype=torch.long),
                torch.empty(0, device=device, dtype=torch.long),
            )
        if support_mask.dim() != 3:
            raise ValueError(
                "distribution_object_support_mask must have shape [F, H, W], "
                f"got {tuple(support_mask.shape)}."
            )
        if tuple(int(v) for v in support_mask.shape) != (int(f), int(h), int(w)):
            raise ValueError(
                "distribution_object_support_mask shape does not match current token grid: "
                f"mask={tuple(int(v) for v in support_mask.shape)} vs grid={(int(f), int(h), int(w))}."
            )

        query_indices: List[torch.Tensor] = []
        query_frames: List[torch.Tensor] = []
        query_frame_ids = self._get_distribution_query_frames(f, device)
        token_limit = int(self.config.probe.distribution_object_query_token_limit_per_frame)
        for frame_id in query_frame_ids.tolist():
            frame_mask = support_mask[int(frame_id)].reshape(-1).to(device=device) > 0.5
            frame_indices = torch.nonzero(frame_mask, as_tuple=False).flatten()
            if frame_indices.numel() == 0:
                continue
            frame_indices = self._sample_evenly_from_indices(frame_indices, token_limit)
            query_indices.append(frame_indices + int(frame_id) * (h * w))
            query_frames.append(torch.full_like(frame_indices, int(frame_id)))

        if not query_indices:
            return (
                torch.empty(0, device=device, dtype=torch.long),
                torch.empty(0, device=device, dtype=torch.long),
            )
        return torch.cat(query_indices, dim=0), torch.cat(query_frames, dim=0)

    def _get_distribution_global_query_indices(
        self,
        f: int,
        h: int,
        w: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return uniformly sampled query-token indices over all spatial positions."""
        query_indices: List[torch.Tensor] = []
        query_frames: List[torch.Tensor] = []
        tokens_per_frame = int(h * w)
        sample_count = max(1, int(self.config.probe.distribution_global_query_tokens_per_frame))
        query_frame_ids = self._get_distribution_query_frames(f, device)
        for frame_id in query_frame_ids.tolist():
            frame_positions = torch.arange(tokens_per_frame, device=device, dtype=torch.long)
            frame_positions = self._sample_evenly_from_indices(frame_positions, sample_count)
            query_indices.append(frame_positions + int(frame_id) * tokens_per_frame)
            query_frames.append(torch.full_like(frame_positions, int(frame_id)))

        if not query_indices:
            return (
                torch.empty(0, device=device, dtype=torch.long),
                torch.empty(0, device=device, dtype=torch.long),
            )
        return torch.cat(query_indices, dim=0), torch.cat(query_frames, dim=0)

    def _ensure_dt_storage(self, bins: int):
        if self.dt_bins is None:
            self.dt_bins = bins
        if self.dt_bins != bins:
            raise RuntimeError("Inconsistent dt bins across runs. Use fixed frame_num for one run.")

        if self.current_step not in self.dt_hist_sum:
            self.dt_hist_sum[self.current_step] = torch.zeros(
                (self.num_layers, self.num_heads, bins), dtype=torch.float64)
            self.dt_hist_count[self.current_step] = torch.zeros((self.num_layers,), dtype=torch.float64)

    def collect(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, seq_lens: torch.Tensor, grid_sizes: torch.Tensor):
        """Collect P(|dt|) and optional MAAS maps from q/k tensors."""
        if not self.should_collect_layer(layer_idx):
            return

        need_dt_histograms = bool(self.config.probe.collect_dt_histograms)
        need_distribution = bool(self.config.probe.collect_distribution) and (
            (not self.config.probe.distribution_layers)
            or (int(layer_idx) in self.config.probe.distribution_layers)
        )
        need_maas = bool(self.config.probe.collect_maas_maps) and (
            self.current_step in self.config.probe.maas_steps
        ) and (
            (not self.config.probe.maas_layers) or (int(layer_idx) in self.config.probe.maas_layers)
        )

        if not (need_dt_histograms or need_distribution or need_maas):
            return

        bsz = q.size(0)
        for b in range(bsz):
            seq_len = int(seq_lens[b].item())
            if seq_len <= 0:
                continue

            f, h, w = [int(v) for v in grid_sizes[b].tolist()]
            q_i = q[b, :seq_len]
            k_i = k[b, :seq_len]
            scale = 1.0 / math.sqrt(q_i.size(-1))

            if need_dt_histograms or need_maas:
                query_indices, query_frames = self._get_query_indices(f, h, w, q.device)
                keep = query_indices < seq_len
                if bool(keep.any().item()):
                    query_indices = query_indices[keep]
                    query_frames = query_frames[keep]
                    q_sel = q_i[query_indices]
                    logits = torch.einsum("qhd,khd->hqk", q_sel.float(), k_i.float()) * scale
                    probs = torch.softmax(logits, dim=-1)

                    key_frames = torch.arange(f, device=q.device).repeat_interleave(h * w)
                    bins = f if self.config.probe.use_abs_dt else (2 * f - 1)
                    if need_dt_histograms:
                        self._ensure_dt_storage(bins)

                    for q_idx, qf in enumerate(query_frames.tolist()):
                        if self.config.probe.use_abs_dt:
                            dt_index = (key_frames - qf).abs()
                        else:
                            dt_index = (key_frames - qf) + (f - 1)

                        if need_dt_histograms:
                            hist = torch.zeros((self.num_heads, bins), device=q.device, dtype=torch.float32)
                            for h_idx in range(self.num_heads):
                                hist[h_idx].scatter_add_(0, dt_index, probs[h_idx, q_idx])

                            self.dt_hist_sum[self.current_step][layer_idx] += hist.detach().cpu().double()
                            self.dt_hist_count[self.current_step][layer_idx] += 1.0

                        if need_maas and qf + 1 < f:
                            probs_map = probs[:, q_idx, :seq_len].reshape(self.num_heads, f, h, w)
                            next_frame_map = probs_map[:, qf + 1].detach().cpu().float()
                            map_key = (self.current_step, layer_idx, qf)

                            if map_key not in self.maas_maps_sum:
                                self.maas_maps_sum[map_key] = torch.zeros_like(next_frame_map)
                                self.maas_maps_count[map_key] = 0

                            self.maas_maps_sum[map_key] += next_frame_map
                            self.maas_maps_count[map_key] += 1
                            self.maas_grid_size = (f, h, w)

            if need_distribution:
                object_support_mask = self.config.probe.distribution_object_support_mask
                if object_support_mask is None:
                    raise ValueError("collect_distribution=True requires distribution_object_support_mask.")
                object_support_mask_device = object_support_mask.to(device=q.device, dtype=torch.float32)
                object_query_indices, object_query_frames = self._get_distribution_object_query_indices(
                    f=f,
                    h=h,
                    w=w,
                    device=q.device,
                )
                object_keep = object_query_indices < seq_len
                object_query_indices = object_query_indices[object_keep]
                object_query_frames = object_query_frames[object_keep]

                if int(object_query_indices.numel()) > 0:
                    object_queries = q_i[object_query_indices]
                    object_logits = torch.einsum("qhd,khd->hqk", object_queries.float(), k_i.float()) * scale
                    object_probs = torch.softmax(object_logits, dim=-1)
                    signed_dt_bins = 2 * f - 1
                    signed_key_frame_indices = torch.arange(f, device=q.device, dtype=torch.long) + (f - 1)
                    for query_position, query_frame in enumerate(object_query_frames.tolist()):
                        probability_map = object_probs[:, query_position, :seq_len].reshape(self.num_heads, f, h, w)
                        frame_mass = probability_map.sum(dim=(-1, -2))
                        object_mass = (probability_map * object_support_mask_device.unsqueeze(0)).sum(dim=(-1, -2))
                        nonobject_mass = frame_mass - object_mass

                        object_key = (self.current_step, int(layer_idx), int(query_frame))
                        if object_key not in self.distribution_object_sum:
                            self.distribution_object_sum[object_key] = torch.zeros(
                                (self.num_heads, f, 3),
                                dtype=torch.float64,
                            )
                            self.distribution_object_count[object_key] = 0
                        stacked_mass = torch.stack([frame_mass, object_mass, nonobject_mass], dim=-1)
                        self.distribution_object_sum[object_key] += stacked_mass.detach().cpu().double()
                        self.distribution_object_count[object_key] += 1

                        object_dt_key = (self.current_step, int(layer_idx))
                        if object_dt_key not in self.distribution_object_dt_sum:
                            self.distribution_object_dt_sum[object_dt_key] = torch.zeros(
                                (self.num_heads, signed_dt_bins, 3),
                                dtype=torch.float64,
                            )
                            self.distribution_object_dt_count[object_dt_key] = 0
                        signed_dt_index = signed_key_frame_indices - int(query_frame)
                        signed_dt_index_cpu = signed_dt_index.detach().cpu()
                        object_dt_sum = self.distribution_object_dt_sum[object_dt_key]
                        frame_mass_cpu = frame_mass.detach().cpu().double()
                        object_mass_cpu = object_mass.detach().cpu().double()
                        nonobject_mass_cpu = nonobject_mass.detach().cpu().double()
                        for head_index in range(self.num_heads):
                            object_dt_sum[head_index, signed_dt_index_cpu, 0] += frame_mass_cpu[head_index]
                            object_dt_sum[head_index, signed_dt_index_cpu, 1] += object_mass_cpu[head_index]
                            object_dt_sum[head_index, signed_dt_index_cpu, 2] += nonobject_mass_cpu[head_index]
                        self.distribution_object_dt_count[object_dt_key] += 1

                global_query_indices, global_query_frames = self._get_distribution_global_query_indices(
                    f=f,
                    h=h,
                    w=w,
                    device=q.device,
                )
                global_keep = global_query_indices < seq_len
                global_query_indices = global_query_indices[global_keep]
                global_query_frames = global_query_frames[global_keep]
                if int(global_query_indices.numel()) > 0:
                    global_queries = q_i[global_query_indices]
                    global_logits = torch.einsum("qhd,khd->hqk", global_queries.float(), k_i.float()) * scale
                    global_probs = torch.softmax(global_logits, dim=-1)
                    signed_dt_bins = 2 * f - 1
                    signed_key_frame_indices = torch.arange(f, device=q.device, dtype=torch.long) + (f - 1)
                    for query_position, query_frame in enumerate(global_query_frames.tolist()):
                        probability_map = global_probs[:, query_position, :seq_len].reshape(self.num_heads, f, h, w)
                        frame_mass = probability_map.sum(dim=(-1, -2))
                        bucket_names = [
                            "all",
                            self._distribution_query_bucket(int(query_frame), int(f)),
                        ]
                        signed_dt_index = signed_key_frame_indices - int(query_frame)
                        signed_dt_index_cpu = signed_dt_index.detach().cpu()
                        frame_mass_cpu = frame_mass.detach().cpu().double()
                        for bucket_name in bucket_names:
                            global_dt_key = (self.current_step, int(layer_idx), str(bucket_name))
                            if global_dt_key not in self.distribution_global_dt_sum:
                                self.distribution_global_dt_sum[global_dt_key] = torch.zeros(
                                    (self.num_heads, signed_dt_bins),
                                    dtype=torch.float64,
                                )
                                self.distribution_global_dt_count[global_dt_key] = 0
                            global_dt_sum = self.distribution_global_dt_sum[global_dt_key]
                            for head_index in range(self.num_heads):
                                global_dt_sum[head_index, signed_dt_index_cpu] += frame_mass_cpu[head_index]
                            self.distribution_global_dt_count[global_dt_key] += 1

                self.distribution_grid_size = (f, h, w)

    def export_dt_histograms(self) -> Dict[str, torch.Tensor]:
        """Export normalized dt histograms as tensors."""
        out = {}
        for step in sorted(self.dt_hist_sum.keys()):
            denom = self.dt_hist_count[step].clamp_min(1.0).view(self.num_layers, 1, 1)
            out[f"dt_hist_step_{step}"] = (self.dt_hist_sum[step] / denom).float()
            out[f"dt_count_step_{step}"] = self.dt_hist_count[step].float()
        return out

    def export_distribution_rows(self) -> Dict[str, List[Dict[str, float]]]:
        """Export self-attention distribution statistics as flat row dictionaries."""
        object_rows: List[Dict[str, float]] = []
        for (step, layer, query_frame), mass_sum in sorted(self.distribution_object_sum.items()):
            count = max(1, int(self.distribution_object_count[(step, layer, query_frame)]))
            mean_mass = (mass_sum / float(count)).float()
            frame_count = int(mean_mass.size(1))
            for head_index in range(int(mean_mass.size(0))):
                for key_frame in range(frame_count):
                    frame_mass = float(mean_mass[head_index, key_frame, 0].item())
                    object_mass = float(mean_mass[head_index, key_frame, 1].item())
                    nonobject_mass = float(mean_mass[head_index, key_frame, 2].item())
                    object_rows.append(
                        {
                            "step": int(step),
                            "layer": int(layer),
                            "head": int(head_index),
                            "query_frame": int(query_frame),
                            "key_frame": int(key_frame),
                            "frame_mass": frame_mass,
                            "object_mass": object_mass,
                            "nonobject_mass": nonobject_mass,
                            "object_fraction": float(object_mass / max(1e-8, frame_mass)),
                            "num_query_tokens": int(count),
                        }
                    )

        object_dt_rows: List[Dict[str, float]] = []
        for (step, layer), mass_sum in sorted(self.distribution_object_dt_sum.items()):
            count = max(1, int(self.distribution_object_dt_count[(step, layer)]))
            mean_mass = (mass_sum / float(count)).float()
            frame_count = int((mean_mass.size(1) + 1) // 2)
            for head_index in range(int(mean_mass.size(0))):
                for dt_index in range(int(mean_mass.size(1))):
                    dt_value = int(dt_index - (frame_count - 1))
                    frame_mass = float(mean_mass[head_index, dt_index, 0].item())
                    object_mass = float(mean_mass[head_index, dt_index, 1].item())
                    nonobject_mass = float(mean_mass[head_index, dt_index, 2].item())
                    object_dt_rows.append(
                        {
                            "step": int(step),
                            "layer": int(layer),
                            "head": int(head_index),
                            "dt": int(dt_value),
                            "frame_mass": frame_mass,
                            "object_mass": object_mass,
                            "nonobject_mass": nonobject_mass,
                            "object_fraction": float(object_mass / max(1e-8, frame_mass)),
                            "num_query_tokens": int(count),
                        }
                    )

        global_dt_rows: List[Dict[str, float]] = []
        for (step, layer, bucket_name), mass_sum in sorted(self.distribution_global_dt_sum.items()):
            count = max(1, int(self.distribution_global_dt_count[(step, layer, bucket_name)]))
            mean_mass = (mass_sum / float(count)).float()
            frame_count = int((mean_mass.size(1) + 1) // 2)
            for head_index in range(int(mean_mass.size(0))):
                for dt_index in range(int(mean_mass.size(1))):
                    dt_value = int(dt_index - (frame_count - 1))
                    global_dt_rows.append(
                        {
                            "step": int(step),
                            "layer": int(layer),
                            "head": int(head_index),
                            "query_bucket": str(bucket_name),
                            "dt": int(dt_value),
                            "attention_mass": float(mean_mass[head_index, dt_index].item()),
                            "num_query_tokens": int(count),
                        }
                    )

        return {
            "object_rows": object_rows,
            "object_dt_rows": object_dt_rows,
            "global_dt_rows": global_dt_rows,
        }

    def compute_maas(self, token_trajectory: Dict[int, Tuple[int, int]], radius: Optional[int] = None):
        """Compute MAAS rows and summary given token-space trajectory targets."""
        if radius is None:
            radius = self.config.probe.maas_radius

        rows = []
        for (step, layer, qf), map_sum in self.maas_maps_sum.items():
            target = token_trajectory.get(qf + 1)
            if target is None:
                continue

            ty, tx = target
            count = self.maas_maps_count[(step, layer, qf)]
            mean_map = map_sum / max(1, count)
            heads, h, w = mean_map.shape

            y0 = max(0, ty - radius)
            y1 = min(h, ty + radius + 1)
            x0 = max(0, tx - radius)
            x1 = min(w, tx + radius + 1)

            local = mean_map[:, y0:y1, x0:x1].sum(dim=(1, 2))
            total = mean_map.sum(dim=(1, 2)).clamp_min(1e-8)
            ratio = local / total

            for head in range(heads):
                rows.append(
                    {
                        "step": step,
                        "layer": layer,
                        "head": head,
                        "query_frame": qf,
                        "target_y": ty,
                        "target_x": tx,
                        "maas_local_mass": float(local[head].item()),
                        "maas_local_ratio": float(ratio[head].item()),
                    }
                )

        if rows:
            ratios = torch.tensor([r["maas_local_ratio"] for r in rows], dtype=torch.float32)
            masses = torch.tensor([r["maas_local_mass"] for r in rows], dtype=torch.float32)
            summary = {
                "maas_mean_local_ratio": float(ratios.mean().item()),
                "maas_std_local_ratio": float(ratios.std(unbiased=False).item()),
                "maas_mean_local_mass": float(masses.mean().item()),
                "maas_std_local_mass": float(masses.std(unbiased=False).item()),
                "maas_num_rows": len(rows),
            }
        else:
            summary = {
                "maas_mean_local_ratio": 0.0,
                "maas_std_local_ratio": 0.0,
                "maas_mean_local_mass": 0.0,
                "maas_std_local_mass": 0.0,
                "maas_num_rows": 0,
            }

        return summary, rows


class Wan21T2VPatchHandle:
    """Handle for restoring all monkey patches."""

    def __init__(
        self,
        target_model,
        state: Wan21T2VProbeState,
        original_forward=None,
        original_attn_forwards=None,
        restore_items=None,
    ):
        self.target_model = target_model
        self.state = state
        self.original_forward = original_forward
        self.original_attn_forwards = [] if original_attn_forwards is None else original_attn_forwards
        self.restore_items = [] if restore_items is None else restore_items

    def restore(self):
        if self.original_forward is not None:
            self.target_model.forward = self.original_forward
        for idx, block in enumerate(self.target_model.blocks):
            if idx < len(self.original_attn_forwards):
                block.self_attn.forward = self.original_attn_forwards[idx]
        for obj, name, value in self.restore_items:
            setattr(obj, name, value)


def _axis_enabled(mode: str) -> Tuple[bool, bool, bool]:
    if mode == "full":
        return True, True, True
    if mode == "no_f":
        return False, True, True
    if mode == "no_h":
        return True, False, True
    if mode == "no_w":
        return True, True, False
    if mode == "only_f":
        return True, False, False
    if mode == "only_hw":
        return False, True, True
    raise ValueError(f"Unknown rope mode: {mode}")


def _rescale_complex_phase(freq: torch.Tensor, scale: float) -> torch.Tensor:
    if scale == 1.0:
        return freq
    angle = torch.angle(freq)
    return torch.polar(torch.ones_like(angle), angle * scale)


@torch.cuda.amp.autocast(enabled=False)
def apply_wan21_t2v_rope_patch(x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor, rope_cfg: Wan21T2VRopePatchConfig):
    """Apply configurable 3D RoPE intervention on q/k tensors."""
    if not rope_cfg.enabled:
        # Full RoPE behavior if patch disabled.
        enable_f, enable_h, enable_w = True, True, True
    else:
        enable_f, enable_h, enable_w = _axis_enabled(rope_cfg.mode)

    n, c = x.size(2), x.size(3) // 2
    freqs_f, freqs_h, freqs_w = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    def _slice_freqs_for_local_shard(freq_i: torch.Tensor, local_len: int) -> torch.Tensor:
        """Slice full RoPE multipliers for a local sequence-parallel shard."""
        full_len = freq_i.size(0)
        if local_len >= full_len:
            return freq_i[:local_len]

        # Try to use USP rank/world-size if available.
        sp_size = max(1, (full_len + local_len - 1) // local_len)
        sp_rank = 0
        try:
            from xfuser.core.distributed import (
                get_sequence_parallel_rank,
                get_sequence_parallel_world_size,
            )
            sp_size = int(get_sequence_parallel_world_size())
            sp_rank = int(get_sequence_parallel_rank())
        except Exception:
            pass

        target_len = local_len * sp_size
        if full_len < target_len:
            pad = torch.ones(
                (target_len - full_len, freq_i.size(1), freq_i.size(2)),
                dtype=freq_i.dtype,
                device=freq_i.device,
            )
            freq_i = torch.cat([freq_i, pad], dim=0)

        start = sp_rank * local_len
        end = start + local_len
        if end > freq_i.size(0):
            start = max(0, freq_i.size(0) - local_len)
            end = freq_i.size(0)
        return freq_i[start:end]

    out = []
    s_local = x.size(1)
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        local_len = min(s_local, seq_len)
        x_i = torch.view_as_complex(x[i, :local_len].to(torch.float64).reshape(local_len, n, -1, 2))

        ff = freqs_f[:f]
        fh = freqs_h[:h]
        fw = freqs_w[:w]

        ff = _rescale_complex_phase(ff, rope_cfg.f_scale) if enable_f else torch.ones_like(ff)
        fh = _rescale_complex_phase(fh, rope_cfg.h_scale) if enable_h else torch.ones_like(fh)
        fw = _rescale_complex_phase(fw, rope_cfg.w_scale) if enable_w else torch.ones_like(fw)

        freq_i_full = torch.cat(
            [
                ff.view(f, 1, 1, -1).expand(f, h, w, -1),
                fh.view(1, h, 1, -1).expand(f, h, w, -1),
                fw.view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        freq_i = _slice_freqs_for_local_shard(freq_i_full, local_len)

        x_i = torch.view_as_real(x_i * freq_i).flatten(2)
        x_i = torch.cat([x_i, x[i, local_len:]])
        out.append(x_i)

    return torch.stack(out).float()


def _build_attention_mask(
    seq_len: int,
    grid_size: Tuple[int, int, int],
    causal_enabled: bool,
    causal_mode: str,
    window_size: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Build float attention mask for torch SDPA.

    Returned mask shape: [seq_len, seq_len], values in {0, -inf}.
    """
    needs_mask = causal_enabled or window_size != (-1, -1)
    if not needs_mask:
        return None

    idx_q = torch.arange(seq_len, device=device).view(seq_len, 1)
    idx_k = torch.arange(seq_len, device=device).view(1, seq_len)
    allowed = torch.ones((seq_len, seq_len), device=device, dtype=torch.bool)

    if causal_enabled:
        if causal_mode == "flat":
            allowed &= idx_k <= idx_q
        elif causal_mode == "temporal":
            f, h, w = grid_size
            spatial = h * w
            frame_q = (idx_q // spatial)
            frame_k = (idx_k // spatial)
            allowed &= frame_k <= frame_q
        elif causal_mode == "none":
            pass
        else:
            raise ValueError(f"Unknown causal mode: {causal_mode}")

    if window_size != (-1, -1):
        left, right = window_size
        if left >= 0:
            allowed &= idx_k >= (idx_q - left)
        if right >= 0:
            allowed &= idx_k <= (idx_q + right)

    mask = torch.zeros((seq_len, seq_len), device=device, dtype=dtype)
    mask = mask.masked_fill(~allowed, float("-inf"))
    return mask


def _attention_torch_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_lens: torch.Tensor,
    grid_sizes: torch.Tensor,
    causal_enabled: bool,
    causal_mode: str,
    window_size: Tuple[int, int],
) -> torch.Tensor:
    """Attention implementation with torch scaled_dot_product_attention."""
    bsz, s, n, d = q.shape
    out = q.new_zeros((bsz, s, n, d))

    for b in range(bsz):
        l = int(seq_lens[b].item())
        if l <= 0:
            continue

        q_b = q[b, :l].permute(1, 0, 2)  # [N, L, D]
        k_b = k[b, :l].permute(1, 0, 2)
        v_b = v[b, :l].permute(1, 0, 2)
        grid_size = tuple(int(x) for x in grid_sizes[b].tolist())

        # Memory-optimized strict temporal-causal path:
        # frame-wise causal mask (frame_k <= frame_q) is equivalent to key-prefix
        # truncation under Wan's [frame-major flatten] ordering.
        if causal_enabled and causal_mode == "temporal" and window_size == (-1, -1):
            f, h, w = grid_size
            spatial = max(1, int(h) * int(w))
            query_chunk_size = 512
            for q_start in range(0, l, query_chunk_size):
                q_end = min(l, q_start + query_chunk_size)
                if q_end <= q_start:
                    continue
                frame_of_query = int((q_end - 1) // spatial)
                key_end = min(l, (frame_of_query + 1) * spatial)

                out_chunk = F.scaled_dot_product_attention(
                    q_b[:, q_start:q_end, :],
                    k_b[:, :key_end, :],
                    v_b[:, :key_end, :],
                    attn_mask=None,
                    is_causal=False,
                    dropout_p=0.0,
                )
                out[b, q_start:q_end] = out_chunk.permute(1, 0, 2)
            continue

        mask = _build_attention_mask(
            seq_len=l,
            grid_size=grid_size,
            causal_enabled=causal_enabled,
            causal_mode=causal_mode,
            window_size=window_size,
            device=q_b.device,
            dtype=q_b.dtype,
        )

        out_b = F.scaled_dot_product_attention(
            q_b,
            k_b,
            v_b,
            attn_mask=mask,
            is_causal=False,
            dropout_p=0.0,
        )
        out[b, :l] = out_b.permute(1, 0, 2)

    return out


def _choose_backend(attn_module, backend: str, causal_mode: str, causal_enabled: bool) -> str:
    if backend not in {"auto", "flash", "torch_sdpa"}:
        raise ValueError(f"Unknown attention backend: {backend}")

    if backend == "torch_sdpa":
        return "torch_sdpa"

    if causal_enabled and causal_mode == "temporal":
        # Flash attention cannot express strict temporal frame-wise mask.
        return "torch_sdpa"

    flash_ok = attn_module.FLASH_ATTN_2_AVAILABLE or attn_module.FLASH_ATTN_3_AVAILABLE
    if backend == "flash":
        if not flash_ok:
            raise RuntimeError("backend=flash requested but flash attention is not available.")
        return "flash"

    # auto
    return "flash" if flash_ok else "torch_sdpa"


def _attention_dispatch(
    attn_module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_lens: torch.Tensor,
    grid_sizes: torch.Tensor,
    causal_enabled: bool,
    causal_mode: str,
    window_size: Tuple[int, int],
    backend: str,
) -> torch.Tensor:
    chosen = _choose_backend(attn_module, backend=backend, causal_mode=causal_mode, causal_enabled=causal_enabled)

    if chosen == "flash":
        return attn_module.flash_attention(
            q=q,
            k=k,
            v=v,
            k_lens=seq_lens,
            window_size=window_size,
            causal=causal_enabled and (causal_mode == "flat"),
        )

    return _attention_torch_sdpa(
        q=q,
        k=k,
        v=v,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        causal_enabled=causal_enabled,
        causal_mode=causal_mode,
        window_size=window_size,
    )


def _unwrap_dit_model(model):
    """Return the actual DiT module containing `blocks` and `forward`.

    For FSDP, this is usually `model.module`; otherwise it's `model` itself.
    """
    if hasattr(model, "module") and hasattr(model.module, "blocks"):
        return model.module
    return model


def _rope_cfg_is_identity(rope_cfg: Wan21T2VRopePatchConfig) -> bool:
    """Return True when rope behavior is equivalent to original full RoPE."""
    if not rope_cfg.enabled:
        return True
    return (
        rope_cfg.mode == "full"
        and rope_cfg.f_scale == 1.0
        and rope_cfg.h_scale == 1.0
        and rope_cfg.w_scale == 1.0
    )


def _model_uses_usp_attention(target_model) -> bool:
    """Detect whether current DiT blocks use USP attention forward."""
    if not getattr(target_model, "blocks", None):
        return False
    fn = getattr(target_model.blocks[0].self_attn.forward, "__func__", None)
    if fn is None:
        return False
    return fn.__name__ == "usp_attn_forward"


def install_wan21_t2v_dit_patch_stack(model, patch_cfg: Wan21T2VPatchBundleConfig) -> Wan21T2VPatchHandle:
    """Install composable patches on Wan2.1-T2V DiT model.

    This single installer applies:
    - RoPE axis intervention,
    - attention probe collection,
    - causal attention schedule.

    Returns:
        Wan21T2VPatchHandle for restoring all original methods.
    """

    target = _unwrap_dit_model(model)
    state = Wan21T2VProbeState(
        config=patch_cfg,
        num_layers=len(target.blocks),
        num_heads=target.num_heads,
    )
    restore_items = []
    needs_custom_attn = patch_cfg.probe.enabled or patch_cfg.causal.enabled
    rope_is_identity = _rope_cfg_is_identity(patch_cfg.rope)

    # Pure RoPE experiments: patch rope_apply only and keep Wan's original
    # attention path (including USP xFuser long-context attention).
    if not needs_custom_attn:
        if not rope_is_identity:
            original_model_rope_apply = wan.modules.model.rope_apply

            def patched_model_rope_apply(x, grid_sizes, freqs):
                return apply_wan21_t2v_rope_patch(x, grid_sizes, freqs, patch_cfg.rope)

            wan.modules.model.rope_apply = patched_model_rope_apply
            restore_items.append((wan.modules.model, "rope_apply", original_model_rope_apply))

            try:
                import wan.distributed.xdit_context_parallel as usp_module

                original_usp_rope_apply = usp_module.rope_apply

                def patched_usp_rope_apply(x, grid_sizes, freqs):
                    return apply_wan21_t2v_rope_patch(x, grid_sizes, freqs, patch_cfg.rope)

                usp_module.rope_apply = patched_usp_rope_apply
                restore_items.append((usp_module, "rope_apply", original_usp_rope_apply))
            except Exception:
                # USP module may be unavailable in single-GPU envs.
                pass

        return Wan21T2VPatchHandle(
            target_model=target,
            state=state,
            original_forward=None,
            original_attn_forwards=[],
            restore_items=restore_items,
        )

    # Attention probe / causal modifications are currently implemented against
    # standard attention forward. Do not mix with USP path.
    if _model_uses_usp_attention(target):
        raise RuntimeError(
            "Attention probe/causal patch is not compatible with USP attention path. "
            "Please disable `use_usp` for this experiment."
        )

    original_forward = target.forward

    def patched_dit_forward(this, *args, **kwargs):
        t = kwargs.get("t", None)
        if t is None and len(args) > 1:
            t = args[1]
        state.on_forward_start(t)
        return original_forward(*args, **kwargs)

    target.forward = MethodType(patched_dit_forward, target)

    original_attn_forwards = []

    def build_patched_self_attn(layer_idx: int):
        """Create per-layer self-attention patch closure."""

        def patched_self_attn(self, x, seq_lens, grid_sizes, freqs):
            b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)

            q = apply_wan21_t2v_rope_patch(q, grid_sizes, freqs, patch_cfg.rope)
            k = apply_wan21_t2v_rope_patch(k, grid_sizes, freqs, patch_cfg.rope)

            state.collect(layer_idx=layer_idx, q=q, k=k, seq_lens=seq_lens, grid_sizes=grid_sizes)

            causal_enabled = state.should_apply_causal()
            out = _attention_dispatch(
                attn_module=wan.modules.attention,
                q=q,
                k=k,
                v=v,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                causal_enabled=causal_enabled,
                causal_mode=patch_cfg.causal.mode,
                window_size=self.window_size,
                backend=patch_cfg.causal.backend,
            )

            out = out.flatten(2)
            out = self.o(out)
            return out

        return patched_self_attn

    for layer_idx, block in enumerate(target.blocks):
        original_attn_forwards.append(block.self_attn.forward)
        block.self_attn.forward = MethodType(build_patched_self_attn(layer_idx), block.self_attn)

    return Wan21T2VPatchHandle(
        target_model=target,
        state=state,
        original_forward=original_forward,
        original_attn_forwards=original_attn_forwards,
        restore_items=restore_items,
    )
