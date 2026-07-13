#!/usr/bin/env python3
"""Analyze timestep-conditioned spatial RoPE lambda distributions for Wan checkpoints.

This script loads the trained `spatial_rope_lambda` module from a Wan fine-tuning
checkpoint, reconstructs the effective height/width RoPE lambda scales used at the
50-step inference timesteps, and exports phase-wise KDE visualizations.

Output layout:

- output_dir/model-wise/diffusion_early_rope_lambda_kde.pdf
- output_dir/model-wise/diffusion_middle_late_rope_lambda_kde.pdf
- output_dir/layer-wise/layer_{i}/diffusion_early_rope_lambda_kde.pdf
- output_dir/layer-wise/layer_{i}/diffusion_middle_late_rope_lambda_kde.pdf

The script also exports lightweight metadata files so the raw lambda parameters
and timestep-conditioned MLP weights can be inspected later.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde

ROOT_DIR = Path(__file__).resolve().parents[1]
WAN_ROOT = ROOT_DIR / "projects" / "Wan2_1"
DIFFSYNTH_ROOT = ROOT_DIR / "DiffSynth-Studio"
if str(WAN_ROOT) not in sys.path:
    sys.path.insert(0, str(WAN_ROOT))
if str(DIFFSYNTH_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFSYNTH_ROOT))

from diffsynth.core import load_state_dict  # noqa: E402
from diffsynth.models.wan_video_dit import WanSpatialRopeLambda  # noqa: E402
from wan.configs import WAN_CONFIGS  # noqa: E402
from wan.utils.fm_solvers import (  # noqa: E402
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # noqa: E402


_STEP_RE = re.compile(r"^(step|epoch)-(\d+)(?:_lambda)?\.safetensors$")
_FLOAT_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def sort_checkpoint_paths(paths: list[Path]) -> list[Path]:
    def key(path: Path):
        match = _STEP_RE.match(path.name)
        if match is None:
            return (0, path.name)
        return (int(match.group(2)), path.name)

    return sorted(paths, key=key)


def parse_experiment_name(experiment_name: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {
        "lambda_enabled": "lambda" in experiment_name,
        "wan_spatial_rope_lambda_scope": None,
        "wan_spatial_rope_lambda_timestep_conditioned": None,
        "wan_spatial_rope_lambda_hidden_dim": None,
        "wan_spatial_rope_lambda_learn_f": None,
        "wan_spatial_rope_lambda_parametrization": None,
        "wan_spatial_rope_lambda_min": None,
        "wan_spatial_rope_lambda_init_eps": None,
        "wan_spatial_rope_lambda_fixed_h": None,
        "wan_spatial_rope_lambda_fixed_w": None,
    }
    patterns = {
        "lambda_scope": r"(?:^|-)lambda_scope_([^\-\s]+)",
        "lambda_tcond": r"(?:^|-)lambda_tcond_([^\-\s]+)",
        "lambda_hidden": r"(?:^|-)lambda_hidden_([^\-\s]+)",
        "lambda_param": r"(?:^|-)lambda_param_([^\-\s]+)",
        "lambda_min": rf"(?:^|-)lambda_min_({_FLOAT_RE})",
        "lambda_init_eps": rf"(?:^|-)lambda_init_eps_({_FLOAT_RE})",
        "lambda_h": rf"(?:^|-)lambda_h_({_FLOAT_RE})",
        "lambda_w": rf"(?:^|-)lambda_w_({_FLOAT_RE})",
    }
    matches = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, experiment_name)
        if match is not None:
            matches[key] = match.group(1)

    if "lambda_scope" in matches:
        parsed["wan_spatial_rope_lambda_scope"] = matches["lambda_scope"]
    if "lambda_tcond" in matches:
        parsed["wan_spatial_rope_lambda_timestep_conditioned"] = bool(int(matches["lambda_tcond"]))
    if "lambda_hidden" in matches:
        parsed["wan_spatial_rope_lambda_hidden_dim"] = int(matches["lambda_hidden"])
    if "lambda_param" in matches:
        parsed["wan_spatial_rope_lambda_parametrization"] = matches["lambda_param"]
    range_match = re.search(
        rf"(?:^|-)range_(unconstrained|softplus_leq_one|bounded_leq_one|fixed)"
        rf"(?:_min_({_FLOAT_RE}))?"
        rf"(?:_eps_({_FLOAT_RE}))?",
        experiment_name,
    )
    if range_match is not None:
        parsed["wan_spatial_rope_lambda_parametrization"] = range_match.group(1)
        if range_match.group(2) is not None:
            parsed["wan_spatial_rope_lambda_min"] = float(range_match.group(2))
        if range_match.group(3) is not None:
            parsed["wan_spatial_rope_lambda_init_eps"] = float(range_match.group(3))
    if "lambda_min" in matches:
        parsed["wan_spatial_rope_lambda_min"] = float(matches["lambda_min"])
    if "lambda_init_eps" in matches:
        parsed["wan_spatial_rope_lambda_init_eps"] = float(matches["lambda_init_eps"])
    if "lambda_h" in matches:
        parsed["wan_spatial_rope_lambda_fixed_h"] = float(matches["lambda_h"])
    if "lambda_w" in matches:
        parsed["wan_spatial_rope_lambda_fixed_w"] = float(matches["lambda_w"])
    return parsed


def resolve_model_artifacts(model_path: str | None, ckpt_step: int | None = None) -> dict[str, Path | None]:
    result: dict[str, Path | None] = {
        "training_args_path": None,
        "main_checkpoint_path": None,
        "lambda_checkpoint_path": None,
    }
    if not model_path:
        return result

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"model_path does not exist: {model_path}")

    if path.is_dir():
        training_args_path = path / "training_args.json"
        if training_args_path.exists():
            result["training_args_path"] = training_args_path

        if ckpt_step is not None:
            step_main = path / f"step-{ckpt_step}.safetensors"
            epoch_main = path / f"epoch-{ckpt_step}.safetensors"
            step_lambda = path / f"step-{ckpt_step}_lambda.safetensors"
            epoch_lambda = path / f"epoch-{ckpt_step}_lambda.safetensors"

            if step_main.exists():
                result["main_checkpoint_path"] = step_main
            elif epoch_main.exists():
                result["main_checkpoint_path"] = epoch_main

            if step_lambda.exists():
                result["lambda_checkpoint_path"] = step_lambda
            elif epoch_lambda.exists():
                result["lambda_checkpoint_path"] = epoch_lambda

            if result["main_checkpoint_path"] is None and result["lambda_checkpoint_path"] is None:
                raise FileNotFoundError(
                    f"Could not find step/epoch checkpoint for CKPT_STEP={ckpt_step} under {path}"
                )

        if result["main_checkpoint_path"] is None:
            main_candidates = [
                p for p in path.glob("*.safetensors")
                if not p.name.endswith("_lambda.safetensors")
            ]
            if main_candidates:
                result["main_checkpoint_path"] = sort_checkpoint_paths(main_candidates)[-1]

        if result["lambda_checkpoint_path"] is None:
            lambda_candidates = [p for p in path.glob("*_lambda.safetensors")]
            if lambda_candidates:
                result["lambda_checkpoint_path"] = sort_checkpoint_paths(lambda_candidates)[-1]
    else:
        if path.name.endswith("_lambda.safetensors"):
            result["lambda_checkpoint_path"] = path
        else:
            result["main_checkpoint_path"] = path
            sibling_lambda = path.with_name(path.stem + "_lambda" + path.suffix)
            if sibling_lambda.exists():
                result["lambda_checkpoint_path"] = sibling_lambda
        training_args_path = path.parent / "training_args.json"
        if training_args_path.exists():
            result["training_args_path"] = training_args_path
    return result


def collect_inferred_training_args(model_path: str | None, training_args_path: Path | None) -> dict[str, Any]:
    inferred: dict[str, Any] = {}
    if training_args_path is not None:
        inferred.update(load_json(training_args_path))
    if model_path:
        path = Path(model_path)
        experiment_name = path.name if path.is_dir() else path.parent.name
        parsed_from_name = parse_experiment_name(experiment_name)
        for key, value in parsed_from_name.items():
            if key not in inferred or inferred.get(key) in (None, "", False):
                inferred[key] = value
    return inferred


def clean_lambda_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("pipe.dit.spatial_rope_lambda."):
            cleaned[key[len("pipe.dit.spatial_rope_lambda."):]] = value
        elif key.startswith("spatial_rope_lambda."):
            cleaned[key[len("spatial_rope_lambda."):]] = value
    return cleaned


def find_lambda_checkpoint(artifacts: dict[str, Path | None], explicit_path: str | None = None) -> Path:
    if explicit_path:
        explicit = Path(explicit_path)
        if not explicit.exists():
            raise FileNotFoundError(f"Explicit lambda checkpoint does not exist: {explicit}")
        return explicit

    if artifacts["lambda_checkpoint_path"] is not None:
        return artifacts["lambda_checkpoint_path"]

    if artifacts["main_checkpoint_path"] is not None:
        state = load_state_dict(str(artifacts["main_checkpoint_path"]))
        cleaned = clean_lambda_state(state)
        if cleaned:
            return artifacts["main_checkpoint_path"]

    raise FileNotFoundError(
        "Could not locate a lambda checkpoint. Please pass --spatial_rope_lambda_checkpoint "
        "or provide a model directory/file that contains spatial_rope_lambda weights."
    )


def load_lambda_module(args) -> tuple[WanSpatialRopeLambda, dict[str, Any]]:
    artifacts = resolve_model_artifacts(args.model_path, ckpt_step=args.ckpt_step)
    inferred = collect_inferred_training_args(args.model_path, artifacts["training_args_path"])
    lambda_checkpoint_path = find_lambda_checkpoint(artifacts, args.spatial_rope_lambda_checkpoint)
    lambda_state = load_state_dict(str(lambda_checkpoint_path))
    cleaned_state = clean_lambda_state(lambda_state)
    if not cleaned_state:
        raise RuntimeError(f"Checkpoint {lambda_checkpoint_path} does not contain spatial_rope_lambda weights.")

    cfg = WAN_CONFIGS[args.task]
    scope = args.lambda_scope or inferred.get("wan_spatial_rope_lambda_scope") or "layer"
    timestep_conditioned = bool(
        args.lambda_timestep_conditioned
        if args.lambda_timestep_conditioned is not None
        else inferred.get("wan_spatial_rope_lambda_timestep_conditioned", False)
    )
    hidden_dim = int(args.lambda_hidden_dim or inferred.get("wan_spatial_rope_lambda_hidden_dim", 128))
    learn_f = bool(inferred.get("wan_spatial_rope_lambda_learn_f", False))
    parametrization = args.lambda_parametrization or inferred.get("wan_spatial_rope_lambda_parametrization") or "unconstrained"
    lambda_min = float(args.lambda_min if args.lambda_min is not None else inferred.get("wan_spatial_rope_lambda_min", 0.5))
    lambda_init_eps = float(
        args.lambda_init_eps
        if args.lambda_init_eps is not None
        else inferred.get("wan_spatial_rope_lambda_init_eps", 1e-4)
    )
    fixed_h = float(args.lambda_fixed_h if args.lambda_fixed_h is not None else inferred.get("wan_spatial_rope_lambda_fixed_h", 1.0))
    fixed_w = float(args.lambda_fixed_w if args.lambda_fixed_w is not None else inferred.get("wan_spatial_rope_lambda_fixed_w", 1.0))

    module = WanSpatialRopeLambda(
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        freq_dim=cfg.freq_dim,
        scope=scope,
        learn_f=learn_f,
        timestep_conditioned=timestep_conditioned,
        hidden_dim=hidden_dim,
        parametrization=parametrization,
        lambda_min=lambda_min,
        init_eps=lambda_init_eps,
        fixed_h=fixed_h,
        fixed_w=fixed_w,
    )
    missing, unexpected = module.load_state_dict(cleaned_state, strict=False)
    logging.info(
        "Loaded lambda checkpoint %s with %d keys, %d missing, %d unexpected.",
        lambda_checkpoint_path,
        len(cleaned_state),
        len(missing),
        len(unexpected),
    )
    module.eval()

    metadata = {
        "lambda_checkpoint_path": str(lambda_checkpoint_path),
        "training_args_path": None if artifacts["training_args_path"] is None else str(artifacts["training_args_path"]),
        "ckpt_step": args.ckpt_step,
        "scope": scope,
        "timestep_conditioned": timestep_conditioned,
        "hidden_dim": hidden_dim,
        "learn_f": learn_f,
        "parametrization": parametrization,
        "lambda_min": lambda_min,
        "lambda_init_eps": lambda_init_eps,
        "fixed_h": fixed_h,
        "fixed_w": fixed_w,
        "num_layers": cfg.num_layers,
        "num_heads": cfg.num_heads,
        "freq_dim": cfg.freq_dim,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }
    return module, metadata


def get_inference_timesteps(sample_solver: str, sample_steps: int, sample_shift: float) -> list[int]:
    if sample_solver == "unipc":
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=1,
            use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(sample_steps, device=torch.device("cpu"), shift=sample_shift)
        return [int(t.item()) for t in scheduler.timesteps]

    if sample_solver == "dpm++":
        scheduler = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            shift=1,
            use_dynamic_shifting=False,
        )
        sampling_sigmas = get_sampling_sigmas(sample_steps, sample_shift)
        timesteps, _ = retrieve_timesteps(scheduler, device=torch.device("cpu"), sigmas=sampling_sigmas)
        return [int(t.item()) for t in timesteps]

    raise ValueError(f"Unsupported sample_solver: {sample_solver}")


def expand_scales_to_layer_head(scales: torch.Tensor, num_layers: int, num_heads: int) -> torch.Tensor:
    if scales.ndim == 1:
        return scales.view(1, 1, 3).expand(num_layers, num_heads, 3)
    if scales.ndim == 2:
        if scales.shape[0] != num_layers or scales.shape[1] != 3:
            raise ValueError(f"Expected layer-wise scales with shape [{num_layers}, 3], got {tuple(scales.shape)}")
        return scales.unsqueeze(1).expand(num_layers, num_heads, 3)
    if scales.ndim == 3:
        if scales.shape[0] != num_layers or scales.shape[1] != num_heads or scales.shape[2] != 3:
            raise ValueError(
                f"Expected head-wise scales with shape [{num_layers}, {num_heads}, 3], got {tuple(scales.shape)}"
            )
        return scales
    raise ValueError(f"Unsupported scale tensor shape: {tuple(scales.shape)}")


def compute_stepwise_rope_lambdas(module: WanSpatialRopeLambda, timesteps: list[int], num_layers: int, num_heads: int) -> list[dict[str, Any]]:
    results = []
    for timestep in timesteps:
        timestep_tensor = torch.tensor([float(timestep)], dtype=torch.float32)
        with torch.no_grad():
            scales = module(timestep_tensor).detach().cpu()
        expanded = expand_scales_to_layer_head(scales, num_layers=num_layers, num_heads=num_heads)
        results.append(
            {
                "timestep": int(timestep),
                "height": expanded[..., 1].numpy(),
                "width": expanded[..., 2].numpy(),
            }
        )
    return results


def pooled_phase_values(stepwise: list[dict[str, Any]], axis: str, layer_id: int | None = None) -> np.ndarray:
    values = []
    for item in stepwise:
        axis_values = item[axis]
        if layer_id is None:
            values.append(axis_values.reshape(-1))
        else:
            values.append(axis_values[layer_id].reshape(-1))
    if not values:
        return np.array([], dtype=np.float64)
    return np.concatenate(values, axis=0).astype(np.float64)


def build_density_curve(values: np.ndarray, x_bounds: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    unique = np.unique(np.round(values, 12))
    if unique.size <= 1:
        return None
    value_min = float(values.min())
    value_max = float(values.max())
    pad = max((value_max - value_min) * 0.15, 1e-4)
    x_min = value_min - pad
    x_max = value_max + pad
    if x_bounds is not None:
        x_min = max(x_min, float(x_bounds[0]))
        x_max = min(x_max, float(x_bounds[1]))
        if x_min >= x_max:
            x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
    xs = np.linspace(x_min, x_max, 512)
    kde = gaussian_kde(values)
    ys = kde(xs)
    return xs, ys


def plot_phase_distribution(
    output_path: Path,
    height_values: np.ndarray,
    width_values: np.ndarray,
    phase_title: str,
    timestep_list: list[int],
    layer_id: int | None = None,
    x_bounds: tuple[float, float] | None = None,
    dpi: int = 200,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    configs = [
        ("height", height_values, "tab:blue"),
        ("width", width_values, "tab:orange"),
    ]

    for ax, (axis_name, values, color) in zip(axes, configs):
        values = np.asarray(values, dtype=np.float64)
        curve = build_density_curve(values, x_bounds=x_bounds)
        if curve is None:
            if values.size == 0:
                ax.text(0.5, 0.5, "No values", ha="center", va="center", transform=ax.transAxes)
            else:
                spike_x = float(values.reshape(-1)[0])
                ax.axvline(spike_x, color=color, linewidth=2.5)
                ax.fill_between([spike_x - 1e-6, spike_x + 1e-6], [0, 0], [1, 1], color=color, alpha=0.18)
        else:
            xs, ys = curve
            ax.plot(xs, ys, color=color, linewidth=2)
            ax.fill_between(xs, 0, ys, color=color, alpha=0.22)

        ax.set_title(f"{axis_name.capitalize()} RoPE lambda scale")
        ax.set_xlabel("Effective lambda scale")
        ax.set_ylabel("Density")
        if x_bounds is not None:
            ax.set_xlim(*x_bounds)
        ax.grid(alpha=0.2, linestyle="--")

        mean = float(values.mean()) if values.size else float("nan")
        std = float(values.std()) if values.size else float("nan")
        vmin = float(values.min()) if values.size else float("nan")
        vmax = float(values.max()) if values.size else float("nan")
        summary_lines = [
            f"count={values.size}",
            f"mean={mean:.6f}",
            f"std={std:.6f}",
            f"min={vmin:.6f}",
            f"max={vmax:.6f}",
        ]
        ax.text(
            0.98,
            0.97,
            "\n".join(summary_lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    title_prefix = phase_title if layer_id is None else f"Layer {layer_id} - {phase_title}"
    fig.suptitle(f"{title_prefix} | timesteps={timestep_list}", fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def split_phase_steps(stepwise: list[dict[str, Any]], early_fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not stepwise:
        return [], []
    early_count = max(1, int(math.ceil(len(stepwise) * early_fraction)))
    early = stepwise[:early_count]
    middle_late = stepwise[early_count:]
    if not middle_late:
        middle_late = stepwise[-1:]
    return early, middle_late


def save_metadata(output_dir: Path, metadata: dict[str, Any], module: WanSpatialRopeLambda, timesteps: list[int], early_steps: list[int], middle_late_steps: list[int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(module.state_dict(), output_dir / "lambda_module_state.pt")
    summary = dict(metadata)
    summary.update(
        {
            "timesteps": timesteps,
            "early_timesteps": early_steps,
            "middle_late_timesteps": middle_late_steps,
            "base_log_scale_shape": list(module.base_log_scale.shape),
            "base_raw_values": module.base_log_scale.detach().cpu().tolist(),
            "base_lambda_values": torch.exp(module._spatial_log_scale_for_summary(timestep=None, update_last=False).detach().cpu()).tolist(),
            "has_timestep_mlp": module.timestep_mlp is not None,
            "timestep_mlp_parameter_shapes": {
                key: list(value.shape) for key, value in module.state_dict().items() if key.startswith("timestep_mlp.")
            },
        }
    )
    with open(output_dir / "analysis_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze trained Wan spatial RoPE lambda distributions.")
    parser.add_argument("--model_path", type=str, required=True, help="Experiment directory or checkpoint path used to resolve the lambda checkpoint.")
    parser.add_argument("--task", type=str, default="t2v-1.3B", choices=("t2v-1.3B", "t2v-14B"), help="Wan task used to determine num_layers and num_heads.")
    parser.add_argument("--ckpt_step", type=int, default=None, help="Optional step or epoch id used to resolve step-{ckpt_step}_lambda.safetensors from an experiment directory.")
    parser.add_argument("--output_dir", type=str, default="./", help="Directory to store visualizations and metadata.")
    parser.add_argument("--sample_solver", type=str, default="unipc", choices=("unipc", "dpm++"), help="Inference solver used to generate the 50-step timestep sequence.")
    parser.add_argument("--sample_steps", type=int, default=50, help="Number of inference denoising steps used for phase analysis.")
    parser.add_argument("--sample_shift", type=float, default=5.0, help="Wan inference shift parameter.")
    parser.add_argument("--early_fraction", type=float, default=0.1, help="Fraction of inference steps treated as diffusion-early.")
    parser.add_argument("--spatial_rope_lambda_checkpoint", type=str, default="", help="Optional explicit lambda checkpoint path.")
    parser.add_argument("--lambda_scope", type=str, default="", choices=("", "model", "layer", "head"), help="Optional manual override for lambda scope.")
    parser.add_argument("--lambda_timestep_conditioned", type=int, choices=(0, 1), default=None, help="Optional manual override for whether timestep-conditioned lambda is enabled.")
    parser.add_argument("--lambda_hidden_dim", type=int, default=None, help="Optional manual override for timestep-conditioned MLP hidden dimension.")
    parser.add_argument("--lambda_parametrization", type=str, default="", choices=("", "unconstrained", "softplus_leq_one", "bounded_leq_one", "fixed"), help="Optional manual override for spatial RoPE lambda parameterization.")
    parser.add_argument("--lambda_min", type=float, default=None, help="Optional manual override for bounded_leq_one lower bound.")
    parser.add_argument("--lambda_init_eps", type=float, default=None, help="Optional manual override for constrained near-identity initialization epsilon.")
    parser.add_argument("--lambda_fixed_h", type=float, default=None, help="Optional manual override for fixed height-axis lambda.")
    parser.add_argument("--lambda_fixed_w", type=float, default=None, help="Optional manual override for fixed width-axis lambda.")
    parser.add_argument("--dpi", type=int, default=200, help="Figure dpi.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    output_dir = Path(args.output_dir)
    model_wise_dir = output_dir / "model-wise"
    layer_wise_dir = output_dir / "layer-wise"

    module, metadata = load_lambda_module(args)
    timesteps = get_inference_timesteps(args.sample_solver, args.sample_steps, args.sample_shift)
    cfg = WAN_CONFIGS[args.task]
    stepwise = compute_stepwise_rope_lambdas(module, timesteps, num_layers=cfg.num_layers, num_heads=cfg.num_heads)
    early_stepwise, middle_late_stepwise = split_phase_steps(stepwise, args.early_fraction)
    parametrization = metadata.get("parametrization", "unconstrained")
    if parametrization == "bounded_leq_one":
        lambda_x_bounds = (float(metadata.get("lambda_min", 0.0)), 1.0)
    elif parametrization == "softplus_leq_one":
        lambda_x_bounds = (0.0, 1.0)
    else:
        lambda_x_bounds = None

    save_metadata(
        output_dir=output_dir,
        metadata=metadata,
        module=module,
        timesteps=timesteps,
        early_steps=[item["timestep"] for item in early_stepwise],
        middle_late_steps=[item["timestep"] for item in middle_late_stepwise],
    )

    plot_phase_distribution(
        output_path=model_wise_dir / "diffusion_early_rope_lambda_kde.pdf",
        height_values=pooled_phase_values(early_stepwise, axis="height"),
        width_values=pooled_phase_values(early_stepwise, axis="width"),
        phase_title="Diffusion early (first 10%)",
        timestep_list=[item["timestep"] for item in early_stepwise],
        x_bounds=lambda_x_bounds,
        dpi=args.dpi,
    )
    plot_phase_distribution(
        output_path=model_wise_dir / "diffusion_middle_late_rope_lambda_kde.pdf",
        height_values=pooled_phase_values(middle_late_stepwise, axis="height"),
        width_values=pooled_phase_values(middle_late_stepwise, axis="width"),
        phase_title="Diffusion middle/late (remaining 90%)",
        timestep_list=[item["timestep"] for item in middle_late_stepwise],
        x_bounds=lambda_x_bounds,
        dpi=args.dpi,
    )

    for layer_id in range(cfg.num_layers):
        layer_dir = layer_wise_dir / f"layer_{layer_id}"
        plot_phase_distribution(
            output_path=layer_dir / "diffusion_early_rope_lambda_kde.pdf",
            height_values=pooled_phase_values(early_stepwise, axis="height", layer_id=layer_id),
            width_values=pooled_phase_values(early_stepwise, axis="width", layer_id=layer_id),
            phase_title="Diffusion early (first 10%)",
            timestep_list=[item["timestep"] for item in early_stepwise],
            layer_id=layer_id,
            x_bounds=lambda_x_bounds,
            dpi=args.dpi,
        )
        plot_phase_distribution(
            output_path=layer_dir / "diffusion_middle_late_rope_lambda_kde.pdf",
            height_values=pooled_phase_values(middle_late_stepwise, axis="height", layer_id=layer_id),
            width_values=pooled_phase_values(middle_late_stepwise, axis="width", layer_id=layer_id),
            phase_title="Diffusion middle/late (remaining 90%)",
            timestep_list=[item["timestep"] for item in middle_late_stepwise],
            layer_id=layer_id,
            x_bounds=lambda_x_bounds,
            dpi=args.dpi,
        )

    logging.info("Lambda analysis finished. Outputs are stored in %s", output_dir)


if __name__ == "__main__":
    main()
