"""CLI entry for Wan2.1-T2V monkey-patch experiments."""

import argparse
import os
import sys
from datetime import datetime
from typing import List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)


def _str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "t", "yes", "y"}


def _parse_csv_ints(s: str) -> List[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_csv_strs(s: str) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_wan21_t2v_channel_profile_target_pairs(s: str) -> List[Tuple[int, int]]:
    if not s:
        return []
    pairs: List[Tuple[int, int]] = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                "self_attention_modulation_channel_profile_targets must use step:layer pairs, "
                f"got {item!r}."
            )
        step_text, layer_text = item.split(":", 1)
        pairs.append((int(step_text.strip()), int(layer_text.strip())))
    return pairs


def _parse_optional_csv_strs_with_none(s: str) -> Optional[List[str]]:
    """Parse a CSV string with an explicit `None` sentinel.

    Semantics:
    - `""` -> `[]`, meaning "use all items by default"
    - `"None"` / `"none"` -> `None`, meaning "use no items"
    - otherwise -> parsed CSV list
    """
    if s is None:
        return []
    text = str(s).strip()
    if text.lower() == "none":
        return None
    if text == "":
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def _parse_size(size: str) -> Tuple[int, int]:
    w, h = size.split("*")
    return int(w), int(h)


def _default_wan21_root() -> str:
    return os.path.normpath(os.path.join(PARENT, "projects", "Wan2_1"))


def _default_output_dir(experiment_name: str) -> str:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.normpath(os.path.join(PARENT, "outputs_wan_2_1", f"wan21_t2v_{experiment_name}_{now}"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wan2.1-T2V experiments via monkey patch.")

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=[
            "rope_axis_ablation",
            "attention_dt_profile",
            "trajectory_entropy",
            "head_evolution",
            "head_trajectory_dynamics",
            "rope_decay_curve",
            "self_attention_temporal_kernel",
            "self_attention_distribution",
            "self_attention_viz",
            "self_attention_modulation",
            "seed_to_trajectory_predictability",
            "event_token_value",
            "motion_aligned_attention",
            "causal_schedule",
            "step_window_cross_attn_off",
            "step_window_ffn_off",
            "cross_attn_head_ablation",
            "step_window_prompt_replace",
            "cross_attention_token_viz",
            "token_trajectory_seed_stability",
            "joint_attention_suite",
            "rope_modification",
            "trajectory_consensus_dynamics",
        ],
    )

    parser.add_argument("--wan21_root", type=str, default=_default_wan21_root())
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument("--task", type=str, default="t2v-14B", choices=["t2v-14B", "t2v-1.3B"])
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--size", type=str, default="832*480")

    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    parser.add_argument("--sampling_steps", type=int, default=50)
    parser.add_argument("--shift", type=float, default=8.0)
    parser.add_argument("--guide_scale", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device_id", type=int, default=None)
    parser.add_argument("--offload_model", type=_str2bool, default=True)

    # Parallel options
    parser.add_argument("--t5_fsdp", action="store_true", default=False)
    parser.add_argument("--dit_fsdp", action="store_true", default=False)
    parser.add_argument("--t5_cpu", action="store_true", default=False)
    parser.add_argument("--use_usp", action="store_true", default=False)
    parser.add_argument("--ulysses_size", type=int, default=1)
    parser.add_argument("--ring_size", type=int, default=1)

    # Probe options
    parser.add_argument("--probe_steps", type=str, default="1,2,3")
    parser.add_argument("--query_frame_count", type=int, default=8)
    parser.add_argument("--probe_query_mode", type=str, default="center", choices=["center", "multi_anchor", "object_guided"])
    parser.add_argument("--probe_branch", type=str, default="uncond", choices=["uncond", "cond", "both"])

    # Rope experiment options
    parser.add_argument("--rope_modes", type=str, default="full,no_f,no_h,no_w,only_f,only_hw")
    parser.add_argument(
        "--rope_modification_mode",
        type=str,
        default="manual",
        choices=["manual", "spatial_temporal_reweight", "timestep_conditioned"],
        help=(
            "RoPE modification mode: training-free manual scaling, "
            "training-free spatial-temporal reweight, or unified "
            "timestep-conditioned scale learning."
        ),
    )
    parser.add_argument("--rope_modification_lambda_f", type=float, default=1.0)
    parser.add_argument("--rope_modification_lambda_h", type=float, default=1.0)
    parser.add_argument("--rope_modification_lambda_w", type=float, default=1.0)
    parser.add_argument(
        "--rope_modification_spatial_temporal_reweight_alpha",
        type=float,
        default=0.5,
        help=(
            "Temporal channel weight alpha for spatial_temporal_reweight mode. "
            "Temporal channels receive sqrt(alpha); spatial channels receive sqrt(1-alpha)."
        ),
    )
    parser.add_argument(
        "--rope_modification_steps",
        type=str,
        default="",
        help="CSV diffusion-step list for rope_modification. Empty means apply to all steps.",
    )
    parser.add_argument(
        "--rope_modification_timestep_conditioned_resolution",
        type=str,
        default="global",
        choices=["global", "head_aware"],
        help="Resolution of timestep-conditioned scale learning.",
    )
    parser.add_argument("--rope_modification_timestep_conditioned_hidden_dim", type=int, default=128)
    parser.add_argument(
        "--rope_modification_timestep_conditioned_checkpoint",
        type=str,
        default="",
        help="Optional state_dict path for the timestep-conditioned RoPE scale learner.",
    )
    parser.add_argument("--rope_modification_semantic_residual_enabled", type=_str2bool, default=False)
    parser.add_argument("--rope_modification_semantic_residual_alpha", type=float, default=0.0)
    parser.add_argument(
        "--rope_modification_semantic_residual_steps",
        type=str,
        default="",
        help="CSV diffusion-step list for semantic residual attention. Empty means apply to all steps.",
    )
    parser.add_argument("--rope_modification_semantic_residual_query_chunk_size", type=int, default=64)
    parser.add_argument("--rope_modification_semantic_residual_timestep_conditioned", type=_str2bool, default=False)
    parser.add_argument(
        "--rope_modification_semantic_residual_timestep_conditioned_resolution",
        type=str,
        default="global",
        choices=["global", "head_aware"],
        help="Resolution of timestep-conditioned semantic residual alpha learning.",
    )
    parser.add_argument("--rope_modification_semantic_residual_timestep_conditioned_hidden_dim", type=int, default=128)
    parser.add_argument(
        "--rope_modification_semantic_residual_timestep_conditioned_checkpoint",
        type=str,
        default="",
        help="Optional state_dict path for the timestep-conditioned semantic residual alpha head.",
    )

    # Motion-aligned options
    parser.add_argument("--maas_layers", type=str, default="0,10,20,30,39")
    parser.add_argument("--maas_radius", type=int, default=1)
    parser.add_argument(
        "--motion_target_source",
        type=str,
        default="motion_centroid",
        choices=["motion_centroid", "object_token_trajectory"],
    )

    # Causal schedule options
    parser.add_argument("--causal_first_n_steps", type=int, default=3)
    parser.add_argument("--causal_mode", type=str, default="flat", choices=["flat", "temporal"])
    parser.add_argument("--attention_backend", type=str, default="auto", choices=["auto", "flash", "torch_sdpa"])
    parser.add_argument("--run_baseline", type=_str2bool, default=True)
    parser.add_argument("--collect_dt_profile", type=_str2bool, default=True)
    parser.add_argument(
        "--condition_remove_start_steps",
        type=str,
        default="9999",
        help=(
            "CSV step list for step_window_cross_attn_off. "
            "Interpretation is controlled by --condition_remove_scope."
        ),
    )
    parser.add_argument(
        "--condition_remove_scope",
        type=str,
        default="from_step",
        choices=["single_step", "from_step"],
        help="Apply condition removal on only selected step or selected step and all later steps.",
    )
    parser.add_argument(
        "--reuse_removed_cond_for_uncond",
        type=_str2bool,
        default=False,
        help="When condition is removed, reuse cond output for uncond branch to avoid duplicated compute.",
    )
    parser.add_argument(
        "--timestep_idx_to_remove_ffn",
        type=str,
        default="1",
        help=(
            "CSV step list for step_window_ffn_off. "
            "Interpretation is controlled by --ffn_remove_scope."
        ),
    )
    parser.add_argument(
        "--layer_idx_to_remove_ffn",
        type=str,
        default="",
        help="Optional CSV layer indices for FFN ablation. Empty means all layers.",
    )
    parser.add_argument(
        "--ffn_remove_scope",
        type=str,
        default="single_step",
        choices=["single_step", "from_step"],
        help="Apply FFN removal on only selected step or selected step and all later steps.",
    )
    parser.add_argument(
        "--ablate_heads",
        type=str,
        default="",
        help=(
            "CSV head specs for cross_attn_head_ablation. "
            "Canonical format: LxHy (e.g., L29H7,L12H3). "
            "Also accepts (x,y)."
        ),
    )
    parser.add_argument(
        "--head_ablation_steps",
        type=str,
        default="",
        help=(
            "CSV diffusion steps for cross_attn_head_ablation. "
            "Empty means apply on all steps [1..sampling_steps]."
        ),
    )
    parser.add_argument(
        "--replacement_prompt",
        type=str,
        default="",
        help="Replacement prompt used by step_window_prompt_replace experiment.",
    )
    parser.add_argument(
        "--prompt_replace_steps",
        type=str,
        default="1",
        help="CSV step list for step_window_prompt_replace experiment.",
    )
    parser.add_argument(
        "--prompt_replace_scope",
        type=str,
        default="from_step",
        choices=["single_step", "from_step"],
        help="Apply prompt replacement on only selected step or selected step and all later steps.",
    )
    parser.add_argument(
        "--replace_cond_only",
        type=_str2bool,
        default=True,
        help="If true, replace prompt only for cond branch; otherwise replace both cond/uncond branches.",
    )

    # Cross-attention token visualization options
    parser.add_argument("--target_object_words", type=str, default="")
    parser.add_argument("--target_verb_words", type=str, default="")
    parser.add_argument("--cross_attn_steps", type=str, default="1,2,3")
    parser.add_argument("--viz_num_frames", type=int, default=5)
    parser.add_argument("--viz_frame_indices", type=str, default="")
    parser.add_argument("--viz_layers", type=str, default="")
    parser.add_argument("--cross_attn_chunk_size", type=int, default=1024)
    parser.add_argument("--traj_enable", type=_str2bool, default=True)
    parser.add_argument("--traj_style", type=str, default="glow_arrow", choices=["glow", "arrow", "glow_arrow"])
    parser.add_argument("--traj_num_frames", type=int, default=0)
    parser.add_argument("--traj_smooth_radius", type=int, default=2)
    parser.add_argument("--traj_power", type=float, default=1.5)
    parser.add_argument("--traj_quantile", type=float, default=0.8)
    parser.add_argument("--traj_arrow_stride", type=int, default=4)
    parser.add_argument("--traj_include_head_mean", type=_str2bool, default=True)
    parser.add_argument("--save_attention_pdfs", type=_str2bool, default=True)
    parser.add_argument(
        "--attention_pdf_per_frame_normalize",
        type=_str2bool,
        default=False,
        help="If true, normalize each cross-attention map frame over H*W before drawing attention PDFs.",
    )
    parser.add_argument(
        "--attention_pdf_share_color_scale",
        type=_str2bool,
        default=False,
        help="If true, all displayed frames in one attention PDF share the same vmin/vmax color scale.",
    )
    parser.add_argument(
        "--skip_existing_pdfs",
        type=_str2bool,
        default=True,
        help=(
            "When saving visualization PDFs (attention/trajectory/timeline), "
            "skip files that already exist in output_dir. Useful for resume after interruption."
        ),
    )
    parser.add_argument("--save_trajectory_pdfs", type=_str2bool, default=True)
    parser.add_argument("--save_trajectory_timeline_pdfs", type=_str2bool, default=True)
    parser.add_argument("--trajectory_timeline_num_frames", type=int, default=10)
    parser.add_argument("--save_video", type=_str2bool, default=True)
    parser.add_argument("--stream_flush_per_step", type=_str2bool, default=False)
    parser.add_argument("--plot_during_sampling", type=_str2bool, default=False)
    parser.add_argument("--draw_attention_map_only", type=_str2bool, default=False)
    parser.add_argument("--draw_attention_maps_path", type=str, default="")
    parser.add_argument("--visualization_output_dir", type=str, default="")
    parser.add_argument(
        "--cross_attention_token_viz_num_workers",
        type=int,
        default=0,
        help=(
            "Number of CPU worker processes used by cross_attention_token_viz to render PDFs "
            "and extract per-head trajectories. A non-positive value means use os.cpu_count()."
        ),
    )

    # Trajectory-entropy options
    parser.add_argument(
        "--trajectory_entropy_steps",
        type=str,
        default="",
        help="CSV steps for trajectory_entropy. Empty means all steps [1..sampling_steps].",
    )
    parser.add_argument(
        "--trajectory_entropy_layerwise_steps",
        type=str,
        default="",
        help=(
            "CSV steps for layer-wise entropy. "
            "Empty defaults to trajectory_entropy_steps (thus all steps by default)."
        ),
    )
    parser.add_argument(
        "--trajectory_entropy_head_layer",
        type=int,
        default=-1,
        help="Layer index for head-wise entropy. -1 means last layer; -2 means all layers.",
    )
    parser.add_argument(
        "--trajectory_entropy_stepwise_layer",
        type=int,
        default=-1,
        help="Layer index used by step-wise entropy. -1 means last DiT layer.",
    )
    parser.add_argument(
        "--trajectory_entropy_save_video",
        type=_str2bool,
        default=True,
        help="Whether trajectory_entropy should save generated video.",
    )

    # Head-evolution options
    parser.add_argument(
        "--head_evolution_steps",
        type=str,
        default="",
        help="CSV steps for head_evolution step-wise/head-wise analysis. Empty means all available steps.",
    )
    parser.add_argument(
        "--head_evolution_layerwise_steps",
        type=str,
        default="",
        help="CSV steps for head_evolution layer-wise analysis. Empty defaults to head_evolution_steps.",
    )
    parser.add_argument(
        "--head_evolution_head_layer",
        type=int,
        default=-2,
        help="Layer index for head-wise curves. -2 means all layers; -1 means last layer.",
    )
    parser.add_argument(
        "--head_evolution_stepwise_layer",
        type=int,
        default=27,
        help="Layer index used for step-wise curves. -1 means last layer.",
    )
    parser.add_argument("--head_evolution_reference_step", type=int, default=50)
    parser.add_argument("--head_evolution_reference_layer", type=int, default=27)
    parser.add_argument(
        "--head_evolution_center_mode",
        type=str,
        default="centroid",
        choices=["peak", "centroid", "geometric_center"],
        help="Reference center mode used to build support masks.",
    )
    parser.add_argument(
        "--head_evolution_support_radius_mode",
        type=str,
        default="adaptive_area",
        choices=["fixed", "adaptive_area"],
        help="Radius mode for trajectory-support disk neighborhood.",
    )
    parser.add_argument("--head_evolution_support_radius_fixed", type=float, default=2.0)
    parser.add_argument("--head_evolution_support_radius_alpha", type=float, default=1.5)
    parser.add_argument("--head_evolution_support_radius_min", type=float, default=1.0)
    parser.add_argument("--head_evolution_support_radius_max_ratio", type=float, default=0.25)
    parser.add_argument("--head_evolution_traj_power", type=float, default=1.5)
    parser.add_argument("--head_evolution_traj_quantile", type=float, default=0.8)
    parser.add_argument("--head_evolution_reference_viz_num_frames", type=int, default=10)
    parser.add_argument("--head_evolution_save_reference_radius_overlay", type=_str2bool, default=True)
    parser.add_argument("--head_evolution_early_step_end", type=int, default=5)
    parser.add_argument("--head_evolution_score_quantile", type=float, default=0.7)
    parser.add_argument(
        "--head_evolution_apply_preprocess_on_metrics",
        type=_str2bool,
        default=True,
        help=(
            "Whether to apply winsorize+despike preprocessing when computing head_evolution metrics. "
            "Reference trajectory extraction still uses preprocessing regardless of this switch."
        ),
    )
    parser.add_argument("--head_evolution_preprocess_winsorize_quantile", type=float, default=0.995)
    parser.add_argument("--head_evolution_preprocess_despike_quantile", type=float, default=0.98)
    parser.add_argument("--head_evolution_preprocess_min_component_area", type=int, default=2)
    parser.add_argument("--head_evolution_concentrated_region_top_ratio", type=float, default=0.05)

    # Head trajectory dynamics options
    parser.add_argument(
        "--head_trajectory_dynamics_heads",
        type=str,
        default="",
        help="CSV head specs `LxHy` for head_trajectory_dynamics. Empty means all heads.",
    )
    parser.add_argument(
        "--head_trajectory_dynamics_steps",
        type=str,
        default="",
        help="CSV diffusion steps for head_trajectory_dynamics. Empty means all available reused-map steps.",
    )
    parser.add_argument(
        "--head_trajectory_dynamics_distance_metrics",
        type=str,
        default="",
        help=(
            "CSV distance metrics for head_trajectory_dynamics: "
            "js,hellinger,wasserstein_map,support_overlap,center_l2 "
            "(legacy alias: wasserstein->center_l2). Empty means all default metrics."
        ),
    )
    parser.add_argument("--head_trajectory_dynamics_reference_step", type=int, default=50)
    parser.add_argument("--head_trajectory_dynamics_reference_layer", type=int, default=27)
    parser.add_argument("--head_trajectory_dynamics_support_quantile", type=float, default=0.9)
    parser.add_argument("--head_trajectory_dynamics_attractor_window", type=int, default=3)
    parser.add_argument(
        "--head_trajectory_dynamics_attractor_distance_metric",
        type=str,
        default="",
        help=(
            "CSV distance metrics used by attractor analysis when asking whether follower heads "
            "move closer to the current leader head. "
            "Allowed: js,hellinger,wasserstein_map,support_overlap,center_l2. "
            "Empty means all supported metrics."
        ),
    )
    parser.add_argument(
        "--head_trajectory_dynamics_center_method",
        type=str,
        default="region_centroid",
        choices=["region_centroid", "preprocessed_component_center"],
        help=(
            "Center-extraction method for head_trajectory_dynamics. "
            "`region_centroid` matches the localized region-center logic used by cross_attention_token_viz; "
            "`preprocessed_component_center` applies preprocessing and extracts "
            "peak / centroid / geometric_center from the dominant component."
        ),
    )
    parser.add_argument("--head_trajectory_dynamics_center_power", type=float, default=1.5)
    parser.add_argument("--head_trajectory_dynamics_center_quantile", type=float, default=0.8)
    parser.add_argument(
        "--head_trajectory_dynamics_preprocessed_center_mode",
        type=str,
        default="geometric_center",
        choices=["peak", "centroid", "geometric_center"],
        help="Center type used when center_method=preprocessed_component_center.",
    )
    parser.add_argument("--head_trajectory_dynamics_preprocess_winsorize_quantile", type=float, default=0.995)
    parser.add_argument("--head_trajectory_dynamics_preprocess_despike_quantile", type=float, default=0.98)
    parser.add_argument("--head_trajectory_dynamics_preprocess_min_component_area", type=int, default=2)
    parser.add_argument(
        "--head_trajectory_dynamics_reference_center_method",
        type=str,
        default="same_as_head",
        choices=["same_as_head", "region_centroid", "preprocessed_component_center"],
        help=(
            "Reference-trajectory center method. `same_as_head` reuses the ordinary-head center method; "
            "otherwise this setting applies only to the reference trajectory."
        ),
    )
    parser.add_argument(
        "--head_trajectory_dynamics_reference_center_power",
        type=float,
        default=-1.0,
        help="Reference-trajectory center power. Negative values fall back to head_trajectory_dynamics_center_power.",
    )
    parser.add_argument(
        "--head_trajectory_dynamics_reference_center_quantile",
        type=float,
        default=-1.0,
        help="Reference-trajectory center quantile. Negative values fall back to head_trajectory_dynamics_center_quantile.",
    )
    parser.add_argument(
        "--head_trajectory_dynamics_reference_preprocessed_center_mode",
        type=str,
        default="same_as_head",
        choices=["same_as_head", "peak", "centroid", "geometric_center"],
        help=(
            "Reference-only center mode used when the reference method is preprocessed_component_center. "
            "`same_as_head` reuses the ordinary-head preprocessed mode."
        ),
    )
    parser.add_argument(
        "--head_trajectory_dynamics_reference_preprocess_winsorize_quantile",
        type=float,
        default=-1.0,
        help="Reference-only winsorize quantile. Negative values fall back to the ordinary-head setting.",
    )
    parser.add_argument(
        "--head_trajectory_dynamics_reference_preprocess_despike_quantile",
        type=float,
        default=-1.0,
        help="Reference-only despike quantile. Negative values fall back to the ordinary-head setting.",
    )
    parser.add_argument(
        "--head_trajectory_dynamics_reference_preprocess_min_component_area",
        type=int,
        default=-1,
        help="Reference-only minimum connected-component area. Negative values fall back to the ordinary-head setting.",
    )
    parser.add_argument(
        "--head_trajectory_dynamics_center_viz_enable",
        type=_str2bool,
        default=False,
        help="If true, render center-overlay PDFs for selected heads.",
    )
    parser.add_argument(
        "--head_trajectory_dynamics_center_viz_step",
        type=int,
        default=-1,
        help="If >= 1, render center-overlay PDFs for one selected step in head_trajectory_dynamics.",
    )
    parser.add_argument(
        "--head_trajectory_dynamics_center_viz_layer",
        type=int,
        default=-1,
        help="If >= 0, render center-overlay PDFs for one selected layer in head_trajectory_dynamics.",
    )
    parser.add_argument(
        "--head_trajectory_dynamics_center_viz_heads",
        type=str,
        default="",
        help="CSV head specs `LxHy` for center-overlay PDFs. Empty means all heads in the selected step/layer.",
    )
    parser.add_argument(
        "--head_trajectory_dynamics_center_viz_num_frames",
        type=int,
        default=10,
        help="Number of frames shown in each head_trajectory_dynamics center-overlay PDF.",
    )
    parser.add_argument(
        "--head_trajectory_dynamics_support_viz_enable",
        type=_str2bool,
        default=False,
        help="If true, render support-overlap binary-mask PDFs and contour PDFs for selected heads.",
    )
    parser.add_argument(
        "--head_trajectory_dynamics_support_viz_step",
        type=int,
        default=-1,
        help="If >= 1, render support-overlap mask PDFs for one selected step.",
    )
    parser.add_argument(
        "--head_trajectory_dynamics_support_viz_layer",
        type=int,
        default=-1,
        help="If >= 0, render support-overlap mask PDFs for one selected layer.",
    )
    parser.add_argument(
        "--head_trajectory_dynamics_support_viz_heads",
        type=str,
        default="",
        help="CSV head specs `LxHy` for support-overlap mask PDFs. Empty means all heads in the selected scope.",
    )
    parser.add_argument(
        "--head_trajectory_dynamics_support_viz_num_frames",
        type=int,
        default=10,
        help="Number of frames shown in each support-overlap mask PDF.",
    )
    parser.add_argument(
        "--head_trajectory_dynamics_support_viz_contour_min_component_area",
        type=int,
        default=4,
        help="Minimum connected-component area required before drawing a green contour on support-overlap masks.",
    )
    parser.add_argument(
        "--head_trajectory_dynamics_support_cache_num_workers",
        type=int,
        default=0,
        help=(
            "Number of CPU worker processes used to build missing support/motion-planning-region caches. "
            "A non-positive value means use os.cpu_count()."
        ),
    )
    parser.add_argument(
        "--head_trajectory_dynamics_center_cache_num_workers",
        type=int,
        default=0,
        help=(
            "Number of CPU worker processes used to extract missing raw head center trajectories. "
            "A non-positive value means use os.cpu_count()."
        ),
    )
    parser.add_argument(
        "--head_trajectory_dynamics_overlay_num_workers",
        type=int,
        default=0,
        help=(
            "Number of CPU worker processes used to render head_trajectory_dynamics center/support overlays. "
            "A non-positive value means use os.cpu_count()."
        ),
    )
    parser.add_argument(
        "--head_trajectory_dynamics_cache_save_interval",
        type=int,
        default=512,
        help=(
            "Flush head_trajectory_dynamics caches every N new entries. Larger values reduce disk writes "
            "but lose more work if the run is interrupted."
        ),
    )
    parser.add_argument(
        "--head_trajectory_dynamics_hypothesis",
        type=str,
        default="attractor",
        choices=["attractor", "reference-leading", "cross-layer-transmission"],
        help="Hypothesis label used to organize head_trajectory_dynamics metric outputs into a dedicated subdirectory.",
    )
    parser.add_argument(
        "--head_trajectory_dynamics_traj_type",
        type=str,
        default="",
        help="Trajectory subset label used to identify which heads are analyzed; written into output metadata.",
    )
    parser.add_argument(
        "--head_trajectory_dynamics_use_motion_planning_region_before_metrics",
        type=_str2bool,
        default=False,
        help=(
            "If true, use the contour-filtered support region as a motion-planning mask before computing "
            "head_trajectory_dynamics metrics. Maps are zeroed outside the region and renormalized per frame."
        ),
    )
    parser.add_argument(
        "--head_trajectory_dynamics_plot_only_from_csv",
        type=_str2bool,
        default=False,
        help="If true, reuse existing head_trajectory_dynamics CSV outputs in output_dir and only redraw plots.",
    )
    parser.add_argument(
        "--head_trajectory_dynamics_overlay_only",
        type=_str2bool,
        default=False,
        help="If true, reuse saved cross-attention maps and center cache to render overlays only, without recomputing metrics or redrawing CSV-based plots.",
    )
    parser.add_argument(
        "--head_trajectory_dynamics_skip_existing_plots",
        type=_str2bool,
        default=True,
        help="If true, head_trajectory_dynamics plotting skips plot files that already exist.",
    )

    # Self-attention temporal-kernel intervention options
    parser.add_argument(
        "--self_attn_kernel_steps",
        type=str,
        default="1,2,3,4,5,6",
        help="CSV diffusion steps for self_attention_temporal_kernel. Empty means all steps.",
    )
    parser.add_argument(
        "--self_attn_kernel_layers",
        type=str,
        default="",
        help="CSV layer ids for self_attention_temporal_kernel. Empty means all layers.",
    )
    parser.add_argument(
        "--self_attn_kernel_heads",
        type=str,
        default="",
        help="CSV head specs `LxHy` for self_attention_temporal_kernel. Empty means all heads in selected layers.",
    )
    parser.add_argument(
        "--self_attn_kernel_branch",
        type=str,
        default="cond",
        choices=["cond", "uncond", "both"],
    )
    parser.add_argument(
        "--self_attn_temporal_intervention_mode",
        type=str,
        default="postoutput_same_position_kernel",
        choices=["postoutput_same_position_kernel", "prelogit_token_temperature"],
        help=(
            "Self-attention intervention mode. "
            "`postoutput_same_position_kernel` smooths selected head outputs over latent frames; "
            "`prelogit_token_temperature` directly flattens selected heads' token-level attention "
            "distribution with an exact softmax temperature."
        ),
    )
    parser.add_argument("--self_attn_kernel_radius", type=int, default=2)
    parser.add_argument("--self_attn_kernel_sigma", type=float, default=1.0)
    parser.add_argument("--self_attn_kernel_mix_alpha", type=float, default=0.25)
    parser.add_argument("--self_attn_token_temperature", type=float, default=1.0)

    # Self-attention distribution options
    parser.add_argument(
        "--self_attention_distribution_steps",
        type=str,
        default="1,2,3",
        help="CSV diffusion steps for self_attention_distribution. Empty means all steps [1..sampling_steps].",
    )
    parser.add_argument(
        "--self_attention_distribution_layers",
        type=str,
        default="",
        help="CSV layer ids for self_attention_distribution. Empty means all layers.",
    )
    parser.add_argument(
        "--self_attention_distribution_branch",
        type=str,
        default="cond",
        choices=["cond", "uncond", "both"],
    )
    parser.add_argument("--self_attention_distribution_reference_step", type=int, default=50)
    parser.add_argument("--self_attention_distribution_reference_layer", type=int, default=27)
    parser.add_argument(
        "--self_attention_distribution_reference_center_mode",
        type=str,
        default="geometric_center",
        choices=["peak", "centroid", "geometric_center"],
    )
    parser.add_argument("--self_attention_distribution_reference_center_power", type=float, default=1.5)
    parser.add_argument("--self_attention_distribution_reference_center_quantile", type=float, default=0.8)
    parser.add_argument("--self_attention_distribution_reference_preprocess_winsorize_quantile", type=float, default=0.995)
    parser.add_argument("--self_attention_distribution_reference_preprocess_despike_quantile", type=float, default=0.98)
    parser.add_argument("--self_attention_distribution_reference_preprocess_min_component_area", type=int, default=2)
    parser.add_argument(
        "--self_attention_distribution_support_radius_mode",
        type=str,
        default="adaptive_area",
        choices=["fixed", "adaptive_area"],
    )
    parser.add_argument("--self_attention_distribution_support_radius_fixed", type=float, default=2.0)
    parser.add_argument("--self_attention_distribution_support_radius_alpha", type=float, default=1.5)
    parser.add_argument("--self_attention_distribution_support_radius_min", type=float, default=1.0)
    parser.add_argument("--self_attention_distribution_support_radius_max_ratio", type=float, default=0.25)
    parser.add_argument("--self_attention_distribution_query_frame_count", type=int, default=8)
    parser.add_argument("--self_attention_distribution_global_query_tokens_per_frame", type=int, default=64)
    parser.add_argument("--self_attention_distribution_object_query_token_limit_per_frame", type=int, default=0)
    parser.add_argument(
        "--self_attention_distribution_plot_per_head",
        type=_str2bool,
        default=False,
        help="If true, export per-head self_attention_distribution plots in addition to layer-level head-mean plots.",
    )
    parser.add_argument(
        "--self_attention_distribution_stop_after_last_probe_step",
        type=_str2bool,
        default=False,
        help="If true, stop diffusion immediately after the last requested self_attention_distribution probe step.",
    )
    parser.add_argument(
        "--self_attention_distribution_plot_only_from_csv",
        type=_str2bool,
        default=False,
        help="If true, reuse existing self_attention_distribution CSV outputs in output_dir and only redraw plots.",
    )
    parser.add_argument(
        "--self_attention_distribution_skip_existing_plots",
        type=_str2bool,
        default=True,
        help="If true, self_attention_distribution plotting skips plot files that already exist.",
    )

    # Self-attention visualization options
    parser.add_argument(
        "--self_attention_viz_steps",
        type=str,
        default="1,2,3",
        help="CSV diffusion steps for self_attention_viz. Empty means all steps [1..sampling_steps].",
    )
    parser.add_argument(
        "--self_attention_viz_layers",
        type=str,
        default="",
        help="CSV layer ids for self_attention_viz. Empty means all layers.",
    )
    parser.add_argument(
        "--self_attention_viz_branch",
        type=str,
        default="cond",
        choices=["cond", "uncond", "both"],
    )
    parser.add_argument("--self_attention_viz_reference_step", type=int, default=50)
    parser.add_argument("--self_attention_viz_reference_layer", type=int, default=27)
    parser.add_argument(
        "--self_attention_viz_reference_center_mode",
        type=str,
        default="geometric_center",
        choices=["peak", "centroid", "geometric_center"],
    )
    parser.add_argument("--self_attention_viz_reference_center_power", type=float, default=1.5)
    parser.add_argument("--self_attention_viz_reference_center_quantile", type=float, default=0.8)
    parser.add_argument("--self_attention_viz_reference_preprocess_winsorize_quantile", type=float, default=0.995)
    parser.add_argument("--self_attention_viz_reference_preprocess_despike_quantile", type=float, default=0.98)
    parser.add_argument("--self_attention_viz_reference_preprocess_min_component_area", type=int, default=2)
    parser.add_argument(
        "--self_attention_viz_support_radius_mode",
        type=str,
        default="adaptive_area",
        choices=["fixed", "adaptive_area"],
    )
    parser.add_argument("--self_attention_viz_support_radius_fixed", type=float, default=2.0)
    parser.add_argument("--self_attention_viz_support_radius_alpha", type=float, default=1.5)
    parser.add_argument("--self_attention_viz_support_radius_min", type=float, default=1.0)
    parser.add_argument("--self_attention_viz_support_radius_max_ratio", type=float, default=0.25)
    parser.add_argument(
        "--self_attention_viz_query_video_frame_indices",
        type=str,
        default="1,33,41,81",
        help=(
            "CSV 1-based video-frame labels used as self_attention_viz query frames. "
            "They are projected to token-frame indices internally."
        ),
    )
    parser.add_argument("--self_attention_viz_object_query_token_limit_per_frame", type=int, default=64)
    parser.add_argument("--self_attention_viz_num_viz_frames", type=int, default=10)
    parser.add_argument("--self_attention_viz_viz_frame_indices", type=str, default="")
    parser.add_argument("--self_attention_viz_save_attention_pdfs", type=_str2bool, default=True)
    parser.add_argument("--self_attention_viz_attention_pdf_share_color_scale", type=_str2bool, default=False)
    parser.add_argument("--self_attention_viz_skip_existing_pdfs", type=_str2bool, default=True)
    parser.add_argument("--self_attention_viz_stop_after_last_probe_step", type=_str2bool, default=False)
    parser.add_argument("--draw_self_attention_maps_only", type=_str2bool, default=False)
    parser.add_argument("--draw_self_attention_maps_path", type=str, default="")
    parser.add_argument("--self_attention_viz_visualization_output_dir", type=str, default="")
    parser.add_argument(
        "--self_attention_viz_num_workers",
        type=int,
        default=0,
        help=(
            "Number of CPU worker processes used by self_attention_viz to render PDFs. "
            "A non-positive value means use os.cpu_count()."
        ),
    )
    parser.add_argument(
        "--self_attention_modulation_steps",
        type=str,
        default="1,2,3",
        help="CSV diffusion steps for self_attention_modulation. Empty means all steps [1..sampling_steps].",
    )
    parser.add_argument(
        "--self_attention_modulation_layers",
        type=str,
        default="",
        help="CSV layer ids for self_attention_modulation. Empty means all layers.",
    )
    parser.add_argument(
        "--self_attention_modulation_branch",
        type=str,
        default="cond",
        choices=["cond", "uncond", "both"],
    )
    parser.add_argument(
        "--self_attention_modulation_stop_after_last_probe_step",
        type=_str2bool,
        default=False,
        help="If true, stop diffusion immediately after the last requested self_attention_modulation probe step.",
    )
    parser.add_argument(
        "--self_attention_modulation_channel_profile_targets",
        type=str,
        default="",
        help=(
            "CSV of step:layer pairs for extra channel-profile bar plots in self_attention_modulation, "
            'for example "1:0,1:28,25:0". Empty means draw bar plots for all collected step-layer pairs.'
        ),
    )
    parser.add_argument(
        "--self_attention_modulation_channel_profile_topk",
        type=int,
        default=12,
        help="Top-k channel indices annotated on each channel-profile bar plot.",
    )
    parser.add_argument(
        "--self_attention_modulation_plot_only_from_saved",
        type=_str2bool,
        default=False,
        help=(
            "If true, skip sampling/model execution and redraw self_attention_modulation figures "
            "only from saved CSV/PT artifacts under output_dir."
        ),
    )

    # Seed-to-trajectory predictability options
    parser.add_argument(
        "--seed_to_trajectory_early_steps",
        type=str,
        default="1,2,3,4,5,6",
        help="CSV early diffusion steps used as trajectory-prediction features.",
    )
    parser.add_argument("--seed_to_trajectory_reference_step", type=int, default=50)
    parser.add_argument("--seed_to_trajectory_reference_layer", type=int, default=27)
    parser.add_argument("--seed_to_trajectory_head", type=str, default="mean")
    parser.add_argument(
        "--seed_to_trajectory_num_points",
        type=int,
        default=0,
        help=(
            "Trajectory resampling point count. "
            "Set <= 0 to keep the original latent-frame trajectory length."
        ),
    )
    parser.add_argument("--seed_to_trajectory_ridge_alpha", type=float, default=1e-3)

    # Event-token value options
    parser.add_argument(
        "--event_token_value_words",
        type=str,
        default="",
        help="CSV event/action words whose cross-attention value contributions are collected.",
    )
    parser.add_argument("--event_token_value_steps", type=str, default="1,2,3,4,5,6")
    parser.add_argument("--event_token_value_layers", type=str, default="")
    parser.add_argument(
        "--event_token_value_branch",
        type=str,
        default="cond",
        choices=["cond", "uncond", "both"],
    )
    parser.add_argument("--event_token_value_chunk_size", type=int, default=512)
    parser.add_argument("--event_token_value_num_viz_frames", type=int, default=10)

    # Seed stability / joint suite options
    parser.add_argument("--seed_list", type=str, default="0,1,2,3")
    parser.add_argument("--stability_num_points", type=int, default=41)
    parser.add_argument("--stability_head", type=str, default="mean")
    parser.add_argument("--object_traj_step", type=int, default=None)
    parser.add_argument("--object_traj_layer", type=int, default=None)
    parser.add_argument("--object_traj_head", type=str, default="mean")
    parser.add_argument(
        "--reuse_cross_attention_dir",
        type=str,
        default="",
        help=(
            "Optional existing cross_attention_token_viz directory. "
            "Used by joint_attention_suite (skip stage-1) and trajectory_entropy (reuse cross-attention maps)."
        ),
    )
    parser.add_argument(
        "--reuse_head_trajectory_dynamics_dir",
        type=str,
        default="",
        help="Optional existing head_trajectory_dynamics directory used for early-alignment scatter plots.",
    )
    parser.add_argument(
        "--trajectory_consensus_stages",
        type=str,
        default="candidate_consensus,head_contribution",
        help=(
            "CSV stages for trajectory_consensus_dynamics. Supported: "
            "candidate_consensus,head_contribution,self_attention_coupling,seed_influence."
        ),
    )
    parser.add_argument(
        "--trajectory_consensus_steps",
        type=str,
        default="1,2,3",
        help="CSV diffusion steps for trajectory_consensus_dynamics. Empty means all reused-map steps.",
    )
    parser.add_argument(
        "--trajectory_consensus_layers",
        type=str,
        default="",
        help="CSV layer ids for trajectory_consensus_dynamics. Empty means all reused-map layers.",
    )
    parser.add_argument(
        "--trajectory_consensus_cross_heads",
        type=str,
        default="",
        help=(
            "CSV cross-attention head specs `LxHy` for trajectory_consensus_dynamics. "
            "Used by candidate visualization and by head_contribution when module `cross` is enabled. "
            "Use empty string to analyze all heads in the selected layers, or `None` to analyze no cross-attention head."
        ),
    )
    parser.add_argument(
        "--trajectory_consensus_self_heads",
        type=str,
        default="",
        help=(
            "CSV self-attention head specs `LxHy` for trajectory_consensus_dynamics "
            "head_contribution when module `self` is enabled. "
            "Use empty string to analyze all heads in the selected layers, or `None` to analyze no self-attention head."
        ),
    )
    parser.add_argument(
        "--trajectory_consensus_modules",
        type=str,
        default="cross,self",
        help="CSV module names for trajectory_consensus_dynamics head_contribution. Supported: cross,self.",
    )
    parser.add_argument(
        "--trajectory_consensus_branch",
        type=str,
        default="cond",
        choices=["cond", "uncond"],
        help="CFG branch analyzed by trajectory_consensus_dynamics head_contribution and self_attention_coupling.",
    )
    parser.add_argument(
        "--trajectory_consensus_reference_distance_metric",
        type=str,
        default="center_l2",
        help="CSV reference-distance metric names reused from head_trajectory_dynamics for early-alignment scatter plots.",
    )
    parser.add_argument(
        "--trajectory_consensus_scatter_outlier_heads",
        type=str,
        default="",
        help="Legacy CSV head tags like `L0H4` removed from filtered scatter plots for both cross and self modules.",
    )
    parser.add_argument(
        "--trajectory_consensus_scatter_outlier_cross_heads",
        type=str,
        default="",
        help="CSV cross-attention head tags like `L0H4` removed only from filtered cross-attention scatter plots.",
    )
    parser.add_argument(
        "--trajectory_consensus_scatter_outlier_self_heads",
        type=str,
        default="",
        help="CSV self-attention head tags like `L0H4` removed only from filtered self-attention scatter plots.",
    )
    parser.add_argument("--trajectory_consensus_candidate_base_quantile", type=float, default=0.85)
    parser.add_argument(
        "--trajectory_consensus_candidate_split_quantiles",
        type=str,
        default="0.92,0.95,0.97",
        help="CSV seed-support quantiles used to propose local maxima in trajectory_consensus_dynamics.",
    )
    parser.add_argument("--trajectory_consensus_candidate_smooth_radius", type=int, default=1)
    parser.add_argument("--trajectory_consensus_candidate_stable_peak_min_levels", type=int, default=2)
    parser.add_argument("--trajectory_consensus_candidate_peak_merge_distance", type=float, default=2.0)
    parser.add_argument("--trajectory_consensus_candidate_preprocess_winsorize_quantile", type=float, default=0.995)
    parser.add_argument("--trajectory_consensus_candidate_preprocess_despike_quantile", type=float, default=0.98)
    parser.add_argument("--trajectory_consensus_candidate_min_component_area", type=int, default=4)
    parser.add_argument("--trajectory_consensus_candidate_viz_num_frames", type=int, default=8)
    parser.add_argument("--trajectory_consensus_candidate_enable_per_head", type=_str2bool, default=True)
    parser.add_argument(
        "--trajectory_consensus_contribution_method",
        type=str,
        default="exact_ablation",
        choices=["exact_ablation", "taylor_approx"],
        help=(
            "Head-contribution method for trajectory_consensus_dynamics. "
            "`exact_ablation` reruns one ablated forward per head. "
            "`taylor_approx` uses attribution patching with a first-order Taylor approximation."
        ),
    )
    parser.add_argument(
        "--trajectory_consensus_ablate_position",
        type=str,
        default="pre_o",
        choices=["pre_o", "post_o"],
        help=(
            "Activation position used by trajectory_consensus_dynamics head contribution. "
            "`pre_o` patches the per-head attention output z before the output projection. "
            "`post_o` patches the per-head residual write after the output projection."
        ),
    )
    parser.add_argument(
        "--trajectory_consensus_do_ablation",
        type=_str2bool,
        default=True,
        help="When True, trajectory_consensus_dynamics runs the ablation branch (exact_ablation or taylor_approx).",
    )
    parser.add_argument(
        "--trajectory_consensus_do_direct_proxy",
        type=_str2bool,
        default=True,
        help="When True, trajectory_consensus_dynamics runs the direct-proxy branch.",
    )
    parser.add_argument(
        "--trajectory_consensus_compute_direct_projection",
        type=_str2bool,
        default=None,
        help=(
            "Deprecated alias for --trajectory_consensus_do_direct_proxy. "
            "Kept for backward compatibility."
        ),
    )
    parser.add_argument("--trajectory_consensus_object_mask_reference_step", type=int, default=50)
    parser.add_argument("--trajectory_consensus_object_mask_reference_layer", type=int, default=27)
    parser.add_argument(
        "--trajectory_consensus_taylor_object_only",
        type=_str2bool,
        default=True,
        help="When True, trajectory_consensus_dynamics Taylor attribution uses only object-region outputs and object-token activations.",
    )
    parser.add_argument(
        "--trajectory_consensus_taylor_num_latent_frames",
        type=int,
        default=10,
        help="Number of evenly sampled latent frames used by trajectory_consensus_dynamics Taylor attribution. Use -1 to keep all latent frames.",
    )
    parser.add_argument(
        "--trajectory_consensus_taylor_metric_scope",
        type=str,
        default="obj",
        choices=["obj", "global"],
        help="Scalar target used by trajectory_consensus_dynamics Taylor attribution: object-region velocity sum or global velocity sum.",
    )
    parser.add_argument(
        "--trajectory_consensus_taylor_use_gradient_checkpointing",
        type=_str2bool,
        default=True,
        help="When True, trajectory_consensus_dynamics Taylor attribution uses gradient checkpointing for downstream Wan blocks.",
    )
    parser.add_argument("--trajectory_consensus_sa_anchor_step", type=int, default=49)
    parser.add_argument("--trajectory_consensus_sa_anchor_layer", type=int, default=27)
    parser.add_argument(
        "--trajectory_consensus_sa_covered_mass_min",
        type=float,
        default=0.0,
        help="Optional candidate-covered-mass threshold used by filtered self_attention_coupling aggregates.",
    )
    parser.add_argument(
        "--trajectory_consensus_sa_precedence_persistence",
        type=int,
        default=2,
        help="Persistence window used by trajectory_consensus_dynamics self_attention_coupling temporal precedence.",
    )
    parser.add_argument("--trajectory_consensus_plot_only_from_csv", type=_str2bool, default=False)
    parser.add_argument("--trajectory_consensus_skip_existing_plots", type=_str2bool, default=True)
    parser.add_argument(
        "--trajectory_consensus_seed_influence_mode",
        type=str,
        default="seed_sensitivity",
        choices=["seed_sensitivity", "anchor_frame"],
        help="Mode used when trajectory_consensus_dynamics runs the seed_influence stage.",
    )
    parser.add_argument(
        "--trajectory_consensus_seed_influence_seeds",
        type=str,
        default="",
        help="CSV seed list used by trajectory_consensus_dynamics seed_influence seed_sensitivity mode.",
    )
    parser.add_argument(
        "--trajectory_consensus_seed_sensitivity_zr_metric",
        type=str,
        default="global_mutual_consistency",
        help=(
            "Candidate-level scalar used to define z_r in trajectory_consensus_dynamics "
            "seed_influence seed_sensitivity mode."
        ),
    )
    parser.add_argument(
        "--trajectory_consensus_seed_sensitivity_steps",
        type=str,
        default="",
        help="CSV step list used only by trajectory_consensus_dynamics seed_influence seed_sensitivity mode.",
    )
    parser.add_argument(
        "--trajectory_consensus_anchor_frame_steps",
        type=str,
        default="",
        help="CSV step list used only by trajectory_consensus_dynamics seed_influence anchor_frame mode.",
    )
    parser.add_argument(
        "--trajectory_consensus_seed_influence_anchor_topk",
        type=int,
        default=2,
        help="Number of anchor frames highlighted in seed_influence anchor_frame scatter plots.",
    )
    parser.add_argument(
        "--trajectory_consensus_seed_influence_arrow_topk",
        type=int,
        default=2,
        help="Top-k routing targets drawn per source frame in seed_influence anchor_frame arrow plots.",
    )
    parser.add_argument(
        "--trajectory_consensus_num_workers",
        type=int,
        default=0,
        help=(
            "Number of CPU worker processes used by trajectory_consensus_dynamics for "
            "candidate-region extraction and candidate visualization rendering. "
            "A non-positive value means use os.cpu_count()."
        ),
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    from wan21_t2v_experiments.wan21_t2v_experiments import (
        Wan21T2VParallelConfig,
        run_wan21_t2v_attention_dt_profile,
        run_wan21_t2v_causal_schedule,
        run_wan21_t2v_cross_attention_token_viz,
        run_wan21_t2v_event_token_value,
        run_wan21_t2v_head_evolution,
        run_wan21_t2v_head_trajectory_dynamics,
        run_wan21_t2v_joint_attention_suite,
        run_wan21_t2v_motion_aligned_attention,
        run_wan21_t2v_rope_modification,
        run_wan21_t2v_rope_axis_ablation,
        run_wan21_t2v_rope_decay_curve,
        run_wan21_t2v_cross_attn_head_ablation,
        run_wan21_t2v_seed_to_trajectory_predictability,
        run_wan21_t2v_self_attention_distribution,
        run_wan21_t2v_self_attention_viz,
        run_wan21_t2v_self_attention_modulation,
        run_wan21_t2v_self_attention_temporal_kernel,
        run_wan21_t2v_step_window_cross_attn_off,
        run_wan21_t2v_step_window_ffn_off,
        run_wan21_t2v_step_window_prompt_replace,
        run_wan21_t2v_trajectory_consensus_dynamics,
        run_wan21_t2v_trajectory_entropy,
        run_wan21_t2v_token_trajectory_seed_stability,
    )

    experiment_name = args.experiment
    output_dir = args.output_dir if args.output_dir else _default_output_dir(experiment_name)

    size = _parse_size(args.size)
    probe_steps = _parse_csv_ints(args.probe_steps)
    cross_attn_steps = _parse_csv_ints(args.cross_attn_steps)
    head_ablation_steps = _parse_csv_ints(args.head_ablation_steps)
    rope_modification_steps = _parse_csv_ints(args.rope_modification_steps)
    rope_modification_semantic_residual_steps = _parse_csv_ints(args.rope_modification_semantic_residual_steps)
    trajectory_entropy_steps = _parse_csv_ints(args.trajectory_entropy_steps)
    trajectory_entropy_layerwise_steps = _parse_csv_ints(args.trajectory_entropy_layerwise_steps)
    head_evolution_steps = _parse_csv_ints(args.head_evolution_steps)
    head_evolution_layerwise_steps = _parse_csv_ints(args.head_evolution_layerwise_steps)
    head_trajectory_dynamics_heads = _parse_csv_strs(args.head_trajectory_dynamics_heads)
    head_trajectory_dynamics_steps = _parse_csv_ints(args.head_trajectory_dynamics_steps)
    head_trajectory_dynamics_distance_metrics = _parse_csv_strs(args.head_trajectory_dynamics_distance_metrics)
    head_trajectory_dynamics_attractor_distance_metrics = _parse_csv_strs(args.head_trajectory_dynamics_attractor_distance_metric)
    self_attn_kernel_steps = _parse_csv_ints(args.self_attn_kernel_steps)
    self_attn_kernel_layers = _parse_csv_ints(args.self_attn_kernel_layers)
    self_attn_kernel_heads = _parse_csv_strs(args.self_attn_kernel_heads)
    self_attention_distribution_steps = _parse_csv_ints(args.self_attention_distribution_steps)
    self_attention_distribution_layers = _parse_csv_ints(args.self_attention_distribution_layers)
    self_attention_viz_steps = _parse_csv_ints(args.self_attention_viz_steps)
    self_attention_viz_layers = _parse_csv_ints(args.self_attention_viz_layers)
    self_attention_viz_query_video_frame_indices = _parse_csv_ints(args.self_attention_viz_query_video_frame_indices)
    self_attention_modulation_steps = _parse_csv_ints(args.self_attention_modulation_steps)
    self_attention_modulation_layers = _parse_csv_ints(args.self_attention_modulation_layers)
    self_attention_modulation_channel_profile_targets = _parse_wan21_t2v_channel_profile_target_pairs(
        args.self_attention_modulation_channel_profile_targets
    )
    seed_to_trajectory_early_steps = _parse_csv_ints(args.seed_to_trajectory_early_steps)
    event_token_value_words = _parse_csv_strs(args.event_token_value_words)
    event_token_value_steps = _parse_csv_ints(args.event_token_value_steps)
    event_token_value_layers = _parse_csv_ints(args.event_token_value_layers)
    ablate_heads = _parse_csv_strs(args.ablate_heads)
    target_object_words = _parse_csv_strs(args.target_object_words)
    target_verb_words = _parse_csv_strs(args.target_verb_words)
    seed_list = _parse_csv_ints(args.seed_list)
    trajectory_consensus_stages = _parse_csv_strs(args.trajectory_consensus_stages)
    trajectory_consensus_steps = _parse_csv_ints(args.trajectory_consensus_steps)
    trajectory_consensus_layers = _parse_csv_ints(args.trajectory_consensus_layers)
    trajectory_consensus_cross_heads = _parse_optional_csv_strs_with_none(args.trajectory_consensus_cross_heads)
    trajectory_consensus_self_heads = _parse_optional_csv_strs_with_none(args.trajectory_consensus_self_heads)
    trajectory_consensus_modules = _parse_csv_strs(args.trajectory_consensus_modules)
    trajectory_consensus_reference_distance_metrics = _parse_csv_strs(args.trajectory_consensus_reference_distance_metric)
    trajectory_consensus_scatter_outlier_heads = _parse_csv_strs(args.trajectory_consensus_scatter_outlier_heads)
    trajectory_consensus_scatter_outlier_cross_heads = _parse_csv_strs(args.trajectory_consensus_scatter_outlier_cross_heads)
    trajectory_consensus_scatter_outlier_self_heads = _parse_csv_strs(args.trajectory_consensus_scatter_outlier_self_heads)
    trajectory_consensus_seed_influence_seeds = _parse_csv_ints(args.trajectory_consensus_seed_influence_seeds)
    trajectory_consensus_seed_sensitivity_steps = _parse_csv_ints(args.trajectory_consensus_seed_sensitivity_steps)
    trajectory_consensus_anchor_frame_steps = _parse_csv_ints(args.trajectory_consensus_anchor_frame_steps)
    viz_layers = _parse_csv_ints(args.viz_layers)
    viz_frame_indices = _parse_csv_ints(args.viz_frame_indices)
    self_attention_viz_viz_frame_indices = _parse_csv_ints(args.self_attention_viz_viz_frame_indices)

    parallel_cfg = Wan21T2VParallelConfig(
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        t5_cpu=args.t5_cpu,
        use_usp=args.use_usp,
        ulysses_size=args.ulysses_size,
        ring_size=args.ring_size,
    )

    common_kwargs = dict(
        wan21_root=args.wan21_root,
        ckpt_dir=args.ckpt_dir,
        output_dir=output_dir,
        prompt=args.prompt,
        size=size,
        task=args.task,
        frame_num=args.frame_num,
        shift=args.shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sampling_steps,
        guide_scale=args.guide_scale,
        seed=args.seed,
        device_id=args.device_id,
        offload_model=args.offload_model,
        parallel_cfg=parallel_cfg,
    )

    if experiment_name == "rope_axis_ablation":
        run_wan21_t2v_rope_axis_ablation(
            **common_kwargs,
            rope_modes=_parse_csv_strs(args.rope_modes),
        )
    elif experiment_name == "attention_dt_profile":
        run_wan21_t2v_attention_dt_profile(
            **common_kwargs,
            probe_steps=probe_steps,
            query_frame_count=args.query_frame_count,
            query_mode=args.probe_query_mode,
            probe_branch=args.probe_branch,
        )
    elif experiment_name == "motion_aligned_attention":
        run_wan21_t2v_motion_aligned_attention(
            **common_kwargs,
            probe_steps=probe_steps,
            query_frame_count=args.query_frame_count,
            query_mode=args.probe_query_mode,
            probe_branch=args.probe_branch,
            maas_layers=_parse_csv_ints(args.maas_layers),
            maas_radius=args.maas_radius,
            motion_target_source=args.motion_target_source,
        )
    elif experiment_name == "trajectory_entropy":
        if not target_object_words:
            raise ValueError("--target_object_words is required for trajectory_entropy.")
        run_wan21_t2v_trajectory_entropy(
            **common_kwargs,
            target_object_words=target_object_words,
            target_verb_words=target_verb_words,
            trajectory_entropy_steps=trajectory_entropy_steps,
            trajectory_entropy_layerwise_steps=trajectory_entropy_layerwise_steps,
            trajectory_entropy_head_layer=args.trajectory_entropy_head_layer,
            trajectory_entropy_stepwise_layer=args.trajectory_entropy_stepwise_layer,
            save_video=args.trajectory_entropy_save_video,
            reuse_cross_attention_dir=args.reuse_cross_attention_dir.strip() or None,
        )
    elif experiment_name == "head_evolution":
        if not target_object_words:
            raise ValueError("--target_object_words is required for head_evolution.")
        if not args.reuse_cross_attention_dir.strip():
            raise ValueError("--reuse_cross_attention_dir is required for head_evolution.")
        run_wan21_t2v_head_evolution(
            **common_kwargs,
            target_object_words=target_object_words,
            target_verb_words=target_verb_words,
            head_evolution_steps=head_evolution_steps,
            head_evolution_layerwise_steps=head_evolution_layerwise_steps,
            head_evolution_head_layer=args.head_evolution_head_layer,
            head_evolution_stepwise_layer=args.head_evolution_stepwise_layer,
            head_evolution_reference_step=args.head_evolution_reference_step,
            head_evolution_reference_layer=args.head_evolution_reference_layer,
            head_evolution_center_mode=args.head_evolution_center_mode,
            head_evolution_support_radius_mode=args.head_evolution_support_radius_mode,
            head_evolution_support_radius_fixed=args.head_evolution_support_radius_fixed,
            head_evolution_support_radius_alpha=args.head_evolution_support_radius_alpha,
            head_evolution_support_radius_min=args.head_evolution_support_radius_min,
            head_evolution_support_radius_max_ratio=args.head_evolution_support_radius_max_ratio,
            head_evolution_traj_power=args.head_evolution_traj_power,
            head_evolution_traj_quantile=args.head_evolution_traj_quantile,
            head_evolution_reference_viz_num_frames=args.head_evolution_reference_viz_num_frames,
            head_evolution_save_reference_radius_overlay=args.head_evolution_save_reference_radius_overlay,
            head_evolution_early_step_end=args.head_evolution_early_step_end,
            head_evolution_score_quantile=args.head_evolution_score_quantile,
            head_evolution_apply_preprocess_on_metrics=args.head_evolution_apply_preprocess_on_metrics,
            head_evolution_preprocess_winsorize_quantile=args.head_evolution_preprocess_winsorize_quantile,
            head_evolution_preprocess_despike_quantile=args.head_evolution_preprocess_despike_quantile,
            head_evolution_preprocess_min_component_area=args.head_evolution_preprocess_min_component_area,
            head_evolution_concentrated_region_top_ratio=args.head_evolution_concentrated_region_top_ratio,
            reuse_cross_attention_dir=args.reuse_cross_attention_dir.strip() or None,
        )
    elif experiment_name == "head_trajectory_dynamics":
        if (not args.head_trajectory_dynamics_plot_only_from_csv) and (not target_object_words):
            raise ValueError("--target_object_words is required for head_trajectory_dynamics.")
        if (not args.head_trajectory_dynamics_plot_only_from_csv) and (not args.reuse_cross_attention_dir.strip()):
            raise ValueError("--reuse_cross_attention_dir is required for head_trajectory_dynamics.")
        run_wan21_t2v_head_trajectory_dynamics(
            **common_kwargs,
            target_object_words=target_object_words,
            target_verb_words=target_verb_words,
            head_trajectory_dynamics_heads=head_trajectory_dynamics_heads,
            head_trajectory_dynamics_steps=head_trajectory_dynamics_steps,
            head_trajectory_dynamics_distance_metrics=head_trajectory_dynamics_distance_metrics,
            head_trajectory_dynamics_reference_step=args.head_trajectory_dynamics_reference_step,
            head_trajectory_dynamics_reference_layer=args.head_trajectory_dynamics_reference_layer,
            head_trajectory_dynamics_support_quantile=args.head_trajectory_dynamics_support_quantile,
            head_trajectory_dynamics_attractor_window=args.head_trajectory_dynamics_attractor_window,
            head_trajectory_dynamics_attractor_distance_metrics=head_trajectory_dynamics_attractor_distance_metrics,
            head_trajectory_dynamics_center_method=args.head_trajectory_dynamics_center_method,
            head_trajectory_dynamics_center_power=args.head_trajectory_dynamics_center_power,
            head_trajectory_dynamics_center_quantile=args.head_trajectory_dynamics_center_quantile,
            head_trajectory_dynamics_preprocessed_center_mode=args.head_trajectory_dynamics_preprocessed_center_mode,
            head_trajectory_dynamics_preprocess_winsorize_quantile=args.head_trajectory_dynamics_preprocess_winsorize_quantile,
            head_trajectory_dynamics_preprocess_despike_quantile=args.head_trajectory_dynamics_preprocess_despike_quantile,
            head_trajectory_dynamics_preprocess_min_component_area=args.head_trajectory_dynamics_preprocess_min_component_area,
            head_trajectory_dynamics_reference_center_method=args.head_trajectory_dynamics_reference_center_method,
            head_trajectory_dynamics_reference_center_power=args.head_trajectory_dynamics_reference_center_power,
            head_trajectory_dynamics_reference_center_quantile=args.head_trajectory_dynamics_reference_center_quantile,
            head_trajectory_dynamics_reference_preprocessed_center_mode=args.head_trajectory_dynamics_reference_preprocessed_center_mode,
            head_trajectory_dynamics_reference_preprocess_winsorize_quantile=args.head_trajectory_dynamics_reference_preprocess_winsorize_quantile,
            head_trajectory_dynamics_reference_preprocess_despike_quantile=args.head_trajectory_dynamics_reference_preprocess_despike_quantile,
            head_trajectory_dynamics_reference_preprocess_min_component_area=args.head_trajectory_dynamics_reference_preprocess_min_component_area,
            head_trajectory_dynamics_center_viz_enable=args.head_trajectory_dynamics_center_viz_enable,
            head_trajectory_dynamics_center_viz_step=args.head_trajectory_dynamics_center_viz_step,
            head_trajectory_dynamics_center_viz_layer=args.head_trajectory_dynamics_center_viz_layer,
            head_trajectory_dynamics_center_viz_heads=_parse_csv_strs(args.head_trajectory_dynamics_center_viz_heads),
            head_trajectory_dynamics_center_viz_num_frames=args.head_trajectory_dynamics_center_viz_num_frames,
            head_trajectory_dynamics_support_viz_enable=args.head_trajectory_dynamics_support_viz_enable,
            head_trajectory_dynamics_support_viz_step=args.head_trajectory_dynamics_support_viz_step,
            head_trajectory_dynamics_support_viz_layer=args.head_trajectory_dynamics_support_viz_layer,
            head_trajectory_dynamics_support_viz_heads=_parse_csv_strs(args.head_trajectory_dynamics_support_viz_heads),
            head_trajectory_dynamics_support_viz_num_frames=args.head_trajectory_dynamics_support_viz_num_frames,
            head_trajectory_dynamics_support_viz_contour_min_component_area=(
                args.head_trajectory_dynamics_support_viz_contour_min_component_area
            ),
            head_trajectory_dynamics_support_cache_num_workers=(
                args.head_trajectory_dynamics_support_cache_num_workers
            ),
            head_trajectory_dynamics_center_cache_num_workers=(
                args.head_trajectory_dynamics_center_cache_num_workers
            ),
            head_trajectory_dynamics_overlay_num_workers=(
                args.head_trajectory_dynamics_overlay_num_workers
            ),
            head_trajectory_dynamics_cache_save_interval=(
                args.head_trajectory_dynamics_cache_save_interval
            ),
            head_trajectory_dynamics_hypothesis=args.head_trajectory_dynamics_hypothesis,
            head_trajectory_dynamics_traj_type=args.head_trajectory_dynamics_traj_type,
            head_trajectory_dynamics_use_motion_planning_region_before_metrics=(
                args.head_trajectory_dynamics_use_motion_planning_region_before_metrics
            ),
            head_trajectory_dynamics_plot_only_from_csv=args.head_trajectory_dynamics_plot_only_from_csv,
            head_trajectory_dynamics_overlay_only=args.head_trajectory_dynamics_overlay_only,
            head_trajectory_dynamics_skip_existing_plots=args.head_trajectory_dynamics_skip_existing_plots,
            reuse_cross_attention_dir=args.reuse_cross_attention_dir.strip() or None,
        )
    elif experiment_name == "rope_decay_curve":
        run_wan21_t2v_rope_decay_curve(
            **common_kwargs,
        )
    elif experiment_name == "rope_modification":
        run_wan21_t2v_rope_modification(
            **common_kwargs,
            rope_modification_mode=args.rope_modification_mode,
            rope_modification_lambda_f=args.rope_modification_lambda_f,
            rope_modification_lambda_h=args.rope_modification_lambda_h,
            rope_modification_lambda_w=args.rope_modification_lambda_w,
            rope_modification_steps=rope_modification_steps,
            rope_modification_spatial_temporal_reweight_alpha=(
                args.rope_modification_spatial_temporal_reweight_alpha
            ),
            rope_modification_timestep_conditioned_resolution=args.rope_modification_timestep_conditioned_resolution,
            rope_modification_timestep_conditioned_hidden_dim=args.rope_modification_timestep_conditioned_hidden_dim,
            rope_modification_timestep_conditioned_checkpoint=args.rope_modification_timestep_conditioned_checkpoint,
            rope_modification_semantic_residual_enabled=args.rope_modification_semantic_residual_enabled,
            rope_modification_semantic_residual_alpha=args.rope_modification_semantic_residual_alpha,
            rope_modification_semantic_residual_steps=rope_modification_semantic_residual_steps,
            rope_modification_semantic_residual_query_chunk_size=args.rope_modification_semantic_residual_query_chunk_size,
            rope_modification_semantic_residual_timestep_conditioned=args.rope_modification_semantic_residual_timestep_conditioned,
            rope_modification_semantic_residual_timestep_conditioned_resolution=args.rope_modification_semantic_residual_timestep_conditioned_resolution,
            rope_modification_semantic_residual_timestep_conditioned_hidden_dim=args.rope_modification_semantic_residual_timestep_conditioned_hidden_dim,
            rope_modification_semantic_residual_timestep_conditioned_checkpoint=args.rope_modification_semantic_residual_timestep_conditioned_checkpoint,
        )
    elif experiment_name == "self_attention_temporal_kernel":
        run_wan21_t2v_self_attention_temporal_kernel(
            **common_kwargs,
            self_attn_kernel_steps=self_attn_kernel_steps,
            self_attn_kernel_layers=self_attn_kernel_layers,
            self_attn_kernel_heads=self_attn_kernel_heads,
            self_attn_kernel_branch=args.self_attn_kernel_branch,
            self_attn_temporal_intervention_mode=args.self_attn_temporal_intervention_mode,
            self_attn_kernel_radius=args.self_attn_kernel_radius,
            self_attn_kernel_sigma=args.self_attn_kernel_sigma,
            self_attn_kernel_mix_alpha=args.self_attn_kernel_mix_alpha,
            self_attn_token_temperature=args.self_attn_token_temperature,
        )
    elif experiment_name == "self_attention_distribution":
        if (not args.self_attention_distribution_plot_only_from_csv) and (not target_object_words):
            raise ValueError("--target_object_words is required for self_attention_distribution.")
        if (not args.self_attention_distribution_plot_only_from_csv) and (not args.reuse_cross_attention_dir.strip()):
            raise ValueError("--reuse_cross_attention_dir is required for self_attention_distribution.")
        run_wan21_t2v_self_attention_distribution(
            **common_kwargs,
            target_object_words=target_object_words,
            target_verb_words=target_verb_words,
            reuse_cross_attention_dir=args.reuse_cross_attention_dir.strip(),
            self_attention_distribution_steps=self_attention_distribution_steps,
            self_attention_distribution_layers=self_attention_distribution_layers,
            self_attention_distribution_branch=args.self_attention_distribution_branch,
            self_attention_distribution_reference_step=args.self_attention_distribution_reference_step,
            self_attention_distribution_reference_layer=args.self_attention_distribution_reference_layer,
            self_attention_distribution_reference_center_mode=args.self_attention_distribution_reference_center_mode,
            self_attention_distribution_reference_center_power=args.self_attention_distribution_reference_center_power,
            self_attention_distribution_reference_center_quantile=args.self_attention_distribution_reference_center_quantile,
            self_attention_distribution_reference_preprocess_winsorize_quantile=args.self_attention_distribution_reference_preprocess_winsorize_quantile,
            self_attention_distribution_reference_preprocess_despike_quantile=args.self_attention_distribution_reference_preprocess_despike_quantile,
            self_attention_distribution_reference_preprocess_min_component_area=args.self_attention_distribution_reference_preprocess_min_component_area,
            self_attention_distribution_support_radius_mode=args.self_attention_distribution_support_radius_mode,
            self_attention_distribution_support_radius_fixed=args.self_attention_distribution_support_radius_fixed,
            self_attention_distribution_support_radius_alpha=args.self_attention_distribution_support_radius_alpha,
            self_attention_distribution_support_radius_min=args.self_attention_distribution_support_radius_min,
            self_attention_distribution_support_radius_max_ratio=args.self_attention_distribution_support_radius_max_ratio,
            self_attention_distribution_query_frame_count=args.self_attention_distribution_query_frame_count,
            self_attention_distribution_global_query_tokens_per_frame=args.self_attention_distribution_global_query_tokens_per_frame,
            self_attention_distribution_object_query_token_limit_per_frame=args.self_attention_distribution_object_query_token_limit_per_frame,
            self_attention_distribution_plot_per_head=args.self_attention_distribution_plot_per_head,
            self_attention_distribution_stop_after_last_probe_step=args.self_attention_distribution_stop_after_last_probe_step,
            self_attention_distribution_plot_only_from_csv=args.self_attention_distribution_plot_only_from_csv,
            self_attention_distribution_skip_existing_plots=args.self_attention_distribution_skip_existing_plots,
            save_video=args.save_video,
        )
    elif experiment_name == "self_attention_viz":
        if not target_object_words:
            raise ValueError("--target_object_words is required for self_attention_viz.")
        if not args.reuse_cross_attention_dir.strip():
            raise ValueError("--reuse_cross_attention_dir is required for self_attention_viz.")
        run_wan21_t2v_self_attention_viz(
            **common_kwargs,
            target_object_words=target_object_words,
            target_verb_words=target_verb_words,
            reuse_cross_attention_dir=args.reuse_cross_attention_dir.strip(),
            self_attention_viz_steps=self_attention_viz_steps,
            self_attention_viz_layers=self_attention_viz_layers,
            self_attention_viz_branch=args.self_attention_viz_branch,
            self_attention_viz_reference_step=args.self_attention_viz_reference_step,
            self_attention_viz_reference_layer=args.self_attention_viz_reference_layer,
            self_attention_viz_reference_center_mode=args.self_attention_viz_reference_center_mode,
            self_attention_viz_reference_center_power=args.self_attention_viz_reference_center_power,
            self_attention_viz_reference_center_quantile=args.self_attention_viz_reference_center_quantile,
            self_attention_viz_reference_preprocess_winsorize_quantile=(
                args.self_attention_viz_reference_preprocess_winsorize_quantile
            ),
            self_attention_viz_reference_preprocess_despike_quantile=(
                args.self_attention_viz_reference_preprocess_despike_quantile
            ),
            self_attention_viz_reference_preprocess_min_component_area=(
                args.self_attention_viz_reference_preprocess_min_component_area
            ),
            self_attention_viz_support_radius_mode=args.self_attention_viz_support_radius_mode,
            self_attention_viz_support_radius_fixed=args.self_attention_viz_support_radius_fixed,
            self_attention_viz_support_radius_alpha=args.self_attention_viz_support_radius_alpha,
            self_attention_viz_support_radius_min=args.self_attention_viz_support_radius_min,
            self_attention_viz_support_radius_max_ratio=args.self_attention_viz_support_radius_max_ratio,
            self_attention_viz_query_video_frame_indices=(
                self_attention_viz_query_video_frame_indices if self_attention_viz_query_video_frame_indices else [1, 33, 41, 81]
            ),
            self_attention_viz_object_query_token_limit_per_frame=(
                args.self_attention_viz_object_query_token_limit_per_frame
            ),
            self_attention_viz_num_viz_frames=args.self_attention_viz_num_viz_frames,
            self_attention_viz_viz_frame_indices=(
                self_attention_viz_viz_frame_indices if self_attention_viz_viz_frame_indices else None
            ),
            self_attention_viz_save_attention_pdfs=args.self_attention_viz_save_attention_pdfs,
            self_attention_viz_attention_pdf_share_color_scale=(
                args.self_attention_viz_attention_pdf_share_color_scale
            ),
            self_attention_viz_skip_existing_pdfs=args.self_attention_viz_skip_existing_pdfs,
            self_attention_viz_stop_after_last_probe_step=args.self_attention_viz_stop_after_last_probe_step,
            draw_self_attention_maps_only=args.draw_self_attention_maps_only,
            draw_self_attention_maps_path=args.draw_self_attention_maps_path,
            self_attention_viz_visualization_output_dir=args.self_attention_viz_visualization_output_dir,
            self_attention_viz_num_workers=args.self_attention_viz_num_workers,
            save_video=args.save_video,
        )
    elif experiment_name == "self_attention_modulation":
        run_wan21_t2v_self_attention_modulation(
            **common_kwargs,
            self_attention_modulation_steps=self_attention_modulation_steps,
            self_attention_modulation_layers=self_attention_modulation_layers,
            self_attention_modulation_branch=args.self_attention_modulation_branch,
            self_attention_modulation_stop_after_last_probe_step=args.self_attention_modulation_stop_after_last_probe_step,
            self_attention_modulation_channel_profile_targets=self_attention_modulation_channel_profile_targets,
            self_attention_modulation_channel_profile_topk=args.self_attention_modulation_channel_profile_topk,
            self_attention_modulation_plot_only_from_saved=args.self_attention_modulation_plot_only_from_saved,
            save_video=args.save_video,
        )
    elif experiment_name == "seed_to_trajectory_predictability":
        if not target_object_words:
            raise ValueError("--target_object_words is required for seed_to_trajectory_predictability.")
        run_wan21_t2v_seed_to_trajectory_predictability(
            **common_kwargs,
            target_object_words=target_object_words,
            target_verb_words=target_verb_words,
            seed_list=seed_list,
            seed_to_trajectory_early_steps=seed_to_trajectory_early_steps,
            seed_to_trajectory_reference_step=args.seed_to_trajectory_reference_step,
            seed_to_trajectory_reference_layer=args.seed_to_trajectory_reference_layer,
            seed_to_trajectory_head=args.seed_to_trajectory_head,
            seed_to_trajectory_num_points=args.seed_to_trajectory_num_points,
            seed_to_trajectory_ridge_alpha=args.seed_to_trajectory_ridge_alpha,
            num_viz_frames=args.viz_num_frames,
            layers_to_collect=viz_layers if viz_layers else None,
            chunk_size=args.cross_attn_chunk_size,
            trajectory_style=args.traj_style,
            trajectory_smooth_radius=args.traj_smooth_radius,
            trajectory_power=args.traj_power,
            trajectory_quantile=args.traj_quantile,
            trajectory_arrow_stride=args.traj_arrow_stride,
        )
    elif experiment_name == "event_token_value":
        if not (target_object_words or target_verb_words or event_token_value_words):
            raise ValueError(
                "--target_object_words, --target_verb_words, or --event_token_value_words "
                "is required for event_token_value."
            )
        run_wan21_t2v_event_token_value(
            **common_kwargs,
            target_object_words=target_object_words,
            target_verb_words=target_verb_words,
            event_token_value_words=event_token_value_words,
            event_token_value_steps=event_token_value_steps,
            event_token_value_layers=event_token_value_layers,
            event_token_value_branch=args.event_token_value_branch,
            event_token_value_chunk_size=args.event_token_value_chunk_size,
            event_token_value_num_viz_frames=args.event_token_value_num_viz_frames,
        )
    elif experiment_name == "causal_schedule":
        run_wan21_t2v_causal_schedule(
            **common_kwargs,
            causal_first_n_steps=args.causal_first_n_steps,
            causal_mode=args.causal_mode,
            attention_backend=args.attention_backend,
            run_baseline=args.run_baseline,
            collect_dt_profile=args.collect_dt_profile,
            probe_steps=probe_steps,
            query_frame_count=args.query_frame_count,
        )
    elif experiment_name == "step_window_cross_attn_off":
        condition_remove_start_steps = _parse_csv_ints(args.condition_remove_start_steps)
        if not condition_remove_start_steps:
            raise ValueError("--condition_remove_start_steps must be non-empty for step_window_cross_attn_off.")
        run_wan21_t2v_step_window_cross_attn_off(
            **common_kwargs,
            condition_remove_start_steps=condition_remove_start_steps,
            condition_remove_scope=args.condition_remove_scope,
            reuse_removed_cond_for_uncond=args.reuse_removed_cond_for_uncond,
        )
    elif experiment_name == "step_window_ffn_off":
        timestep_idx_to_remove_ffn = _parse_csv_ints(args.timestep_idx_to_remove_ffn)
        if not timestep_idx_to_remove_ffn:
            raise ValueError("--timestep_idx_to_remove_ffn must be non-empty for step_window_ffn_off.")
        layer_idx_to_remove_ffn = _parse_csv_ints(args.layer_idx_to_remove_ffn)
        run_wan21_t2v_step_window_ffn_off(
            **common_kwargs,
            timestep_idx_to_remove_ffn=timestep_idx_to_remove_ffn,
            layer_idx_to_remove_ffn=layer_idx_to_remove_ffn if layer_idx_to_remove_ffn else None,
            ffn_remove_scope=args.ffn_remove_scope,
        )
    elif experiment_name == "cross_attn_head_ablation":
        if not ablate_heads:
            raise ValueError("--ablate_heads must be non-empty for cross_attn_head_ablation.")
        run_wan21_t2v_cross_attn_head_ablation(
            **common_kwargs,
            ablate_heads=ablate_heads,
            head_ablation_steps=head_ablation_steps,
        )
    elif experiment_name == "step_window_prompt_replace":
        prompt_replace_steps = _parse_csv_ints(args.prompt_replace_steps)
        if not prompt_replace_steps:
            raise ValueError("--prompt_replace_steps must be non-empty for step_window_prompt_replace.")
        if not args.replacement_prompt.strip():
            raise ValueError("--replacement_prompt is required for step_window_prompt_replace.")
        run_wan21_t2v_step_window_prompt_replace(
            **common_kwargs,
            replacement_prompt=args.replacement_prompt,
            prompt_replace_steps=prompt_replace_steps,
            prompt_replace_scope=args.prompt_replace_scope,
            replace_cond_only=args.replace_cond_only,
        )
    elif experiment_name == "cross_attention_token_viz":
        if (not args.draw_attention_map_only) and (not target_object_words and not target_verb_words):
            raise ValueError(
                "--target_object_words or --target_verb_words is required for cross_attention_token_viz."
            )
        run_wan21_t2v_cross_attention_token_viz(
            **common_kwargs,
            target_object_words=target_object_words,
            target_verb_words=target_verb_words,
            collect_steps=cross_attn_steps,
            num_viz_frames=args.viz_num_frames,
            viz_frame_indices=viz_frame_indices if viz_frame_indices else None,
            layers_to_collect=viz_layers if viz_layers else None,
            chunk_size=args.cross_attn_chunk_size,
            trajectory_enable=args.traj_enable,
            trajectory_style=args.traj_style,
            trajectory_num_frames=args.traj_num_frames,
            trajectory_smooth_radius=args.traj_smooth_radius,
            trajectory_power=args.traj_power,
            trajectory_quantile=args.traj_quantile,
            trajectory_arrow_stride=args.traj_arrow_stride,
            trajectory_include_head_mean=args.traj_include_head_mean,
            save_attention_pdfs=args.save_attention_pdfs,
            attention_pdf_per_frame_normalize=args.attention_pdf_per_frame_normalize,
            attention_pdf_share_color_scale=args.attention_pdf_share_color_scale,
            skip_existing_pdfs=args.skip_existing_pdfs,
            save_trajectory_pdfs=args.save_trajectory_pdfs,
            save_trajectory_timeline_pdfs=args.save_trajectory_timeline_pdfs,
            trajectory_timeline_num_frames=args.trajectory_timeline_num_frames,
            save_video=args.save_video,
            stream_flush_per_step=args.stream_flush_per_step,
            plot_during_sampling=args.plot_during_sampling,
            draw_attention_map_only=args.draw_attention_map_only,
            draw_attention_maps_path=args.draw_attention_maps_path,
            visualization_output_dir=args.visualization_output_dir,
            cross_attention_token_viz_num_workers=args.cross_attention_token_viz_num_workers,
        )
    elif experiment_name == "token_trajectory_seed_stability":
        if not target_object_words:
            raise ValueError("--target_object_words is required for token_trajectory_seed_stability.")
        run_wan21_t2v_token_trajectory_seed_stability(
            **common_kwargs,
            target_object_words=target_object_words,
            target_verb_words=target_verb_words,
            seed_list=seed_list,
            collect_steps=cross_attn_steps,
            num_viz_frames=args.viz_num_frames,
            layers_to_collect=viz_layers if viz_layers else None,
            chunk_size=args.cross_attn_chunk_size,
            stability_num_points=args.stability_num_points,
            stability_head=args.stability_head,
            trajectory_style=args.traj_style,
            trajectory_smooth_radius=args.traj_smooth_radius,
            trajectory_power=args.traj_power,
            trajectory_quantile=args.traj_quantile,
            trajectory_arrow_stride=args.traj_arrow_stride,
        )
    elif experiment_name == "joint_attention_suite":
        if not target_object_words:
            raise ValueError("--target_object_words is required for joint_attention_suite.")
        run_wan21_t2v_joint_attention_suite(
            **common_kwargs,
            target_object_words=target_object_words,
            target_verb_words=target_verb_words,
            collect_steps=probe_steps,
            layers_to_collect=viz_layers if viz_layers else None,
            query_frame_count=args.query_frame_count,
            maas_layers=_parse_csv_ints(args.maas_layers),
            maas_radius=args.maas_radius,
            probe_branch=args.probe_branch,
            motion_target_source=args.motion_target_source,
            object_trajectory_step=args.object_traj_step,
            object_trajectory_layer=args.object_traj_layer,
            object_trajectory_head=args.object_traj_head,
            reuse_cross_attention_dir=args.reuse_cross_attention_dir.strip() or None,
        )
    elif experiment_name == "trajectory_consensus_dynamics":
        if (not args.trajectory_consensus_plot_only_from_csv) and (not target_object_words):
            raise ValueError("--target_object_words is required for trajectory_consensus_dynamics.")
        if (not args.trajectory_consensus_plot_only_from_csv) and (not args.reuse_cross_attention_dir.strip()):
            raise ValueError("--reuse_cross_attention_dir is required for trajectory_consensus_dynamics.")
        run_wan21_t2v_trajectory_consensus_dynamics(
            **common_kwargs,
            target_object_words=target_object_words,
            target_verb_words=target_verb_words,
            reuse_cross_attention_dir=args.reuse_cross_attention_dir.strip() or None,
            reuse_head_trajectory_dynamics_dir=args.reuse_head_trajectory_dynamics_dir.strip() or None,
            trajectory_consensus_stages=trajectory_consensus_stages,
            trajectory_consensus_steps=trajectory_consensus_steps,
            trajectory_consensus_layers=trajectory_consensus_layers,
            trajectory_consensus_cross_heads=trajectory_consensus_cross_heads,
            trajectory_consensus_self_heads=trajectory_consensus_self_heads,
            trajectory_consensus_modules=trajectory_consensus_modules,
            trajectory_consensus_branch=args.trajectory_consensus_branch,
            trajectory_consensus_reference_distance_metrics=trajectory_consensus_reference_distance_metrics,
            trajectory_consensus_scatter_outlier_heads=trajectory_consensus_scatter_outlier_heads,
            trajectory_consensus_scatter_outlier_cross_heads=trajectory_consensus_scatter_outlier_cross_heads,
            trajectory_consensus_scatter_outlier_self_heads=trajectory_consensus_scatter_outlier_self_heads,
            trajectory_consensus_candidate_base_quantile=args.trajectory_consensus_candidate_base_quantile,
            trajectory_consensus_candidate_split_quantiles=tuple(
                float(x) for x in _parse_csv_strs(args.trajectory_consensus_candidate_split_quantiles)
            ),
            trajectory_consensus_candidate_smooth_radius=args.trajectory_consensus_candidate_smooth_radius,
            trajectory_consensus_candidate_stable_peak_min_levels=args.trajectory_consensus_candidate_stable_peak_min_levels,
            trajectory_consensus_candidate_peak_merge_distance=args.trajectory_consensus_candidate_peak_merge_distance,
            trajectory_consensus_candidate_preprocess_winsorize_quantile=args.trajectory_consensus_candidate_preprocess_winsorize_quantile,
            trajectory_consensus_candidate_preprocess_despike_quantile=args.trajectory_consensus_candidate_preprocess_despike_quantile,
            trajectory_consensus_candidate_min_component_area=args.trajectory_consensus_candidate_min_component_area,
            trajectory_consensus_candidate_viz_num_frames=args.trajectory_consensus_candidate_viz_num_frames,
            trajectory_consensus_candidate_enable_per_head=args.trajectory_consensus_candidate_enable_per_head,
            trajectory_consensus_do_ablation=args.trajectory_consensus_do_ablation,
            trajectory_consensus_contribution_method=args.trajectory_consensus_contribution_method,
            trajectory_consensus_ablate_position=args.trajectory_consensus_ablate_position,
            trajectory_consensus_do_direct_proxy=args.trajectory_consensus_do_direct_proxy,
            trajectory_consensus_compute_direct_projection=args.trajectory_consensus_compute_direct_projection,
            trajectory_consensus_object_mask_reference_step=args.trajectory_consensus_object_mask_reference_step,
            trajectory_consensus_object_mask_reference_layer=args.trajectory_consensus_object_mask_reference_layer,
            trajectory_consensus_taylor_object_only=args.trajectory_consensus_taylor_object_only,
            trajectory_consensus_taylor_num_latent_frames=args.trajectory_consensus_taylor_num_latent_frames,
            trajectory_consensus_taylor_metric_scope=args.trajectory_consensus_taylor_metric_scope,
            trajectory_consensus_taylor_use_gradient_checkpointing=args.trajectory_consensus_taylor_use_gradient_checkpointing,
            trajectory_consensus_sa_anchor_step=args.trajectory_consensus_sa_anchor_step,
            trajectory_consensus_sa_anchor_layer=args.trajectory_consensus_sa_anchor_layer,
            trajectory_consensus_sa_covered_mass_min=args.trajectory_consensus_sa_covered_mass_min,
            trajectory_consensus_sa_precedence_persistence=args.trajectory_consensus_sa_precedence_persistence,
            trajectory_consensus_plot_only_from_csv=args.trajectory_consensus_plot_only_from_csv,
            trajectory_consensus_seed_influence_mode=args.trajectory_consensus_seed_influence_mode,
            trajectory_consensus_seed_influence_seeds=trajectory_consensus_seed_influence_seeds,
            trajectory_consensus_seed_sensitivity_zr_metric=args.trajectory_consensus_seed_sensitivity_zr_metric,
            trajectory_consensus_seed_sensitivity_steps=trajectory_consensus_seed_sensitivity_steps,
            trajectory_consensus_anchor_frame_steps=trajectory_consensus_anchor_frame_steps,
            trajectory_consensus_seed_influence_anchor_topk=args.trajectory_consensus_seed_influence_anchor_topk,
            trajectory_consensus_seed_influence_arrow_topk=args.trajectory_consensus_seed_influence_arrow_topk,
            trajectory_consensus_skip_existing_plots=args.trajectory_consensus_skip_existing_plots,
            trajectory_consensus_num_workers=args.trajectory_consensus_num_workers,
        )

    print(f"[wan21_t2v_experiments] done. outputs: {output_dir}")


if __name__ == "__main__":
    main()
