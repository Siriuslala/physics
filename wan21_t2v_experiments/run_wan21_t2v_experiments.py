"""CLI entry for Wan2.1-T2V monkey-patch experiments."""

import argparse
import os
import sys
from datetime import datetime
from typing import List, Tuple

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
            "motion_aligned_attention",
            "causal_schedule",
            "step_window_cross_attn_off",
            "step_window_ffn_off",
            "cross_attn_head_ablation",
            "step_window_prompt_replace",
            "cross_attention_token_viz",
            "token_trajectory_seed_stability",
            "joint_attention_suite",
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

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    from wan21_t2v_experiments.wan21_t2v_experiments import (
        Wan21T2VParallelConfig,
        run_wan21_t2v_attention_dt_profile,
        run_wan21_t2v_causal_schedule,
        run_wan21_t2v_cross_attention_token_viz,
        run_wan21_t2v_joint_attention_suite,
        run_wan21_t2v_motion_aligned_attention,
        run_wan21_t2v_rope_axis_ablation,
        run_wan21_t2v_cross_attn_head_ablation,
        run_wan21_t2v_step_window_cross_attn_off,
        run_wan21_t2v_step_window_ffn_off,
        run_wan21_t2v_step_window_prompt_replace,
        run_wan21_t2v_trajectory_entropy,
        run_wan21_t2v_token_trajectory_seed_stability,
    )

    experiment_name = args.experiment
    output_dir = args.output_dir if args.output_dir else _default_output_dir(experiment_name)

    size = _parse_size(args.size)
    probe_steps = _parse_csv_ints(args.probe_steps)
    cross_attn_steps = _parse_csv_ints(args.cross_attn_steps)
    head_ablation_steps = _parse_csv_ints(args.head_ablation_steps)
    trajectory_entropy_steps = _parse_csv_ints(args.trajectory_entropy_steps)
    trajectory_entropy_layerwise_steps = _parse_csv_ints(args.trajectory_entropy_layerwise_steps)
    ablate_heads = _parse_csv_strs(args.ablate_heads)
    target_object_words = _parse_csv_strs(args.target_object_words)
    target_verb_words = _parse_csv_strs(args.target_verb_words)
    seed_list = _parse_csv_ints(args.seed_list)
    viz_layers = _parse_csv_ints(args.viz_layers)
    viz_frame_indices = _parse_csv_ints(args.viz_frame_indices)

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

    print(f"[wan21_t2v_experiments] done. outputs: {output_dir}")


if __name__ == "__main__":
    main()
