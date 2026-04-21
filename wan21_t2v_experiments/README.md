# Wan2.1-T2V Experiments (Monkey Patch)

This folder contains standalone experiment scripts for Wan2.1 text-to-video analysis.
All modifications are runtime monkey patches and do not edit `projects/Wan2_1` source files.

本目录提供 Wan2.1-T2V 的可解释性实验脚本。  
所有改动都通过运行时 monkey patch 完成，不会改动 `projects/Wan2_1` 原始代码。

## Implemented Experiments

- `rope_axis_ablation`
- `attention_dt_profile`
- `trajectory_entropy`
- `head_evolution`
- `head_trajectory_dynamics`
- `self_attention_temporal_kernel`
- `seed_to_trajectory_predictability`
- `event_token_value`
- `motion_aligned_attention`
- `causal_schedule`
- `cross_attention_token_viz`
- `token_trajectory_seed_stability`
- `joint_attention_suite`
- `step_window_cross_attn_off`
- `step_window_ffn_off`
- `cross_attn_head_ablation`
- `step_window_prompt_replace`

## Experiment Overview (EN + 中文)

### 1) `rope_axis_ablation`
- Motivation: test which RoPE axes are necessary for motion and appearance formation.
- 动机：验证 3D RoPE 的时间/空间轴对视频运动与外观生成的贡献。
- Input: prompt + rope modes (`full/no_f/no_h/no_w/only_f/only_hw`).
- Output: one video per mode + summary CSV/JSON.
- Key readout: qualitative motion collapse patterns across modes.

### 2) `attention_dt_profile`
- Motivation: profile temporal distance preference \(P(|\Delta t|)\) in early denoising steps.
- 动机：分析扩散早期 self-attention 的时间偏移分布 \(P(|\Delta t|)\)。
- Input: probe steps, query sampling mode (`center/multi_anchor/object_guided`), probe branch (`uncond/cond/both`).
- Output: generated video + `attention_dt_histograms.pt` + summary + dt visualization PDFs.
- Key readout: near-frame mass vs far-frame mass.

### 3) `motion_aligned_attention`
- Motivation: measure whether next-frame attention mass lands near motion target positions (MAAS).
- 动机：测量 next-frame 注意力是否对齐到目标运动位置（MAAS）。
- Input: MAAS layers/radius, query mode/branch, target source:
  - `motion_centroid` (from generated video)
  - `object_token_trajectory` (from cross-attention trajectory)
- Output: MAAS rows CSV + dt histogram + dt visualization PDFs + motion-alignment visualizations.
- Key readout:
  - `maas_local_ratio` (higher means better spatial alignment)
  - layer/time curves and next-frame map overlays.

### 4) `causal_schedule`
- Motivation: compare bidirectional vs causal masks in early steps.
- 动机：比较早期双向注意力和因果约束注意力对结果的影响。
- Input: causal mode (`flat/temporal`), first N steps, backend (`auto/flash/torch_sdpa`).
- Output: videos per variant + optional dt profiles.

### 5) `cross_attention_token_viz`
- Motivation: visualize video-token to text-token cross-attention per timestep/layer/head.
- 动机：逐 timestep/layer/head 可视化视频序列对文本 token 的交叉注意力。
- Input:
  - `target_object_words`: object words used for map + trajectory
  - `target_verb_words`: verb words used for map only
  - `viz_num_frames`: uniform frame count for map panels
  - `viz_frame_indices`: optional explicit frame index list; overrides uniform sampling
  - `skip_existing_pdfs`: if `true`, skip already-generated PDFs (attention map + trajectory + timeline)
  - `stream_flush_per_step`: flush one completed diffusion step to disk and free memory
  - `plot_during_sampling`: if `true`, render PDFs during sampling; if `false`, defer plotting
  - `draw_attention_map_only`: redraw mode from saved map tensors without rerunning sampling
  - `visualization_output_dir`: optional directory to store newly rendered PDFs
- Output:
  - per-token attention PDFs
  - object-token trajectory PDFs/CSV
  - object-token trajectory timeline PDFs
  - map tensor `cross_attention_maps.pt`.
- Key readout:
  - object trajectory shape
  - attention concentration and drift across layers/steps.

### 6) `token_trajectory_seed_stability`
- Motivation: quantify same-prompt variability across seeds using object-token trajectories.
- 动机：在同一 prompt 下统计不同 seed 的 object token 轨迹稳定性。
- Input: seed list + cross-attention settings.
- Output:
  - per-seed runs
  - aggregated stability CSV/JSON
  - trajectory overlay and DTW matrix PDFs.
- Key readout:
  - `pointwise_dispersion`
  - `path_length_cv`
  - `direction_consistency`
  - `pairwise_dtw_mean`.

### 7) `joint_attention_suite`
- Motivation: unified 3-stage workflow with separate modules and unified report.
- 动机：按三步联动执行，同时保持模块解耦并输出统一报告。
- Pipeline:
  1. run `cross_attention_token_viz` to get object trajectories
  2. run object-guided `attention_dt_profile`
  3. run `motion_aligned_attention` with object trajectory or motion centroid target
- Output:
  - each sub-experiment keeps its own report
  - unified `joint_attention_suite_summary.json`.

### 8) `step_window_cross_attn_off`
- Motivation: identify when condition injection is essential by removing cross-attention within selected diffusion windows.
- 动机：通过在指定扩散步窗口移除 cross-attention，定位 condition 注入的关键时段。
- Input:
  - `condition_remove_start_steps`: step list
  - `condition_remove_scope`: `single_step` or `from_step`
  - `reuse_removed_cond_for_uncond`: optional compute-saving reuse when condition is removed
- Output:
  - one video per step setting
  - `step_window_cross_attn_off_summary.json/.csv`.
- Key readout:
  - trajectory/semantic collapse threshold vs step
  - `cross_attn_removed_calls`, `uncond_reused_calls`.

### 9) `step_window_ffn_off`
- Motivation: measure FFN contribution to motion-plan commitment at different denoising steps/layers.
- 动机：分析 FFN 在不同扩散步/层对运动规划收敛的作用。
- Input:
  - `timestep_idx_to_remove_ffn`: step list
  - `layer_idx_to_remove_ffn`: optional layer list (empty means all layers)
  - `ffn_remove_scope`: `single_step` or `from_step`
- Output:
  - one video per step setting
  - `step_window_ffn_off_summary.json/.csv`.
- Key readout:
  - sensitivity of motion quality to FFN removal by step/layer.

### 10) `cross_attn_head_ablation`
- Motivation: test the causal role of specific cross-attention heads found from visualization.
- 动机：针对在可视化中观察到的关键 cross-attention heads 做定点消融，检验其因果作用。
- Input:
  - `ablate_heads`: head list, canonical format `LxHy` (e.g. `L29H7,L12H3`)
  - `head_ablation_steps`: optional step list (empty => all diffusion steps)
- Output:
  - one ablated video
  - `cross_attn_head_ablation_summary.json/.csv`
  - `cross_attn_head_ablation_heads.csv`.
- Key readout:
  - video-level semantic/motion changes after removing selected heads
  - `cross_attn_ablation_calls`, `ablated_head_instances`.

### 11) `step_window_prompt_replace`
- Motivation: test plan-locking by replacing the conditional prompt at chosen denoising steps.
- 动机：在指定扩散步替换条件文本，测试运动规划的不可逆锁定窗口。
- Input:
  - `prompt`: original prompt A
  - `replacement_prompt`: replacement prompt B
  - `prompt_replace_steps`: step list
  - `prompt_replace_scope`: `single_step` or `from_step`
  - `replace_cond_only`: whether to replace only cond branch
- Output:
  - one video per step setting
  - `step_window_prompt_replace_summary.json/.csv`.
- Key readout:
  - earliest step where replacing prompt no longer changes trajectory semantics.

### 12) `trajectory_entropy`
- Motivation: quantify cross-attention spatial concentration for motion-plan analysis.
- 动机：量化 cross-attention 的空间分布熵，分析轨迹规划何时从“多候选”收敛到“已锁定”。
- Three sub-analyses:
  - step-wise: entropy vs diffusion step (selected layer, mean over per-head entropy)
  - layer-wise: entropy vs layer at selected steps (mean over per-head entropy)
  - head-wise: entropy vs diffusion step for each head at one selected layer (or all layers)
- Input:
  - `target_object_words`: object words to build trajectory-entropy maps
  - `trajectory_entropy_steps`: step list (empty => all steps)
  - `trajectory_entropy_stepwise_layer`: layer index used for step-wise curves (`-1` => last layer)
  - `trajectory_entropy_layerwise_steps`: layer-wise step list (empty => same as `trajectory_entropy_steps`, i.e., all steps by default)
  - `trajectory_entropy_head_layer`: layer index for head-wise curves (`-1` => last layer, `-2` => all layers)
  - `reuse_cross_attention_dir`: optional existing `cross_attention_token_viz` output dir; if set, entropy is computed from saved maps without resampling
- Output:
  - `trajectory_entropy_stepwise.csv`
  - `trajectory_entropy_layerwise.csv`
  - `trajectory_entropy_headwise.csv`
  - `trajectory_entropy_stepwise_object_mean.pdf` (frame-wise normalization)
  - `trajectory_entropy_stepwise_object_mean_videowise.pdf` (video-wise normalization on `F*H*W`)
  - layer/head entropy curve PDFs
  - `trajectory_entropy_summary.json`.

### 13) `head_evolution`
- Motivation: analyze dynamic head patterns over denoising steps and identify likely planning heads vs state-driven/modulation-driven heads.
- 动机：分析 head pattern 随扩散步的动态演化，区分可能的规划头、状态驱动头和调制驱动头。
- Core input:
  - `reuse_cross_attention_dir` (required): existing `cross_attention_token_viz` output directory
  - `target_object_words`: object words used to build object-mean maps
  - `head_evolution_reference_step/layer`: reference map location for trajectory-support construction
  - `head_evolution_center_mode`: `peak` / `centroid` / `geometric_center` (`bbox_center` is removed)
  - `head_evolution_support_radius_mode`: `fixed` or `adaptive_area`
- Core outputs:
  - `head_evolution_stepwise.csv` / `head_evolution_layerwise.csv` / `head_evolution_headwise.csv`
  - `head_evolution_head_scores.csv` (planning/readout/modulation/state-coupling scores + category)
  - `head_evolution_reference_radius_overlay.pdf` (reference map with center+r overlay for radius sanity-check)
  - Plot metric keys now use clean `frame` / `video` naming:
    - `entropy_frame`, `entropy_video`
    - `support_quality_frame`, `support_quality_video`
  - `head_evolution_summary.json`.

- MSS (Modulation Sensitivity Score):
  - Full name: `Modulation Sensitivity Score`.
  - Motivation: distinguish heads that simply follow denoising state evolution from heads with stronger step-selective behavior.
  - Meaning: larger MSS indicates stronger head-specific residual dynamics relative to layer-level average dynamics across diffusion steps.

### 14) `head_trajectory_dynamics`
- Motivation: test whether heads that show object-token trajectories become mutually consistent over early denoising, and whether some heads behave like attractors that pull other heads' soft trajectories closer over subsequent steps.
- 动机：分析多个 cross-attention heads 的 object-token 轨迹是否在扩散早期形成共识，以及是否存在“吸引子头”使其他 head 的软轨迹逐步靠近。
- Type: offline analysis. It reuses saved maps from an existing `cross_attention_token_viz` directory and does not resample video.
- Input:
  - `reuse_cross_attention_dir` (required): existing `cross_attention_token_viz` output directory
  - `target_object_words`: object words to aggregate into object-token maps
  - `head_trajectory_dynamics_heads`: optional head list such as `L4H1,L7H8`; empty means all heads
  - `head_trajectory_dynamics_steps`: optional diffusion-step list; empty means all steps available in reused maps
  - `head_trajectory_dynamics_distance_metrics`: `js`, `wasserstein`, or empty for both
  - `head_trajectory_dynamics_reference_step/layer`: final reference map used for distance-to-final-soft-center
- Output:
  - `head_trajectory_dynamics_head_maps.csv`: map inventory after filtering
  - `head_trajectory_dynamics_pairwise.csv`: pairwise head trajectory distances within each step/layer
  - `head_trajectory_dynamics_consensus.csv`: consensus score per step/layer
  - `head_trajectory_dynamics_attractor.csv`: attractor score for each candidate leader head
  - `head_trajectory_dynamics_final_distance.csv`: distance from each head's soft trajectory to final reference soft trajectory
  - `head_trajectory_dynamics_soft_centers.csv`: per-frame soft centers for each step/layer/head
  - `head_trajectory_dynamics_plots/`: consensus heatmaps, consensus curves, attractor curves, and soft-center spaghetti plots.
- Metrics:
  - `js_distance`: Jensen-Shannon distance between per-frame spatial probability maps.
  - `wasserstein_distance`: approximate Wasserstein-like distance computed by soft-center displacement, not full 2D optimal transport.
  - `consensus = 1 / (1 + mean_pairwise_distance)`: larger means heads in the same layer/step are more similar.
  - `attractor_score_mean`: average decrease in follower-head distance to a leader head from one selected step to the next selected step; positive means followers move closer.

### 15) `self_attention_temporal_kernel`
- Motivation: intervene on self-attention's temporal coherence path and test whether smoother longer-range temporal mixing improves or disrupts motion planning.
- 动机：干预 self-attention 的时序一致性通路，观察把局部帧输出平滑分配到邻近/更远帧是否改善或破坏轨迹规划。
- Type: online monkey patch. It does not modify Wan source files.
- Important implementation detail:
  - The experiment does not materialize full `[heads, L, L]` attention matrices, because that is too expensive for long video sequences.
  - Instead, Wan self-attention is computed normally, then its output `y` is mixed along the latent-frame axis for the same spatial token position:
    `y' = (1 - alpha) * y + alpha * temporal_kernel(y)`.
  - The temporal kernel is normalized, so the intervention is a residual output smoothing operation rather than an unnormalized attention rewrite.
- Input:
  - `self_attn_kernel_steps`: 1-based diffusion steps; empty means all steps
  - `self_attn_kernel_layers`: DiT layer list; empty means all layers
  - `self_attn_kernel_branch`: `cond`, `uncond`, or `both`
  - `self_attn_kernel_radius`: frame radius in latent-token time
  - `self_attn_kernel_sigma`: Gaussian kernel width
  - `self_attn_kernel_mix_alpha`: mixing coefficient in `[0,1]`
- Output:
  - generated video
  - `self_attention_temporal_kernel_calls.csv`
  - `self_attention_temporal_kernel_summary.json`.
- Key readout:
  - qualitative video motion changes vs baseline
  - whether increasing temporal smoothing stabilizes trajectory, over-smooths motion, or introduces artifacts.

### 16) `seed_to_trajectory_predictability`
- Motivation: test whether the selected reference object trajectory is predictable from the initial latent noise and/or early cross-attention trajectories.
- 动机：检验指定的参考轨迹能否从初始 latent 噪声和/或扩散早期的 cross-attention 轨迹中预测出来。
- Type: multi-seed analysis. It runs one generation per seed, captures the initial Gaussian latent `z_T`, captures early/reference object-token cross-attention maps in the same run, and then fits lightweight probes.
- Four built-in variants:
  - `noise_only`: use compact features from initial latent noise `z_T`
  - `step1_attention`: use only step-1 object-token trajectory
  - `steps1_to_k_attention`: use early-step object-token trajectories, e.g. steps `1..6`
  - `noise_plus_attention`: concatenate `z_T` features and early attention features
- Core idea:
  - For each seed, sample one video and save `initial_noise.pt`.
  - In the same run, collect selected early steps and one reference step of object-token cross-attention maps.
  - Use leave-one-seed-out ridge regression to predict the reference trajectory from each feature variant.
- Input:
  - `seed_list`: seeds to analyze
  - `seed_to_trajectory_early_steps`: early steps used by the attention-based variants
  - `seed_to_trajectory_reference_step`: reference trajectory step, usually `50`
  - `seed_to_trajectory_reference_layer`: layer used for early and reference trajectories, usually `27`
  - `seed_to_trajectory_head`: `mean` or a head id string
  - `seed_to_trajectory_num_points`: trajectory resampling point count; set `<= 0` to keep the original latent-frame trajectory length
- Output:
  - `seed_000000/initial_noise.pt`: saved initial Gaussian latent
  - `seed_000000/seed_to_trajectory_cross_attention_maps.pt`: early/reference attention maps collected in the same run
  - `seed_to_trajectory_seed_runs.csv`
  - `seed_to_trajectory_features.csv`
  - `seed_to_trajectory_targets.csv`
  - `seed_to_trajectory_predictions.csv`: leave-one-seed-out error for all four variants
  - `seed_to_trajectory_plots/<variant>/`: target vs predicted reference trajectory PDFs
  - `seed_to_trajectory_predictability_summary.json`.
- Key readout:
  - compare `noise_only`, `step1_attention`, `steps1_to_k_attention`, `noise_plus_attention`
  - `mean_point_error`: mean token-grid distance between predicted and reference trajectory
  - `normalized_mean_point_error`: error normalized by average final-trajectory step displacement.

### 17) `event_token_value`
- Motivation: compare different prompt tokens not only by where their cross-attention maps look, but by how large their value-side contribution is to the cross-attention residual.
- 动机：不只看某个文本 token 被视频 token 注意到哪里，还看该 token 的 value 对 cross-attention 输出 residual 的贡献强度。
- Type: online monkey patch. Token matching reuses the same `_locate_wan21_t2v_prompt_words` helper as `cross_attention_token_viz`.
- What is measured:
  - `attention_mean`: mean attention probability to the selected word positions.
  - `pre_output_value_norm_mean`: norm of the selected token's pre-output per-head value contribution.
  - `projected_value_norm_mean`: norm after applying the corresponding slice of the cross-attention output projection `o.weight`; the projection bias is not assigned to individual tokens.
- Input:
  - `target_object_words`: object words
  - `target_verb_words`: verb/action words
  - `event_token_value_words`: additional event words to mark as `event`; they are located through the same tokenizer matching path
  - `event_token_value_steps`: collection steps
  - `event_token_value_layers`: collection layers; empty means all layers
  - `event_token_value_branch`: `cond`, `uncond`, or `both`
- Output:
  - generated video
  - `event_token_value_summary.csv`
  - `event_token_value_maps.pt`: value-contribution maps
  - `event_token_attention_maps.pt`: matching attention maps
  - `event_token_value_maps/`: head-mean projected-value-norm timeline PDFs
  - `event_token_value_summary.json`.
- Key readout:
  - Tokens may have similar attention maps but different projected value norms. Such a gap indicates that a token is visually co-located in attention but contributes differently to the actual residual stream.

## Important Notes

- Cross-attention and early-step probe statistics are computed directly from projected \(q/k\) tensors in patches.
- These analyses do not rely on flash-attention returning attention weights.
- For causal temporal masking experiments, `torch_sdpa` may still be required by mask expressiveness.

中文说明：
- 本仓库里 cross-attention 和早期 self-attention 统计都通过 patch 后的 \(q/k\) 显式计算，因此不依赖 flash-attention 输出权重。
- 但严格时域 mask 依然可能需要 `torch_sdpa` 后端来表达。

## Entry Script

```bash
python wan21_t2v_experiments/run_wan21_t2v_experiments.py \
  --experiment attention_dt_profile \
  --ckpt_dir /path/to/Wan2.1-T2V-14B \
  --prompt "A basketball falls to the ground and bounces up several times." \
  --size 832*480 \
  --frame_num 81 \
  --sampling_steps 50 \
  --shift 8.0 \
  --guide_scale 12.0 \
  --seed 14
```

Multi-GPU example:

```bash
torchrun --nproc_per_node=2 wan21_t2v_experiments/run_wan21_t2v_experiments.py \
  --experiment attention_dt_profile \
  --ckpt_dir /path/to/Wan2.1-T2V-14B \
  --prompt "A basketball falls to the ground and bounces up several times." \
  --size 832*480 \
  --frame_num 81 \
  --sampling_steps 50 \
  --shift 8.0 \
  --guide_scale 12.0 \
  --seed 14 \
  --dit_fsdp \
  --t5_cpu
```

## Typical Commands

### RoPE Axis Ablation

```bash
python wan21_t2v_experiments/run_wan21_t2v_experiments.py \
  --experiment rope_axis_ablation \
  --ckpt_dir /path/to/Wan2.1-T2V-14B \
  --prompt "A basketball falls to the ground and bounces up several times." \
  --rope_modes full,no_f,no_h,no_w,only_f,only_hw
```

### Attention DT Profile

```bash
python wan21_t2v_experiments/run_wan21_t2v_experiments.py \
  --experiment attention_dt_profile \
  --ckpt_dir /path/to/Wan2.1-T2V-14B \
  --prompt "A ping pong ball falls and bounces on the table." \
  --probe_steps 1,2,3 \
  --query_frame_count 8 \
  --probe_query_mode center \
  --probe_branch uncond
```

### Motion-Aligned Attention

```bash
python wan21_t2v_experiments/run_wan21_t2v_experiments.py \
  --experiment motion_aligned_attention \
  --ckpt_dir /path/to/Wan2.1-T2V-14B \
  --prompt "A basketball falls to the ground and bounces up several times." \
  --probe_steps 1,2,3 \
  --probe_query_mode center \
  --probe_branch cond \
  --maas_layers 0,10,20,30,39 \
  --maas_radius 1 \
  --motion_target_source motion_centroid
```

### Causal Schedule

```bash
python wan21_t2v_experiments/run_wan21_t2v_experiments.py \
  --experiment causal_schedule \
  --ckpt_dir /path/to/Wan2.1-T2V-14B \
  --prompt "A basketball falls to the ground and bounces up several times." \
  --causal_first_n_steps 3 \
  --causal_mode temporal \
  --attention_backend torch_sdpa \
  --run_baseline true \
  --collect_dt_profile true
```

### Step-Window Cross-Attn Off

```bash
python wan21_t2v_experiments/run_wan21_t2v_experiments.py \
  --experiment step_window_cross_attn_off \
  --ckpt_dir /path/to/Wan2.1-T2V-14B \
  --prompt "A basketball falls to the ground and bounces up several times." \
  --condition_remove_start_steps 1,2,3,5,8,13,21,34,51 \
  --condition_remove_scope from_step \
  --reuse_removed_cond_for_uncond false
```

### Step-Window FFN Off

```bash
python wan21_t2v_experiments/run_wan21_t2v_experiments.py \
  --experiment step_window_ffn_off \
  --ckpt_dir /path/to/Wan2.1-T2V-14B \
  --prompt "A basketball falls to the ground and bounces up several times." \
  --timestep_idx_to_remove_ffn 1,2,3,5,8,13,21,34,50 \
  --layer_idx_to_remove_ffn "" \
  --ffn_remove_scope single_step
```

### Cross-Attn Head Ablation

```bash
python wan21_t2v_experiments/run_wan21_t2v_experiments.py \
  --experiment cross_attn_head_ablation \
  --ckpt_dir /path/to/Wan2.1-T2V-14B \
  --prompt "A basketball falls to the ground and bounces up several times." \
  --ablate_heads L29H7,L29H11 \
  --head_ablation_steps ""   # empty => all steps

# optional: only ablate selected steps
python wan21_t2v_experiments/run_wan21_t2v_experiments.py \
  --experiment cross_attn_head_ablation \
  --ckpt_dir /path/to/Wan2.1-T2V-14B \
  --prompt "A basketball falls to the ground and bounces up several times." \
  --ablate_heads L29H7,L29H11 \
  --head_ablation_steps 1,2,3,5
```

### Step-Window Prompt Replace

```bash
python wan21_t2v_experiments/run_wan21_t2v_experiments.py \
  --experiment step_window_prompt_replace \
  --ckpt_dir /path/to/Wan2.1-T2V-14B \
  --prompt "A basketball falls to the ground and bounces up several times." \
  --replacement_prompt "A basketball rises upward into the air." \
  --prompt_replace_steps 1,2,3,5,8,13,21,34,50 \
  --prompt_replace_scope from_step \
  --replace_cond_only true
```

### Trajectory Entropy

```bash
python wan21_t2v_experiments/run_wan21_t2v_experiments.py \
  --experiment trajectory_entropy \
  --ckpt_dir /path/to/Wan2.1-T2V-14B \
  --prompt "A basketball falls to the ground and bounces up several times." \
  --target_object_words basketball \
  --target_verb_words falls,bounces \
  --trajectory_entropy_steps "" \
  --trajectory_entropy_stepwise_layer -1 \
  --trajectory_entropy_layerwise_steps "" \
  --trajectory_entropy_head_layer -1 \
  --trajectory_entropy_save_video true

# Optional: reuse an existing cross_attention_token_viz directory (no resampling)
python wan21_t2v_experiments/run_wan21_t2v_experiments.py \
  --experiment trajectory_entropy \
  --ckpt_dir /path/to/Wan2.1-T2V-14B \
  --prompt "A basketball falls to the ground and bounces up several times." \
  --target_object_words basketball \
  --target_verb_words falls,bounces \
  --reuse_cross_attention_dir /path/to/previous/cross_attention_token_viz
```

### Head Evolution

```bash
python wan21_t2v_experiments/run_wan21_t2v_experiments.py \
  --experiment head_evolution \
  --ckpt_dir /path/to/Wan2.1-T2V-14B \
  --prompt "A basketball falls to the ground and bounces up several times." \
  --target_object_words basketball \
  --target_verb_words falls,bounces \
  --head_evolution_steps "" \
  --head_evolution_layerwise_steps "" \
  --head_evolution_stepwise_layer 27 \
  --head_evolution_head_layer -2 \
  --head_evolution_reference_step 50 \
  --head_evolution_reference_layer 27 \
  --head_evolution_center_mode centroid \
  --head_evolution_support_radius_mode adaptive_area \
  --head_evolution_support_radius_alpha 1.5 \
  --head_evolution_save_reference_radius_overlay true \
  --reuse_cross_attention_dir /path/to/previous/cross_attention_token_viz
```

### Head Trajectory Dynamics

```bash
python wan21_t2v_experiments/run_wan21_t2v_experiments.py \
  --experiment head_trajectory_dynamics \
  --ckpt_dir /path/to/Wan2.1-T2V-1.3B \
  --prompt "A basketball falls to the ground and bounces up several times." \
  --target_object_words basketball \
  --target_verb_words falls,bounces \
  --head_trajectory_dynamics_heads "" \
  --head_trajectory_dynamics_steps 1,2,3,4,5,6 \
  --head_trajectory_dynamics_distance_metrics "" \
  --head_trajectory_dynamics_reference_step 50 \
  --head_trajectory_dynamics_reference_layer 27 \
  --reuse_cross_attention_dir /path/to/previous/cross_attention_token_viz
```

### Self-Attention Temporal Kernel

```bash
python wan21_t2v_experiments/run_wan21_t2v_experiments.py \
  --experiment self_attention_temporal_kernel \
  --ckpt_dir /path/to/Wan2.1-T2V-1.3B \
  --prompt "A basketball falls to the ground and bounces up several times." \
  --self_attn_kernel_steps 1,2,3,4,5,6 \
  --self_attn_kernel_layers "" \
  --self_attn_kernel_branch cond \
  --self_attn_kernel_radius 2 \
  --self_attn_kernel_sigma 1.0 \
  --self_attn_kernel_mix_alpha 0.25
```

### Seed-To-Trajectory Predictability

```bash
python wan21_t2v_experiments/run_wan21_t2v_experiments.py \
  --experiment seed_to_trajectory_predictability \
  --ckpt_dir /path/to/Wan2.1-T2V-1.3B \
  --prompt "A basketball falls to the ground and bounces up several times." \
  --target_object_words basketball \
  --target_verb_words falls,bounces \
  --seed_list 0,1,2,3,4,5,6,7 \
  --seed_to_trajectory_early_steps 1,2,3,4,5,6 \
  --seed_to_trajectory_reference_step 50 \
  --seed_to_trajectory_reference_layer 27 \
  --seed_to_trajectory_head mean \
  --seed_to_trajectory_num_points 0
```

### Event-Token Value

```bash
python wan21_t2v_experiments/run_wan21_t2v_experiments.py \
  --experiment event_token_value \
  --ckpt_dir /path/to/Wan2.1-T2V-1.3B \
  --prompt "A basketball falls to the ground and bounces up several times." \
  --target_object_words basketball \
  --target_verb_words falls,bounces \
  --event_token_value_words falls,bounces \
  --event_token_value_steps 1,2,3,4,5,6 \
  --event_token_value_layers 27 \
  --event_token_value_branch cond
```

### Cross-Attention Token Visualization

```bash
python wan21_t2v_experiments/run_wan21_t2v_experiments.py \
  --experiment cross_attention_token_viz \
  --ckpt_dir /path/to/Wan2.1-T2V-14B \
  --prompt "A basketball falls to the ground and bounces up several times." \
  --target_object_words basketball \
  --target_verb_words falls,bounces \
  --cross_attn_steps 1,2,3 \
  --viz_num_frames 5 \
  --viz_frame_indices "" \
  --traj_enable true \
  --traj_style glow_arrow \
  --traj_num_frames 21 \
  --traj_smooth_radius 2 \
  --traj_power 1.5 \
  --traj_quantile 0.8 \
  --skip_existing_pdfs true \
  --save_trajectory_timeline_pdfs true \
  --trajectory_timeline_num_frames 10 \
  --stream_flush_per_step true \
  --plot_during_sampling false

# Draw-only mode: reuse saved attention maps (.pt) and regenerate PDFs/trajectories
python wan21_t2v_experiments/run_wan21_t2v_experiments.py \
  --experiment cross_attention_token_viz \
  --output_dir /path/to/existing/cross_attention_token_viz \
  --draw_attention_map_only true \
  --draw_attention_maps_path /path/to/existing/cross_attention_token_viz/cross_attention_maps.pt \
  --visualization_output_dir /path/to/new/redraw_outputs \
  --save_attention_pdfs true \
  --skip_existing_pdfs true \
  --save_trajectory_pdfs true \
  --save_trajectory_timeline_pdfs true
```

### Token Trajectory Seed Stability

```bash
python wan21_t2v_experiments/run_wan21_t2v_experiments.py \
  --experiment token_trajectory_seed_stability \
  --ckpt_dir /path/to/Wan2.1-T2V-14B \
  --prompt "A basketball falls to the ground and bounces up several times." \
  --target_object_words basketball \
  --target_verb_words falls,bounces \
  --seed_list 0,1,2,3,4,5,6,7 \
  --cross_attn_steps 1,2,3 \
  --stability_head mean \
  --stability_num_points 41
```

### Joint Attention Suite

```bash
python wan21_t2v_experiments/run_wan21_t2v_experiments.py \
  --experiment joint_attention_suite \
  --ckpt_dir /path/to/Wan2.1-T2V-14B \
  --prompt "A basketball falls to the ground and bounces up several times." \
  --target_object_words basketball \
  --target_verb_words falls,bounces \
  --probe_steps 1,2,3 \
  --probe_branch cond \
  --probe_query_mode object_guided \
  --motion_target_source object_token_trajectory \
  --object_traj_head mean

# Optional: reuse an existing stage-1 run and skip cross_attention_token_viz
python wan21_t2v_experiments/run_wan21_t2v_experiments.py \
  --experiment joint_attention_suite \
  --ckpt_dir /path/to/Wan2.1-T2V-14B \
  --prompt "A basketball falls to the ground and bounces up several times." \
  --target_object_words basketball \
  --target_verb_words falls,bounces \
  --reuse_cross_attention_dir /path/to/previous/cross_attention_token_viz
```

## Output Structure

Default output root:

`outputs_wan_2_1/wan21_t2v_<experiment>_<timestamp>/`

Typical files:

- `*_summary.json`: experiment-level report
- `*.csv`: row-level metrics/indices
- `*.pt`: tensor artifacts
- `*.pdf`: attention maps / trajectory / stability / alignment plots
- `*.mp4`: generated videos (optional for some runs)

For `joint_attention_suite`:

- `cross_attention_token_viz/` with its own report
- `object_guided_attention_dt_profile/` with its own report
- `motion_aligned_attention/` with its own report
- `joint_attention_suite_summary.json` unified report
- if `--reuse_cross_attention_dir` is set, stage-1 is loaded from that external directory and only stage-2/3 are generated under current output dir

For `head_evolution`:

- `head_evolution_reference_trajectory.csv`: reference per-frame centers/radius metadata
- `head_evolution_reference_head_mean_map.pt`: reference map used to construct support mask
- `head_evolution_support_mask_fhw.pt`: support mask tensor `[F, H, W]`
- `head_evolution_reference_radius_overlay.pdf`: center+radius sanity-check figure on reference maps
- `head_evolution_stepwise.csv`: step-wise metrics (selected layer, head-mean map)
- `head_evolution_layerwise.csv`: layer-wise metrics (selected steps, head-mean map)
- `head_evolution_headwise.csv`: head-wise metrics (selected layers, per-head curves)
- `head_evolution_head_scores.csv`: per-head scores and category labels
- `head_evolution_summary.json`: full configuration, thresholds, and artifact paths

## How To Read Main Metrics

- `maas_local_ratio`:
  - higher means next-frame attention places more mass near target trajectory position.
- `pointwise_dispersion`:
  - lower means multi-seed trajectory is more concentrated.
- `path_length_cv`:
  - lower means motion magnitude is more seed-stable.
- `direction_consistency`:
  - higher means velocity direction is more consistent across seeds.
- `pairwise_dtw_mean`:
  - lower means trajectory shapes are more similar across seeds.

## Parameter Guide (重点参数解释 + 推荐值)

- `probe_steps`: early diffusion steps used for probing. Example: `1,2,3`.
- 中文：在第几个扩散步收集注意力统计。推荐先用 `1,2,3`。

- `query_frame_count`: how many query frames are sampled from video-token timeline.
- 中文：从视频帧维度上抽多少帧作为 query。推荐 `8`（81帧视频）；更大更细但更慢。

- `probe_branch`: CFG branch to probe.
- Values: `uncond` / `cond` / `both`.
- 中文：采集无条件分支、有条件分支或两者。推荐 `cond`（更贴近文本驱动主体），对照实验再看 `uncond`。

- `maas_layers`: layer indices used to compute MAAS.
- 中文：在哪些层计算 MAAS（next-frame 对齐指标）。推荐稀疏覆盖如 `0,10,20,30,39`；若想更细可全层。

- `maas_radius`: local window radius around target token.
- 中文：目标点周围局部窗口半径（token网格单位）。推荐 `1`；目标较模糊时可试 `2`。

- `motion_target_source`: source of target trajectory for MAAS.
- Values: `motion_centroid` / `object_token_trajectory`.
- 中文：MAAS 的目标轨迹来源。`motion_centroid` 来自生成视频像素运动中心；`object_token_trajectory` 来自 cross-attention object 轨迹。研究语义跟踪建议先用 `object_token_trajectory`。

- `condition_remove_scope` / `ffn_remove_scope` / `prompt_replace_scope`:
- Values: `single_step` / `from_step`.
- 中文：干预范围控制。`single_step` 仅干预指定 step；`from_step` 干预指定 step 及其后续步骤。

- `reuse_removed_cond_for_uncond`:
- Values: `true` / `false` (default `false`).
- 中文：当 cross-attention 被移除时，是否复用 cond 分支结果给 uncond 分支以节省计算。

- `replacement_prompt`:
- Used by `step_window_prompt_replace`.
- 中文：用于 prompt 替换实验的目标文本（Prompt B）。

- `replace_cond_only`:
- Values: `true` / `false` (default `true`).
- 中文：是否只替换 cond 分支的 prompt（推荐 `true`，更符合 CFG 分析语义）。

- `ablate_heads`:
- Used by `cross_attn_head_ablation`.
- Format: CSV `LxHy` (recommended), e.g. `L29H7,L12H3` (also accepts `(x,y)`).
- 中文：指定要消融的 cross-attention 头。推荐写法 `LxHy`，如 `L29H7`。

- `head_ablation_steps`:
- Used by `cross_attn_head_ablation`.
- Values: CSV steps or empty string `""`.
- 中文：指定在哪些扩散步做 head 消融。留空表示对所有扩散步都消融。

- `skip_existing_pdfs`:
- Used by `cross_attention_token_viz`.
- Values: `true` / `false` (default `true`).
- 中文：保存可视化 PDF 时（attention map / trajectory / timeline），若目标文件已存在则直接跳过，支持中断后续跑。

- `trajectory_entropy_steps`:
- Values: CSV steps or empty string `""`.
- 中文：轨迹熵的 step-wise / head-wise 采集步。留空表示全步长 `[1..sampling_steps]`。

- `trajectory_entropy_stepwise_layer`:
- Values: integer layer index, `-1` for last layer.
- 中文：step-wise 分析使用的层号，`-1` 表示最后一层。

- `trajectory_entropy_layerwise_steps`:
- Values: CSV steps or empty string `""`.
- 中文：layer-wise 曲线使用的 step 列表。留空时默认等于 `trajectory_entropy_steps`（因此默认覆盖所有 steps）。

- `trajectory_entropy_head_layer`:
- Values: integer layer index, `-1` for last layer, `-2` for all layers.
- 中文：head-wise 分析所选层号，`-1` 表示最后一层，`-2` 表示全部层。

- `reuse_cross_attention_dir`:
- Values: existing `cross_attention_token_viz` output directory path.
- 中文：可复用已有 cross-attention 结果目录。`trajectory_entropy` 将直接读取 `cross_attention_maps.pt`（或 stream 文件），不重新采样；`joint_attention_suite` 会跳过 stage-1。

- `head_evolution_steps`:
- Values: CSV steps or empty string `""`.
- 中文：`head_evolution` 的 step-wise/head-wise 分析步列表。留空表示使用复用目录中的所有可用 steps。

- `head_evolution_layerwise_steps`:
- Values: CSV steps or empty string `""`.
- 中文：`head_evolution` 的 layer-wise 分析步列表。留空时默认等于 `head_evolution_steps`。

- `head_evolution_stepwise_layer`:
- Values: layer index (`-1` means last layer).
- 中文：step-wise 曲线使用的层号。推荐 `27`（与你现有实验约定一致）。

- `head_evolution_head_layer`:
- Values: `-2` (all layers), `-1` (last layer), or non-negative layer index.
- 中文：head-wise 曲线分析层号，`-2` 表示所有层。

- `head_evolution_reference_step` / `head_evolution_reference_layer`:
- 中文：构建参考轨迹和支撑掩码时使用的参考图位置（step 1-based，layer 0-based）。

- `head_evolution_center_mode`:
- Values: `peak` / `centroid` / `geometric_center`.
- 中文：参考中心点选取方式。`peak` 取峰值；`centroid` 使用分位阈值+连通域加权质心；`geometric_center` 使用连通域几何中心。`bbox_center` 不再支持。

- `head_evolution_support_radius_mode`:
- Values: `fixed` / `adaptive_area`.
- 中文：半径模式。`adaptive_area` 按参考高亮连通域面积自适应半径，`fixed` 使用固定半径。

- `head_evolution_save_reference_radius_overlay`:
- Values: `true` / `false` (default `true`).
- 中文：是否保存参考图中心点+半径圆盘叠加图，用于检查 `r` 是否合理。

- `head_evolution_reference_viz_num_frames`:
- Values: integer.
- 中文：参考半径叠加图展示多少帧（时间轴并排）。

- `head_evolution_early_step_end`:
- Values: integer.
- 中文：head 打分时用于区分“早期窗口”和“后期窗口”的边界 step。

- `head_evolution_score_quantile`:
- Values: float in `[0,1]`.
- 中文：head 分类阈值采用的分位数参数（用于 early/late/state/MSS 阈值）。

- `head_evolution_apply_preprocess_on_metrics`:
- Values: `true` / `false` (default `true`).
- 中文：是否在各项指标计算前对 map 做 winsorize+despike。设为 `false` 时，会跳过指标阶段所有离群点相关逻辑（包括 top-k 连通域集中度计算）；注意参考轨迹（object center 与邻域）提取仍固定使用去离群点后的 map。

- `head_evolution_preprocess_winsorize_quantile`:
- Values: float in `[0,1]`.
- 中文：每帧 winsorize 截断分位点，用于压制极端高值离群点（attention sink）。

- `head_evolution_preprocess_despike_quantile`:
- Values: float in `[0,1]`.
- 中文：每帧空间去尖刺分位阈值。先取高值掩码，再做连通域面积筛选，移除零星高亮点。

- `head_evolution_preprocess_min_component_area`:
- Values: integer `>=1`.
- 中文：空间去尖刺时保留连通域的最小面积阈值（像素/token 数）。

- `head_evolution_concentrated_region_top_ratio`:
- Values: float in `(0,1]`.
- 中文：计算 concentrated-region score 时选取前多少比例的高权重 token。

- `head_trajectory_dynamics_heads`:
- Values: CSV `LxHy` specs or empty string `""`.
- 中文：指定参与共识/吸引子动态分析的 heads。留空表示使用复用 map 里的所有 heads。

- `head_trajectory_dynamics_steps`:
- Values: CSV steps or empty string `""`.
- 中文：指定分析哪些扩散步。留空表示使用复用 map 里的所有可用 steps；若只关心 motion plan，建议先用 `1,2,3,4,5,6`。

- `head_trajectory_dynamics_distance_metrics`:
- Values: `js`, `wasserstein`, `js,wasserstein`, or empty string `""`.
- 中文：head 轨迹距离度量。`wasserstein` 当前是 soft-center displacement 近似，不是完整二维最优传输。

- `head_trajectory_dynamics_reference_step` / `head_trajectory_dynamics_reference_layer`:
- 中文：用于构造最终参考软轨迹的位置，默认 `step=50, layer=27`。

- `self_attn_kernel_steps`:
- Values: CSV steps or empty string `""`.
- 中文：在哪些扩散步对 self-attention 输出做时间核混合。留空表示所有 steps。

- `self_attn_kernel_layers`:
- Values: CSV layer ids or empty string `""`.
- 中文：在哪些 DiT 层做时间核混合。留空表示所有层。

- `self_attn_kernel_branch`:
- Values: `cond` / `uncond` / `both`.
- 中文：指定干预 CFG 的哪个分支。Wan T2V 中每个 step 先跑 cond，再跑 uncond。

- `self_attn_kernel_radius` / `self_attn_kernel_sigma` / `self_attn_kernel_mix_alpha`:
- 中文：时间核半径、核宽度、残差混合强度。`mix_alpha=0` 是 no-op，`mix_alpha=1` 完全使用平滑后的 self-attention 输出。

- `seed_to_trajectory_early_steps`:
- Values: CSV steps.
- 中文：用于预测最终轨迹的早期扩散步。推荐先关注前几步，例如 `1,2,3,4,5,6`。

- `seed_to_trajectory_reference_step` / `seed_to_trajectory_reference_layer` / `seed_to_trajectory_head`:
- 中文：参考轨迹标签的提取位置。默认 `step=50, layer=27, head=mean`。

- `seed_to_trajectory_num_points`:
- Values: integer.
- 中文：每条轨迹重采样为多少个点后进入预测模型；若设为 `<= 0`，则直接使用原始 latent-frame 个数，不做插值。

- `seed_to_trajectory_ridge_alpha`:
- Values: non-negative float.
- 中文：leave-one-seed-out ridge regression 的 L2 正则强度。

- `event_token_value_words`:
- Values: CSV words.
- 中文：额外标记为 `event` 的动作/事件词。token 匹配复用 `cross_attention_token_viz` 的 `_locate_wan21_t2v_prompt_words`。

- `event_token_value_steps` / `event_token_value_layers`:
- 中文：采集 event-token value contribution 的扩散步和层。layers 留空表示所有层。

- `event_token_value_branch`:
- Values: `cond` / `uncond` / `both`.
- 中文：指定采集 CFG 的哪个分支。

- `event_token_value_chunk_size`:
- Values: integer.
- 中文：计算 cross-attention token contribution 时 query 维分块大小。调小可省显存，调大可更快。

- `event_token_value_num_viz_frames`:
- Values: integer.
- 中文：head-mean projected-value-norm timeline PDF 中展示多少帧。

## How To Read DT Plots

- `attention_dt_profile_global_dt_curve.pdf`:
- 含义：每条曲线对应一个 diffusion step，横轴是 `|dt|`（帧间距离），纵轴是注意力质量。
- 解读：曲线越靠左，表示更偏好邻近帧；越平坦/右移，表示更依赖远帧信息。

- `attention_dt_profile_layer_heatmaps/step_XXX.pdf`:
- 含义：固定某个 step，纵轴层号，横轴 `|dt|`，颜色是该层对不同时间偏移的概率质量。
- 解读：可看“浅层局部、深层全局”是否成立，以及不同 step 是否发生明显迁移。

## How To Read Motion-Aligned Maps

- 文件位置：`motion_alignment_visualizations/next_frame_attention_maps/step_xxx/layer_xx/query_frame_xxx.pdf`
- `query_frame_xxx` 的含义：以第 `query_frame` 帧的 query token 作为查询，画其对 `query_frame+1` 帧的注意力热力图。
- 图中红色 `x`：目标位置（target point），来自你选择的 `motion_target_source` 轨迹在 `query_frame+1` 的位置。
- 图中橙色方框：以 `maas_radius` 为半径的局部窗口（MAAS 统计区域）。
- 热力图亮区若集中在橙框内，表示 next-frame 注意力与目标运动位置更一致。

### Stream Flush Flags

- `--stream_flush_per_step` (default: `false`)
  - `true`: write per-step attention maps to `cross_attention_maps_stream/` and clear in-memory cache.
- `--plot_during_sampling` (default: `false`)
  - only takes effect when stream flush is enabled.
  - `true`: plot PDFs while sampling progresses.
  - `false`: plot after sampling from saved per-step map files.
- `--viz_frame_indices` (default: empty)
  - explicit frame indices, comma-separated; if set, overrides `--viz_num_frames`.
  - accepts video-frame indices in either 1-based (`1..frame_num`) or 0-based (`0..frame_num-1`) form.
  - maps are indexed on attention time grid internally, but displayed frame labels follow video-frame numbering.
- `--draw_attention_map_only` (default: `false`)
  - `true`: skip model sampling and redraw visualizations from saved attention maps.
- `--draw_attention_maps_path` (default: empty)
  - optional path to a map file; if empty, load `output_dir/cross_attention_maps.pt` or stream index.
- `--visualization_output_dir` (default: empty)
  - optional separate directory for PDF redraw outputs; useful to keep previous figures unchanged.
- `--skip_existing_pdfs` (default: `true`)
  - affects all PDF rendering in `cross_attention_token_viz` (map/trajectory/timeline).
  - `true`: if target PDF already exists, skip drawing this file.
  - useful for resume after OOM/kill during long visualization rendering.
