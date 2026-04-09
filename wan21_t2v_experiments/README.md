# Wan2.1-T2V Experiments (Monkey Patch)

This folder contains standalone experiment scripts for Wan2.1 text-to-video analysis.
All modifications are runtime monkey patches and do not edit `projects/Wan2_1` source files.

本目录提供 Wan2.1-T2V 的可解释性实验脚本。  
所有改动都通过运行时 monkey patch 完成，不会改动 `projects/Wan2_1` 原始代码。

## Implemented Experiments

- `rope_axis_ablation`
- `attention_dt_profile`
- `trajectory_entropy`
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
