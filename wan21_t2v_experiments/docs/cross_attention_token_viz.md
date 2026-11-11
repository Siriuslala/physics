# `cross_attention_token_viz` Technical Note

## 1. Motivation

`cross_attention_token_viz` is the base experiment used to inspect how video latent tokens attend to selected text tokens during denoising. Its most important derived object is the object-token trajectory: for a chosen text token such as `basketball`, the experiment extracts one spatial center from each latent frame and connects these centers into a trajectory.

This trajectory is used later by several downstream analyses, so the exact center-extraction logic matters.

## 2. Input / Output

### Inputs

- Saved or online-collected cross-attention maps.
- Selected object words and verb words.
- Trajectory parameters:
  - `trajectory_power`
  - `trajectory_quantile`
  - `trajectory_smooth_radius`
  - `trajectory_num_frames`
  - `trajectory_timeline_num_frames`

### Outputs

For each selected `(step, layer, head, token)`:

- cross-attention map PDF
- trajectory PDF
- trajectory timeline PDF
- trajectory CSV rows

## 3. Tensor Definition

Let

\[
A \in \mathbb{R}_{\ge 0}^{F \times H \times W}
\]

be the cross-attention map for one selected text token after choosing one diffusion step, one DiT layer, and one attention head.

Where:

- \(F\) is the latent-frame count.
- \(H\) is the token-grid height.
- \(W\) is the token-grid width.
- \(A_f(y, x)\) is the attention weight at frame \(f\), row \(y\), column \(x\).

The trajectory is a sequence of 2D points

\[
\mathcal{T} = \{(\hat y_f, \hat x_f)\}_{f=1}^{F}.
\]

## 4. Center Extraction Used by `cross_attention_token_viz`

The actual trajectory extraction used by this experiment is implemented by

- `_extract_wan21_t2v_attention_region_center_trajectory`

in [utils.py]

This is **not** the simple global soft center over the whole frame. It is a more localized region-center method.

### Step 1. Frame-wise nonnegative attention map

For each frame \(f\), define

\[
A_f(y, x) = \max(A_f(y, x), 0).
\]

### Step 2. Peak location

Flatten the frame and find the peak index

\[
(y_f^{\mathrm{peak}}, x_f^{\mathrm{peak}}) = \arg\max_{y, x} A_f(y, x).
\]

This peak is used as the anchor for connected-component selection.

### Step 3. Quantile thresholding

Given quantile parameter \(q \in [0, 1]\), compute the scalar threshold

\[
\tau_f = \operatorname{Quantile}(\{A_f(y, x)\}_{y,x}, q).
\]

Then construct the binary mask

\[
M_f(y, x) = \mathbf{1}[A_f(y, x) \ge \tau_f].
\]

If the peak location is not included, it is force-added into the mask.

### Step 4. Connected component containing the peak

Among all 8-neighborhood connected components of \(M_f\), the algorithm keeps **only the component that contains the peak**.

Denote this component by

\[
\Omega_f \subseteq \{0, \dots, H-1\} \times \{0, \dots, W-1\}.
\]

This is the most important design choice: the algorithm does **not** compute the centroid on all threshold-surviving pixels; it computes it only on the dominant high-attention region that contains the peak.

### Step 5. Power sharpening inside the selected component

Given `trajectory_power = p > 0`, define weights on the selected component:

\[
w_f(y, x) = A_f(y, x)^p, \qquad (y, x) \in \Omega_f.
\]

The exponent \(p\) sharpens large values when \(p > 1\).

### Step 6. Weighted centroid of the component

The final frame center is

\[ \hat y_f = \frac{\sum_{(y,x) \in \Omega_f} y \, w_f(y, x)}{\sum_{(y,x) \in \Omega_f} w_f(y, x)}, \qquad \hat x_f = \frac{\sum_{(y,x) \in \Omega_f} x \, w_f(y, x)}{\sum_{(y,x) \in \Omega_f} w_f(y, x)}. \]

If the denominator is numerically zero, the algorithm falls back to the peak point.

## 5. Post-processing of the Trajectory

After extracting the raw center sequence, the experiment performs two downstream operations.

### 5.1 Temporal smoothing for plotting

The plotted trajectory may be smoothed by a moving average with radius `trajectory_smooth_radius`. If the raw trajectory is

\[
\mathcal{T} = \{(\hat y_f, \hat x_f)\}_{f=1}^{F},
\]

then the smoothed point at frame \(f\) is the average over a local temporal window.

This smoothing is used **only for visualization**.

### 5.2 Exported CSV uses raw centers

The CSV export intentionally uses the **raw frame-local centers before temporal smoothing**, so downstream analyses do not inherit plotting-specific smoothing bias.

## 6. CPU Parallel Materialization

After the cross-attention maps have been collected or reloaded from disk, the experiment materializes per-head attention PDFs and object-token trajectories using a CPU worker pool.

Let

\[
\mathcal{K} = \{(s_i, \ell_i, t_i, h_i)\}_{i=1}^{N}
\]

be the ordered set of materialization tasks, where each element corresponds to one diffusion step \(s_i\), one layer \(\ell_i\), one token \(t_i\), and one head \(h_i\). Each task independently renders:

- the per-head attention PDF,
- the optional trajectory PDF,
- the optional trajectory timeline PDF,
- the associated CSV rows for that head or head-mean bundle.

The worker count is controlled by `cross_attention_token_viz_num_workers`. If this value is non-positive, the implementation falls back to `os.cpu_count()` and then clamps the effective worker count to the number of tasks. This parallelization changes only the execution schedule, not the mathematical definition of the exported maps or trajectories.

## 7. Interpretation

This experiment's center extractor can be summarized as:

- localize the frame's dominant attention mode,
- reject disconnected high-valued clutter away from the peak,
- then compute a weighted centroid **inside that dominant region**.

Compared with a global soft center, it is much less sensitive to unrelated bright regions such as background or floor highlights, as long as those regions are not connected to the peak region.

## 8. Current Limitation

This method still depends on the peak-containing component. If the peak itself lands on a spurious region, the extracted center will follow that region. So the method is more robust than a global expectation, but it is not immune to all failure cases.


## Fine-tuned LoRA and early spatial RoPE intervention

Both attention visualization experiments accept the same optional model intervention. The base `--ckpt_dir` remains the official Wan checkpoint directory. `--finetune_lora_checkpoint` selects an exact adapter `.safetensors` file, `--finetune_lora_rank` and `--finetune_lora_alpha` specify the adapter rank and scaling numerator, and `--finetune_lora_modules` selects `attn` (self/cross-attention projections `q,k,v,o`) or `attn_ffn` (these projections plus `ffn.0,ffn.2`). The loader checks the complete adapter key set and every tensor shape before loading. Missing, unexpected, duplicate, and non-adapter tensors are rejected. Base-model missing keys are expected when loading an adapter-only checkpoint. Adapters remain unmerged, matching `wan_eval/infer_wan_t2v.py`.

For a projection input $x\in\mathbb{R}^{d_{\mathrm{in}}}$, let $W\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}$ be its frozen weight, $A\in\mathbb{R}^{r\times d_{\mathrm{in}}}$ and $B\in\mathbb{R}^{d_{\mathrm{out}}\times r}$ its loaded adapter matrices, $r\in\mathbb{N}$ the rank, and $\alpha\in\mathbb{R}_{>0}$ the scaling numerator. The adapted projection, omitting its unchanged bias, is:

$$y=Wx+(\alpha/r)BAx.$$

`--rope_modification_lambda_h` and `--rope_modification_lambda_w` set fixed spatial phase scales. The temporal scale is one. `--rope_modification_steps` is a CSV list of **one-based denoising iteration numbers**, not scheduler timestep values; an empty list applies the scales to every iteration. For a step index $s\in\{1,\ldots,S\}$, where $S\in\mathbb{N}$ is the total number of sampling steps, let $\mathcal{I}\subseteq\{1,\ldots,S\}$ be the selected intervention iterations, and let $\lambda_a\in\mathbb{R}_{>0}$ be the scale for axis $a\in\{f,h,w\}$. Each original complex RoPE multiplier $z_{a,j}(p_a)\in\mathbb{C}$ has unit modulus; $p_a\in\mathbb{N}_0$ is an axis position and $j$ indexes that axis's complex channel. The implementation uses:

$$\widetilde z_{a,j}^{(s)}(p_a)=\exp\!\left(i\lambda_a\operatorname{Arg}z_{a,j}(p_a)\right)\quad\text{for }s\in\mathcal{I}.$$

Outside the selected iterations the original multiplier is retained. Here $i$ is the imaginary unit and $\operatorname{Arg}$ is the principal complex argument. This convention deliberately matches the fixed-lambda inference implementation in `wan_eval/infer_wan_t2v.py`: scaling a wrapped complex phase is not generally identical to multiplying the unwrapped position-frequency product by the scale. Identity axes retain their original complex values rather than reconstructing them through an angle/polar round trip. Fixed spatial scales follow the evaluation module precision convention: the requested values are converted to float32 logarithms on the model device and pass through exp, clamped log, and exp before promotion to the RoPE computation dtype. Thus requested 0.70 has effective value 0.699999988079071, while 0.75 remains exact. The summary preserves requested `rope_lambda_h/w` and separately records `rope_lambda_h_effective`, `rope_lambda_w_effective`, and `rope_lambda_precision`. Both conditional and unconditional CFG branches receive the same intervention at each selected iteration.

The self-attention experiment passes this configuration to its existing q/k probe patch, so its recorded maps use the same modified q/k as generation. The cross-attention experiment installs the RoPE patch before the recording patch and restores them in reverse order. Neither path edits official model source. `model_intervention` in the summary JSON records the exact checkpoint path, adapter rank/alpha/modules and tensor count, spatial scales, iteration list, and phase convention. Redrawing previously saved maps does not reload a model and records no new intervention provenance; the original collection summary remains the authoritative source.

The basketball comparison uses the following paired settings; each modified run must use its own cross-attention directory for object support extraction:

| Seed | LoRA modules | Checkpoint filename | Rank / alpha | Spatial scales (h, w) | Intervention iterations | Cross-attention reference |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | `attn` | `step-1000.safetensors` | 64 / 32 | 0.75, 0.75 | 1,2,3,4,5 | step 45, layer 27 |
| 20 | `attn` | `step-1000.safetensors` | 64 / 32 | 0.75, 0.75 | 1,2,3,4,5 | step 50, layer 27 |
| 29 | `attn_ffn` | `step-700.safetensors` | 64 / 32 | 0.70, 0.70 | 1,2,3,4,5 | step 50, layer 27 |

The `attn` checkpoint belongs to the fixed-lambda training run with early-timestep probability 0.9; the `attn_ffn` checkpoint belongs to the run with probability 0.5. Both training runs use spatial lambda 0.75, while seed 29 deliberately uses inference lambda 0.70. Use `--self_attention_viz_query_video_frame_indices 1` to query only the first video frame. Inspect object-support overlays before interpreting attention maps. A comparison between the original model and LoRA plus modified RoPE measures their joint intervention; isolating the causal contribution of RoPE requires a matched fine-tuned-model run with identity RoPE.

Sampling summaries record the base checkpoint directory, seed, resolution, solver, step count, shift, and guidance scale alongside the model intervention.

## Selected-token PDF materialization from saved maps

`redraw_cross_attention.py` renders selected tokens from `cross_attention_maps.pt` without constructing a model or allocating GPU tensors. Token filtering occurs before creating rendering tasks. For each saved step/layer/token tensor, it renders every attention head and the arithmetic probability-map mean across heads, using the existing `_save_wan21_t2v_cross_attention_pdf` helper. Display settings are read from the original collection summary. `--num-viz-frames` overrides frame sampling using the standard sampler and clears explicit frame indices. The modified seed 20 launcher explicitly selects 10 frames to match the baseline visualization: video frames 1, 9, 17, 25, 33, 41, 49, 57, 65, and 81. Per-frame normalization and shared-scale settings retain their collection values.

The command writes PDFs directly into `timestep_SSS/token_object_basketball/layer_LL_head_HH.pdf`, with `head_mean` for the head average. It preserves the saved tensors, original collection summary, and collection index. A separate `cross_attention_redraw_basketball.json` records source, selected tokens, covered steps/layers, rendering settings, and generated paths. Original `save_attention_pdfs=False` metadata describes collection-time behavior; subsequent PDF availability is recorded in this separate manifest.

```bash
bash scripts/redraw_cross_attention_modified_seed20.sh
```

The supplied launcher selects only `basketball` for modified seed 20, using ten displayed frames and eight CPU workers. With 50 steps, 30 layers, and 12 heads, it produces 19,500 PDFs including head means. The general entry point is `python -m wan21_t2v_experiments.redraw_cross_attention --directory /absolute/cross_attention/path --tokens basketball --workers 8`. It regenerates the selected PDFs and does not render other saved tokens or trajectory plots.
