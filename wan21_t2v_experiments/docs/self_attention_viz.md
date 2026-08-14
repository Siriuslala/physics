# Self Attention Viz

## Motivation

This experiment studies how Wan2.1 T2V self-attention redistributes mass from an object-centered query region to the full video latent grid. The main target is not the full self-attention matrix over all video tokens, which is too large to store densely, but a structured subset of rows: only query tokens inside a reference object region are probed, while all video-latent key tokens remain visible.

The experiment is designed to preserve the original diffusion trajectory. Wan's native self-attention forward path is left unchanged. The probe only reads the patched query and key tensors and computes an additional sampled attention map on the side.

## Reference Object Region

The object query region is derived from reused cross-attention outputs. Let `A_ref \in \mathbb{R}^{F \times H \times W}` denote the head-mean cross-attention map for the target object words at diffusion step `s_ref = 50` and layer `\ell_ref = 27`, where:

- `F` is the latent-frame count,
- `H` is the latent token-grid height,
- `W` is the latent token-grid width.

The map is first preprocessed by winsorization and despiking. A dominant connected component is then extracted frame by frame, and one center trajectory `c_f = (y_f, x_f)` is constructed for `f = 0, \dots, F-1`. The center type can be `peak`, `centroid`, or `geometric_center`.

Given `c_f`, the reference support mask `M \in \{0, 1\}^{F \times H \times W}` is built as a disk neighborhood:

`M[f, y, x] = \mathbf{1}\{(y - y_f)^2 + (x - x_f)^2 \le r_f^2\}`,

where `r_f` is either a fixed radius or an area-adaptive radius derived from the connected-component area of frame `f`.

## Query Sampling

Let `F_video` denote the original video-frame count, and let `Q_video = \{1, 33, 41, 81\}` be the default set of 1-based query video-frame labels. These video-frame labels are projected to latent token-frame indices by:

`q_f = \operatorname{round}\left((v_f - 1)\frac{F - 1}{F_{video} - 1}\right)`,

where `v_f \in Q_video`.

For each selected token frame `q_f`, all object-region query tokens are collected from `M[q_f]`. If `N_{raw}(q_f)` exceeds the configured limit `N_{max}`, the query-token set is uniformly subsampled in latent-token index order to size `N(q_f) = \min(N_{raw}(q_f), N_{max})`.

## Self-Attention Probe

At one selected diffusion step `s` and one selected DiT layer `\ell`, let:

- `Q \in \mathbb{R}^{L \times N_h \times D_h}` be the self-attention query tensor,
- `K \in \mathbb{R}^{L \times N_h \times D_h}` be the self-attention key tensor,
- `L = FHW` be the full latent-token sequence length,
- `N_h` be the head count,
- `D_h` be the head dimension.

For one selected query frame `q_f`, the experiment extracts only the object-region rows. If the selected query-token indices are `I_{q_f} = \{i_1, \dots, i_{N(q_f)}\}`, the sampled logits for one head `h` are:

`\Lambda_h[a, b] = \frac{\langle Q[i_a, h, :], K[b, h, :] \rangle}{\sqrt{D_h}}`,

for `a = 1, \dots, N(q_f)` and `b = 1, \dots, L`.

The corresponding probabilities are:

`P_h[a, b] = \frac{\exp(\Lambda_h[a, b])}{\sum_{b'=1}^{L}\exp(\Lambda_h[a, b'])}`.

Each row `P_h[a, :]` is reshaped to `\mathbb{R}^{F \times H \times W}`. The final self-attention visualization map for head `h` and query frame `q_f` is the average over all sampled object-region query tokens:

`S_h(q_f) = \frac{1}{N(q_f)}\sum_{a=1}^{N(q_f)} \operatorname{reshape}(P_h[a, :], F, H, W)`.

Therefore the saved tensor per `(step, layer, query_frame)` key has shape `N_h \times F \times H \times W`.

This confirms the intended aggregation rule: the per-head result is first a stack of `N(q_f)` maps with shape `F \times H \times W`, and then the query-token axis is averaged away.

## Engineering Design

### Non-interference with Diffusion

The probe is executed before Wan's own self-attention output is dispatched, but it never overwrites the tensors used by the forward path. The original diffusion branch still computes:

- the same RoPE-transformed `q`, `k`, `v`,
- the same Wan attention backend (`flash` or `torch_sdpa`),
- the same self-attention output tensor written back to the residual stream.

The experiment only performs an auxiliary `einsum + softmax` on a small subset of query rows.

### Output Format

The main tensor artifact is:

- `self_attention_viz_maps.pt`

It stores a Python dictionary keyed by `(step, layer, query_token_frame)` with value shape `[num_heads, F, H, W]`.

The experiment also exports:

- `self_attention_viz_index.csv`
- `self_attention_viz_reference_support.csv`
- `self_attention_viz_summary.json`

Visualization PDFs are organized under:

- `self_attention_viz/step_XXX/layer_YY/query_frame_ZZZ/head_XX.pdf`
- `self_attention_viz/step_XXX/layer_YY/query_frame_ZZZ/head_mean.pdf`

All self-attention PDFs use the `viridis` colormap.

During visualization materialization, PDF rendering can be parallelized over CPU workers, and a progress bar is shown for the plotting stage.

## Main Parameters

- `self_attention_viz_steps`: diffusion steps to probe. Empty means all steps.
- `self_attention_viz_layers`: layer ids to probe. Empty means all layers.
- `self_attention_viz_branch`: `cond`, `uncond`, or `both`.
- `self_attention_viz_reference_step`: cross-attention reference step, default `50`.
- `self_attention_viz_reference_layer`: cross-attention reference layer, default `27`.
- `self_attention_viz_reference_center_mode`: center type for the reference trajectory.
- `self_attention_viz_support_radius_mode`: support-radius mode, `fixed` or `adaptive_area`.
- `self_attention_viz_query_video_frame_indices`: selected 1-based query video-frame labels.
- `self_attention_viz_object_query_token_limit_per_frame`: optional per-frame query-token subsampling budget.
- `self_attention_viz_num_viz_frames`: number of key-frame panels shown in each PDF.
- `self_attention_viz_viz_frame_indices`: optional explicit key-frame labels for rendering; when empty, the same uniform frame-sampling rule as `cross_attention_token_viz` is used.
- `self_attention_viz_attention_pdf_share_color_scale`: if `true`, all panels in one PDF share the same color scale.
- `self_attention_viz_stop_after_last_probe_step`: if `true`, diffusion stops immediately after the final requested probe step.
- `draw_self_attention_maps_only`: if `true`, skip sampling and redraw visualization PDFs only from a saved `.pt` map file.
- `draw_self_attention_maps_path`: optional explicit `.pt` path used by redraw-only mode.
- `self_attention_viz_visualization_output_dir`: optional directory used to store newly rendered PDFs.
- `self_attention_viz_num_workers`: CPU worker count for visualization materialization. Non-positive values fall back to `os.cpu_count()`.

## Running

The standard launcher script is:

`scripts/self_attention_viz_wan2_1.sh`

The experiment requires an existing `cross_attention_token_viz` output directory because the reference object region is built from reused cross-attention maps.

## Interpretation Notes

This experiment is most informative when comparing multiple query-frame positions. Early and late query frames can access the largest signed temporal offsets, while middle query frames reveal whether long-range decay remains symmetric after conditioning on a more central temporal position.
