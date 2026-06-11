# `self_attention_modulation` Technical Note

## 1. Goal

This experiment profiles the self-attention modulation tensors inside each Wan2.1 DiT block.

For one block, Wan2.1 computes six modulation tensors from the shared timestep embedding and the block-specific modulation bias. The self-attention branch uses

\[ e_0,\; e_1,\; e_2 \in \mathbb{R}^{B \times 1 \times C}, \]

where:

- \(B\) is the batch size of the current forward call,
- \(C\) is the hidden width of the DiT block,
- \(e_0\) is the additive modulation before self-attention,
- \(e_1\) is the multiplicative modulation before self-attention,
- \(e_2\) is the write gate that multiplies the self-attention output before it is added back to the residual stream.

The experiment answers:

1. how \(e_0\), \(e_1\), and \(e_2\) vary across diffusion steps,
2. how they vary across DiT layers,
3. whether the effective self-attention write controlled by \(e_2\) becomes stronger or weaker during early denoising.

## 2. Official Wan2.1 Block Equation

For one block, after modulation chunking, the self-attention branch is

\[ \hat{x} = \operatorname{LN}(x) \odot (1 + e_1) + e_0, \]

\[ y = \operatorname{SA}(\hat{x}), \]

\[ x' = x + y \odot e_2. \]

Here:

- \(x \in \mathbb{R}^{B \times L \times C}\) is the residual stream before the self-attention write,
- \(L\) is the latent token sequence length,
- \(\hat{x} \in \mathbb{R}^{B \times L \times C}\) is the modulated self-attention input,
- \(y \in \mathbb{R}^{B \times L \times C}\) is the raw self-attention output before the gate,
- \(x' \in \mathbb{R}^{B \times L \times C}\) is the updated residual stream.

The key point is that all three tensors \(e_0\), \(e_1\), and \(e_2\) have shape \(B \times 1 \times C\), so they are shared across the token dimension and act channel-wise.

The current patch records:

- \(e_0\) directly from the additive pre-SA modulation term,
- \(e_1\) directly from the multiplicative pre-SA modulation term,
- \(e_2\) directly from the residual write gate in \(x' = x + y \odot e_2\).

## 3. Recorded Statistics

For each selected modulation tensor \(e_m \in \{e_0, e_1, e_2\}\), diffusion step \(s\), and layer \(\ell\), the experiment stores:

\[ \mu_{m,s,\ell} = \operatorname{mean}(e_m), \]

\[ \mu^{|\cdot|}_{m,s,\ell} = \operatorname{mean}(|e_m|), \]

\[ \operatorname{rms}_{m,s,\ell}(e_m) = \sqrt{\operatorname{mean}(e_m^2)}, \]

\[ p^{+}_{m,s,\ell} = \operatorname{mean}(\mathbf{1}[e_m > 0]), \]

\[ p^{-}_{m,s,\ell} = \operatorname{mean}(\mathbf{1}[e_m < 0]), \]

\[ \maxabs_{m,s,\ell} = \max |e_m|. \]

These six statistics are collected for all three tensors \(e_0\), \(e_1\), and \(e_2\).

For \(e_2\) only, the experiment additionally records write-strength statistics tied to the actual self-attention branch output:

\[ \operatorname{rms}_{s,\ell}(y) = \sqrt{\operatorname{mean}(y^2)}, \]

\[ \operatorname{rms}_{s,\ell}(y \odot e_2) = \sqrt{\operatorname{mean}((y \odot e_2)^2)}, \]

\[ \rho_{s,\ell} = \frac{\operatorname{rms}_{s,\ell}(y \odot e_2)}{\operatorname{rms}_{s,\ell}(y) + \varepsilon}. \]

Here \(\rho_{s,\ell}\) is the `gated_to_raw_rms_ratio`. It is useful because a large \(|e_2|\) does not necessarily imply a large effective residual write if the raw self-attention output \(y\) is already weak.

The experiment now also records one per-head family of write statistics for \(e_2\).
This family follows the same self-attention head-write definition used by `trajectory_consensus_dynamics`.

Let \(Z_h^{(s,\ell)} \in \mathbb{R}^{L \times d_h}\) be the post-attention, pre-output-projection tensor of self-attention head \(h\), and let \(W_O^{(h)} \in \mathbb{R}^{d_h \times C}\) be the corresponding slice of the self-attention output projection.
Define the per-head hidden-state write

\[ U_h^{(s,\ell)} = Z_h^{(s,\ell)} W_O^{(h)} \in \mathbb{R}^{L \times C}. \]

The gated per-head write is

\[ \widetilde U_h^{(s,\ell)} = U_h^{(s,\ell)} \odot e_2^{(s,\ell)}. \]

For each selected step \(s\), layer \(\ell\), and head \(h\), the per-head CSV stores

\[ \operatorname{rms}^{\mathrm{head}}_{s,\ell,h} = \sqrt{\operatorname{mean}((U_h^{(s,\ell)})^2)}, \]

\[ \operatorname{gated\_rms}^{\mathrm{head}}_{s,\ell,h} = \sqrt{\operatorname{mean}((\widetilde U_h^{(s,\ell)})^2)}, \]

\[ \rho^{\mathrm{head}}_{s,\ell,h} = \frac{\operatorname{gated\_rms}^{\mathrm{head}}_{s,\ell,h}}{\operatorname{rms}^{\mathrm{head}}_{s,\ell,h} + \varepsilon}. \]

These are exported as:

- `sa_head_write_rms`
- `gated_sa_head_write_rms`
- `gated_to_raw_sa_head_write_rms_ratio`

The experiment additionally records one decomposition family for the self-attention branch itself.
Using the same notation as Section 2, define

\[ \hat{x} = \operatorname{LN}(x) \odot (1 + e_1) + e_0, \]

\[ V = W_V \hat{x}, \]

\[ A_{\mathrm{pre}\text{-}O} = \operatorname{Attn}(Q, K, V), \]

\[ y = W_O A_{\mathrm{pre}\text{-}O}. \]

Here:

- \(\hat{x}\) is the modulated self-attention input,
- \(V\) is the value-path activation before attention aggregation,
- \(A_{\mathrm{pre}\text{-}O}\) is the attention output before the output projection \(W_O\),
- \(y\) is the final self-attention output already described above.

For each selected step \(s\) and layer \(\ell\), the decomposition CSV stores

\[ \operatorname{rms}_{s,\ell}(\hat{x}), \qquad \operatorname{rms}_{s,\ell}(V), \qquad \operatorname{rms}_{s,\ell}(A_{\mathrm{pre}\text{-}O}). \]

These are exported as:

- `x_hat_rms`
- `v_rms`
- `attn_out_pre_o_rms`

Finally, for selected `(step, layer)` pairs, the experiment can render channel-profile bar plots.
For one selected step \(s\), layer \(\ell\), and hidden channel \(c\), it stores:

\[ \mathrm{sa\_channel\_energy}_{s,\ell,c} = \operatorname{mean}_{B,L}(y_c^2), \]

\[ \mathrm{gate\_channel\_rms}_{s,\ell,c} = \sqrt{\operatorname{mean}_{B}(e_{2,c}^2)}, \]

\[ \mathrm{gated\_channel\_energy}_{s,\ell,c} = \operatorname{mean}_{B,L}((y_c e_{2,c})^2). \]

The full tensors are saved for all collected `(step, layer)` pairs, while the PDF bar plots are only rendered for user-selected targets.
Internally, the collected channel-profile tensors are accumulated on CPU during probing, and the plotting stage runs after the experiment has already exported CSV/PT artifacts and released the heavy runtime references to the pipeline, the patched DiT model, and the generated video tensor.
The experiment also supports a redraw-only mode that skips diffusion sampling entirely and regenerates the PDFs only from the previously saved CSV/PT artifacts inside one existing output directory.

## 4. Probe Scope

The probe uses the same step and branch selection logic as the existing self-attention analysis patches:

- `cond`: collect only the conditional CFG forward,
- `uncond`: collect only the unconditional CFG forward,
- `both`: collect both forwards and average them in the exported rows.

If `self_attention_modulation_layers` is empty, all DiT layers are collected.

If `self_attention_modulation_steps` is empty, all diffusion steps from `1` to `sampling_steps` are collected.

If `self_attention_modulation_stop_after_last_probe_step=true`, diffusion stops immediately after the largest requested probe step has completed.

## 5. Outputs

- `self_attention_modulation_rows.csv`
- `self_attention_modulation_head_rows.csv`
- `self_attention_modulation_decomposition_rows.csv`
- `self_attention_modulation_channel_profiles.pt`
- `self_attention_modulation_weight_norms.csv`
- `self_attention_modulation_summary.json`
- `self_attention_modulation_plots/e0/...`
- `self_attention_modulation_plots/e1/...`
- `self_attention_modulation_plots/e2/...`
- `self_attention_modulation_per_head_plots/...`
- `self_attention_modulation_decomposition_plots/...`
- `self_attention_modulation_channel_profiles/...`

The CSV contains one row per `(modulation_name, step, layer)` after averaging over all selected forward samples.

For each modulation tensor \(e_0\), \(e_1\), and \(e_2\), the plot root is

\[ \texttt{self\_attention\_modulation\_plots/<modulation\_name>/<metric>/}. \]

For all three modulation tensors, the following metrics are rendered:

- `gate_mean`
- `gate_abs_mean`
- `gate_rms`
- `gate_positive_fraction`
- `gate_negative_fraction`
- `gate_max_abs`

For each such metric, the experiment exports:

- `heatmap.pdf`: step \(\times\) layer heatmap
- `step_curves_bucketed.pdf`: one curve for `all`, `shallow`, `middle`, and `deep`, where the selected layers are split into three count-balanced contiguous layer-index buckets and the legend explicitly shows the layer ranges
- `step_curves_per_layer.pdf`: one overview figure that overlays one diffusion-step curve per selected layer and includes a layer-index color bar
- `per_layer/layer_xx.pdf`: one dedicated diffusion-step curve for one specific layer

For `e2` only, three additional metrics are rendered:

- `sa_output_rms`
- `gated_sa_output_rms`
- `gated_to_raw_rms_ratio`

So, for example, the `e2` gate RMS heatmap is located at

\[ \texttt{self\_attention\_modulation\_plots/e2/gate\_rms/heatmap.pdf}. \]

The bucketed step curve for the same metric is

\[ \texttt{self\_attention\_modulation\_plots/e2/gate\_rms/step\_curves\_bucketed.pdf}. \]

The per-layer overview figure is

\[ \texttt{self\_attention\_modulation\_plots/e2/gate\_rms/step\_curves\_per\_layer.pdf}. \]

One dedicated single-layer figure is, for example,

\[ \texttt{self\_attention\_modulation\_plots/e2/gate\_rms/per\_layer/layer\_12.pdf}. \]

The per-head outputs are stored separately so that the existing layer-wise outputs remain unchanged.

The per-head CSV contains one row per `(step, layer, head)` after averaging over all selected forward samples.

The per-head plot root is

\[ \texttt{self\_attention\_modulation\_per\_head\_plots/<metric>/}. \]

Currently it renders three metrics:

- `sa_head_write_rms`
- `gated_sa_head_write_rms`
- `gated_to_raw_sa_head_write_rms_ratio`

For each metric and each selected layer, two figures are exported:

- `layer_xx_heatmap.pdf`: step \(\times\) head heatmap
- `layer_xx_step_curves.pdf`: one step curve per head together with a head-index color bar

For example,

\[ \texttt{self\_attention\_modulation\_per\_head\_plots/gated\_sa\_head\_write\_rms/layer\_26\_heatmap.pdf}. \]

The decomposition plot root is

\[ \texttt{self\_attention\_modulation\_decomposition\_plots/}. \]

It renders:

- `x_hat_rms_heatmap.pdf`
- `v_rms_heatmap.pdf`
- `attn_out_pre_o_rms_heatmap.pdf`

The layer-wise self-attention parameter norms are exported as one CSV plus one summary PDF:

- `self_attention_modulation_weight_norms.csv`
- `self_attention_modulation_weight_norms.pdf`

The channel-profile tensor file

\[ \texttt{self\_attention\_modulation\_channel\_profiles.pt} \]

stores all collected `(step, layer)` profiles.

By default, channel-profile PDFs are rendered for all collected `(step, layer)` pairs.
The CLI can optionally restrict this to selected target pairs.
When all collected pairs are rendered, the plotting loop shows an explicit progress bar so that long full-grid rendering jobs remain observable.
The per-head plot rendering loop also shows an explicit progress bar because it traverses all selected metric-layer pairs.
For one target `(step, layer)`, the plot directory is

\[ \texttt{self\_attention\_modulation\_channel\_profiles/step\_xxx/layer\_xx/}. \]

It contains:

- `sa_channel_energy.pdf`
- `sa_channel_energy_topk_annotated.pdf`
- `gate_channel_rms.pdf`
- `gate_channel_rms_topk_annotated.pdf`
- `gated_channel_energy.pdf`
- `gated_channel_energy_topk_annotated.pdf`

In the annotated channel-profile PDFs, all non-annotated bars are blue, while the top-\(k\) annotated bars are green.
If one annotation would exceed the right plot boundary, its text anchor is switched so that the annotation right edge is clamped to the plot boundary instead of overflowing outside the figure.
If one annotation would exceed the lower plot boundary, its text anchor is also switched so that the annotation bottom edge is clamped to the lower plot boundary instead of overflowing below the figure.

## 6. How To Read

If `gate_mean` is close to zero but `gate_abs_mean` or `gate_rms` is clearly nonzero, then signed cancellation is happening and the modulation is not weak. It only means that positive and negative channels partially cancel in the signed average.

If `gate_abs_mean`, `gate_rms`, or `gate_max_abs` grows with layer depth, then deeper layers are using stronger channel-wise modulation even if the signed mean remains small.

If `gate_positive_fraction` is close to `0.5` while `gate_mean` is near zero, then the tensor is roughly sign-balanced across channels.

If `gate_positive_fraction` and `gate_negative_fraction` are highly asymmetric, then the modulation is biased toward amplification or suppression in a large fraction of channels.

If `sa_output_rms` is large but `gated_to_raw_rms_ratio` is small, then the raw self-attention branch is active but the residual write is being suppressed by \(e_2\).

If `gated_sa_output_rms` and `gated_to_raw_rms_ratio` are strongest in deeper layers during early denoising steps, then those layers are plausible candidates for cross-frame coordination during motion-planning consolidation.

If one layer has a modest layer-wise `gate_rms` but some heads show very large `gated_to_raw_sa_head_write_rms_ratio`, then the gate is not uniformly large. Instead it is selectively amplifying the channels used by those heads' actual writes.

If one layer has large layer-wise `sa_output_rms` and `gate_rms` but many heads still show small `gated_to_raw_sa_head_write_rms_ratio`, then the layer is active overall but the gate is not preferentially opening on the dominant head-write directions.

If the step curves are smooth and similar across seeds or prompts, that is not automatically suspicious. These tensors are produced from timestep-conditioned modulation pathways and are therefore expected to have a strong deterministic component tied to diffusion time. The layer dimension is the more likely source of structural variation.

## 7. Run Modes

Standard profiling mode runs the full diffusion process, exports the CSV/PT artifacts, and then renders all requested PDFs.

Redraw-only mode is enabled with `--self_attention_modulation_plot_only_from_saved True`.
In this mode, the experiment does not load the Wan2.1 pipeline and does not run sampling.
Instead, it reads the following saved artifacts from `output_dir` and redraws the figures:

- `self_attention_modulation_rows.csv`
- `self_attention_modulation_head_rows.csv`
- `self_attention_modulation_decomposition_rows.csv`
- `self_attention_modulation_channel_profiles.pt`
- `self_attention_modulation_weight_norms.csv`

Under distributed launch, redraw-only mode follows the same rank semantics as the standard profiling mode:
only rank `0` performs the actual plotting work, while the other ranks exit after synchronization.
