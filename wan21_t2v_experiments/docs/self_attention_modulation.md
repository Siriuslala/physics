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
- `self_attention_modulation_summary.json`
- `self_attention_modulation_plots/e0/...`
- `self_attention_modulation_plots/e1/...`
- `self_attention_modulation_plots/e2/...`

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

## 6. How To Read

If `gate_mean` is close to zero but `gate_abs_mean` or `gate_rms` is clearly nonzero, then signed cancellation is happening and the modulation is not weak. It only means that positive and negative channels partially cancel in the signed average.

If `gate_abs_mean`, `gate_rms`, or `gate_max_abs` grows with layer depth, then deeper layers are using stronger channel-wise modulation even if the signed mean remains small.

If `gate_positive_fraction` is close to `0.5` while `gate_mean` is near zero, then the tensor is roughly sign-balanced across channels.

If `gate_positive_fraction` and `gate_negative_fraction` are highly asymmetric, then the modulation is biased toward amplification or suppression in a large fraction of channels.

If `sa_output_rms` is large but `gated_to_raw_rms_ratio` is small, then the raw self-attention branch is active but the residual write is being suppressed by \(e_2\).

If `gated_sa_output_rms` and `gated_to_raw_rms_ratio` are strongest in deeper layers during early denoising steps, then those layers are plausible candidates for cross-frame coordination during motion-planning consolidation.

If the step curves are smooth and similar across seeds or prompts, that is not automatically suspicious. These tensors are produced from timestep-conditioned modulation pathways and are therefore expected to have a strong deterministic component tied to diffusion time. The layer dimension is the more likely source of structural variation.
