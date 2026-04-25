# `motion_aligned_attention` Technical Note

## 1. Motivation

`motion_aligned_attention` asks a stronger question than `attention_dt_profile`:

- not only how far in time self-attention reaches,
- but whether that attention lands on motion-relevant target regions.

The target can come from:

- `motion_centroid`: a motion trajectory extracted from the generated video itself
- `object_token_trajectory`: a token-space trajectory extracted from `cross_attention_token_viz`

## 2. Query and Target Construction

The experiment reuses the same self-attention probe stack as `attention_dt_profile`, but additionally collects motion-alignment maps.

Let the query token belong to latent frame \(f\). The experiment defines a target token position in the next latent frame \(f+1\):

\[
(y_{f+1}^{\star}, x_{f+1}^{\star}).
\]

This target is obtained either from a projected video-space motion centroid trajectory or from an object-token trajectory supplied by `cross_attention_token_viz` / `joint_attention_suite`.

## 3. Motion-Aligned Attention Score

For a fixed query token, layer, head, and step, let the self-attention map on the next frame be

\[
A_{f \rightarrow f+1}(y,x).
\]

Given a radius \(r\), define the local target neighborhood

\[
\Omega_r(f+1) = \{(y,x): \|(y,x) - (y_{f+1}^{\star}, x_{f+1}^{\star})\|_{\infty} \le r\}.
\]

The local aligned mass is

\[
M_{\text{local}} = \sum_{(y,x) \in \Omega_r(f+1)} A_{f \rightarrow f+1}(y,x).
\]

The experiment records both:

- local mass itself
- local ratio, namely the fraction of the next-frame attention mass that falls inside the target neighborhood

After averaging over sampled queries, it reports MAAS statistics per layer/step.

## 4. Outputs

- `motion_aligned_attention_rows.csv`
- `motion_aligned_attention_dt_histograms.pt`
- `motion_alignment_visualizations/`
- `motion_aligned_attention_summary.json`

## 5. Relation to `attention_dt_profile`

- `attention_dt_profile`: measures temporal distance preference only
- `motion_aligned_attention`: measures whether attention lands on motion-relevant target positions

So the two experiments are complementary:

- one studies temporal reach
- the other studies temporal usefulness
