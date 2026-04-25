# `attention_dt_profile` Technical Note

## 1. Motivation

`attention_dt_profile` measures how Wan self-attention distributes mass over temporal distance in early denoising. It answers the question:

- for a query token chosen from a selected latent frame,
- how much self-attention mass goes to keys at temporal offset \(|\Delta t|\)?

This is a temporal-distance profile experiment, not a full motion-alignment experiment.

## 2. Patch and Data Collection

The experiment runs one normal generation with a self-attention probe patch enabled. The patch collects self-attention statistics only at selected diffusion steps.

Inputs:

- `probe_steps`: diffusion steps to collect
- `query_frame_count`: number of query frames sampled for the profile
- `query_mode`: how query tokens are chosen
  - `center`: use central spatial positions
  - `object_guided`: use an externally provided object-token trajectory
- `probe_branch`: `cond`, `uncond`, or `both`

## 3. Core Quantity

Let a query token belong to latent frame \(f_q\), and let a key token belong to latent frame \(f_k\). Define temporal offset

\[
\Delta t = f_k - f_q.
\]

When `use_abs_dt=True`, the experiment records

\[
|\Delta t|.
\]

For a selected layer/head/step, let the attention probability from the query token to the key token be \(\alpha(i,j)\). The experiment aggregates attention mass into a histogram over temporal distance:

\[
H(d) = \sum_{(i,j):\, |f(j)-f(i)| = d} \alpha(i,j).
\]

After aggregating over sampled query tokens, the histogram is normalized into a probability distribution over \(|\Delta t|\).

## 4. Outputs

- `attention_dt_histograms.pt`: raw histogram tensors
- `attention_dt_profile_global_dt_curve.pdf`
- `attention_dt_profile_layer_heatmaps/step_XXX.pdf`
- `attention_dt_profile_summary.json`

## 5. Interpretation

If early self-attention is very local in time, the mass of the histogram concentrates near small \(|\Delta t|\). If a layer/head has broader temporal reach, the profile spreads toward larger temporal offsets.

This experiment therefore characterizes temporal receptive range in self-attention, but it does not yet check whether that temporal reach aligns with true motion targets.
