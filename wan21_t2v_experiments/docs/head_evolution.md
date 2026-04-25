# `head_evolution` Technical Note

## 1. Motivation

`head_evolution` studies how cross-attention head patterns evolve during denoising. It does not try to reconstruct one head's trajectory directly and compare it to a target trajectory. Instead, it first builds a reference support tube from a reference object map, and then scores every step/layer/head map by how concentrated it is and how much of its probability mass lies inside that support tube.

Because of this design, the center extraction used in `head_evolution` is asymmetric:

- the reference trajectory is extracted carefully with preprocessing;
- the ordinary head maps are not assigned their own explicit center trajectories during scoring;
- instead they are measured against the fixed support mask built from the reference trajectory.

## 2. Input / Output

### Inputs

- Reused cross-attention maps from `cross_attention_token_viz`.
- Reference location:
  - `head_evolution_reference_step`
  - `head_evolution_reference_layer`
- Reference center choice:
  - `head_evolution_center_mode \in \{\texttt{peak}, \texttt{centroid}, \texttt{geometric_center}\}`
- Reference-support radius parameters:
  - `head_evolution_support_radius_mode`
  - `head_evolution_support_radius_fixed`
  - `head_evolution_support_radius_alpha`
  - `head_evolution_support_radius_min`
  - `head_evolution_support_radius_max_ratio`
- Preprocessing parameters:
  - `head_evolution_preprocess_winsorize_quantile`
  - `head_evolution_preprocess_despike_quantile`
  - `head_evolution_preprocess_min_component_area`
- Metric switch:
  - `head_evolution_apply_preprocess_on_metrics`

### Outputs

- reference trajectory CSV
- support mask tensor
- step-wise / layer-wise / head-wise metric CSVs
- score CSVs and plots

## 3. Reference Map Construction

Let

\[
A^{\mathrm{ref}} \in \mathbb{R}_{\ge 0}^{F \times H \times W}
\]

be the reference object map. In code this is built as the head-mean object-token map at the selected reference step and layer:

\[
A^{\mathrm{ref}}_f(y,x) = \frac{1}{N} \sum_{h=1}^{N} A^{\mathrm{ref},h}_f(y,x),
\]

where:

- \(N\) is the number of attention heads in that layer,
- \(A^{\mathrm{ref},h}\) is the object-word mean map of head \(h\).

## 4. Preprocessing of the Reference Map

The reference map is preprocessed by `_preprocess_wan21_t2v_attention_map_fhw` before any center is extracted.

### Step 1. Winsorization

For each frame \(f\), let the flattened values be \(\{A_f(i)\}_{i=1}^{HW}\). Given quantile \(q_{\mathrm{win}}\), define

\[
\tau_f^{\mathrm{win}} = \operatorname{Quantile}(\{A_f(i)\}, q_{\mathrm{win}}).
\]

Then clip the frame:

\[
\tilde A_f(y,x) = \min(A_f(y,x), \tau_f^{\mathrm{win}}).
\]

This suppresses extremely large outliers without changing most values.

### Step 2. High-value mask for despiking

Given quantile \(q_{\mathrm{despike}}\), compute

\[
\tau_f^{\mathrm{despike}} = \operatorname{Quantile}(\{\tilde A_f(i)\}, q_{\mathrm{despike}}).
\]

Construct a binary mask

\[
M_f(y,x) = \mathbf{1}[\tilde A_f(y,x) \ge \tau_f^{\mathrm{despike}}].
\]

### Step 3. Remove tiny connected components

Extract all 8-neighborhood connected components of \(M_f\). Keep only components whose area is at least

\[
a_{\min} = \texttt{head\_evolution\_preprocess\_min\_component\_area}.
\]

All pixels belonging to smaller components are zeroed out in the attention map.

This removes isolated spike-like bright points.

## 5. Reference Center Extraction

After preprocessing, the code extracts three candidate center trajectories by `_extract_wan21_t2v_reference_peak_and_centroid_trajectory`.

For each frame \(f\):

### 5.1 Peak center

\[
(y_f^{\mathrm{peak}}, x_f^{\mathrm{peak}}) = \arg\max_{y,x} \tilde A_f(y,x).
\]

### 5.2 Peak-containing component

Using quantile threshold `head_evolution_traj_quantile = q`, form

\[
M_f^{\mathrm{traj}}(y,x) = \mathbf{1}[\tilde A_f(y,x) \ge \tau_f^{\mathrm{traj}}],
\qquad
\tau_f^{\mathrm{traj}} = \operatorname{Quantile}(\{\tilde A_f(y,x)\}_{y,x}, q).
\]

Then keep only the connected component \(\Omega_f\) that contains the peak.

### 5.3 Weighted centroid inside the component

With sharpening exponent `head_evolution_traj_power = p`, define

\[
w_f(y,x) = \tilde A_f(y,x)^p, \qquad (y,x) \in \Omega_f.
\]

Then the centroid is

\[
(y_f^{\mathrm{centroid}}, x_f^{\mathrm{centroid}})
=
\left(
\frac{\sum_{(y,x)\in\Omega_f} y \, w_f(y,x)}{\sum_{(y,x)\in\Omega_f} w_f(y,x)},
\frac{\sum_{(y,x)\in\Omega_f} x \, w_f(y,x)}{\sum_{(y,x)\in\Omega_f} w_f(y,x)}
\right).
\]

### 5.4 Geometric center of the component

If \(|\Omega_f|\) is the number of pixels in the component, then

\[
(y_f^{\mathrm{geom}}, x_f^{\mathrm{geom}})
=
\left(
\frac{1}{|\Omega_f|} \sum_{(y,x)\in\Omega_f} y,
\frac{1}{|\Omega_f|} \sum_{(y,x)\in\Omega_f} x
\right).
\]

### 5.5 Final reference trajectory choice

The experiment chooses one of the above three trajectories according to

- `peak`
- `centroid`
- `geometric_center`

This selected sequence is the reference center trajectory:

\[
\mathcal{T}^{\mathrm{ref}} = \{(\hat y_f^{\mathrm{ref}}, \hat x_f^{\mathrm{ref}})\}_{f=1}^{F}.
\]

## 6. Support Mask Construction

The experiment does not compare each head to the reference by point-to-point center distance. Instead it builds a frame-wise circular support mask around the reference center.

For each frame \(f\), let the chosen reference center be \((\hat y_f^{\mathrm{ref}}, \hat x_f^{\mathrm{ref}})\).

### 6.1 Fixed-radius mode

If `support_radius_mode = fixed`, then

\[
r_f = r_{\mathrm{fixed}}.
\]

### 6.2 Adaptive-area mode

If `support_radius_mode = adaptive_area`, let \(a_f\) be the area of the peak-containing reference component. Define its equivalent-circle radius

\[
r_f^{\mathrm{eq}} = \sqrt{\frac{a_f}{\pi}}.
\]

Then the actual radius is

\[
r_f = \alpha \, r_f^{\mathrm{eq}},
\]

where \(\alpha = \texttt{head\_evolution\_support\_radius\_alpha}\), then clipped by the configured minimum and maximum bounds.

### 6.3 Binary support mask

The frame-wise support mask is

\[
S_f(y,x) = \mathbf{1}\left[(y-\hat y_f^{\mathrm{ref}})^2 + (x-\hat x_f^{\mathrm{ref}})^2 \le r_f^2\right].
\]

All frames together give

\[
S \in \{0,1\}^{F \times H \times W}.
\]

## 7. How Ordinary Head Maps Are Scored

This is the crucial point for comparison with `head_trajectory_dynamics`:

`head_evolution` does not extract a separate explicit center trajectory for every ordinary head map during metric computation.

Instead, for every analyzed map

\[
A \in \mathbb{R}_{\ge 0}^{F \times H \times W},
\]

the experiment computes metrics against the fixed reference support mask \(S\).

### Optional preprocessing on ordinary maps

If `head_evolution_apply_preprocess_on_metrics = true`, the same winsorize + despike preprocessing is applied to the current map before metric computation.

If it is `false`, the map is only clamped to nonnegative values and no outlier-oriented connected-component logic is used.

### Support-quality metric

Frame-wise normalized probability:

\[
P_f(y,x) = \frac{A_f(y,x)}{\sum_{y',x'} A_f(y',x') + \varepsilon}.
\]

Frame-level support quality:

\[
Q_f^{\mathrm{frame}} = \sum_{y,x} P_f(y,x) S_f(y,x).
\]

Then average over frames:

\[
Q^{\mathrm{frame}} = \frac{1}{F} \sum_{f=1}^{F} Q_f^{\mathrm{frame}}.
\]

Video-wise support quality is also computed after normalizing over the full spatiotemporal support \(FHW\).

## 8. Interpretation

So the center-related asymmetry in `head_evolution` is:

- Reference trajectory: carefully extracted after preprocessing and component analysis.
- Ordinary head maps: usually not assigned their own trajectory centers; instead they are evaluated by whether their mass falls into the reference support region.

This design is good for asking:

- does this head support the reference trajectory?
- does this head become concentrated around that trajectory early or late?

But it is not designed for asking:

- what exact center trajectory does this head itself follow?

That second question is exactly why `head_trajectory_dynamics` currently feels too weak: it tries to answer that second question, but with a much simpler global soft-center extractor.
