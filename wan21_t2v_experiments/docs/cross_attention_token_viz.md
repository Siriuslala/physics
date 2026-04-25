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

in [utils.py](/home/liyueyan/Interpretability/physics/wan21_t2v_experiments/utils.py#L1570).

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

\[
\hat y_f = \frac{\sum_{(y,x) \in \Omega_f} y \, w_f(y, x)}{\sum_{(y,x) \in \Omega_f} w_f(y, x)},
\qquad
\hat x_f = \frac{\sum_{(y,x) \in \Omega_f} x \, w_f(y, x)}{\sum_{(y,x) \in \Omega_f} w_f(y, x)}.
\]

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

## 6. Interpretation

This experiment's center extractor can be summarized as:

- localize the frame's dominant attention mode,
- reject disconnected high-valued clutter away from the peak,
- then compute a weighted centroid **inside that dominant region**.

Compared with a global soft center, it is much less sensitive to unrelated bright regions such as background or floor highlights, as long as those regions are not connected to the peak region.

## 7. Current Limitation

This method still depends on the peak-containing component. If the peak itself lands on a spurious region, the extracted center will follow that region. So the method is more robust than a global expectation, but it is not immune to all failure cases.
