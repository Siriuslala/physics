# `head_trajectory_dynamics` Technical Note

## 1. Motivation

`head_trajectory_dynamics` reuses saved cross-attention maps from `cross_attention_token_viz` and studies how heads evolve toward a common object-trajectory pattern.

The experiment now separates two kinds of quantities:

1. **map-level distances** between full per-frame 2D attention distributions
2. **center-trajectory distances** between extracted per-frame centers

This separation is important because early denoising maps are often multi-modal or diffuse, so reducing them to a single center can discard useful structure.

## 2. Inputs

For one diffusion step \(s\), layer \(\ell\), and head \(h\), let the object-token mean map be

\[
A^{(s,\ell,h)} \in \mathbb{R}_{\ge 0}^{F \times H \times W}.
\]

Here:

- \(F\): latent-frame count
- \(H, W\): token-grid height and width

The experiment also extracts a per-frame center trajectory

\[
\mathcal{T}^{(s,\ell,h)} = \{c_f^{(s,\ell,h)}\}_{f=1}^{F},
\qquad
c_f^{(s,\ell,h)} \in \mathbb{R}^{2},
\]

using either:

- `region_centroid`
- `preprocessed_component_center`

The detailed center extraction logic is shared with `cross_attention_token_viz` and `head_evolution`.

## 3. Per-frame Spatial Normalization

For map-level distances, each frame is normalized over its spatial support:

\[
P_f(y,x) = \frac{A_f(y,x)}{\sum_{y',x'} A_f(y',x') + \varepsilon}.
\]

So each frame becomes a probability distribution over \(H \times W\).

## 4. Map-Level Distances

### 4.1 Jensen-Shannon distance

For flattened frame distributions \(p_f, q_f \in \mathbb{R}^{HW}\), define

\[
m_f = \frac{1}{2}(p_f + q_f).
\]

Then

\[
D_{\mathrm{JS}}(f) = \sqrt{\frac{1}{2}\operatorname{KL}(p_f\|m_f) + \frac{1}{2}\operatorname{KL}(q_f\|m_f)}.
\]

The stored map-level distance is the frame mean:

\[
\bar D_{\mathrm{JS}} = \frac{1}{F}\sum_{f=1}^{F} D_{\mathrm{JS}}(f).
\]

### 4.2 Hellinger distance

\[
D_{\mathrm{Hell}}(f) = \frac{1}{\sqrt 2}\left\|\sqrt{p_f} - \sqrt{q_f}\right\|_2.
\]

Again the experiment stores the frame mean.

### 4.3 Map-level Wasserstein proxy

Full 2D optimal transport is too expensive to run for all pairwise head comparisons. The current implementation therefore uses an efficient map-level Wasserstein proxy based on row and column marginals.

Let

\[
p_f^y(y) = \sum_x P_f(y,x),
\qquad
p_f^x(x) = \sum_y P_f(y,x),
\]

and similarly for \(q_f^y, q_f^x\). The 1D Wasserstein-1 distance on a unit-spaced grid is computed from cumulative sums:

\[
W_1(p_f^y, q_f^y) = \sum_y \left|\operatorname{CDF}(p_f^y)(y) - \operatorname{CDF}(q_f^y)(y)\right|,
\]

\[
W_1(p_f^x, q_f^x) = \sum_x \left|\operatorname{CDF}(p_f^x)(x) - \operatorname{CDF}(q_f^x)(x)\right|.
\]

The experiment uses

\[
D_{\mathrm{W-map}}(f) = \frac{1}{2}\Big(W_1(p_f^y, q_f^y) + W_1(p_f^x, q_f^x)\Big).
\]

This is a 2D map-level Wasserstein proxy, not full 2D OT.

### 4.4 Support-overlap distance

For each frame, define a high-response support mask by a quantile threshold \(q_{\mathrm{sup}}\):

\[
S_f = \{(y,x): P_f(y,x) \ge Q_{q_{\mathrm{sup}}}(P_f)\}.
\]

The support-overlap IoU is

\[
\operatorname{IoU}(f) = \frac{|S_f^{(1)} \cap S_f^{(2)}|}{|S_f^{(1)} \cup S_f^{(2)}|}.
\]

The experiment records the corresponding distance

\[
D_{\mathrm{sup}}(f) = 1 - \operatorname{IoU}(f).
\]

## 5. Center-Trajectory Distance

For two center trajectories

\[
\mathcal{T}^{(1)} = \{c_f^{(1)}\}_{f=1}^{F},
\qquad
\mathcal{T}^{(2)} = \{c_f^{(2)}\}_{f=1}^{F},
\]

the center distance is

\[
D_{\mathrm{center}} = \frac{1}{F}\sum_{f=1}^{F} \|c_f^{(1)} - c_f^{(2)}\|_2.
\]

This is the current `center_l2` metric. The legacy metric name `wasserstein` is still accepted as an alias for backward compatibility, but it no longer refers to a map-level Wasserstein quantity.

## 6. Consensus

For a fixed `(step, layer)`, the experiment computes all pairwise distances between heads. For one metric, let these distances be \(d_1, \dots, d_K\). The mean pairwise distance is

\[
\bar d = \frac{1}{K} \sum_{k=1}^{K} d_k.
\]

Consensus is defined as

\[
\mathrm{consensus} = \frac{1}{1 + \bar d}.
\]

So smaller pairwise distance implies larger consensus.

## 7. Reference Distance and Convergence Speed

The reference map is built from the head-mean object map at `(reference_step, reference_layer)`. For each head and each analyzed step, the experiment computes its distance to this reference under every selected metric.

From the reference-distance curve \(D_h(s)\), it also derives convergence summaries:

- `reference_distance_auc`: mean distance across selected steps
- `lock_in_step_rho_0p2`
- `lock_in_step_rho_0p5`

These lock-in steps indicate how early a head enters a neighborhood close to its final reference distance.

## 8. Attractor Metrics

Attractor analysis currently operates on center trajectories. For a leader head trajectory at step \(s\), and a follower head, let

\[
d_{\mathrm{current}} = D_{\mathrm{center}}\big(\mathcal{T}^{(s)}_{\mathrm{follower}}, \mathcal{T}^{(s)}_{\mathrm{leader}}\big).
\]

### 8.1 One-step attractor

\[
\Delta^{\mathrm{one}} = d_{\mathrm{current}} - D_{\mathrm{center}}\big(\mathcal{T}^{(s+1)}_{\mathrm{follower}}, \mathcal{T}^{(s)}_{\mathrm{leader}}\big).
\]

### 8.2 Window-mean attractor

For a future window \(s+1, \dots, s+K\), define

\[
\Delta^{\mathrm{mean}} = d_{\mathrm{current}} - \frac{1}{K}\sum_{k=1}^{K} D_{\mathrm{center}}\big(\mathcal{T}^{(s+k)}_{\mathrm{follower}}, \mathcal{T}^{(s)}_{\mathrm{leader}}\big).
\]

### 8.3 Best-future attractor

\[
\Delta^{\mathrm{best}} = d_{\mathrm{current}} - \min_{1 \le k \le K} D_{\mathrm{center}}\big(\mathcal{T}^{(s+k)}_{\mathrm{follower}}, \mathcal{T}^{(s)}_{\mathrm{leader}}\big).
\]

The experiment reports all three methods:

- `one_step`
- `window_mean`
- `best_future`

## 9. Outputs

- `head_trajectory_dynamics_pairwise.csv`
- `head_trajectory_dynamics_consensus.csv`
- `head_trajectory_dynamics_reference_distance.csv`
- `head_trajectory_dynamics_convergence.csv`
- `head_trajectory_dynamics_attractor.csv`
- `head_trajectory_dynamics_trajectory_centers.csv`
- center-overlay PDFs for manual inspection
- consensus / reference-distance / attractor plots
- reference-distance curve PDFs:
  - one plot per `(layer, metric)` with one curve per head
- convergence-summary heatmaps:
  - `reference_distance_auc`
  - `lock_in_step_rho_0p2`
  - `lock_in_step_rho_0p5`

The old spaghetti plot has been removed because early-step center trajectories are often too unstable to be a reliable primary visualization.
