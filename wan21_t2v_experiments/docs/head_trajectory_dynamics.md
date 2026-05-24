# `head_trajectory_dynamics` Technical Note

## 1. Motivation

`head_trajectory_dynamics` reuses saved cross-attention maps from `cross_attention_token_viz` and studies how object-related cross-attention heads evolve during diffusion.

The experiment is built to answer three classes of questions:

1. Do heads within the same layer become more mutually consistent over diffusion?
2. Do some heads behave as candidate leaders under an attractor-style hypothesis?
3. How early does each head align with a final reference trajectory or reference map?

The experiment deliberately keeps two complementary object representations:

1. a **full per-frame 2D attention distribution**
2. a **per-frame center trajectory**

The first representation preserves multi-modal spatial structure. The second representation compresses one frame into one point and is therefore easier to compare as a trajectory.

## 2. Core Inputs

For one diffusion step \(s\), layer \(\ell\), and head \(h\), let the object-token mean cross-attention map be

\[
A^{(s,\ell,h)} \in \mathbb{R}_{\ge 0}^{F \times H \times W},
\]

where:

- \(F\) is the latent-frame count.
- \(H\) is the token-grid height.
- \(W\) is the token-grid width.

For map-level distances, each frame is normalized spatially:

\[ P_f^{(s,\ell,h)}(y,x) = \frac{A_f^{(s,\ell,h)}(y,x)}{\sum_{y'=1}^{H}\sum_{x'=1}^{W} A_f^{(s,\ell,h)}(y',x') + \varepsilon}, \qquad \varepsilon>0. \]

Hence

\[ P^{(s,\ell,h)} \in \mathbb{R}_{\ge 0}^{F \times H \times W}, \qquad \sum_{y=1}^{H}\sum_{x=1}^{W} P_f^{(s,\ell,h)}(y,x)=1 \quad \forall f. \]

For center-based distances, the experiment extracts a trajectory

\[ \mathcal{T}^{(s,\ell,h)} = \{c_f^{(s,\ell,h)}\}_{f=1}^{F}, \qquad c_f^{(s,\ell,h)} \in \mathbb{R}^{2}, \]

where the two coordinates represent \((y,x)\) in token-grid space.

## 3. Reference Pipeline and Ordinary-Head Pipeline

The experiment maintains two distinct pipelines.

### 3.1 Reference Pipeline

The reference object is not a single head. It is the head-mean object map at a user-selected reference location

\[
(s_{\mathrm{ref}}, \ell_{\mathrm{ref}}).
\]

Let \(N_{\ell_{\mathrm{ref}}}\) be the number of heads at layer \(\ell_{\mathrm{ref}}\). The reference map is

\[ A^{\mathrm{ref}} = \frac{1}{N_{\ell_{\mathrm{ref}}}} \sum_{h=1}^{N_{\ell_{\mathrm{ref}}}} A^{(s_{\mathrm{ref}}, \ell_{\mathrm{ref}}, h)}. \]

Its normalized map is

\[
P^{\mathrm{ref}} \in \mathbb{R}_{\ge 0}^{F \times H \times W}.
\]

The reference trajectory is extracted from \(A^{\mathrm{ref}}\), not from a single head. It is used by:

- `*_reference_distance`
- `center_l2_reference_distance`
- convergence summaries derived from reference-distance curves

The reference pipeline has its own center-extraction parameters:

- `head_trajectory_dynamics_reference_center_method`
- `head_trajectory_dynamics_reference_center_power`
- `head_trajectory_dynamics_reference_center_quantile`
- `head_trajectory_dynamics_reference_preprocessed_center_mode`
- `head_trajectory_dynamics_reference_preprocess_winsorize_quantile`
- `head_trajectory_dynamics_reference_preprocess_despike_quantile`
- `head_trajectory_dynamics_reference_preprocess_min_component_area`

If these values are set to `same_as_head` or to negative fallback sentinels, the ordinary-head configuration is reused.

### 3.2 Ordinary-Head Pipeline

For each analyzed triple \((s,\ell,h)\), the experiment constructs:

1. a normalized probability map \(P^{(s,\ell,h)}\)
2. a center trajectory \(\mathcal{T}^{(s,\ell,h)}\)

The probability map is used by:

- `js`
- `hellinger`
- `wasserstein_map`
- `support_overlap`

The center trajectory is used by:

- `center_l2`
- center-overlay visualization
- leader-follower distance calculations when the selected attractor metric is center-based

The ordinary-head pipeline is controlled by:

- `head_trajectory_dynamics_center_method`
- `head_trajectory_dynamics_center_power`
- `head_trajectory_dynamics_center_quantile`
- `head_trajectory_dynamics_preprocessed_center_mode`
- `head_trajectory_dynamics_preprocess_winsorize_quantile`
- `head_trajectory_dynamics_preprocess_despike_quantile`
- `head_trajectory_dynamics_preprocess_min_component_area`

## 4. Center Extraction

Two center-extraction families are supported.

### 4.1 `region_centroid`

This method reuses the localized region-center logic used by `cross_attention_token_viz`.

For one frame \(A_f(y,x)\):

1. Find the peak location \((y_f^\star, x_f^\star)\).
2. Threshold the frame by a quantile \(q\).
3. Keep only the connected component that contains the peak.
4. Inside that component, apply power weighting with exponent \(p\).
5. Compute the weighted centroid.

If the retained component is denoted by \(\Omega_f\), the weights are

\[ w_f(y,x)=A_f(y,x)^p, \qquad (y,x)\in\Omega_f. \]

The extracted center is

\[ \hat y_f = \frac{\sum_{(y,x)\in\Omega_f} y\,w_f(y,x)}{\sum_{(y,x)\in\Omega_f} w_f(y,x)}, \qquad \hat x_f = \frac{\sum_{(y,x)\in\Omega_f} x\,w_f(y,x)}{\sum_{(y,x)\in\Omega_f} w_f(y,x)}. \]

### 4.2 `preprocessed_component_center`

This method first denoises the raw attention map, then extracts one dominant component from the cleaned map.

For one frame \(A_f(y,x)\), let the preprocessed frame be \(\tilde A_f(y,x)\). The preprocessing consists of:

1. winsorization at `preprocess_winsorize_quantile`
2. high-value masking at `preprocess_despike_quantile`
3. removal of tiny connected components below `preprocess_min_component_area`

After preprocessing, the method again extracts the peak-containing component and returns one of:

- `peak`
- `centroid`
- `geometric_center`

If the retained cleaned component is \(\tilde\Omega_f\), then:

- for `centroid`,

\[ \tilde w_f(y,x)=\tilde A_f(y,x)^p, \qquad (y,x)\in\tilde\Omega_f, \]

\[ \hat y_f = \frac{\sum_{(y,x)\in\tilde\Omega_f} y\,\tilde w_f(y,x)}{\sum_{(y,x)\in\tilde\Omega_f} \tilde w_f(y,x)}, \qquad \hat x_f = \frac{\sum_{(y,x)\in\tilde\Omega_f} x\,\tilde w_f(y,x)}{\sum_{(y,x)\in\tilde\Omega_f} \tilde w_f(y,x)}; \]

- for `geometric_center`,

\[ \hat y_f=\frac{1}{|\tilde\Omega_f|}\sum_{(y,x)\in\tilde\Omega_f} y, \qquad \hat x_f=\frac{1}{|\tilde\Omega_f|}\sum_{(y,x)\in\tilde\Omega_f} x; \]

- for `peak`, the output is the argmax coordinate itself.

## 5. Motion-Planning Region

The experiment supports an additional intermediate object representation called the **motion-planning region**.

For one normalized frame \(P_f(y,x)\), define the raw support set by a quantile threshold \(q\):

\[ S_f = \{(y,x)\in\{1,\dots,H\}\times\{1,\dots,W\}: P_f(y,x)\ge Q_q(P_f)\}, \]

where \(Q_q(P_f)\) is the frame-wise empirical \(q\)-quantile of the flattened frame values.

The support visualization does not use \(S_f\) directly. Instead, it applies 4-neighborhood connected-component filtering and removes components whose area is smaller than a user-specified threshold \(a_{\min}\). The resulting denoised set is

\[
M_f \subseteq S_f.
\]

This document refers to \(M_f\) as the **motion-planning region**.

In practice:

- \(S_f\) is the exact set used by the raw `support_overlap` IoU definition.
- \(M_f\) is the contour-filtered region used by support-overlap contour visualization.
- when the motion-planning-region preprocessing switch is enabled, metrics are computed from maps restricted to \(M_f\), not from the full map.

### 5.1 Why Filtering Requires Renormalization

If the experiment keeps only the motion-planning region, the filtered frame becomes

\[
\bar P_f(y,x)=P_f(y,x)\,\mathbf{1}\{(y,x)\in M_f\}.
\]

However, \(\bar P_f\) is not guaranteed to sum to one. Therefore the experiment renormalizes it:

\[ P_f^{\mathrm{mpr}}(y,x) = \frac{\bar P_f(y,x)}{\sum_{y'=1}^{H}\sum_{x'=1}^{W}\bar P_f(y',x')+\varepsilon}. \]

This renormalization is necessary because:

- `JS` and `Hellinger` are defined on probability distributions.
- the row/column marginal Wasserstein proxy is also intended to compare probability mass distributions.
- a filtered map with reduced total mass would otherwise entangle object-region geometry with trivial mass shrinkage.

If a frame becomes empty after filtering, the implementation falls back to the unfiltered normalized frame for that frame only.

### 5.2 Optional Pre-Metric Filtering

If `head_trajectory_dynamics_use_motion_planning_region_before_metrics=True`, then for every analyzed head the experiment replaces

\[ P^{(s,\ell,h)} \longrightarrow P_{\mathrm{mpr}}^{(s,\ell,h)} \]

before computing:

- pairwise distances
- consensus
- attractor scores
- reference distances
- convergence summaries

For `region_centroid`, the corresponding center trajectories are re-extracted from the filtered map.
For `preprocessed_component_center`, the center extraction itself already includes the denoising stage, so the experiment reuses the original center trajectories and does not build a separate filtered-center cache.

## 6. Map-Level Distance Metrics

Let \(p_f, q_f \in \mathbb{R}_{\ge 0}^{HW}\) denote flattened normalized frame distributions.

### 6.1 Jensen-Shannon Distance

Define

\[
m_f=\frac{1}{2}(p_f+q_f).
\]

Then

\[ D_{\mathrm{JS}}(f) = \sqrt{ \frac{1}{2}\operatorname{KL}(p_f\|m_f) + \frac{1}{2}\operatorname{KL}(q_f\|m_f) }. \]

The stored distance is the frame mean:

\[ \bar D_{\mathrm{JS}} = \frac{1}{F}\sum_{f=1}^{F} D_{\mathrm{JS}}(f). \]

### 6.2 Hellinger Distance

\[ D_{\mathrm{Hell}}(f) = \frac{1}{\sqrt{2}} \left\| \sqrt{p_f}-\sqrt{q_f} \right\|_2. \]

Again the experiment stores the frame mean.

### 6.3 `wasserstein_map`

This metric is a project-specific map-level Wasserstein proxy, not full 2D optimal transport.

For one frame, define row and column marginals:

\[ p_f^y(y)=\sum_{x=1}^{W} P_f(y,x), \qquad p_f^x(x)=\sum_{y=1}^{H} P_f(y,x), \]

and similarly \(q_f^y, q_f^x\).

The 1D Wasserstein-1 distance on the row axis is computed by cumulative sums:

\[ W_1(p_f^y,q_f^y) = \sum_{y=1}^{H} \left| \operatorname{CDF}(p_f^y)(y)-\operatorname{CDF}(q_f^y)(y) \right|. \]

Likewise on the column axis:

\[ W_1(p_f^x,q_f^x) = \sum_{x=1}^{W} \left| \operatorname{CDF}(p_f^x)(x)-\operatorname{CDF}(q_f^x)(x) \right|. \]

The final proxy is

\[ D_{\mathrm{W-map}}(f) = \frac{1}{2}\left[ W_1(p_f^y,q_f^y) + W_1(p_f^x,q_f^x) \right]. \]

There is no claim that `wasserstein_map` is a standard published benchmark name. It is a descriptive project-specific label for this row/column-marginal proxy.

### 6.4 `support_overlap`

For two normalized frame distributions \(P_f\) and \(Q_f\), define raw support sets

\[ S_f^P = \{(y,x):P_f(y,x)\ge Q_q(P_f)\}, \qquad S_f^Q = \{(y,x):Q_f(y,x)\ge Q_q(Q_f)\}. \]

The frame-wise IoU is

\[ \operatorname{IoU}_f = \frac{|S_f^P \cap S_f^Q|}{|S_f^P \cup S_f^Q|}. \]

The stored overlap score is the frame mean \(\frac{1}{F}\sum_f \operatorname{IoU}_f\), and the corresponding distance is

\[
D_{\mathrm{supp}} = 1 - \frac{1}{F}\sum_{f=1}^{F}\operatorname{IoU}_f.
\]

This definition uses the raw quantile support \(S_f\), not the contour-filtered motion-planning region \(M_f\), unless the optional pre-metric filtering step has already replaced the input probability maps.

## 7. Center-Trajectory Distance

For two center trajectories

\[ \mathcal{T}^A=\{c_f^A\}_{f=1}^{F}, \qquad \mathcal{T}^B=\{c_f^B\}_{f=1}^{F}, \qquad c_f^A,c_f^B\in\mathbb{R}^{2}, \]

the frame-wise Euclidean distance is

\[
D_{\mathrm{center}}(f)=\|c_f^A-c_f^B\|_2.
\]

The stored `center_l2` distance is

\[ \bar D_{\mathrm{center}} = \frac{1}{F}\sum_{f=1}^{F} D_{\mathrm{center}}(f). \]

## 8. Consensus

For one fixed \((s,\ell)\), let the analyzed heads be indexed by \(h_1,\dots,h_n\).

For a chosen metric \(d\), define all unordered pairwise distances

\[ d_{ij}^{(s,\ell)} = d\big(h_i,h_j\big), \qquad 1\le i<j\le n. \]

Their mean is

\[ \bar d^{(s,\ell)} = \frac{1}{\binom{n}{2}} \sum_{1\le i<j\le n} d_{ij}^{(s,\ell)}. \]

The consensus score is

\[
C^{(s,\ell)} = \frac{1}{1+\bar d^{(s,\ell)}}.
\]

Larger consensus means smaller mean pairwise distance.

The current implementation computes consensus layer-wise. Heads from different layers are not pooled into one pairwise set.

## 9. Reference Distance and Convergence

For one head \((s,\ell,h)\), the experiment measures its distance to the reference object under each selected metric.

This produces a step-wise curve

\[
r_{s}^{(\ell,h,m)},
\]

where \(m\) denotes the metric name.

The experiment stores:

- the raw reference-distance rows in `head_trajectory_dynamics_reference_distance.csv`
- a convergence summary in `head_trajectory_dynamics_convergence.csv`

The convergence summary is derived from the reference-distance curve. For one head and one metric:

- let \(r_{\mathrm{init}}\) be the first step value
- let \(r_{\mathrm{final}}\) be the last step value
- let \(\Delta=\max(0, r_{\mathrm{init}}-r_{\mathrm{final}})\)

Then the lock-in thresholds are

\[ \tau_{0.2}=r_{\mathrm{final}}+0.2\Delta, \qquad \tau_{0.5}=r_{\mathrm{final}}+0.5\Delta. \]

The reported lock-in step is the earliest analyzed step whose reference distance falls below the chosen threshold.

The summary also stores

\[ \operatorname{AUC} = \frac{1}{T}\sum_{t=1}^{T} r_t, \]

that is, the arithmetic mean of the sampled reference-distance values across analyzed diffusion steps.

In this experiment, the word **convergence** therefore means "how a head approaches the reference object over diffusion", not "how all heads approach one another". The latter behavior is captured by consensus.

## 10. Attractor Metrics

Attractor analysis compares a follower head and a leader head at the same diffusion step.
For a fixed layer \(\ell\), current step \(s\), candidate leader head \(h_L\), candidate follower head \(h_F\), metric \(d\), and future-window length \(K\), define:

1. the current-step distance

\[
d_{\mathrm{cur}} = d\big((s,\ell,h_F), (s,\ell,h_L)\big)
\]

2. the future-step distances

\[
d_{\mathrm{future}}(k) = d\big((s+k,\ell,h_F), (s+k,\ell,h_L)\big), \qquad 1 \le k \le K
\]

The implementation reports three aggregated follower deltas:

- `one_step`:

\[
\Delta_{\mathrm{one}} = d_{\mathrm{cur}} - d_{\mathrm{future}}(1)
\]

- `window_mean`:

\[
\Delta_{\mathrm{mean}} = d_{\mathrm{cur}} - \frac{1}{K}\sum_{k=1}^{K} d_{\mathrm{future}}(k)
\]

- `best_future`:

\[
\Delta_{\mathrm{best}} = d_{\mathrm{cur}} - \min_{1 \le k \le K} d_{\mathrm{future}}(k)
\]

Positive values mean the follower becomes closer to the leader head at later steps under the chosen metric.

The current implementation performs this analysis layer-wise.

## 11. Support Visualization and Motion-Planning Cache

If `head_trajectory_dynamics_support_viz_enable=True`, the experiment renders support-region contour PDFs for selected heads.

The rendered object is the contour-filtered motion-planning region \(M_f\), not the raw support set \(S_f\).

The support contour uses:

- a raw frame-wise support quantile `head_trajectory_dynamics_support_quantile`
- 4-neighborhood connected components
- a minimum component area `head_trajectory_dynamics_support_viz_contour_min_component_area`

The support cache stores one binary motion-planning mask per analyzed \((s,\ell,h)\). To reduce storage, the cache stores sparse positive indices frame by frame, together with the tensor shape, rather than saving dense floating-point maps.

## 12. Output Structure

Let the experiment root be `output_dir`.

Shared outputs remain in the root directory:

- center trajectory cache JSON
- motion-planning-region cache JSON
- center-overlay PDFs
- support contour PDFs

Metric-dependent outputs are written to a dedicated subdirectory:

```text
output_dir/
  head_trajectory_dynamics_head_center_overlays_motion_planning_region_<on|off>_preprocessed_<on|off>_center_mode_<peak|centroid|geometric_center>/
  head_trajectory_dynamics_support_overlap_masks/
  head_trajectory_dynamics_trajectory_cache_*.json
  head_trajectory_dynamics_motion_planning_region_cache_*.json
  head_trajectory_dynamics_metrics_hypothesis_<name>_motion_planning_region_<on|off>_preprocessed_<on|off>_center_mode_<peak|centroid|geometric_center>/
```

The metric subdirectory contains:

- `head_trajectory_dynamics_head_maps.csv`
- `head_trajectory_dynamics_pairwise.csv`
- `head_trajectory_dynamics_consensus.csv`
- `head_trajectory_dynamics_attractor.csv`
- `head_trajectory_dynamics_reference_distance.csv`
- `head_trajectory_dynamics_convergence.csv`
- `head_trajectory_dynamics_trajectory_centers.csv`
- `head_trajectory_dynamics_soft_centers.csv`
- `head_trajectory_dynamics_plots/`
- `head_trajectory_dynamics_summary.json`

This layout is designed so that future hypothesis variants such as `reference-leading` or `cross-layer-transmission` can share the same common overlays and caches while keeping their metric outputs separate.

## 13. Key Parameters

### 13.1 Hypothesis and Metric Preprocessing

- `head_trajectory_dynamics_hypothesis`
  - hypothesis label used to name the metric-output subdirectory
- `head_trajectory_dynamics_use_motion_planning_region_before_metrics`
  - whether to filter every map by the motion-planning region before metric computation

### 13.2 Reference Selection

- `head_trajectory_dynamics_reference_step`
- `head_trajectory_dynamics_reference_layer`

These specify the step/layer location from which the reference head-mean object map is built.

### 13.3 Support and Motion-Planning Region

- `head_trajectory_dynamics_support_quantile`
  - frame-wise support quantile
- `head_trajectory_dynamics_support_viz_contour_min_component_area`
  - minimum 4-connected component area used by contour filtering and motion-planning-region masking

### 13.4 Center Extraction

- `head_trajectory_dynamics_center_method`
- `head_trajectory_dynamics_center_power`
- `head_trajectory_dynamics_center_quantile`
- `head_trajectory_dynamics_preprocessed_center_mode`
- `head_trajectory_dynamics_preprocess_winsorize_quantile`
- `head_trajectory_dynamics_preprocess_despike_quantile`
- `head_trajectory_dynamics_preprocess_min_component_area`

### 13.5 Reference-Only Center Extraction

- `head_trajectory_dynamics_reference_center_method`
- `head_trajectory_dynamics_reference_center_power`
- `head_trajectory_dynamics_reference_center_quantile`
- `head_trajectory_dynamics_reference_preprocessed_center_mode`
- `head_trajectory_dynamics_reference_preprocess_winsorize_quantile`
- `head_trajectory_dynamics_reference_preprocess_despike_quantile`
- `head_trajectory_dynamics_reference_preprocess_min_component_area`

### 13.6 Overlay Controls

- `head_trajectory_dynamics_center_viz_enable`
- `head_trajectory_dynamics_center_viz_step`
- `head_trajectory_dynamics_center_viz_layer`
- `head_trajectory_dynamics_center_viz_heads`
- `head_trajectory_dynamics_center_viz_num_frames`

- `head_trajectory_dynamics_support_viz_enable`
- `head_trajectory_dynamics_support_viz_step`
- `head_trajectory_dynamics_support_viz_layer`
- `head_trajectory_dynamics_support_viz_heads`
- `head_trajectory_dynamics_support_viz_num_frames`

### 13.7 Reuse Modes

- `head_trajectory_dynamics_plot_only_from_csv`
  - redraw plots only from an existing metric subdirectory
- `head_trajectory_dynamics_overlay_only`
  - redraw overlays only from reused cross-attention maps and caches
- `head_trajectory_dynamics_skip_existing_plots`
  - skip existing plot files instead of overwriting them

## 14. Reading the Outputs

- `consensus`
  - larger means heads within the same layer-step group are more mutually similar
- `reference_distance`
  - measures how far one head is from the final reference object
- `convergence`
  - summarizes the reference-distance curve by AUC and lock-in thresholds
- `attractor_score_mean`
  - positive means followers become closer to the current leader prototype under the chosen metric

When the motion-planning-region preprocessing switch is enabled, all of the map-based outputs above should be interpreted as distances inside a denoised object-focused spatial support rather than over the full attention map.

因此它更像是“基于二维 map 行/列边缘的 Wasserstein proxy”，而不是某个通用标准 OT 求解器的别名。

### 6.4 Support-overlap distance

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

这里用到的唯一参数是：

- `head_trajectory_dynamics_support_quantile`
  - 控制“高响应支撑区域”的分位阈值
  - 它只影响 `support_overlap` 指标
  - 不参与 center extraction

## 7. Center-Trajectory Distance

For two center trajectories

\[ \mathcal{T}^{(1)} = \{c_f^{(1)}\}_{f=1}^{F}, \qquad \mathcal{T}^{(2)} = \{c_f^{(2)}\}_{f=1}^{F}, \]

the center distance is

\[
D_{\mathrm{center}} = \frac{1}{F}\sum_{f=1}^{F} \|c_f^{(1)} - c_f^{(2)}\|_2.
\]

This is the current `center_l2` metric. The legacy metric name `wasserstein` is still accepted as an alias for backward compatibility, but it no longer refers to a map-level Wasserstein quantity.

## 8. Consensus

当前实现里，`consensus` 是 **逐层** 计算的，不会把不同 layer 的 heads 混在一起。

For a fixed `(step, layer)`, the experiment computes all pairwise distances between heads in that same layer only. For one metric, let these distances be \(d_1, \dots, d_K\). The mean pairwise distance is

\[
\bar d = \frac{1}{K} \sum_{k=1}^{K} d_k.
\]

Consensus is defined as

\[
\mathrm{consensus} = \frac{1}{1 + \bar d}.
\]

So smaller pairwise distance implies larger consensus.

## 9. Reference Distance and Convergence Speed

The reference map is built from the head-mean object map at `(reference_step, reference_layer)`. For each head and each analyzed step, the experiment computes its distance to this reference under every selected metric.

From the reference-distance curve \(D_h(s)\), it also derives convergence summaries:

- `reference_distance_auc`: mean distance across selected steps
- `lock_in_step_rho_0p2`
- `lock_in_step_rho_0p5`

These lock-in steps indicate how early a head enters a neighborhood close to its final reference distance.

这里的“final”是该 head 在所分析 step 列表最后一个 step 上相对 reference 的距离，不是模型训练意义上的全局 final state。

更具体地说，若该 head 的初始和最终 reference distance 分别为

\[
D_h(s_{\mathrm{init}}), \qquad D_h(s_{\mathrm{final}}),
\]

则定义距离收缩量

\[
\Delta_h = D_h(s_{\mathrm{init}}) - D_h(s_{\mathrm{final}}).
\]

`lock_in_step_rho_0p2` 取的是最早满足

\[
D_h(s) \le D_h(s_{\mathrm{final}}) + 0.2 \Delta_h
\]

的 step。`lock_in_step_rho_0p5` 同理，只是把 \(0.2\) 改成 \(0.5\)。

因此：

- `reference_distance_auc` 是整条 reference-distance 曲线的面积型摘要
- `lock_in_step` 是“多早进入最终邻域”的时间型摘要

这里要强调一件事：`convergence` 不是一个新的原始观测量，而是从 head-wise `reference_distance` 曲线

\[
D_h(s)
\]

导出的摘要量。因此二者的关系是：

- `reference_distance`：原始的逐 step 曲线
- `convergence`：对这条曲线做的低维摘要

所以二者相关，但不重复。

这里要明确一点：`convergence` 只保留 CSV 摘要，不再单独绘制 curves。

原因是：

- `reference_distance` 已经是原始的逐 step 曲线
- 所谓 `convergence curves` 本质上只是对这同一批 `reference_distance` 数据的重复呈现或重标度呈现

因此，如果你的问题是“哪个 head 先接近 reference”，直接看：

- `head_trajectory_dynamics_plots/reference_distance_curves/<metric>/reference_distance_layer_XX.pdf`

就足够了。

该实验的可视化部分保留 `reference_distance` multi-head curves，不包含 `reference_distance` heatmaps、`convergence` heatmaps 或单独的 `convergence curves`。

## 10. Attractor Metrics

Attractor analysis uses a selectable distance metric. For a leader head at step \(s\), and a follower head, define the current-step distance

\[
d_{\mathrm{current}} = D\big(\mathcal{H}^{(s)}_{\mathrm{follower}}, \mathcal{H}^{(s)}_{\mathrm{leader}}\big),
\]

where \(D\) is controlled by

- `head_trajectory_dynamics_attractor_distance_metric`
  - `center_l2`: compare center trajectories
  - `js`: compare frame-wise probability maps with Jensen-Shannon distance
  - `hellinger`: compare frame-wise probability maps with Hellinger distance
  - `wasserstein_map`: compare frame-wise probability maps with the axis-marginal Wasserstein proxy
  - `support_overlap`: compare frame-wise support masks with distance \(1-\mathrm{IoU}\)

如果当前设置为默认的 `center_l2`，那么上式退化为：

\[
d_{\mathrm{current}} = D_{\mathrm{center}}\big(\mathcal{T}^{(s)}_{\mathrm{follower}}, \mathcal{T}^{(s)}_{\mathrm{leader}}\big).
\]

当前实现中，attractor analysis 也是 **逐层** 的，而不是跨层混算：

- leader head 和 follower heads 必须来自同一个 `(step, layer)`
- follower 的 future states 也只在同一 layer 的 future steps 中比较

所以目前代码的实际假设是：同层 heads 的时序演化更可比，不直接把不同层的 heads 放进同一个 attractor pool。

未来步上的距离也必须使用同一个 future step 的 follower 和 leader：

\[ d_{\mathrm{future}}(k) = D\big(\mathcal{H}^{(s+k)}_{\mathrm{follower}}, \mathcal{H}^{(s+k)}_{\mathrm{leader}}\big), \qquad 1 \le k \le K. \]

### 8.1 One-step attractor

\[
\Delta^{\mathrm{one}} = d_{\mathrm{current}} - d_{\mathrm{future}}(1).
\]

### 8.2 Window-mean attractor

For a future window \(s+1, \dots, s+K\), define

\[
\Delta^{\mathrm{mean}} = d_{\mathrm{current}} - \frac{1}{K}\sum_{k=1}^{K} d_{\mathrm{future}}(k).
\]

### 8.3 Best-future attractor

\[
\Delta^{\mathrm{best}} = d_{\mathrm{current}} - \min_{1 \le k \le K} d_{\mathrm{future}}(k).
\]

The experiment reports all three methods:

- `one_step`
- `window_mean`
- `best_future`

这里唯一直接控制未来窗口长度的参数是：

- `head_trajectory_dynamics_attractor_window`
  - 若设为 `K`，则在 step \(s\) 处，只向未来看 \(s+1,\dots,s+K\)
  - `one_step` 只用其中第一个未来 step
  - `window_mean` 和 `best_future` 会用整个窗口

- `head_trajectory_dynamics_attractor_distance_metric`
  - 这个参数虽然名字是单数，但接受 CSV 多值
  - 例如：
    - `center_l2`
    - `center_l2,js`
    - `js,hellinger,wasserstein_map`
  - 空字符串表示使用所有支持的 attractor 距离：
    - `js`
    - `hellinger`
    - `wasserstein_map`
    - `support_overlap`
    - `center_l2`
  - 它指定上面公式里的 \(D\)
  - 代码里不再是“默认偷偷用 center trajectory distance”

## 11. Parameter Map

下面把 bash 脚本中你提到的参数逐个对应到处理流程。

- `HEAD_TRAJECTORY_DYNAMICS_SUPPORT_QUANTILE=0.9`
  - 只用于 `support_overlap` 指标
  - 含义：取每帧 probability map 的 top 10% 高响应区域来做 IoU
  - 精确定义是：

\[
S_f = \{(y,x): P_f(y,x) \ge Q_{q_{\mathrm{sup}}}(P_f)\}
\]

  - support-mask 可视化对应的就是这个二值区域，而不是别的预处理版本

- `HEAD_TRAJECTORY_DYNAMICS_ATTRACTOR_WINDOW=3`
  - 只用于 attractor analysis
  - 含义：对 step \(s\) 的 leader，查看 follower 在未来 3 个 steps 内是否更靠近它

- `HEAD_TRAJECTORY_DYNAMICS_ATTRACTOR_DISTANCE_METRIC=""`
  - 只用于 attractor analysis
  - 含义：决定“更靠近”是按哪种距离定义
  - 这个变量接受 CSV 多值，例如：
    - `HEAD_TRAJECTORY_DYNAMICS_ATTRACTOR_DISTANCE_METRIC="center_l2,js"`
  - 空字符串表示所有支持的 attractor 距离都算一遍
  - `center_l2` 比较 center trajectory
  - `js/hellinger/wasserstein_map/support_overlap` 比较 probability map 级别的距离

- `HEAD_TRAJECTORY_DYNAMICS_PLOT_ONLY_FROM_CSV`
  - 若为 `True`，不再重新做整套离线分析
  - 直接复用 `output_dir` 下已有的 CSV，重新绘制 plots
  - 这适合只改了可视化代码、但不想重跑分析的情况
  - 但它只能重绘依赖 CSV 的曲线图，不能重建需要原始 probability map 的 center overlay 和 support-mask 可视化

- `HEAD_TRAJECTORY_DYNAMICS_SKIP_EXISTING_PLOTS`
  - 若为 `True`，发现目标 plot 路径已经存在时直接跳过，不覆盖旧图

- `HEAD_TRAJECTORY_DYNAMICS_CENTER_METHOD`
  - 选择 center extraction 主方法
  - 合法值只有：
    - `region_centroid`
    - `preprocessed_component_center`

- `HEAD_TRAJECTORY_DYNAMICS_CENTER_POWER=1.5`
  - 主连通域内的 attention 幂次加权指数
  - 数值越大，越偏向高亮峰值附近

- `HEAD_TRAJECTORY_DYNAMICS_CENTER_QUANTILE=0.8`
  - 提取主连通域时的分位阈值
  - 例如 `0.8` 表示保留每帧 top 20% 左右的高值区域，再取峰值所在主连通域

- `HEAD_TRAJECTORY_DYNAMICS_PREPROCESSED_CENTER_MODE="geometric_center"`
  - 只在 `center_method=preprocessed_component_center` 时生效
  - 表示最后取主连通域的几何中心

- `HEAD_TRAJECTORY_DYNAMICS_PREPROCESS_WINSORIZE_QUANTILE=0.995`
  - 只在 `preprocessed_component_center` 时生效
  - 表示先把每帧极端高值裁到 99.5% 分位点，压制 attention sink

- `HEAD_TRAJECTORY_DYNAMICS_PREPROCESS_DESPIKE_QUANTILE=0.98`
  - 只在 `preprocessed_component_center` 时生效
  - 表示先取每帧 top 2% 左右的高值掩码，再做连通域筛选

- `HEAD_TRAJECTORY_DYNAMICS_PREPROCESS_MIN_COMPONENT_AREA=2`
  - 只在 `preprocessed_component_center` 时生效
  - 表示面积小于 2 个 token 的小尖刺连通域会被移除

- `HEAD_TRAJECTORY_DYNAMICS_REFERENCE_CENTER_METHOD`
  - reference trajectory 专用中心提取方法
  - `same_as_head` 表示直接复用普通 head 的中心提取方法

- `HEAD_TRAJECTORY_DYNAMICS_REFERENCE_CENTER_POWER`
- `HEAD_TRAJECTORY_DYNAMICS_REFERENCE_CENTER_QUANTILE`
- `HEAD_TRAJECTORY_DYNAMICS_REFERENCE_PREPROCESSED_CENTER_MODE`
- `HEAD_TRAJECTORY_DYNAMICS_REFERENCE_PREPROCESS_WINSORIZE_QUANTILE`
- `HEAD_TRAJECTORY_DYNAMICS_REFERENCE_PREPROCESS_DESPIKE_QUANTILE`
- `HEAD_TRAJECTORY_DYNAMICS_REFERENCE_PREPROCESS_MIN_COMPONENT_AREA`
  - 这些都是 reference trajectory 专用参数
  - 若设为默认的负数 / `same_as_head`，就回退到普通 head 的对应参数

## 12. Outputs

- `head_trajectory_dynamics_pairwise.csv`
- `head_trajectory_dynamics_consensus.csv`
- `head_trajectory_dynamics_reference_distance.csv`
- `head_trajectory_dynamics_convergence.csv`
- `head_trajectory_dynamics_attractor.csv`
- `head_trajectory_dynamics_trajectory_centers.csv`
- center-overlay PDFs for manual inspection
- support-overlap binary-mask PDFs for manual inspection
- consensus / reference-distance / attractor plots
- reference-distance curve PDFs:
  - one plot per `(layer, metric)` with one curve per head
- attractor curve PDFs:
  - `head_trajectory_dynamics_plots/attractor_curves/<metric>/<method>/attractor_layer_XX.pdf`
    - 每层一张图
  - `head_trajectory_dynamics_plots/attractor_curves/<metric>/<method>/attractor_all_heads.pdf`
    - 所有分析 head 放在一张图里，共享同一纵轴范围，便于跨层直接比较谁更 attractive
- support-overlap mask PDFs:
  - `head_trajectory_dynamics_support_overlap_masks/step_xxx/layer_xx/support_mask_step_xxx_layer_xx_head_xx.pdf`
    - 黑白二值图，显示 `support_overlap` 指标实际使用的保留区域
  - `head_trajectory_dynamics_support_overlap_masks/step_xxx/layer_xx/support_mask_contour_step_xxx_layer_xx_head_xx.pdf`
    - 在黑白二值图基础上，额外绘制绿色轮廓线
    - 仅对面积不小于 `head_trajectory_dynamics_support_viz_contour_min_component_area` 的连通域绘制轮廓，避免对零散单点画轮廓

仅从 CSV 重绘时，以上这些 metric plots 都可以恢复，因为它们都只依赖这些 CSV 数值表。

但 `center overlay` 和 `support-overlap mask` 这类需要原始 probability map 的图，不属于“只靠指标 CSV 就能恢复”的部分。
