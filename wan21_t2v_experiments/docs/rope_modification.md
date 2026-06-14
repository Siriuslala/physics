# `rope_modification` Technical Note

## 1. Goal

`rope_modification` studies how to modify Wan2.1 T2V self-attention RoPE without editing the official model source code.
All changes are applied through runtime monkey patching.

The experiment currently contains three RoPE-side schemes:

1. `manual`: a training-free scheme with manually chosen axis-wise scales.
2. `spatial_temporal_reweight`: a training-free scheme that reweights temporal and spatial channel groups after RoPE and before attention.
3. `timestep_conditioned`: a training-oriented scale-learning scheme with `global` and `head_aware` modes.

Both schemes preserve one joint attention softmax over the full token sequence.
This experiment does **not** split attention into separate spatial and temporal softmax branches.

## 2. Current Wan2.1 RoPE

Let

\[
X \in \mathbb{R}^{B \times L \times C}
\]

be the patchified video token tensor, where:

- \(B\) is the batch size.
- \(L\) is the padded token length.
- \(C\) is the model width.

For one self-attention layer, the model computes

\[
Q, K, V \in \mathbb{R}^{B \times L \times N \times d},
\]

where:

- \(N\) is the number of heads.
- \(d = C / N\) is the per-head dimension.

For token \(i\), its latent coordinate is

\[
\mathbf{p}_i = (f_i, h_i, w_i),
\]

where:

- \(f_i\) is the latent frame index,
- \(h_i\) is the latent height index,
- \(w_i\) is the latent width index.

Wan2.1 splits one head into three consecutive RoPE channel blocks:

- one block for the frame axis,
- one block for the height axis,
- one block for the width axis.

For `t2v-1.3B`, \(d = 128\), so the real channel split is \(44/42/42\) across \((f, h, w)\).
The implementation is in [wan/modules/model.py](</home/liyueyan/Interpretability/physics/projects/Wan2_1/wan/modules/model.py:42>) and [wan/modules/model.py](</home/liyueyan/Interpretability/physics/projects/Wan2_1/wan/modules/model.py:477>).

The attention logit for token pair \((i, j)\), layer \(\ell\), and head \(m\) is

\[
s_{ij}^{(\ell, m)} = \frac{\langle \widetilde q_i^{(\ell, m)}, \widetilde k_j^{(\ell, m)} \rangle}{\sqrt d},
\]

where:

- \(\widetilde q_i^{(\ell, m)}\) is the RoPE-rotated query,
- \(\widetilde k_j^{(\ell, m)}\) is the RoPE-rotated key.

The key point is that the three axes are still combined inside one dot product and one softmax.

## 3. Notation for the Modification

This note uses the following notation:

- \(b_f, b_h, b_w\): the RoPE base values for frame, height, and width.
- \(\theta_{a,r}\): the angular frequency of axis \(a\) and complex-pair index \(r\).
- \(\lambda_f, \lambda_h, \lambda_w\): the axis-wise RoPE scales.

For axis \(a \in \{f, h, w\}\), let \(d_a\) be the number of real channels assigned to that axis and let

\[
m_a = d_a / 2
\]

be the number of complex pairs on that axis.

Then the angular frequency of the \(r\)-th complex pair is

\[ \theta_{a,r} = b_a^{-2r/d_a}, \qquad r = 0, 1, \dots, m_a - 1. \]

So:

- \(b_a\) is the base,
- \(\theta_{a,r}\) is the angular frequency generated from that base.

This is the notation used in the rest of the note.

## 4. Scheme 1: Manual Axis-Wise Scaling

### 4.1 Motivation

The first scheme is the simplest one and should be implemented first.

The motivation is not that the temporal axis must always be much longer than the spatial axes.
Instead, the more stable argument is the following:

- video is anisotropic data, and the temporal axis and the spatial axes correspond to different physical constraints, so the three axes do not have to share exactly the same RoPE phase-growth speed,
- reducing \(\lambda_f\) is mainly a way to relax the temporal phase constraint when cross-frame token matching is needed, while keeping \(\lambda_h\) and \(\lambda_w\) unchanged or larger is mainly a way to preserve spatial locality and spatial sharpness.

Under this view, Scheme 1 does not assume that temporal decay is necessarily the only bottleneck.
It simply introduces a controlled axis-wise inductive bias.

The implementation idea is:

- keep the original joint attention softmax,
- keep the original head-channel split,
- only change how fast the RoPE phase grows along each axis.

The manual scale parameters are

\[
\lambda_f > 0,\qquad \lambda_h > 0,\qquad \lambda_w > 0.
\]

They are set by hand and are not learned.

### 4.2 Phase definition

For token \(i\), axis \(a\), and complex-pair index \(r\), define the RoPE phase as

\[
\phi_{i,r}^{(a)} = \lambda_a \, p_i^{(a)} \, \theta_{a,r},
\]

where:

- \(p_i^{(a)}\) is the coordinate value of token \(i\) on axis \(a\),
- \(\theta_{a,r}\) is the angular frequency defined above,
- \(\lambda_a\) is the axis-wise scale.

The role of \(\lambda_a\) is:

- larger \(\lambda_a\) means faster phase growth and stronger effective decay,
- smaller \(\lambda_a\) means slower phase growth and longer-range coupling.

So if we want slower temporal decay than spatial decay, a natural direction is

\[
\lambda_f < \lambda_h,\qquad \lambda_f < \lambda_w.
\]

In the current code, this is implemented by rescaling the RoPE complex phase directly in [wan21_t2v_experiment_patch.py](</home/liyueyan/Interpretability/physics/wan21_t2v_experiments/wan21_t2v_experiment_patch.py:745>).

### 4.3 Rotation and attention

After the phase is defined, the query and key are rotated in the standard RoPE form.
For one real pair \((x_1, x_2)\), the rotated pair is

\[ \begin{bmatrix} x_1' \\ x_2' \end{bmatrix} = \begin{bmatrix} \cos \phi & -\sin \phi \\ \sin \phi & \cos \phi \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}. \]

Then the model still computes

\[
A_{ij}^{(\ell, m)} = \mathrm{softmax}_j \left( s_{ij}^{(\ell, m)} \right).
\]

So Scheme 1 changes only the RoPE phase, not the attention operator itself.

## 5. Diffusion-Step Gating

Scheme 1 also supports a diffusion-step window.
Let

\[
S \subseteq \{1, 2, \dots, K\}
\]

be the set of diffusion steps on which the modification is active, where \(K\) is the total number of denoising steps.

If \(S\) is empty, the modification is applied to all diffusion steps.
If \(S\) is non-empty, then the effective scale is

\[ \lambda_a^{\text{used}}(t) = \begin{cases} \lambda_a, & t \in S, \\ 1, & t \notin S. \end{cases} \]

This means that outside the selected steps, the model falls back to the original Wan2.1 RoPE.

In the current implementation, the step index is tracked from the model forward pass, and the step window is applied inside the monkey-patched RoPE function.

## 6. Scheme 1B: `spatial_temporal_reweight`

### 6.1 Motivation

The manual \(\lambda\)-scaling scheme changes the RoPE phase itself.
That is useful when we want to alter how fast positional phase grows along each axis.

However, there is an even lighter intervention that does not touch the phase at all.
Instead, it changes the relative contribution of temporal channels and spatial channels **after** RoPE has already been applied.

This is useful when the hypothesis is:

- the original 3D RoPE geometry is still basically correct,
- but the final dot product may over-emphasize spatial alignment relative to temporal alignment,
- so we want to rebalance temporal and spatial evidence with minimal extra computation.

This leads to the `spatial_temporal_reweight` scheme.

### 6.2 Channel partition

For one head, after RoPE, write the query and key as

\[
\widetilde q_i
=
\big[
\widetilde q_i^{f},
\widetilde q_i^{h},
\widetilde q_i^{w}
\big]
\in
\mathbb{R}^{d},
\qquad
\widetilde k_j
=
\big[
\widetilde k_j^{f},
\widetilde k_j^{h},
\widetilde k_j^{w}
\big]
\in
\mathbb{R}^{d},
\]

where:

- \(\widetilde q_i^{f}, \widetilde k_j^{f} \in \mathbb{R}^{d_f}\) are the frame-axis channels,
- \(\widetilde q_i^{h}, \widetilde k_j^{h} \in \mathbb{R}^{d_h}\) are the height-axis channels,
- \(\widetilde q_i^{w}, \widetilde k_j^{w} \in \mathbb{R}^{d_w}\) are the width-axis channels,
- \(d_f + d_h + d_w = d\).

Define the combined spatial block

\[
\widetilde q_i^{s}
=
\big[
\widetilde q_i^{h},
\widetilde q_i^{w}
\big]
\in
\mathbb{R}^{d_s},
\qquad
\widetilde k_j^{s}
=
\big[
\widetilde k_j^{h},
\widetilde k_j^{w}
\big]
\in
\mathbb{R}^{d_s},
\]

where \(d_s = d_h + d_w\).

For Wan2.1 `t2v-1.3B`, one head uses

\[
d_f = 44,\qquad d_h = 42,\qquad d_w = 42,\qquad d_s = 84.
\]

The partition is still the original consecutive axis split from the official model.
No interleaving is introduced.

### 6.3 Reweight definition

Let

\[
\alpha \in [0, 1]
\]

be the temporal reweight parameter.

Define the reweighted query and key by

\[
\widehat q_i
=
\big[
\sqrt{\alpha}\,\widetilde q_i^{f},
\sqrt{1-\alpha}\,\widetilde q_i^{h},
\sqrt{1-\alpha}\,\widetilde q_i^{w}
\big],
\]

\[
\widehat k_j
=
\big[
\sqrt{\alpha}\,\widetilde k_j^{f},
\sqrt{1-\alpha}\,\widetilde k_j^{h},
\sqrt{1-\alpha}\,\widetilde k_j^{w}
\big].
\]

Equivalently, in spatial-vs-temporal notation,

\[
\widehat q_i
=
\big[
\sqrt{\alpha}\,\widetilde q_i^{f},
\sqrt{1-\alpha}\,\widetilde q_i^{s}
\big],
\qquad
\widehat k_j
=
\big[
\sqrt{\alpha}\,\widetilde k_j^{f},
\sqrt{1-\alpha}\,\widetilde k_j^{s}
\big].
\]

### 6.4 Resulting attention logit

The modified attention logit is

\[
\widehat s_{ij}
=
\frac{
\langle \widehat q_i, \widehat k_j \rangle
}{
\sqrt d
}.
\]

By direct expansion,

\[
\widehat s_{ij}
=
\frac{
\alpha \langle \widetilde q_i^{f}, \widetilde k_j^{f} \rangle
+
(1-\alpha)\langle \widetilde q_i^{h}, \widetilde k_j^{h} \rangle
+
(1-\alpha)\langle \widetilde q_i^{w}, \widetilde k_j^{w} \rangle
}{
\sqrt d
}.
\]

If we use the combined spatial block, the same formula can be written as

\[
\widehat s_{ij}
=
\frac{
\alpha \langle \widetilde q_i^{f}, \widetilde k_j^{f} \rangle
+
(1-\alpha)\langle \widetilde q_i^{s}, \widetilde k_j^{s} \rangle
}{
\sqrt d
}.
\]

So this scheme does **not** change the RoPE phase.
It changes only the relative energy of temporal and spatial channel groups inside the final dot product.

### 6.5 Difference from axis-wise \(\lambda\) scaling

The difference between Scheme 1 and `spatial_temporal_reweight` is fundamental:

- Scheme 1 changes the phase growth rate before the query-key dot product is computed,
- `spatial_temporal_reweight` keeps the original phase and changes only the channel-group weighting after RoPE.

So Scheme 1 modifies the positional geometry, while Scheme 1B modifies the final temporal-vs-spatial balance given that geometry.

### 6.6 Complexity and engineering advantage

This scheme is computationally cheap.
It does not create a second attention branch and it does not create a second softmax.

The implementation cost is only:

- split the last channel dimension into \((f, h, w)\),
- multiply the frame block by \(\sqrt{\alpha}\),
- multiply the height and width blocks by \(\sqrt{1-\alpha}\),
- concatenate the blocks back.

So the asymptotic attention complexity is unchanged.
Compared with semantic residual attention, this scheme adds almost no extra cost.

### 6.7 Step control

The same diffusion-step window from Section 5 is used here.
If the current diffusion step is outside the selected step set, the reweight is not applied and the model falls back to the original RoPE-based attention.

This is especially useful when we only want to bias the early motion-planning steps.

## 7. Scheme 2: Timestep-Conditioned Scale Learning

Scheme 2 is the unified training-oriented formulation of axis-wise RoPE scaling.

The core idea is to stop treating the scale vector

\[
\lambda = [\lambda_f, \lambda_h, \lambda_w]
\]

as a purely manual hyperparameter.
Instead, the model is allowed to learn:

- a base axis-wise scale,
- and a timestep-dependent correction on top of that base.

Under this view, the old manual scheme becomes a special case in which the base scale is fixed and the timestep-dependent correction is removed.

### 7.1 Learnable base scale

Define the log-scale base parameter

\[
\mu = [\mu_f, \mu_h, \mu_w] \in \mathbb{R}^3.
\]

The corresponding positive base scale is

\[
\lambda^{\text{base}} = \exp(\mu)
=
\big[
\lambda_f^{\text{base}},
\lambda_h^{\text{base}},
\lambda_w^{\text{base}}
\big]
\in
\mathbb{R}_{>0}^3.
\]

Here:

- \(\mu_f, \mu_h, \mu_w\) are unconstrained trainable parameters in log space,
- \(\lambda_f^{\text{base}}, \lambda_h^{\text{base}}, \lambda_w^{\text{base}}\) are the positive base scales actually used by the model.

This formulation has two advantages:

- positivity is guaranteed by the exponential map,
- the base scale itself can now be learned rather than manually fixed forever.

If one wants to recover the manual scheme, one can simply freeze \(\mu\) at

\[
\mu_a = \log \lambda_a,
\qquad a \in \{f,h,w\}.
\]

So Scheme 1 is not a different family.
It is the fixed-base special case of Scheme 2.

### 7.2 Timestep embedding

Let

\[
e_t \in \mathbb{R}^{d_e}
\]

be the timestep embedding for diffusion step \(t\), where \(d_e\) is the timestep-embedding width.

In the implementation, \(e_t\) is produced by the same sinusoidal timestep embedding used by Wan2.1.

### 7.3 Global timestep-conditioned mode

In the simplest learned variant, the model predicts one axis-wise correction vector for each timestep:

\[
g(e_t) \in \mathbb{R}^3.
\]

The effective scale vector is then

\[
\lambda(t)
=
\exp\big(\mu + g(e_t)\big)
\in
\mathbb{R}_{>0}^3.
\]

Write

\[
\lambda(t)
=
\big[
\lambda_f(t),
\lambda_h(t),
\lambda_w(t)
\big].
\]

Then:

- \(\lambda_f(t)\) is the frame-axis scale used at step \(t\),
- \(\lambda_h(t)\) is the height-axis scale used at step \(t\),
- \(\lambda_w(t)\) is the width-axis scale used at step \(t\).

Conceptually, if diffusion had only a finite set of timesteps and we tabulated these values directly, this mode would correspond to one learned \(3\)-vector per timestep.
So its role is exactly the global step-wise scale controller.

### 7.4 Effective scale under step gating

If a diffusion-step window \(S\) is also specified, then the actually used scale is

\[
\lambda^{\text{used}}_a(t)
=
\begin{cases}
\lambda_a(t), & t \in S, \\
1, & t \notin S,
\end{cases}
\qquad a \in \{f,h,w\}.
\]

So outside the selected steps, the model falls back to the original Wan2.1 RoPE.

### 7.5 Why this formulation is preferable

This unified view is preferable to "manual \(\lambda\) times learned multiplier" as a primary formulation.

The reason is simple:

- if the base scale is always frozen by hand, the learned module can only make multiplicative corrections around a manually chosen anchor,
- if the base scale is itself learnable, the model can jointly determine the long-range prior and the timestep-dependent modulation.

So the cleanest training-oriented parameterization is:

- manual mode: fix \(\mu\) and remove \(g(e_t)\),
- learned global mode: learn both \(\mu\) and \(g(e_t)\).

## 8. Why We Do Not Split the Softmax

The experiment deliberately keeps one joint attention softmax.
We do **not** compute one spatial softmax and one temporal softmax inside the same head.

The reason is that separate softmax branches can destroy joint spatiotemporal binding.
For example:

- a purely spatial branch cannot tell whether a token belongs to an early frame or a late frame if the spatial coordinate is the same,
- a purely temporal branch cannot tell whether the token is foreground or background inside the same frame.

So the current experiment changes the RoPE phase but preserves the original joint attention probability.

## 9. Scheme 2B: Head-Aware Timestep-Conditioned Scaling

The global timestep-conditioned mode in Section 7 uses one scale vector for all heads.
That is the cleanest starting point, but it may still be too restrictive.

The natural refinement is not a completely different scheme.
It is simply a different mode of the same timestep-conditioned family:

- `global` mode: one scale vector for the whole attention layer at step \(t\),
- `head_aware` mode: different heads may use different scale vectors at the same step \(t\).

This is why the head-aware design should be treated as a mode choice inside Scheme 2 rather than as a separate family.

### 9.1 Head-aware correction tensor

For self-attention layer \(\ell\), head \(m\), and diffusion step \(t\), define

\[
g_{\ell,m}(e_t)
\in
\mathbb{R}^3.
\]

Its three coordinates are:

- \(g_{\ell,m}^{(f)}(e_t)\): frame-axis correction for head \(m\),
- \(g_{\ell,m}^{(h)}(e_t)\): height-axis correction for head \(m\),
- \(g_{\ell,m}^{(w)}(e_t)\): width-axis correction for head \(m\).

So the correction now depends on:

- timestep \(t\),
- layer index \(\ell\),
- head index \(m\).

Conceptually, if these values were tabulated directly for discrete timesteps, this mode would correspond to a head-resolved family of timestep-specific scale factors rather than a single shared \(3\)-vector.

### 9.2 Effective head-aware scale

The head-aware scale vector is

\[
\lambda_{\ell,m}(t)
=
\exp\big(\mu + g_{\ell,m}(e_t)\big)
\in
\mathbb{R}_{>0}^3.
\]

Write

\[
\lambda_{\ell,m}(t)
=
\big[
\lambda_{\ell,m}^{(f)}(t),
\lambda_{\ell,m}^{(h)}(t),
\lambda_{\ell,m}^{(w)}(t)
\big].
\]

Then:

- \(\lambda_{\ell,m}^{(f)}(t)\) is the frame-axis scale used by head \(m\) at layer \(\ell\),
- \(\lambda_{\ell,m}^{(h)}(t)\) is the height-axis scale used by head \(m\) at layer \(\ell\),
- \(\lambda_{\ell,m}^{(w)}(t)\) is the width-axis scale used by head \(m\) at layer \(\ell\).

These three scales are inserted into the same RoPE phase definition from Section 4.
The attention operator itself is unchanged.

### 9.3 Relationship to the global mode

The global timestep-conditioned mode from Section 7 is a special case of the head-aware mode.

If all heads share the same correction,

\[
g_{\ell,m}(e_t) = g(e_t)
\]

for all \(\ell\) and \(m\), then

\[
\lambda_{\ell,m}(t) = \lambda(t)
\]

for all heads.

So the difference between the two modes is not conceptual.
It is only the resolution at which the timestep-conditioned correction is applied.

### 9.4 Why this unification is cleaner

This unification avoids the previous duplication in which:

- one scheme learned timestep-dependent global multipliers,
- another scheme separately introduced head-specific gains.

Both objects were controlling the same mathematical quantity: the axis-wise RoPE scale.

The cleaner view is:

- the object being controlled is always the scale vector \(\lambda\),
- the only design choice is the resolution of the controller.

That resolution can be:

- global across all heads,
- or head-aware.

### 9.5 Engineering interpretation

Under this unified view, the mode switch should control which tensor shape is predicted by the timestep-conditioned module:

- `global` mode predicts one \(3\)-vector,
- `head_aware` mode predicts a head-indexed collection of \(3\)-vectors.

So the two modes differ only in output shape and parameter sharing pattern.
They do not need to be presented as unrelated mechanisms.

## 10. Semantic Residual Self-Attention as a Compatible Extension

This section describes another compatible extension.
It is implemented in the current code as an optional self-attention logit correction.

The purpose of this extension is different from axis-wise RoPE scaling.
Axis-wise scaling changes the positional phase prior.
Semantic residual self-attention adds a content-based correction term on top of the original RoPE attention logit.

The two ideas are therefore compatible.

### 10.1 Motivation

In video self-attention, a query token from a moving object may attend strongly to the same spatial coordinate in other frames, instead of attending to the new location of the same object.

That bias is not always wrong:

- it helps static background consistency,
- it preserves a natural same-coordinate prior across frames.

However, if that bias becomes too strong, it can hurt motion planning.
The model may over-read features from a static coordinate and under-read features from the true object trajectory.

The goal of semantic residual self-attention is therefore not to remove the positional prior.
The goal is to add a controlled semantic correction for cross-frame token matching.

### 10.2 Standard RoPE logit

For layer \(\ell\), head \(m\), query token \(i\), and key token \(j\), let

- \(q_i^{(\ell,m)} \in \mathbb{R}^d\) be the query before RoPE,
- \(k_j^{(\ell,m)} \in \mathbb{R}^d\) be the key before RoPE,
- \(\widetilde q_i^{(\ell,m)} \in \mathbb{R}^d\) be the query after RoPE,
- \(\widetilde k_j^{(\ell,m)} \in \mathbb{R}^d\) be the key after RoPE.

The standard RoPE attention logit is

\[
s_{ij}^{\mathrm{rope},(\ell,m)}
=
\frac{
\langle \widetilde q_i^{(\ell,m)}, \widetilde k_j^{(\ell,m)} \rangle
}{
\sqrt d
}
\in
\mathbb{R}.
\]

### 10.3 Semantic residual logit

Define a pure semantic logit by using the pre-RoPE query and key:

\[
s_{ij}^{\mathrm{sem},(\ell,m)}
=
\frac{
\langle q_i^{(\ell,m)}, k_j^{(\ell,m)} \rangle
}{
\sqrt d
}
\in
\mathbb{R}.
\]

This term is intentionally computed before RoPE, so that it emphasizes feature similarity rather than positional phase alignment.

Next define a cross-frame mask

\[
M_{ij}
=
\begin{cases}
1, & f_i \neq f_j, \\
0, & f_i = f_j,
\end{cases}
\]

where \(f_i\) and \(f_j\) are the frame indices of tokens \(i\) and \(j\).

So \(M_{ij}\) activates the residual term only for cross-frame token pairs.

### 10.4 Final attention logit

Let

\[
\alpha(t) \in \mathbb{R}_{\ge 0}
\]

be the semantic residual weight at diffusion step \(t\).

In the current implementation, \(\alpha(t)\) can be used in two ways:

- as a manually specified constant scalar,
- as a manual base scalar multiplied by an optional timestep-conditioned positive scalar head.

The final attention logit is

\[
\hat s_{ij}^{(\ell,m)}
=
s_{ij}^{\mathrm{rope},(\ell,m)}
+
\alpha(t)\, M_{ij}\, s_{ij}^{\mathrm{sem},(\ell,m)}.
\]

Then the model computes the attention probability in the usual way:

\[
A_{ij}^{(\ell,m)}
=
\mathrm{softmax}_j
\left(
\hat s_{ij}^{(\ell,m)}
\right).
\]

This construction preserves a single attention matrix and a single softmax.
It does **not** split the head into separate spatial and temporal branches.

### 10.5 Relation to the RoPE scale schemes

The role of the RoPE scale schemes is to modify the positional prior through the phase.
The role of the semantic residual term is to modify cross-frame content matching at the logit level.

So the two mechanisms act at different places:

- manual \(\lambda\) scaling, `spatial_temporal_reweight`, and timestep-conditioned scaling change how \(\widetilde q\) and \(\widetilde k\) are formed or weighted,
- semantic residual attention changes the final logit after the RoPE term has already been computed.

This is why Scheme 10 is compatible with axis-wise \(\lambda\) scaling and with `spatial_temporal_reweight`.
One can use both at the same time:

- the RoPE term keeps geometric structure and positional stability,
- the semantic residual term helps cross-frame semantic correspondence.

### 10.6 Step control

The semantic residual term can use the same diffusion-step window idea from Section 5.
Let \(S \subseteq \{1, 2, \dots, K\}\) be the active step set.

Define

\[
\alpha^{\mathrm{used}}(t)
=
\begin{cases}
\alpha(t), & t \in S, \\
0, & t \notin S.
\end{cases}
\]

Then the final logit becomes

\[
\hat s_{ij}^{(\ell,m)}
=
s_{ij}^{\mathrm{rope},(\ell,m)}
+
\alpha^{\mathrm{used}}(t)\, M_{ij}\, s_{ij}^{\mathrm{sem},(\ell,m)}.
\]

This makes the extension especially suitable for early denoising steps, where global layout and rough motion planning are most strongly formed.

## 11. Current Code Path

The current code implements:

- Scheme 1: manual axis-wise \(\lambda_f, \lambda_h, \lambda_w\),
- Scheme 1B: `spatial_temporal_reweight`,
- diffusion-step gating,
- Scheme 2, `global` mode: a timestep-conditioned scale head with checkpoint-loading support,
- Scheme 2B, `head_aware` mode: a head-aware timestep-conditioned scaling module with runtime attachment and checkpoint-loading support,
- Scheme 10: Semantic Residual Self-Attention with cross-frame masking, step gating, and optional timestep-conditioned \(\alpha(t)\).

In the current implementation, the global and head-aware modes are still exposed through separate switches.
The unified presentation in Sections 6 and 8 is the cleaner mathematical view of the same scale-learning family.

The main files are:

- [rope_modification.py](/home/liyueyan/Interpretability/physics/wan21_t2v_experiments/rope_modification.py)
- [wan21_t2v_experiment_patch.py](/home/liyueyan/Interpretability/physics/wan21_t2v_experiments/wan21_t2v_experiment_patch.py)
- [run_wan21_t2v_experiments.py](/home/liyueyan/Interpretability/physics/wan21_t2v_experiments/run_wan21_t2v_experiments.py)

## 12. Recommended Evaluation

The most relevant existing experiments for evaluating this proposal are:

- `rope_decay_curve`
- `attention_dt_profile`
- `motion_aligned_attention`
- `cross_attention_token_viz`
- `head_evolution`

Together, these experiments can test whether the modified RoPE produces:

- slower temporal decay,
- stronger temporal coherence in early denoising,
- preserved spatial sharpness,
- meaningful head-level behavior changes.
