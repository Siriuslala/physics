# `rope_modification` Technical Note

## 1. Goal

`rope_modification` studies how to modify Wan2.1 T2V self-attention RoPE without editing the official model source code.
All changes are applied through runtime monkey patching.

The experiment currently contains two schemes:

1. `manual`: a training-free scheme with manually chosen axis-wise scales.
2. `step_conditioned`: a training-oriented scheme in which the axis-wise scales are produced by a small timestep-conditioned module.

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

\[
\theta_{a,r} = b_a^{-2r/d_a},
\qquad r = 0, 1, \dots, m_a - 1.
\]

So:

- \(b_a\) is the base,
- \(\theta_{a,r}\) is the angular frequency generated from that base.

This is the notation used in the rest of the note.

## 4. Scheme 1: Manual Axis-Wise Scaling

### 4.1 Motivation

The first scheme is the simplest one and should be implemented first.
The idea is:

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

\[
\begin{bmatrix}
x_1' \\
x_2'
\end{bmatrix}
=
\begin{bmatrix}
\cos \phi & -\sin \phi \\
\sin \phi & \cos \phi
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}.
\]

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

\[
\lambda_a^{\text{used}}(t) =
\begin{cases}
\lambda_a, & t \in S, \\
1, & t \notin S.
\end{cases}
\]

This means that outside the selected steps, the model falls back to the original Wan2.1 RoPE.

In the current implementation, the step index is tracked from the model forward pass, and the step window is applied inside the monkey-patched RoPE function.

## 6. Scheme 2: Step-Conditioned Extension

Scheme 2 keeps the same idea as Scheme 1, but replaces the fixed scales with a small trainable timestep-conditioned module.

### 6.1 Timestep embedding

Let

\[
e_t \in \mathbb{R}^{d_e}
\]

be the timestep embedding for diffusion step \(t\), where \(d_e\) is the timestep-embedding width.

In the implementation, the timestep embedding is produced by the same sinusoidal embedding function used by Wan2.1.

### 6.2 Scale head output

The step-conditioned scale head is a small MLP:

\[
g(e_t) \in \mathbb{R}^3.
\]

Its output is converted into positive dynamic scales:

\[
\lambda^{\text{dyn}}(t) =
\exp(g(e_t))
=
\big[
\lambda_f^{\text{dyn}}(t),
\lambda_h^{\text{dyn}}(t),
\lambda_w^{\text{dyn}}(t)
\big].
\]

Here:

- \(\lambda_f^{\text{dyn}}(t)\) is the dynamic frame-axis multiplier,
- \(\lambda_h^{\text{dyn}}(t)\) is the dynamic height-axis multiplier,
- \(\lambda_w^{\text{dyn}}(t)\) is the dynamic width-axis multiplier.

### 6.3 Effective scale

Scheme 2 still uses the manual scales as a base.
The final scale applied at diffusion step \(t\) is

\[
\lambda_a^{\text{eff}}(t) =
\lambda_a \cdot \lambda_a^{\text{dyn}}(t).
\]

So:

- \(\lambda_a\) is the manual base scale,
- \(\lambda_a^{\text{dyn}}(t)\) is the learned timestep-dependent multiplier,
- \(\lambda_a^{\text{eff}}(t)\) is the scale actually used by RoPE.

If a diffusion-step window \(S\) is also specified, then

\[
\lambda_a^{\text{used}}(t) =
\begin{cases}
\lambda_a^{\text{eff}}(t), & t \in S, \\
1, & t \notin S.
\end{cases}
\]

### 6.4 Engineering implementation

The step-conditioned scale head is implemented as a small module attached to the monkey-patched model at runtime.
This is done in [wan21_t2v_experiment_patch.py](</home/liyueyan/Interpretability/physics/wan21_t2v_experiments/wan21_t2v_experiment_patch.py:57>).

Important engineering properties:

- it is attached as an `nn.Module`, so its parameters appear in `state_dict`,
- it can optionally load a checkpoint by path,
- it does not require editing `projects/Wan2_1`,
- it is therefore compatible with a later training framework.

## 7. Why We Do Not Split the Softmax

The experiment deliberately keeps one joint attention softmax.
We do **not** compute one spatial softmax and one temporal softmax inside the same head.

The reason is that separate softmax branches can destroy joint spatiotemporal binding.
For example:

- a purely spatial branch cannot tell whether a token belongs to an early frame or a late frame if the spatial coordinate is the same,
- a purely temporal branch cannot tell whether the token is foreground or background inside the same frame.

So the current experiment changes the RoPE phase but preserves the original joint attention probability.

## 8. Soft Head Specialization as a Future Extension

Soft Head Specialization is a later-stage design.
It is not implemented in the current code.

The goal is to let different heads prefer different RoPE scale patterns without hard-coding head ids by hand.

### 8.1 Router definition

For layer \(\ell\), head \(m\), and diffusion step \(t\), define a router output

\[
\beta_{\ell,m}(t)
=
\big[
\beta_{\ell,m}^{(f)}(t),
\beta_{\ell,m}^{(s)}(t),
\beta_{\ell,m}^{(j)}(t)
\big]
\in \mathbb{R}^3,
\]

where:

- \(\beta_{\ell,m}^{(f)}(t)\) is the temporal-specialist weight,
- \(\beta_{\ell,m}^{(s)}(t)\) is the spatial-specialist weight,
- \(\beta_{\ell,m}^{(j)}(t)\) is the joint-head weight.

The router is normalized by softmax:

\[
\beta_{\ell,m}(t) = \mathrm{softmax}(W_{\ell,m} e_t + c_{\ell,m}),
\]

so its three coordinates are nonnegative and sum to \(1\).

### 8.2 Three reference scale triplets

Define three reference scale triplets:

\[
\lambda^{(f)} =
\big[
\lambda_f^{(f)},
\lambda_h^{(f)},
\lambda_w^{(f)}
\big],
\]

\[
\lambda^{(s)} =
\big[
\lambda_f^{(s)},
\lambda_h^{(s)},
\lambda_w^{(s)}
\big],
\]

\[
\lambda^{(j)} =
\big[
\lambda_f^{(j)},
\lambda_h^{(j)},
\lambda_w^{(j)}
\big].
\]

Their meanings are:

- \(\lambda^{(f)}\): the reference scale triplet for a temporal-specialist head,
- \(\lambda^{(s)}\): the reference scale triplet for a spatial-specialist head,
- \(\lambda^{(j)}\): the reference scale triplet for a joint head.

### 8.3 Effective per-head scale

The effective scale triplet for layer \(\ell\), head \(m\), and step \(t\) is

\[
\lambda_{\ell,m}(t)
=
\beta_{\ell,m}^{(f)}(t)\lambda^{(f)}
+
\beta_{\ell,m}^{(s)}(t)\lambda^{(s)}
+
\beta_{\ell,m}^{(j)}(t)\lambda^{(j)}.
\]

This formula means:

- if \(\beta_{\ell,m}^{(f)}(t)\) is large, the head behaves more like a temporal specialist,
- if \(\beta_{\ell,m}^{(s)}(t)\) is large, the head behaves more like a spatial specialist,
- if \(\beta_{\ell,m}^{(j)}(t)\) is large, the head stays closer to a joint head.

Crucially, this still modifies RoPE through scales only.
It does **not** create three separate attention matrices.

### 8.4 Why this is safer than hard head assignment

This future extension is softer than manually declaring:

- "head 0 to head 3 are temporal heads",
- "head 4 to head 7 are spatial heads".

Instead, the model learns a continuous preference distribution over head behaviors.
That is the main reason why this idea is safer, but it should still be treated as a second-stage extension rather than the first implementation target.

## 9. Current Code Path

The current code implements:

- Scheme 1: manual axis-wise \(\lambda_f, \lambda_h, \lambda_w\),
- diffusion-step gating,
- Scheme 2: a timestep-conditioned scale head with checkpoint-loading support.

The main files are:

- [rope_modification.py](/home/liyueyan/Interpretability/physics/wan21_t2v_experiments/rope_modification.py)
- [wan21_t2v_experiment_patch.py](/home/liyueyan/Interpretability/physics/wan21_t2v_experiments/wan21_t2v_experiment_patch.py)
- [run_wan21_t2v_experiments.py](/home/liyueyan/Interpretability/physics/wan21_t2v_experiments/run_wan21_t2v_experiments.py)

## 10. Recommended Evaluation

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
