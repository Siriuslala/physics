# `self_attention_temporal_kernel` Technical Note

## 1. Motivation

`self_attention_temporal_kernel` is a causal-intervention experiment on Wan2.1-T2V self-attention. The current implementation supports two conceptually different intervention modes:

1. `postoutput_same_position_kernel`
2. `prelogit_token_temperature`

The first is a post-attention proxy intervention. The second is an exact token-level attention-temperature intervention.

## 2. Why Full Attention Maps Are Not Materialized

Let the patchified video-latent sequence length be

\[
L = F \cdot S,
\]

where:

- \(F\) is the latent-frame count,
- \(S\) is the spatial token count per latent frame.

For common Wan2.1-T2V-1.3B settings, \(L\) is on the order of \(3.2 \times 10^4\), so a full attention map per head has shape \(L \times L\), which is too large to materialize directly. Therefore the experiment intervenes without explicitly exporting the full attention matrix.

## 3. Baseline Self-Attention

For query token \(i\), key token \(j\), and head \(h\), define

\[
\ell_h(i,j) = \frac{\langle q_{i,h}, k_{j,h} \rangle}{\sqrt d},
\]

where:

- \(q_{i,h} \in \mathbb{R}^d\) is the query vector,
- \(k_{j,h} \in \mathbb{R}^d\) is the key vector,
- \(d\) is the head dimension.

The baseline attention probability is

\[
\alpha_h(i,j) = \operatorname{softmax}_{j}(\ell_h(i,j)).
\]

The head output is

\[
y_{i,h} = \sum_j \alpha_h(i,j) v_{j,h}.
\]

## 4. Mode A: `postoutput_same_position_kernel`

This is the older proxy scheme. It does **not** change \(\alpha_h(i,j)\) directly.

### 4.1 Tensor view

After self-attention is computed, the per-head outputs are reshaped into

\[
Y \in \mathbb{R}^{F \times H \times W \times N \times d},
\]

where:

- \(F\): latent frames,
- \(H, W\): token-grid height and width,
- \(N\): number of heads,
- \(d\): head dimension.

For fixed spatial location \((u,v)\), head \(h\), and channel \(c\), define the temporal signal

\[
z_{f,u,v,h,c} = Y[f,u,v,h,c].
\]

### 4.2 Temporal smoothing kernel

Let the temporal offsets be

\[
\Delta \in \{-r, \dots, r\},
\]

where \(r\) is `self_attn_kernel_radius`. The Gaussian kernel is

\[
K(\Delta) = \exp\!\left(-\frac{\Delta^2}{2\sigma^2}\right),
\]

with \(\sigma = \texttt{self_attn_kernel_sigma}\), and is normalized to

\[
\widetilde K(\Delta) = \frac{K(\Delta)}{\sum_{\delta=-r}^{r} K(\delta)}.
\]

### 4.3 Output mixing

The smoothed signal is

\[
\bar z_{f,u,v,h,c} = \sum_{\delta=-r}^{r} \widetilde K(\delta) \, z_{f+\delta,u,v,h,c},
\]

using replicate padding at the temporal boundaries. The final mixed output is

\[
Y'_{f,u,v,h,c} = (1-\alpha)Y_{f,u,v,h,c} + \alpha \, \bar Y_{f,u,v,h,c},
\]

where \(\alpha = \texttt{self_attn_kernel_mix_alpha}\).

### 4.4 Interpretation

This mode keeps Wan's original attention routing unchanged and only smooths the resulting head outputs across latent frames at the same spatial token position. It is therefore a proxy intervention, not a direct attention-weight rewrite.

## 5. Mode B: `prelogit_token_temperature`

This is the exact token-level temperature intervention.

### 5.1 Goal

For selected heads, replace the attention probabilities by

\[
\alpha_h^{(T)}(i,j) = \operatorname{softmax}_{j}\!\left(\frac{\ell_h(i,j)}{T}\right),
\qquad T > 0,
\]

where \(T = \texttt{self_attn_token_temperature}\).

- \(T = 1\): no change
- \(T > 1\): flatter token-level attention distribution
- \(0 < T < 1\): sharper token-level attention distribution

### 5.2 Exact implementation without materializing the full attention matrix

Because

\[
\ell_h(i,j) = \frac{\langle q_{i,h}, k_{j,h} \rangle}{\sqrt d},
\]

we can realize the temperature scaling exactly by scaling both query and key:

\[
q'_{i,h} = T^{-1/2} q_{i,h},
\qquad
k'_{j,h} = T^{-1/2} k_{j,h}.
\]

Then

\[
\frac{\langle q'_{i,h}, k'_{j,h} \rangle}{\sqrt d}
= \frac{\langle q_{i,h}, k_{j,h} \rangle}{\sqrt d \, T}.
\]

More generally, if both query and key are scaled by the same scalar \(s\), then

\[
\frac{\langle s q_{i,h}, s k_{j,h} \rangle}{\sqrt d}
= s^2 \frac{\langle q_{i,h}, k_{j,h} \rangle}{\sqrt d}.
\]

So to obtain an exact division by \(T\), we must set

\[
s^2 = \frac{1}{T}
\quad\Longrightarrow\quad
s = T^{-1/2}.
\]

So feeding \(q'\) and \(k'\) into flash attention is mathematically equivalent to applying an exact softmax temperature \(T\) to the selected heads' token-level attention logits.

### 5.3 Interpretation

This mode directly changes the attention weights themselves. It is therefore the most faithful implementation of the idea:

- make the selected heads attend less sharply to their top choices,
- and relatively increase mass on weaker alternatives,
- without explicitly exporting the full \(L \times L\) attention matrix.

The tradeoff is that the temperature is applied to the full token-token attention distribution, so it affects both temporal and spatial competition, not only temporal competition.

## 6. Inputs

Shared inputs:

- `self_attn_kernel_steps`
- `self_attn_kernel_layers`
- `self_attn_kernel_heads`
- `self_attn_kernel_branch`
- `self_attn_temporal_intervention_mode`

Mode-specific inputs:

- `postoutput_same_position_kernel`:
  - `self_attn_kernel_radius`
  - `self_attn_kernel_sigma`
  - `self_attn_kernel_mix_alpha`
- `prelogit_token_temperature`:
  - `self_attn_token_temperature`

## 7. Outputs

- generated video
- `self_attention_temporal_kernel_calls.csv`
- `self_attention_temporal_kernel_head_calls.csv`
- `self_attention_temporal_kernel_summary.json`

## 8. Practical Reading

- Use `prelogit_token_temperature` when the scientific question is about changing attention sharpness itself.
- Use `postoutput_same_position_kernel` only as a weaker proxy baseline, because it does not directly modify the attention probabilities.
