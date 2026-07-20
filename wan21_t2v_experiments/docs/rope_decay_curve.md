# `rope_decay_curve` Technical Note

## 1. Goal

This experiment does not load Wan2.1 model weights. It directly reuses the
official Wan2.1 RoPE parameterization and visualizes:

1. temporal frame-level decay curves
2. spatial decay curves along the height and width axes
3. a same-frame spatial coherence heatmap around one center anchor
4. a spatial radial profile derived from that heatmap
5. a flattened video-token anchor curve

这里所谓的“衰减”不是指 RoPE 显式乘了一个小于 1 的系数，而是指：

- 随着相对距离增大，不同频率分量的相位差越来越大
- 在多频率平均之后，query/key 的旋转后内积核会逐渐失去相干性
- 因而其平均余弦核会从 1 附近下降，并伴随振荡

这个实验要可视化的正是这种由相位错位带来的有效衰减。

## 2. Wan2.1 中的 RoPE 定义

Wan2.1 官方代码中：

```python
freqs = torch.outer(
    torch.arange(max_seq_len),
    1.0 / torch.pow(theta, torch.arange(0, dim, 2) / dim)
)
freqs = torch.polar(torch.ones_like(freqs), freqs)
```

对应的数学形式是：

对于某一条轴上的 real 维度 \(d\)，要求 \(d\) 为偶数。定义频率索引

\[ k = 0,1,\dots,\frac d2-1, \]

则该轴上的 base 记为

\[ b = 10000, \]

对应的第 \(k\) 个角频率记为

\[ \theta_k = b^{- \frac{2k}{d}}. \]

位置 \(p\) 上的复数旋转因子为

\[ r_k(p)=e^{i p \theta_k}. \]

因此两个位置 \(p\) 与 \(q\) 在该频率上的相对相位只依赖于相对距离

\[ \Delta = p-q, \]

并满足

\[ r_k(p)\overline{r_k(q)} = e^{i \Delta \theta_k}. \]

## 2.1 为什么 current experiment 里的 RoPE kernel 可以定义成 cosine mean

先看单个 2D rotary pair。记旋转矩阵为

\[ R(\alpha)=\begin{bmatrix} \cos\alpha & -\sin\alpha\\ \sin\alpha & \cos\alpha \end{bmatrix}. \]

对第 \(k\) 个频率分量，如果某个 pre-RoPE 2D 向量是 \(u_k \in \mathbb{R}^2\)，那么它在位置 \(p\) 处经过 RoPE 后变成

\[ \tilde u_k(p)=R(p\theta_k)u_k. \]

如果比较“同一个 2D 特征对自己在两个位置上的相干性”，那么有

\[ \langle \tilde u_k(p), \tilde u_k(q)\rangle = u_k^\top R((q-p)\theta_k)u_k. \]

把 \(u_k=(a_k,b_k)^\top\) 展开，可得

\[ u_k^\top R(\Delta\theta_k)u_k = (a_k^2+b_k^2)\cos(\Delta\theta_k). \]

也就是说，对于“同向量自相关”这个最基本的 RoPE-only 场景，位置差 \(\Delta\) 带来的相干性变化恰好就是

\[ \cos(\Delta\theta_k) \]

乘上一个与内容能量有关但与位置无关的系数。

进一步地，如果不是完全取自相关，而是考虑一般的两个 2D 向量 \(u_k,v_k\)，则有

\[ \langle R(p\theta_k)u_k,\; R(q\theta_k)v_k\rangle = \cos(\Delta\theta_k)\langle u_k,v_k\rangle \;+\; \sin(\Delta\theta_k)\langle J u_k,v_k\rangle, \]

其中

\[ J=\begin{bmatrix} 0 & -1\\ 1 & 0 \end{bmatrix}. \]

因此：

- cosine 项控制“同向相干”
- sine 项控制“正交相位耦合”

如果我们只想抽出 RoPE 自身的“距离导致的平均相干性 envelope”，最自然的 RoPE-only proxy 就是保留 cosine 主项，并对所有频率取平均。

所以，当前实验里的 axis kernel 不是“唯一正确”的定义，而是：

- 在忽略内容分布细节时
- 从 RoPE 相位结构中抽出的最直接、最可解释的相干性 proxy

在这个意义下，轴向 kernel 定义为

\[ K_{\text{axis}}(\Delta) = \frac{2}{d} \sum_{k=0}^{d/2-1}\cos(\Delta \theta_k). \]

由于上式前面的 \(\frac{2}{d}\) 与求和项中的项数 \(\frac d2\) 正好抵消，所以代码里直接写成 mean 即可：

\[ K_{\text{axis}}(\Delta) = \frac{1}{d/2}\sum_{k=0}^{d/2-1}\cos(\Delta \theta_k). \]

这正是当前实现里的 `_wan21_t2v_mean_rope_cosine_kernel(...)`。

## 3. Wan2.1 的三轴 RoPE 拆分

Wan2.1 的 self-attention head dimension 记为

\[ d_{\text{head}}. \]

官方实现中先取

\[ c = \frac{d_{\text{head}}}{2}, \]

然后把 RoPE 频率按三轴拆分为：

\[ c_f = c - 2\left\lfloor\frac c3\right\rfloor, \qquad c_h = \left\lfloor\frac c3\right\rfloor, \qquad c_w = \left\lfloor\frac c3\right\rfloor. \]

在代码里，这对应：

```python
freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
```

又因为每个 complex 频率对应 2 个 real channels，所以三个轴对应的 real 维度分别是：

\[ d_f = d_{\text{head}} - 4\left\lfloor\frac{d_{\text{head}}}{6}\right\rfloor, \]

\[ d_h = 2\left\lfloor\frac{d_{\text{head}}}{6}\right\rfloor, \qquad d_w = 2\left\lfloor\frac{d_{\text{head}}}{6}\right\rfloor. \]

对于 Wan2.1 T2V 1.3B：

- `dim = 1536`
- `num_heads = 12`

因此

\[ d_{\text{head}}=\frac{1536}{12}=128. \]

于是

\[ d_f = 128 - 4\left\lfloor\frac{128}{6}\right\rfloor = 44, \]

\[ d_h=d_w=2\left\lfloor\frac{128}{6}\right\rfloor = 42. \]

注意：

- \(44, 42, 42\) 都是 real 维度
- 它们对应的 complex 频率数分别是 \(22, 21, 21\)

## 4. 三轴联合的 RoPE Kernel

令两个 video tokens 的三轴坐标分别为

\[ (f_1,h_1,w_1), \qquad (f_2,h_2,w_2), \]

定义相对位移

\[ \Delta_f = f_1-f_2,\qquad \Delta_h = h_1-h_2,\qquad \Delta_w = w_1-w_2. \]

忽略内容向量，只看 RoPE 本身的平均相干性，则三轴联合 kernel 定义为

\[ K_{\text{full}}(\Delta_f,\Delta_h,\Delta_w) = \frac{ \sum_{k=0}^{d_f/2-1}\cos(\Delta_f \theta^{(f)}_k) \;+\; \sum_{k=0}^{d_h/2-1}\cos(\Delta_h \theta^{(h)}_k) \;+\; \sum_{k=0}^{d_w/2-1}\cos(\Delta_w \theta^{(w)}_k) }{ d_f/2 + d_h/2 + d_w/2 }. \]

当前实现里的 `_wan21_t2v_full_rope_kernel(...)` 正是在计算这个量。

## 5. 832x480, frame_num=81 时的 token 网格

Wan2.1 T2V 1.3B 配置：

- `vae_stride = (4, 8, 8)`
- `patch_size = (1, 2, 2)`

因此对于视频设置：

- `size = 832*480`
- `frame_num = 81`

latent frame 数为

\[ F_{\text{latent}} = \frac{81-1}{4}+1 =21. \]

latent 空间分辨率为

\[ H_{\text{latent}}=\frac{480}{8}=60, \qquad W_{\text{latent}}=\frac{832}{8}=104. \]

再经过 patchify：

\[ H_{\text{tok}}=\frac{60}{2}=30, \qquad W_{\text{tok}}=\frac{104}{2}=52. \]

所以最终视频 token 网格大小是

\[ (F_{\text{latent}}, H_{\text{tok}}, W_{\text{tok}}) = (21,30,52). \]

总 token 数是

\[ L = 21 \times 30 \times 52 = 32760. \]

## 6. 各张图分别画什么

### 6.1 Frame-level curve

frame-level 图的横轴是 latent token-frame 相对距离

\[ \Delta_f = 0,1,\dots,20. \]

当前实现画两条曲线：

1. `temporal_axis_only`

\[ K_{\text{temp-only}}(\Delta_f) = \frac{1}{d_f/2}\sum_{k=0}^{d_f/2-1}\cos(\Delta_f \theta^{(f)}_k). \]

它只看 temporal 轴本身的 RoPE 衰减。

2. `full_head_same_spatial`

\[ K_{\text{full}}(\Delta_f,0,0). \]

它表示在保持空间位置不变时，完整 head 子空间上的 RoPE 相干性随 frame distance 的变化。

当前代码会导出两个 temporal frame-level PDF：

- `rope_decay_curve_frame_level.pdf`
- `rope_decay_curve_temporal_frame_level.pdf`

这两份文件当前画的是同一组数学对象，只是后者的标题更明确地强调它是 temporal 可视化；前者保留是为了兼容旧的输出命名习惯。

### 6.2 Spatial axis curves

为了把空间衰减和时间衰减分开，当前实现还会在同一 frame 内分别画 height 轴和 width 轴上的曲线。

#### 6.2.1 Height-axis curve

固定

\[ \Delta_f = 0,\qquad \Delta_w = 0, \]

只让

\[ \Delta_h = 0,1,\dots,H_{\text{tok}}-1 \]

变化。

当前实现画两条曲线：

1. `height_axis_only`

\[K_{\text{height-only}}(\Delta_h)=\frac{1}{d_h/2}\sum_{k=0}^{d_h/2-1}\cos(\Delta_h \theta_k^{(h)}).\]

它只看 height 轴自身的 RoPE 衰减。

2. `full_head_same_frame_same_width`

\[K_{\text{full}}(0,\Delta_h,0).\]

它表示在保持 frame 和 width 位置不变时，完整 head 子空间上的 RoPE 相干性如何随 height offset 变化。

对应输出文件：

- `rope_decay_curve_spatial_height_axis.pdf`

#### 6.2.2 Width-axis curve

固定

\[\Delta_f = 0,\qquad \Delta_h = 0,\]

只让

\[\Delta_w = 0,1,\dots,W_{\text{tok}}-1\]

变化。

当前实现画两条曲线：

1. `width_axis_only`

\[ K_{\text{width-only}}(\Delta_w) = \frac{1}{d_w/2}\sum_{k=0}^{d_w/2-1}\cos(\Delta_w \theta_k^{(w)}). \]

它只看 width 轴自身的 RoPE 衰减。

2. `full_head_same_frame_same_height`

\[ K_{\text{full}}(0,0,\Delta_w). \]

它表示在保持 frame 和 height 位置不变时，完整 head 子空间上的 RoPE 相干性如何随 width offset 变化。

对应输出文件：

- `rope_decay_curve_spatial_width_axis.pdf`

### 6.3 Same-frame spatial heatmap and radial profile

为了直接看“一帧里的一个 anchor token 能看到多大的空间区域”，当前实现还增加了 same-frame spatial heatmap。

#### 6.3.1 Spatial center-anchor heatmap

当前实现选择同一帧内的中心 anchor token：

\[ x_{\mathrm{anchor}} = (f_{\ast}, h_{\ast}, w_{\ast}), \]

其中

\[ f_{\ast}=0,\qquad h_{\ast}=\left\lfloor \frac{H_{\text{tok}}}{2} \right\rfloor,\qquad w_{\ast}=\left\lfloor \frac{W_{\text{tok}}}{2} \right\rfloor. \]

然后在保持 frame 不变的条件下，对整张 token grid 上的每个位置

\[ (0,h,w) \]

计算

\[ K_{\text{heatmap}}(h,w) = K_{\text{full}}\bigl( 0,\; h-h_{\ast},\; w-w_{\ast} \bigr). \]

也就是说，这张图画的是：

- anchor 固定在同一帧内的中心位置
- 目标点遍历该帧全部空间坐标
- 每个像素值都是该目标点相对 anchor 的 RoPE-only full-head coherence

The corresponding output files are:

- `rope_decay_curve_spatial_center_heatmap.pdf`
- `rope_decay_curve_spatial_center_heatmap_lambda_h{lambda_h}_lambda_w{lambda_w}.pdf`

The second heatmap uses scaled spatial RoPE phases. The height-axis phase is evaluated as \(\lambda_h\Delta_h\omega_h\), and the width-axis phase is evaluated as \(\lambda_w\Delta_w\omega_w\). The temporal phase is unchanged. Its colorbar uses the same lower and upper limits as `rope_decay_curve_spatial_center_heatmap.pdf` so that the two heatmaps are visually comparable.

summary 里的 `spatial_anchor_token_index` 保存的正是

\[ (h_{\ast}, w_{\ast}). \]

#### 6.3.2 Spatial radial profile

基于上面的 heatmap，当前实现进一步定义离散半径

\[ r(h,w) = \operatorname{round} \left( \sqrt{(h-h_{\ast})^2 + (w-w_{\ast})^2} \right). \]

对每个整数半径 \(\rho\)，把所有满足

\[ r(h,w)=\rho \]

的空间位置收集起来，记为

\[ \mathcal{S}_{\rho}. \]

然后定义径向平均曲线：

\[ K_{\text{radial}}(\rho) = \frac{1}{|\mathcal{S}_{\rho}|} \sum_{(h,w)\in\mathcal{S}_{\rho}} K_{\text{heatmap}}(h,w). \]

This curve answers how the same-frame spatial coherence envelope changes when spatial direction is marginalized and only center distance is retained. The figure contains two curves: `original RoPE` for the unscaled Wan2.1 RoPE basis, and `RoPE with λ_h=..., λ_w=...` for the auxiliary spatial phases \(\lambda_h\Delta_h\omega_h\) and \(\lambda_w\Delta_w\omega_w\).

The corresponding output file is:

- `rope_decay_curve_spatial_radial_profile.pdf`

### 6.4 Token-level curve

token-level 图的横轴是 flattened video-token 相对距离。先定义 flatten index

\[ \ell(f,h,w)=f(H_{\text{tok}}W_{\text{tok}})+hW_{\text{tok}}+w. \]

于是 token 序列长度为 \(L\)，flatten 后的 token index 属于

\[ \{0,1,\dots,L-1\}. \]

当前图的横轴是

\[ \delta = 0,1,\dots,L-1. \]

这里必须区分两种完全不同的定义。

### 6.4.1 当前实现：canonical anchor curve

当前实现固定 anchor token 为

\[ x_0=(0,0,0), \]

并取唯一满足

\[ \ell(x_\delta)=\delta \]

的 token

\[ x_\delta=(f_\delta,h_\delta,w_\delta). \]

然后画

\[ K_{\text{anchor}}(\delta) = K_{\text{full}}(f_\delta,\;h_\delta,\;w_\delta). \]

这正是代码里通过

\[ \Delta_f = \left\lfloor \frac{\delta}{H_{\text{tok}}W_{\text{tok}}} \right\rfloor, \qquad r = \delta \bmod (H_{\text{tok}}W_{\text{tok}}), \]

\[ \Delta_h = \left\lfloor \frac{r}{W_{\text{tok}}} \right\rfloor, \qquad \Delta_w = r \bmod W_{\text{tok}} \]

构造出来的 canonical 位移。

也就是说，当前图画的是：

- 固定一个 anchor
- 对每个 flatten distance \(\delta\)
- 只取一个代表性 target token
- 画这一个 pair 的 RoPE kernel

### 6.4.2 你提到的另一种定义：pair-average curve

另一种更“统计化”的定义是：对于每个 flatten distance \(\delta\)，把所有满足

\[ \ell(y)-\ell(x)=\delta \]

的 token pair 全部收集起来，记为

\[ \mathcal{P}_\delta = \{(x,y): \ell(y)-\ell(x)=\delta\}. \]

然后定义 pair-average kernel：

\[ K_{\text{pair-avg}}(\delta) = \frac{1}{|\mathcal{P}_\delta|} \sum_{(x,y)\in\mathcal{P}_\delta} K_{\text{full}}(y_f-x_f,\; y_h-x_h,\; y_w-x_w). \]

这才叫“对所有同样 flatten distance 的 token pair 做平均”。

### 6.4.3 为什么这两者不一样

因为给定同一个 flatten distance \(\delta\)，三轴位移

\[ (\Delta_f,\Delta_h,\Delta_w) \]

并不唯一。

例如固定 \(W_{\text{tok}}=52\)，当 \(\delta=1\) 时：

- 在同一行内部，相邻 token pair 可能对应
\[ (\Delta_f,\Delta_h,\Delta_w)=(0,0,1) \]
- 但在一行末尾跨到下一行开头时，也可能对应
\[ (\Delta_f,\Delta_h,\Delta_w)=(0,1,-51) \]
- 甚至在 frame 边界附近，还可能跨到下一帧

因此：

- `canonical anchor curve` 只选其中一个代表性分解
- `pair-average curve` 则对所有可能分解做平均

当前实现画的是前者，不是后者。

## 7. 当前实验的局限

当前实现有意保持简单，因此它画的是 RoPE-only kernel proxy，而不是：

- 数据分布下真实 query/key 的统计平均
- 加入内容向量协方差后的完整 attention logit 期望
- 对同一 flatten distance 的所有 token pairs 做 Monte Carlo 或全枚举平均

所以这个实验回答的是：

- RoPE 本身在 Wan2.1 的三轴分配下，随着相对距离变化，会如何破坏相位相干性

它不直接回答：

- 模型实际注意力权重一定如何衰减

换句话说，这是一个结构先验曲线，不是经验 attention 曲线。

## 8. Outputs

- `rope_decay_curve_frame_level.pdf`
- `rope_decay_curve_temporal_frame_level.pdf`
- `rope_decay_curve_spatial_height_axis.pdf`
- `rope_decay_curve_spatial_width_axis.pdf`
- `rope_decay_curve_spatial_center_heatmap.pdf`
- `rope_decay_curve_spatial_center_heatmap_lambda_h{lambda_h}_lambda_w{lambda_w}.pdf`
- `rope_decay_curve_spatial_radial_profile.pdf`
- `rope_decay_curve_token_level.pdf`
- `rope_decay_curve_summary.json`

summary 里还会保存：

- `head_dim`
- `temporal_real_dim`
- `spatial_real_dim_per_axis`
- `latent_frames`
- `token_grid_height`
- `token_grid_width`
- `sequence_token_count`
- `spatial_anchor_token_index`
- `scaled_spatial_heatmap_path`
- `rope_modification_lambda_h`
- `rope_modification_lambda_w`

## 9. Reading Guide

If the temporal frame-level curve drops quickly, then:

- even before content enters the picture, RoPE phase mismatch already weakens long-range temporal coherence

If the spatial height/width curves drop quickly, then:

- same-frame spatial coherence is more local under the current 3-axis split

If the same-frame spatial heatmap is sharply concentrated around the anchor, then:

- RoPE alone already induces a strong locality prior inside one frame

If the token-level curve shows early oscillation or decay, then:

- long-range token pairs in flattened order encounter multi-frequency decorrelation quickly

特别要注意：

- curve 可能不是单调下降的
- RoPE 的典型现象是“整体趋于去相干，同时伴有多频率振荡”

因此这里的“decay”更准确地说是：

- 平均相干性 envelope 的下降
- 而不是严格单调的指数衰减
