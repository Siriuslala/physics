# `self_attention_distribution` Technical Note

## 1. Motivation

这个实验研究 self-attention 的分布本身，而不是只看时间距离直方图。

它回答两个问题：

1. 对于位于 object 区域内的 query tokens，self-attention 更偏向看其他帧里的 object 区域，还是看非 object 区域？
2. 如果不只看 object，而看全局采样的 query tokens，它们对不同时间偏移 `dt` 的注意力分布是什么样的？这种分布在序列前端、中段、后端是否不同？

## 2. Reference Object Support

实验先复用 `cross_attention_token_viz` 的输出，在 `(reference_step, reference_layer)` 处取目标 object words 的 head-mean cross-attention map：

\[
A^{\mathrm{ref}} \in \mathbb{R}_{\ge 0}^{F \times H \times W}.
\]

其中：

- \(F\) 是 latent frame 数
- \(H, W\) 是 token-grid 高宽

然后像 `head_evolution` 一样，从每一帧里提取一个 reference center：

- `peak`
- `centroid`
- `geometric_center`

得到参考轨迹

\[
\mathcal{C}^{\mathrm{ref}} = \{c_f^{\mathrm{ref}}\}_{f=1}^{F},
\qquad
c_f^{\mathrm{ref}} = (y_f^{\mathrm{ref}}, x_f^{\mathrm{ref}}).
\]

再围绕每一帧的参考中心构建圆盘支撑区域

\[
M_f^{\mathrm{obj}}(y,x) \in \{0,1\},
\]

把所有帧拼起来得到 object support mask

\[
M^{\mathrm{obj}} \in \{0,1\}^{F \times H \times W}.
\]

这个 mask 后面既用于选 query tokens，也用于定义 key 端的 object / non-object 区域。

## 3. Baseline Self-Attention

对 step \(s\)、layer \(\ell\)、head \(h\)，记 query / key logits 为

\[
\ell_{h}(i,j)=\frac{\langle q_{i,h}, k_{j,h}\rangle}{\sqrt d},
\]

其中：

- \(i\) 是 query token 索引
- \(j\) 是 key token 索引
- \(d\) 是 head dimension

attention 概率为

\[
\alpha_h(i,j)=\operatorname{softmax}_{j}\big(\ell_h(i,j)\big).
\]

我们不显式保存完整 \(L \times L\) attention matrix，而是在 probe 时对选定 query 集合直接计算它们对全部 key 的 attention 分布。

## 4. Object-Region Query Analysis

### 4.1 Query token selection

在若干个均匀采样的 query frames 上，选出所有落在 object support 里的 token：

\[
Q_f^{\mathrm{obj}}=\{(f,y,x)\mid M_f^{\mathrm{obj}}(y,x)=1\}.
\]

如果某一帧内 token 太多，则按近似均匀间隔下采样到上限 `distribution_object_query_token_limit_per_frame`。

### 4.2 Key-side object / non-object mass

对一个 query token \(i \in Q_f^{\mathrm{obj}}\)，其在 key frame \(f'\) 上的总注意力质量定义为

\[
A_{h}^{\mathrm{frame}}(i,f')=\sum_{y=1}^{H}\sum_{x=1}^{W}\alpha_h\big(i,(f',y,x)\big).
\]

其中落在 reference object support 内的质量定义为

\[
A_{h}^{\mathrm{obj}}(i,f')=
\sum_{y=1}^{H}\sum_{x=1}^{W}
\alpha_h\big(i,(f',y,x)\big)\,M_{f'}^{\mathrm{obj}}(y,x).
\]

non-object 质量为

\[
A_{h}^{\mathrm{nonobj}}(i,f')=
A_{h}^{\mathrm{frame}}(i,f')-A_{h}^{\mathrm{obj}}(i,f').
\]

object fraction 定义为

\[
R_h(i,f')=
\frac{A_{h}^{\mathrm{obj}}(i,f')}
{A_{h}^{\mathrm{frame}}(i,f')+\varepsilon}.
\]

实验会对同一 `(step, layer, query_frame)` 内的所有 object queries 求平均，输出：

- `frame_mass`
- `object_mass`
- `nonobject_mass`
- `object_fraction`

并同时按 signed dt 聚合：

\[
\Delta t = f' - f.
\]

这样就能看 object query 是更偏向看别的帧里的 object 区域，还是更偏向看背景。

## 5. Global Query Analysis

### 5.1 Query token selection

对于若干个均匀采样的 query frames，每帧在整个 \(H \times W\) token 网格上均匀采样若干 query tokens，记为

\[
Q_f^{\mathrm{global}}.
\]

每帧最多采样 `distribution_global_query_tokens_per_frame` 个 token。

### 5.2 Signed-dt frame-mass distribution

对于任意全局 query token \(i \in Q_f^{\mathrm{global}}\)，定义其对 key frame \(f'\) 的总注意力质量

\[
A_h^{\mathrm{global}}(i,f')=
\sum_{y=1}^{H}\sum_{x=1}^{W}\alpha_h\big(i,(f',y,x)\big).
\]

再按 signed dt

\[
\Delta t=f'-f
\]

聚合，得到 head 对不同时间偏移的全局注意力分布。

为了处理“序列中间 token 和两端 token 最远可见距离不同”的问题，实验不只输出总体 `all`，还按 query frame 位置把 query 分成：

- `early`
- `middle`
- `late`

分别画出 signed-dt 曲线。

## 6. Outputs

- `self_attention_distribution_object_rows.csv`
  - object queries 的绝对 `(query_frame, key_frame)` 统计
- `self_attention_distribution_object_dt_rows.csv`
  - object queries 的 signed-dt 统计
- `self_attention_distribution_global_dt_rows.csv`
  - global queries 的 signed-dt 统计
- `self_attention_distribution_reference_support.csv`
  - 参考中心、半径、支撑面积
- `self_attention_distribution_plots/`
  - object query-key heatmaps
  - object dt curves
  - global dt curves
- `self_attention_distribution_summary.json`

## 7. How To Read

### 7.1 object query-key heatmap

横轴是 key frame，纵轴是 query frame。

- 如果 `object_fraction` 热力图沿着接近对角或某条轨迹相关带状区域较亮，说明 object-region queries 确实更倾向于去看其他帧的 object 区域。
- 如果热力图整体偏暗，而 `nonobject_mass` 很高，说明这些 queries 虽然来自 object 区域，但 self-attention 仍大量分配到了背景。

### 7.2 object dt curves

- `object_mass` 高：表示 query 更关注其他帧中的 object 区域
- `nonobject_mass` 高：表示 query 更多看向背景
- `object_fraction` 高：表示在给定 key frame 的总注意力质量里，落入 object support 的比例高

### 7.3 global dt curves

- `all`：总体 signed-dt 分布
- `early/middle/late`：分别看序列前段、中段、后段 query token 的时间偏移偏好

如果三者差异很大，说明 self-attention 的时间分布显著依赖 query 在序列中的位置；如果差异很小，说明这种位置依赖较弱。
