# `self_attention_distribution` Technical Note

## 1. Motivation

这个实验研究 self-attention 的分布本身，而不是只看时间距离直方图。

它回答两个问题：

1. 对于位于 object 区域内的 query tokens，self-attention 更偏向看其他帧里的 object 区域，还是看非 object 区域？
2. 如果不只看 object，而看全局采样的 query tokens，它们对不同时间偏移 `dt` 的注意力分布是什么样的？这种分布在序列前端、中段、后端是否不同？

## 2. Reference Object Support

实验先复用 `cross_attention_token_viz` 的输出，在 `(reference_step, reference_layer)` 处取目标 object words 的 head-mean cross-attention map：

\[ A^{\mathrm{ref}} \in \mathbb{R}_{\ge 0}^{F \times H \times W}. \]

其中：

- \(F\) 是 latent frame 数
- \(H, W\) 是 token-grid 高宽

当前实现中，reference support 的构造已经改成与 `head_evolution` 一致的标准流程：

1. 先对 reference map 做 winsorize + despike 预处理
2. 再在预处理后的 map 上提 reference center
3. 最后围绕该 center 构建 object support mask

记预处理前 reference map 为

\[ A^{\mathrm{ref}}, \]

预处理后的 map 为

\[ \tilde A^{\mathrm{ref}}. \]

预处理步骤是：

1. winsorize：
   - 对每一帧，把高于 `reference_preprocess_winsorize_quantile` 的值截到该分位点
2. despike：
   - 对每一帧取高值 mask，阈值由 `reference_preprocess_despike_quantile` 决定
   - 提取高值连通域
   - 删除面积小于 `reference_preprocess_min_component_area` 的小连通域

因此，后续的 reference center 和 object support mask 都基于 \(\tilde A^{\mathrm{ref}}\)，而不是直接基于原始 \(A^{\mathrm{ref}}\)。

然后像 `head_evolution` 一样，从每一帧里提取一个 reference center：

- `peak`
- `centroid`
- `geometric_center`

得到参考轨迹

\[ \mathcal{C}^{\mathrm{ref}} = \{c_f^{\mathrm{ref}}\}_{f=1}^{F}, \qquad c_f^{\mathrm{ref}} = (y_f^{\mathrm{ref}}, x_f^{\mathrm{ref}}). \]

再围绕每一帧的参考中心构建圆盘支撑区域

\[ M_f^{\mathrm{obj}}(y,x) \in \{0,1\}, \]

把所有帧拼起来得到 object support mask

\[ M^{\mathrm{obj}} \in \{0,1\}^{F \times H \times W}. \]

这个 mask 后面既用于选 query tokens，也用于定义 key 端的 object / non-object 区域。

补充说明：

- 如果 `reference_center_mode = geometric_center`，那么 `reference_center_power` 不直接参与最终 center 坐标计算
- 但 `reference_center_quantile` 仍然会影响“峰值所在主连通域”的定义
- 所以即使在 `geometric_center` 模式下，`reference_center_quantile` 依然有作用

## 3. Baseline Self-Attention

对 step \(s\)、layer \(\ell\)、head \(h\)，记 query / key logits 为

\[ \ell_{h}(i,j)=\frac{\langle q_{i,h}, k_{j,h}\rangle}{\sqrt d}, \]

其中：

- \(i\) 是 query token 索引
- \(j\) 是 key token 索引
- \(d\) 是 head dimension

attention 概率为

\[ \alpha_h(i,j)=\operatorname{softmax}_{j}\big(\ell_h(i,j)\big). \]

Let the latent token grid for one sample be

\[ \mathcal{T}=\{(f,y,x)\mid 0 \le f < F,\; 0 \le y < H,\; 0 \le x < W\}. \]

Then the key index \(j\) ranges over the entire video-token sequence \(\mathcal{T}\), not only over other frames. Therefore:

- the key set includes the query frame itself,
- the key set also includes the query token itself when \(j=i\),
- the diagonal self-copy term is one ordinary entry inside the same-frame block.

We do not save the full \(L \times L\) attention matrix explicitly. Instead, for each selected query subset we recompute the logits against all video keys in \(\mathcal{T}\), apply the exact softmax over that full key set, and then aggregate the resulting probability mass into frame-level summaries.

## 4. Object-Region Query Analysis

### 4.1 Query token selection

在若干个均匀采样的 query frames 上，选出所有落在 object support 里的 token：

\[ Q_f^{\mathrm{obj}}=\{(f,y,x)\mid M_f^{\mathrm{obj}}(y,x)=1\}. \]

这里的 `query_frame` 指的是 latent token frame，不是最终导出视频的 RGB frame。

如果本次采样的 latent-frame 数是 \(F\)，那么实际参与 object-query 统计的 query frames 个数至多为

\[ \min(F,\texttt{distribution\_query\_frame\_count}). \]

如果某一帧内 token 太多，则按近似均匀间隔下采样到上限 `distribution_object_query_token_limit_per_frame`。

- `distribution_object_query_token_limit_per_frame = 0`
  - 不设上限，保留该 query frame 内 support 区域的所有 token
- `distribution_object_query_token_limit_per_frame > 0`
  - 每个 query frame 最多保留这么多个 object queries

### 4.2 Key-side object / non-object mass

对一个 query token \(i \in Q_f^{\mathrm{obj}}\)，其在 key frame \(f'\) 上的总注意力质量定义为

\[ A_{h}^{\mathrm{frame}}(i,f')=\sum_{y=1}^{H}\sum_{x=1}^{W}\alpha_h\big(i,(f',y,x)\big). \]

其中落在 reference object support 内的质量定义为

\[ A_{h}^{\mathrm{obj}}(i,f')= \sum_{y=1}^{H}\sum_{x=1}^{W} \alpha_h\big(i,(f',y,x)\big)\,M_{f'}^{\mathrm{obj}}(y,x). \]

non-object 质量为

\[ A_{h}^{\mathrm{nonobj}}(i,f')= A_{h}^{\mathrm{frame}}(i,f')-A_{h}^{\mathrm{obj}}(i,f'). \]

object fraction 定义为

\[ R_h(i,f')= \frac{A_{h}^{\mathrm{obj}}(i,f')}{A_{h}^{\mathrm{frame}}(i,f')+\varepsilon}. \]

实验会对同一 `(step, layer, query_frame)` 内的所有 object queries 求平均，输出：

- `frame_mass`
- `object_mass`
- `nonobject_mass`
- `object_fraction`

并同时按 signed dt 聚合：

\[ \Delta t = f' - f. \]

这样就能看 object query 是更偏向看别的帧里的 object 区域，还是更偏向看背景。

### 4.3 什么叫“绝对 `(query_frame, key_frame)` 统计”

`self_attention_distribution_object_rows.csv` 里保存的是下面这种量：

- 固定一个 diffusion `step`
- 固定一个 DiT `layer`
- 固定一个 attention `head`
- 固定一个 object-side `query_frame = f`
- 固定一个 key-side `key_frame = f'`

然后把该 `query_frame=f` 中所有被选中的 object queries 的统计量做平均，得到一行：

\[ (\texttt{step}, \texttt{layer}, \texttt{head}, \texttt{query\_frame}=f, \texttt{key\_frame}=f'). \]

这一行里会记录：

- `frame_mass`
  - query 对整个 key frame \(f'\) 的总注意力质量
- `object_mass`
  - 这部分质量中落入 object support 的部分
- `nonobject_mass`
  - 这部分质量中落在非 object 区域的部分
- `object_fraction`
  - `object_mass / frame_mass`

这里说“绝对”是为了强调：它保留了原始的 `(query_frame, key_frame)` 二元索引，没有先压缩成 `dt = key_frame - query_frame`。

所以这份 CSV 不是时间差统计，而是 query-frame 到 key-frame 的二维配对统计。

## 5. Global Query Analysis

### 5.1 Query token selection

对于若干个均匀采样的 query frames，每帧在整个 \(H \times W\) token 网格上均匀采样若干 query tokens，记为

\[ Q_f^{\mathrm{global}}. \]

每帧最多采样 `distribution_global_query_tokens_per_frame` 个 token。

这里同样只在 `distribution_query_frame_count` 个均匀采样的 latent query frames 上取样，而不是在全部 \(F\) 个 query frames 上都取。

### 5.2 Signed-dt frame-mass distribution

对于任意全局 query token \(i \in Q_f^{\mathrm{global}}\)，定义其对 key frame \(f'\) 的总注意力质量

\[ A_h^{\mathrm{global}}(i,f')= \sum_{y=1}^{H}\sum_{x=1}^{W}\alpha_h\big(i,(f',y,x)\big). \]

再按 signed dt

\[ \Delta t=f'-f \]

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

这些图默认保存在：

\[ \texttt{output\_dir/self\_attention\_distribution\_plots/step\_xxx/layer\_xx/} \]

如果打开逐 head 可视化开关：

- `self_attention_distribution_plot_per_head=True`

则还会额外生成：

\[ \texttt{output\_dir/self\_attention\_distribution\_plots/step\_xxx/layer\_xx/head\_xx/} \]

下面同名的 heatmap / curve 会在该目录下按单个 head 输出。

另外现在还支持：

- `self_attention_distribution_plot_only_from_csv=True`
  - 直接复用 `output_dir` 下已有的：
    - `self_attention_distribution_object_rows.csv`
    - `self_attention_distribution_object_dt_rows.csv`
    - `self_attention_distribution_global_dt_rows.csv`
  - 仅重绘 plots，不重新跑 self-attention 采集

- `self_attention_distribution_skip_existing_plots=True`
  - 如果目标 plot 路径已经存在，就直接跳过，不覆盖旧图

## 7. Plot Shapes

### 7.1 `object query-key heatmaps` 画的是什么

它不是对所有 token 画一个 \(L \times L\) attention heatmap。

它画的是一个更小的 frame-level summary heatmap：

- 横轴：`key_frame`
- 纵轴：`query_frame`
- 每个格子：对该 `(query_frame, key_frame)` 的统计量，在同一 `(step, layer)` 下再对 heads 做平均后的结果

因此热图尺寸是

\[ (\#\text{sampled query frames}) \times (\#\text{all key frames}). \]

更具体地说：

- 行数至多是 `distribution_query_frame_count`
- 列数是 latent-frame 数 \(F\)

例如 Wan2.1 T2V 常见设置 `frame_num=81`，而 VAE stride 在时间维是 4，则 latent-frame 数通常是

\[ F = \frac{81-1}{4} + 1 = 21. \]

如果 `distribution_query_frame_count=8`，那么常见的 heatmap 尺寸更接近

\[ 8 \times 21, \]

而不是最终视频帧意义下的 \(81 \times 81\)，更不是 token-level 的 \(L \times L\)。

当前实现会分别画：

- `object_query_key_heatmap_object_fraction.pdf`
- `object_query_key_heatmap_object_mass.pdf`

前者强调“看向 object 的比例”，后者强调“看向 object 的绝对质量”。

默认情况下，这两张图是在固定 `(step, layer)` 后，对所有 heads 做平均再画。

如果设置：

- `self_attention_distribution_plot_per_head=True`

那么就会额外对每个 head 单独画一组对应的图。

### 7.2 `object dt curves`

这张图把 `object_rows.csv` 里原本按 `(query_frame, key_frame)` 保存的统计，进一步按

\[ dt = key\_frame - query\_frame \]

聚合，画出：

## 8. Computation Notes

- `self_attention_distribution_plot_per_head`
  - 主要增加的是绘图文件数量，不会重新跑一套 self-attention probe 数学
  - 因为当前 CSV 本身已经是逐 `head` 保存的
  - 所以额外成本主要是：
    - 更多的 pandas / matplotlib 分组与出图
    - 更多的 PDF 文件写盘
  - 不是额外做一遍 qk attention 计算

如果当前选了：

- \(S\) 个 diffusion steps
- \(L\) 个 layers
- \(H\) 个 heads

那么 layer-level 默认大约是每个 `(step, layer)` 生成 4 张图：

- 2 张 object heatmap
- 1 张 object dt curve
- 1 张 global dt curve

逐 head 额外大约会再乘一个 \(H\)。

以 Wan2.1 T2V 1.3B 常见的 12 heads 为例：

- 每个 `(step, layer)` 默认约 4 张图
- 开启逐 head 后，额外约 \(4 \times 12 = 48\) 张图

## 9. Early Stop

如果设置：

- `self_attention_distribution_stop_after_last_probe_step=True`

那么程序会在最后一个请求的 probe step 采集完成后，立刻停止后续扩散。

例如：

- `self_attention_distribution_steps="1,2,3,4,5,6"`
- `self_attention_distribution_stop_after_last_probe_step=True`

那么在 step 6 采集完成后，不再继续跑 step 7 到 step 50。

注意：

- 这会跳过最终完整视频的生成
- 因为此时没有继续走完扩散，也就没有最终 video sample
- `summary.json` 里会记录：
  - `early_stop_triggered`
  - `early_stop_completed_step`
  - `early_stop_reason`

## 10. CSV-Only Replot

`self_attention_distribution` 现在的三类主要 plot：

- object query-key heatmaps
- object dt curves
- global dt curves

都可以仅由已有 CSV 重新绘制，因为它们本来就是对 CSV 里的 frame-level / dt-level 数值做聚合可视化。

所以如果你只是改了绘图样式、颜色、排版或分组逻辑，不需要重新跑 attention probe。

- `frame_mass`
- `object_mass`
- `nonobject_mass`
- `object_fraction`

随 signed `dt` 的变化曲线。

The current implementation normalizes each signed-`dt` bin by the number of query tokens that actually have a valid `(query_frame, key_frame)` pair for that `dt`. In particular, large \(|dt|\) bins near sequence boundaries are no longer divided by the total query count of the whole layer.

### 7.3 `global dt curves`

这张图不再区分 object / non-object，只看全局均匀采样 queries 对不同 signed `dt` 的 frame-level attention mass 分布。

同一张图会画多个 query bucket：

- `all`
- `early`
- `middle`
- `late`

用来检查 query 所处序列位置是否影响时间偏移偏好。

The same signed-`dt` normalization rule is used here as well: each bin is averaged over the subset of sampled global queries for which that temporal offset exists.

## 8. Key Parameters

### 8.1 `SELF_ATTENTION_DISTRIBUTION_QUERY_FRAME_COUNT`

控制从全部 latent frames 中均匀抽取多少个 query frames。

设 latent-frame 数为 \(F\)，则实际使用的 query-frame 个数是

\[ \min(F,\texttt{SELF\_ATTENTION\_DISTRIBUTION\_QUERY\_FRAME\_COUNT}). \]

影响：

- 值越大，时间覆盖越细
- 值越小，计算更省，heatmap 行数更少

### 8.2 `SELF_ATTENTION_DISTRIBUTION_GLOBAL_QUERY_TOKENS_PER_FRAME`

只用于 global-query 分析。

对每个被选中的 query frame，在整张 \(H \times W\) token 网格上均匀采样这么多个 query tokens。

影响：

- 值越大，global dt 分布估计越稳定
- 值越小，计算更快，但方差更大

### 8.3 `SELF_ATTENTION_DISTRIBUTION_OBJECT_QUERY_TOKEN_LIMIT_PER_FRAME`

只用于 object-region query 分析。

对于每个被选中的 query frame，先找出 support 区域中的所有 token，然后：

- 若值为 `0`，保留全部 object queries
- 若值为正整数，只保留这么多个，按近似均匀间隔下采样

这个参数主要是用来控制 object support 很大时的计算量。

### 8.4 Reference-support parameters

- `SELF_ATTENTION_DISTRIBUTION_REFERENCE_PREPROCESS_WINSORIZE_QUANTILE`
  - reference map 的 winsorize 分位点
- `SELF_ATTENTION_DISTRIBUTION_REFERENCE_PREPROCESS_DESPIKE_QUANTILE`
  - reference map 的高值掩码分位点，用于 despike
- `SELF_ATTENTION_DISTRIBUTION_REFERENCE_PREPROCESS_MIN_COMPONENT_AREA`
  - reference map 去尖刺时保留的最小连通域面积
- `SELF_ATTENTION_DISTRIBUTION_REFERENCE_CENTER_MODE`
  - `peak` / `centroid` / `geometric_center`
- `SELF_ATTENTION_DISTRIBUTION_REFERENCE_CENTER_POWER`
  - 只在 `centroid` 下直接影响加权质心；`geometric_center` 下不直接参与最终 center 坐标
- `SELF_ATTENTION_DISTRIBUTION_REFERENCE_CENTER_QUANTILE`
  - 用于定义峰值所在主连通域；即使 `center_mode=geometric_center`，它仍然会影响最终 center

## 9. How To Read

### 9.1 object query-key heatmap

横轴是 key frame，纵轴是 query frame。

- 如果 `object_fraction` 热力图沿着接近对角或某条轨迹相关带状区域较亮，说明 object-region queries 确实更倾向于去看其他帧的 object 区域。
- 如果 `object_mass` 高但 `object_fraction` 不高，说明 query 确实看向了 object，但同时也把不少质量给了背景。
- 如果热力图整体偏暗，而 `nonobject_mass` 很高，说明这些 queries 虽然来自 object 区域，但 self-attention 仍大量分配到了背景。

### 9.2 object dt curves

- `object_mass` 高：表示 query 更关注其他帧中的 object 区域
- `nonobject_mass` 高：表示 query 更多看向背景
- `object_fraction` 高：表示在给定 key frame 的总注意力质量里，落入 object support 的比例高
- 若曲线在远距离 `|dt|` 处明显下降，说明存在远距离衰减

### 9.3 global dt curves

- `all`：总体 signed-dt 分布
- `early/middle/late`：分别看序列前段、中段、后段 query token 的时间偏移偏好

如果三者差异很大，说明 self-attention 的时间分布显著依赖 query 在序列中的位置；如果差异很小，说明这种位置依赖较弱。
