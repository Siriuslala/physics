# `trajectory_consensus_dynamics` Technical Note

## 1. Motivation

我们研究的是视频模型的motion planning，这个阶段发生在扩散早期的几步里，中后期主要是细节填充。视频 latent 对condition里object token的cross attention map可以显示模型规划的运动路径。早期的cross attn heads的pattern通常比较混乱，似乎每个head有很多object位置的candidates，或者不同heads有不同的object candidate position。但是最后都会收敛到某个确定的位置。我们这个实验的终极目标，就是解释这个现象。一方面，将这个过程更为详细地分解开来，探究这些head是如何“达成共识”（trajectory disambiguation）的？究竟是如何从混乱叠加态收敛到某一个具体的运动轨迹的？从模型结构上来看，self attention，cross attention，FFN和Layernorm是模型最主要的模块，其中FFN在本研究中不太关注，主要是其他三个。cross attn 负责将 condition 中的语义传给 video latent，但是并不确定 object 在哪一帧里究竟存在于什么位置。而 self attention 则负责协调各帧来捕捉帧间关系，从而选出一条合理的轨迹。本研究会将这些模块分开来分别研究。

`trajectory_consensus_dynamics` is a next-generation experiment for early motion planning in Wan2.1-T2V.

The central question is not merely whether object-related cross-attention heads eventually become similar. That empirical fact is already visible in existing trajectory and reference-distance analyses. The real question is:

- why do several early candidate trajectories coexist at first;
- why are some candidates later suppressed;
- why do many heads eventually support the same winner trajectory;
- and why can a small seed change move the model toward a different final trajectory.

This experiment is designed to study that transition explicitly.

It treats early motion planning as a **candidate-competition process**:

1. cross-attention proposes object-location candidates;
2. self-attention evaluates whether candidates across frames can form a mutually compatible motion path;
3. the residual stream accumulates these preferences across layers and diffusion steps;
4. small early score differences can be amplified into different final winner trajectories.

The experiment therefore focuses on four concrete objects:

1. head contribution to the final velocity prediction;
2. frame-wise candidate regions extracted from cross-attention maps;
3. candidate-level consensus and winner-gap dynamics;
4. self-attention coupling between candidate regions across latent frames.

## 2. Why This Is a New Experiment

`head_trajectory_dynamics` is an offline cross-attention-map analysis. Its main objects are:

- head-to-head distances;
- consensus curves;
- reference distances;
- attractor-style summaries.

That design is useful for descriptive dynamics, but it is not the right engineering container for the present goal.

The present experiment additionally requires:

- explicit decomposition of attention-module outputs into per-head writes;
- contribution analysis from early heads to the final `v_pred`;
- candidate-region extraction and visualization;
- self-attention region-to-region coupling analysis;
- targeted head-wise interventions for both cross-attention and self-attention;
- trajectory-graph decoding by dynamic programming.

Because these additions change the data model, the cached tensors, and the intervention logic, this experiment should be implemented as a **new module** rather than as another branch inside `head_trajectory_dynamics`.

## 3. Wan2.1 Architecture Segment Relevant to `v_pred`

This section maps the mathematical notation directly to the Wan2.1-T2V code path in `projects/Wan2_1/wan/modules/model.py`.

### 3.1 One transformer block

For one diffusion step and one transformer block \(\ell\), let the input residual state be

\[
X^{(\ell)} \in \mathbb{R}^{L \times C},
\]

where:

- \(L\) is the patchified video-token sequence length;
- \(C\) is the transformer hidden dimension.

The Wan block applies self-attention first:

\[ Y_{\mathrm{sa}}^{(\ell)} = \operatorname{SelfAttn}^{(\ell)}\!\left(\operatorname{Norm}_1^{(\ell)}(X^{(\ell)}) \odot (1 + e_1^{(\ell)}) + e_0^{(\ell)}\right). \]

then writes it back through the residual connection with timestep-dependent modulation:

\[ \widetilde X^{(\ell)} = X^{(\ell)} + Y_{\mathrm{sa}}^{(\ell)} \odot e_2^{(\ell)}. \]

Next, the block applies text-conditioned cross-attention:

\[ Y_{\mathrm{ca}}^{(\ell)} = \operatorname{CrossAttn}^{(\ell)}\!\left(\operatorname{Norm}_3^{(\ell)}(\widetilde X^{(\ell)}),\ \text{context}\right). \]

and writes it back as

\[ \widehat X^{(\ell)} = \widetilde X^{(\ell)} + Y_{\mathrm{ca}}^{(\ell)}. \]

Finally, the block applies the FFN:

\[ Y_{\mathrm{ffn}}^{(\ell)} = \operatorname{FFN}^{(\ell)}\!\left(\operatorname{Norm}_2^{(\ell)}(\widehat X^{(\ell)}) \odot (1 + e_4^{(\ell)}) + e_3^{(\ell)}\right). \]

and outputs

\[ X^{(\ell+1)} = \widehat X^{(\ell)} + Y_{\mathrm{ffn}}^{(\ell)} \odot e_5^{(\ell)}. \]

These equations correspond exactly to the order in `WanAttentionBlock.forward`.

### 3.2 Final model head after the last transformer block

After the last transformer block \(L_{\max}\), Wan2.1 does **not** apply another transformer stack. The tail is short:

1. `self.head.norm`
2. timestep-conditioned modulation inside `Head.forward`
3. `self.head.head`, a linear layer from hidden dimension \(C\) to patch output dimension
4. `unpatchify`

Formally, let the final hidden sequence be

\[ X^{(\mathrm{final})} \in \mathbb{R}^{L \times C}. \]

The head module produces patch outputs

\[ O = \operatorname{Head}(X^{(\mathrm{final})}, e) \in \mathbb{R}^{L \times P C_{\mathrm{out}}}. \]

where:

- \(P = p_t p_h p_w\) is the patch volume,
- \(C_{\mathrm{out}}\) is the output channel count of the diffusion model.

After `unpatchify`, the model returns

\[ v_{\mathrm{pred}} \in \mathbb{R}^{C_{\mathrm{out}} \times F \times H \times W}. \]

So, from the last transformer block to `v_pred`, the Wan2.1 path is:

\[ \text{last block output} \rightarrow \text{Head norm + modulation + linear} \rightarrow \text{unpatchify} \rightarrow v_{\mathrm{pred}}. \]

## 4. Per-Head Write and Downstream Contribution

### 4.1 Per-head write in one attention module

Inside one attention module with \(H\) heads, let the post-attention, pre-output-projection tensor of head \(h\) be

\[ Z_h \in \mathbb{R}^{L \times d_h}. \]

where \(d_h = C / H\).

The output projection `o` is linear. Therefore its weight can be partitioned by heads:

\[ W_O = [W_O^{(1)}, W_O^{(2)}, \dots, W_O^{(H)}], \qquad W_O^{(h)} \in \mathbb{R}^{d_h \times C}. \]

The hidden-state write contributed by head \(h\) is

\[ U_h = Z_h W_O^{(h)} \in \mathbb{R}^{L \times C}. \]

For self-attention in Wan2.1, the actual residual write is further modulated by \(e_2^{(\ell)}\):

\[ U^{\mathrm{sa}}_h = U_h \odot e_2^{(\ell)}. \]

For cross-attention, the residual add is unscaled:

\[ U^{\mathrm{ca}}_h = U_h. \]

These \(U_h\) tensors are the natural objects for contribution analysis, because they are exactly what enters the residual stream.

### 4.2 What the downstream computation means here

Suppose we analyze one head in block \(\ell\) at diffusion step \(s\).

If the source is a **self-attention head**, then after its write \(U^{\mathrm{sa}}_h\) enters the residual stream, the remaining computation still includes:

1. the cross-attention sublayer of the same block;
2. the FFN sublayer of the same block;
3. all later transformer blocks \(\ell+1, \dots, L_{\max}\);
4. the final `Head` module and `unpatchify`.

If the source is a **cross-attention head**, then after its write \(U^{\mathrm{ca}}_h\) enters the residual stream, the remaining computation includes:

1. the FFN sublayer of the same block;
2. all later transformer blocks \(\ell+1, \dots, L_{\max}\);
3. the final `Head` module and `unpatchify`.

This remaining computation is the **downstream computation** for that head.

### 4.3 Exact head ablation effect

The primary contribution definition in this experiment should be exact zero-ablation.

For one analyzed head \(h\), keep all clean activations fixed except that head's residual write, and define the downstream output as a function of the inserted head write:

\[ v_{s,\ell,h}(U). \]

Here:

- \(U\) is the write inserted at that head's residual insertion point;
- \(v_{s,\ell,h}(U)\) is the final `v_pred` after running the downstream computation from that point onward.

Then:

- the clean output is

\[ v^{\mathrm{clean}}_{s} = v_{s,\ell,h}(U_{s,\ell,h}); \]

- the ablated output is

\[ v^{\mathrm{ablate}}_{s,\ell,h} = v_{s,\ell,h}(0). \]

The exact head ablation effect is therefore

\[ \Delta v_{s,\ell,h} = v^{\mathrm{clean}}_{s} - v^{\mathrm{ablate}}_{s,\ell,h}. \]

Interpretation:

- if \(\Delta v_{s,\ell,h}\) is large, then removing this head substantially changes the final velocity prediction;
- if \(\Delta v_{s,\ell,h}\) is small, then this head has little causal effect under zero-ablation.

This exact ablation effect should be the main contribution quantity in the experiment.

### 4.4 First-order Taylor approximation of head ablation effect

Exact ablation requires one downstream rerun per head. To reduce cost, the experiment may also compute a first-order Taylor approximation.

Starting from

\[ \Delta v_{s,\ell,h} = v_{s,\ell,h}(U_{s,\ell,h}) - v_{s,\ell,h}(0), \]

take the first-order Taylor expansion of \(v_{s,\ell,h}(U)\) around the clean head write \(U_{s,\ell,h}\). Then

\[ v_{s,\ell,h}(0) \approx v_{s,\ell,h}(U_{s,\ell,h}) + \left. \frac{\partial v_{s,\ell,h}(U)}{\partial U} \right|_{U = U_{s,\ell,h}} (0 - U_{s,\ell,h}). \]

Rearranging gives

\[ \Delta v_{s,\ell,h} \approx \left. \frac{\partial v_{s,\ell,h}(U)}{\partial U} \right|_{U = U_{s,\ell,h}} U_{s,\ell,h}. \]

This approximation means:

- use the clean run as the expansion point;
- use the local gradient of the final `v_pred` with respect to the head write;
- multiply that gradient by the clean head write itself.

This is a fast approximation to the exact ablation effect and is suitable for large-scale head screening.

### 4.5 Direct final-head projection proxy

The user-proposed idea "map one head's output to a separate \(v_{\mathrm{pred}}'\), then compare it with the clean \(v_{\mathrm{pred}}\)" is valid and should be kept as a proxy.

Take one head write \(U_h\), map it through the final output head described in Section 3.2, and obtain

\[ v^{\prime,\mathrm{proj}}_{s,\ell,h}. \]

Then compare \(v^{\prime,\mathrm{proj}}_{s,\ell,h}\) with the clean \(v_{\mathrm{pred}}\).

This proxy is attractive because it is cheap and easy to interpret. However, it is not an exact decomposition for an arbitrary early head. There are two reasons:

- after the source head writes into the residual stream, the signal is still rewritten by later transformer blocks;
- the final `Head` module itself contains a `LayerNorm`, so the readout is not globally linear in one head write.

Therefore this quantity should be described as a **direct readout proxy**, not as the main causal contribution definition.

If the implementation uses this method, the recommended engineering choice is:

- reuse the clean run's final-head normalization statistics or the clean final hidden state as the readout anchor;
- then apply the same final linear readout and `unpatchify` path to the single-head signal.

This keeps the definition stable across heads.

### 4.6 Similarity and share for the direct projection proxy

If the direct projection proxy is used, let

\[ v^{\prime,\mathrm{proj}}_{s,\ell,h} \]

be the projected output of one head.

The most direct similarity measures are:

\[ \mathrm{ProjCos}(s,\ell,h) = \frac{\langle v^{\prime,\mathrm{proj}}_{s,\ell,h},\, v_{\mathrm{pred}} \rangle}{\|v^{\prime,\mathrm{proj}}_{s,\ell,h}\|_2 \, \|v_{\mathrm{pred}}\|_2}. \]

\[ \mathrm{ProjDot}(s,\ell,h) = \langle v^{\prime,\mathrm{proj}}_{s,\ell,h},\, v_{\mathrm{pred}} \rangle. \]

If one wants a relative share among analyzed heads, a practical definition is:

\[ \mathrm{ProjShare}(s,\ell,h) = \frac{\max\!\big(0, \mathrm{ProjDot}(s,\ell,h)\big)}{\sum_{j \in \mathcal{H}_{\mathrm{ana}}} \max\!\big(0, \mathrm{ProjDot}(s,\ell,j)\big) + \varepsilon}. \]

where \(\mathcal{H}_{\mathrm{ana}}\) is the set of analyzed heads.

Interpretation:

- `ProjCos` asks whether this head's projected output points in a direction similar to the clean `v_pred`;
- `ProjDot` asks how strongly this projected output aligns with the clean `v_pred`;
- `ProjShare` asks what fraction of the positive aligned projected contribution is carried by this head among the analyzed heads.

## 5. Head-Contribution Metrics

This experiment measures both cross-attention heads and self-attention heads.

Unless otherwise stated, the primary input to the following metrics is the exact head ablation effect

\[ \Delta v_{s,\ell,h} = v^{\mathrm{clean}}_{s} - v^{\mathrm{ablate}}_{s,\ell,h}. \]

If the experiment uses the first-order Taylor approximation instead of exact ablation, the same metric formulas may be reused after replacing \(\Delta v_{s,\ell,h}\) by its Taylor approximation from Section 4.4.

If the experiment uses the direct final-head projection proxy, then the dedicated proxy metrics from Section 4.6 should be reported separately rather than mixed into the exact-ablation tables.

### 5.1 Full-field similarity

Given one head-level contribution tensor \(\Delta v_{s,\ell,h}\) and the clean final prediction \(v_s\), define:

\[ \mathrm{CosFull}(s,\ell,h) = \frac{\langle \Delta v_{s,\ell,h}, v_s \rangle}{\|\Delta v_{s,\ell,h}\|_2 \, \|v_s\|_2}. \]

\[ \mathrm{DotFull}(s,\ell,h) = \langle \Delta v_{s,\ell,h}, v_s \rangle. \]

### 5.2 Object-masked similarity

Let the object region mask be

\[ M^{\mathrm{obj}} \in \{0,1\}^{F \times H \times W}. \]

In this experiment, the default object mask should reuse the existing project logic based on the head-mean object map at:

- reference step `50`
- reference layer `27`

The same reference-center and support-mask extraction pipeline already used in `head_evolution` and `self_attention_distribution` should be reused here.

Then define:

\[ \mathrm{CosObj}(s,\ell,h) = \frac{\langle M^{\mathrm{obj}} \odot \Delta v_{s,\ell,h},\, M^{\mathrm{obj}} \odot v_s \rangle}{\|M^{\mathrm{obj}} \odot \Delta v_{s,\ell,h}\|_2 \, \|M^{\mathrm{obj}} \odot v_s\|_2}. \]

\[ \mathrm{DotObj}(s,\ell,h) = \langle M^{\mathrm{obj}} \odot \Delta v_{s,\ell,h},\, M^{\mathrm{obj}} \odot v_s \rangle. \]

The experiment should implement all four metrics:

- `cos_full`
- `dot_full`
- `cos_obj`
- `dot_obj`

for both self-attention heads and cross-attention heads.

### 5.3 Early-alignment correlation plots

Let \(D^{\mathrm{ref}}_{s,\ell,h}\) be the existing reference distance from `head_trajectory_dynamics`.

The experiment should draw at least two scatter plots:

1. x-axis = early-alignment score based on negative AUC

\[ E^{\mathrm{auc}}_{\ell,h} = -\frac{1}{|S|}\sum_{s \in S} D^{\mathrm{ref}}_{s,\ell,h}. \]

2. x-axis = raw reference distance itself, for example the step-wise or step-averaged

\[ E^{\mathrm{raw}}_{\ell,h} = \frac{1}{|S|}\sum_{s \in S} D^{\mathrm{ref}}_{s,\ell,h}. \]

The y-axis should be one selected head-contribution metric, such as `cos_obj`.

### 5.4 Contribution visualizations

The experiment should generate:

- layer-wise curves:
  - average contribution over heads in each layer;
- layer-by-head heatmaps:
  - for one fixed step;
- step-by-head heatmaps:
  - for one fixed layer;
- step-by-layer heatmaps:
  - after averaging over heads;
- scatter plots:
  - early-alignment score vs head contribution.

These plots are needed to test whether high-contribution heads are stable over diffusion or only dominate in a narrow early window.

If direct projection is also enabled, the experiment should additionally draw the same plot families for:

- `ProjCos`
- `ProjDot`
- `ProjShare`

and should keep them visually separate from the exact-ablation metrics.

## 6. Candidate Region Extraction

Candidate regions are the most important intermediate object in this experiment.

### 6.1 Input map

For one step \(s\), layer \(\ell\), head \(h\), and latent frame \(f\), let the normalized object-token cross-attention map be

\[ P^{(s,\ell,h)}_f(y,x), \qquad \sum_{y,x} P^{(s,\ell,h)}_f(y,x) = 1. \]

### 6.2 Extraction procedure

The candidate extractor should avoid assuming that early maps are already a set of isolated bright points. It must support:

- one broad elongated high-response band;
- one band that later splits into several local maxima;
- several clearly separated blobs.

The recommended procedure is:

1. apply the same winsorize + despike preprocessing used by `head_evolution`;
2. optionally apply a small Gaussian smoothing;
3. threshold by a broad base quantile \(q_{\mathrm{base}}\) to obtain a large support set;
4. remove tiny connected components below area threshold \(a_{\min}\);
5. inside each broad component, raise the threshold through a sequence

\[ q_1 < q_2 < \cdots < q_R. \]

6. detect local peaks that survive across several threshold levels;
7. treat these stable peaks as seeds;
8. if multiple stable peaks exist, run watershed or seeded region growing inside the broad support component;
9. if no stable split exists, keep the whole broad component as one candidate region.

This procedure allows:

- one continuous trajectory band to remain one candidate when no stable internal split exists;
- the same band to split into two or three candidate regions only when the split is persistent and visually meaningful.

### 6.3 Candidate-region weights

Let the extracted candidate regions on frame \(f\) be

\[ R_{f,1}, R_{f,2}, \dots, R_{f,K_f}, \]

where \(K_f\) may vary across frames.

Then the soft mass assigned by head \((s,\ell,h)\) to region \(k\) is

\[ a_{s,\ell,h,f,k} = \sum_{(y,x)\in R_{f,k}} P^{(s,\ell,h)}_f(y,x). \]

These \(a_{s,\ell,h,f,k}\) values are the candidate weights used later for consensus and transmission analysis.

### 6.4 Candidate-region visualization

For each analyzed head, the experiment should generate a two-row visualization:

- Row 1:
  - raw attention map panels;
  - each frame uses its **own** color scale;
  - this should match the non-shared-scale setting in `cross_attention_token_viz`;
- Row 2:
  - binary candidate-region visualization;
  - each candidate region is shown as a mask panel;
  - the visual style may reuse the motion-planning-region rendering already available in the repository.

This visualization is required to verify that the extracted candidate regions are scientifically meaningful.

## 7. Candidate Consensus and Winner Gap

### 7.1 Layer-mean candidate weights

For one step \(s\), layer \(\ell\), frame \(f\), and region \(k\), define the layer-mean candidate weight

\[ \bar a_{s,\ell,f,k} = \frac{1}{H_\ell}\sum_{h=1}^{H_\ell} a_{s,\ell,h,f,k}, \]

where \(H_\ell\) is the number of heads in layer \(\ell\).

### 7.2 Winner gap

Let \(k_1\) and \(k_2\) be the top-1 and top-2 candidate indices under \(\bar a_{s,\ell,f,k}\). Define the winner gap

\[ G_{s,\ell,f} = \bar a_{s,\ell,f,k_1} - \bar a_{s,\ell,f,k_2}. \]

This quantity directly measures how strongly one candidate is winning over the nearest competitor.

### 7.3 Candidate entropy

For completeness, define the frame-wise candidate entropy

\[ H^{\mathrm{cand}}_{s,\ell,f} = -\sum_{k=1}^{K_f} \bar a_{s,\ell,f,k}\log\!\big(\bar a_{s,\ell,f,k} + \varepsilon\big). \]

However, the primary object of interpretation should be the winner gap \(G_{s,\ell,f}\), not entropy alone.

## 8. Cross-Attention and Self-Attention Module Attribution

This experiment studies only:

- cross-attention;
- self-attention.

FFN is not a primary focus here.

### 8.1 Head-wise intervention

For one selected source head \(h\) in one selected step \(s\) and layer \(\ell\), perform a targeted intervention:

- zero out one self-attention head write;
- or zero out one cross-attention head write.

The intervention must preserve the rest of the block computation.

This zero-ablation intervention is the primary causal intervention in the experiment.

If the experiment also uses the first-order Taylor approximation from Section 4.4, that approximation should be treated as a **screening method** for large-scale head ranking, not as a replacement for the exact zero-ablation definition.

### 8.2 Target readouts after intervention

For a downstream location \((s', \ell')\), let \(k^\star\) be the winner candidate in the clean run. Define:

1. winner-support drop

\[ \Delta W_{h \to s',\ell',f} = \bar a^{\mathrm{clean}}_{s',\ell',f,k^\star} - \bar a^{\mathrm{ablate}(h)}_{s',\ell',f,k^\star}. \]

2. winner-gap drop

\[ \Delta G_{h \to s',\ell',f} = G^{\mathrm{clean}}_{s',\ell',f} - G^{\mathrm{ablate}(h)}_{s',\ell',f}. \]

3. winner-flip indicator

\[ \mathbf{1}\!\left[\arg\max_k \bar a^{\mathrm{clean}}_{s',\ell',f,k} \ne \arg\max_k \bar a^{\mathrm{ablate}(h)}_{s',\ell',f,k}\right]. \]

These are better aligned with the scientific question than generic head-to-head distances, because they ask whether a source head helps one candidate become the shared winner.

### 8.3 Comparing cross-attention and self-attention

The same contribution and intervention protocol should be applied to:

- selected cross-attention heads;
- selected self-attention heads.

This supports two comparisons:

1. which module contributes more strongly to the final `v_pred`;
2. which module contributes more strongly to winner-gap amplification.

The experiment should not assume that cross-attention dominates self-attention or vice versa. It should measure both.

### 8.4 Transmission visualizations

The final intervention results should be shown by:

- step-wise heatmaps of \(\Delta G\);
- layer-wise heatmaps of \(\Delta G\);
- frame-wise winner-gap curves for selected cases;
- winner-flip summary tables;
- module-level comparison plots:
  - cross-attention vs self-attention.

If Taylor screening is also enabled, its plots and CSV files should be stored separately from the exact zero-ablation outputs so that approximation results are not confused with exact causal interventions.

## 9. Self-Attention Candidate Coupling

This is one of the central analyses in the experiment.

### 9.1 Region-to-region coupling

For one self-attention head at step \(s\), layer \(\ell\), query-frame candidate \(R_{f,a}\), and key-frame candidate \(R_{g,b}\), define

\[ C_{s,\ell,h}(f,a \to g,b) = \frac{1}{|R_{f,a}|}\sum_{i \in R_{f,a}}\sum_{j \in R_{g,b}} \alpha_{s,\ell,h}(i,j). \]

where:

- \(i\) is a query token index inside candidate region \(R_{f,a}\);
- \(j\) is a key token index inside candidate region \(R_{g,b}\);
- \(\alpha_{s,\ell,h}(i,j)\) is the self-attention probability from query token \(i\) to key token \(j\).

Normalize over all candidate regions in the target frame:

\[ \widetilde C_{s,\ell,h}(f,a \to g,b) = \frac{C_{s,\ell,h}(f,a \to g,b)}{\sum_{b'} C_{s,\ell,h}(f,a \to g,b') + \varepsilon}. \]

### 9.2 Candidate-coupling entropy

For one query candidate \(R_{f,a}\), define

\[ H^{\mathrm{sa}}_{s,\ell,h}(f,a \to g) = -\sum_b \widetilde C_{s,\ell,h}(f,a \to g,b)\log\!\big(\widetilde C_{s,\ell,h}(f,a \to g,b) + \varepsilon\big). \]

If one query candidate spreads attention across many future candidates, this entropy is high. If it becomes concentrated on one future candidate, this entropy is low.

### 9.3 Dominant-link ratio

\[ D^{\mathrm{sa}}_{s,\ell,h}(f,a \to g) = \max_b \widetilde C_{s,\ell,h}(f,a \to g,b). \]

This directly measures whether the candidate-to-candidate interaction becomes close to one-to-one.

### 9.4 Candidate compatibility score

For candidate \(R_{f,k}\), define a simple compatibility score:

\[ \Lambda_{s,\ell}(f,k) = \sum_{k'} \widetilde C_{s,\ell}(f-1,k' \to f,k) + \sum_{k'} \widetilde C_{s,\ell}(f,k \to f+1,k'). \]

This score asks whether a candidate is well connected to plausible candidates in neighboring frames.

The working hypothesis is:

- candidates with weak compatibility are more likely to be suppressed later;
- candidates that survive into the final trajectory should form a high-compatibility chain across frames.

### 9.5 Self-attention visualizations

The experiment should generate:

- candidate-to-candidate coupling heatmaps;
- coupling-entropy curves over diffusion steps;
- dominant-link curves over diffusion steps;
- compatibility overlays on candidate masks;
- selected qualitative panels showing many-to-many coupling changing into sparse one-to-one coupling.

## 10. Trajectory Graph and Dynamic Programming

Frame-wise candidate extraction is the primary representation. However, a trajectory-level summary is still useful if done without exponential enumeration.

### 10.1 Node score

For candidate \(R_{f,k}\), define the node score

\[ u_{f,k} = \log\!\big(\bar a_{s,\ell,f,k} + \varepsilon\big). \]

### 10.2 Edge score

Let \(c_{f,k}\) be the candidate center. Let \(C_f(k \to k')\) be a self-attention coupling summary between adjacent frames. Define

\[ \Psi_f(k \to k') = \lambda_{\mathrm{sa}} \log\!\big(C_f(k \to k') + \varepsilon\big) - \lambda_{\mathrm{geo}}\|c_{f+1,k'} - c_{f,k}\|_2^2. \]

### 10.3 Trajectory score

For one candidate trajectory

\[ T = (k_1, k_2, \dots, k_F), \]

define

\[ S(T) = \sum_{f=1}^{F} u_{f,k_f} + \sum_{f=1}^{F-1} \Psi_f(k_f \to k_{f+1}). \]

### 10.4 Decoding

The best trajectory and top-\(K\) trajectories should be decoded by dynamic programming or beam search, not by brute-force enumeration.

This makes trajectory-wise analysis feasible even when each frame contains more than one candidate region.

## 11. Seed Sensitivity Interpretation

The intended interpretation is:

- different seeds do not create unrelated physical mechanisms;
- rather, they perturb early candidate node scores \(u_{f,k}\) and candidate-coupling edge scores \(C_f(k \to k')\);
- when several trajectories have close total scores, these small perturbations can change the top-ranked path;
- later layers and later diffusion steps then amplify that early difference into a different final trajectory.

This is the main mechanistic link between seed sensitivity and motion-planning instability.

## 12. Phase Specialization as a Secondary Analysis

`phase_specialization` should not be treated as a separate primary data-collection experiment.

Instead, it should be built on top of the caches produced here:

- head contribution;
- candidate weights;
- winner-gap drops;
- self-attention coupling metrics;
- decoded trajectory-path scores.

Heads can then be grouped into roles such as:

- candidate proposal;
- candidate pruning;
- trajectory commitment;
- later grounding.

This role analysis is secondary and should reuse the outputs of this experiment.

## 13. Core Outputs

Recommended outputs include:

- `trajectory_consensus_head_contribution.csv`
- `trajectory_consensus_head_contribution_plots/`
- `trajectory_consensus_candidate_regions.csv`
- `trajectory_consensus_candidate_regions.pt`
- `trajectory_consensus_candidate_region_viz/`
- `trajectory_consensus_candidate_weights.csv`
- `trajectory_consensus_winner_gap.csv`
- `trajectory_consensus_intervention.csv`
- `trajectory_consensus_intervention_plots/`
- `trajectory_consensus_self_attention_coupling.csv`
- `trajectory_consensus_self_attention_plots/`
- `trajectory_consensus_trajectory_graph.csv`
- `trajectory_consensus_topk_paths.json`
- `trajectory_consensus_phase_scores.csv`
- `trajectory_consensus_summary.json`

## 14. Reading Guide

The most important readout order is:

1. candidate-region visualization:
   - are the extracted regions scientifically plausible;
2. head-contribution heatmaps and scatter plots:
   - which heads align early and which heads actually push `v_pred`;
3. winner-gap curves:
   - when does one candidate begin to dominate;
4. self-attention coupling plots:
   - does many-to-many candidate coupling become sparse and selective;
5. intervention results:
   - which cross-attention or self-attention heads actually increase the winner gap;
6. top-\(K\) trajectory decoding:
   - are multiple near-tied paths present early, and does one path later separate.

## 15. Practical Notes

- The candidate extractor must be validated visually before any quantitative claim is trusted.
- Frame-wise candidate regions are the primary representation; trajectory-wise decoding is a derived summary.
- Object-mask logic should reuse the existing reference-support pipeline already implemented in the repository.
- Cross-attention and self-attention should be analyzed under the same metric family whenever possible.
- The experiment should prefer clear winner-gap and winner-flip metrics over abstract similarity curves when the scientific question is consensus formation.
