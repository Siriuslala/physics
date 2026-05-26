# `trajectory_consensus_dynamics` Technical Note

## 1. Motivation

`trajectory_consensus_dynamics` studies how Wan2.1-T2V moves from an early multi-candidate planning state to a later single-trajectory consensus state.

The empirical phenomenon is already clear from existing cross-attention visualization:

- early cross-attention maps often contain several candidate object locations;
- different heads may emphasize different candidates;
- later the maps become sharper and more consistent;
- different random seeds can still produce very different final trajectories.

The scientific question is therefore not only whether heads become similar, but **how the system eliminates competing candidates and commits to one trajectory**.

This experiment treats early motion planning as a candidate-competition process:

1. cross-attention proposes candidate object locations;
2. self-attention coordinates candidate compatibility across frames;
3. the residual stream propagates and amplifies candidate preferences across layers and diffusion steps;
4. small early score differences can be magnified into different final winner trajectories.

## 2. Experiment Map and Stage Dependencies

This project discussion originally contained five scientific directions.
After revision, this module keeps four primary directions and one optional appendix.
The previous `cross-layer-transmission` idea is no longer treated as an independent experiment.
Its useful intervention logic is absorbed into the broader `shared-latent coordination` program.

### 2.1 Primary scientific experiments

1. `early-reference-alignment`
   - Question: which heads align with the final reference trajectory earlier than others?
   - Main readout: head contribution to final `v_pred` together with reference-distance scatter plots.

2. `shared-latent coordination`
   - Question: which internal factors actually drive trajectory disambiguation, that is, the suppression of competing candidates and the amplification of one winner?
   - Main readout:
     - clean candidate regions and candidate weights;
     - winner gap and candidate entropy;
     - exact zero-ablation of selected cross-attention and self-attention heads;
     - downstream winner-gap drop, entropy increase, clean-winner support drop, and winner-flip statistics.

3. `self-attention candidate coupling`
   - Question: does self-attention drive candidate convergence, and can its coupling pattern predict which candidate will later win or be eliminated?
   - Main readout:
     - local and global region-to-region coupling matrices;
     - coupling entropy, dominant-link ratio, link margin, mutual consistency, and head agreement;
     - candidate-level compatibility and chainability scores;
     - winner-versus-loser early-feature gaps and temporal-precedence analysis against cross-attention proposal strength.

4. `phase-specialization`
   - Question: do different heads specialize into proposal, pruning, commitment, or later grounding?
   - Main readout: secondary role analysis built on caches produced by the earlier stages.

### 2.2 Optional appendix

1. `trajectory-graph summary`
   - Status: de-prioritized.
   - Role: optional derived visualization only.
   - Reason: by itself it mostly re-encodes an already known convergence phenomenon and does not directly identify the causal factors that drive disambiguation.

### 2.3 Engineering stages in this module

The code module is organized by engineering stages rather than by hypothesis names.
This is the key reason why the document is structured differently from the scientific questions.
One scientific question may rely on several engineering stages, and one engineering stage may support more than one scientific question.

1. `candidate_consensus`
   - Status: implemented.
   - Covers the clean baseline part of `shared-latent coordination`.
   - Produces candidate regions, candidate weights, winner gap, candidate entropy, and candidate visualizations.

2. `head_contribution`
   - Status: implemented.
   - Covers the early-reference-alignment direction.
   - Produces exact zero-ablation contribution metrics for separately selected cross-attention and self-attention heads, plus optional direct readout proxy metrics.
   - Head-list semantics:
     - empty string means "analyze all heads of that module in the selected layers";
     - `None` means "analyze no head of that module".

3. `candidate_intervention`
   - Status: planned, not yet implemented in the current code.
   - Covers the causal attribution part of `shared-latent coordination`.
   - Will ablate one selected cross-attention or self-attention head, rerun the downstream denoising trajectory, and measure downstream candidate-competition changes on the clean candidate partition.

4. `self_attention_coupling`
   - Status: implemented.
   - Covers candidate-to-candidate coupling collapse, winner-versus-loser early predictors, and temporal-precedence analysis.

5. `phase_specialization`
   - Status: planned as a secondary analysis.
   - Will reuse caches from the stages above rather than recollecting model activations.

6. `trajectory_graph`
   - Status: optional appendix only.
   - Should not be prioritized before `candidate_intervention` and `self_attention_coupling` are mature.

### 2.4 Dependency structure

The stages are not intended to be run in one expensive monolithic pass.

The recommended engineering practice is to reuse one common `output_dir` for all stages of the same prompt and seed.
This allows later stages to read the cached CSV or tensor files produced by earlier runs, while still keeping each stage invocation short and controllable.

- `candidate_consensus`
  - Requires only `reuse_cross_attention_dir`.
  - Can be run independently.
  - Candidate-region visualization uses the selected cross-attention head list only.

- `head_contribution`
  - Requires `reuse_cross_attention_dir`.
  - Optionally uses `reuse_head_trajectory_dynamics_dir` for early-alignment scatter plots.
  - Uses one head list for cross-attention and another head list for self-attention.
  - Does not require `candidate_consensus` outputs as a hard runtime dependency, although the two are usually studied together.

- `candidate_intervention`
  - Planned dependency: should reuse candidate-region caches from `candidate_consensus`.
  - Planned dependency: should reuse clean cross-attention maps from `reuse_cross_attention_dir`.
  - For self-attention source heads it additionally requires collected self-attention writes or a runtime patch that can zero one selected self-attention head.
  - This is the main causal stage for the `shared-latent coordination` question.

- `self_attention_coupling`
  - Planned dependency: should reuse candidate-region caches from `candidate_consensus`.
  - Planned dependency: should collect self-attention probabilities for selected heads and selected frame pairs.
  - Planned dependency: should reuse `trajectory_consensus_candidate_weights.csv` from `candidate_consensus` as the cross-attention proposal baseline, so cross-attention early bias and self-attention early bias can be compared on the same candidate partition.

- `trajectory_graph`
  - Optional appendix dependency: should reuse candidate weights from `candidate_consensus` and coupling scores from `self_attention_coupling`.

- `phase_specialization`
  - Planned dependency: should reuse outputs from `head_contribution`, `candidate_intervention`, and `self_attention_coupling`.

### 2.5 Plot-only mode

`trajectory_consensus_plot_only_from_csv=True` is an engineering mode, not a scientific stage.

- It should be used after one expensive stage has already saved its CSV and tensor caches.
- It redraws plots without rerunning expensive extraction or runtime ablation.
- For `head_contribution`, the current code can redraw plots from the saved CSV alone.
- For `candidate_consensus`, step-layer heatmaps are redrawn from the saved CSV files; per-head candidate-region visualizations additionally need `reuse_cross_attention_dir`, because the first row of that visualization uses the original per-head attention map.
- For the planned `candidate_intervention` stage, plot-only regeneration should redraw source-head and downstream-target plots from saved intervention CSV files without rerunning the ablations.

## 3. Wan2.1 Segment Relevant to `v_pred`

This section maps the analysis notation to the Wan2.1-T2V code path in `projects/Wan2_1/wan/modules/model.py`.

### 3.1 One transformer block

For one diffusion step and one transformer block \(\ell\), let the input residual state be \(X^{(\ell)} \in \mathbb{R}^{L \times C}\), where \(L\) is the patchified token length and \(C\) is the hidden dimension.

The Wan block computes

\[ Y_{\mathrm{sa}}^{(\ell)} = \operatorname{SelfAttn}^{(\ell)}\!\left(\operatorname{Norm}_1^{(\ell)}(X^{(\ell)}) \odot (1 + e_1^{(\ell)}) + e_0^{(\ell)}\right). \]

\[ \widetilde X^{(\ell)} = X^{(\ell)} + Y_{\mathrm{sa}}^{(\ell)} \odot e_2^{(\ell)}. \]

\[ Y_{\mathrm{ca}}^{(\ell)} = \operatorname{CrossAttn}^{(\ell)}\!\left(\operatorname{Norm}_3^{(\ell)}(\widetilde X^{(\ell)}),\ \mathrm{context}\right). \]

\[ \widehat X^{(\ell)} = \widetilde X^{(\ell)} + Y_{\mathrm{ca}}^{(\ell)}. \]

\[ Y_{\mathrm{ffn}}^{(\ell)} = \operatorname{FFN}^{(\ell)}\!\left(\operatorname{Norm}_2^{(\ell)}(\widehat X^{(\ell)}) \odot (1 + e_4^{(\ell)}) + e_3^{(\ell)}\right). \]

\[ X^{(\ell+1)} = \widehat X^{(\ell)} + Y_{\mathrm{ffn}}^{(\ell)} \odot e_5^{(\ell)}. \]

These equations match the order in `WanAttentionBlock.forward`.

### 3.2 Final readout path

After the final transformer block, Wan2.1 applies the `Head` module and then `unpatchify`.

Let \(X^{(\mathrm{final})} \in \mathbb{R}^{L \times C}\) be the final hidden sequence. Let the timestep modulation produce two vectors

\[ m_0(e), m_1(e) \in \mathbb{R}^{C}. \]

In code, these are produced by `self.head.modulation + e.unsqueeze(1)` and then split into two chunks.

Let \(W_{\mathrm{head}} \in \mathbb{R}^{C \times P C_{\mathrm{out}}}\) be the weight of `self.head.head`, where \(P = p_t p_h p_w\) is the patch volume. The final readout before `unpatchify` is

\[ O = W_{\mathrm{head}}\!\left(\operatorname{LN}(X^{(\mathrm{final})}) \odot (1 + m_1(e)) + m_0(e)\right) \in \mathbb{R}^{L \times P C_{\mathrm{out}}}. \]

After `unpatchify`, the model returns

\[ v_{\mathrm{pred}} \in \mathbb{R}^{C_{\mathrm{out}} \times F \times H \times W}. \]

## 4. Per-Head Write and Contribution Definitions

### 4.1 Per-head write

For one attention module with \(H\) heads, let the post-attention, pre-output-projection tensor of head \(h\) be \(Z_h \in \mathbb{R}^{L \times d_h}\), where \(d_h = C / H\).

Partition the output-projection weight as

\[ W_O = [W_O^{(1)}, W_O^{(2)}, \dots, W_O^{(H)}], \qquad W_O^{(h)} \in \mathbb{R}^{d_h \times C}. \]

The hidden-state write of head \(h\) is

\[ U_h = Z_h W_O^{(h)} \in \mathbb{R}^{L \times C}. \]

For self-attention, the actual residual write is

\[ U_h^{\mathrm{sa}} = U_h \odot e_2^{(\ell)}. \]

For cross-attention, the actual residual write is

\[ U_h^{\mathrm{ca}} = U_h. \]

### 4.2 Exact zero-ablation effect

For one analyzed head at diffusion step \(s\), layer \(\ell\), and head index \(h\), define the downstream final prediction as a function of the inserted head write \(U\):

\[ v_{s,\ell,h}(U). \]

Then the clean and ablated outputs are

\[ v_s^{\mathrm{clean}} = v_{s,\ell,h}(U_{s,\ell,h}), \qquad v_{s,\ell,h}^{\mathrm{ablate}} = v_{s,\ell,h}(0). \]

The exact zero-ablation effect is

\[ \Delta v_{s,\ell,h} = v_s^{\mathrm{clean}} - v_{s,\ell,h}^{\mathrm{ablate}}. \]

This is the primary causal contribution definition in the current implementation.

### 4.3 First-order Taylor approximation of head ablation effect

Exact ablation requires one downstream rerun per head.
To reduce cost, the experiment may also compute a first-order Taylor approximation.

Starting from

\[ \Delta v_{s,\ell,h} = v_{s,\ell,h}(U_{s,\ell,h}) - v_{s,\ell,h}(0), \]

take the first-order Taylor expansion of \(v_{s,\ell,h}(U)\) around the clean head write \(U_{s,\ell,h}\).
Then

\[ v_{s,\ell,h}(0) \approx v_{s,\ell,h}(U_{s,\ell,h}) + \left. \frac{\partial v_{s,\ell,h}(U)}{\partial U} \right|_{U = U_{s,\ell,h}} (0 - U_{s,\ell,h}). \]

Rearranging gives

\[ \Delta v_{s,\ell,h} \approx \left. \frac{\partial v_{s,\ell,h}(U)}{\partial U} \right|_{U = U_{s,\ell,h}} U_{s,\ell,h}. \]

This approximation means:

- use the clean run as the expansion point;
- use the local Jacobian of the final `v_pred` with respect to the injected head write;
- multiply that Jacobian by the clean head write itself.

This is a fast approximation to the exact ablation effect and is suitable for large-scale head screening.

### 4.4 Relation to JVP and attribution patching

The Taylor approximation above can be written in two computationally useful forms.

First, define the downstream map

\[ F_{s,\ell,h}(U) = v_{s,\ell,h}(U). \]

Then the first-order Taylor approximation is the Jacobian-vector product

\[ \Delta v_{s,\ell,h}^{\mathrm{Taylor}} = JF_{s,\ell,h}(U_{s,\ell,h})[U_{s,\ell,h}], \]

where \(JF(U)[dU]\) denotes the directional derivative of \(F\) at \(U\) along direction \(dU\).

This is exactly what `jvp` computes:

- input point: \(U_{s,\ell,h}\);
- direction vector: \(U_{s,\ell,h}\);
- output: the first-order approximation to the clean-to-ablate change in `v_pred`.

Second, if the quantity of interest is not the whole vector \(\Delta v\) itself, but a scalar patch metric

\[ m(v), \]

then the first-order change of that scalar metric is

\[ \Delta m_{s,\ell,h}^{\mathrm{Taylor}} \approx \left\langle \nabla_U m\!\big(F_{s,\ell,h}(U_{s,\ell,h})\big),\ U_{s,\ell,h} \right\rangle. \]

This is the same local-linear idea used in attribution patching.

More generally, if one compares a clean activation \(U^{\mathrm{clean}}\) and a corrupted activation \(U^{\mathrm{corr}}\), attribution patching approximates the scalar patch effect by

\[ \Delta m^{\mathrm{attr}} \approx \left\langle \nabla_U m\!\big(F(U^{\mathrm{corr}})\big),\ U^{\mathrm{clean}} - U^{\mathrm{corr}} \right\rangle. \]

Our ablation setting is a special case with

\[ U^{\mathrm{clean}} = U_{s,\ell,h}, \qquad U^{\mathrm{corr}} = 0, \qquad U^{\mathrm{clean}} - U^{\mathrm{corr}} = U_{s,\ell,h}. \]

So the attribution-patching style approximation becomes

\[ \Delta m_{s,\ell,h}^{\mathrm{ablate}} \approx \left\langle \nabla_U m\!\big(F_{s,\ell,h}(U_{s,\ell,h})\big),\ U_{s,\ell,h} \right\rangle. \]

This is a scalarized version of the vector Taylor approximation above.

The distinction is important:

- `jvp` naturally approximates the **vector-valued** output change \(\Delta v\);
- attribution patching naturally approximates the change of a **scalar metric** defined on the output.

The current code now implements the vector-valued Taylor path through a dedicated
`trajectory_consensus_contribution_method=taylor_approx` option.
The implementation:

- runs one clean forward for each selected `(step, layer, module, branch)`;
- captures the clean head writes and the clean downstream suffix payload;
- replays the clean downstream suffix from the targeted head write;
- uses `jvp` to compute the first-order approximation
  \(JF_{s,\ell,h}(U_{s,\ell,h})[U_{s,\ell,h}]\).

The scalar attribution-patching style approximation from the equations above is
still documented for interpretation, but it is not yet emitted as a separate
metric family in the current CSV outputs.

### 4.5 Direct final-head projection proxy

The direct projection proxy is intended to answer a simpler question:

- if one takes the isolated head write \(U_{s,\ell,h}\),
- and reads it out immediately by the model's final readout head,
- how aligned is that readout with the clean final prediction?

Let the clean run provide the final-head timestep modulation input \(e_s^{\mathrm{clean}}\) and the clean `grid_sizes`.
Define the two clean modulation vectors at the final readout head by

\[ m_0^{\mathrm{clean}} = m_0(e_s^{\mathrm{clean}}), \qquad m_1^{\mathrm{clean}} = m_1(e_s^{\mathrm{clean}}). \]

Then the proxy first applies the same final-head readout transform to the isolated head write:

\[ O_{s,\ell,h}^{\prime} = W_{\mathrm{head}}\!\left(\operatorname{LN}(U_{s,\ell,h}) \odot (1 + m_1^{\mathrm{clean}}) + m_0^{\mathrm{clean}}\right). \]

The proxy video-space output is then

\[ v_{s,\ell,h}^{\prime,\mathrm{proj}} = \operatorname{Unpatchify}(O_{s,\ell,h}^{\prime}, \mathrm{grid\_sizes}^{\mathrm{clean}}). \]

This formula matches the current code implementation.

Two points are important.

1. Wan `LayerNorm` has no running mean or running variance.
   - Therefore the phrase "reuse the clean run's normalization statistics" is not correct for Wan.
   - The current code does **not** reuse any running statistics.
   - The normalization inside the proxy is computed directly from \(U_{s,\ell,h}\) itself, because `Head.forward` calls `self.norm(x)` on its own input.

2. The current implementation reuses only:
   - the trained final `Head` module parameters,
   - the clean run's timestep modulation input \(e_s^{\mathrm{clean}}\),
   - the clean run's `grid_sizes`.

This is why the quantity is called a **direct readout proxy** rather than an exact causal decomposition of an early head.
It asks whether the isolated head write is already aligned with the final readout direction under the clean final readout head, not whether that head alone causally reconstructs the full downstream computation.

### 4.6 Proxy similarity and share

\[ \mathrm{ProjCos}(s,\ell,h) = \frac{\langle v_{s,\ell,h}^{\prime,\mathrm{proj}}, v_s^{\mathrm{clean}} \rangle}{\|v_{s,\ell,h}^{\prime,\mathrm{proj}}\|_2 \, \|v_s^{\mathrm{clean}}\|_2}. \]

\[ \mathrm{ProjDot}(s,\ell,h) = \langle v_{s,\ell,h}^{\prime,\mathrm{proj}}, v_s^{\mathrm{clean}} \rangle. \]

\[ \mathrm{ProjShare}(s,\ell,h) = \frac{\max(0, \mathrm{ProjDot}(s,\ell,h))}{\sum_{j \in \mathcal{H}_{\mathrm{ana}}}\max(0, \mathrm{ProjDot}(s,\ell,j)) + \varepsilon}. \]

## 5. Head-Contribution Metrics

The current code reports two complementary metric families.

### 5.1 Effect-direction metrics based on \(\Delta v\)

\[ \mathrm{CosFull}(s,\ell,h) = \frac{\langle \Delta v_{s,\ell,h}, v_s^{\mathrm{clean}} \rangle}{\|\Delta v_{s,\ell,h}\|_2 \, \|v_s^{\mathrm{clean}}\|_2}. \]

\[ \mathrm{DotFull}(s,\ell,h) = \langle \Delta v_{s,\ell,h}, v_s^{\mathrm{clean}} \rangle. \]

Let the object mask be \(M^{\mathrm{obj}} \in \{0,1\}^{F \times H \times W}\). Then

\[ \mathrm{CosObj}(s,\ell,h) = \frac{\langle M^{\mathrm{obj}} \odot \Delta v_{s,\ell,h}, M^{\mathrm{obj}} \odot v_s^{\mathrm{clean}} \rangle}{\|M^{\mathrm{obj}} \odot \Delta v_{s,\ell,h}\|_2 \, \|M^{\mathrm{obj}} \odot v_s^{\mathrm{clean}}\|_2}. \]

\[ \mathrm{DotObj}(s,\ell,h) = \langle M^{\mathrm{obj}} \odot \Delta v_{s,\ell,h}, M^{\mathrm{obj}} \odot v_s^{\mathrm{clean}} \rangle. \]

Interpretation:

- these metrics ask whether the removed head contributed in a direction similar to the clean final prediction;
- they are causal effect metrics because they are based on the difference between clean and ablated outputs.

### 5.2 Output-preservation metrics based on \(v^{\mathrm{ablate}}\)

The user-raised alternative is also meaningful and is now part of the current implementation.

\[ \mathrm{CosAblateFull}(s,\ell,h) = \frac{\langle v_{s,\ell,h}^{\mathrm{ablate}}, v_s^{\mathrm{clean}} \rangle}{\|v_{s,\ell,h}^{\mathrm{ablate}}\|_2 \, \|v_s^{\mathrm{clean}}\|_2}. \]

\[ \mathrm{DotAblateFull}(s,\ell,h) = \langle v_{s,\ell,h}^{\mathrm{ablate}}, v_s^{\mathrm{clean}} \rangle. \]

\[ \mathrm{CosAblateObj}(s,\ell,h) = \frac{\langle M^{\mathrm{obj}} \odot v_{s,\ell,h}^{\mathrm{ablate}}, M^{\mathrm{obj}} \odot v_s^{\mathrm{clean}} \rangle}{\|M^{\mathrm{obj}} \odot v_{s,\ell,h}^{\mathrm{ablate}}\|_2 \, \|M^{\mathrm{obj}} \odot v_s^{\mathrm{clean}}\|_2}. \]

\[ \mathrm{DotAblateObj}(s,\ell,h) = \langle M^{\mathrm{obj}} \odot v_{s,\ell,h}^{\mathrm{ablate}}, M^{\mathrm{obj}} \odot v_s^{\mathrm{clean}} \rangle. \]

Interpretation:

- these metrics ask how much the final output is preserved after the head is removed;
- they are not direct effect vectors, but they are useful for judging whether the head is functionally important for keeping the clean prediction stable.

The current code therefore reports both:

- effect-direction metrics from \(\Delta v\);
- output-preservation metrics from \(v^{\mathrm{ablate}}\).

### 5.3 Why both metric families are useful

The two metric families answer different questions.

- \(\Delta v = v^{\mathrm{clean}} - v^{\mathrm{ablate}}\) measures the direction of the part removed by ablation.
- \(v^{\mathrm{ablate}}\) measures how much of the clean output remains after the head is removed.

Therefore:

- if one wants a closer analogue of an indirect causal effect, the \(\Delta v\)-based metrics are the primary choice;
- if one wants a direct stability metric of the ablated output itself, the \(v^{\mathrm{ablate}}\)-based metrics are also informative.

### 5.4 Early-alignment correlation

Let \(D_{s,\ell,h}^{\mathrm{ref}}\) be the reference distance from `head_trajectory_dynamics`. Define

\[ E_{\ell,h}^{\mathrm{auc}} = -\frac{1}{|S|}\sum_{s \in S} D_{s,\ell,h}^{\mathrm{ref}}, \qquad E_{\ell,h}^{\mathrm{raw}} = \frac{1}{|S|}\sum_{s \in S} D_{s,\ell,h}^{\mathrm{ref}}. \]

The current code draws scatter plots of early-alignment scores against selected cosine-based contribution metrics from both families:

- `cos_obj`, `cos_full`;
- `ablate_cos_obj`, `ablate_cos_full`.


## 6. Candidate Region Extraction

Candidate regions are the primary intermediate representation in the current implementation.

### 6.1 Input

For step \(s\), layer \(\ell\), head \(h\), and latent frame \(f\), let the normalized object-token cross-attention map be \(P_f^{(s,\ell,h)}(y,x)\), where \(\sum_{y,x} P_f^{(s,\ell,h)}(y,x) = 1\).

The current code first averages over heads to obtain the shared frame map
\[ P_f^{(s,\ell,\mathrm{mean})}(y,x) = \frac{1}{H_\ell}\sum_{h=1}^{H_\ell} P_f^{(s,\ell,h)}(y,x). \]

Candidate regions are extracted from this shared map and then reused for all heads when computing per-head candidate weights.

### 6.2 Detailed extraction algorithm

The goal is to recover compact candidate regions from a frame map that is often shaped like a bright motion stripe rather than a set of already-separated blobs.

For one frame \(f\), the current algorithm is:

1. `Coarse denoising and smoothing`
   - The extractor first reuses the shared winsorization-and-despike preprocessing from the helper in `head_evolution.py`.
   - This removes isolated spikes with a minimum connected-component area threshold and clamps extreme upper tails.
   - Let the resulting frame map be \(A_f(y,x)\).
   - A small uniform smoothing kernel is then applied:
     \[ \bar A_f(y,x) = \frac{1}{(2r+1)^2}\sum_{|u|\le r}\sum_{|v|\le r} A_f(y+u,x+v), \]
     where \(r\) is the smoothing radius.
   - In code, this is a standard box filter, so it preserves local structure while suppressing single-patch jaggedness.

2. `Background suppression and contrast boosting`
   - Let \(q_{\mathrm{base}} \in (0,1)\) be the shared support quantile.
   - Define a background quantile \(q_{\mathrm{bg}} = \max(q_{\mathrm{base}} - 0.10, 0.50)\).
   - Compute the background threshold \(t_{\mathrm{bg}} = Q_{q_{\mathrm{bg}}}(\bar A_f)\).
   - Form the positive contrast map
     \[ B_f(y,x) = \left[\max(\bar A_f(y,x) - t_{\mathrm{bg}}, 0)\right]^\gamma, \]
     with fixed exponent \(\gamma = 2\).
   - This transformation is designed to suppress weak floor-like background bands while making compact bright nuclei much sharper.

3. `Support set construction`
   - Let \(B_f^+\) be the set of positive entries of \(B_f\).
   - Define the candidate support threshold \(t_{\mathrm{sup}} = Q_{q_{\mathrm{base}}}(B_f^+)\).
   - The high-confidence support is
     \[ S_f = \{(y,x): B_f(y,x) \ge t_{\mathrm{sup}}\}. \]
   - Every later step only operates on \(S_f\), not on the entire frame.

4. `Local-maximum proposal at multiple seed levels`
   - Let \(q_{\mathrm{seed},1}, \dots, q_{\mathrm{seed},R}\) be the seed-support levels.
   - In the current implementation these are taken from `trajectory_consensus_candidate_split_quantiles`, clamped into \([q_{\mathrm{base}}, 0.995]\), with a fallback level \(q_{\mathrm{base}}+0.08\) when the list is empty.
   - For each level \(q_{\mathrm{seed},r}\), compute the score threshold \(t_r = Q_{q_{\mathrm{seed},r}}(B_f^+)\).
   - A point \(p\in S_f\) is a proposal if it is an 8-neighborhood local maximum above that threshold:
     \[ p \in \mathcal{P}_r \iff B_f(p)\ge t_r \ \wedge\ B_f(p)=\max_{q\in \mathcal{N}_8(p)\cup\{p\}} B_f(q). \]
   - In code, the local-max test is implemented by a \(3\times3\) max-pooling operator followed by equality testing.

5. `Greedy seed consolidation`
   - The proposal set \(\cup_r \mathcal{P}_r\) is sorted by score and merged greedily with a spatial non-maximum suppression radius \(d_{\mathrm{merge}}=2\).
   - A proposal is kept as a stable seed only if it is sufficiently far from all previously kept seeds and it appears on at least \(n_{\mathrm{stable}}\) seed levels.
   - The kept seeds are capped at \(K_{\max}=5\) per frame.
   - This step separates the roles of proposal generation and final seed selection: step 4 proposes local maxima, while step 5 decides which maxima are stable enough to seed clustering.

6. `Seeded weighted clustering and core trimming`
   - Let the support points be \(x_i \in \mathbb{R}^2\) with weights \(w_i = B_f(x_i)\).
   - Let \(c_1^{(0)}, \dots, c_K^{(0)}\) be the stable seed centers.
   - The clustering objective is
     \[ \min_{z_1,\dots,z_n,\ c_1,\dots,c_K} \sum_{i=1}^n w_i \|x_i - c_{z_i}\|_2^2. \]
   - The assignment and update steps are the usual weighted k-means iterations:
     \[ z_i^{(t)} = \arg\min_j \|x_i - c_j^{(t)}\|_2, \qquad c_j^{(t+1)} = \frac{\sum_{i:z_i^{(t)}=j} w_i x_i}{\sum_{i:z_i^{(t)}=j} w_i}. \]
   - The iteration stops when either the maximum center displacement is at most \(0.25\) token units or \(10\) iterations have been executed.
   - For each cluster \(C_j\), the final region is not the full Voronoi cell. Instead, the cluster is trimmed to a compact core: sort the assigned points by distance to \(c_j\), keep the smallest radius that retains at least \(80\%\) of the cluster mass, and then keep only the largest connected component inside that core.

7. `Strong pruning`
   - A trimmed cluster is kept only if it satisfies all of the following:
     \[ |R_{f,j}| \ge 4,\qquad \frac{\mathrm{bbox\_width}(R_{f,j})}{W} \le 0.40,\qquad \frac{\mathrm{bbox\_height}(R_{f,j})}{H} \le 0.95. \]
   - The region must also be at least as dense as its support cluster in average intensity:
     \[ \frac{1}{|R_{f,j}|}\sum_{p\in R_{f,j}} B_f(p) \ge \frac{1}{|C_j|}\sum_{p\in C_j} B_f(p). \]
   - The minimum-area threshold is intentionally raised from the old value of \(2\) to \(4\), because two- or three-patch speckles are not treated as meaningful candidates.
   - The asymmetric width bound is the most important geometry prior here, because floor-like false positives are usually too wide horizontally even when their height is modest.

If no seed survives the full process, the extractor falls back to the global argmax as a one-pixel candidate.
The current code therefore implements a peak-seeded weighted clustering algorithm rather than the previous threshold-ladder split heuristic.

The default values are intentionally conservative:
\[ q_{\mathrm{base}} = 0.85,\quad q_{\mathrm{bg}} = 0.75,\quad (q_{\mathrm{seed},r}) = (0.92, 0.95, 0.97),\quad n_{\mathrm{stable}} = 2,\quad d_{\mathrm{merge}} = 2,\quad K_{\max} = 5,\quad \gamma = 2,\quad \eta = 0.8. \]
The final geometry filter uses \(\min\)-area \(=4\), maximum width ratio \(=0.40\), and maximum height ratio \(=0.95\).
These defaults are chosen to keep the compact object-centered bright nuclei while suppressing two failure modes seen in the qualitative examples: small multi-patch speckles and wide floor-like background bands.

### 6.3 Candidate weights

Let the extracted regions on frame \(f\) be \(R_{f,1}, \dots, R_{f,K_f}\). The per-head candidate weight is

\[ a_{s,\ell,h,f,k} = \sum_{(y,x) \in R_{f,k}} P_f^{(s,\ell,h)}(y,x). \]

These weights are computed after candidate extraction, using the per-head normalized map and the shared candidate regions.

### 6.4 Candidate visualization

For each selected step \(s\) and selected layer \(\ell\), the current code renders one visualization for the shared head-mean map and optionally additional visualizations for selected individual cross-attention heads.
Every visualization has two rows:

- row 1: raw attention maps, with one color scale per frame and no per-frame color bar;
- row 2: binary candidate support masks.

This visualization is mandatory for validating whether candidate extraction is scientifically reasonable.

## 7. Candidate Consensus and Winner Gap

For step \(s\), layer \(\ell\), frame \(f\), and candidate \(k\), define the layer-mean candidate weight

\[ \bar a_{s,\ell,f,k} = \frac{1}{H_\ell}\sum_{h=1}^{H_\ell} a_{s,\ell,h,f,k}. \]

Normalize these weights over the extracted candidates:

\[ \pi_{s,\ell,f,k} = \frac{\bar a_{s,\ell,f,k}}{\sum_{j=1}^{K_f}\bar a_{s,\ell,f,j} + \varepsilon}. \]

The winner gap is

\[ G_{s,\ell,f} = \pi_{s,\ell,f,k_1} - \pi_{s,\ell,f,k_2}, \]

where \(k_1\) and \(k_2\) are the top-1 and top-2 candidates under \(\pi_{s,\ell,f,k}\).

The candidate entropy is

\[ H_{s,\ell,f}^{\mathrm{cand}} = -\sum_{k=1}^{K_f} \pi_{s,\ell,f,k}\log(\pi_{s,\ell,f,k} + \varepsilon). \]

The current `candidate_consensus` stage saves both quantities, but the intended primary interpretation is the winner gap.

## 8. Shared-Latent Coordination via Head Intervention

This section describes the planned `candidate_intervention` stage.
It is the main causal experiment for the question:

- which internal heads actually help one candidate trajectory defeat its competitors;
- whether cross-attention and self-attention contribute differently to this disambiguation process.

The stage should analyze only:

- cross-attention heads;
- self-attention heads.

FFN is not a primary target here.

### 8.1 Clean candidate reference at each downstream observation

Fix one clean run.
For every selected downstream observation point \((s', \ell')\), candidate extraction from Section 6 provides clean frame-wise candidate regions

\[ R_{s',\ell',f,1}^{\mathrm{clean}}, \dots, R_{s',\ell',f,K_{s',\ell',f}}^{\mathrm{clean}}. \]

Using the clean head-mean candidate weights from Section 7, define the clean candidate-support coverage

\[ S_{s',\ell',f}^{\mathrm{clean}} = \sum_{k=1}^{K_{s',\ell',f}} \bar a_{s',\ell',f,k}^{\mathrm{clean}}. \]

Define the clean candidate distribution normalized **within the extracted clean candidates**

\[ \pi_{s',\ell',f,k}^{\mathrm{clean}} = \frac{\bar a_{s',\ell',f,k}^{\mathrm{clean}}}{S_{s',\ell',f}^{\mathrm{clean}} + \varepsilon}. \]

Let the clean winner be

\[ k_{s',\ell',f}^{\star} = \arg\max_k \pi_{s',\ell',f,k}^{\mathrm{clean}}. \]

Let the top-1 and top-2 clean candidate indices under \(\pi^{\mathrm{clean}}\) be \(k_1^{\mathrm{clean}}\) and \(k_2^{\mathrm{clean}}\).
Define the clean winner gap and clean candidate entropy by

\[ G_{s',\ell',f}^{\mathrm{clean}} = \pi_{s',\ell',f,k_1^{\mathrm{clean}}}^{\mathrm{clean}} - \pi_{s',\ell',f,k_2^{\mathrm{clean}}}^{\mathrm{clean}}, \]

\[ H_{s',\ell',f}^{\mathrm{clean}} = -\sum_{k=1}^{K_{s',\ell',f}} \pi_{s',\ell',f,k}^{\mathrm{clean}} \log(\pi_{s',\ell',f,k}^{\mathrm{clean}} + \varepsilon). \]

This normalization is important.
It separates two different effects:

- how the head changes competition **among the extracted candidates**;
- how the head changes the **total amount of mass** that remains inside the clean candidate support.

### 8.2 Source-head ablation protocol

Let one analyzed source head be

\[ q = (m, s_0, \ell_0, h_0), \qquad m \in \{\mathrm{ca}, \mathrm{sa}\}. \]

Here:

- \(m\) identifies whether the source is a cross-attention head or a self-attention head;
- \(s_0\) is the source diffusion step;
- \(\ell_0\) is the source layer;
- \(h_0\) is the source head index.

The intervention is exact zero-ablation of that head write:

\[ U_q \leftarrow 0. \]

All other activations and parameters are kept unchanged.
After that intervention, the model should continue the downstream denoising trajectory and collect downstream cross-attention maps again.

The downstream observation set should depend on the source module:

- if \(m = \mathrm{ca}\), observe
  \[ \mathcal{D}(q) = \{(s', \ell') : s' > s_0 \ \text{or}\ (s' = s_0 \text{ and } \ell' > \ell_0)\}; \]
- if \(m = \mathrm{sa}\), observe
  \[ \mathcal{D}(q) = \{(s', \ell') : s' > s_0 \ \text{or}\ (s' = s_0 \text{ and } \ell' \ge \ell_0)\}. \]

The distinction is necessary because in Wan one self-attention sublayer is followed by cross-attention inside the same block, so self-attention ablation at layer \(\ell_0\) can already affect the same block's cross-attention map.

### 8.3 Anchored downstream readout on the clean candidate partition

For each ablated run and each downstream observation point \((s', \ell') \in \mathcal{D}(q)\), compute the ablated layer-mean candidate weights on the **same clean candidate regions**

\[ \bar a_{s',\ell',f,k}^{\mathrm{ablate}(q)} = \frac{1}{H_{\ell'}} \sum_{h=1}^{H_{\ell'}} a_{s',\ell',h,f,k}^{\mathrm{ablate}(q)}. \]

Define the ablated candidate-support coverage

\[ S_{s',\ell',f}^{\mathrm{ablate}(q)} = \sum_{k=1}^{K_{s',\ell',f}} \bar a_{s',\ell',f,k}^{\mathrm{ablate}(q)}. \]

Define the ablated candidate distribution on the clean partition

\[ \pi_{s',\ell',f,k}^{\mathrm{ablate}(q)} = \frac{\bar a_{s',\ell',f,k}^{\mathrm{ablate}(q)}}{S_{s',\ell',f}^{\mathrm{ablate}(q)} + \varepsilon}. \]

Then define the ablated winner gap and candidate entropy

\[ G_{s',\ell',f}^{\mathrm{ablate}(q)} = \pi_{s',\ell',f,k_1^{\mathrm{ablate}(q)}}^{\mathrm{ablate}(q)} - \pi_{s',\ell',f,k_2^{\mathrm{ablate}(q)}}^{\mathrm{ablate}(q)}, \]

\[ H_{s',\ell',f}^{\mathrm{ablate}(q)} = -\sum_{k=1}^{K_{s',\ell',f}} \pi_{s',\ell',f,k}^{\mathrm{ablate}(q)} \log(\pi_{s',\ell',f,k}^{\mathrm{ablate}(q)} + \varepsilon). \]

This anchored evaluation is the core of the stage.
It does **not** ask whether the ablated run discovers a new segmentation.
It asks whether the source head was helping the clean winner candidate remain dominant on a fixed and interpretable candidate basis.

### 8.4 Primary intervention metrics

For each source head \(q\) and downstream target \((s', \ell', f)\), define the following quantities.

1. clean-winner support drop

\[ \Delta A_{q \to s',\ell',f}^{\star} = \pi_{s',\ell',f,k_{s',\ell',f}^{\star}}^{\mathrm{clean}} - \pi_{s',\ell',f,k_{s',\ell',f}^{\star}}^{\mathrm{ablate}(q)}. \]

2. winner-gap drop

\[ \Delta G_{q \to s',\ell',f} = G_{s',\ell',f}^{\mathrm{clean}} - G_{s',\ell',f}^{\mathrm{ablate}(q)}. \]

3. candidate-entropy increase

\[ \Delta H_{q \to s',\ell',f} = H_{s',\ell',f}^{\mathrm{ablate}(q)} - H_{s',\ell',f}^{\mathrm{clean}}. \]

4. candidate-support coverage drop

\[ \Delta S_{q \to s',\ell',f} = S_{s',\ell',f}^{\mathrm{clean}} - S_{s',\ell',f}^{\mathrm{ablate}(q)}. \]

5. winner-flip indicator on the clean partition

\[ F_{q \to s',\ell',f} = \mathbf{1}\!\left[\arg\max_k \pi_{s',\ell',f,k}^{\mathrm{clean}} \ne \arg\max_k \pi_{s',\ell',f,k}^{\mathrm{ablate}(q)}\right]. \]

These metrics answer different mechanistic questions:

- \(\Delta G\) asks whether the source head helps one candidate separate from the runner-up;
- \(\Delta H\) asks whether the source head helps reduce candidate ambiguity;
- \(\Delta A^{\star}\) asks whether the source head specifically supports the clean winner;
- \(\Delta S\) asks whether the source head helps keep mass inside the clean candidate support at all;
- \(F\) asks whether removing the source head can change the actual winner identity.

### 8.5 Aggregate source-head scores

Let \(\mathcal{T}(q)\) be the set of all valid downstream target triples \((s', \ell', f)\) for source head \(q\).
The stage should report at least the following aggregated source-head summaries:

\[ \mathrm{GapImpact}(q) = \frac{1}{|\mathcal{T}(q)|}\sum_{(s',\ell',f)\in \mathcal{T}(q)} \Delta G_{q \to s',\ell',f}, \]

\[ \mathrm{EntropyImpact}(q) = \frac{1}{|\mathcal{T}(q)|}\sum_{(s',\ell',f)\in \mathcal{T}(q)} \Delta H_{q \to s',\ell',f}, \]

\[ \mathrm{WinnerSupportImpact}(q) = \frac{1}{|\mathcal{T}(q)|}\sum_{(s',\ell',f)\in \mathcal{T}(q)} \Delta A_{q \to s',\ell',f}^{\star}, \]

\[ \mathrm{CoverageImpact}(q) = \frac{1}{|\mathcal{T}(q)|}\sum_{(s',\ell',f)\in \mathcal{T}(q)} \Delta S_{q \to s',\ell',f}, \]

\[ \mathrm{FlipRate}(q) = \frac{1}{|\mathcal{T}(q)|}\sum_{(s',\ell',f)\in \mathcal{T}(q)} F_{q \to s',\ell',f}. \]

In practice, it is also useful to compute the same summaries on an explicitly restricted early planning window, because late diffusion steps may dilute the signal.

### 8.6 Planned visualizations

The planned `candidate_intervention` stage should produce:

- source-head heatmaps of `GapImpact`, `EntropyImpact`, `CoverageImpact`, and `FlipRate`;
- source-step and source-layer averages, separately for cross-attention and self-attention;
- downstream target heatmaps for one selected source head, showing where its effect appears;
- frame-wise curves of \(\Delta G\), \(\Delta H\), and \(\Delta A^{\star}\) for selected source heads;
- clean-vs-ablated candidate-distribution bar plots on selected frames;
- module-level comparison plots:
  - cross-attention source heads versus self-attention source heads.

### 8.7 Interpretation logic

This stage is intended to distinguish several mechanistically different cases.

- Large \(\Delta S\) but small \(\Delta G\) and small \(\Delta H\):
  - the head mostly helps maintain object support or grounding, but is not a primary disambiguation driver.
- Large \(\Delta G\), large \(\Delta H\), and high `FlipRate`:
  - the head is a strong causal driver of candidate disambiguation.
- Small winner-gap effect but high final `v_pred` contribution from Section 5:
  - the head may be more important for later appearance or motion rendering than for early trajectory selection.

This is the central reason to prioritize `candidate_intervention` over a separate `cross-layer-transmission` experiment.
The real question is not merely whether information can propagate forward, but **which heads actually make the winner candidate win**.

## 9. Self-Attention Candidate Coupling

This is one of the central planned analyses in the experiment.
It aims to explain why some candidates survive while others are eliminated, and whether self-attention is the module that actually drives this convergence.

The central working view is:

- cross-attention proposes candidate object locations inside each frame;
- self-attention coordinates those candidate locations across frames;
- winner selection is not read off from one frame alone, but from whether one candidate is more compatible with a coherent cross-frame trajectory than its competitors.

### 9.1 Analysis unit and candidate partition

The basic analysis unit is one observation point \((s, \ell)\), where \(s\) is the diffusion step and \(\ell\) is the transformer layer.

At that observation point, for each latent frame \(f\), let

\[ R_{s,\ell,f,1}, \dots, R_{s,\ell,f,K_{s,\ell,f}} \]

be the candidate regions extracted by `candidate_consensus` from the cross-attention `head-mean` map at the same step and layer.

This means the stage is fundamentally layer-wise:

- for fixed step \(s\), it compares how the metrics change across layers \(\ell\);
- for fixed layer \(\ell\), it compares how the metrics change across diffusion steps \(s\);
- for selected candidates, it can also follow one ordered observation trace through denoising time.

All self-attention heads inside the same layer are evaluated on the same candidate partition.
This is important because the scientific goal is not to let each self-attention head define its own regions, but to ask how different self-attention heads treat the same set of motion-planning candidates.

When a layer-level aggregation is needed, let

\[ \mathcal{H}^{\mathrm{sa}}_{\ell} \]

be the selected self-attention head set for layer \(\ell\).
By default, this set should contain all self-attention heads in that layer.
An explicit subset is optional and should be used only for focused follow-up analysis.

### 9.2 Eventual-winner anchor and candidate labels

To study why one early candidate later wins, every early candidate needs a later reference target.

Use the cross-attention `head-mean` candidate extraction at step `49`, layer `27` as the anchor observation.
For each frame \(f\), let the anchor candidate regions be

\[ A_{f,1}, \dots, A_{f,M_f}. \]

Because the final object location can sometimes be split into two or more adjacent candidate regions, define the anchor winner region as the union

\[ W_f = \bigcup_{m=1}^{M_f} A_{f,m}. \]

For one earlier candidate \(R_{s,\ell,f,k}\), define its anchor-overlap score

\[ \Omega_{s,\ell,f,k} = \operatorname{IoU}(R_{s,\ell,f,k}, W_f). \]

Also define its anchor-distance score

\[ d^{\mathrm{anchor}}_{s,\ell,f,k} = \left\| \mu(R_{s,\ell,f,k}) - \mu(W_f) \right\|_2, \]

where \(\mu(\cdot)\) denotes the geometric center of a region.

The primary winner-aligned candidate at observation \((s,\ell,f)\) is

\[ k^{+}_{s,\ell,f} = \arg\max_k \Omega_{s,\ell,f,k}. \]

This definition avoids having to set a hard overlap threshold in the main analysis and ensures that each frame contributes exactly one later-winner reference candidate.

For threshold-based pooled analysis, an optional binary survivor label can also be defined by

\[ y_{s,\ell,f,k} = \mathbf{1}[k = k^{+}_{s,\ell,f}], \]

with optional sensitivity checks based on \(\Omega_{s,\ell,f,k}\) or \(d^{\mathrm{anchor}}_{s,\ell,f,k}\).

For winner-versus-loser comparison, define the strongest loser as the non-winner candidate with the largest cross-attention proposal strength:

\[ k^{-}_{s,\ell,f} = \arg\max_{k \ne k^{+}_{s,\ell,f}} \pi_{s,\ell,f,k}. \]

Here \(\pi_{s,\ell,f,k}\) is the normalized layer-mean candidate weight from Section 7.

### 9.3 Frame-pair protocols: local and global

The stage must cover both local frame consistency and global motion-planning consistency.

For a query frame \(f\), define:

\[ \mathcal{G}_{\mathrm{local}}(f) = \{ g : |g-f| = 1 \}, \]

\[ \mathcal{G}_{\mathrm{all}}(f) = \{ g : g \ne f \}. \]

The stage should report two complementary views:

1. local view
   - uses only \(g \in \mathcal{G}_{\mathrm{local}}(f)\);
   - tests whether candidates are coordinated coherently with adjacent frames.

2. global view
   - uses all \(g \in \mathcal{G}_{\mathrm{all}}(f)\);
   - summarizes the overall degree to which one candidate is compatible with the rest of the sequence.

For global diagnostics, it is also useful to group frame pairs by signed temporal offset

\[ d = g - f, \qquad d \ne 0, \]

and visualize metrics as functions of \(d\).
This makes it possible to see whether planning is only local or whether stable links already exist to distant future or past frames.

### 9.4 Region-to-region coupling and candidate-covered mass

For one self-attention head at step \(s\), layer \(\ell\), query-frame candidate \(R_{s,\ell,f,a}\), and key-frame candidate \(R_{s,\ell,g,b}\), define the raw candidate-to-candidate coupling

\[ C_{s,\ell,h}(f,a \to g,b) = \frac{1}{|R_{s,\ell,f,a}|}\sum_{i \in R_{s,\ell,f,a}}\sum_{j \in R_{s,\ell,g,b}} \alpha_{s,\ell,h}(i,j). \]

Here:

- \(i\) is a query token index inside the query candidate;
- \(j\) is a key token index inside the target candidate;
- \(\alpha_{s,\ell,h}(i,j)\) is the self-attention probability from query token \(i\) to key token \(j\).

Let the union of all candidates in frame \(g\) be

\[ \mathcal{R}_{s,\ell,g} = \bigcup_{b=1}^{K_{s,\ell,g}} R_{s,\ell,g,b}. \]

Define the candidate-covered mass

\[ M_{s,\ell,h}(f,a \to g) = \frac{1}{|R_{s,\ell,f,a}|}\sum_{i \in R_{s,\ell,f,a}}\sum_{j \in \mathcal{R}_{s,\ell,g}} \alpha_{s,\ell,h}(i,j). \]

This quantity measures how much of the query candidate's attention to frame \(g\) actually falls inside the extracted candidate support, rather than leaking into background tokens.

Then define the candidate-normalized coupling

\[ \widetilde C_{s,\ell,h}(f,a \to g,b) = \frac{C_{s,\ell,h}(f,a \to g,b)}{M_{s,\ell,h}(f,a \to g) + \varepsilon}. \]

This normalization answers a conditional question:
given that the attention mass falls on candidate support in frame \(g\), how is that mass distributed across the candidates in that frame?

Both versions are needed:

- raw metrics computed directly from \(C_{s,\ell,h}\) or \(M_{s,\ell,h}\);
- filtered or weighted metrics that use \(M_{s,\ell,h}\) as an optional reliability gate.

The filtered version is optional because the extracted candidate support already covers most of the meaningful cross-attention mass in many cases.
Nevertheless, it is important to keep it as a controlled comparison, since some self-attention heads may still place substantial mass outside the candidate support.

### 9.5 Pairwise sharpness metrics

For one query candidate \(R_{s,\ell,f,a}\) and one target frame \(g\), define the following metrics on the candidate-normalized coupling.

1. coupling entropy

\[ H^{\mathrm{sa}}_{s,\ell,h}(f,a \to g) = -\sum_b \widetilde C_{s,\ell,h}(f,a \to g,b)\log(\widetilde C_{s,\ell,h}(f,a \to g,b) + \varepsilon). \]

This measures how diffusely the query candidate spreads its support across candidates in the target frame.
High entropy means unresolved many-to-many coupling.
Low entropy means the head is concentrating on one or a few specific target candidates.

2. dominant-link ratio

\[ D^{\mathrm{sa}}_{s,\ell,h}(f,a \to g) = \max_b \widetilde C_{s,\ell,h}(f,a \to g,b). \]

This directly measures whether one target candidate dominates the link.

3. link margin

Let \(\widetilde C^{(1)}_{s,\ell,h}(f,a \to g)\) and \(\widetilde C^{(2)}_{s,\ell,h}(f,a \to g)\) be the largest and second-largest values of \(\widetilde C_{s,\ell,h}(f,a \to g,b)\) over \(b\).
Define

\[ \Delta^{\mathrm{link}}_{s,\ell,h}(f,a \to g) = \widetilde C^{(1)}_{s,\ell,h}(f,a \to g) - \widetilde C^{(2)}_{s,\ell,h}(f,a \to g). \]

This margin is more sensitive than the dominant-link ratio when two target candidates are still in a close competition.
If the top two links are nearly tied, the system has not yet clearly committed.
If the margin becomes large, the system is selecting one target candidate over its nearest competitor.

### 9.6 Layer-mean coupling, mutual consistency, and head agreement

When a layer-level summary is needed, average the candidate-normalized coupling over the selected self-attention heads:

\[ \overline C_{s,\ell}(f,a \to g,b) = \frac{1}{|\mathcal{H}^{\mathrm{sa}}_{\ell}|}\sum_{h \in \mathcal{H}^{\mathrm{sa}}_{\ell}} \widetilde C_{s,\ell,h}(f,a \to g,b). \]

This layer-mean coupling is the default object for candidate-level compatibility analysis.

Also define the layer-mean candidate-covered mass

\[ \overline M_{s,\ell}(f,a \to g) = \frac{1}{|\mathcal{H}^{\mathrm{sa}}_{\ell}|}\sum_{h \in \mathcal{H}^{\mathrm{sa}}_{\ell}} M_{s,\ell,h}(f,a \to g). \]

From the layer-mean coupling, define the layer-level versions of the sharpness metrics:

\[ \overline H^{\mathrm{sa}}_{s,\ell}(f,a \to g) = -\sum_b \overline C_{s,\ell}(f,a \to g,b)\log(\overline C_{s,\ell}(f,a \to g,b) + \varepsilon), \]

\[ \overline D^{\mathrm{sa}}_{s,\ell}(f,a \to g) = \max_b \overline C_{s,\ell}(f,a \to g,b). \]

Let \(\overline C^{(1)}_{s,\ell}(f,a \to g)\) and \(\overline C^{(2)}_{s,\ell}(f,a \to g)\) be the largest and second-largest values of \(\overline C_{s,\ell}(f,a \to g,b)\) over \(b\).
Define the layer-level link margin

\[ \overline{\Delta}^{\mathrm{link}}_{s,\ell}(f,a \to g) = \overline C^{(1)}_{s,\ell}(f,a \to g) - \overline C^{(2)}_{s,\ell}(f,a \to g). \]

Define mutual consistency between two candidates in different frames by

\[ \mathrm{MC}_{s,\ell}(f,a; g,b) = \overline C_{s,\ell}(f,a \to g,b)\,\overline C_{s,\ell}(g,b \to f,a). \]

This score is high only when the pair supports each other in both directions.
It is intended as one early sign that two candidates belong to the same stable trajectory chain.

Next define head agreement.
For one head \(h\), let the winning target candidate in frame \(g\) be

\[ b^{\star}_{s,\ell,h}(f,a \to g) = \arg\max_b \widetilde C_{s,\ell,h}(f,a \to g,b). \]

Here \(b\) always refers to the shared `head-mean` candidate partition in key frame \(g\).
It is **not** a head-specific candidate index from a per-head region extraction.
Therefore the vote comparison across heads is well defined.

Then define the head-vote distribution over target candidates

\[ V_{s,\ell}(f,a \to g,b) = \frac{1}{|\mathcal{H}^{\mathrm{sa}}_{\ell}|}\sum_{h \in \mathcal{H}^{\mathrm{sa}}_{\ell}} \mathbf{1}[b^{\star}_{s,\ell,h}(f,a \to g) = b]. \]

The head-agreement score is

\[ A^{\mathrm{head}}_{s,\ell}(f,a \to g) = \max_b V_{s,\ell}(f,a \to g,b). \]

If different self-attention heads vote for very different target candidates, agreement is low.
If many heads vote for the same target candidate, agreement is high.
This is a direct measure of whether the layer is internally converging toward one trajectory continuation.

### 9.7 Candidate compatibility and chainability

The metrics above are defined for one query candidate and one target frame.
To explain winner selection, we also need candidate-level summaries that aggregate over target frames.

For any pairwise metric \(\Psi_{s,\ell}(f,k \to g)\), define the local and global candidate-level averages

\[ \Psi_{s,\ell,\mathrm{local}}(f,k) = \frac{1}{|\mathcal{G}_{\mathrm{local}}(f)|}\sum_{g \in \mathcal{G}_{\mathrm{local}}(f)} \Psi_{s,\ell}(f,k \to g), \]

\[ \Psi_{s,\ell,\mathrm{global}}(f,k) = \frac{1}{|\mathcal{G}_{\mathrm{all}}(f)|}\sum_{g \in \mathcal{G}_{\mathrm{all}}(f)} \Psi_{s,\ell}(f,k \to g). \]

In implementation, boundary frames should use only valid frame pairs in the denominator.
This rule applies to all local and global averages below.

First define local compatibility.
For one candidate \(R_{s,\ell,f,k}\), let

\[ \Lambda^{\mathrm{in}}_{s,\ell,\mathrm{local}}(f,k) = \sum_{k'} \overline C_{s,\ell}(f-1,k' \to f,k), \]

\[ \Lambda^{\mathrm{out}}_{s,\ell,\mathrm{local}}(f,k) = \sum_{k'} \overline C_{s,\ell}(f,k \to f+1,k'). \]

Then define the local compatibility score

\[ \Lambda_{s,\ell,\mathrm{local}}(f,k) = \Lambda^{\mathrm{in}}_{s,\ell,\mathrm{local}}(f,k) + \Lambda^{\mathrm{out}}_{s,\ell,\mathrm{local}}(f,k). \]

This asks whether the candidate is well connected to neighboring frames on both sides.

Now define the local chainability score

\[ \Gamma_{s,\ell,\mathrm{local}}(f,k) = \min\!\left( \Lambda^{\mathrm{in}}_{s,\ell,\mathrm{local}}(f,k),\ \Lambda^{\mathrm{out}}_{s,\ell,\mathrm{local}}(f,k) \right). \]

The minimum is important.
A candidate should not be considered a strong trajectory node if it only receives support from the previous frame but does not link onward, or vice versa.

For global motion-planning analysis, define

\[ \Lambda^{\mathrm{in}}_{s,\ell,\mathrm{global}}(f,k) = \frac{1}{|\{g : g < f\}|}\sum_{g < f}\sum_b \overline C_{s,\ell}(g,b \to f,k), \]

\[ \Lambda^{\mathrm{out}}_{s,\ell,\mathrm{global}}(f,k) = \frac{1}{|\{g : g > f\}|}\sum_{g > f}\sum_b \overline C_{s,\ell}(f,k \to g,b). \]

Then define

\[ \Lambda_{s,\ell,\mathrm{global}}(f,k) = \Lambda^{\mathrm{in}}_{s,\ell,\mathrm{global}}(f,k) + \Lambda^{\mathrm{out}}_{s,\ell,\mathrm{global}}(f,k), \]

\[ \Gamma_{s,\ell,\mathrm{global}}(f,k) = \min\!\left( \Lambda^{\mathrm{in}}_{s,\ell,\mathrm{global}}(f,k),\ \Lambda^{\mathrm{out}}_{s,\ell,\mathrm{global}}(f,k) \right). \]

The local scores emphasize adjacent-frame continuity.
The global scores emphasize whether one candidate is compatible with the broader motion state of the whole video.

At the first or last frame, some directions are missing.
In implementation, missing local or global directions should be treated as unavailable measurements rather than forced zeros, and the downstream aggregation should average only over valid terms.

For mutual consistency, the candidate-level summary should first take the best reciprocal partner in each target frame:

\[ \mathrm{MC}^{\max}_{s,\ell}(f,k \to g) = \max_b \mathrm{MC}_{s,\ell}(f,k; g,b). \]

Then use \(\mathrm{MC}^{\max}_{s,\ell,\mathrm{local}}(f,k)\) and \(\mathrm{MC}^{\max}_{s,\ell,\mathrm{global}}(f,k)\) through the averaging rule above.

### 9.8 Cross-attention proposal baseline reused from `candidate_consensus`

The self-attention coupling stage must not be interpreted in isolation.
To answer whether self-attention is truly selecting the winner, it must be compared against what cross-attention already preferred.

The good news is that `candidate_consensus` already saves the quantities needed for this baseline:

- `trajectory_consensus_candidate_weights.csv` stores per-head candidate weights \(a_{s,\ell,h,f,k}\);
- `trajectory_consensus_winner_gap.csv` stores layer-level winner gap and candidate entropy;
- from those weights we can recover the normalized layer-mean proposal strength \(\pi_{s,\ell,f,k}\) from Section 7.

These cached values should be reused directly.
No new candidate extraction is needed.

One implementation rule must be stated explicitly:
all cross-head comparisons in this subsection are performed on the shared `head-mean` candidate partition \(R_{s,\ell,f,k}\) from Section 9.1.
They do **not** compare candidate indices from different heads' own region extractions.
Instead, every head is asked how much mass it assigns to the same reference candidates.

To measure whether cross-attention heads already lean toward one candidate, define the per-head proposal winner

\[ k^{\star}_{s,\ell,h,f} = \arg\max_k a_{s,\ell,h,f,k}. \]

Here \(a_{s,\ell,h,f,k}\) is the mass that cross-attention head \(h\) assigns to the shared candidate region \(R_{s,\ell,f,k}\).
So the index \(k\) is comparable across heads because the partition is fixed before the per-head mass is measured.

Then define the cross-attention head-vote share for candidate \(k\):

\[ S^{\mathrm{ca}}_{s,\ell,f,k} = \frac{1}{H_{\ell}}\sum_{h=1}^{H_{\ell}} \mathbf{1}[k^{\star}_{s,\ell,h,f} = k]. \]

The overall cross-attention proposal agreement is

\[ A^{\mathrm{ca}}_{s,\ell,f} = \max_k S^{\mathrm{ca}}_{s,\ell,f,k}. \]

This baseline is important because one possible outcome is that the later winner already had a weak cross-attention advantage.
Another possible outcome is that cross-attention is still nearly tied while self-attention already shows a clear chainability or agreement advantage.
The second case would be much stronger evidence that self-attention is actively driving convergence.

### 9.9 Winner-versus-loser feature analysis

This subsection is the most directly tied to the scientific goal:
which properties make one early candidate survive while another candidate is later eliminated?

For every observation triple \((s,\ell,f)\), compare the winner-aligned candidate \(k^{+}_{s,\ell,f}\) with the strongest loser \(k^{-}_{s,\ell,f}\).

For any candidate-level metric \(\phi\), define the winner-minus-loser gap

\[ \Delta \phi_{s,\ell,f} = \phi(s,\ell,f,k^{+}_{s,\ell,f}) - \phi(s,\ell,f,k^{-}_{s,\ell,f}). \]

The primary feature list should include at least:

- cross-attention proposal strength \(\pi_{s,\ell,f,k}\);
- cross-attention vote share \(S^{\mathrm{ca}}_{s,\ell,f,k}\);
- local and global averages of layer-mean candidate-covered mass \(\overline M\), in both raw and filtered variants;
- local and global averages of layer-mean coupling entropy \(\overline H^{\mathrm{sa}}\);
- local and global averages of layer-mean dominant-link ratio \(\overline D^{\mathrm{sa}}\);
- local and global averages of layer-mean link margin \(\overline{\Delta}^{\mathrm{link}}\);
- local and global averages of mutual consistency \(\mathrm{MC}^{\max}\);
- local and global averages of self-attention head agreement \(A^{\mathrm{head}}\);
- local compatibility \(\Lambda_{\mathrm{local}}\);
- local chainability \(\Gamma_{\mathrm{local}}\);
- global compatibility \(\Lambda_{\mathrm{global}}\);
- global chainability \(\Gamma_{\mathrm{global}}\).

For pooled statistics across many candidates, the stage should provide three analyses.

1. continuous overlap correlation
   - correlate each feature with the anchor-overlap score \(\Omega_{s,\ell,f,k}\);
   - this tests whether the feature changes smoothly with how close the early candidate is to the eventual anchor region.
   - In implementation, the per-\((s,\ell)\) feature summary stores one scalar overlap-correlation field `anchor_iou_correlation`, while the visualization layer additionally exposes selected feature-versus-\(\Omega\) scatter plots for direct inspection.

2. winner-versus-loser gap analysis
   - use \(\Delta \phi_{s,\ell,f}\) to measure whether the winner already has a systematic early advantage over the strongest loser;
   - large positive values indicate an early winner-specific signal.

3. ranking power analysis
   - treat \(y_{s,\ell,f,k} = \mathbf{1}[k = k^{+}_{s,\ell,f}]\) as the binary label;
   - evaluate how well each feature ranks winner-aligned candidates above non-winners, for example by AUROC or average precision.

The interpretation is:

- if a feature has high ranking power very early, it is an informative early signature of survival;
- if a feature only becomes discriminative late, it is more likely a consequence of convergence than a cause or precursor;
- if self-attention features become predictive before cross-attention proposal strength does, that is direct evidence that self-attention is not merely following the winner but helping to determine it.

### 9.10 Temporal-precedence analysis

The previous subsection measures whether a feature distinguishes winners from losers.
Temporal-precedence asks a stronger question:
which feature family separates winner from loser first?

Let the selected observation points be ordered by actual denoising execution order:

\[ t = 1, \dots, T, \qquad (s_t, \ell_t). \]

For one frame \(f\) and one candidate-level feature \(\phi\), define

\[ \Delta \phi_t(f) = \phi(s_t,\ell_t,f,k^{+}_{s_t,\ell_t,f}) - \phi(s_t,\ell_t,f,k^{-}_{s_t,\ell_t,f}). \]

The first stable separation time of feature \(\phi\) on frame \(f\) is

\[ \tau_{\phi}(f) = \min \left\{ t : \Delta \phi_{t'}(f) > 0 \text{ for all } t' \in \{t, \dots, t+r-1\} \right\}, \]

where \(r\) is a small persistence window.

This definition intentionally avoids declaring a precedence event from a one-step fluctuation.
It asks when the winner's advantage becomes consistently positive.

The main comparison should be:

- cross-attention proposal features such as \(\pi\) or \(S^{\mathrm{ca}}\);
- self-attention coupling features such as \(\Gamma_{\mathrm{local}}\), \(\Gamma_{\mathrm{global}}\), \(\mathrm{MC}\), and \(A^{\mathrm{head}}\).

If self-attention features achieve stable winner-versus-loser separation earlier than cross-attention proposal features, that strongly supports the hypothesis that self-attention is actively driving candidate convergence.

### 9.11 Planned implementation protocol

The planned `self_attention_coupling` stage should:

1. reuse clean candidate-region caches from `candidate_consensus`;
2. reuse `trajectory_consensus_candidate_weights.csv` to recover the cross-attention proposal baseline;
3. collect self-attention probabilities for selected heads, selected steps, selected layers, and selected frame pairs;
4. compute pairwise coupling statistics \(C\), \(M\), \(\widetilde C\), \(H^{\mathrm{sa}}\), \(D^{\mathrm{sa}}\), and \(\Delta^{\mathrm{link}}\);
5. compute layer-level summaries \(\overline C\), \(\mathrm{MC}\), and \(A^{\mathrm{head}}\);
6. aggregate them into candidate-level local and global scores such as \(\Lambda\) and \(\Gamma\);
7. construct the anchor-overlap score \(\Omega\), the winner-aligned candidate \(k^{+}\), and the strongest loser \(k^{-}\);
8. compare winner-aligned candidates and losers by overlap correlation, winner-minus-loser gaps, and ranking power;
9. run temporal-precedence analysis to compare self-attention features against the cross-attention proposal baseline;
10. later combine with `candidate_intervention` so that heads selected by this stage can be tested causally.

### 9.12 Planned visualizations

Because this stage simultaneously contains step, layer, head, frame, and candidate dimensions, one single figure type is never sufficient.
The visualization design should therefore be explicitly split into three layers:

- Layer I: mechanistic qualitative figures
  - answer: what exactly is the model attending to, and how does one candidate defeat another?
- Layer II: mesoscopic trend figures
  - answer: how do the important quantities evolve across time offsets and observation order?
- Layer III: global scan figures
  - answer: where in the step-layer plane should we look first?

The following figure families are the recommended default set.

#### Layer I. Mechanistic qualitative figures

1. candidate score overlay panel
   - Purpose:
     - directly show which candidate regions are already favored inside one frame.
   - Content:
     - for one selected observation \((s,\ell)\), draw the candidate masks on selected frames;
     - color each candidate region by one selected scalar, such as \(\pi\), \(\Gamma_{\mathrm{local}}\), \(\Gamma_{\mathrm{global}}\), or \(\mathrm{MC}^{\max}_{\mathrm{global}}\);
     - mark the winner-aligned candidate and the strongest loser with different contour colors.
   - Drawing method:
     - each subplot corresponds to one frame;
     - candidate regions are filled with a continuous color scale according to the selected metric;
     - annotate the candidate index and the scalar value near the candidate centroid.
   - Interpretation:
     - if the winner region is already visually more intense than its competitors, that metric is an early winner signature;
     - if the loser still looks equally strong, then convergence has not yet occurred under that metric.
   - Current default implementation:
     - render one panel for each of `proposal_pi`, `global_chainability`, and `global_mutual_consistency`;
     - choose one representative observation \((s,\ell)\), then draw several evenly spaced frames and always include the most informative representative frame used by the evolution analysis.

2. winner-versus-loser coupling storyboard
   - Purpose:
     - show the actual candidate-to-candidate routing pattern, instead of only showing aggregated statistics.
   - Content:
     - choose one observation \((s,\ell)\), one query frame \(f\), and compare two query candidates:
       - the winner-aligned candidate \(k^{+}\),
       - the strongest loser \(k^{-}\);
     - for each query candidate, show how it couples to candidates in several target frames.
   - Drawing method:
     - the first panel highlights the chosen query candidate in its own frame;
     - the following panels show the target-frame candidate masks, with each target candidate colored by either raw coupling \(C\) or normalized coupling \(\widetilde C\);
     - the panel title should also report candidate-covered mass \(M\), so the reader can judge whether the coupling is reliable.
   - Interpretation:
     - a winner candidate should gradually show more selective and coherent cross-frame routing;
     - a loser candidate often remains diffuse, fragmented, or inconsistent across target frames.
   - Current default implementation:
     - choose one representative observation \((s,\ell)\) and one representative query frame \(f\);
     - compare the winner-aligned candidate and the strongest loser on the same query frame;
     - the target-frame set always contains both local neighbors when available and representative global frames such as the beginning, middle, and end of the sequence.

3. winner-loser feature evolution panel
   - Purpose:
     - show when the winner starts separating from the loser along the denoising trajectory.
   - Content:
     - for one selected feature, plot the winner value and strongest-loser value across ordered observation index \(t\).
   - Drawing method:
     - x-axis is observation index \(t\), ordered by actual denoising execution order;
     - y-axis is one selected feature, such as \(\pi\), \(\Gamma_{\mathrm{global}}\), or \(\mathrm{MC}^{\max}_{\mathrm{global}}\);
     - optionally add a vertical marker for the first stable separation time \(\tau_{\phi}\).
   - Interpretation:
     - if the winner curve consistently rises above the loser curve early, the feature is a plausible early selection signal;
     - if the two curves remain entangled until late, the feature is more likely a consequence of convergence than a precursor.
   - Current default implementation:
     - use one representative frame and render a multi-panel figure rather than only one scalar feature;
     - the default feature set is `proposal_pi`, `proposal_vote_share`, `local_chainability`, `global_chainability`, `global_mutual_consistency`, and `global_head_agreement`;
     - the stable-separation marker \(\tau_{\phi}\) is overlaid when it exists.

#### Layer II. Mesoscopic trend figures

4. signed-offset planning curve
   - Purpose:
     - measure whether self-attention only enforces local continuity or already coordinates long-range motion structure.
   - Content:
     - summarize one pairwise metric as a function of signed frame offset \(d = g-f\), for example entropy, dominant-link ratio, link margin, or head agreement.
   - Drawing method:
     - x-axis is signed offset \(d\);
     - y-axis is the selected metric averaged over candidates at one observation \((s,\ell)\);
     - plot several representative observations on the same figure as separate curves.
   - Interpretation:
     - if later or stronger planning observations show lower entropy and larger link margin even at large \(|d|\), the model is not only doing local smoothing but also organizing global motion.
   - Current default implementation:
     - render a four-panel figure for entropy, dominant-link ratio, link margin, and head agreement;
     - compare a small set of representative observations sampled from early, middle, and late parts of the observation order.

5. cross-attention versus self-attention competition curve
   - Purpose:
     - directly compare whether early winner bias first appears in cross-attention proposal or in self-attention coordination.
   - Content:
     - place winner-minus-loser gap curves of selected CA features and selected SA features on the same observation-order axis.
   - Drawing method:
     - x-axis is observation index \(t\);
     - y-axis is the frame-averaged winner-minus-loser gap;
     - recommended CA features are \(\pi\) and \(S^{\mathrm{ca}}\);
     - recommended SA features are \(\Gamma_{\mathrm{local}}\), \(\Gamma_{\mathrm{global}}\), \(\mathrm{MC}^{\max}_{\mathrm{global}}\), and \(A^{\mathrm{head}}_{\mathrm{global}}\).
   - Interpretation:
     - if SA curves pull away from zero earlier than CA curves, that supports the hypothesis that self-attention actively drives convergence;
     - if CA already shows a clear gap first, then SA may be amplifying an existing proposal bias rather than creating it.
   - Current default implementation:
     - draw one shared plot whose CA curves are `proposal_pi` and `proposal_vote_share`;
     - the default SA curves are `local_chainability`, `global_chainability`, `global_mutual_consistency`, and `global_head_agreement`.

6. temporal-precedence summary panel
   - Purpose:
     - compress many winner-loser evolution curves into one ranking of which feature family separates first.
   - Content:
     - compare \(\tau_{\phi}\) across features.
   - Drawing method:
     - the detailed figure should still be the observation-order curve from item 3;
     - the summary figure can be a bar chart or small table of mean precedence index across frames.
   - Interpretation:
     - this panel is not the primary evidence by itself;
     - it is a compact summary of the richer temporal curves.
   - Current default implementation:
     - output one bar chart of mean first-stable-separation index across frames.

#### Layer III. Global scan figures

7. winner-loser gap heatmap
   - Purpose:
     - quickly locate where one feature most strongly distinguishes winners from losers.
   - Content:
     - one heatmap per feature.
   - Drawing method:
     - x-axis is layer, y-axis is step;
     - the value is the frame-averaged winner-minus-loser gap \(\Delta \phi_{s,\ell,f}\).
   - Interpretation:
     - use this as a navigation map, not as the final mechanistic evidence;
     - the role of this figure is to tell us which observations deserve closer qualitative inspection.
   - Current default implementation:
     - save these heatmaps under the Layer III navigation directory, separate from the qualitative and trend figures, so they are visually de-emphasized.

8. ranking-power heatmap
   - Purpose:
     - identify which features are genuinely predictive of survival, rather than only large in magnitude.
   - Content:
     - one heatmap per feature.
   - Drawing method:
     - x-axis is layer, y-axis is step;
     - the value is AUROC or average precision for predicting \(y_{s,\ell,f,k}\).
   - Interpretation:
     - high ranking power early in diffusion is strong evidence that the feature contains useful selection information.

9. optional diagnostic scatter plot
   - Purpose:
     - check whether one feature changes smoothly with the anchor-overlap score \(\Omega\), or whether its apparent usefulness is driven by a few outliers.
   - Content:
     - scatter one selected feature against \(\Omega\).
   - Drawing method:
     - x-axis is the feature value, y-axis is \(\Omega\);
     - color points by winner versus loser when possible.
   - Interpretation:
     - this is a diagnostic tool, not a primary result figure.
   - Current default implementation:
     - render selected diagnostic scatters for `proposal_pi`, `global_chainability`, and `global_mutual_consistency`.

In summary, Layer I figures should be treated as the primary mechanistic evidence, Layer II figures as the main temporal-trend analysis, and Layer III figures only as large-scale navigation maps.

## 10. Optional Appendix: Trajectory Graph and Dynamic Programming

This section is retained as an optional appendix only.
It is **not** a near-term implementation priority.

The reason is simple:

- by itself, trajectory decoding mostly re-describes an already observed convergence phenomenon;
- it does not directly identify which module or which head drives disambiguation;
- it becomes much more meaningful only after the coupling stage in Section 9 is mature.

The formulas are kept here because they may later be useful as a derived summary.

### 10.1 Node score

\[ u_{f,k} = \log(\bar a_{s,\ell,f,k} + \varepsilon). \]

### 10.2 Edge score

Let \(c_{f,k}\) be the candidate center.
Let \(C_f(k \to k')\) be a self-attention coupling summary between adjacent frames.
Define

\[ \Psi_f(k \to k') = \lambda_{\mathrm{sa}} \log(C_f(k \to k') + \varepsilon) - \lambda_{\mathrm{geo}}\|c_{f+1,k'} - c_{f,k}\|_2^2. \]

### 10.3 Trajectory score

For one candidate trajectory \(T = (k_1, k_2, \dots, k_F)\), define

\[ S(T) = \sum_{f=1}^{F} u_{f,k_f} + \sum_{f=1}^{F-1} \Psi_f(k_f \to k_{f+1}). \]

### 10.4 Decoding

The best trajectory can be decoded by Viterbi-style max-sum dynamic programming:

\[ \mathrm{DP}_1(k) = u_{1,k}, \]

\[ \mathrm{DP}_{f+1}(k') = u_{f+1,k'} + \max_k \left( \mathrm{DP}_f(k) + \Psi_f(k \to k') \right). \]

Store a backpointer

\[ \mathrm{bp}_{f+1}(k') = \arg\max_k \left( \mathrm{DP}_f(k) + \Psi_f(k \to k') \right). \]

The final best path is obtained by tracing back from \(\arg\max_k \mathrm{DP}_F(k)\).

For top-\(K\) paths, beam search is the recommended first implementation.

## 11. Seed Sensitivity Interpretation

The intended mechanistic interpretation is now centered on Sections 8 and 9.

- seeds perturb early candidate weights \(a_{s,\ell,h,f,k}\) and therefore also perturb the clean winner gap \(G_{s,\ell,f}\);
- seeds also perturb candidate-to-candidate compatibility scores \(C_{s,\ell,h}(f,a \to g,b)\), chainability scores \(\Gamma_{s,\ell,\mathrm{local}}(f,k)\) and \(\Gamma_{s,\ell,\mathrm{global}}(f,k)\), mutual consistency \(\mathrm{MC}_{s,\ell}(f,a;g,b)\), and head agreement \(A^{\mathrm{head}}_{s,\ell}(f,a \to g)\);
- when several candidates are close in score, a small perturbation can change the clean winner identity;
- later layers and later diffusion steps then amplify that early difference through residual accumulation and self-attention coordination.

This is the connection between seed sensitivity and physical inconsistency.
The main tools for testing it are therefore:

- head intervention on candidate competition;
- self-attention candidate coupling;
- cross-attention proposal baseline versus self-attention precedence comparison;
- phase-specialization analysis on top of those two.

## 12. Phase Specialization

`phase_specialization` should be treated as a secondary analysis built on saved caches.
It should not recollect activations by itself.

### 12.1 Why this is secondary

The main scientific question is not whether a head can be given a label, but which mechanisms drive consensus formation.
Therefore phase labels should be assigned **after** the primary causal and coupling readouts are available.

The recommended input caches are:

- head contribution metrics from Section 5;
- candidate weights and winner-gap curves from Sections 6 and 7;
- head intervention metrics from Section 8;
- self-attention coupling metrics from Section 9.

### 12.2 Phase windows from clean consensus dynamics

Let the ordered downstream observation index be \(t = 1, \dots, T\), where one index \(t\) corresponds to one selected pair \((s_t, \ell_t)\).
Using the clean candidate statistics, define four phase windows.

1. proposal window \(\mathcal{P}\)
   - observations where more than one candidate is present and the winner gap is still small.

2. pruning window \(\mathcal{R}\)
   - observations where the winner gap increases rapidly and candidate entropy decreases rapidly.

3. commitment window \(\mathcal{C}\)
   - the first sustained window in which the clean winner identity stays unchanged and the winner gap remains above a high threshold.

4. grounding window \(\mathcal{A}\)
   - later observations after commitment, where winner-gap changes are small but final `v_pred` contribution can still be substantial.

The exact thresholds should be derived from the clean winner-gap curve rather than fixed globally whenever possible.

### 12.3 Role scores

For one analyzed head \(q\), first aggregate frame-wise intervention metrics into one observation-level summary at observation index \(t\):

\[ \Delta S_{q \to t} = \frac{1}{F_t}\sum_{f=1}^{F_t} \Delta S_{q \to s_t,\ell_t,f}, \]

\[ \Delta G_{q \to t} = \frac{1}{F_t}\sum_{f=1}^{F_t} \Delta G_{q \to s_t,\ell_t,f}, \]

\[ \Delta H_{q \to t} = \frac{1}{F_t}\sum_{f=1}^{F_t} \Delta H_{q \to s_t,\ell_t,f}, \]

\[ \Delta A_{q \to t}^{\star} = \frac{1}{F_t}\sum_{f=1}^{F_t} \Delta A_{q \to s_t,\ell_t,f}^{\star}, \]

\[ F_{q \to t} = \frac{1}{F_t}\sum_{f=1}^{F_t} F_{q \to s_t,\ell_t,f}. \]

Here \(F_t\) denotes the number of latent frames evaluated at observation \(t\).

Then define four role scores from existing caches.

1. proposal score

\[ \mathrm{Proposal}(q) = \frac{1}{|\mathcal{P}|}\sum_{t \in \mathcal{P}} \Delta S_{q \to t}. \]

2. pruning score

\[ \mathrm{Pruning}(q) = \frac{1}{|\mathcal{R}|}\sum_{t \in \mathcal{R}} \left( \Delta G_{q \to t} + \lambda_H \Delta H_{q \to t} \right). \]

3. commitment score

\[ \mathrm{Commitment}(q) = \frac{1}{|\mathcal{C}|}\sum_{t \in \mathcal{C}} \left( \Delta A_{q \to t}^{\star} + \lambda_F F_{q \to t} \right). \]

4. grounding score

\[ \mathrm{Grounding}(q) = \mathrm{Contr}_{v}(q) - \lambda_G \frac{1}{|\mathcal{A}|}\sum_{t \in \mathcal{A}} \Delta G_{q \to t}. \]

Here \(\mathrm{Contr}_{v}(q)\) denotes one selected final-output contribution metric from Section 5, such as `cos_obj` or `dot_obj`.

The intended interpretation is:

- high `Proposal` means the head mainly supports candidate presence or object grounding;
- high `Pruning` means the head helps suppress ambiguity among candidates;
- high `Commitment` means the head stabilizes the winner identity;
- high `Grounding` means the head contributes strongly to the final output while affecting candidate competition only weakly.

### 12.4 Assignment strategy

The first implementation should use a transparent rule-based assignment:

- z-score all four role scores across analyzed heads;
- assign a head to the role with the largest z-score only if that z-score exceeds a minimum confidence threshold;
- otherwise mark the head as `mixed`.

This is preferable to clustering as a first pass because it is easier to audit scientifically.

## 13. Current and Planned Outputs

The current implementation can produce:

- `trajectory_consensus_candidate_regions.csv`
- `trajectory_consensus_candidate_regions.pt`
- `trajectory_consensus_candidate_regions_per_head.csv`
- `trajectory_consensus_candidate_regions_per_head.pt`
- `trajectory_consensus_candidate_weights.csv`
- `trajectory_consensus_winner_gap.csv`
- `trajectory_consensus_candidate_region_viz/`
- `trajectory_consensus_candidate_plots/`
- `trajectory_consensus_head_contribution.csv`
- `trajectory_consensus_head_contribution_plots/`
- `trajectory_consensus_head_contribution/<method>/trajectory_consensus_summary.json`

The two `.pt` candidate caches store only compact integer label maps. The CSV files
store only the frame-wise candidate metadata required for plotting and later analysis.

Planned later outputs include:

- `trajectory_consensus_intervention.csv`
- `trajectory_consensus_intervention_plots/`
- `trajectory_consensus_self_attention_coupling_pairwise.csv`
- `trajectory_consensus_self_attention_coupling_candidate_features.csv`
- `trajectory_consensus_self_attention_coupling_temporal_precedence.csv`
- `trajectory_consensus_self_attention_plots/`
- `trajectory_consensus_phase_scores.csv`
- optional `trajectory_consensus_trajectory_graph.csv`
- optional `trajectory_consensus_topk_paths.json`

## 14. Recommended Execution Order

The recommended workflow is:

1. run `cross_attention_token_viz` and save reusable cross-attention maps;
2. optionally run `head_trajectory_dynamics` if early-alignment scatter plots are needed;
3. run `trajectory_consensus_dynamics` with `trajectory_consensus_stages=candidate_consensus`;
4. run the planned `self_attention_coupling` stage, because it reuses the cached candidate partition and helps identify the most interesting self-attention heads and candidate features;
5. run `trajectory_consensus_dynamics` with `trajectory_consensus_stages=head_contribution`;
6. later run the planned `candidate_intervention` stage on a smaller set of selected source heads selected from the earlier analyses;
7. run `phase_specialization` only after the earlier caches exist.

Rerun any completed stage with `trajectory_consensus_plot_only_from_csv=True` whenever only visualization updates are needed.

This order keeps expensive runtime ablation separate from offline candidate extraction and makes the later phase analysis auditable.

## 15. Practical Notes

- The current heatmaps in this experiment use a diverging `bwr` colormap.
- The current code parallelizes candidate-region extraction and candidate visualization rendering over CPU workers.
- The candidate extractor must be validated visually before any quantitative interpretation is trusted.
- Candidate intervention should always evaluate ablated runs on the clean candidate partition before introducing any more complex rematching logic.
- Frame-wise candidate regions are the primary representation; the trajectory-graph appendix is optional and not a current priority.
- The object mask for contribution analysis is reused from the existing reference-support pipeline at step `50`, layer `27`.
