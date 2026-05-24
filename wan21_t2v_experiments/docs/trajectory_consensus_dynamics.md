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
   - Question: does self-attention convert many-to-many candidate coupling into sparse and selective candidate-to-candidate coupling?
   - Main readout: region-to-region coupling matrices, coupling entropy, dominant-link ratio, and candidate compatibility scores.

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
   - Status: planned, not yet implemented in the current code.
   - Will cover candidate-to-candidate coupling collapse and compatibility scores.

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
It aims to explain why some candidates survive while others are eliminated.

### 9.1 Region-to-region coupling

For one self-attention head at diffusion step \(s\), layer \(\ell\), query-frame candidate \(R_{f,a}\), and key-frame candidate \(R_{g,b}\), define

\[ C_{s,\ell,h}(f,a \to g,b) = \frac{1}{|R_{f,a}|}\sum_{i \in R_{f,a}}\sum_{j \in R_{g,b}} \alpha_{s,\ell,h}(i,j). \]

Here:

- \(i\) is a query token index inside candidate region \(R_{f,a}\);
- \(j\) is a key token index inside candidate region \(R_{g,b}\);
- \(\alpha_{s,\ell,h}(i,j)\) is the self-attention probability from query token \(i\) to key token \(j\).

Normalize over all candidate regions in the target frame:

\[ \widetilde C_{s,\ell,h}(f,a \to g,b) = \frac{C_{s,\ell,h}(f,a \to g,b)}{\sum_{b'} C_{s,\ell,h}(f,a \to g,b') + \varepsilon}. \]

### 9.2 Candidate-coupling entropy

For one query candidate \(R_{f,a}\), define

\[ H_{s,\ell,h}^{\mathrm{sa}}(f,a \to g) = -\sum_b \widetilde C_{s,\ell,h}(f,a \to g,b)\log(\widetilde C_{s,\ell,h}(f,a \to g,b) + \varepsilon). \]

If one query candidate spreads attention across many future candidates, this entropy is high.
If it becomes concentrated on one future candidate, this entropy is low.

### 9.3 Dominant-link ratio

\[ D_{s,\ell,h}^{\mathrm{sa}}(f,a \to g) = \max_b \widetilde C_{s,\ell,h}(f,a \to g,b). \]

This directly measures whether candidate-to-candidate interaction becomes close to one-to-one.

### 9.4 Candidate compatibility score

For one layer-level summary, first average normalized coupling over a selected self-attention head set \(\mathcal{H}_{\ell}^{\mathrm{sa}}\):

\[ \overline C_{s,\ell}(f,a \to g,b) = \frac{1}{|\mathcal{H}_{\ell}^{\mathrm{sa}}|}\sum_{h \in \mathcal{H}_{\ell}^{\mathrm{sa}}} \widetilde C_{s,\ell,h}(f,a \to g,b). \]

Then define the candidate compatibility score

\[ \Lambda_{s,\ell}(f,k) = \sum_{k'} \overline C_{s,\ell}(f-1,k' \to f,k) + \sum_{k'} \overline C_{s,\ell}(f,k \to f+1,k'). \]

This score asks whether candidate \(R_{f,k}\) is well connected to plausible candidates in neighboring frames.

The working hypothesis is:

- candidates with weak compatibility are more likely to be suppressed later;
- candidates that survive into the final trajectory should form a high-compatibility chain across frames.

### 9.5 Planned implementation protocol

The planned `self_attention_coupling` stage should:

1. reuse clean candidate regions from `candidate_consensus`;
2. collect self-attention probabilities for selected heads, selected steps, and selected frame pairs;
3. compute \(C_{s,\ell,h}(f,a \to g,b)\), \(\widetilde C_{s,\ell,h}(f,a \to g,b)\), \(H_{s,\ell,h}^{\mathrm{sa}}(f,a \to g)\), \(D_{s,\ell,h}^{\mathrm{sa}}(f,a \to g)\), and \(\Lambda_{s,\ell}(f,k)\);
4. compare whether low-compatibility candidates are precisely the ones later losing winner-gap competition;
5. later combine with `candidate_intervention` to test whether removing one self-attention head weakens the compatibility chain of the clean winner.

### 9.6 Planned visualizations

The stage should generate:

- candidate-to-candidate coupling heatmaps;
- coupling-entropy curves over diffusion steps;
- dominant-link curves over diffusion steps;
- compatibility overlays on candidate masks;
- selected qualitative panels showing many-to-many coupling changing into sparse one-to-one coupling.

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
- seeds also perturb candidate-to-candidate compatibility scores \(C_{s,\ell,h}(f,a \to g,b)\) and \(\Lambda_{s,\ell}(f,k)\);
- when several candidates are close in score, a small perturbation can change the clean winner identity;
- later layers and later diffusion steps then amplify that early difference through residual accumulation and self-attention coordination.

This is the connection between seed sensitivity and physical inconsistency.
The main tools for testing it are therefore:

- head intervention on candidate competition;
- self-attention candidate coupling;
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
- `trajectory_consensus_self_attention_coupling.csv`
- `trajectory_consensus_self_attention_plots/`
- `trajectory_consensus_phase_scores.csv`
- optional `trajectory_consensus_trajectory_graph.csv`
- optional `trajectory_consensus_topk_paths.json`

## 14. Recommended Execution Order

The recommended workflow is:

1. run `cross_attention_token_viz` and save reusable cross-attention maps;
2. optionally run `head_trajectory_dynamics` if early-alignment scatter plots are needed;
3. run `trajectory_consensus_dynamics` with `trajectory_consensus_stages=candidate_consensus`;
4. run `trajectory_consensus_dynamics` with `trajectory_consensus_stages=head_contribution`;
5. later run the planned `candidate_intervention` stage on a smaller set of selected source heads;
6. later run the planned `self_attention_coupling` stage;
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
