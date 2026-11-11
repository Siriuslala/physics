# RoPE Posthoc Attention Comparison

## Motivation and interpretation

This CPU-only analysis compares baseline and finetuned, spatial-RoPE-modified runs using the first latent frame's object region as the query support. It tests whether attention from those queries preferentially reaches other frames' object regions or remains near the first frame's spatial coordinates. It does not assume that the modified condition improves this preference.

The inputs are the saved probability maps from `self_attention_viz` and visually audited object supports from `region_audit`. Each condition uses its own late-denoising cross-attention reference. Key-region extraction settings may differ from those used when collecting query tokens. Optional separate query audits verify the fixed query support; changing key masks does not retroactively change the sampled queries. Seed 8 uses reference step 45, layer 27; seeds 20 and 29 use reference step 50, layer 27. These reference supports are retrospective proxies for object position, not direct measurements of early-step candidate positions or independently annotated ground truth.

A baseline versus modified comparison changes finetuned weights as well as spatial RoPE. The generated trajectories and query supports may also differ. Consequently, a favorable attention redistribution is mechanistic evidence consistent with the proposed explanation, but cannot independently identify a causal effect attributable exclusively to RoPE. An isolated RoPE intervention requires matched weights and additional controlled comparisons. The requested first-frame probe also cannot establish behavior for every possible query frame.

## Probability maps and reference regions

Let $F$, $H$, and $W$ be the latent frame count, token-grid height, and token-grid width, and let $N$ be the attention head count. The condition index is $c\in\{\mathrm{baseline},\mathrm{modified}\}$, $s$ is a one-based denoising step, $\ell$ is the saved layer index, and $h\in\{0,\ldots,N-1\}$ is a zero-based head index. The saved first-frame-query map is $A^{c,s,\ell,h}\in\mathbb{R}_{\geq0}^{F\times H\times W}$, averaged over sampled query tokens and normalized over all key tokens. Its entries sum to one within numerical tolerance.

The audited key mask is $M^c\in\{0,1\}^{F\times H\times W}$, and the independently verified actual query support is $Q^c\in\{0,1\}^{H\times W}$. For key frame $f\in\{1,\ldots,F-1\}$, define the object region $O_f^c=\{(y,x):M^c[f,y,x]=1\}$ and the same-coordinate region $C^c=\{(y,x):Q^c[y,x]=1\}$. All reported metrics exclude key frame zero, which is also the query frame. Let $\mathcal V\subseteq\{1,\ldots,F-1\}$ denote the intersection of key frames explicitly marked valid by visual review in both conditions. Only frames in $\mathcal V$ enter probability metrics and their summaries. Invalid, uncertain, and unreviewed key regions are excluded in both conditions, ensuring paired comparisons use the same frames. The program refuses to proceed if this intersection is empty or the actual first-frame query lacks positive visual review. The comparison additionally evaluates the overlap $I_f^c=O_f^c\cap C^c$ and the disjoint regions $U_f^c=O_f^c\setminus C^c$ and $V_f^c=C^c\setminus O_f^c$.

The scalar overlap statistic is:

$$\operatorname{IoU}_f^c=\frac{|I_f^c|}{|O_f^c\cup C^c|}.$$

An identical object position across frames makes the full object and same-coordinate scores indistinguishable. Reporting overlap and exclusive regions prevents that shared support from being mistaken for evidence of object tracking.

## Region metrics

Suppress condition, step, layer, and head superscripts temporarily. Let $R$ be any of the five regions $O_f,C,I_f,U_f,V_f$, with area $n_R=|R|$. The key-frame global probability mass is the scalar $b_f=\sum_{y,x}A[f,y,x]$. The global region mass $m_{f,R}$, within-frame conditional mass $p_{f,R}$, global per-token density $d_{f,R}$, and conditional per-token density $e_{f,R}$ are:

$$m_{f,R}=\sum_{(y,x)\in R}A[f,y,x],\qquad p_{f,R}=m_{f,R}/b_f,\qquad d_{f,R}=m_{f,R}/n_R,\qquad e_{f,R}=p_{f,R}/n_R.$$

The conditional probability removes changes in total attention received by a frame; the density additionally adjusts for region area. Both global and conditional metrics are retained because either normalization alone can conceal a relevant change. Zero frame mass gives undefined conditional quantities; zero region area gives undefined densities. Undefined entries are written as `nan` in CSV files.

The exclusive density contrast is the scalar:

$$\Delta_f=e_{f,U_f}-e_{f,V_f}.$$

A positive contrast indicates greater conditional attention per token inside the object-only region than inside the same-coordinate-only region. It does not by itself establish physically correct motion. A frame with either exclusive region empty has undefined contrast and contributes no value to its mean.

For each step, layer, and head, the output includes an unweighted arithmetic mean over finite values for frames in $\mathcal V$ of each metric. The count of frames with defined exclusive contrasts is reported separately. Global masses are additionally summed over $\mathcal V$, preserving their interpretation relative to the full attention distribution. Excluded-frame probability mass is not renormalized away. A `mean` head row is computed from $\bar A=N^{-1}\sum_{h=0}^{N-1}A^h$ before conditional normalization; it is not the mean of separately normalized head scores.

## Implementation and artifacts

`rope_posthoc_compare.py` loads the two `self_attention_viz_maps.pt` dictionaries, keeps first-frame query keys, and compares the intersection of their step/layer keys. Unpaired keys are recorded in `comparison.json`. Tensor shapes, nonnegative finite probabilities, and probability sums are validated. The first-frame query audit is checked against the center, radius, and area in the self-attention run's `self_attention_viz_reference_support.csv`. Other key masks may use an independently tuned extraction configuration, which is recorded explicitly. The same-coordinate comparator always uses the verified original query support, not the independently tuned key audit's first-frame support.

Each audit directory supplies `reference_support.pt` and `region_audit.json`. The former contains `support_mask_fhw`, `center_trajectory`, and `support_radius_per_frame`; the latter records extraction settings and a `frames` list with zero-based `latent_frame`, textual `review_status`, and Boolean `valid_for_object_metrics`. Missing or non-true validity flags are excluded. Query and key audit metadata are both retained. Only trusted local Torch artifacts should be supplied.

The output consists of:

- `frame_inclusion.csv`: every nonquery frame's per-condition review status and whether it enters paired metrics.
- `per_frame_metrics.csv`: all region areas, overlap, masses, densities, and exclusive contrasts for every condition, step, layer, head, and jointly valid nonquery key frame.
- `per_head_summary.csv`: frame averages and summed global masses, including the probability-map head mean.
- `step_XX_layer_YY_headmean.png`: paired global-probability maps. The full visualization pass covers steps 1–20 and all layers 0–29, with baseline above modified and ten frames per row. It uses `viridis` with alpha 0.95, green solid object outlines, and yellow dashed query-coordinate outlines. Only positively reviewed object supports receive green outlines. There are no per-frame titles or colorbars. Limits are shared within each figure and recorded in `overview_index.csv`; limits are not shared between figures.
- `step_XX_layer_YY_headmean_conditional.png`: the same layout after normalizing each head-mean key-frame map to unit mass. Head averaging precedes normalization. This view measures spatial allocation within a frame while discarding how much total attention that frame receives.
- `overview_metadata.json`: normalization, frame selection, colors, and outline semantics. The standard `self_attention_viz` sampler selects one-based video frames 1, 9, 17, 25, 33, 41, 49, 57, 65, 81, corresponding to zero-based latent frames 0, 2, 4, 6, 8, 10, 12, 14, 16, 20. The query frame appears as a visual reference but is excluded from all cross-frame metrics.
- `step_trends_layer_YY.png`: all 50 steps for each layer; `step_trends.png` is the layer-27 compatibility copy. The three panels show summed object global mass, mean conditional object probability, and strictly paired mean exclusive density contrast. The shaded interval denotes steps 1–5.
- `alignment_step_layer.png`: baseline, modified, and their paired difference over the first 20 steps and all 30 layers. `alignment_first5_steps.png` shows the intervention window separately with its own scales, preventing later large magnitudes from hiding early changes.
- `alignment_key_frames_layer_YY.png`: the same triptych without averaging key frames, for each layer. Grey cells have no defined paired contrast.
- `alignment_paired_scatter.png`: each point represents one step/layer pair. The horizontal coordinate is baseline contrast and the vertical coordinate is modified contrast. Points above the identity diagonal indicate increased relative object preference; orange indicates steps 1–5 and blue indicates steps 6–20.
- `alignment_valid_frame_counts.png`: the number of identical, defined frames used per step/layer cell.
- `paired_alignment_frames.csv`, `paired_alignment_step_layer.csv`, and `alignment_metrics.json`: first-20-step paired values, counts, and interpretation metadata.

The three row-pair overview entry points are `comparison_overview_seed_8.png`, `comparison_overview_seed_20.png`, and `comparison_overview_seed_29.png` in the experiment root. Each is a copy of its seed's step-5, layer-27 conditional figure. Layer 27 is an illustrative entry point, not a restriction on either data collection or the full visualization pass. The original combined `comparison_overview.png` is a legacy artifact.

### Trend axes and paired changes

For the head-mean map $\bar A$, the first trend ordinate is $G=\sum_{f\in\mathcal V}m_{f,O_f}$. It is the fraction of total attention probability reaching reviewed object regions in valid nonquery frames, without renormalizing excluded frames. The second is $P=|\mathcal V|^{-1}\sum_{f\in\mathcal V}p_{f,O_f}$, the arithmetic mean of within-frame object probability. Both are dimensionless probabilities. The first retains allocation across frames; the second gives equal weight to each reviewed frame.

For step $s$ and layer $\ell$, define $\mathcal T_{s,\ell}=\{f\in\mathcal V:\Delta_f^{\mathrm{baseline},s,\ell}\text{ and }\Delta_f^{\mathrm{modified},s,\ell}\text{ are finite}\}$. The third trend ordinate is $D^{c}_{s,\ell}=|\mathcal T_{s,\ell}|^{-1}\sum_{f\in\mathcal T_{s,\ell}}\Delta_f^{c,s,\ell}$, measured in conditional probability per spatial token. An empty paired set gives an undefined value. The full visualization pass recomputes this axis from paired per-frame records rather than separately dropping undefined values within each condition.

The difference panel shows $B_{s,\ell}=D^{\mathrm{modified}}_{s,\ell}-D^{\mathrm{baseline}}_{s,\ell}$. Positive $B$ means an increase in object preference relative to same-coordinate preference. It does not require the modified contrast itself to be positive. Baseline and modified metric panels share a centered color scale; the difference panel has its own centered scale. Metric matrices retain numeric color legends to make signed magnitudes readable; the attention overview panels have no colorbars. Frame-level triptychs show the corresponding unaveraged $\Delta_f$ and paired differences.

`comparison.json` records input paths, matched and unmatched map keys, source metadata, and interpretation limitations.

## Running

The program does not instantiate Wan or allocate GPU tensors. Torch CPU execution is limited to two threads. Run it in the project environment after both self-attention runs and their matching region audits finish:

```bash
conda activate video
python -m wan21_t2v_experiments.rope_posthoc_compare \
  --baseline /absolute/path/to/baseline_self_attention \
  --modified /absolute/path/to/modified_self_attention \
  --baseline-audit /absolute/path/to/baseline_region_audit \
  --modified-audit /absolute/path/to/modified_region_audit \
  --baseline-query-audit /absolute/path/to/original_baseline_query_audit \
  --modified-query-audit /absolute/path/to/original_modified_query_audit \
  --output /absolute/path/to/comparison \
  --plot-layer 27
```

The optional `--baseline-query-audit` and `--modified-query-audit` default to the corresponding key audit. Supply them explicitly if tuned key extraction changes the first-frame support.

Run the comparison separately for seeds 8, 20, and 29. Visual inspection of the input audit images remains necessary: matching metadata and geometry cannot establish that the extracted support is actually on the basketball.

## Full saved-data visualization pass

After the per-frame CSVs and reviewed masks exist, run:

```bash
bash scripts/rope_posthoc_visualize_wan2_1.sh
```

This invokes `python -m wan21_t2v_experiments.rope_posthoc_visualize` in the `video` environment. It renders 1,200 paired attention images per seed (20 steps × 30 layers × two normalizations), using four CPU workers and no GPU inference. Optional `--seeds`, `--layers`, `--step-end`, `--workers`, `--metrics-only`, and `--overviews-only` arguments support targeted regeneration. The root path is configured in the launcher. The earlier comparison command computes the source CSVs and a limited illustrative plot set; the full visualization pass supplies the current presentation and all-layer diagnostics.
