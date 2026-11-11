# Reference region audit

The CPU-only `region_audit` utility checks the object supports used by `self_attention_viz` against both cross-attention maps and the generated video. It calls the production support builder directly and does not modify the model or inference path. The reference step and layer must match the subsequent self-attention run.

Let $A\in\mathbb{R}^{F\times H\times W}$ denote the selected head-averaged basketball cross-attention map, where $F$ is the latent frame count and $H,W$ are spatial token-grid dimensions. The production extractor preprocesses each frame, selects the eight-connected threshold component containing its maximum, and derives a center $c_f\in\mathbb{R}^2$ for latent frame $f$. The output support $M\in\{0,1\}^{F\times H\times W}$ is a disk around that center, with the configured fixed or adaptive radius. The full extraction equations and parameter definitions are maintained in `self_attention_viz.md` and `head_evolution.md`.

The audit displays the exact disk boundary in red and its center in cyan. A video panel uses zero-based video frame $4f$ as a temporal reference for latent frame $f$; the temporal receptive field of the VAE prevents interpreting this correspondence as exact segmentation. Cross-attention panels use independent color scales, so color intensity is not comparable between panels.

Run from the repository root after activating the `video` environment:

```bash
python -m wan21_t2v_experiments.region_audit \
  --cross-dir /absolute/path/to/cross_attention_run \
  --output-dir /absolute/path/to/region_audit \
  --reference-step 45 --reference-layer 27
```

`--center-quantile` controls the component threshold, `--center-mode` selects geometric center, weighted centroid, or peak, and the support-radius options reproduce the production builder. Defaults match `self_attention_viz`. `--video-path` explicitly chooses the video; otherwise the first MP4 in the cross-attention directory is used.

Outputs include `cross_attention_regions.png`, `video_regions.png` when a video is available, `region_audit.json` with all extraction parameters and per-frame coordinates, and `reference_support.pt` with the original and preprocessed maps and the exact boolean support tensor. JSON frame indices are zero-based for latent frames and one-based for video frames. A human visual review is still necessary: the tool does not automatically certify that a bright region is the basketball. Ground highlights, reflections, and objects leaving the image can cause the peak-based extractor to select the wrong region. Failed or uncertain key-frame supports must be excluded from object-alignment metrics, even when the first-frame query is correct.
