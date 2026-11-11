# Basketball attention comparison

## Purpose and scope

This workflow compares the original Wan2.1-T2V-1.3B model with a specified LoRA checkpoint and an early spatial RoPE intervention. It collects cross-attention for object localization and self-attention from the first video frame's object region. This is a joint weights-and-RoPE intervention; isolating the causal contribution of RoPE requires an additional comparison at fixed LoRA weights.

## Fixed generation settings

The prompt is `Against a pure white background, a basketball falls vertically from mid-air onto a wooden floor and bounces up several times.` Generation uses 832 × 480 pixels, 81 video frames, 50 UniPC steps, shift 5.0, guidance 5.0, and the original negative prompt. All sampling runs require an NVIDIA A800. The launcher checks the device name and at least 40000 MiB of free memory before starting. It uses model offloading and records device information in `gpu_provenance.csv`.

| Seed | LoRA modules | Checkpoint | Early probability in training run | Inference spatial lambda | Reference cross-attention step | Reference layer |
|---|---|---|---|---|---|---|
| 8 | attn | step-1000.safetensors | 0.9 | 0.75 | 45 | 27 |
| 20 | attn | step-1000.safetensors | 0.9 | 0.75 | 50 | 27 |
| 29 | attn_ffn | step-700.safetensors | 0.5 | 0.70 | 50 | 27 |

All three adapters use rank 64 and alpha 32, from the batch-size-32, learning-rate-1e-4, 1500-step training directories explicitly encoded in the launcher. Both spatial lambdas act on one-based denoising steps 1–5, in both classifier-free guidance branches. The temporal lambda is 1. The adapter remains active throughout denoising. The implementation follows the principal-complex-phase convention of `wan_eval/infer_wan_t2v.py`; it does not silently substitute unwrapped frequency scaling.

## Collection and audit

The launcher is `scripts/rope_posthoc_attention_wan2_1.sh`. Its positional arguments are stage (`cross`, `audit`, or `self`), seed, and condition (`baseline` or `modified`). Existing baseline cross-attention is reused without overwriting it. New outputs live under `outputs_wan_2_1_t2v-1.3B/rope_posthoc_attention/<prompt_tag>/<condition>/seed_<seed>`.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/rope_posthoc_attention_wan2_1.sh cross 8 modified
bash scripts/rope_posthoc_attention_wan2_1.sh audit 8 modified
CUDA_VISIBLE_DEVICES=0 bash scripts/rope_posthoc_attention_wan2_1.sh self 8 modified
```

Cross-attention collection stores all 50 steps, all layers, and all heads for `basketball`, `falls`, `bounces`, and `up`, plus the video. Bulk cross-attention PDF rendering is disabled; the raw tensors and reference audit sheets remain available. Self-attention stores all 50 steps and all layers, uses the conditional branch, and averages all object-region query tokens from video frame 1 without subsampling. The standard per-head and head-mean PDFs show ten key frames with a shared color scale within each PDF. Cross-condition comparison figures additionally share their color scale between conditions.

The audit utility `python -m wan21_t2v_experiments.region_audit` calls the production region extractor and overlays its exact masks on both cross-attention and generated video. `REGION_QUANTILE` and `REGION_RADIUS_ALPHA` configure the launcher consistently for audit and sampling; defaults are 0.8 and 1.5. The query mask must be visually checked before sampling. Key-region metrics require a separate explicit per-frame visual validity assessment, since a correctly localized first-frame query does not establish that all later masks are correct. Ground highlights, reflections, and objects leaving the image are particular failure cases. Do not treat those masks as valid object regions merely because extraction returned a result.

## Comparison

Use `rope_posthoc_compare.py` after collecting both conditions and reviewing the audit sheets. Its exact metrics, independent query/key audits, valid-frame intersection, area normalization, and overlap handling are documented in [rope_posthoc_comparison.md](rope_posthoc_comparison.md). Different generated trajectories and different query supports remain interpretation limitations even when invalid object masks are excluded.
