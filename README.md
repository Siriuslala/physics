<h1 align="center">Why Do Video Diffusion Models Violate Physics? Unveiling the Flaws in Attention Mechanisms</h1>

---
<p align="center">
  <b>Yueyan Li</b> · <b>Haibo Wang</b> · <b>Caixia Yuan</b> · <b>Xiaojie Wang</b>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/xxxx.xxxxx"><img src="https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg" alt="arXiv"></a>
</p>

<!-- <p align="center"><em>Under review at ICLR 2027</em></p> -->

This repository contains the code for our interpretability study of **motion planning** in text-to-video diffusion models. We analyze how Wan2.1-T2V forms object trajectories during early denoising, locate the attention-head circuits that drive this process, and show that a lightweight RoPE frequency scaling improves physical commonsense.

<p align="center">
  <img src="assets/teaser.png" width="100%"/>
</p>
<p align="center">
  <em>Figure 1. Same prompt, different seeds. Wan2.1-T2V-1.3B often produces mid-air bouncing, anti-gravity floating, or sudden freezing instead of a physically plausible bounce.</em>
</p>

---

## TL;DR

State-of-the-art video diffusion models look realistic, but they frequently violate basic physics. Existing fixes inject external simulators, rewrite prompts, or add specialized data. We instead look **inside** the model:

1. **Where** does motion planning happen? In the first few denoising steps, visible in object-token cross-attention.
2. **How** does it happen? A small subset of trajectory-forming heads writes motion semantics; self-attention then selects among competing candidate regions.
3. **Why** does it fail? 3D RoPE induces a **spatial anchoring** bias: early, physically wrong positions can suppress better candidates in nearby frames.
4. **What can we do?** Scale the height/width RoPE frequencies with $\lambda^{h/w}<1$ in early steps, with optional LoRA adaptation.

No extra physics simulator, no extra foundation-model teacher, and no change to the official Wan source. All analysis is applied by runtime monkey patches.

---

## Findings

### Motion planning emerges in the first 5 denoising steps

T2V latents are too noisy to decode early on, so we track **video-to-object-token cross-attention**. In 50-step sampling, object locations go from noise to multiple candidate regions to a deterministic trajectory around step 5.

<p align="center">
  <img src="assets/cross_attention.png" width="100%"/>
</p>
<p align="center">
  <em>Head-averaged cross-attention in layer 27. The bounce trajectory is already visible by step 7.</em>
</p>

### A clear trajectory pattern is not enough

We score each cross-attention head by **convergence speed** (how fast its map locks onto the final trajectory) and **causal contribution** (attribution patching on the flow-matching velocity in the object region). Zero-ablating heads then reveals four types:

| Type | Trajectory pattern | Contribution | Ablation effect |
| --- | --- | --- | --- |
| (1)(2) | Weak / chaotic | Can be large | Appearance / background, not motion |
| (3) | Clear | High | Trajectory collapses |
| (4) | Clear | Near zero | Trajectory almost unchanged |

A visible trajectory is a **sufficient but not necessary** condition for a motion-planning head.

<p align="center">
  <img src="assets/head_ablation.png" width="100%"/>
</p>
<p align="center">
  <em>Zero-ablation of the four head types. Only Type (3) heads are necessary for a coherent bounce.</em>
</p>

### Self-attention fails through RoPE spatial anchoring

Cross-attention injects *what* to generate. Self-attention decides *where* the object should be in each frame. Early on, every frame contains several **candidate regions**. Self-attention then votes among them via mutual consistency.

<p align="center">
  <img src="assets/self_attention.png" width="100%"/>
</p>
<p align="center">
  <em>A query region in frame 0 attends to the same spatial coordinate in other frames (RoPE spatial anchoring).</em>
</p>
<p align="center">
  <img src="assets/candidates.png" width="100%"/>
</p>
<p align="center">
  <em>Multiple candidate regions extracted from early cross-attention.</em>
</p>

Because 3D RoPE decays sharply along height and width, a query prefers spatially nearby keys across frames. If a few frames lock into a physically wrong location first, neighboring frames are pulled toward that same coordinate. Reasonable but more distant candidates lose the competition, producing the failures in Figure 1.

### A one-factor RoPE fix

We keep temporal RoPE unchanged and scale only the spatial axes:

$$
f^{h}(q,p)=q^{h}e^{i p^{h}\lambda^{h}\theta},\qquad
f^{w}(q,p)=q^{w}e^{i p^{w}\lambda^{w}\theta},\qquad \lambda^{h/w}<1.
$$

Smaller $\lambda^{h/w}$ slows spatial attention decay, so early denoising can explore more candidate regions. Training-free inference applies the scale on the first 5 steps. Training-based fine-tuning combines the same scale with LoRA on attention modules and an early-step timestep sampler.

---

## Results

On **VideoPhy** (344 cases; Semantic Adherence / Physical Commonsense, human evaluation):

<p align="center">
  <img src="assets/videophy_table.png" width="100%"/>
</p>

The gain is largest on **solid-\*** interactions, which is the regime our analysis targets. Prompt refinement mainly helps instruction following; combining it with modified RoPE further boosts physical consistency.

<p align="center">
  <img src="assets/qualitative_basketball.png" width="100%"/>
</p>
<p align="center">
  <em>Basketball free-fall. Top: Wan2.1-T2V-1.3B. Bottom: LoRA + modified RoPE ($\lambda^{h}=\lambda^{w}=0.70$ on the first 5 steps).</em>
</p>

<p align="center">
  <img src="assets/qualitative_cork.png" width="90%"/>
</p>

Training only attention LoRA (not FFN) preserves object appearance while improving motion, which is consistent with the interpretability result that physical failures live in attention rather than in the feed-forward layers.

---

## Repository Structure

```text
.
├── wan21_t2v_experiments/   # interpretability toolkit (monkey patches)
│   ├── docs/                # per-experiment notes
│   └── run_wan21_t2v_experiments.py
├── scripts/                 # launchers for each experiment
├── wan21_train/             # DiffSynth training utilities + RoPE-lambda LoRA
├── wan_eval/                # batch inference on VideoPhy
├── projects/Wan2_1/         # official Wan2.1 inference code (unmodified)
└── DiffSynth-Studio/        # training backend (unmodified)
```

Analysis never edits `projects/Wan2_1`. Every intervention is a runtime patch.

---

## Setup

```bash
git clone https://github.com/Siriuslala/physics.git
cd physics

conda create -n video python=3.10 -y
conda activate video
pip install -r projects/Wan2_1/requirements.txt
```

Download [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) (or the 14B checkpoint) and copy the environment template:

```bash
cp scripts/.env_example scripts/env.sh
```

Set `ROOT_DIR`, checkpoint directory, and output directory in `scripts/env.sh`. The launchers in `scripts/` source this file.

---

## Interpretability Toolkit

Launchers live in `scripts/`. Edit the corresponding `.sh` file, then run it:

```bash
bash scripts/<experiment>_wan2_1.sh
```

A typical analysis path follows the paper:

| Paper section | Script | What it shows |
| --- | --- | --- |
| 4.1 Cross-attention evolution | `scripts/cross_attention_token_viz_wan2_1.sh`, `scripts/head_evolution_wan2_1.sh` | Trajectory appears in the first 5 steps; entropy / support quality |
| 4.2 Motion-planning heads | `scripts/cross_attn_head_ablation_wan2_1.sh`, `scripts/trajectory_consensus_dynamics_wan2_1.sh` | Convergence speed vs. causal contribution; zero ablation |
| 5 Self-attention / candidates | `scripts/self_attention_viz_wan2_1.sh`, `scripts/trajectory_consensus_dynamics_wan2_1.sh` | Candidate-region competition and mutual consistency |
| 5–6 RoPE spatial decay | `scripts/rope_decay_curve_wan2_1.sh`, `scripts/rope_ablation_wan2_1.sh` | Height/width decay vs. temporal decay |

Per-experiment math, flags, and outputs are documented in [`wan21_t2v_experiments/docs/`](wan21_t2v_experiments/docs) and summarized in [`wan21_t2v_experiments/README.md`](wan21_t2v_experiments/README.md).

---

## Training-based RoPE Modification

Training uses DiffSynth-Studio on a filtered subset of [WISA-80K](https://wisav1.github.io/WISA/) (~48k physics videos after duration / motion-score filtering). The recommended recipe is:

- LoRA on self- and cross-attention (`r=64`, `alpha=32`), not FFN
- fixed spatial RoPE scale $\lambda^{h/w}=0.75$ during training
- mixed timestep sampler with $p_{\mathrm{early}}=0.9$ on the first 10% of denoising steps
- at test time, apply $\lambda^{h/w}=0.70$ on the first 5 steps

Prepare metadata:

```bash
python wan21_train/prepare_wisa80k_for_diffsynth.py \
  --extract \
  --build-metadata \
  --metadata-mode categories \
  --merge-category-metadata /path/to/physics_metadata \
  --reflection-sample-n 4000
```

Launch LoRA + fixed-lambda training:

```bash
bash wan21_train/scripts/train_wan21_t2v_1b3_fixed_lambda_lora.sh
```

Details: [`wan21_train/README.md`](wan21_train/README.md) and [`wan21_train/docs/wan21_spatial_rope_lambda_training.md`](wan21_train/docs/wan21_spatial_rope_lambda_training.md).

---

## Evaluation

Edit the settings in `wan_eval/scripts/infer_eval_2.1.sh`, then run:

```bash
bash wan_eval/scripts/infer_eval_2.1.sh
```

Prompt files are under `wan_eval/datasets/`. See [`wan_eval/README.md`](wan_eval/README.md).

---

## Citation

```bibtex
@inproceedings{video-physics-attention-2027,
  title     = {Why Do Video Diffusion Models Violate Physics?
               Unveiling the Flaws in Attention Mechanisms},
  author    = {Yueyan Li and Haibo Wang and Caixia Yuan and Xiaojie Wang},
  booktitle = {International Conference on Learning Representations (under review)},
  year      = {2027}
}
```

---

## Acknowledgements

This project builds on [Wan2.1](https://github.com/Wan-Video/Wan2.1), [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), [WISA](https://wisav1.github.io/WISA/), and [VideoPhy](https://github.com/Hritikbansal/videophy).
