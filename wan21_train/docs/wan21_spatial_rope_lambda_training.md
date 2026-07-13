# Wan2.1 Spatial RoPE Lambda Training

## 1. Goal

This document specifies the first training-based RoPE modification method for Wan2.1 T2V fine-tuning. The method adds learnable spatial RoPE scale parameters to Wan self-attention so that the model can reduce the excessive same-coordinate spatial prior during early denoising.

The target failure mode is an early self-attention pattern: a query token from an object region in frame `f` often attends strongly to the same spatial coordinate in other frames, even when the object has moved. This is consistent with the height and width components of 3D RoPE giving a logit advantage to small spatial phase differences. In high-noise early denoising, object semantics in hidden states are still weak, so this spatial prior can dominate trajectory candidate selection.

The method changes only self-attention RoPE. It does not modify cross-attention RoPE because Wan cross-attention does not use RoPE.

## 2. Wan Self-Attention RoPE

In DiffSynth-Studio, the relevant Wan implementation is:

- `DiffSynth-Studio/diffsynth/models/wan_video_dit.py`
- `SelfAttention.forward`
- `rope_apply`
- `WanModel.forward`

Let $q_i,k_j\in\mathbb{R}^{d}$ be the pre-RoPE query and key for video tokens `i` and `j` in one self-attention head. Each token has latent grid coordinate $p_i=(f_i,h_i,w_i)$. Wan splits the per-head dimension into frame, height, and width RoPE channel groups. For Wan2.1 T2V 1.3B, $d=128$ and $(d_f,d_h,d_w)=(44,42,42)$.

For axis $a\in\{f,h,w\}$, the original RoPE phase is $\phi_{i,r}^{(a)}=p_i^{(a)}\theta_{a,r}$, where $r$ is the complex-pair index. The modified phase is $\phi_{i,r}^{(a)}=\lambda_a p_i^{(a)}\theta_{a,r}$.

Smaller $\lambda_a$ means slower phase growth and weaker effective distance decay on axis $a$. Larger $\lambda_a$ means faster phase growth and a stronger locality prior.

## 3. Core Method

The primary method is spatial-only RoPE scaling: $\lambda_f=1$, $\lambda_h=\exp z_h$, and $\lambda_w=\exp z_w$.

The trainable variables are log-scales $z_h$ and $z_w$, not the raw $\lambda$ values. This guarantees positive scales while allowing unconstrained optimization in log space.

The identity initialization is $z_h=0,z_w=0$, hence $\lambda_h=\lambda_w=1$. At initialization, the modified model is exactly the original Wan model.

Keeping $\lambda_f=1$ is the recommended default because the current hypothesis is about an excessive spatial same-coordinate prior, not an obviously excessive temporal RoPE penalty. Learning $\lambda_f$ should be a later ablation.

## 4. Lambda Scope

The spatial log-scale tensor $z$ can be shared at different resolutions.

Model-wise sharing uses $z\in\mathbb{R}^{2}$. The two entries correspond to height and width. This is the cleanest smoke test.

Layer-wise sharing uses $z\in\mathbb{R}^{L\times 2}$. For Wan2.1 T2V 1.3B, $L=30$, so this adds 60 base parameters when $\lambda_f$ is fixed. Layer-wise is the recommended default for the main method because it is still tiny but allows different DiT layers to learn different spatial RoPE biases.

Head-wise sharing uses $z\in\mathbb{R}^{L\times H\times 2}$. For Wan2.1 T2V 1.3B, $L=30$ and $H=12$, so this adds 720 base parameters when $\lambda_f$ is fixed. Head-wise is useful after layer-wise experiments show a positive signal.

The recommended experiment order is `model-wise -> layer-wise -> head-wise`. The recommended default is `layer-wise` with `learn_lambda_f=false`.

## 5. Timestep-Conditioned Lambda

### 5.1 Definition

Let $\tau\in[0,1000]$ be the raw Wan timestep value passed into the DiT, where larger $\tau$ means higher noise and earlier denoising. Let $e_\tau\in\mathbb{R}^{d_t}$ be the sinusoidal timestep embedding.

The timestep-conditioned log-scale is $z(\tau)=\mu+g_\psi(e_\tau)$, and the applied scale is $\lambda(\tau)=\exp z(\tau)$.

Here $\mu$ is the trainable base log-scale tensor, and $g_\psi$ is a trainable MLP with parameters $\psi$. The lambda module parameters are $\theta_\lambda=\{\mu,\psi\}$ when timestep conditioning is enabled, and $\theta_\lambda=\{\mu\}$ when it is disabled.

### 5.2 Structure of $g_\psi$

The recommended $g_\psi$ is one shared MLP that maps the timestep embedding to the whole log-scale correction vector: $g_\psi(e)=W_2\,\mathrm{SiLU}(W_1 e+b_1)+b_2$.

The output dimension equals the number of trainable lambda log-scale values:

- model-wise spatial-only: output dimension 2,
- layer-wise spatial-only: output dimension $L\times 2$,
- head-wise spatial-only: output dimension $L\times H\times 2$.

Thus, $g_\psi$ is shared in the sense that one MLP produces the entire vector of timestep-dependent corrections. It is not necessary to create one MLP per layer or per head. A per-layer or per-head independent MLP would add redundant parameters and make the method harder to interpret.

The last linear layer should be zero-initialized, $W_2=0,b_2=0$. Together with $\mu=0$, this gives $z(\tau)=0$ and $\lambda(\tau)=1$ for every timestep at initialization.

### 5.3 Whether $g_\psi$ Is Trained

Yes. If timestep conditioning is enabled, $g_\psi$ must be trained in all three training modes:

- mode 1 trains $\theta_\lambda=\{\mu,\psi\}$ only,
- mode 2 trains $\theta_\lambda=\{\mu,\psi\}$ and LoRA adapters,
- mode 3 trains $\theta_\lambda=\{\mu,\psi\}$ and the full DiT.

Freezing $g_\psi$ while claiming timestep-conditioned lambda would make the timestep conditioning ineffective unless a checkpointed $g_\psi$ is loaded.

## 6. Training Modes

Mode 1 is lambda-only training. The original Wan DiT is frozen, no LoRA is inserted, and the only trainable parameters are $\theta_\lambda$. If `timestep_conditioned=false`, this means only $\mu$. If `timestep_conditioned=true`, this means both $\mu$ and $g_\psi$.

Mode 2 is lambda plus LoRA training. The base Wan DiT weights are frozen, LoRA adapters are inserted into selected DiT linear layers, and $\theta_\lambda$ is trained together with the LoRA parameters. If timestep conditioning is enabled, $g_\psi$ is trained.

Mode 3 is lambda plus full DiT fine-tuning. The full Wan DiT is trainable, and $\theta_\lambda$ is trained together with all DiT parameters. If timestep conditioning is enabled, $g_\psi$ is trained.

The recommended optimizer implementation is to use separate parameter groups: one group for $\theta_\lambda$, one group for LoRA parameters in mode 2, and one group for DiT parameters in mode 3. This allows the lambda learning rate to be larger than the LoRA or full-DiT learning rate.

## 7. Rectified Flow, Sigma, and Shift

### 7.1 From $x_t=(1-t)x_0+t x_1$ to DiffSynth Notation

In rectified flow or flow matching, a noisy latent can be written as $x_t=(1-t)x_0+t\epsilon$, where $x_0$ is the clean video latent, $\epsilon\sim\mathcal{N}(0,I)$ is noise, and $t\in[0,1]$ is the noise level. The target velocity is $v^\star=\epsilon-x_0$.

DiffSynth uses the name $\sigma$ for this noise level. In the Wan SFT loss, the noised latent is $x_\sigma=(1-\sigma)x_0+\sigma\epsilon$, and the target is $v^\star=\epsilon-x_0$.

So for this training code, $\sigma$ plays the same conceptual role as the $t$ in $x_t=(1-t)x_0+t\epsilon$. Large $\sigma$ is high noise and early denoising. Small $\sigma$ is low noise and late denoising.

### 7.2 Scheduler Index and Denoising Index

The scheduler discretizes the continuous interval into a list $\{\sigma_i\}_{i=0}^{N-1}$, ordered from high noise to low noise. The integer $i$ is the scheduler index. During inference, the loop index over denoising steps is often called the denoising index or progress index. Both refer to the position in the high-to-low noise denoising sequence.

For training, DiffSynth calls `set_timesteps(1000, training=True)`, so $N=1000$. For 50-step inference, $N=50$. In both cases, index $i=0$ is the beginning of denoising and has the largest $\sigma_i$.

### 7.3 Wan Shift

Wan first builds a linearly spaced base noise level $u_i\in[1,0]$. With default denoising strength 1, this is approximately $u_i=1-i/N$. Then it applies the shift map $\sigma_i=S_s(u_i)=\frac{s u_i}{1+(s-1)u_i}$, where $s=5$ by default.

For $s>1$, $S_s(u)>u$ for $0<u<1$. Therefore, shift moves each interior noise level toward the high-noise side. For example, with $s=5$, $S_5(0.5)=\frac{2.5}{3}=0.8333$. The midpoint scheduler index no longer corresponds to $\sigma=0.5$; it corresponds to a much noisier latent.

The derivative is $S_s'(u)=\frac{s}{(1+(s-1)u)^2}$. Near $u=1$, the derivative is $1/s$, so the high-noise end is compressed in $\sigma$ value. Near $u=0$, the derivative is $s$, so the low-noise end is expanded in $\sigma$ value.

This is what "shift changes sigma values" means: it changes the actual noise level $\sigma_i$ assigned to a fixed scheduler index $i$.

### 7.4 Why Shift Does Not Replace Timestep Boundary Sampling

DiffSynth samples a scheduler index, not a continuous $\sigma$, inside `FlowMatchSFTLoss`. The code computes $i_{\min}=\lfloor b_{\min}N\rfloor$ and $i_{\max}=\lfloor b_{\max}N\rfloor$, then samples $i\sim\mathrm{Uniform}\{i_{\min},\ldots,i_{\max}-1\}$.

Changing $s$ changes $\sigma_i$ for every index, but it does not change the number of sampled indices in the early denoising region. If $[b_{\min},b_{\max}]=[0,1]$, every scheduler index is still sampled with equal probability. A larger shift makes many of those indices noisier, but it does not specifically sample the first 4 or 5 denoising indices more often.

Therefore, for motion-planning-focused training, the clean control knob is the boundary interval or a custom mixture sampler over scheduler indices. Shift should usually stay at the Wan default $s=5$, unless inference will also use a different shift.

### 7.5 Scheduler-Dependent Weight

After the timestep is sampled, DiffSynth multiplies the MSE loss by a scheduler-dependent scalar $w_i$. In `FlowMatchScheduler.set_training_weight`, the unnormalized weight is a function of the raw timestep $\tau_i=1000\sigma_i$, roughly highest around middle timesteps and lower near the endpoints.

The training loss is therefore $L_{\mathrm{flow}}=w_i\,\|\hat v_\theta(x_{\sigma_i},\tau_i,c)-(\epsilon-x_0)\|_2^2$.

This matters because the earliest high-noise timesteps can receive smaller weights than middle timesteps. If mode 1 samples only the early region, the early interval should not be too narrow.

## 8. `min_timestep_boundary` and `max_timestep_boundary`

### 8.1 Meaning

Let $b_{\min}$ be `min_timestep_boundary` and $b_{\max}$ be `max_timestep_boundary`. DiffSynth samples $i\sim\mathrm{Uniform}\{\lfloor b_{\min}N\rfloor,\ldots,\lfloor b_{\max}N\rfloor-1\}$.

Since $i=0$ is high noise, $[0,0.1]$ means the first 10 percent of denoising indices. Since large $i$ is low noise, $[0.8,1.0]$ means the final 20 percent of denoising indices.

### 8.2 About the "Default" Boundary

The current DiffSynth Wan parser defaults are `min_timestep_boundary=0.0` and `max_timestep_boundary=1.0`. A setting such as `[0.8,1.0]` is not the default in `examples/wanvideo/model_training/train.py`.

The boundary-split examples in DiffSynth are mostly for Wan2.2 mixture models, where high-noise and low-noise models are trained or fine-tuned separately. For example, some Wan2.2 scripts use a high-noise interval such as `[0,0.417]` and a low-noise interval such as `[0.417,1]`. Those settings are model-family specific and should not be treated as the default recipe for Wan2.1 T2V 1.3B.

If a local script uses `[0.8,1.0]`, it is a late-denoising fine-tune. It is appropriate for detail/style refinement, but not for the motion-planning question.

### 8.3 Fair Comparison

There are two different comparison goals.

For an architecture ablation, the ordinary fine-tuning baseline and the lambda method should use the same timestep sampler. Otherwise, an improvement could be caused by the timestep distribution rather than the lambda modification.

For a practical best-recipe comparison, each method can use its own best timestep sampler, but the result should be reported as a combined method and training-recipe improvement.

The recommended evaluation is to report both:

1. Same-sampler comparison: baseline LoRA/full fine-tuning and lambda LoRA/full fine-tuning use the same timestep distribution.
2. Best-recipe comparison: ordinary baseline uses full-range sampling, while lambda training uses an early-biased mixture sampler.

This makes the causal claim and the practical performance claim both clean.

### 8.4 Recommended Mixture Sampler

A safer alternative to training only early timesteps is a mixture sampler. Let $\rho=0.12$ be the early interval boundary. Sample a Bernoulli variable $B\sim\mathrm{Bernoulli}(p_{\mathrm{early}})$. If $B=1$, sample $i\sim\mathrm{Uniform}\{0,\ldots,\lfloor\rho N\rfloor-1\}$. If $B=0$, sample $i\sim\mathrm{Uniform}\{\lfloor\rho N\rfloor,\ldots,N-1\}$.

This gives explicit early supervision while preserving middle and late denoising ability.

Recommended values:

- mode 1 lambda-only: use early-only `[0.0,0.12]` first, then validate a mixture if needed,
- mode 2 lambda plus LoRA: use $p_{\mathrm{early}}\in[0.5,0.7]$,
- mode 3 lambda plus full fine-tuning: use $p_{\mathrm{early}}\in[0.3,0.5]$ or full-range sampling.

Until the custom sampler is implemented, mode 2 should use full range for the most conservative comparison, or `[0.0,0.35]` for a more motion-focused run. Mode 3 should use full range unless there is a clear reason to bias early.

## 9. Lambda Parameterization and Regularization

The default parameterization is unconstrained: $\lambda_h=\exp z_h$ and $\lambda_w=\exp z_w$. This preserves the original training behavior and allows learned spatial RoPE scales to move above or below one.

For the constrained ablation, the implementation supports two scalable parameterizations that enforce $\lambda_h\leq 1$ and $\lambda_w\leq 1$ for every layer, head, and timestep. Both forms learn an unconstrained raw variable $r_a(\tau)$ and map it to a valid RoPE scale, where $a\in\{h,w\}$.

The strict softplus form is $\lambda_a(\tau)=\exp(-\mathrm{softplus}(r_a(\tau)))$. It guarantees $0<\lambda_a<1$. However, near the identity point $\lambda_a\approx 1$, this form requires a large negative raw value, so $\sigma(r_a)$ is very small and the effective gradient on $\lambda_a$ can be weak. This form is mathematically clean but can be slow to move when initialized extremely close to one.

The bounded form is $\lambda_a(\tau)=\lambda_{\min}+(1-\lambda_{\min})\sigma(r_a(\tau))$, where $0\leq\lambda_{\min}<1$. It guarantees $\lambda_{\min}<\lambda_a<1$. The lower bound is configured by `wan_spatial_rope_lambda_min`. This form is the recommended constrained parameterization because it avoids the strongest near-one saturation of the softplus form while still forbidding spatial RoPE scales above one.

For bounded initialization, choose a near-identity target $\lambda_0=1-\epsilon$. The raw initial value is $r_0=\mathrm{logit}((\lambda_0-\lambda_{\min})/(1-\lambda_{\min}))$. The implementation uses `wan_spatial_rope_lambda_init_eps` for $\epsilon$ and initializes every spatial lambda value to $\lambda_0$. This requires $\lambda_{\min}<1-\epsilon$.

Exact identity initialization is impossible with finite raw parameters under a strict upper-bound parameterization. The implementation therefore uses a near-identity initialization $\lambda_a=1-\epsilon$. The default `wan_spatial_rope_lambda_init_eps=1e-4` is almost identical to the original Wan RoPE, but it may be too conservative for constrained training. For practical bounded runs, `wan_spatial_rope_lambda_init_eps=1e-2` is a stronger default because it starts from $\lambda_a=0.99$ and gives a larger effective gradient.

The script-level parameters are:

```text
LAMBDA_PARAMETRIZATION=bounded_leq_one
LAMBDA_MIN=0.5
LAMBDA_INIT_EPS=1e-2
```

The corresponding CLI arguments are:

```text
--wan_spatial_rope_lambda_parametrization bounded_leq_one
--wan_spatial_rope_lambda_min 0.5
--wan_spatial_rope_lambda_init_eps 1e-2
```

The original Wan model corresponds to $\log\lambda=0$. The primary regularizer is an L2 penalty on the effective log-scale: $L_\lambda=\beta_\lambda\,\mathbb{E}_{\tau}[\|\log\lambda(\tau)\|_2^2/M]$, where $M$ is the number of spatial lambda values. In the unconstrained form, $\log\lambda(\tau)=z(\tau)$. In constrained forms, $\log\lambda(\tau)$ is computed after the raw parameter $r(\tau)$ is transformed into a valid lambda value.

If timestep conditioning is disabled, the effective raw parameter is the base tensor only. If timestep conditioning is enabled, the effective raw parameter is the base tensor plus $g_\psi(e_\tau)$, and both the base tensor and $g_\psi$ are trainable in mode 1, mode 2, and mode 3.

The total objective used by the current implementation is $L=L_{\mathrm{flow}}+L_\lambda$. Recommended initial values are `beta_lambda=1e-4` for lambda-only runs and `0` to `1e-4` for lambda plus LoRA runs, depending on whether the experiment is testing unconstrained movement or constrained shrinkage.

A fixed manual lambda mode is also supported for LoRA adaptation experiments. In this mode, $\lambda_h$ and $\lambda_w$ are target constants provided by `wan_spatial_rope_lambda_fixed_h` and `wan_spatial_rope_lambda_fixed_w`; the lambda module has no trainable parameters, and only LoRA adapts to the manually modified RoPE geometry.

Fixed manual lambda can optionally use a training-step schedule. Let $\lambda_a^*$ be the fixed target for axis $a\in\{h,w\}$ and let $\alpha(s)\in[0,1]$ be the schedule progress at optimizer update step $s$. The applied scale is $\lambda_a(s)=1+\alpha(s)(\lambda_a^*-1)$. `constant` uses $\alpha(s)=1$ from the first step. `linear` uses $\alpha(s)=\min(s/S,1)$. `cosine` uses $\alpha(s)=0.5-0.5\cos(\pi\min(s/S,1))$. Here $S$ is configured by `wan_spatial_rope_lambda_fixed_schedule_steps`; if it is zero, the total number of training update steps is used.

The recommended fixed-lambda LoRA setting is `wan_spatial_rope_lambda_fixed_schedule=cosine`, because it starts from the original RoPE geometry, moves smoothly toward the target, and lets LoRA adapt before the full manual RoPE bias is applied.

Fixed manual lambda also supports a timestep-activation switch. `wan_spatial_rope_lambda_global=true` applies the fixed lambda on every sampled training timestep. `wan_spatial_rope_lambda_global=false` applies the fixed lambda only when the sampled training timestep index is inside the early interval defined by `timestep_mixture_early_boundary`; other sampled timesteps use the original RoPE with $\lambda_h=\lambda_w=1$. This keeps the structural bias focused on early motion-planning timesteps while leaving middle and late detail-refinement training closer to the original model.

## 10. LoRA and Full Fine-Tuning Recommendations

### 10.1 LoRA Targets

The first LoRA experiment should use attention-only targets:

```text
lora_target_modules = q,k,v,o
```

In the current DiffSynth naming scheme, this patches both self-attention and cross-attention Q/K/V/O. Self-attention is directly responsible for the RoPE-modified video-token routing. Cross-attention is useful because physical motion still depends on grounding object and event words from the text condition.

FFN LoRA should be a second ablation:

```text
lora_target_modules = q,k,v,o,ffn.0,ffn.2
```

FFN can help because the dataset includes diverse dynamics, thermodynamics, and optics events, but it makes the mechanism comparison less isolated.

### 10.2 Learning Rates

DiffSynth's parser default learning rate is `1e-4`, and the current `wan21_train` LoRA script also uses `1e-4`. Therefore, the default LoRA learning rate for mode 2 should be `1e-4`, not `1e-5`.

Recommended mode 2 learning rates:

```text
lambda_lr = 1e-3 for lambda-only warmup or 5e-4 when trained with LoRA
lora_lr = 1e-4 by default
lora_lr sweep = 5e-5, 1e-4, 2e-4
```

Use the lower end if adding FFN LoRA or if validation quality becomes unstable.

Recommended mode 3 learning rates:

```text
lambda_lr = 1e-4 to 5e-4
dit_lr = 1e-6 to 1e-5
```

Full DiT fine-tuning should use a much smaller learning rate than LoRA.

## 11. Recommended Experiment Order

The recommended order is:

1. Mode 1, model-wise, no timestep conditioning.
2. Mode 1, layer-wise, timestep-conditioned.
3. Baseline LoRA with the same timestep sampler as mode 2.
4. Mode 2, layer-wise lambda plus attention-only LoRA.
5. Mode 2, layer-wise lambda plus attention-and-FFN LoRA.
6. Mode 2, head-wise lambda plus attention-only LoRA.
7. Baseline full fine-tuning with the same timestep sampler as mode 3.
8. Mode 3, layer-wise lambda plus full DiT fine-tuning.

The most important causal comparison is:

```text
same timestep sampler + same trainable backbone capacity,
without lambda vs with lambda.
```

The most important practical comparison is:

```text
best ordinary fine-tuning recipe vs best lambda fine-tuning recipe.
```

## 12. Default Configurations

The first serious mode 1 run should use:

```text
training_mode = lambda_only
lambda_scope = layer-wise
learn_lambda_f = false
timestep_conditioned = true
timestep_conditioned_hidden_dim = 128
lambda_lr = 1e-3
beta_lambda = 1e-4
beta_smooth = 0
shift = 5
min_timestep_boundary = 0.0
max_timestep_boundary = 0.12
```

The first mode 2 run should use:

```text
training_mode = lambda_lora
lambda_scope = layer-wise
learn_lambda_f = false
timestep_conditioned = true
lambda_lr = 5e-4
lora_lr = 1e-4
lora_rank = 32
lora_alpha = 32
lora_target_modules = q,k,v,o
beta_lambda = 1e-4
shift = 5
timestep_sampling = mixture if implemented, otherwise full range
```

The first mode 3 run should use:

```text
training_mode = lambda_full
lambda_scope = layer-wise
learn_lambda_f = false
timestep_conditioned = true
lambda_lr = 1e-4 to 5e-4
dit_lr = 1e-6 to 1e-5
beta_lambda = 1e-4
shift = 5
min_timestep_boundary = 0.0
max_timestep_boundary = 1.0
```

## 13. Checkpoint Contents

The training logger saves the normal DiffSynth checkpoint and additionally saves lambda-specific artifacts when a spatial RoPE lambda module is attached:

```text
step-N.safetensors
step-N_lambda.safetensors, if trainable lambda parameters are present
lambda_heatmaps/step-N_lambda_heatmaps.pt
lambda_heatmaps/step-N_lambda_heatmaps.json
lambda_heatmaps/step-N_lambda_t{0,50,100,500,900}_{h,w}.png
```

The `.pt` heatmap payload stores the lambda scope, parameterization, lambda bounds or fixed values, and the layer-by-head lambda tensors. The `.png` files visualize per-layer or per-head $\lambda_h$ and $\lambda_w$ at fixed timesteps. The same export runs for learnable lambda and fixed manual lambda, so checkpoints can be compared by their applied RoPE geometry even when fixed lambda has no trainable state.

## 14. Validation

Validation should include both generation quality and mechanism diagnostics:

- fixed prompt and seed grid for falling/bouncing, collision, rigid body motion, liquid motion, gas motion, and deformation,
- trajectory stability across seeds,
- self-attention maps from object query regions to other frames,
- same-coordinate attention mass versus object-coordinate attention mass,
- cross-attention object-token trajectory maps,
- learned $\lambda_h(\tau)$ and $\lambda_w(\tau)$ curves,
- per-layer or per-head lambda heatmaps.

For timestep-conditioned lambda, plot $\lambda_h(\tau)$ and $\lambda_w(\tau)$ over the 1000 training timesteps and the 50 inference timestep locations. A positive mechanism signal is $\lambda_h(\tau)<1$ and $\lambda_w(\tau)<1$ mainly at high-noise early timesteps, with values closer to one at late low-noise timesteps.
