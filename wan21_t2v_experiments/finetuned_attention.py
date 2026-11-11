"""Strict LoRA loading and fixed spatial RoPE configuration for attention probes."""

import math
from pathlib import Path

import torch

from .wan21_t2v_experiment_patch import Wan21T2VRopePatchConfig


def _eval_fixed_spatial_scales(lambda_h, lambda_w, device):
    """Return effective h/w scales using the eval fixed-module float32 chain.

    The two positive Python floats are stored as float32 logarithms, then pass
    through exp, clamped log, and exp on the model device, exactly as
    WanSpatialRopeLambda(parametrization='fixed', scope='model'). The returned
    Python floats preserve the resulting float32 values when RoPE promotes them
    to float64; e.g. requested 0.70 becomes 0.699999988079071.
    """
    raw = torch.tensor([math.log(lambda_h), math.log(lambda_w)], dtype=torch.float32, device=device)
    return tuple(float(value) for value in raw.exp().clamp_min(1e-12).log().exp().tolist())


def prepare_finetuned_attention(model, checkpoint, rank, alpha, modules, lambda_h, lambda_w, steps, sampling_steps):
    """Load a complete adapter and return its provenance plus a shared RoPE config.

    ``model`` is the unsharded official Wan DiT. The safetensors checkpoint must
    contain exactly all A/B adapter tensors for ``attn`` or ``attn_ffn``; missing,
    unexpected, non-adapter, and shape-incompatible tensors are rejected. Adapters
    remain unmerged, matching wan_eval inference. Spatial scales apply to both CFG
    branches on the supplied one-based denoising steps (empty means every step).
    """
    if not all(math.isfinite(v) and v > 0 for v in (lambda_h, lambda_w)):
        raise ValueError('Spatial RoPE lambdas must be finite and positive.')
    steps = tuple(sorted(set(int(s) for s in steps)))
    if any(s < 1 or s > sampling_steps for s in steps):
        raise ValueError('RoPE steps must be within the one-based sampling-step range.')
    effective_h, effective_w = _eval_fixed_spatial_scales(
        lambda_h, lambda_w, next(model.parameters()).device,
    )
    rope = Wan21T2VRopePatchConfig(
        enabled=True, mode='full', lambda_h=effective_h, lambda_w=effective_w, apply_steps=steps,
    )
    metadata = {
        'lora_checkpoint': '', 'lora_loaded': False,
        'rope_lambda_f': 1.0, 'rope_lambda_h': float(lambda_h), 'rope_lambda_w': float(lambda_w),
        'rope_lambda_h_effective': effective_h, 'rope_lambda_w_effective': effective_w,
        'rope_lambda_precision': 'eval_fixed_float32_log_exp_log_exp_on_model_device',
        'rope_steps': list(steps), 'rope_steps_empty_means': 'all',
        'rope_phase_convention': 'principal_complex_phase_matching_wan_eval',
    }
    if not checkpoint:
        return metadata, rope
    if rank <= 0 or alpha <= 0:
        raise ValueError('LoRA rank and alpha must be positive.')
    targets = {'attn': ['q', 'k', 'v', 'o'], 'attn_ffn': ['q', 'k', 'v', 'o', 'ffn.0', 'ffn.2']}[modules]
    from peft import LoraConfig, inject_adapter_in_model
    from safetensors.torch import load_file

    path = Path(checkpoint).expanduser().resolve(strict=True)
    raw = load_file(str(path), device='cpu')
    state = {}
    for key, value in raw.items():
        if not any(part in key for part in ('lora_A.', 'lora_B.')):
            raise ValueError(f'Unexpected non-adapter checkpoint tensor: {key}')
        mapped = key.replace('lora_A.weight', 'lora_A.default.weight').replace('lora_B.weight', 'lora_B.default.weight')
        if mapped in state:
            raise ValueError(f'Duplicate adapter tensor after key mapping: {mapped}')
        state[mapped] = value
    inject_adapter_in_model(LoraConfig(r=rank, lora_alpha=alpha, target_modules=targets), model)
    expected = {k: v for k, v in model.state_dict().items() if 'lora_A.' in k or 'lora_B.' in k}
    missing, unexpected = set(expected) - set(state), set(state) - set(expected)
    if missing or unexpected:
        raise ValueError(f'Incomplete/incompatible LoRA checkpoint: missing={sorted(missing)}, unexpected={sorted(unexpected)}')
    for key, value in state.items():
        if value.shape != expected[key].shape:
            raise ValueError(f'LoRA shape mismatch for {key}: {tuple(value.shape)} != {tuple(expected[key].shape)}')
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or any('lora_' in k for k in result.missing_keys):
        raise RuntimeError(f'LoRA loading failed validation: {result}')
    model.eval().requires_grad_(False)
    metadata.update(lora_checkpoint=str(path), lora_loaded=True, lora_rank=int(rank),
                    lora_alpha=int(alpha), lora_modules=modules, lora_target_modules=targets,
                    lora_tensor_count=len(state), lora_checkpoint_bytes=path.stat().st_size)
    return metadata, rope
