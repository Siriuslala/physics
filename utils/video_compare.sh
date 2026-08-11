#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source ../scripts/env.sh

VIDEO_PATHS=(
  "/work/liyueyan/Interpretability/physics/wan_eval/videophy/a800/42/wan2.1_1B3/ckpt_step_latest"
  "/work/liyueyan/Interpretability/physics/wan_eval/videophy/a800/42/wan2.1_1B3/ckpt_step_latest_lambda_steps_1-2-3-4-5_lambda_manual_0.70_0.70"
  "/work/liyueyan/Interpretability/physics/wan_eval/videophy/a800/42/fixed_lambda_lora-bsz_32-lr_1e-4-lora_rank_64-lora_alpha_32-lora_modules_attn-lambda_scope_model-lambda_h_0.75-lambda_w_0.75-lambda_global_0-lam_schedulecosine_0-steps_1500-warmup_0.03-timestep_mixed_early_0.1_prob_0.9-seed_42/ckpt_step_800_lambda_steps_1-2-3-4-5_lambda_manual_0.70_0.70"
)

VIDEO_LABELS=(
  "wan"
  "wan_rope"
  "wan_rope_lora"
)

if [[ "${#VIDEO_PATHS[@]}" -ne "${#VIDEO_LABELS[@]}" ]]; then
  echo "VIDEO_PATHS and VIDEO_LABELS must have the same length." >&2
  exit 1
fi

python "$ROOT_DIR/utils/video_compare.py" \
  --paths "${VIDEO_PATHS[@]}" \
  --labels "${VIDEO_LABELS[@]}" \
  --port 7861
