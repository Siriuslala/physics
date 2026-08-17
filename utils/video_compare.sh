#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source ../scripts/env.sh

VIDEO_PATHS=(
  "$WORK_DIR/wan_eval/videophy/a800/42/wan2.1_1B3/_ckpt_step_latest"
  # "$WORK_DIR/wan_eval/videophy_rewrite/a800/42/wan2.1_1B3/_ckpt_step_latest_lambda_steps_1-2-3-4-5_lambda_manual_0.70_0.70"
  # "$WORK_DIR/wan_eval/videophy_rewrite/a800/42/fixed_lambda_lora-bsz_32-lr_1e-4-lora_rank_64-lora_alpha_32-lora_modules_attn-lambda_scope_model-lambda_h_0.75-lambda_w_0.75-lambda_global_0-lam_schedulecosine_0-steps_1500-warmup_0.03-timestep_mixed_early_0.1_prob_0.9-seed_42/_ckpt_step_800_lambda_steps_1-2-3-4-5_lambda_manual_0.70_0.70"
  "$WORK_DIR/wan_eval/videophy/a800/42/bsz_32-lr_1e-4-lora_rank_64-lora_alpha_32-lora_modules_attn-steps_1500-warmup_0.03-adam_beta1_0.9-beta2_0.999-timestep_uniform_0.0_1.0-seed_42/_ckpt_step_800"
)
VIDEO_LABELS=(
  "wan"
  # "wan_rope"
  # "wan_rope_lora"
  "wan_lora"
)

if [[ "${#VIDEO_PATHS[@]}" -ne "${#VIDEO_LABELS[@]}" ]]; then
  echo "VIDEO_PATHS and VIDEO_LABELS must have the same length." >&2
  exit 1
fi

python "$ROOT_DIR/utils/video_compare.py" \
  --paths "${VIDEO_PATHS[@]}" \
  --labels "${VIDEO_LABELS[@]}" \
  --port 7861
