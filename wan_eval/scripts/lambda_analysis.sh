#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source ../../scripts/env.sh
cd $ROOT_DIR

set +u
source ~/miniforge3/etc/profile.d/conda.sh
conda activate video
set -u

MODEL_PATH=$WORK_TRAIN_DIR/Wan2.1-T2V-1B3/lambda_lora/lambda_lora-bsz_16-lr_1e-4-lora_rank_64-lora_alpha_32-lora_modules_attn-lambda_scope_head-lambda_lr_5e-4-lambda_beta_0-lambda_tcond_1-lambda_hidden_128-range_bounded_leq_one_min_0.5-steps_3000-warmup_0.03-timestep_mixed_early_0.10_prob_1.00-seed_42
# MODEL_PATH=$WORK_TRAIN_DIR/Wan2.1-T2V-1B3/lambda_only/lambda_only-bsz_16-lr_1e-3-lambda_scope_head-lambda_lr_5e-4-lambda_beta_0-lambda_tcond_1-lambda_hidden_128-range_bounded_leq_one_min_0.50_eps_5e-2-steps_3000-warmup_0.03-adam_beta1_0.9-beta2_0.999-timestep_mixed_early_0.10_prob_1.00-seed_42
CKPT_STEP=3000

MODEL_NAME="$(basename "$MODEL_PATH")"
OUTPUT_DIR=$WORK_DIR/wan_eval/lambda_analysis/${MODEL_NAME}/${CKPT_STEP}

TASK=t2v-1.3B
SAMPLE_SOLVER=unipc
SAMPLE_STEPS=50
SAMPLE_SHIFT=5.0
EARLY_FRACTION=0.1
DPI=300

python wan_eval/analyze_wan_lambda.py \
  --model_path "$MODEL_PATH" \
  --ckpt_step "$CKPT_STEP" \
  --task "$TASK" \
  --output_dir "$OUTPUT_DIR" \
  --sample_solver "$SAMPLE_SOLVER" \
  --sample_steps "$SAMPLE_STEPS" \
  --sample_shift "$SAMPLE_SHIFT" \
  --early_fraction "$EARLY_FRACTION" \
  --dpi "$DPI"
