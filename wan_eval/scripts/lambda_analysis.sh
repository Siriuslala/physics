#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source ../../scripts/env.sh
cd $ROOT_DIR

set +u
source ~/miniforge3/etc/profile.d/conda.sh
conda activate video
set -u

MODEL_PATH="/work/liyueyan/Interpretability/physics_train/Wan2.1-T2V-1B3/lambda_lora/lambda_lora-bsz_16-lr_1e-4-lora_rank_32-lora_alpha_32-lora_modules_attn-lambda_scope_head-lambda_lr_5e-4-lambda_beta_1e-4-lambda_tcond_1-lambda_hidden_128-steps_3000-warmup_0.03-adam_beta1_0.9-beta2_0.999-timestep_mixed_early_0.14_prob_0.60-seed_42"
CKPT_STEP=2000
TASK=t2v-1.3B
OUTPUT_DIR=/work/liyueyan/Interpretability/physics/wan_eval/lambda_analysis
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
