cd "$(dirname "${BASH_SOURCE[0]}")"
source ../scripts/env.sh
cd $ROOT_DIR/projects/Wan2_1

set +u
source ~/miniforge3/etc/profile.d/conda.sh
conda activate video
set -u

python $ROOT_DIR/utils/visualize_wandb_offline.py \
  --wandb-dir $WORK_TRAIN_DIR/Wan2.1-T2V-1B3/lambda_only/lambda_only-bsz_16-lr_1e-3-lambda_scope_head-lambda_lr_5e-4-lambda_beta_1e-4-lambda_tcond_1-lambda_hidden_128-steps_3000-warmup_0.03-adam_beta1_0.9-beta2_0.999-timestep_mixed_early_0.10_prob_1.00-seed_42/wandb_log/wandb/offline-run-20260627_141923-z1zc9yr2 \
  --output-dir $WORK_DIR/wan_train_viz
