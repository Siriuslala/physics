#!/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")"
source ../../scripts/env.sh

wandb_log_paths=(
    $WORK_TRAIN_DIR/Wan2.1-T2V-1B3/lambda_only/lambda_only-bsz_16-lr_1e-3-lambda_scope_head-lambda_lr_5e-4-lambda_beta_1e-4-lambda_tcond_1-lambda_hidden_128-steps_3000-warmup_0.03-adam_beta1_0.9-beta2_0.999-timestep_mixed_early_0.10_prob_0.80-seed_42/wandb_log/wandb/offline-run-20260625_101602-03h5332a
    $WORK_TRAIN_DIR/Wan2.1-T2V-1B3/lambda_lora/lambda_lora-bsz_16-lr_1e-4-lora_rank_64-lora_alpha_32-lora_modules_attn-lambda_scope_head-lambda_lr_5e-4-lambda_beta_1e-4-lambda_tcond_1-lambda_hidden_128-steps_3000-warmup_0.03-adam_beta1_0.9-beta2_0.999-timestep_mixed_early_0.10_prob_0.80-seed_42/wandb_log/wandb/offline-run-20260622_230228-tgkuzs5d
    $WORK_TRAIN_DIR/Wan2.1-T2V-1B3/lambda_lora/lambda_lora-bsz_16-lr_1e-4-lora_rank_128-lora_alpha_64-lora_modules_attn_ffn-lambda_scope_head-lambda_lr_5e-4-lambda_beta_1e-4-lambda_tcond_1-lambda_hidden_128-steps_3000-warmup_0.03-adam_beta1_0.9-beta2_0.999-timestep_mixed_early_0.10_prob_0.80-seed_42/wandb_log/wandb/offline-run-20260624_112348-no2xdln9
)

for log_dir in "${wandb_log_paths[@]}"; do

    if [ ! -d "$log_dir" ]; then
        echo "Warning: $log_dir inexisted, skip sync"
        continue
    fi

    wandb sync "$log_dir"

done

echo "Sync completed"