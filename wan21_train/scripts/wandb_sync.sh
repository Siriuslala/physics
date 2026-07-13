#!/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")"
source ../../scripts/env.sh

wandb_log_paths=(
    # $WORK_TRAIN_DIR/Wan2.1-T2V-1B3/lambda_lora/lambda_lora-bsz_16-lr_1e-4-lora_rank_64-lora_alpha_32-lora_modules_attn-lambda_scope_head-lambda_lr_5e-4-lambda_beta_0-lambda_tcond_1-lambda_hidden_128-range_bounded_leq_one_min_0.5-steps_3000-warmup_0.03-timestep_mixed_early_0.10_prob_1.00-seed_42/wandb_log/wandb/offline-run-20260701_105945-kutfwli8
    # $WORK_TRAIN_DIR/Wan2.1-T2V-1B3/lambda_only/lambda_only-bsz_16-lr_1e-3-lambda_scope_head-lambda_lr_5e-4-lambda_beta_0-lambda_tcond_1-lambda_hidden_128-range_bounded_leq_one_min_0.50_eps_5e-2-steps_3000-warmup_0.03-adam_beta1_0.9-beta2_0.999-timestep_mixed_early_0.10_prob_1.00-seed_42/wandb_log/wandb/offline-run-20260702_101518-jtihsmc5
    $WORK_TRAIN_DIR/Wan2.1-T2V-1B3/fixed_lambda_lora/fixed_lambda_lora-bsz_16-lr_1e-4-lora_rank_64-lora_alpha_32-lora_modules_attn-lambda_scope_model-lambda_h_0.75-lambda_w_0.75-lam_schedulecosine_0-steps_3000-warmup_0.03-adam_beta1_0.9-beta2_0.999-timestep_mixed_early_0.1_prob_0.9-seed_42/wandb_log/wandb/offline-run-20260706_114341-cnco030a
    $WORK_TRAIN_DIR/Wan2.1-T2V-1B3/fixed_lambda_lora/fixed_lambda_lora-bsz_16-lr_1e-4-lora_rank_64-lora_alpha_32-lora_modules_attn-lambda_scope_model-lambda_h_0.75-lambda_w_0.75-lam_schedulecosine_1500-steps_3000-warmup_0.03-adam_beta1_0.9-beta2_0.999-timestep_mixed_early_0.1_prob_0.9-seed_42/wandb_log/wandb/offline-run-20260707_180237-wp0epjen
)

for log_dir in "${wandb_log_paths[@]}"; do

    if [ ! -d "$log_dir" ]; then
        echo "Warning: $log_dir inexisted, skip sync"
        continue
    fi

    wandb sync "$log_dir"

done

echo "Sync completed"