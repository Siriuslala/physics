#!/usr/bin/env bash
# Reproduce the basketball baseline / fine-tuned RoPE attention comparison.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh
cd "$ROOT_DIR"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate video
set -u

STAGE=${1:?Usage: bash scripts/rope_posthoc_attention_wan2_1.sh cross|audit|self SEED baseline|modified}
SEED=${2:?Missing seed}
CONDITION=${3:?Missing condition}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLBACKEND=Agg
PROMPT='Against a pure white background, a basketball falls vertically from mid-air onto a wooden floor and bounces up several times.'
PROMPT_TAG=${PROMPT// /_}
BASE="$WORK_DIR/outputs_wan_2_1_t2v-1.3B"
OUT="$BASE/rope_posthoc_attention/$PROMPT_TAG"
TRAIN="$WORK_TRAIN_DIR/Wan2.1-T2V-1B3/fixed_lambda_lora"
case "$SEED" in
    8|20)
        MODEL="$TRAIN/fixed_lambda_lora-bsz_32-lr_1e-4-lora_rank_64-lora_alpha_32-lora_modules_attn-lambda_scope_model-lambda_h_0.75-lambda_w_0.75-lambda_global_0-lam_schedulecosine_0-steps_1500-warmup_0.03-timestep_mixed_early_0.1_prob_0.9-seed_42/step-1000.safetensors"
        LAMBDA=0.75; MODULES=attn; REFERENCE_STEP=50
        if [[ "$SEED" == 8 ]]; then REFERENCE_STEP=45; fi
        ;;
    29)
        MODEL="$TRAIN/fixed_lambda_lora-bsz_32-lr_1e-4-lora_rank_64-lora_alpha_32-lora_modules_attn_ffn-lambda_scope_model-lambda_h_0.75-lambda_w_0.75-lambda_global_0-lam_schedulecosine_0-steps_1500-warmup_0.03-timestep_mixed_early_0.1_prob_0.5-seed_42/step-700.safetensors"
        LAMBDA=0.70; MODULES=attn_ffn; REFERENCE_STEP=50
        ;;
    *) echo 'Only seeds 8, 20, and 29 are defined.' >&2; exit 2 ;;
esac
MODIFICATION=()
case "$CONDITION" in
    baseline) CROSS="$BASE/cross_attention_token_viz/$PROMPT_TAG/seed_${SEED}_shift_5.0_guide_5.0" ;;
    modified)
        [[ -f "$MODEL" ]]
        CROSS="$OUT/modified/seed_$SEED/cross_attention"
        MODIFICATION=(--finetune_lora_checkpoint "$MODEL" --finetune_lora_rank 64 --finetune_lora_alpha 32 --finetune_lora_modules "$MODULES" --rope_modification_lambda_h "$LAMBDA" --rope_modification_lambda_w "$LAMBDA" --rope_modification_steps 1,2,3,4,5)
        ;;
    *) echo 'Condition must be baseline or modified.' >&2; exit 2 ;;
esac
COMMON=(--wan21_root "$ROOT_DIR/projects/Wan2_1" --ckpt_dir "$MODEL_DIR/Wan2.1-T2V-1.3B" --task t2v-1.3B --prompt "$PROMPT" --size '832*480' --frame_num 81 --sample_solver unipc --sampling_steps 50 --shift 5.0 --guide_scale 5.0 --seed "$SEED" --offload_model True --target_object_words basketball --target_verb_words falls,bounces,up)
AUDIT="$OUT/$CONDITION/seed_$SEED/region_audit"
QUANTILE=${REGION_QUANTILE:-0.8}
RADIUS_ALPHA=${REGION_RADIUS_ALPHA:-1.5}
if [[ "$STAGE" != audit ]]; then
    GPU_NAME=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name --format=csv,noheader)
    [[ "$GPU_NAME" == *A800* ]] || { echo "A800 required; got $GPU_NAME" >&2; exit 2; }
    FREE_MIB=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.free --format=csv,noheader,nounits)
    (( FREE_MIB >= 40000 )) || { echo "At least 40000 MiB free required; got $FREE_MIB" >&2; exit 2; }
fi
case "$STAGE" in
    cross)
        [[ "$CONDITION" == modified ]] || { echo 'Reuse the existing baseline cross-attention.' >&2; exit 2; }
        mkdir -p "$CROSS"
        nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name,uuid,driver_version,memory.used,memory.free --format=csv > "$CROSS/gpu_provenance.csv"
        python -u wan21_t2v_experiments/run_wan21_t2v_experiments.py --experiment cross_attention_token_viz "${COMMON[@]}" "${MODIFICATION[@]}" --output_dir "$CROSS" --cross_attn_steps "$(seq -s, 1 50)" --save_video True --traj_enable False --save_attention_pdfs False --save_trajectory_pdfs False --save_trajectory_timeline_pdfs False --cross_attention_token_viz_num_workers 2
        ;;
    audit)
        python -u -m wan21_t2v_experiments.region_audit --cross-dir "$CROSS" --output-dir "$AUDIT" --reference-step "$REFERENCE_STEP" --reference-layer 27 --center-quantile "$QUANTILE" --support-radius-alpha "$RADIUS_ALPHA"
        ;;
    self)
        SELF="$OUT/$CONDITION/seed_$SEED/self_attention"
        mkdir -p "$SELF"
        nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name,uuid,driver_version,memory.used,memory.free --format=csv > "$SELF/gpu_provenance.csv"
        python -u wan21_t2v_experiments/run_wan21_t2v_experiments.py --experiment self_attention_viz "${COMMON[@]}" "${MODIFICATION[@]}" --output_dir "$SELF" --reuse_cross_attention_dir "$CROSS" --self_attention_viz_steps "$(seq -s, 1 50)" --self_attention_viz_reference_step "$REFERENCE_STEP" --self_attention_viz_reference_layer 27 --self_attention_viz_reference_center_quantile "$QUANTILE" --self_attention_viz_support_radius_alpha "$RADIUS_ALPHA" --self_attention_viz_query_video_frame_indices 1 --self_attention_viz_object_query_token_limit_per_frame 0 --self_attention_viz_num_viz_frames 10 --self_attention_viz_save_attention_pdfs True --self_attention_viz_attention_pdf_share_color_scale True --self_attention_viz_stop_after_last_probe_step False --self_attention_viz_num_workers 8 --save_video True
        ;;
    *) echo "Unknown stage: $STAGE" >&2; exit 2 ;;
esac
