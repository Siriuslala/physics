#!/usr/bin/env bash
set -euo pipefail
trap 'stty sane 2>/dev/null || true' EXIT INT TERM

cd "$(dirname "${BASH_SOURCE[0]}")"
source ../../scripts/env.sh
cd "$ROOT_DIR"

set +u
source ~/miniforge3/etc/profile.d/conda.sh
conda activate video
set -u

if [[ -n "${PYTHONPATH+x}" ]]; then
  export PYTHONPATH="$ROOT_DIR/DiffSynth-Studio:$PYTHONPATH"
else
  export PYTHONPATH="$ROOT_DIR/DiffSynth-Studio"
fi
export DIFFSYNTH_MODEL_BASE_PATH=/
export DIFFSYNTH_SKIP_DOWNLOAD=true


# GPU settings ==============================
export CUDA_VISIBLE_DEVICES=1
NUM_PROCESSES=1

# export CUDA_VISIBLE_DEVICES=0,1
# NUM_PROCESSES=2

if [[ "$NUM_PROCESSES" -gt 1 ]]; then LAUNCH_MODE=multi; else LAUNCH_MODE=single; fi

# General settings ==============================
USE_CACHE=1
MODEL_SIZE=1.3B
MIXED_PRECISION=bf16
SEED=42
DETERMINISTIC=0  # 1 enables deterministic PyTorch algorithms when available; slower and may warn.

DATASET_BASE_PATH=/datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/video_data
DATASET_METADATA_PATH=/datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/video_data/physics_metadata_new/metadata_physics_merged_reflection4000.csv
DATASET_NUM_WORKERS=12
DATASET_REPEAT=1

HEIGHT=480
WIDTH=832
NUM_FRAMES=81
VIDEO_CLIP_SECONDS=5.0

# Hyper-params settings ==============================
GLOBAL_BATCH_SIZE=16
DATASET_BATCH_SIZE=1
ENABLE_BATCHED_SFT=1
if [[ "$NUM_PROCESSES" -gt 1 ]]; then WORLD_SIZE=$NUM_PROCESSES; else WORLD_SIZE=1; fi
GRADIENT_ACCUMULATION_STEPS=$(( (GLOBAL_BATCH_SIZE + DATASET_BATCH_SIZE * WORLD_SIZE - 1) / (DATASET_BATCH_SIZE * WORLD_SIZE) ))

LEARNING_RATE=1e-3  # Unused. lambda-only has no main trainable params, so LAMBDA_LR is the effective LR.
LR_SCHEDULER=constant_with_warmup
WARMUP_RATIO=0.03
ADAM_BETA1=0.9
ADAM_BETA2=0.999
ADAM_EPSILON=1e-8
WEIGHT_DECAY=0.0
MAX_GRAD_NORM=1.0
NUM_EPOCHS=3
MAX_TRAIN_STEPS=3000
SAVE_STEPS=200

# Full-range training is [0.0, 1.0]. Do not set min=max; DiffSynth requires min < max.
MIN_TIMESTEP_BOUNDARY=0.0
MAX_TIMESTEP_BOUNDARY=1.0
TIMESTEP_SAMPLING_STRATEGY=uniform  # uniform | early_rest_mixture
TIMESTEP_MIXTURE_EARLY_BOUNDARY=0.12
TIMESTEP_MIXTURE_EARLY_PROB=0.5

LAMBDA_SCOPE=layer  # model | layer | head
LAMBDA_LR=1e-3
LAMBDA_BETA=1e-4
LAMBDA_TIMESTEP_CONDITIONED=1
LAMBDA_HIDDEN_DIM=128
LAMBDA_CHECKPOINT=

# Tensorboard settings ==============================
ENABLE_WANDB=1
WANDB_PROJECT=wan21-physics-lambda-only

# Model path, Cache path & Output settings ==============================
if [[ "$MODEL_SIZE" == "1.3B" || "$MODEL_SIZE" == "1B3" ]]; then
  PRETRAINED_MODEL_DIR="${MODEL_DIR%/}/Wan2.1-T2V-1.3B"
  CKPT_ROOT="$WORK_TRAIN_DIR/Wan2.1-T2V-1B3"
elif [[ "$MODEL_SIZE" == "14B" ]]; then
  PRETRAINED_MODEL_DIR="${MODEL_DIR%/}/Wan2.1-T2V-14B"
  CKPT_ROOT="$WORK_TRAIN_DIR/Wan2.1-T2V-14B"
else
  echo "Unsupported MODEL_SIZE: $MODEL_SIZE" >&2
  exit 1
fi

MODEL_CONFIGS="${PRETRAINED_MODEL_DIR}:diffusion_pytorch_model*.safetensors,${PRETRAINED_MODEL_DIR}:models_t5_umt5-xxl-enc-bf16.pth,${PRETRAINED_MODEL_DIR}:Wan2.1_VAE.pth"
TOKENIZER_PATH="${PRETRAINED_MODEL_DIR}/google/umt5-xxl"
CACHE_DIR=/datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/wan_cache/Wan2.1-T2V-${MODEL_SIZE}/physics_81f_5s_compact
TRAIN_ROOT="$CKPT_ROOT/lambda_only"

TRAIN_LEN=$(python - <<PY2
import pandas as pd
print(len(pd.read_csv('$DATASET_METADATA_PATH')))
PY2
)
if [[ -n "$MAX_TRAIN_STEPS" ]]; then TRAIN_TOKEN="steps_${MAX_TRAIN_STEPS}"; else TRAIN_TOKEN="epochs_${NUM_EPOCHS}"; fi
EXP_NAME="lambda_only-bsz_${GLOBAL_BATCH_SIZE}-lr_${LEARNING_RATE}-lambda_scope_${LAMBDA_SCOPE}-lambda_lr_${LAMBDA_LR}-lambda_beta_${LAMBDA_BETA}-lambda_tcond_${LAMBDA_TIMESTEP_CONDITIONED}-lambda_hidden_${LAMBDA_HIDDEN_DIM}-${TRAIN_TOKEN}-warmup_${WARMUP_RATIO}-adam_beta1_${ADAM_BETA1}-beta2_${ADAM_BETA2}-timestep_${MIN_TIMESTEP_BOUNDARY}_${MAX_TIMESTEP_BOUNDARY}-seed_${SEED}"
OUTPUT_PATH="$TRAIN_ROOT/$EXP_NAME"
mkdir -p "$CACHE_DIR" "$OUTPUT_PATH"
WANDB_NAME="$EXP_NAME"

# Prepare args ==============================
COMMON_ARGS=(
  --height "$HEIGHT"
  --width "$WIDTH"
  --num_frames "$NUM_FRAMES"
  --video_sampling_mode first_seconds_uniform
  --video_clip_seconds "$VIDEO_CLIP_SECONDS"
  --dataset_num_workers "$DATASET_NUM_WORKERS"
  --model_id_with_origin_paths "$MODEL_CONFIGS"
  --tokenizer_path "$TOKENIZER_PATH"
  --remove_prefix_in_ckpt "pipe.dit."
  --use_gradient_checkpointing
  --seed "$SEED"
)
if [[ "$DETERMINISTIC" == "1" ]]; then COMMON_ARGS+=(--deterministic); fi

TRAIN_ARGS=(
  --dataset_batch_size "$DATASET_BATCH_SIZE"
  --dataset_repeat "$DATASET_REPEAT"
  --learning_rate "$LEARNING_RATE"
  --lr_scheduler "$LR_SCHEDULER"
  --warmup_ratio "$WARMUP_RATIO"
  --adam_beta1 "$ADAM_BETA1"
  --adam_beta2 "$ADAM_BETA2"
  --adam_epsilon "$ADAM_EPSILON"
  --weight_decay "$WEIGHT_DECAY"
  --max_grad_norm "$MAX_GRAD_NORM"
  --num_epochs "$NUM_EPOCHS"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --save_steps "$SAVE_STEPS"
  --output_path "$OUTPUT_PATH"
  --min_timestep_boundary "$MIN_TIMESTEP_BOUNDARY"
  --max_timestep_boundary "$MAX_TIMESTEP_BOUNDARY"
  --timestep_sampling_strategy "$TIMESTEP_SAMPLING_STRATEGY"
  --timestep_mixture_early_boundary "$TIMESTEP_MIXTURE_EARLY_BOUNDARY"
  --timestep_mixture_early_prob "$TIMESTEP_MIXTURE_EARLY_PROB"
  --wan_spatial_rope_lambda_enabled
  --wan_spatial_rope_lambda_scope "$LAMBDA_SCOPE"
  --wan_spatial_rope_lambda_lr "$LAMBDA_LR"
  --wan_spatial_rope_lambda_beta "$LAMBDA_BETA"
  --wan_spatial_rope_lambda_hidden_dim "$LAMBDA_HIDDEN_DIM"
)
if [[ "$LAMBDA_TIMESTEP_CONDITIONED" == "1" ]]; then TRAIN_ARGS+=(--wan_spatial_rope_lambda_timestep_conditioned); fi
if [[ -n "$LAMBDA_CHECKPOINT" ]]; then TRAIN_ARGS+=(--wan_spatial_rope_lambda_checkpoint "$LAMBDA_CHECKPOINT"); fi
if [[ -n "$MAX_TRAIN_STEPS" ]]; then TRAIN_ARGS+=(--max_train_steps "$MAX_TRAIN_STEPS"); fi
if [[ "$ENABLE_BATCHED_SFT" == "1" ]]; then TRAIN_ARGS+=(--enable_batched_sft); fi

if [[ "$ENABLE_WANDB" == "1" ]]; then
  export WANDB_PROJECT WANDB_NAME
  TRAIN_ARGS+=(--enable_wandb_log --wandb_project "$WANDB_PROJECT")
fi

launch_train() {
  if [[ "$LAUNCH_MODE" == "multi" ]]; then
    accelerate launch --num_processes "$NUM_PROCESSES" --num_machines 1 --mixed_precision "$MIXED_PRECISION" --dynamo_backend no DiffSynth-Studio/examples/wanvideo/model_training/train.py "$@"
  else
    accelerate launch --num_processes 1 --num_machines 1 --mixed_precision "$MIXED_PRECISION" --dynamo_backend no DiffSynth-Studio/examples/wanvideo/model_training/train.py "$@"
  fi
}

if [[ "$USE_CACHE" == "1" ]]; then
  launch_train --dataset_base_path "$CACHE_DIR" --task "sft:train" "${COMMON_ARGS[@]}" "${TRAIN_ARGS[@]}"
else
  launch_train --dataset_base_path "$DATASET_BASE_PATH" --dataset_metadata_path "$DATASET_METADATA_PATH" --task "sft" "${COMMON_ARGS[@]}" "${TRAIN_ARGS[@]}"
fi

echo "Experiment output: $OUTPUT_PATH"
echo "Condition cache: $CACHE_DIR"
echo "Effective global batch: $((DATASET_BATCH_SIZE * WORLD_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "Training rows: $TRAIN_LEN"
