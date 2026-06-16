#!/usr/bin/env bash
set -euo pipefail

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

export CUDA_VISIBLE_DEVICES=2

# Usage examples:
#   bash wan21_train/scripts/train_wan21_t2v_1b3_full.sh
#   Edit the parameter block below, then run this script.

MODE=train  # cache | train | both
USE_CACHE=0  # 0: online video/text encoding; 1: train from cached .pth files
COMPACT_CACHE=1  # save only compact Wan T2V cache fields during MODE=cache
CACHE_BATCH_SIZE=8  # per-process batch size for MODE=cache
ENABLE_CACHE_TIMING=1  # print text/video/VAE timing during MODE=cache
VIDEO_OUTPUT_FORMAT=tensor_uint8  # pil | tensor_uint8
CACHE_PREFETCH_BATCHES=2  # overlap video preprocessing with VAE/cache saving; 0 disables it
CACHE_RESUME=1  # single-process only: skip existing continuous cache files
CACHE_SKIP_MISMATCHED_SHAPES=1  # skip decoded videos whose shape is not NUM_FRAMES x HEIGHT x WIDTH
VAE_ENCODE_BATCH_SIZE=2  # 0 means encode the full cache batch at once
MODEL_SIZE=1.3B         # 1.3B | 14B
NUM_PROCESSES=1
if [[ "$NUM_PROCESSES" -gt 1 ]]; then
  LAUNCH_MODE=multi
else
  LAUNCH_MODE=single
fi
MIXED_PRECISION=bf16

DATASET_BASE_PATH=/datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/video_data
DATASET_METADATA_PATH=/datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/video_data/physics_metadata/metadata_physics_merged_reflection4000.csv
DATASET_NUM_WORKERS=4
DATASET_REPEAT=1

HEIGHT=480
WIDTH=832
NUM_FRAMES=81
VIDEO_CLIP_SECONDS=5.0

GLOBAL_BATCH_SIZE=64
DATASET_BATCH_SIZE=1
ENABLE_BATCHED_SFT=1
if [[ "$NUM_PROCESSES" -gt 1 ]]; then
  WORLD_SIZE=$NUM_PROCESSES
else
  WORLD_SIZE=1
fi
GRADIENT_ACCUMULATION_STEPS=$(( (GLOBAL_BATCH_SIZE + DATASET_BATCH_SIZE * WORLD_SIZE - 1) / (DATASET_BATCH_SIZE * WORLD_SIZE) ))

LEARNING_RATE=1e-5
LR_SCHEDULER=constant_with_warmup
WARMUP_RATIO=0.03
ADAM_BETA1=0.9
ADAM_BETA2=0.999
ADAM_EPSILON=1e-8
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0
NUM_EPOCHS=1
MAX_TRAIN_STEPS=
SAVE_STEPS=500

# Timestep sampling settings ==============================
MIN_TIMESTEP_BOUNDARY=0.0
MAX_TIMESTEP_BOUNDARY=1.0
TIMESTEP_SAMPLING_STRATEGY=uniform  # uniform | early_rest_mixture
TIMESTEP_MIXTURE_EARLY_BOUNDARY=0.12
TIMESTEP_MIXTURE_EARLY_PROB=0.5

ENABLE_WANDB=0
WANDB_PROJECT=wan21-physics-full
WANDB_NAME=

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
CACHE_DIR=$WORK_TRAIN_DIR/condition_cache/Wan2.1-T2V-${MODEL_SIZE}/physics_81f_5s
if [[ "$COMPACT_CACHE" == "1" ]]; then
  CACHE_DIR=$WORK_TRAIN_DIR/condition_cache/Wan2.1-T2V-${MODEL_SIZE}/physics_81f_5s_compact
fi
TRAIN_ROOT="$CKPT_ROOT/full"

TRAIN_LEN=$(python - <<PY
import pandas as pd
print(len(pd.read_csv('$DATASET_METADATA_PATH')))
PY
)
if [[ -n "$MAX_TRAIN_STEPS" ]]; then
  TRAIN_TOKEN="steps_${MAX_TRAIN_STEPS}"
else
  TRAIN_TOKEN="epochs_${NUM_EPOCHS}"
fi
EXP_NAME="bsz_${GLOBAL_BATCH_SIZE}-lr_${LEARNING_RATE}-${TRAIN_TOKEN}-warmup_${WARMUP_RATIO}-adam_beta1_${ADAM_BETA1}-beta2_${ADAM_BETA2}"
OUTPUT_PATH=$TRAIN_ROOT/$EXP_NAME
mkdir -p "$CACHE_DIR" "$OUTPUT_PATH"

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
)

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
  --trainable_models "dit"
)
if [[ -n "$MAX_TRAIN_STEPS" ]]; then
  TRAIN_ARGS+=(--max_train_steps "$MAX_TRAIN_STEPS")
fi
if [[ "$ENABLE_BATCHED_SFT" == "1" ]]; then
  TRAIN_ARGS+=(--enable_batched_sft)
fi
if [[ "$ENABLE_WANDB" == "1" ]]; then
  export WANDB_PROJECT
  if [[ -n "$WANDB_NAME" ]]; then export WANDB_NAME; fi
  TRAIN_ARGS+=(--enable_wandb_log --wandb_project "$WANDB_PROJECT")
fi

CACHE_ARGS=(
  --dataset_batch_size "$CACHE_BATCH_SIZE"
  --video_output_format "$VIDEO_OUTPUT_FORMAT"
  --vae_encode_batch_size "$VAE_ENCODE_BATCH_SIZE"
  --cache_prefetch_batches "$CACHE_PREFETCH_BATCHES"
)
if [[ "$COMPACT_CACHE" == "1" ]]; then
  CACHE_ARGS+=(--compact_cache)
fi
if [[ "$ENABLE_CACHE_TIMING" == "1" ]]; then
  CACHE_ARGS+=(--cache_timing)
fi
if [[ "$CACHE_RESUME" == "1" ]]; then
  CACHE_ARGS+=(--cache_resume)
fi
if [[ "$CACHE_SKIP_MISMATCHED_SHAPES" == "1" ]]; then
  CACHE_ARGS+=(--cache_skip_mismatched_shapes)
fi

launch_train() {
  if [[ "$LAUNCH_MODE" == "multi" ]]; then
    accelerate launch --num_processes "$NUM_PROCESSES" --num_machines 1 --mixed_precision "$MIXED_PRECISION" --dynamo_backend no DiffSynth-Studio/examples/wanvideo/model_training/train.py "$@"
  else
    accelerate launch --num_processes 1 --num_machines 1 --mixed_precision "$MIXED_PRECISION" --dynamo_backend no DiffSynth-Studio/examples/wanvideo/model_training/train.py "$@"
  fi
}

run_cache() {
  launch_train \
    --dataset_base_path "$DATASET_BASE_PATH" \
    --dataset_metadata_path "$DATASET_METADATA_PATH" \
    --dataset_repeat 1 \
    --output_path "$CACHE_DIR" \
    --task "sft:data_process" \
    --offload_models "${PRETRAINED_MODEL_DIR}:diffusion_pytorch_model*.safetensors" \
    "${COMMON_ARGS[@]}" \
    "${CACHE_ARGS[@]}"
}

run_train() {
  if [[ "$USE_CACHE" == "1" ]]; then
    launch_train \
      --dataset_base_path "$CACHE_DIR" \
      --task "sft:train" \
      "${COMMON_ARGS[@]}" \
      "${TRAIN_ARGS[@]}"
  else
    launch_train \
      --dataset_base_path "$DATASET_BASE_PATH" \
      --dataset_metadata_path "$DATASET_METADATA_PATH" \
      --task "sft" \
      "${COMMON_ARGS[@]}" \
      "${TRAIN_ARGS[@]}"
  fi
}

case "$MODE" in
  cache) run_cache ;;
  train) run_train ;;
  both) run_cache; run_train ;;
  *) echo "Unsupported MODE: $MODE" >&2; exit 1 ;;
esac

echo "Experiment output: $OUTPUT_PATH"
echo "Condition cache: $CACHE_DIR"
echo "Cache batch per process: $CACHE_BATCH_SIZE"
echo "Effective global batch: $((DATASET_BATCH_SIZE * WORLD_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "Training rows: $TRAIN_LEN"
