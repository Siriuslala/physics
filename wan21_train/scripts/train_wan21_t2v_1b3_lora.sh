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
export CUDA_VISIBLE_DEVICES=0
NUM_PROCESSES=1

# export CUDA_VISIBLE_DEVICES=0,1
# NUM_PROCESSES=2

if [[ "$NUM_PROCESSES" -gt 1 ]]; then
  LAUNCH_MODE=multi
else
  LAUNCH_MODE=single
fi


# Experiment mode ==============================
MODE=cache  # cache | train | both

# General settings [for cache] ============================== 
COMPACT_CACHE=1  # save only compact Wan T2V cache fields during MODE=cache
CACHE_BATCH_SIZE=16  # per-process batch size for MODE=cache
ENABLE_CACHE_TIMING=1  # print text/video/VAE timing during MODE=cache
VIDEO_OUTPUT_FORMAT=tensor_uint8  # pil | tensor_uint8
VAE_ENCODE_BATCH_SIZE=8  # 0 means encode the full cache batch at once
CACHE_PREFETCH_BATCHES=2  # overlap video preprocessing with VAE/cache saving; 0 disables it
CACHE_RESUME=1  # single-process only: skip existing continuous cache files

# General settings [for train] ============================== 
USE_CACHE=0  # 0: online video/text encoding; 1: train from cached .pth files
MODEL_SIZE=1.3B  # 1.3B | 14B
MIXED_PRECISION=bf16

DATASET_BASE_PATH=/datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/video_data
DATASET_METADATA_PATH=/datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/video_data/physics_metadata_new/metadata_physics_merged_reflection4000.csv
DATASET_NUM_WORKERS=8
DATASET_REPEAT=1

HEIGHT=480
WIDTH=832
NUM_FRAMES=81
VIDEO_CLIP_SECONDS=5.0

# Hyper-params settings ============================== 
# With ENABLE_BATCHED_SFT=1, standard Wan T2V SFT uses DATASET_BATCH_SIZE as real per-process batch size.
# Effective global batch = DATASET_BATCH_SIZE * NUM_PROCESSES * GRADIENT_ACCUMULATION_STEPS.
GLOBAL_BATCH_SIZE=16
DATASET_BATCH_SIZE=4  # batchsize per forward
ENABLE_BATCHED_SFT=1
if [[ "$NUM_PROCESSES" -gt 1 ]]; then
  WORLD_SIZE=$NUM_PROCESSES
else
  WORLD_SIZE=1
fi
GRADIENT_ACCUMULATION_STEPS=$(( (GLOBAL_BATCH_SIZE + DATASET_BATCH_SIZE * WORLD_SIZE - 1) / (DATASET_BATCH_SIZE * WORLD_SIZE) ))

LEARNING_RATE=1e-4
LR_SCHEDULER=constant_with_warmup
WARMUP_RATIO=0.03
ADAM_BETA1=0.9
ADAM_BETA2=0.999
ADAM_EPSILON=1e-8
WEIGHT_DECAY=0.0
MAX_GRAD_NORM=1.0
NUM_EPOCHS=3
MAX_TRAIN_STEPS=3000
SAVE_STEPS=500

LORA_RANK=32
LORA_ALPHA=32
LORA_MODULE_PRESET=attn_ffn  # attn | ffn | attn_ffn
case "$LORA_MODULE_PRESET" in
  attn) LORA_TARGET_MODULES=q,k,v,o ;;
  ffn) LORA_TARGET_MODULES=ffn.0,ffn.2 ;;
  attn_ffn) LORA_TARGET_MODULES=q,k,v,o,ffn.0,ffn.2 ;;
  *) echo "Unsupported LORA_MODULE_PRESET: $LORA_MODULE_PRESET" >&2; exit 1 ;;
esac

ENABLE_WANDB=1
WANDB_PROJECT=wan21-physics-lora
WANDB_NAME="bsz_${GLOBAL_BATCH_SIZE}-lr_${LEARNING_RATE}-lora_rank_${LORA_RANK}-lora_alpha_${LORA_ALPHA}-lora_modules_${LORA_MODULE_PRESET}-warmup_${WARMUP_RATIO}-adam_beta1_${ADAM_BETA1}-beta2_${ADAM_BETA2}"

# Model path & Output settings ==============================
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
CACHE_DIR=/datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/wan_cache/Wan2.1-T2V-${MODEL_SIZE}/physics_81f_5s
if [[ "$COMPACT_CACHE" == "1" ]]; then
  CACHE_DIR=/datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/wan_cache/Wan2.1-T2V-${MODEL_SIZE}/physics_81f_5s_compact
fi
TRAIN_ROOT="$CKPT_ROOT/lora"

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
EXP_NAME="bsz_${GLOBAL_BATCH_SIZE}-lr_${LEARNING_RATE}-lora_rank_${LORA_RANK}-lora_alpha_${LORA_ALPHA}-lora_modules_${LORA_MODULE_PRESET}-${TRAIN_TOKEN}-warmup_${WARMUP_RATIO}-adam_beta1_${ADAM_BETA1}-beta2_${ADAM_BETA2}"
OUTPUT_PATH=$TRAIN_ROOT/$EXP_NAME
mkdir -p "$CACHE_DIR" "$OUTPUT_PATH"

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
  --lora_base_model "dit"
  --lora_target_modules "$LORA_TARGET_MODULES"
  --lora_rank "$LORA_RANK"
  --lora_alpha "$LORA_ALPHA"
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

# Prepare commands ============================== 
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

# Start ============================== 
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
