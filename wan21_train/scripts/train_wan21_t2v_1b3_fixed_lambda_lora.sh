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
export CUDA_VISIBLE_DEVICES=2
NUM_PROCESSES=1

# export CUDA_VISIBLE_DEVICES=5,6
# NUM_PROCESSES=2

if [[ "$NUM_PROCESSES" -gt 1 ]]; then LAUNCH_MODE=multi; else LAUNCH_MODE=single; fi


# Experiment mode ==============================
MODE=train  # cache | train | both

# General settings [for cache] ============================== 
COMPACT_CACHE=1  # save only compact Wan T2V cache fields during MODE=cache
CACHE_BATCH_SIZE=16  # per-process batch size for MODE=cache
ENABLE_CACHE_TIMING=1  # print text/video/VAE timing during MODE=cache
VIDEO_OUTPUT_FORMAT=tensor_uint8  # pil | tensor_uint8
VAE_ENCODE_BATCH_SIZE=4  # 0 means encode the full cache batch at once
CACHE_PREFETCH_BATCHES=2  # overlap video preprocessing with VAE/cache saving; 0 disables it
CACHE_RESUME=1  # single-process only: skip existing continuous cache files
CACHE_SKIP_MISMATCHED_SHAPES=1  # skip decoded videos whose shape is not NUM_FRAMES x HEIGHT x WIDTH

# General settings [for train] ============================== 
USE_CACHE=1  # 0: online video/text encoding; 1: train from cached .pth files
MODEL_SIZE=1.3B  # 1.3B | 14B
MIXED_PRECISION=bf16
SEED=42
DETERMINISTIC=0  # 1 enables deterministic PyTorch algorithms when available; slower and may warn.
FIND_UNUSED_PARAMETERS=0  # Set to 1 if DDP reports unused trainable parameters.
USE_GRADIENT_CHECKPOINTING=1
USE_GRADIENT_CHECKPOINTING_OFFLOAD=0  # use gradient checkpointing, but whether to offload it to CPU
NUM_CPU_THREADS_PER_PROCESS=8
ENABLE_CPU_AFFINITY=0  # Set to 1 to let accelerate bind CPU cores per process when supported.

DATASET_BASE_PATH=/datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/video_data
DATASET_METADATA_PATH=/datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/video_data/physics_metadata_new/metadata_physics_merged_reflection4000.csv
DATASET_NUM_WORKERS=4
DATASET_REPEAT=1
CACHE_FILTER_INVALID_LATENTS=1
CACHE_FILTER_INVALID_LATENTS_REBUILD_INDEX=0

HEIGHT=480
WIDTH=832
NUM_FRAMES=81
VIDEO_CLIP_SECONDS=5.0

# Hyper-params settings ============================== 
# With ENABLE_BATCHED_SFT=1, standard Wan T2V SFT uses DATASET_BATCH_SIZE as real per-process batch size.
# Effective global batch = DATASET_BATCH_SIZE * NUM_PROCESSES * GRADIENT_ACCUMULATION_STEPS.
GLOBAL_BATCH_SIZE=32
DATASET_BATCH_SIZE=8  # batchsize per forward
ENABLE_BATCHED_SFT=1
if [[ "$NUM_PROCESSES" -gt 1 ]]; then WORLD_SIZE=$NUM_PROCESSES; else WORLD_SIZE=1; fi
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
MAX_TRAIN_STEPS=1500
SAVE_STEPS=100
SAVE_TRAINING_STATE=1
RESUME_TRAINING=0  # 0 | auto | /path/to/training_state_latest

# Timestep sampling settings ==============================
MIN_TIMESTEP_BOUNDARY=0.0
MAX_TIMESTEP_BOUNDARY=1.0
TIMESTEP_SAMPLING_STRATEGY=early_rest_mixture  # uniform | early_rest_mixture
TIMESTEP_MIXTURE_EARLY_BOUNDARY=0.1
TIMESTEP_MIXTURE_EARLY_PROB=0.9


# Fixed manual spatial RoPE lambda settings ==============================
# This mode keeps lambda frozen and trains LoRA to adapt to the manual RoPE bias.
FIXED_LAMBDA_SCOPE=model  # ignore this
FIXED_LAMBDA_H=0.75
FIXED_LAMBDA_W=0.75
FIXED_LAMBDA_SCHEDULE=cosine  # constant | linear | cosine. cosine anneals from 1 to the target lambda.
FIXED_LAMBDA_SCHEDULE_STEPS=0  # 0 means use MAX_TRAIN_STEPS / total update steps.
FIXED_LAMBDA_GLOBAL=0  # 1: apply lambda on all training timesteps; 0: only apply it in the early timestep boundary.

LORA_RANK=64
LORA_ALPHA=32
LORA_MODULE_PRESET=attn  # attn | ffn | attn_ffn
case "$LORA_MODULE_PRESET" in
  attn) LORA_TARGET_MODULES=q,k,v,o ;;
  ffn) LORA_TARGET_MODULES=ffn.0,ffn.2 ;;
  attn_ffn) LORA_TARGET_MODULES=q,k,v,o,ffn.0,ffn.2 ;;
  *) echo "Unsupported LORA_MODULE_PRESET: $LORA_MODULE_PRESET" >&2; exit 1 ;;
esac

# CPU offload settings ==============================
# Formal training does not use CPU offload unless this is enabled.
ENABLE_MODEL_CPU_OFFLOAD=0
ENABLE_OPTIMIZER_CPU_OFFLOAD=0
CPU_OFFLOAD_SPLIT_THRESHOLD=""  # empty means DiffSynth default

# Tensorboard / wandb settings ==============================
# wandb sync ...
WANDB_MODE=offline  # offline | online | disabled. Offline is syncable later and never depends on api.wandb.ai.
WANDB_ALLOW_ONLINE=0  # set to 1 together with WANDB_MODE=online if online streaming is explicitly needed
WANDB_INIT_TIMEOUT=15
export WANDB_INIT_TIMEOUT
export WANDB_MODE
export WANDB_ALLOW_ONLINE
ENABLE_WANDB=1
WANDB_PROJECT=wan21-physics-lambda-lora

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
CACHE_DIR=/datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/wan_cache/Wan2.1-T2V-${MODEL_SIZE}/physics_81f_5s
if [[ "$COMPACT_CACHE" == "1" ]]; then
  CACHE_DIR=/datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/wan_cache/Wan2.1-T2V-${MODEL_SIZE}/physics_81f_5s_compact
fi
TRAIN_ROOT="$CKPT_ROOT/fixed_lambda_lora"

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
case "$TIMESTEP_SAMPLING_STRATEGY" in
  uniform)
    TIMESTEP_TOKEN="timestep_uniform_${MIN_TIMESTEP_BOUNDARY}_${MAX_TIMESTEP_BOUNDARY}"
    ;;
  early_rest_mixture)
    TIMESTEP_TOKEN="timestep_mixed_early_${TIMESTEP_MIXTURE_EARLY_BOUNDARY}_prob_${TIMESTEP_MIXTURE_EARLY_PROB}"
    ;;
  *)
    echo "Unsupported TIMESTEP_SAMPLING_STRATEGY: $TIMESTEP_SAMPLING_STRATEGY" >&2
    exit 1
    ;;
esac
EXP_NAME="fixed_lambda_lora-bsz_${GLOBAL_BATCH_SIZE}-lr_${LEARNING_RATE}-lora_rank_${LORA_RANK}-lora_alpha_${LORA_ALPHA}-lora_modules_${LORA_MODULE_PRESET}-lambda_scope_${FIXED_LAMBDA_SCOPE}-lambda_h_${FIXED_LAMBDA_H}-lambda_w_${FIXED_LAMBDA_W}-lambda_global_${FIXED_LAMBDA_GLOBAL}-lam_schedule${FIXED_LAMBDA_SCHEDULE}_${FIXED_LAMBDA_SCHEDULE_STEPS}-${TRAIN_TOKEN}-warmup_${WARMUP_RATIO}-${TIMESTEP_TOKEN}-seed_${SEED}"
WANDB_NAME="$EXP_NAME"
WANDB_RUN_ID=
WANDB_RESUME=allow
if [[ -n "$WANDB_RUN_ID" ]]; then export WANDB_RUN_ID; else unset WANDB_RUN_ID; fi
export WANDB_RESUME
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
  --seed "$SEED"
)
if [[ "$USE_GRADIENT_CHECKPOINTING" == "1" ]]; then
  COMMON_ARGS+=(--use_gradient_checkpointing)
fi
if [[ "$USE_GRADIENT_CHECKPOINTING_OFFLOAD" == "1" ]]; then
  COMMON_ARGS+=(--use_gradient_checkpointing_offload)
fi
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
  --wan_spatial_rope_lambda_scope "$FIXED_LAMBDA_SCOPE"
  --wan_spatial_rope_lambda_parametrization fixed
  --wan_spatial_rope_lambda_fixed_h "$FIXED_LAMBDA_H"
  --wan_spatial_rope_lambda_fixed_w "$FIXED_LAMBDA_W"
  --wan_spatial_rope_lambda_fixed_schedule "$FIXED_LAMBDA_SCHEDULE"
  --wan_spatial_rope_lambda_fixed_schedule_steps "$FIXED_LAMBDA_SCHEDULE_STEPS"
)
if [[ "$FIXED_LAMBDA_GLOBAL" == "1" ]]; then
  TRAIN_ARGS+=(--wan_spatial_rope_lambda_global)
else
  TRAIN_ARGS+=(--no-wan_spatial_rope_lambda_global)
fi
if [[ -n "$MAX_TRAIN_STEPS" ]]; then
  TRAIN_ARGS+=(--max_train_steps "$MAX_TRAIN_STEPS")
fi
if [[ "$ENABLE_BATCHED_SFT" == "1" ]]; then
  TRAIN_ARGS+=(--enable_batched_sft)
fi
if [[ "$SAVE_TRAINING_STATE" == "1" ]]; then
  TRAIN_ARGS+=(--save_training_state)
fi
case "$RESUME_TRAINING" in
  0|""|none) ;;
  auto) TRAIN_ARGS+=(--resume_from_latest_state) ;;
  *) TRAIN_ARGS+=(--resume_training_state "$RESUME_TRAINING") ;;
esac
if [[ "$FIND_UNUSED_PARAMETERS" == "1" ]]; then
  TRAIN_ARGS+=(--find_unused_parameters)
fi
if [[ "$ENABLE_MODEL_CPU_OFFLOAD" == "1" ]]; then
  TRAIN_ARGS+=(--enable_model_cpu_offload)
fi
if [[ "$ENABLE_OPTIMIZER_CPU_OFFLOAD" == "1" ]]; then
  TRAIN_ARGS+=(--enable_optimizer_cpu_offload)
fi
if [[ -n "$CPU_OFFLOAD_SPLIT_THRESHOLD" ]]; then
  TRAIN_ARGS+=(--cpu_offload_split_threshold "$CPU_OFFLOAD_SPLIT_THRESHOLD")
fi
if [[ "$CACHE_FILTER_INVALID_LATENTS" == "1" ]]; then
  TRAIN_ARGS+=(--cache_filter_invalid_latents)
else
  TRAIN_ARGS+=(--no-cache_filter_invalid_latents)
fi
if [[ "$CACHE_FILTER_INVALID_LATENTS_REBUILD_INDEX" == "1" ]]; then
  TRAIN_ARGS+=(--cache_filter_invalid_latents_rebuild_index)
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
if [[ "$ENABLE_MODEL_CPU_OFFLOAD" == "1" ]]; then
  CACHE_ARGS+=(--enable_model_cpu_offload)
fi
if [[ "$ENABLE_OPTIMIZER_CPU_OFFLOAD" == "1" ]]; then
  CACHE_ARGS+=(--enable_optimizer_cpu_offload)
fi
if [[ -n "$CPU_OFFLOAD_SPLIT_THRESHOLD" ]]; then
  CACHE_ARGS+=(--cpu_offload_split_threshold "$CPU_OFFLOAD_SPLIT_THRESHOLD")
fi

# Prepare commands ============================== 
launch_train() {
  ACCELERATE_COMMON_ARGS=(
    --num_machines 1
    --mixed_precision "$MIXED_PRECISION"
    --dynamo_backend no
    --main_process_port 0
    --num_cpu_threads_per_process "$NUM_CPU_THREADS_PER_PROCESS"
  )
  if [[ "$ENABLE_CPU_AFFINITY" == "1" ]]; then
    ACCELERATE_COMMON_ARGS+=(--enable_cpu_affinity)
  fi

  if [[ "$LAUNCH_MODE" == "multi" ]]; then
    accelerate launch --multi_gpu --num_processes "$NUM_PROCESSES" "${ACCELERATE_COMMON_ARGS[@]}" DiffSynth-Studio/examples/wanvideo/model_training/train.py "$@"
  else
    accelerate launch --num_processes 1 "${ACCELERATE_COMMON_ARGS[@]}" DiffSynth-Studio/examples/wanvideo/model_training/train.py "$@"
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
