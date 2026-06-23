#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source ../../scripts/env.sh
cd $ROOT_DIR/projects/Wan2_1

set +u
source ~/miniforge3/etc/profile.d/conda.sh
conda activate video
set -u

# infer settings
GPU_IDS=2
export CUDA_VISIBLE_DEVICES=$GPU_IDS
GPU_TAG=a800

# SEEDS=(1 8 20 23 29)
SEEDS=(42)
BATCH_SIZE=4

BENCHMARK_NAME=videophy2_rewrite  # test | videophy2 | videophy2_rewrite | phygenbench
MODEL_TYPE=1.3B
MODEL_PATH=/work/liyueyan/Interpretability/physics_train/Wan2.1-T2V-1B3/lambda_lora/lambda_lora-bsz_16-lr_1e-4-lora_rank_32-lora_alpha_32-lora_modules_attn-lambda_scope_head-lambda_lr_5e-4-lambda_beta_1e-4-lambda_tcond_1-lambda_hidden_128-steps_3000-warmup_0.03-adam_beta1_0.9-beta2_0.999-timestep_mixed_early_0.14_prob_0.60-seed_42
CKPT_STEP=2000
MODEL_PATH=
CKPT_STEP=

# dir
# /work/liyueyan/Interpretability/physics_train/Wan2.1-T2V-1B3/lora
# /work/liyueyan/Interpretability/physics_train/Wan2.1-T2V-1B3/lambda_lora
# model_path
# bsz_16-lr_1e-4-lora_rank_32-lora_alpha_32-lora_modules_attn-steps_3000-warmup_0.03-adam_beta1_0.9-beta2_0.999-timestep_0.0_1.0-seed_42
# lambda_lora-bsz_16-lr_1e-4-lora_rank_32-lora_alpha_32-lora_modules_attn-lambda_scope_head-lambda_lr_5e-4-lambda_beta_1e-4-lambda_tcond_1-lambda_hidden_128-steps_3000-warmup_0.03-adam_beta1_0.9-beta2_0.999-timestep_mixed_early_0.14_prob_0.60-seed_42


SIZE=832*480
FRAME_NUM=81  # 81
SAMPLE_SOLVER=unipc
SAMPLE_STEPS=50
SAMPLE_SHIFT=5.0
SAMPLE_GUIDE_SCALE=5.0
OFFLOAD_MODEL=1
NEGATIVE_PROMPT=
SKIP_EXISTING=1
T5_CPU=0

# benchmark_name -> input jsonl
declare -A BENCHMARK_TO_INPUT_JSONL
BENCHMARK_TO_INPUT_JSONL[test]="$ROOT_DIR/wan_eval/datasets/test/prompts.jsonl"
BENCHMARK_TO_INPUT_JSONL[videophy2]="$ROOT_DIR/wan_eval/datasets/videophy2/prompts.jsonl"
BENCHMARK_TO_INPUT_JSONL[videophy2_rewrite]="$ROOT_DIR/wan_eval/datasets/videophy2_rewrite/prompts.jsonl"
BENCHMARK_TO_INPUT_JSONL[phygenbench]="$ROOT_DIR/wan_eval/datasets/phygenbench/prompts.jsonl"

INPUT_JSONL="${BENCHMARK_TO_INPUT_JSONL[$BENCHMARK_NAME]:-}"
if [[ -z "$INPUT_JSONL" ]]; then
  echo "Unsupported BENCHMARK_NAME: $BENCHMARK_NAME" >&2
  exit 1
fi
if [[ ! -f "$INPUT_JSONL" ]]; then
  echo "Input JSONL does not exist: $INPUT_JSONL" >&2
  exit 1
fi

if [[ "$MODEL_TYPE" == "1.3B" || "$MODEL_TYPE" == "1B3" ]]; then
  TASK=t2v-1.3B
  CKPT_DIR=$MODEL_DIR/Wan2.1-T2V-1.3B
  DEFAULT_MODEL_NAME=wan2.1_1B3
elif [[ "$MODEL_TYPE" == "14B" ]]; then
  TASK=t2v-14B
  CKPT_DIR=$MODEL_DIR/Wan2.1-T2V-14B
  DEFAULT_MODEL_NAME=wan2.1_14B
else
  echo "Unsupported MODEL_TYPE: $MODEL_TYPE" >&2
  exit 1
fi

RESOLVED_MODEL_PATH=$MODEL_PATH
if [[ -n "$MODEL_PATH" && -n "$CKPT_STEP" && -d "$MODEL_PATH" ]]; then
  STEP_MODEL_PATH=$MODEL_PATH/step-$CKPT_STEP.safetensors
  EPOCH_MODEL_PATH=$MODEL_PATH/epoch-$CKPT_STEP.safetensors
  if [[ -f "$STEP_MODEL_PATH" ]]; then
    RESOLVED_MODEL_PATH=$STEP_MODEL_PATH
  elif [[ -f "$EPOCH_MODEL_PATH" ]]; then
    RESOLVED_MODEL_PATH=$EPOCH_MODEL_PATH
  else
    echo "Cannot find checkpoint for CKPT_STEP=$CKPT_STEP under $MODEL_PATH" >&2
    exit 1
  fi
fi

if [[ -n "$MODEL_PATH" && -n "$CKPT_STEP" && ! -d "$MODEL_PATH" ]]; then
  echo "MODEL_PATH is already a file, so CKPT_STEP is ignored: $MODEL_PATH" >&2
fi

if [[ -z "$RESOLVED_MODEL_PATH" ]]; then
  MODEL_NAME=$DEFAULT_MODEL_NAME
else
  if [[ -d "$MODEL_PATH" ]]; then
    MODEL_NAME="$(basename "$MODEL_PATH")"
  else
    MODEL_NAME="$(basename "$(dirname "$RESOLVED_MODEL_PATH")")"
  fi
fi

cd $ROOT_DIR

if [[ -n "$CKPT_STEP" ]]; then
  CKPT_STEP_TAG=ckpt_step_$CKPT_STEP
else
  CKPT_STEP_TAG=ckpt_step_latest
fi

for BASE_SEED in "${SEEDS[@]}"; do
  OUTPUT_DIR=$WORK_DIR/wan_eval/$BENCHMARK_NAME/$GPU_TAG/$BASE_SEED/$MODEL_NAME/$CKPT_STEP_TAG
  mkdir -p "$OUTPUT_DIR"

  ARGS=(
    python wan_eval/infer_wan_t2v.py
    --input_jsonl "$INPUT_JSONL"
    --output_dir "$OUTPUT_DIR"
    --ckpt_dir "$CKPT_DIR"
    --task "$TASK"
    --size "$SIZE"
    --frame_num "$FRAME_NUM"
    --sample_solver "$SAMPLE_SOLVER"
    --sample_steps "$SAMPLE_STEPS"
    --sample_shift "$SAMPLE_SHIFT"
    --sample_guide_scale "$SAMPLE_GUIDE_SCALE"
    --batch_size "$BATCH_SIZE"
    --base_seed "$BASE_SEED"
    --gpu_ids "$GPU_IDS"
  )

  if [[ -n "$RESOLVED_MODEL_PATH" ]]; then
    ARGS+=(--model_path "$RESOLVED_MODEL_PATH")
  fi
  if [[ -n "$NEGATIVE_PROMPT" ]]; then
    ARGS+=(--negative_prompt "$NEGATIVE_PROMPT")
  fi
  if [[ "$OFFLOAD_MODEL" == "1" ]]; then
    ARGS+=(--offload_model)
  else
    ARGS+=(--no_offload_model)
  fi
  if [[ "$SKIP_EXISTING" == "1" ]]; then
    ARGS+=(--skip_existing)
  fi
  if [[ "$T5_CPU" == "1" ]]; then
    ARGS+=(--t5_cpu)
  fi

  printf 'Benchmark: %s\n' "$BENCHMARK_NAME"
  printf 'Input JSONL: %s\n' "$INPUT_JSONL"
  printf 'Output Dir: %s\n' "$OUTPUT_DIR"
  printf 'Task: %s\n' "$TASK"
  printf 'Visible GPUs: %s\n' "$GPU_IDS"
  printf 'Batch Size Per GPU Worker: %s\n' "$BATCH_SIZE"
  printf 'Base Seed: %s\n' "$BASE_SEED"
  printf 'Model Path: %s\n' "${MODEL_PATH:-<original>}"
  printf 'CKPT_STEP: %s\n' "${CKPT_STEP:-<latest>}"
  printf 'Resolved Model Path: %s\n' "${RESOLVED_MODEL_PATH:-<original>}"
  printf 'Result JSONL: %s\n' "$OUTPUT_DIR/$(basename "$INPUT_JSONL" .jsonl)_with_video_url.jsonl"
  printf 'LoRA/Lambda Config Source: %s\n' "${RESOLVED_MODEL_PATH:-official base model}"

  "${ARGS[@]}"
done
