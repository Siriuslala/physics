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
GPU_IDS="2"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
GPU_TAG=a800

BENCHMARK_NAME=test  # test | videophy2 | phygenbench
MODEL_TYPE=1.3B
MODEL_PATH=
BATCH_SIZE=1
BASE_SEED=26

SIZE=832*480
FRAME_NUM=81
SAMPLE_SOLVER=unipc
SAMPLE_STEPS=50
SAMPLE_SHIFT=5.0
SAMPLE_GUIDE_SCALE=5.0
OFFLOAD_MODEL=1
NEGATIVE_PROMPT=
SKIP_EXISTING=1
T5_CPU=0

# benchmark_name -> input jsonl
# Add new benchmarks here and then switch BENCHMARK_NAME above.
declare -A BENCHMARK_TO_INPUT_JSONL
BENCHMARK_TO_INPUT_JSONL[test]="$ROOT_DIR/wan_eval/datasets/test/prompts.jsonl"
BENCHMARK_TO_INPUT_JSONL[videophy2]="$ROOT_DIR/wan_eval/datasets/videophy2/prompts.jsonl"
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

if [[ -z "$MODEL_PATH" ]]; then
  MODEL_NAME=$DEFAULT_MODEL_NAME
else
  if [[ -d "$MODEL_PATH" ]]; then
    MODEL_NAME="$(basename "$MODEL_PATH")"
  else
    MODEL_NAME="$(basename "$(dirname "$MODEL_PATH")")"
  fi
fi

OUTPUT_DIR=$WORK_DIR/wan_eval/$BENCHMARK_NAME/$GPU_TAG/$MODEL_NAME
mkdir -p "$OUTPUT_DIR"

cd $ROOT_DIR

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

if [[ -n "$MODEL_PATH" ]]; then
  ARGS+=(--model_path "$MODEL_PATH")
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
printf 'Model Path: %s\n' "${MODEL_PATH:-<original>}"
printf 'Result JSONL: %s\n' "$OUTPUT_DIR/$(basename "$INPUT_JSONL" .jsonl)_with_video_url.jsonl"
printf 'LoRA/Lambda Config Source: %s\n' "${MODEL_PATH:-official base model}"

"${ARGS[@]}"
