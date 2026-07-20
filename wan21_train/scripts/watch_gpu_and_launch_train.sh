#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

TARGET_GPU_INDEX=3  # ！
MEMORY_THRESHOLD_MIB=50000
LOG_ENABLED=0  # ！
MODE="eval"  # train | eval
TRAIN_SCRIPT="$(pwd)/train_wan21_t2v_1b3_fixed_lambda_lora.sh"
EVAL_SCRIPT="$(pwd)/../../wan_eval/scripts/infer_eval_2.1.sh"


REQUIRED_IDLE_SECONDS=60
POLL_INTERVAL_SECONDS=5
LOG_PATH="$(pwd)/logs/watch_gpu_and_launch_train_wan21_t2v_1b3.log"

if [[ "$LOG_ENABLED" == "1" ]]; then
  mkdir -p "$(dirname "$LOG_PATH")"
fi

log() {
  local timestamp
  timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
  if [[ "$LOG_ENABLED" == "1" ]]; then
    echo "[$timestamp] $*" | tee -a "$LOG_PATH"
  else
    echo "[$timestamp] $*"
  fi
}

require_command() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    log "Missing required command: $cmd"
    exit 1
  fi
}

get_gpu_memory_used_mib() {
  nvidia-smi \
    --query-gpu=memory.used \
    --format=csv,noheader,nounits \
    -i "$TARGET_GPU_INDEX" \
    | tr -d '[:space:]'
}

resolve_target_script() {
  case "$MODE" in
    train)
      echo "$TRAIN_SCRIPT"
      ;;
    eval)
      echo "$EVAL_SCRIPT"
      ;;
    *)
      log "Unsupported MODE: $MODE. Expected one of: train, eval"
      exit 1
      ;;
  esac
}

TARGET_SCRIPT="$(resolve_target_script)"

is_target_script_running() {
  pgrep -f "$TARGET_SCRIPT" >/dev/null 2>&1
}

launch_target_script() {
  if is_target_script_running; then
    log "Target script is already running: $TARGET_SCRIPT"
    exit 0
  fi

  log "GPU ${TARGET_GPU_INDEX} stayed below ${MEMORY_THRESHOLD_MIB} MiB for ${REQUIRED_IDLE_SECONDS} seconds. Launching mode=${MODE}."
  log "Replacing watcher process with target script in the current terminal: $TARGET_SCRIPT"
  exec bash "$TARGET_SCRIPT"
}

require_command nvidia-smi
require_command pgrep

if [[ ! -f "$TARGET_SCRIPT" ]]; then
  log "Target script not found for MODE=${MODE}: $TARGET_SCRIPT"
  exit 1
fi

if (( POLL_INTERVAL_SECONDS <= 0 )); then
  log "POLL_INTERVAL_SECONDS must be positive, got ${POLL_INTERVAL_SECONDS}"
  exit 1
fi

if (( REQUIRED_IDLE_SECONDS <= 0 )); then
  log "REQUIRED_IDLE_SECONDS must be positive, got ${REQUIRED_IDLE_SECONDS}"
  exit 1
fi

if (( MEMORY_THRESHOLD_MIB < 0 )); then
  log "MEMORY_THRESHOLD_MIB must be non-negative, got ${MEMORY_THRESHOLD_MIB}"
  exit 1
fi

idle_seconds=0

log "Start watching GPU ${TARGET_GPU_INDEX}. Threshold=${MEMORY_THRESHOLD_MIB} MiB, required_idle=${REQUIRED_IDLE_SECONDS}s, poll_interval=${POLL_INTERVAL_SECONDS}s."
log "Mode: $MODE"
log "Target script: $TARGET_SCRIPT"

while true; do
  if is_target_script_running; then
    log "Detected existing running target script. Exiting watcher without launching a duplicate job."
    exit 0
  fi

  memory_used_mib="$(get_gpu_memory_used_mib || true)"
  if [[ ! "$memory_used_mib" =~ ^[0-9]+$ ]]; then
    log "Failed to read GPU ${TARGET_GPU_INDEX} memory usage. Raw output: ${memory_used_mib:-<empty>}. Retrying after ${POLL_INTERVAL_SECONDS}s."
    sleep "$POLL_INTERVAL_SECONDS"
    continue
  fi

  if (( memory_used_mib < MEMORY_THRESHOLD_MIB )); then
    idle_seconds=$((idle_seconds + POLL_INTERVAL_SECONDS))
    log "GPU ${TARGET_GPU_INDEX} memory used=${memory_used_mib} MiB < ${MEMORY_THRESHOLD_MIB} MiB. Idle countdown: ${idle_seconds}/${REQUIRED_IDLE_SECONDS}s."
    if (( idle_seconds >= REQUIRED_IDLE_SECONDS )); then
      launch_target_script
      exit 0
    fi
  else
    if (( idle_seconds > 0 )); then
      log "GPU ${TARGET_GPU_INDEX} memory bounced back to ${memory_used_mib} MiB. Reset idle countdown."
    else
      log "GPU ${TARGET_GPU_INDEX} memory used=${memory_used_mib} MiB. Waiting."
    fi
    idle_seconds=0
  fi

  sleep "$POLL_INTERVAL_SECONDS"
done
