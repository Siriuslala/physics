#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

TARGET_GPU_INDEX=3  # ！
LOG_ENABLED=0  # ！
TRAIN_SCRIPT="$(pwd)/train_wan21_t2v_1b3_lambda_lora.sh"

MEMORY_THRESHOLD_MIB=10000
REQUIRED_IDLE_SECONDS=60
POLL_INTERVAL_SECONDS=5
LOG_PATH="$(pwd)/logs/watch_gpu3_and_launch_train_wan21_t2v_1b3_lambda_lora.log"

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

is_train_script_running() {
  pgrep -f "$TRAIN_SCRIPT" >/dev/null 2>&1
}

launch_train() {
  if is_train_script_running; then
    log "Training script is already running: $TRAIN_SCRIPT"
    exit 0
  fi

  log "GPU ${TARGET_GPU_INDEX} stayed below ${MEMORY_THRESHOLD_MIB} MiB for ${REQUIRED_IDLE_SECONDS} seconds. Launching training."
  log "Replacing watcher process with training script in the current terminal."
  exec bash "$TRAIN_SCRIPT"
}

require_command nvidia-smi
require_command pgrep

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  log "Training script not found: $TRAIN_SCRIPT"
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
log "Target training script: $TRAIN_SCRIPT"

while true; do
  if is_train_script_running; then
    log "Detected existing running training script. Exiting watcher without launching a duplicate job."
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
      launch_train
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
