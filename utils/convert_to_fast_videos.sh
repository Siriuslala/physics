#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source ../scripts/env.sh
cd $ROOT_DIR/projects/Wan2_1

set +u
source ~/miniforge3/etc/profile.d/conda.sh
conda activate video
set -u

USER_DIR=$(dirname "$(dirname "$ROOT_DIR")")
echo $USER_DIR
FFMPEG=$USER_DIR/.conda/envs/video/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2

SRC_DIR="$WORK_DIR/wan_eval/videophy/a800/42/bsz_32-lr_1e-4-lora_rank_64-lora_alpha_32-lora_modules_attn-steps_1500-warmup_0.03-adam_beta1_0.9-beta2_0.999-timestep_uniform_0.0_1.0-seed_42/ckpt_step_800"
DST_DIR="$WORK_DIR/wan_eval/videophy/a800/42/bsz_32-lr_1e-4-lora_rank_64-lora_alpha_32-lora_modules_attn-steps_1500-warmup_0.03-adam_beta1_0.9-beta2_0.999-timestep_uniform_0.0_1.0-seed_42/_ckpt_step_800"
mkdir -p "$DST_DIR"

for f in "$SRC_DIR"/*.mp4; do
  name="$(basename "$f")"
  "$FFMPEG" -y -i "$f" -c copy -movflags +faststart "$DST_DIR/$name"
done