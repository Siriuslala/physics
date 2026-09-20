#!/usr/bin/env bash
set -euo pipefail

# Run in an environment with imageio-ffmpeg and Pillow; no model or GPU is needed.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${1:-/work/liyueyan/Interpretability/physics/viz}"
python "${ROOT_DIR}/wan21_t2v_experiments/prepare_project_media.py" --source "${SOURCE_DIR}"
