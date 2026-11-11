#!/usr/bin/env bash
# Render only basketball maps, using the baseline ten-frame sampling configuration.
set -eo pipefail
cd /home/liyueyan/Interpretability/physics
source ~/miniforge3/etc/profile.d/conda.sh
conda activate video
set -u
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -u -m wan21_t2v_experiments.redraw_cross_attention \
  --directory '/work/liyueyan/Interpretability/physics/outputs_wan_2_1_t2v-1.3B/rope_posthoc_attention/Against_a_pure_white_background,_a_basketball_falls_vertically_from_mid-air_onto_a_wooden_floor_and_bounces_up_several_times./modified/seed_20/cross_attention' \
  --tokens basketball --num-viz-frames 10 --workers 8
