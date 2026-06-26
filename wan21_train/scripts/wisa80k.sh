# data processing for wisa80k

cd "$(dirname "${BASH_SOURCE[0]}")"
source ../../scripts/env.sh
cd $ROOT_DIR

source ~/miniforge3/etc/profile.d/conda.sh
conda activate video

# ============================================================ Unzip and get metadata.csv
# python ${ROOT_DIR}/wan21_train/prepare_wisa80k_for_diffsynth.py \
#   --extract \
#   --build-metadata \
#   --metadata-mode all \
#   --metadata-out /datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/video_data/metadata.csv
#   --num-workers 8

# ============================================================ Get meta info for dynamics, thermodynamics, and optics.
# This writes metadata_dynamics.csv, metadata_thermodynamics.csv, and metadata_optics.csv.
# python ${ROOT_DIR}/wan21_train/prepare_wisa80k_for_diffsynth.py \
#   --build-metadata \
#   --metadata-mode categories \
#   --metadata-out /datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/video_data/physics_metadata_new \
#   --min-duration 2.0 \
#   --video-check off \
#   --video-check-workers 32

# ============================================================ Jsonl for viz
SEED=28
python ${ROOT_DIR}/wan21_train/prepare_wisa80k_for_diffsynth.py \
  --sample-csv /datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/video_data/physics_metadata_new/metadata_optics.csv \
  --sample-n 40 \
  --sample-random \
  --sample-seed $SEED \
  --sample-out /datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/video_data/viz/wisa_sample_videos_seed${SEED}/metadata_sample.jsonl \
  --sample-copy-dir /datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/video_data/viz/wisa_sample_videos_seed${SEED} \
  --sample-base-dir /datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/video_data

# ============================================================ Merge 3 classes
# python ${ROOT_DIR}/wan21_train/prepare_wisa80k_for_diffsynth.py \
#   --merge-category-metadata /datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/video_data/physics_metadata_new \
#   --merge-out /datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/video_data/physics_metadata_new/metadata_physics_merged_reflection4000.csv \
#   --reflection-sample-n 4000 \
#   --merge-seed 0
