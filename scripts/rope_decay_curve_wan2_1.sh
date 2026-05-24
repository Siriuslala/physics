source ./env.sh
cd $ROOT_DIR

source ~/miniforge3/etc/profile.d/conda.sh
conda activate video

task="t2v-1.3B"
SIZE="832*480"
FRAME_NUM=81

SAVE_DIR="$WORK_DIR/outputs_wan_2_1_${task}/rope_decay_curve/832x480_f${FRAME_NUM}"
mkdir -p "$SAVE_DIR"

python $ROOT_DIR/wan21_t2v_experiments/run_wan21_t2v_experiments.py \
    --experiment rope_decay_curve \
    --wan21_root $ROOT_DIR/projects/Wan2_1 \
    --ckpt_dir "$MODEL_DIR/Wan2.1-T2V-1.3B" \
    --output_dir "$SAVE_DIR" \
    --task $task \
    --prompt "rope decay curve" \
    --size $SIZE \
    --frame_num $FRAME_NUM
