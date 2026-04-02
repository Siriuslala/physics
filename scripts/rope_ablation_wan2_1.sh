source ./env.sh
cd ../wan21_t2v_experiments

# -- prompts --
PROMPTS=(
    # "A basketball falls to the ground from the air."
    # "A basketball falls to the ground from the air and bounces up."
    "A basketball falls to the ground from the air and bounces up several times."
    # "A basketball falls to the ground from the air and bounces up several times, with the height of each bounce gradually decreasing until it comes to a stop."
)

# -- model settings --
task="t2v-1.3B"
CKPT_DIR="$MODEL_DIR/Wan2.1-T2V-1.3B"

# task="t2v-14B"
# CKPT_DIR="$MODEL_DIR/Wan2.1-T2V-14B"

# -- gpu settings --
NPROC_PER_NODE=2
export CUDA_VISIBLE_DEVICES=3,4  # 5,7

NPROC_PER_NODE=1
export CUDA_VISIBLE_DEVICES=4

# -- inference settings --
SIZE="832*480"  # 
FRAME_NUM=81
SAMPLE_STEPS=50
SAMPLE_SHIFT=5.0
SAMPLE_GUIDE_SCALE=5.0
SEED=14

# start inference
for PROMPT in "${PROMPTS[@]}"; do
    echo "=================================================================================="
    echo "Starting inference for prompt: $PROMPT"
    echo "=================================================================================="

    SAVE_DIR="$WORK_DIR/outputs_wan_2_1_${task}/rope_axis_ablation/$(echo $PROMPT | tr ' ' '_')_seed_${SEED}_shift_${SAMPLE_SHIFT}_guide_${SAMPLE_GUIDE_SCALE}"

    # single
    python run_wan21_t2v_experiments.py \
        --experiment rope_axis_ablation \
        --ckpt_dir "$CKPT_DIR" \
        --output_dir $SAVE_DIR \
        --task $task \
        --prompt "$PROMPT" \
        --size $SIZE \
        --frame_num $FRAME_NUM \
        --sampling_steps $SAMPLE_STEPS \
        --shift $SAMPLE_SHIFT \
        --guide_scale $SAMPLE_GUIDE_SCALE \
        --seed $SEED \
        --rope_modes full,no_f

    # # parallel
    # PORT_NUMBER=$((20000 + RANDOM % 10000))
    # torchrun --nproc_per_node=$NPROC_PER_NODE --master_port $PORT_NUMBER run_wan21_t2v_experiments.py \
    #     --experiment rope_axis_ablation \
    #     --ckpt_dir "$CKPT_DIR" \
    #     --output_dir $SAVE_DIR \
    #     --task $task \
    #     --prompt "$PROMPT" \
    #     --size $SIZE \
    #     --frame_num $FRAME_NUM \
    #     --sampling_steps $SAMPLE_STEPS \
    #     --shift $SAMPLE_SHIFT \
    #     --guide_scale $SAMPLE_GUIDE_SCALE \
    #     --seed $SEED \
    #     --dit_fsdp \
    #     --t5_cpu \
    #     --use_usp \
    #     --ulysses_size $NPROC_PER_NODE \
    #     --rope_modes full,no_f,no_h,no_w,only_f

    echo "Finished inference for prompt: $PROMPT"
    echo ""
done
