source ./env.sh
cd $ROOT_DIR

# size
# t2v-14B: '720*1280', '1280*720', '480*832', '832*480'
# t2v-1.3B: '480*832', '832*480'

export CUDA_VISIBLE_DEVICES=3

build_prompt_tag() {
    local prompt="$1"
    local name_max
    local cleaned
    local max_len
    name_max=$(getconf NAME_MAX . 2>/dev/null || echo 255)
    max_len="${2:-$name_max}"
    if [ "$max_len" -gt "$name_max" ]; then
        max_len="$name_max"
    fi
    cleaned=$(echo "$prompt" | tr ' ' '_' | tr -cd '[:alnum:]_.,-' | tr -s '_')
    cleaned="${cleaned#_}"
    cleaned="${cleaned%_}"
    if [ ${#cleaned} -gt "$max_len" ]; then
        cleaned="${cleaned:0:$max_len}"
        cleaned="${cleaned%_}"
    fi
    if [ -z "$cleaned" ]; then
        cleaned="prompt"
    fi
    echo "$cleaned"
}

# ==============================
# t2v-1.3B
task="t2v-1.3B"
SIZE="832*480"
CKPT_DIR="$MODEL_DIR/Wan2.1-T2V-1.3B"
FRAME_NUM=81
SAMPLE_STEPS=50
SAMPLE_SHIFT=5.0
SAMPLE_GUIDE_SCALE=5.0

# t2v-14B
# task="t2v-14B"
# SIZE="832*480"
# CKPT_DIR="$MODEL_DIR/Wan2.1-T2V-14B"
# FRAME_NUM=81
# SAMPLE_STEPS=50
# SAMPLE_SHIFT=5.0
# SAMPLE_GUIDE_SCALE=5.0

# Head config ==============================
# Canonical format: LxHy, x/y are 0-based.
# Example: "L29H7,L29H11"

## Black hole heads
BLACK_HOLE_HEADS_1B3_T40="L0H3,L1H9,L3H0,L3H4,L3H10,L5H3,L6H3,L10H1,L12H4,L17H8,L20H8,L21H0,L21H5,L22H8,L23H6,L23H10,L27H7,L28H2,L28H8,L29H6,L29H9"

## Traj heads
TRAJ_HEADS_1B3_T2="L6H9,L10H3,L13H1,L13H8,L16H3,L16H6,L18H10,L23H3,L27H1,L27H3"
TRAJ_HEADS_1B3_T6="L2H2,L2H6,L2H11,L3H7,L3H8,L3H9,L4H3,L4H7,L5H0,L5H1,L5H4,L5H5,L5H6,L5H9,L6H4,L6H6,L6H7,L6H8,L6H9,L7H0,L7H1,L7H2,L7H3,L7H5,L7H9"
TRAJ_HEADS_1B3_T40=""

# Exp config ==============================
SEEDS=(26)  #
PROMPTS=(
    "Against a pure white background, a basketball falls vertically from mid-air onto a wooden floor and bounces up several times."
)  #
HEAD_TYPE="traj"  # "black_hole", "traj"

BLACK_HOLE_HEADS_1B3_SOURCE="BLACK_HOLE_HEADS_1B3_T40"  #
BLACK_HOLE_HEADS_1B3="${!BLACK_HOLE_HEADS_1B3_SOURCE}"
TRAJ_HEADS_1B3_SOURCE="TRAJ_HEADS_1B3_T6"  #
TRAJ_HEADS_1B3="${!TRAJ_HEADS_1B3_SOURCE}"

BLACK_HOLE_HEADS_14B_SOURCE=""  #
BLACK_HOLE_HEADS_14B="${!BLACK_HOLE_HEADS_14B_SOURCE}"
TRAJ_HEADS_14B_SOURCE=""  #
TRAJ_HEADS_14B="${!TRAJ_HEADS_14B_SOURCE}"


# Empty -> ablate on all diffusion steps.
# Example: "1,2,3,5"
HEAD_ABLATION_STEPS=""

# ==============================
if [ "$task" == "t2v-1.3B" ]; then
    if [ "$HEAD_TYPE" == "black_hole" ]; then
        ABLATE_HEADS="$BLACK_HOLE_HEADS_1B3"
        HEAD_TAG="${HEAD_TYPE}_${BLACK_HOLE_HEADS_1B3_SOURCE##*_}"
    elif [ "$HEAD_TYPE" == "traj" ]; then
        ABLATE_HEADS="$TRAJ_HEADS_1B3"
        HEAD_TAG="${HEAD_TYPE}_${TRAJ_HEADS_1B3_SOURCE##*_}"
    fi
elif [ "$task" == "t2v-14B" ]; then
    if [ "$HEAD_TYPE" == "black_hole" ]; then
        ABLATE_HEADS="$BLACK_HOLE_HEADS_14B"
        HEAD_TAG="${HEAD_TYPE}_${BLACK_HOLE_HEADS_14B_SOURCE##*_}"
    elif [ "$HEAD_TYPE" == "traj" ]; then
        ABLATE_HEADS="$TRAJ_HEADS_14B"
        HEAD_TAG="${HEAD_TYPE}_${TRAJ_HEADS_14B_SOURCE##*_}"
    fi
fi

if [ "$HEAD_ABLATION_STEPS" == "" ]; then
    HEAD_ABLATION_STEPS_TAG="all_steps"
else
    # Replace commas with underscores for the tag.
    HEAD_ABLATION_STEPS_TAG=$(echo "$HEAD_ABLATION_STEPS" | tr ',' '_')
fi

# ==============================
for PROMPT in "${PROMPTS[@]}"; do
for SEED in "${SEEDS[@]}"; do
    echo "=================================================================================="
    echo "Running cross_attn_head_ablation | prompt: $PROMPT | seed: $SEED"
    echo "=================================================================================="

    PROMPT_TAG=$(build_prompt_tag "$PROMPT")
    SAVE_DIR="$WORK_DIR/outputs_wan_2_1_${task}/cross_attn_head_ablation/${PROMPT_TAG}/seed_${SEED}_shift_${SAMPLE_SHIFT}_guide_${SAMPLE_GUIDE_SCALE}/ablate_${HEAD_TAG}-steps_${HEAD_ABLATION_STEPS_TAG}"
    SUMMARY_FILE="$SAVE_DIR/cross_attn_head_ablation_summary.json"
    if [ -f "$SUMMARY_FILE" ]; then
        echo "Summary already exists: $SUMMARY_FILE"
        echo "Skip."
        echo ""
        continue
    fi
    mkdir -p "$SAVE_DIR"

    python $ROOT_DIR/wan21_t2v_experiments/run_wan21_t2v_experiments.py \
        --experiment cross_attn_head_ablation \
        --wan21_root $ROOT_DIR/projects/Wan2_1 \
        --ckpt_dir $CKPT_DIR \
        --output_dir $SAVE_DIR \
        --task $task \
        --prompt "$PROMPT" \
        --size $SIZE \
        --frame_num $FRAME_NUM \
        --sample_solver unipc \
        --sampling_steps $SAMPLE_STEPS \
        --shift $SAMPLE_SHIFT \
        --guide_scale $SAMPLE_GUIDE_SCALE \
        --seed $SEED \
        --offload_model True \
        --ablate_heads "$ABLATE_HEADS" \
        --head_ablation_steps "$HEAD_ABLATION_STEPS"

    echo "Finished cross_attn_head_ablation | prompt: $PROMPT | seed: $SEED"
    echo ""
done
done
