source ./env.sh
cd $ROOT_DIR

# size
# t2v-14B: '720*1280', '1280*720', '480*832', '832*480'
# t2v-1.3B: '480*832', '832*480'

export CUDA_VISIBLE_DEVICES=0

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

## Head types
# black_hole: as its name suggests. Some black hole heads have extreamly high attentions on the edge of the object.
# traj: heads that showing the pattern of the trajectory
# motion_plan: heads responsible for motion planning
# others: uniform distribution heads, sink heads, background heads

## Black hole heads
BLACK_HOLE_HEADS_1B3_T40="L0H3,L1H9,L3H0,L3H4,L3H10,L5H3,L6H3,L10H1,L12H4,L17H8,L20H8,L21H0,L21H5,L22H8,L23H6,L23H10,L27H7,L28H2,L28H8,L29H6,L29H9"

## Traj heads
TRAJ_HEADS_1B3_T2="L6H9,L10H3,L13H1,L13H8,L16H3,L16H6,L18H10,L23H3,L27H1,L27H3"
TRAJ_HEADS_1B3_T6="L2H2,L2H6,L2H11,L3H7,L3H8,L3H9,L4H3,L4H7,L5H0,L5H1,L5H4,L5H5,L5H6,L5H9,L6H4,L6H6,L6H7,L6H8,L6H9,L7H0,L7H1,L7H2,L7H3,L7H5,L7H9"
TRAJ_HEADS_1B3_T40=""

TRAJ_HEADS_1B3_T40_no_influence4_4="L0H3,L0H5,L1H9,L2H0,L2H3,L2H5,L2H8,L3H0,L3H4,L3H6,L3H10,L4H0,L4H3,L4H8,L5H0,L5H2,L5H3,L5H8,L5H10,L6H3,L6H10,L7H4,L9H0,L9H3,L9H4,L9H5,L9H8,L10H1,L11H7,L11H8,L11H9,L12H3,L12H4,L12H6,L12H7,L12H8,L13H5,L13H10,L14H1,L14H7,L14H11,L15H1,L15H3,L16H0,L17H1,L17H2,L17H8,L17H10,L18H0,L18H1,L18H2,L18H6,L18H7,L18H10,L19H5,L19H11,L20H1,L20H3,L20H6,L20H7,L20H8,L20H11,L21H0,L21H4,L21H6,L21H7,L22H3,L22H5,L22H6,L22H7,L22H8,L22H9,L23H3,L23H6,L23H8,L23H9,L23H10,L24H7,L24H8,L24H11,L25H3,L25H4,L25H5,L26H3,L26H4,L26H5,L26H10,L27H7,L28H2,L28H4,L28H8,L28H9,L29H6,L29H9,L29H11,L0H10,L1H0,L6H1,L20H4,L20H5,L25H0,L25H10,L26H11,L28H0,L28H5,L28H7,L28H10,L3H1,L7H10,L21H5,L7H8,L17H6,L17H9,L3H3"  # black_hole/pure_black/some bright_edge + a few from maybe_no_influence + a few from no_influence_false_true w/o L4H1 (add L4H1 would harm trajectory)
TRAJ_HEADS_1B3_T40_sink="L2H8,L4H0,L4H8,L5H3,L5H8,L5H10,L6H3,L7H4,L7H10,L9H4,L9H5,L9H8,L11H8,L11H9,L12H3,L12H4,L12H6,L12H7,L12H8,L13H5,L13H10,L14H1,L14H11,L15H1,L15H3,L16H0,L17H1,L17H2,L17H8,L17H10,L18H0,L18H1,L18H6,L18H7,L18H10,L19H11,L20H1,L20H6,L20H11,L21H4,L22H3,L22H7,L22H9,L23H8,L23H9,L24H11,L26H4,L26H5,L26H10,L26H11,L27H7,L28H9,"
TRAJ_HEADS_1B3_T40_sink_line="L12H4,L14H11,L17H8,L20H1,L22H3,L23H9,L26H4,L26H10,L26H11,L28H9,"  # contained in _sink
TRAJ_HEADS_1B3_T40_sink_weak="L6H10,L9H3,L11H7"
TRAJ_HEADS_1B3_T40_mess="L0H5,L0H10,L19H5,L20H2,L23H2,L24H7,L25H4,L28H0,L28H2,"
TRAJ_HEADS_1B3_T40_black_hole="L0H3,L1H0,L1H9,L2H0,L2H3,L2H5,L3H0,L3H1,L3H3,L3H4,L3H6,L3H10,L4H3,L5H2,L6H1,L9H0,L10H1,L14H7,L17H9,L18H2,L20H3,L20H4,L20H5,L20H7,L20H8,L21H0,L21H5,L21H6,L21H7,L22H5,L22H6,L22H8,L23H3,L23H6,L24H8,L25H0,L25H3,L25H5,L25H10,L26H3,L28H4,L28H5,L28H7,L28H8,L28H10,L29H9,L29H11"
TRAJ_HEADS_1B3_T40_bright_edge="L1H9,L2H0,L2H5,L3H0,L3H1,L3H3,L3H4,L6H1,L11H7,L17H9,L20H3,L20H4,L20H5,L20H7,L20H8,L21H5,L22H5,L24H8,L25H0,L25H10,L28H4,L28H5,L28H7,L28H8,L28H10,L29H9,L29H11"  # contained in _black_hole
TRAJ_HEADS_1B3_T40_traj_no_influence="L0H10,L17H6,L17H9,L23H10,L29H6,L3H3,L5H0,L6H1,L7H8"

TRAJ_HEADS_1B3_T40_no_influence4_14="L0H3,L0H5,L1H9,L2H0,L2H3,L2H5,L2H8,L3H0,L3H4,L3H6,L3H10,L4H0,L4H3,L4H8,L5H0,L5H2,L5H3,L5H8,L5H10,L6H3,L6H10,L7H4,L9H0,L9H3,L9H4,L9H5,L9H8,L10H1,L11H7,L11H8,L11H9,L12H3,L12H4,L12H6,L12H7,L12H8,L13H5,L13H10,L14H1,L14H7,L14H11,L15H1,L15H3,L16H0,L17H1,L17H2,L17H8,L17H10,L18H0,L18H1,L18H2,L18H6,L18H7,L18H10,L19H5,L19H11,L20H1,L20H3,L20H4,L20H6,L20H7,L20H8,L20H11,L21H0,L21H4,L21H6,L21H7,L22H3,L22H5,L22H6,L22H7,L22H8,L22H9,L23H3,L23H6,L23H8,L23H9,L23H10,L24H7,L24H8,L24H11,L25H3,L25H4,L25H5,L26H3,L26H4,L26H5,L26H10,L27H7,L28H2,L28H4,L28H8,L28H9,L29H6,L29H9,L29H11,L0H10,L1H0,L6H1,L20H5,L25H0,L25H10,L26H11,L28H0,L28H10,L3H1,L7H10,L7H8,L17H6,L17H9,L21H5,L3H3,L20H2,L20H5,L23H2,L25H0,L26H11,L28H0,L28H10,L29H0"  # black_hole + sink + sink_weak + mess + traj_no_influence

TRAJ_HEADS_1B3_T40_clear_traj_pattern="L0H0,L0H1,L0H6,L0H7,L0H8,L0H9,L0H11,L1H1,L1H2,L1H3,L1H4,L1H5,L1H8,L1H10,L1H11,L2H1,L2H2,L2H4,L2H6,L2H7,L2H9,L2H10,L2H11,L3H2,L3H5,L3H7,L3H8,L3H9,L3H11,L4H4,L4H5,L4H6,L4H7,L4H10,L5H1,L5H4,L5H5,L5H6,L5H9,L6H0,L6H2,L6H4,L6H5,L6H6,L6H7,L6H8,L6H9,L6H11,L7H0,L7H1,L7H2,L7H3,L7H5,L7H6,L7H9,L7H11,L8H0,L8H2,L8H3,L8H5,L8H6,L8H7,L8H8,L8H9,L8H10,L8H11,L9H1,L9H2,L9H6,L9H7,L9H9,L9H10,L10H2,L10H3,L10H4,L10H5,L10H7,L11H0,L11H2,L11H3,L11H5,L11H6,L11H10,L11H11,L12H1,L12H5,L12H9,L12H10,L12H11,L13H0,L13H1,L13H3,L13H4,L13H6,L13H8,L13H9,L13H11,L14H0,L14H2,L14H4,L14H6,L15H0,L15H2,L15H5,L15H6,L15H8,L15H9,L15H10,L15H11,L16H2,L16H3,L16H4,L16H5,L16H7,L16H8,L16H9,L16H10,L16H11,L17H0,L17H3,L17H4,L17H5,L17H7,L17H11,L18H3,L18H11,L19H0,L19H1,L19H3,L19H6,L19H9,L19H10,L21H1,L21H2,L21H8,L21H9,L21H10,L21H11,L22H0,L22H2,L22H4,L22H11,L23H1,L23H4,L23H5,L23H7,L23H11,L24H0,L24H1,L24H2,L24H9,L24H10,L25H7,L25H8,L25H9,L26H0,L26H2,L26H6,L27H0,L27H1,L27H3,L27H4,L27H5,L27H6,L27H8,L27H9,L27H10,L27H11,L28H1,L28H3,L29H1,L29H2,L29H3,L29H4,L29H5,L29H7,L29H8,L29H10"
TRAJ_HEADS_1B3_T40_vague_traj_pattern="L0H2,L0H4,L0H10,L1H6,L1H7,L3H3,L4H1,L4H2,L4H9,L4H11,L5H0,L5H7,L5H11,L6H1,L7H7,L7H8,L8H1,L8H4,L9H11,L10H0,L10H6,L10H8,L10H9,L10H10,L10H11,L11H1,L11H4,L12H0,L12H2,L13H2,L13H7,L14H3,L14H5,L14H8,L14H9,L14H10,L15H4,L15H7,L16H1,L16H6,L17H6,L17H9,L18H4,L18H5,L18H8,L18H9,L19H2,L19H4,L19H7,L19H8,L20H0,L20H9,L20H10,L21H3,L22H1,L22H10,L23H0,L23H10,L24H3,L24H4,L24H5,L24H6,L25H1,L25H2,L25H6,L25H11,L26H1,L26H7,L26H8,L26H9,L27H2,L28H5,L28H6,L28H7,L28H11,L29H6,"

# update
# complete_series = clear_traj_pattern + vague_traj_pattern + no_influence4_4 + no_influence_false_true + maybe_no_influence
# del from 4_4/4_7: L5H0,L7H8,

# The following heads are selected according to the head-wise curves
TRAJ_HEADS_1B3_early_plan=""  # selected according to the head-wise analysis

# Motion plan heads

# Exp config ==============================
SEEDS=(26)  #
PROMPTS=(
    "Against a pure white background, a basketball falls vertically from mid-air onto a wooden floor and bounces up several times."
)

HEAD_TYPE="traj"  # "black_hole", "traj"

BLACK_HOLE_HEADS_1B3_SOURCE="BLACK_HOLE_HEADS_1B3_T40"  #
BLACK_HOLE_HEADS_1B3="${!BLACK_HOLE_HEADS_1B3_SOURCE}"
TRAJ_HEADS_1B3_SOURCE="TRAJ_HEADS_1B3_T40_no_influence4_15"  #
TRAJ_HEADS_1B3="${!TRAJ_HEADS_1B3_SOURCE}"

BLACK_HOLE_HEADS_14B_SOURCE=""  #
BLACK_HOLE_HEADS_14B="${!BLACK_HOLE_HEADS_14B_SOURCE}"
TRAJ_HEADS_14B_SOURCE=""  #
TRAJ_HEADS_14B="${!TRAJ_HEADS_14B_SOURCE}"


# Empty -> ablate on all diffusion steps.
# Example: "1,2,3,5"
HEAD_ABLATION_STEPS=""
# HEAD_ABLATION_STEPS="1,2,3,4,5"

# ==============================
if [ "$task" == "t2v-1.3B" ]; then
    if [ "$HEAD_TYPE" == "black_hole" ]; then
        ABLATE_HEADS="$BLACK_HOLE_HEADS_1B3"
        PREFIX="BLACK_HOLE_HEADS_1B3_"
        SRC_TAG="${BLACK_HOLE_HEADS_1B3_SOURCE#$PREFIX}"
        HEAD_TAG="${HEAD_TYPE}_${SRC_TAG}"
    elif [ "$HEAD_TYPE" == "traj" ]; then
        ABLATE_HEADS="$TRAJ_HEADS_1B3"
        PREFIX="TRAJ_HEADS_1B3_"
        SRC_TAG="${TRAJ_HEADS_1B3_SOURCE#$PREFIX}"
        HEAD_TAG="${HEAD_TYPE}_${SRC_TAG}"
    fi
elif [ "$task" == "t2v-14B" ]; then
    if [ "$HEAD_TYPE" == "black_hole" ]; then
        ABLATE_HEADS="$BLACK_HOLE_HEADS_14B"
        PREFIX="BLACK_HOLE_HEADS_14B_"
        SRC_TAG="${BLACK_HOLE_HEADS_14B_SOURCE#$PREFIX}"
        HEAD_TAG="${HEAD_TYPE}_${SRC_TAG}"
    elif [ "$HEAD_TYPE" == "traj" ]; then
        ABLATE_HEADS="$TRAJ_HEADS_14B"
        PREFIX="TRAJ_HEADS_14B_"
        SRC_TAG="${TRAJ_HEADS_14B_SOURCE#$PREFIX}"
        HEAD_TAG="${HEAD_TYPE}_${SRC_TAG}"
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
