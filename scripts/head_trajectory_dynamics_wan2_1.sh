source ./env.sh
cd $ROOT_DIR

# Analyze head consensus and attractor dynamics from an existing
# cross_attention_token_viz directory. This script does not resample videos.

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
task="t2v-1.3B"
SIZE="832*480"
CKPT_DIR="$MODEL_DIR/Wan2.1-T2V-1.3B"
FRAME_NUM=81
SAMPLE_STEPS=50
SAMPLE_SHIFT=5.0
SAMPLE_GUIDE_SCALE=5.0

SEEDS=(26)
PROMPTS=(
    "Against a pure white background, a basketball falls vertically from mid-air onto a wooden floor and bounces up several times."
)

TARGET_OBJECT_WORDS="basketball"
TARGET_VERB_WORDS="falls,bounces,up"

# Empty -> all heads
HEAD_TRAJECTORY_DYNAMICS_HEADS="L0H0,L0H1,L0H6,L0H7,L0H8,L0H9,L0H11,L1H1,L1H2,L1H3,L1H4,L1H5,L1H8,L1H10,L1H11,L2H1,L2H2,L2H4,L2H6,L2H7,L2H9,L2H10,L2H11,L3H2,L3H5,L3H7,L3H8,L3H9,L3H11,L4H4,L4H5,L4H6,L4H7,L4H10,L5H1,L5H4,L5H5,L5H6,L5H9,L6H0,L6H2,L6H4,L6H5,L6H6,L6H7,L6H8,L6H9,L6H11,L7H0,L7H1,L7H2,L7H3,L7H5,L7H6,L7H9,L7H11,L8H0,L8H2,L8H3,L8H5,L8H6,L8H7,L8H8,L8H9,L8H10,L8H11,L9H1,L9H2,L9H6,L9H7,L9H9,L9H10,L10H2,L10H3,L10H4,L10H5,L10H7,L11H0,L11H2,L11H3,L11H5,L11H6,L11H10,L11H11,L12H1,L12H5,L12H9,L12H10,L12H11,L13H0,L13H1,L13H3,L13H4,L13H6,L13H8,L13H9,L13H11,L14H0,L14H2,L14H4,L14H6,L15H0,L15H2,L15H5,L15H6,L15H8,L15H9,L15H10,L15H11,L16H2,L16H3,L16H4,L16H5,L16H7,L16H8,L16H9,L16H10,L16H11,L17H0,L17H3,L17H4,L17H5,L17H7,L17H11,L18H3,L18H11,L19H0,L19H1,L19H3,L19H6,L19H9,L19H10,L21H1,L21H2,L21H8,L21H9,L21H10,L21H11,L22H0,L22H2,L22H4,L22H11,L23H1,L23H4,L23H5,L23H7,L23H11,L24H0,L24H1,L24H2,L24H9,L24H10,L25H7,L25H8,L25H9,L26H0,L26H2,L26H6,L27H0,L27H1,L27H3,L27H4,L27H5,L27H6,L27H8,L27H9,L27H10,L27H11,L28H1,L28H3,L29H1,L29H2,L29H3,L29H4,L29H5,L29H7,L29H8,L29H10"
# Empty -> all available steps in reused maps
HEAD_TRAJECTORY_DYNAMICS_STEPS=""
# Empty -> use default metric bundle
HEAD_TRAJECTORY_DYNAMICS_DISTANCE_METRICS=""
HEAD_TRAJECTORY_DYNAMICS_REFERENCE_STEP=50
HEAD_TRAJECTORY_DYNAMICS_REFERENCE_LAYER=27
HEAD_TRAJECTORY_DYNAMICS_SUPPORT_QUANTILE=0.9
HEAD_TRAJECTORY_DYNAMICS_ATTRACTOR_WINDOW=3
HEAD_TRAJECTORY_DYNAMICS_CENTER_METHOD="region_centroid"
HEAD_TRAJECTORY_DYNAMICS_CENTER_POWER=1.5
HEAD_TRAJECTORY_DYNAMICS_CENTER_QUANTILE=0.8
HEAD_TRAJECTORY_DYNAMICS_PREPROCESSED_CENTER_MODE="geometric_center"
HEAD_TRAJECTORY_DYNAMICS_PREPROCESS_WINSORIZE_QUANTILE=0.995
HEAD_TRAJECTORY_DYNAMICS_PREPROCESS_DESPIKE_QUANTILE=0.98
HEAD_TRAJECTORY_DYNAMICS_PREPROCESS_MIN_COMPONENT_AREA=2
# Center-overlay visualization:
# - If step/layer are both valid, only render that step/layer.
# - If both keep default -1, code will automatically render all analyzed heads
#   selected by HEAD_TRAJECTORY_DYNAMICS_HEADS into a dedicated directory.
HEAD_TRAJECTORY_DYNAMICS_CENTER_VIZ_STEP=-1
HEAD_TRAJECTORY_DYNAMICS_CENTER_VIZ_LAYER=-1
HEAD_TRAJECTORY_DYNAMICS_CENTER_VIZ_HEADS=""
HEAD_TRAJECTORY_DYNAMICS_CENTER_VIZ_NUM_FRAMES=10

for PROMPT in "${PROMPTS[@]}"; do
for SEED in "${SEEDS[@]}"; do
    echo "=================================================================================="
    echo "Running head_trajectory_dynamics | prompt: $PROMPT | seed: $SEED"
    echo "=================================================================================="

    PROMPT_TAG=$(build_prompt_tag "$PROMPT")
    SAVE_DIR="$WORK_DIR/outputs_wan_2_1_${task}/head_trajectory_dynamics/${PROMPT_TAG}/seed_${SEED}_shift_${SAMPLE_SHIFT}_guide_${SAMPLE_GUIDE_SCALE}"
    SUMMARY_FILE="$SAVE_DIR/head_trajectory_dynamics_summary.json"
    if [ -f "$SUMMARY_FILE" ]; then
        echo "Summary already exists: $SUMMARY_FILE"
        echo "Skip."
        echo ""
        continue
    fi
    mkdir -p "$SAVE_DIR"

    REUSE_CROSS_ATTENTION_DIR="$WORK_DIR/outputs_wan_2_1_${task}/cross_attention_token_viz/${PROMPT_TAG}/seed_${SEED}_shift_${SAMPLE_SHIFT}_guide_${SAMPLE_GUIDE_SCALE}"

    python $ROOT_DIR/wan21_t2v_experiments/run_wan21_t2v_experiments.py \
        --experiment head_trajectory_dynamics \
        --wan21_root $ROOT_DIR/projects/Wan2_1 \
        --ckpt_dir $CKPT_DIR \
        --output_dir $SAVE_DIR \
        --task $task \
        --prompt "$PROMPT" \
        --size $SIZE \
        --frame_num $FRAME_NUM \
        --sampling_steps $SAMPLE_STEPS \
        --shift $SAMPLE_SHIFT \
        --guide_scale $SAMPLE_GUIDE_SCALE \
        --seed $SEED \
        --target_object_words "$TARGET_OBJECT_WORDS" \
        --target_verb_words "$TARGET_VERB_WORDS" \
        --head_trajectory_dynamics_heads "$HEAD_TRAJECTORY_DYNAMICS_HEADS" \
        --head_trajectory_dynamics_steps "$HEAD_TRAJECTORY_DYNAMICS_STEPS" \
        --head_trajectory_dynamics_distance_metrics "$HEAD_TRAJECTORY_DYNAMICS_DISTANCE_METRICS" \
        --head_trajectory_dynamics_reference_step $HEAD_TRAJECTORY_DYNAMICS_REFERENCE_STEP \
        --head_trajectory_dynamics_reference_layer $HEAD_TRAJECTORY_DYNAMICS_REFERENCE_LAYER \
        --head_trajectory_dynamics_support_quantile $HEAD_TRAJECTORY_DYNAMICS_SUPPORT_QUANTILE \
        --head_trajectory_dynamics_attractor_window $HEAD_TRAJECTORY_DYNAMICS_ATTRACTOR_WINDOW \
        --head_trajectory_dynamics_center_method $HEAD_TRAJECTORY_DYNAMICS_CENTER_METHOD \
        --head_trajectory_dynamics_center_power $HEAD_TRAJECTORY_DYNAMICS_CENTER_POWER \
        --head_trajectory_dynamics_center_quantile $HEAD_TRAJECTORY_DYNAMICS_CENTER_QUANTILE \
        --head_trajectory_dynamics_preprocessed_center_mode $HEAD_TRAJECTORY_DYNAMICS_PREPROCESSED_CENTER_MODE \
        --head_trajectory_dynamics_preprocess_winsorize_quantile $HEAD_TRAJECTORY_DYNAMICS_PREPROCESS_WINSORIZE_QUANTILE \
        --head_trajectory_dynamics_preprocess_despike_quantile $HEAD_TRAJECTORY_DYNAMICS_PREPROCESS_DESPIKE_QUANTILE \
        --head_trajectory_dynamics_preprocess_min_component_area $HEAD_TRAJECTORY_DYNAMICS_PREPROCESS_MIN_COMPONENT_AREA \
        --head_trajectory_dynamics_center_viz_step $HEAD_TRAJECTORY_DYNAMICS_CENTER_VIZ_STEP \
        --head_trajectory_dynamics_center_viz_layer $HEAD_TRAJECTORY_DYNAMICS_CENTER_VIZ_LAYER \
        --head_trajectory_dynamics_center_viz_heads "$HEAD_TRAJECTORY_DYNAMICS_CENTER_VIZ_HEADS" \
        --head_trajectory_dynamics_center_viz_num_frames $HEAD_TRAJECTORY_DYNAMICS_CENTER_VIZ_NUM_FRAMES \
        --reuse_cross_attention_dir "$REUSE_CROSS_ATTENTION_DIR"

    echo "Finished head_trajectory_dynamics | prompt: $PROMPT | seed: $SEED"
    echo ""
done
done
