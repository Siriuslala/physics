source ./env.sh
cd $ROOT_DIR

source ~/miniforge3/etc/profile.d/conda.sh
conda activate video

# Analyze head consensus and attractor dynamics from an existing
# cross_attention_token_viz directory. This script does not resample videos.

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

# SEEDS=(4)
# PROMPTS=(
#     "Against a pure white background, a wooden cube block at the top of a smooth slope slides straight down the slope with steadily and uniformly increasing speed."
# )
# TARGET_OBJECT_WORDS="cube"
# TARGET_VERB_WORDS="slope,slides"

# SEEDS=(2)
# PROMPTS=(
#     "Against a pure white background, there is a wooden horizontal surface, with one single wooden slope attached to its left end. One small green ball starts from rest at the top of the slope, slides straight along the slope the entire time with its speed increasing steadily and uniformly, then rolls rightward along the wooden horizontal surface after reaching it."
# )
# TARGET_OBJECT_WORDS="ball"  # "basketball"
# TARGET_VERB_WORDS="slope,slides,rolls"  # "falls,bounces,up"


CLEAR_TRAJ_HEADS="L0H0,L0H1,L0H6,L0H7,L0H8,L0H9,L0H11,L1H1,L1H2,L1H3,L1H4,L1H5,L1H8,L1H10,L1H11,L2H1,L2H2,L2H4,L2H6,L2H7,L2H9,L2H10,L2H11,L3H2,L3H5,L3H7,L3H8,L3H9,L3H11,L4H4,L4H5,L4H6,L4H7,L4H10,L5H1,L5H4,L5H5,L5H6,L5H9,L6H0,L6H2,L6H4,L6H5,L6H6,L6H7,L6H8,L6H9,L6H11,L7H0,L7H1,L7H2,L7H3,L7H5,L7H6,L7H9,L7H11,L8H0,L8H2,L8H3,L8H5,L8H6,L8H7,L8H8,L8H9,L8H10,L8H11,L9H1,L9H2,L9H6,L9H7,L9H9,L9H10,L10H2,L10H3,L10H4,L10H5,L10H7,L11H0,L11H2,L11H3,L11H5,L11H6,L11H10,L11H11,L12H1,L12H5,L12H9,L12H10,L12H11,L13H0,L13H1,L13H3,L13H4,L13H6,L13H8,L13H9,L13H11,L14H0,L14H2,L14H4,L14H6,L15H0,L15H2,L15H5,L15H6,L15H8,L15H9,L15H10,L15H11,L16H2,L16H3,L16H4,L16H5,L16H7,L16H8,L16H9,L16H10,L16H11,L17H0,L17H3,L17H4,L17H5,L17H7,L17H11,L18H3,L18H11,L19H0,L19H1,L19H3,L19H6,L19H9,L19H10,L21H1,L21H2,L21H8,L21H9,L21H10,L21H11,L22H0,L22H2,L22H4,L22H11,L23H1,L23H4,L23H5,L23H7,L23H11,L24H0,L24H1,L24H2,L24H9,L24H10,L25H7,L25H8,L25H9,L26H0,L26H2,L26H6,L27H0,L27H1,L27H3,L27H4,L27H5,L27H6,L27H8,L27H9,L27H10,L27H11,L28H1,L28H3,L29H1,L29H2,L29H3,L29H4,L29H5,L29H7,L29H8,L29H10"
VAGUE_TRAJ_HEADS="L0H2,L0H4,L0H10,L1H6,L1H7,L3H3,L4H1,L4H2,L4H9,L4H11,L5H0,L5H7,L5H11,L6H1,L7H7,L7H8,L8H1,L8H4,L9H11,L10H0,L10H6,L10H8,L10H9,L10H10,L10H11,L11H1,L11H4,L12H0,L12H2,L13H2,L13H7,L14H3,L14H5,L14H8,L14H9,L14H10,L15H4,L15H7,L16H1,L16H6,L17H6,L17H9,L18H4,L18H5,L18H8,L18H9,L19H2,L19H4,L19H7,L19H8,L20H0,L20H9,L20H10,L21H3,L22H1,L22H10,L23H0,L23H10,L24H3,L24H4,L24H5,L24H6,L25H1,L25H2,L25H6,L25H11,L26H1,L26H7,L26H8,L26H9,L27H2,L28H5,L28H6,L28H7,L28H11,L29H6"

# Empty -> all heads
TRAJ_TYPE="all"  # ! all, traj_clear, traj_vague, traj_clear_vague
if [ "$TRAJ_TYPE" == "traj_clear" ]; then
    HEAD_TRAJECTORY_DYNAMICS_HEADS="$CLEAR_TRAJ_HEADS"
elif [ "$TRAJ_TYPE" == "traj_vague" ]; then
    HEAD_TRAJECTORY_DYNAMICS_HEADS="$VAGUE_TRAJ_HEADS"
elif [ "$TRAJ_TYPE" == "traj_clear_vague" ]; then
    HEAD_TRAJECTORY_DYNAMICS_HEADS="$CLEAR_TRAJ_HEADS,$VAGUE_TRAJ_HEADS"
elif [ "$TRAJ_TYPE" == "all" ]; then
    HEAD_TRAJECTORY_DYNAMICS_HEADS=""
else
    echo "Unknown traj type: $TRAJ_TYPE"
    exit 1
fi

# Empty -> all available steps in reused maps
HEAD_TRAJECTORY_DYNAMICS_STEPS=""

# metric settings
HEAD_TRAJECTORY_DYNAMICS_DISTANCE_METRICS=""  # if empty, use all metrics
HEAD_TRAJECTORY_DYNAMICS_SUPPORT_QUANTILE=0.8  # for IoU-based metric
HEAD_TRAJECTORY_DYNAMICS_ATTRACTOR_WINDOW=5  # ! attractor window
HEAD_TRAJECTORY_DYNAMICS_ATTRACTOR_DISTANCE_METRIC=""  # metrics for attractor analysis, include `onestep`, `best_future` and `window_mean`; empty means all supported metrics

# experiment mode
HEAD_TRAJECTORY_DYNAMICS_PLOT_ONLY_FROM_CSV=False  # ! if True, only plot head dynamics metrics, no calculations
HEAD_TRAJECTORY_DYNAMICS_OVERLAY_ONLY=True  # ! if True, only render center/support overlays from saved maps and center cache
HEAD_TRAJECTORY_DYNAMICS_SKIP_EXISTING_PLOTS=True
HEAD_TRAJECTORY_DYNAMICS_HYPOTHESIS="attractor"  # ! hypothesis for analysis
HEAD_TRAJECTORY_DYNAMICS_USE_MOTION_PLANNING_REGION_BEFORE_METRICS=True  # ! If True, perform support overlay first to find the motion planning region (patches inside contours)

# region_centroid: the one used in cross_attention_token_viz (centroid);
# preprocessed_component_center: the one used in head_evolution
HEAD_TRAJECTORY_DYNAMICS_REFERENCE_STEP=50
HEAD_TRAJECTORY_DYNAMICS_REFERENCE_LAYER=27
HEAD_TRAJECTORY_DYNAMICS_REFERENCE_CENTER_METHOD="preprocessed_component_center"
HEAD_TRAJECTORY_DYNAMICS_REFERENCE_PREPROCESSED_CENTER_MODE="geometric_center"
HEAD_TRAJECTORY_DYNAMICS_REFERENCE_CENTER_POWER=1.5  # if geometric center, not used
HEAD_TRAJECTORY_DYNAMICS_REFERENCE_CENTER_QUANTILE=0.8
HEAD_TRAJECTORY_DYNAMICS_REFERENCE_PREPROCESS_WINSORIZE_QUANTILE=0.995
HEAD_TRAJECTORY_DYNAMICS_REFERENCE_PREPROCESS_DESPIKE_QUANTILE=0.98
HEAD_TRAJECTORY_DYNAMICS_REFERENCE_PREPROCESS_MIN_COMPONENT_AREA=2

HEAD_TRAJECTORY_DYNAMICS_APPLY_PREPROCESS_ON_HEAD_CENTERS=True
if [ "${HEAD_TRAJECTORY_DYNAMICS_APPLY_PREPROCESS_ON_HEAD_CENTERS}" = "True" ]; then
    HEAD_TRAJECTORY_DYNAMICS_CENTER_METHOD="preprocessed_component_center"
else
    HEAD_TRAJECTORY_DYNAMICS_CENTER_METHOD="region_centroid"
fi
HEAD_TRAJECTORY_DYNAMICS_PREPROCESSED_CENTER_MODE="geometric_center"  # peak / centroid / geometric_center; only used when APPLY_PREPROCESS_ON_HEAD_CENTERS=True
HEAD_TRAJECTORY_DYNAMICS_CENTER_POWER=1.5
HEAD_TRAJECTORY_DYNAMICS_CENTER_QUANTILE=0.8
HEAD_TRAJECTORY_DYNAMICS_PREPROCESS_WINSORIZE_QUANTILE=0.995
HEAD_TRAJECTORY_DYNAMICS_PREPROCESS_DESPIKE_QUANTILE=0.98
HEAD_TRAJECTORY_DYNAMICS_PREPROCESS_MIN_COMPONENT_AREA=2

# Center-overlay visualization:
# - If step/layer are both valid, only render that step/layer.
# - If both keep default -1, code will automatically render all analyzed heads
#   selected by HEAD_TRAJECTORY_DYNAMICS_HEADS into a dedicated directory.
HEAD_TRAJECTORY_DYNAMICS_CENTER_VIZ_ENABLE=True
HEAD_TRAJECTORY_DYNAMICS_CENTER_VIZ_STEP=-1
HEAD_TRAJECTORY_DYNAMICS_CENTER_VIZ_LAYER=-1
HEAD_TRAJECTORY_DYNAMICS_CENTER_VIZ_HEADS=""
HEAD_TRAJECTORY_DYNAMICS_CENTER_VIZ_NUM_FRAMES=10

# Support-overlap mask visualization:
# - Render only the contour-overlay PDF of the denoised motion-planning region.
HEAD_TRAJECTORY_DYNAMICS_SUPPORT_VIZ_ENABLE=True
HEAD_TRAJECTORY_DYNAMICS_SUPPORT_VIZ_STEP=-1
HEAD_TRAJECTORY_DYNAMICS_SUPPORT_VIZ_LAYER=-1
HEAD_TRAJECTORY_DYNAMICS_SUPPORT_VIZ_HEADS=""
HEAD_TRAJECTORY_DYNAMICS_SUPPORT_VIZ_NUM_FRAMES=10
HEAD_TRAJECTORY_DYNAMICS_SUPPORT_VIZ_CONTOUR_MIN_COMPONENT_AREA=4
HEAD_TRAJECTORY_DYNAMICS_SUPPORT_CACHE_NUM_WORKERS=16  # multi-cpu for acceleration (build motion-planning region mask)
HEAD_TRAJECTORY_DYNAMICS_CENTER_CACHE_NUM_WORKERS=16  # multi-cpu for acceleration (center extraction)
HEAD_TRAJECTORY_DYNAMICS_OVERLAY_NUM_WORKERS=16  # multi-cpu for acceleration (center/support overlay rendering)
HEAD_TRAJECTORY_DYNAMICS_CACHE_SAVE_INTERVAL=512


for PROMPT in "${PROMPTS[@]}"; do
for SEED in "${SEEDS[@]}"; do
    echo "=================================================================================="
    echo "Running head_trajectory_dynamics | prompt: $PROMPT | seed: $SEED"
    echo "=================================================================================="

    PROMPT_TAG=$(build_prompt_tag "$PROMPT")
    SAVE_DIR="$WORK_DIR/outputs_wan_2_1_${task}/head_trajectory_dynamics/${PROMPT_TAG}/seed_${SEED}_shift_${SAMPLE_SHIFT}_guide_${SAMPLE_GUIDE_SCALE}/${TRAJ_TYPE}"
    SUMMARY_FILE="$SAVE_DIR/head_trajectory_dynamics_summary.json"
    if [ -f "$SUMMARY_FILE" ] && [ "${HEAD_TRAJECTORY_DYNAMICS_PLOT_ONLY_FROM_CSV}" != "True" ] && [ "${HEAD_TRAJECTORY_DYNAMICS_OVERLAY_ONLY}" != "True" ]; then
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
        --head_trajectory_dynamics_attractor_distance_metric "$HEAD_TRAJECTORY_DYNAMICS_ATTRACTOR_DISTANCE_METRIC" \
        --head_trajectory_dynamics_center_method $HEAD_TRAJECTORY_DYNAMICS_CENTER_METHOD \
        --head_trajectory_dynamics_center_power $HEAD_TRAJECTORY_DYNAMICS_CENTER_POWER \
        --head_trajectory_dynamics_center_quantile $HEAD_TRAJECTORY_DYNAMICS_CENTER_QUANTILE \
        --head_trajectory_dynamics_preprocessed_center_mode $HEAD_TRAJECTORY_DYNAMICS_PREPROCESSED_CENTER_MODE \
        --head_trajectory_dynamics_preprocess_winsorize_quantile $HEAD_TRAJECTORY_DYNAMICS_PREPROCESS_WINSORIZE_QUANTILE \
        --head_trajectory_dynamics_preprocess_despike_quantile $HEAD_TRAJECTORY_DYNAMICS_PREPROCESS_DESPIKE_QUANTILE \
        --head_trajectory_dynamics_preprocess_min_component_area $HEAD_TRAJECTORY_DYNAMICS_PREPROCESS_MIN_COMPONENT_AREA \
        --head_trajectory_dynamics_reference_center_method $HEAD_TRAJECTORY_DYNAMICS_REFERENCE_CENTER_METHOD \
        --head_trajectory_dynamics_reference_center_power $HEAD_TRAJECTORY_DYNAMICS_REFERENCE_CENTER_POWER \
        --head_trajectory_dynamics_reference_center_quantile $HEAD_TRAJECTORY_DYNAMICS_REFERENCE_CENTER_QUANTILE \
        --head_trajectory_dynamics_reference_preprocessed_center_mode $HEAD_TRAJECTORY_DYNAMICS_REFERENCE_PREPROCESSED_CENTER_MODE \
        --head_trajectory_dynamics_reference_preprocess_winsorize_quantile $HEAD_TRAJECTORY_DYNAMICS_REFERENCE_PREPROCESS_WINSORIZE_QUANTILE \
        --head_trajectory_dynamics_reference_preprocess_despike_quantile $HEAD_TRAJECTORY_DYNAMICS_REFERENCE_PREPROCESS_DESPIKE_QUANTILE \
        --head_trajectory_dynamics_reference_preprocess_min_component_area $HEAD_TRAJECTORY_DYNAMICS_REFERENCE_PREPROCESS_MIN_COMPONENT_AREA \
        --head_trajectory_dynamics_center_viz_enable $HEAD_TRAJECTORY_DYNAMICS_CENTER_VIZ_ENABLE \
        --head_trajectory_dynamics_center_viz_step $HEAD_TRAJECTORY_DYNAMICS_CENTER_VIZ_STEP \
        --head_trajectory_dynamics_center_viz_layer $HEAD_TRAJECTORY_DYNAMICS_CENTER_VIZ_LAYER \
        --head_trajectory_dynamics_center_viz_heads "$HEAD_TRAJECTORY_DYNAMICS_CENTER_VIZ_HEADS" \
        --head_trajectory_dynamics_center_viz_num_frames $HEAD_TRAJECTORY_DYNAMICS_CENTER_VIZ_NUM_FRAMES \
        --head_trajectory_dynamics_support_viz_enable $HEAD_TRAJECTORY_DYNAMICS_SUPPORT_VIZ_ENABLE \
        --head_trajectory_dynamics_support_viz_step $HEAD_TRAJECTORY_DYNAMICS_SUPPORT_VIZ_STEP \
        --head_trajectory_dynamics_support_viz_layer $HEAD_TRAJECTORY_DYNAMICS_SUPPORT_VIZ_LAYER \
        --head_trajectory_dynamics_support_viz_heads "$HEAD_TRAJECTORY_DYNAMICS_SUPPORT_VIZ_HEADS" \
        --head_trajectory_dynamics_support_viz_num_frames $HEAD_TRAJECTORY_DYNAMICS_SUPPORT_VIZ_NUM_FRAMES \
        --head_trajectory_dynamics_support_viz_contour_min_component_area $HEAD_TRAJECTORY_DYNAMICS_SUPPORT_VIZ_CONTOUR_MIN_COMPONENT_AREA \
        --head_trajectory_dynamics_support_cache_num_workers $HEAD_TRAJECTORY_DYNAMICS_SUPPORT_CACHE_NUM_WORKERS \
        --head_trajectory_dynamics_center_cache_num_workers $HEAD_TRAJECTORY_DYNAMICS_CENTER_CACHE_NUM_WORKERS \
        --head_trajectory_dynamics_overlay_num_workers $HEAD_TRAJECTORY_DYNAMICS_OVERLAY_NUM_WORKERS \
        --head_trajectory_dynamics_cache_save_interval $HEAD_TRAJECTORY_DYNAMICS_CACHE_SAVE_INTERVAL \
        --head_trajectory_dynamics_hypothesis $HEAD_TRAJECTORY_DYNAMICS_HYPOTHESIS \
        --head_trajectory_dynamics_traj_type "$TRAJ_TYPE" \
        --head_trajectory_dynamics_use_motion_planning_region_before_metrics $HEAD_TRAJECTORY_DYNAMICS_USE_MOTION_PLANNING_REGION_BEFORE_METRICS \
        --head_trajectory_dynamics_plot_only_from_csv $HEAD_TRAJECTORY_DYNAMICS_PLOT_ONLY_FROM_CSV \
        --head_trajectory_dynamics_overlay_only $HEAD_TRAJECTORY_DYNAMICS_OVERLAY_ONLY \
        --head_trajectory_dynamics_skip_existing_plots $HEAD_TRAJECTORY_DYNAMICS_SKIP_EXISTING_PLOTS \
        --reuse_cross_attention_dir "$REUSE_CROSS_ATTENTION_DIR"

    echo "Finished head_trajectory_dynamics | prompt: $PROMPT | seed: $SEED"
    echo ""
done
done
