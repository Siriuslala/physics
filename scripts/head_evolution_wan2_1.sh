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

# ==============================
SEEDS=(26)
PROMPTS=(
    "Against a pure white background, a basketball falls vertically from mid-air onto a wooden floor and bounces up several times."
)

TARGET_OBJECT_WORDS="basketball"
TARGET_VERB_WORDS="falls,bounces,up"

# Step-wise analysis. Empty -> all available steps from reused cross-attention maps
HEAD_EVOLUTION_STEPS=""
# Step-wise analysis. Layer used for step-wise curves
HEAD_EVOLUTION_STEPWISE_LAYER=27
# Layer-wise analysis. Empty -> same as HEAD_EVOLUTION_STEPS
HEAD_EVOLUTION_LAYERWISE_STEPS=""
# Layer-wise analysis. -2 all layers; -1 last layer; >=0 specific layer
HEAD_EVOLUTION_HEAD_LAYER=-2

# Reference map for support-mask construction
HEAD_EVOLUTION_REFERENCE_STEP=50
HEAD_EVOLUTION_REFERENCE_LAYER=27
HEAD_EVOLUTION_CENTER_MODE="geometric_center"   # peak / centroid / geometric_center

# Radius settings
HEAD_EVOLUTION_SUPPORT_RADIUS_MODE="adaptive_area"  # fixed / adaptive_area
HEAD_EVOLUTION_SUPPORT_RADIUS_FIXED=2.0
HEAD_EVOLUTION_SUPPORT_RADIUS_ALPHA=1.0
HEAD_EVOLUTION_SUPPORT_RADIUS_MIN=1.0
HEAD_EVOLUTION_SUPPORT_RADIUS_MAX_RATIO=0.25

# Trajectory extraction settings on reference map
HEAD_EVOLUTION_TRAJ_POWER=1.5
HEAD_EVOLUTION_TRAJ_QUANTILE=0.95

# Save a timeline PDF with center point + radius circle overlays for radius sanity-check
HEAD_EVOLUTION_SAVE_REFERENCE_RADIUS_OVERLAY=True
HEAD_EVOLUTION_REFERENCE_VIZ_NUM_FRAMES=10

# Head scoring settings
HEAD_EVOLUTION_EARLY_STEP_END=5
HEAD_EVOLUTION_SCORE_QUANTILE=0.7

# Whether to apply preprocessing (remove outliers) during metric computation.
# Note: reference trajectory extraction still uses preprocessing.
HEAD_EVOLUTION_APPLY_PREPROCESS_ON_METRICS=False

# Attention-map preprocessing and concentrated-region metric
HEAD_EVOLUTION_PREPROCESS_WINSORIZE_QUANTILE=0.995
HEAD_EVOLUTION_PREPROCESS_DESPIKE_QUANTILE=0.98
HEAD_EVOLUTION_PREPROCESS_MIN_COMPONENT_AREA=2
HEAD_EVOLUTION_CONCENTRATED_REGION_TOP_RATIO=0.05

# ==============================
for PROMPT in "${PROMPTS[@]}"; do
for SEED in "${SEEDS[@]}"; do
    echo "=================================================================================="
    echo "Running head_evolution | prompt: $PROMPT | seed: $SEED"
    echo "=================================================================================="

    PROMPT_TAG=$(build_prompt_tag "$PROMPT")
    SAVE_DIR="$WORK_DIR/outputs_wan_2_1_${task}/head_evolution/${PROMPT_TAG}/seed_${SEED}_shift_${SAMPLE_SHIFT}_guide_${SAMPLE_GUIDE_SCALE}"
    SUMMARY_FILE="$SAVE_DIR/head_evolution_summary.json"
    if [ -f "$SUMMARY_FILE" ]; then
        echo "Summary already exists: $SUMMARY_FILE"
        echo "Skip."
        echo ""
        continue
    fi
    mkdir -p "$SAVE_DIR"

    REUSE_CROSS_ATTENTION_DIR="$WORK_DIR/outputs_wan_2_1_${task}/cross_attention_token_viz/${PROMPT_TAG}/seed_${SEED}_shift_${SAMPLE_SHIFT}_guide_${SAMPLE_GUIDE_SCALE}"

    python $ROOT_DIR/wan21_t2v_experiments/run_wan21_t2v_experiments.py \
        --experiment head_evolution \
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
        --target_object_words "$TARGET_OBJECT_WORDS" \
        --target_verb_words "$TARGET_VERB_WORDS" \
        --head_evolution_steps "$HEAD_EVOLUTION_STEPS" \
        --head_evolution_layerwise_steps "$HEAD_EVOLUTION_LAYERWISE_STEPS" \
        --head_evolution_head_layer $HEAD_EVOLUTION_HEAD_LAYER \
        --head_evolution_stepwise_layer $HEAD_EVOLUTION_STEPWISE_LAYER \
        --head_evolution_reference_step $HEAD_EVOLUTION_REFERENCE_STEP \
        --head_evolution_reference_layer $HEAD_EVOLUTION_REFERENCE_LAYER \
        --head_evolution_center_mode $HEAD_EVOLUTION_CENTER_MODE \
        --head_evolution_support_radius_mode $HEAD_EVOLUTION_SUPPORT_RADIUS_MODE \
        --head_evolution_support_radius_fixed $HEAD_EVOLUTION_SUPPORT_RADIUS_FIXED \
        --head_evolution_support_radius_alpha $HEAD_EVOLUTION_SUPPORT_RADIUS_ALPHA \
        --head_evolution_support_radius_min $HEAD_EVOLUTION_SUPPORT_RADIUS_MIN \
        --head_evolution_support_radius_max_ratio $HEAD_EVOLUTION_SUPPORT_RADIUS_MAX_RATIO \
        --head_evolution_traj_power $HEAD_EVOLUTION_TRAJ_POWER \
        --head_evolution_traj_quantile $HEAD_EVOLUTION_TRAJ_QUANTILE \
        --head_evolution_reference_viz_num_frames $HEAD_EVOLUTION_REFERENCE_VIZ_NUM_FRAMES \
        --head_evolution_save_reference_radius_overlay $HEAD_EVOLUTION_SAVE_REFERENCE_RADIUS_OVERLAY \
        --head_evolution_early_step_end $HEAD_EVOLUTION_EARLY_STEP_END \
        --head_evolution_score_quantile $HEAD_EVOLUTION_SCORE_QUANTILE \
        --head_evolution_apply_preprocess_on_metrics $HEAD_EVOLUTION_APPLY_PREPROCESS_ON_METRICS \
        --head_evolution_preprocess_winsorize_quantile $HEAD_EVOLUTION_PREPROCESS_WINSORIZE_QUANTILE \
        --head_evolution_preprocess_despike_quantile $HEAD_EVOLUTION_PREPROCESS_DESPIKE_QUANTILE \
        --head_evolution_preprocess_min_component_area $HEAD_EVOLUTION_PREPROCESS_MIN_COMPONENT_AREA \
        --head_evolution_concentrated_region_top_ratio $HEAD_EVOLUTION_CONCENTRATED_REGION_TOP_RATIO \
        --reuse_cross_attention_dir "$REUSE_CROSS_ATTENTION_DIR"

    echo "Finished head_evolution | prompt: $PROMPT | seed: $SEED"
    echo ""
done
done
