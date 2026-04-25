source ./env.sh
cd $ROOT_DIR

# Self-attention distribution experiment.
# It reuses a cross_attention_token_viz directory to build the reference object
# support mask, then probes:
# 1) object-region query tokens -> object/non-object key mass
# 2) global query tokens -> signed-dt frame-mass distribution

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

SELF_ATTENTION_DISTRIBUTION_STEPS="1,2,3,4,5,6"
SELF_ATTENTION_DISTRIBUTION_LAYERS=""
SELF_ATTENTION_DISTRIBUTION_BRANCH="cond"
SELF_ATTENTION_DISTRIBUTION_REFERENCE_STEP=50
SELF_ATTENTION_DISTRIBUTION_REFERENCE_LAYER=27
SELF_ATTENTION_DISTRIBUTION_REFERENCE_CENTER_MODE="geometric_center"
SELF_ATTENTION_DISTRIBUTION_REFERENCE_CENTER_POWER=1.5
SELF_ATTENTION_DISTRIBUTION_REFERENCE_CENTER_QUANTILE=0.8
SELF_ATTENTION_DISTRIBUTION_SUPPORT_RADIUS_MODE="adaptive_area"
SELF_ATTENTION_DISTRIBUTION_SUPPORT_RADIUS_FIXED=2.0
SELF_ATTENTION_DISTRIBUTION_SUPPORT_RADIUS_ALPHA=1.5
SELF_ATTENTION_DISTRIBUTION_SUPPORT_RADIUS_MIN=1.0
SELF_ATTENTION_DISTRIBUTION_SUPPORT_RADIUS_MAX_RATIO=0.25
SELF_ATTENTION_DISTRIBUTION_QUERY_FRAME_COUNT=8
SELF_ATTENTION_DISTRIBUTION_GLOBAL_QUERY_TOKENS_PER_FRAME=64
SELF_ATTENTION_DISTRIBUTION_OBJECT_QUERY_TOKEN_LIMIT_PER_FRAME=0
SAVE_VIDEO=True

for PROMPT in "${PROMPTS[@]}"; do
for SEED in "${SEEDS[@]}"; do
    echo "=================================================================================="
    echo "Running self_attention_distribution | prompt: $PROMPT | seed: $SEED"
    echo "=================================================================================="

    PROMPT_TAG=$(build_prompt_tag "$PROMPT")
    SAVE_DIR="$WORK_DIR/outputs_wan_2_1_${task}/self_attention_distribution/${PROMPT_TAG}/seed_${SEED}_shift_${SAMPLE_SHIFT}_guide_${SAMPLE_GUIDE_SCALE}"
    SUMMARY_FILE="$SAVE_DIR/self_attention_distribution_summary.json"
    if [ -f "$SUMMARY_FILE" ]; then
        echo "Summary already exists: $SUMMARY_FILE"
        echo "Skip."
        echo ""
        continue
    fi
    mkdir -p "$SAVE_DIR"

    REUSE_CROSS_ATTENTION_DIR="$WORK_DIR/outputs_wan_2_1_${task}/cross_attention_token_viz/${PROMPT_TAG}/seed_${SEED}_shift_${SAMPLE_SHIFT}_guide_${SAMPLE_GUIDE_SCALE}"

    python $ROOT_DIR/wan21_t2v_experiments/run_wan21_t2v_experiments.py \
        --experiment self_attention_distribution \
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
        --reuse_cross_attention_dir "$REUSE_CROSS_ATTENTION_DIR" \
        --self_attention_distribution_steps "$SELF_ATTENTION_DISTRIBUTION_STEPS" \
        --self_attention_distribution_layers "$SELF_ATTENTION_DISTRIBUTION_LAYERS" \
        --self_attention_distribution_branch "$SELF_ATTENTION_DISTRIBUTION_BRANCH" \
        --self_attention_distribution_reference_step $SELF_ATTENTION_DISTRIBUTION_REFERENCE_STEP \
        --self_attention_distribution_reference_layer $SELF_ATTENTION_DISTRIBUTION_REFERENCE_LAYER \
        --self_attention_distribution_reference_center_mode "$SELF_ATTENTION_DISTRIBUTION_REFERENCE_CENTER_MODE" \
        --self_attention_distribution_reference_center_power $SELF_ATTENTION_DISTRIBUTION_REFERENCE_CENTER_POWER \
        --self_attention_distribution_reference_center_quantile $SELF_ATTENTION_DISTRIBUTION_REFERENCE_CENTER_QUANTILE \
        --self_attention_distribution_support_radius_mode "$SELF_ATTENTION_DISTRIBUTION_SUPPORT_RADIUS_MODE" \
        --self_attention_distribution_support_radius_fixed $SELF_ATTENTION_DISTRIBUTION_SUPPORT_RADIUS_FIXED \
        --self_attention_distribution_support_radius_alpha $SELF_ATTENTION_DISTRIBUTION_SUPPORT_RADIUS_ALPHA \
        --self_attention_distribution_support_radius_min $SELF_ATTENTION_DISTRIBUTION_SUPPORT_RADIUS_MIN \
        --self_attention_distribution_support_radius_max_ratio $SELF_ATTENTION_DISTRIBUTION_SUPPORT_RADIUS_MAX_RATIO \
        --self_attention_distribution_query_frame_count $SELF_ATTENTION_DISTRIBUTION_QUERY_FRAME_COUNT \
        --self_attention_distribution_global_query_tokens_per_frame $SELF_ATTENTION_DISTRIBUTION_GLOBAL_QUERY_TOKENS_PER_FRAME \
        --self_attention_distribution_object_query_token_limit_per_frame $SELF_ATTENTION_DISTRIBUTION_OBJECT_QUERY_TOKEN_LIMIT_PER_FRAME \
        --save_video $SAVE_VIDEO

    echo "Finished self_attention_distribution | prompt: $PROMPT | seed: $SEED"
    echo ""
done
done
