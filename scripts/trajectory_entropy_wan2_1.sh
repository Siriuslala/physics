source ./env.sh
cd $ROOT_DIR

# size
# t2v-14B: '720*1280', '1280*720', '480*832', '832*480'
# t2v-1.3B: '480*832', '832*480'

export CUDA_VISIBLE_DEVICES=1

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

# (For step-wise) Empty -> all steps [1..SAMPLE_STEPS]
TRAJECTORY_ENTROPY_STEPS=""
# (For step-wise) -1 means last layer
TRAJECTORY_ENTROPY_STEPWISE_LAYER=27
# (For layer-wise) Empty -> same as TRAJECTORY_ENTROPY_STEPS (thus all steps by default)
TRAJECTORY_ENTROPY_LAYERWISE_STEPS=""
# (For head-wise) -1 means last layer; -2 means all layers
TRAJECTORY_ENTROPY_HEAD_LAYER=-2

SAVE_VIDEO=True

# ==============================
for PROMPT in "${PROMPTS[@]}"; do
for SEED in "${SEEDS[@]}"; do
    echo "=================================================================================="
    echo "Running trajectory_entropy | prompt: $PROMPT | seed: $SEED"
    echo "=================================================================================="

    PROMPT_TAG=$(build_prompt_tag "$PROMPT")
    SAVE_DIR="$WORK_DIR/outputs_wan_2_1_${task}/trajectory_entropy/${PROMPT_TAG}/seed_${SEED}_shift_${SAMPLE_SHIFT}_guide_${SAMPLE_GUIDE_SCALE}"
    SUMMARY_FILE="$SAVE_DIR/trajectory_entropy_summary.json"
    if [ -f "$SUMMARY_FILE" ]; then
        echo "Summary already exists: $SUMMARY_FILE"
        echo "Skip."
        echo ""
        continue
    fi
    mkdir -p "$SAVE_DIR"

    # Optional: reuse an existing cross_attention_token_viz output directory.
    # If non-empty, trajectory_entropy reads saved maps directly and skips resampling.
    REUSE_CROSS_ATTENTION_DIR="$WORK_DIR/outputs_wan_2_1_${task}/cross_attention_token_viz/${PROMPT_TAG}/seed_${SEED}_shift_${SAMPLE_SHIFT}_guide_${SAMPLE_GUIDE_SCALE}"
    # REUSE_CROSS_ATTENTION_DIR=""

    python $ROOT_DIR/wan21_t2v_experiments/run_wan21_t2v_experiments.py \
        --experiment trajectory_entropy \
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
        --trajectory_entropy_steps "$TRAJECTORY_ENTROPY_STEPS" \
        --trajectory_entropy_layerwise_steps "$TRAJECTORY_ENTROPY_LAYERWISE_STEPS" \
        --trajectory_entropy_stepwise_layer $TRAJECTORY_ENTROPY_STEPWISE_LAYER \
        --trajectory_entropy_head_layer $TRAJECTORY_ENTROPY_HEAD_LAYER \
        --trajectory_entropy_save_video $SAVE_VIDEO \
        --reuse_cross_attention_dir "$REUSE_CROSS_ATTENTION_DIR"

    echo "Finished trajectory_entropy | prompt: $PROMPT | seed: $SEED"
    echo ""
done
done
