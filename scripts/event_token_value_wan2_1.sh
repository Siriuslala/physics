source ./env.sh
cd $ROOT_DIR

# Collect token-wise cross-attention value contributions. Event/action words
# reuse the same Wan tokenizer matching logic as cross_attention_token_viz.

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
EVENT_TOKEN_VALUE_WORDS="falls,bounces"
EVENT_TOKEN_VALUE_STEPS="1,2,3,4,5,6"
EVENT_TOKEN_VALUE_LAYERS="27"
EVENT_TOKEN_VALUE_BRANCH="cond"
EVENT_TOKEN_VALUE_CHUNK_SIZE=512
EVENT_TOKEN_VALUE_NUM_VIZ_FRAMES=10

for PROMPT in "${PROMPTS[@]}"; do
for SEED in "${SEEDS[@]}"; do
    PROMPT_TAG=$(build_prompt_tag "$PROMPT")
    SAVE_DIR="$WORK_DIR/outputs_wan_2_1_${task}/event_token_value/${PROMPT_TAG}/seed_${SEED}_shift_${SAMPLE_SHIFT}_guide_${SAMPLE_GUIDE_SCALE}"
    SUMMARY_FILE="$SAVE_DIR/event_token_value_summary.json"
    if [ -f "$SUMMARY_FILE" ]; then
        echo "Summary already exists: $SUMMARY_FILE"
        echo "Skip."
        continue
    fi
    mkdir -p "$SAVE_DIR"

    python $ROOT_DIR/wan21_t2v_experiments/run_wan21_t2v_experiments.py \
        --experiment event_token_value \
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
        --event_token_value_words "$EVENT_TOKEN_VALUE_WORDS" \
        --event_token_value_steps "$EVENT_TOKEN_VALUE_STEPS" \
        --event_token_value_layers "$EVENT_TOKEN_VALUE_LAYERS" \
        --event_token_value_branch "$EVENT_TOKEN_VALUE_BRANCH" \
        --event_token_value_chunk_size $EVENT_TOKEN_VALUE_CHUNK_SIZE \
        --event_token_value_num_viz_frames $EVENT_TOKEN_VALUE_NUM_VIZ_FRAMES
done
done
