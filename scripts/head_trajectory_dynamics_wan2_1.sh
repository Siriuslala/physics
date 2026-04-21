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

# Empty -> all heads. Example: "L4H1,L7H8,L17H9"
HEAD_TRAJECTORY_DYNAMICS_HEADS=""
# Empty -> all available steps in reused maps. Planning-focused example: "1,2,3,4,5,6"
HEAD_TRAJECTORY_DYNAMICS_STEPS="1,2,3,4,5,6"
# Empty -> js,wasserstein
HEAD_TRAJECTORY_DYNAMICS_DISTANCE_METRICS=""
HEAD_TRAJECTORY_DYNAMICS_REFERENCE_STEP=50
HEAD_TRAJECTORY_DYNAMICS_REFERENCE_LAYER=27

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
        --reuse_cross_attention_dir "$REUSE_CROSS_ATTENTION_DIR"

    echo "Finished head_trajectory_dynamics | prompt: $PROMPT | seed: $SEED"
    echo ""
done
done
