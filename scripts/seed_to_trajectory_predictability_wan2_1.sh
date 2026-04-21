source ./env.sh
cd $ROOT_DIR

# Multi-seed analysis: test whether the final object trajectory is predictable
# from the initial latent noise and/or early cross-attention trajectories.

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

SEED_LIST="0,1,2,3,4,5,6,7"
PROMPTS=(
    "Against a pure white background, a basketball falls vertically from mid-air onto a wooden floor and bounces up several times."
)

TARGET_OBJECT_WORDS="basketball"
TARGET_VERB_WORDS="falls,bounces,up"

SEED_TO_TRAJECTORY_EARLY_STEPS="1,2,3,4,5,6"
SEED_TO_TRAJECTORY_REFERENCE_STEP=50
SEED_TO_TRAJECTORY_REFERENCE_LAYER=27
SEED_TO_TRAJECTORY_HEAD="mean"
# Set <= 0 to keep the original latent-frame trajectory length.
SEED_TO_TRAJECTORY_NUM_POINTS=0
SEED_TO_TRAJECTORY_RIDGE_ALPHA=0.001

for PROMPT in "${PROMPTS[@]}"; do
    PROMPT_TAG=$(build_prompt_tag "$PROMPT")
    SAVE_DIR="$WORK_DIR/outputs_wan_2_1_${task}/seed_to_trajectory_predictability/${PROMPT_TAG}/shift_${SAMPLE_SHIFT}_guide_${SAMPLE_GUIDE_SCALE}"
    SUMMARY_FILE="$SAVE_DIR/seed_to_trajectory_predictability_summary.json"
    if [ -f "$SUMMARY_FILE" ]; then
        echo "Summary already exists: $SUMMARY_FILE"
        echo "Skip."
        continue
    fi
    mkdir -p "$SAVE_DIR"

    python $ROOT_DIR/wan21_t2v_experiments/run_wan21_t2v_experiments.py \
        --experiment seed_to_trajectory_predictability \
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
        --seed 0 \
        --offload_model True \
        --target_object_words "$TARGET_OBJECT_WORDS" \
        --target_verb_words "$TARGET_VERB_WORDS" \
        --seed_list "$SEED_LIST" \
        --seed_to_trajectory_early_steps "$SEED_TO_TRAJECTORY_EARLY_STEPS" \
        --seed_to_trajectory_reference_step $SEED_TO_TRAJECTORY_REFERENCE_STEP \
        --seed_to_trajectory_reference_layer $SEED_TO_TRAJECTORY_REFERENCE_LAYER \
        --seed_to_trajectory_head "$SEED_TO_TRAJECTORY_HEAD" \
        --seed_to_trajectory_num_points $SEED_TO_TRAJECTORY_NUM_POINTS \
        --seed_to_trajectory_ridge_alpha $SEED_TO_TRAJECTORY_RIDGE_ALPHA \
        --cross_attn_chunk_size 1024 \
        --viz_num_frames 5
done
