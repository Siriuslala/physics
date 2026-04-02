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
SEEDS=(20)
# SEEDS=($(seq 1 32))

PROMPTS=(
    # "Against a pure white background, a basketball falls vertically from mid-air onto a wooden floor."
    "Against a pure white background, a basketball falls vertically from mid-air onto a wooden floor and bounces up several times."
    # "Against a pure white background, a basketball falls vertically from mid-air onto a wooden floor and bounces up several times, with the height of each bounce gradually decreasing until it comes to a stop."
)

TARGET_OBJECT_WORDS="basketball"
TARGET_VERB_WORDS="falls,bounces,up"

PROBE_STEPS="1,2,3,4,5,6,7,8,9,10"  # the timesteps to probe motion planning
MAAS_LAYERS="0,10,20,30,39"
MAAS_RADIUS=1
QUERY_FRAME_COUNT=8
PROBE_BRANCH="cond"
MOTION_TARGET_SOURCE="object_token_trajectory"   # object_token_trajectory / motion_centroid
OBJECT_TRAJ_HEAD="mean"
OBJECT_TRAJ_STEP=50  # this is to decide which step's cross-attention map to use for the object trajectory extraction
OBJECT_TRAJ_LAYER=27  # same as above


for PROMPT in "${PROMPTS[@]}"; do
for SEED in "${SEEDS[@]}"; do
    echo "=================================================================================="
    echo "Running joint_attention_suite | prompt: $PROMPT | seed: $SEED"
    echo "=================================================================================="

    PROMPT_TAG=$(build_prompt_tag "$PROMPT")
    SAVE_DIR="$WORK_DIR/outputs_wan_2_1_${task}/joint_attention_suite/${PROMPT_TAG}/seed_${SEED}_shift_${SAMPLE_SHIFT}_guide_${SAMPLE_GUIDE_SCALE}"
    SUMMARY_FILE="$SAVE_DIR/joint_attention_suite_summary.json"
    if [ -f "$SUMMARY_FILE" ]; then
        echo "Summary already exists: $SUMMARY_FILE"
        echo "Skip."
        echo ""
        continue
    fi
    mkdir -p "$SAVE_DIR"

    REUSE_CROSS_ATTN_DIR="$WORK_DIR/outputs_wan_2_1_${task}/cross_attention_token_viz/${PROMPT_TAG}/seed_${SEED}_shift_${SAMPLE_SHIFT}_guide_${SAMPLE_GUIDE_SCALE}"
    # REUSE_CROSS_ATTN_DIR=""

    CMD=(
        python $ROOT_DIR/wan21_t2v_experiments/run_wan21_t2v_experiments.py
        --experiment joint_attention_suite
        --wan21_root $ROOT_DIR/projects/Wan2_1
        --ckpt_dir $CKPT_DIR
        --output_dir $SAVE_DIR
        --task $task
        --prompt "$PROMPT"
        --size $SIZE
        --frame_num $FRAME_NUM
        --sample_solver unipc
        --sampling_steps $SAMPLE_STEPS
        --shift $SAMPLE_SHIFT
        --guide_scale $SAMPLE_GUIDE_SCALE
        --seed $SEED
        --offload_model True
        --target_object_words "$TARGET_OBJECT_WORDS"
        --target_verb_words "$TARGET_VERB_WORDS"
        --probe_steps $PROBE_STEPS
        --query_frame_count $QUERY_FRAME_COUNT
        --probe_branch $PROBE_BRANCH
        --probe_query_mode object_guided
        --maas_layers $MAAS_LAYERS
        --maas_radius $MAAS_RADIUS
        --motion_target_source $MOTION_TARGET_SOURCE
        --object_traj_head $OBJECT_TRAJ_HEAD
    )
    if [ -n "$OBJECT_TRAJ_STEP" ]; then
        CMD+=(--object_traj_step "$OBJECT_TRAJ_STEP")
    fi
    if [ -n "$OBJECT_TRAJ_LAYER" ]; then
        CMD+=(--object_traj_layer "$OBJECT_TRAJ_LAYER")
    fi
    if [ -n "$REUSE_CROSS_ATTN_DIR" ]; then
        CMD+=(--reuse_cross_attention_dir "$REUSE_CROSS_ATTN_DIR")
    fi

    "${CMD[@]}"

    echo "Finished joint_attention_suite | prompt: $PROMPT | seed: $SEED"
    echo ""
done
done
