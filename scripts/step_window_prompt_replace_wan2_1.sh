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

# ==============================
SEEDS=(23)
PROMPTS=(
    "A basketball falls to the ground."
)
REPLACEMENT_PROMPT="A basketball rises upward into the air."

# Replace prompt at selected steps.
# PROMPT_REPLACE_SCOPE controls behavior:
# - single_step: replace only on that step
# - from_step: replace on that step and all later steps
# REPLACE_COND_ONLY=True means only cond branch is replaced.
PROMPT_REPLACE_STEPS="1,2,3,5,8,13,21,34,50"
PROMPT_REPLACE_SCOPE="from_step"  # single_step / from_step
REPLACE_COND_ONLY=True

# ==============================
for PROMPT in "${PROMPTS[@]}"; do
for SEED in "${SEEDS[@]}"; do
    echo "=================================================================================="
    echo "Running step_window_prompt_replace | prompt: $PROMPT | seed: $SEED"
    echo "=================================================================================="

    PROMPT_TAG=$(build_prompt_tag "$PROMPT")
    SAVE_DIR="$WORK_DIR/outputs_wan_2_1_${task}/step_window_prompt_replace/${PROMPT_TAG}/seed_${SEED}_shift_${SAMPLE_SHIFT}_guide_${SAMPLE_GUIDE_SCALE}"
    SUMMARY_FILE="$SAVE_DIR/step_window_prompt_replace_summary.json"
    if [ -f "$SUMMARY_FILE" ]; then
        echo "Summary already exists: $SUMMARY_FILE"
        echo "Skip."
        echo ""
        continue
    fi
    mkdir -p "$SAVE_DIR"

    python $ROOT_DIR/wan21_t2v_experiments/run_wan21_t2v_experiments.py \
        --experiment step_window_prompt_replace \
        --wan21_root $ROOT_DIR/projects/Wan2_1 \
        --ckpt_dir $CKPT_DIR \
        --output_dir $SAVE_DIR \
        --task $task \
        --prompt "$PROMPT" \
        --replacement_prompt "$REPLACEMENT_PROMPT" \
        --size $SIZE \
        --frame_num $FRAME_NUM \
        --sample_solver unipc \
        --sampling_steps $SAMPLE_STEPS \
        --shift $SAMPLE_SHIFT \
        --guide_scale $SAMPLE_GUIDE_SCALE \
        --seed $SEED \
        --offload_model True \
        --prompt_replace_steps $PROMPT_REPLACE_STEPS \
        --prompt_replace_scope $PROMPT_REPLACE_SCOPE \
        --replace_cond_only $REPLACE_COND_ONLY

    echo "Finished step_window_prompt_replace | prompt: $PROMPT | seed: $SEED"
    echo ""
done
done
