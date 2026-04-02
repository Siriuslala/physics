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
# SEEDS=($(seq 1 32))

PROMPTS=(
    "Against a pure white background, a basketball falls vertically from mid-air onto a wooden floor and bounces up several times."
)

CAUSAL_FIRST_N_STEPS=5
CAUSAL_MODE="temporal"          # flat / temporal
ATTENTION_BACKEND="auto"        # auto / flash / torch_sdpa
RUN_BASELINE=False               # whether to run bidirectional baseline together
COLLECT_DT_PROFILE=True         # whether to save dt histogram for each variant
PROBE_STEPS="1,2,3,4,5,6,7,8,9,10"
QUERY_FRAME_COUNT=8

# ==============================
for PROMPT in "${PROMPTS[@]}"; do
for SEED in "${SEEDS[@]}"; do
    echo "=================================================================================="
    echo "Running causal_schedule | prompt: $PROMPT | seed: $SEED"
    echo "=================================================================================="

    PROMPT_TAG=$(build_prompt_tag "$PROMPT")
    SAVE_DIR="$WORK_DIR/outputs_wan_2_1_${task}/causal_schedule/${PROMPT_TAG}/seed_${SEED}_shift_${SAMPLE_SHIFT}_guide_${SAMPLE_GUIDE_SCALE}"
    SUMMARY_FILE="$SAVE_DIR/causal_schedule_summary.json"
    if [ -f "$SUMMARY_FILE" ]; then
        echo "Summary already exists: $SUMMARY_FILE"
        echo "Skip."
        echo ""
        continue
    fi
    mkdir -p "$SAVE_DIR"

    python $ROOT_DIR/wan21_t2v_experiments/run_wan21_t2v_experiments.py \
        --experiment causal_schedule \
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
        --causal_first_n_steps $CAUSAL_FIRST_N_STEPS \
        --causal_mode $CAUSAL_MODE \
        --attention_backend $ATTENTION_BACKEND \
        --run_baseline $RUN_BASELINE \
        --collect_dt_profile $COLLECT_DT_PROFILE \
        --probe_steps $PROBE_STEPS \
        --query_frame_count $QUERY_FRAME_COUNT

    echo "Finished causal_schedule | prompt: $PROMPT | seed: $SEED"
    echo ""
done
done

# 1.3B
# 24722M for causal_schedule (temporal + auto)