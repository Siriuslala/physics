source ./env.sh
cd $ROOT_DIR

source ~/miniforge3/etc/profile.d/conda.sh
conda activate video

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

SELF_ATTENTION_MODULATION_STEPS=""
SELF_ATTENTION_MODULATION_LAYERS=""
SELF_ATTENTION_MODULATION_BRANCH="cond"
SELF_ATTENTION_MODULATION_STOP_AFTER_LAST_PROBE_STEP=True
SELF_ATTENTION_MODULATION_CHANNEL_PROFILE_TARGETS=""  # for channel visualization
SELF_ATTENTION_MODULATION_CHANNEL_PROFILE_TOPK=5
SELF_ATTENTION_MODULATION_PLOT_ONLY_FROM_SAVED=True
SAVE_VIDEO=False

for PROMPT in "${PROMPTS[@]}"; do
for SEED in "${SEEDS[@]}"; do
    echo "=================================================================================="
    echo "Running self_attention_modulation | prompt: $PROMPT | seed: $SEED"
    echo "=================================================================================="

    PROMPT_TAG=$(build_prompt_tag "$PROMPT")
    SAVE_DIR="$WORK_DIR/outputs_wan_2_1_${task}/self_attention_modulation/${PROMPT_TAG}/seed_${SEED}_shift_${SAMPLE_SHIFT}_guide_${SAMPLE_GUIDE_SCALE}"
    mkdir -p "$SAVE_DIR"

    python $ROOT_DIR/wan21_t2v_experiments/run_wan21_t2v_experiments.py \
        --experiment self_attention_modulation \
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
        --self_attention_modulation_steps "$SELF_ATTENTION_MODULATION_STEPS" \
        --self_attention_modulation_layers "$SELF_ATTENTION_MODULATION_LAYERS" \
        --self_attention_modulation_branch "$SELF_ATTENTION_MODULATION_BRANCH" \
        --self_attention_modulation_stop_after_last_probe_step $SELF_ATTENTION_MODULATION_STOP_AFTER_LAST_PROBE_STEP \
        --self_attention_modulation_channel_profile_targets "$SELF_ATTENTION_MODULATION_CHANNEL_PROFILE_TARGETS" \
        --self_attention_modulation_channel_profile_topk $SELF_ATTENTION_MODULATION_CHANNEL_PROFILE_TOPK \
        --self_attention_modulation_plot_only_from_saved $SELF_ATTENTION_MODULATION_PLOT_ONLY_FROM_SAVED \
        --save_video $SAVE_VIDEO

    echo "Finished self_attention_modulation | prompt: $PROMPT | seed: $SEED"
    echo ""
done
done
