source ./env.sh
cd $ROOT_DIR

# Intervention experiment: after Wan self-attention, mix a fraction of each
# token output with same-spatial-position outputs from nearby latent frames.

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

# Empty -> all steps/layers.
SELF_ATTN_KERNEL_STEPS="1,2,3,4,5,6"
SELF_ATTN_KERNEL_LAYERS=""
SELF_ATTN_KERNEL_BRANCH="cond"  # cond / uncond / both
SELF_ATTN_KERNEL_RADIUS=2
SELF_ATTN_KERNEL_SIGMA=1.0
SELF_ATTN_KERNEL_MIX_ALPHA=0.25

for PROMPT in "${PROMPTS[@]}"; do
for SEED in "${SEEDS[@]}"; do
    PROMPT_TAG=$(build_prompt_tag "$PROMPT")
    SAVE_DIR="$WORK_DIR/outputs_wan_2_1_${task}/self_attention_temporal_kernel/${PROMPT_TAG}/seed_${SEED}_shift_${SAMPLE_SHIFT}_guide_${SAMPLE_GUIDE_SCALE}_alpha_${SELF_ATTN_KERNEL_MIX_ALPHA}_radius_${SELF_ATTN_KERNEL_RADIUS}"
    SUMMARY_FILE="$SAVE_DIR/self_attention_temporal_kernel_summary.json"
    if [ -f "$SUMMARY_FILE" ]; then
        echo "Summary already exists: $SUMMARY_FILE"
        echo "Skip."
        continue
    fi
    mkdir -p "$SAVE_DIR"

    python $ROOT_DIR/wan21_t2v_experiments/run_wan21_t2v_experiments.py \
        --experiment self_attention_temporal_kernel \
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
        --self_attn_kernel_steps "$SELF_ATTN_KERNEL_STEPS" \
        --self_attn_kernel_layers "$SELF_ATTN_KERNEL_LAYERS" \
        --self_attn_kernel_branch "$SELF_ATTN_KERNEL_BRANCH" \
        --self_attn_kernel_radius $SELF_ATTN_KERNEL_RADIUS \
        --self_attn_kernel_sigma $SELF_ATTN_KERNEL_SIGMA \
        --self_attn_kernel_mix_alpha $SELF_ATTN_KERNEL_MIX_ALPHA
done
done
