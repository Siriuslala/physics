source ./env.sh
cd $ROOT_DIR

source ~/miniforge3/etc/profile.d/conda.sh
conda activate video

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
SEEDS=(8)
# SEEDS=($(seq 1 32))

PROMPTS=(
    "Against a pure white background, a basketball falls vertically from mid-air onto a wooden floor and bounces up several times."
)

# Manual mode: training-free axis-wise scaling.
ROPE_MODIFICATION_MODE="manual"
ROPE_MODIFICATION_LAMBDA_F=0.75
ROPE_MODIFICATION_LAMBDA_H=1.00
ROPE_MODIFICATION_LAMBDA_W=1.00

# Apply the modification only to the first five denoising steps.
ROPE_MODIFICATION_STEPS="1,2,3,4,5"

# Step-conditioned mode: uncomment if you want to use the trainable scale head.
# ROPE_MODIFICATION_MODE="step_conditioned"
# ROPE_MODIFICATION_STEP_CONDITIONED_HIDDEN_DIM=128
# ROPE_MODIFICATION_STEP_CONDITIONED_CHECKPOINT="/path/to/rope_scale_head_state_dict.pt"
ROPE_MODIFICATION_STEP_CONDITIONED_HIDDEN_DIM=128
ROPE_MODIFICATION_STEP_CONDITIONED_CHECKPOINT=""

for PROMPT in "${PROMPTS[@]}"; do
for SEED in "${SEEDS[@]}"; do
    echo "=================================================================================="
    echo "Running rope_modification | prompt: $PROMPT | seed: $SEED | mode: $ROPE_MODIFICATION_MODE"
    echo "=================================================================================="

    PROMPT_TAG=$(build_prompt_tag "$PROMPT")
    EXP_NAME="rope_modification"
    SAVE_DIR="$WORK_DIR/outputs_wan_2_1_${task}/${EXP_NAME}/${PROMPT_TAG}/seed_${SEED}_shift_${SAMPLE_SHIFT}_guide_${SAMPLE_GUIDE_SCALE}"
    SUMMARY_FILE="$SAVE_DIR/rope_modification_summary.json"
    if [ -f "$SUMMARY_FILE" ]; then
        echo "Summary already exists: $SUMMARY_FILE"
        echo "Skip."
        echo ""
        continue
    fi
    mkdir -p "$SAVE_DIR"

    python $ROOT_DIR/wan21_t2v_experiments/run_wan21_t2v_experiments.py \
        --experiment rope_modification \
        --wan21_root $ROOT_DIR/projects/Wan2_1 \
        --ckpt_dir $CKPT_DIR \
        --output_dir "$SAVE_DIR" \
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
        --rope_modification_mode $ROPE_MODIFICATION_MODE \
        --rope_modification_lambda_f $ROPE_MODIFICATION_LAMBDA_F \
        --rope_modification_lambda_h $ROPE_MODIFICATION_LAMBDA_H \
        --rope_modification_lambda_w $ROPE_MODIFICATION_LAMBDA_W \
        --rope_modification_steps "$ROPE_MODIFICATION_STEPS" \
        --rope_modification_step_conditioned_hidden_dim $ROPE_MODIFICATION_STEP_CONDITIONED_HIDDEN_DIM \
        --rope_modification_step_conditioned_checkpoint "$ROPE_MODIFICATION_STEP_CONDITIONED_CHECKPOINT"

    echo "Finished rope_modification | prompt: $PROMPT | seed: $SEED | mode: $ROPE_MODIFICATION_MODE"
    echo ""
done
done
