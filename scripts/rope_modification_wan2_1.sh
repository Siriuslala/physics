source ./env.sh
cd $ROOT_DIR

source ~/miniforge3/etc/profile.d/conda.sh
conda activate video

# size
# t2v-14B: '720*1280', '1280*720', '480*832', '832*480'
# t2v-1.3B: '480*832', '832*480'

export CUDA_VISIBLE_DEVICES=2

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

format_step_tag() {
    local csv="$1"
    if [ -z "$csv" ]; then
        echo "all"
    else
        echo "${csv//,/-}"
    fi
}

build_rope_output_filename() {
    local tags=()
    tags+=("lambda_modify_mode_${ROPE_MODIFICATION_MODE}")

    if [ "$ROPE_MODIFICATION_MODE" = "manual" ]; then
        tags+=("lambdaf_${ROPE_MODIFICATION_LAMBDA_F}")
        tags+=("lambdah_${ROPE_MODIFICATION_LAMBDA_H}")
        tags+=("lambdaw_${ROPE_MODIFICATION_LAMBDA_W}")
        tags+=("lambda_steps_$(format_step_tag "$ROPE_MODIFICATION_STEPS")")
    elif [ "$ROPE_MODIFICATION_MODE" = "spatial_temporal_reweight" ]; then
        tags+=("spatial_temporal_reweight_alpha_${ROPE_MODIFICATION_SPATIAL_TEMPORAL_REWEIGHT_ALPHA}")
        tags+=("spatial_temporal_reweight_steps_$(format_step_tag "$ROPE_MODIFICATION_STEPS")")
    elif [ "$ROPE_MODIFICATION_MODE" = "timestep_conditioned" ]; then
        tags+=("lambda_timestep_condition_mode_${ROPE_MODIFICATION_TIMESTEP_CONDITIONED_RESOLUTION}")
    fi

    if [ "$ROPE_MODIFICATION_SEMANTIC_RESIDUAL_ENABLED" = "True" ]; then
        tags+=("use_semantic_SA")
        tags+=("semantic_alpha_${ROPE_MODIFICATION_SEMANTIC_RESIDUAL_ALPHA}")
        tags+=("semantic_steps_$(format_step_tag "$ROPE_MODIFICATION_SEMANTIC_RESIDUAL_STEPS")")
        if [ "$ROPE_MODIFICATION_SEMANTIC_RESIDUAL_TIMESTEP_CONDITIONED" = "True" ]; then
            tags+=(
                "semantic_timestep_condition_mode_${ROPE_MODIFICATION_SEMANTIC_RESIDUAL_TIMESTEP_CONDITIONED_RESOLUTION}"
            )
        fi
    fi

    local filename="${tags[0]}"
    local i
    for ((i = 1; i < ${#tags[@]}; ++i)); do
        filename="${filename}-${tags[$i]}"
    done
    echo "${filename}.mp4"
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
SEEDS=(1 8 13 20 23 26 29)
# SEEDS=($(seq 1 32))

PROMPTS=(
    "Against a pure white background, a basketball falls vertically from mid-air onto a wooden floor and bounces up several times."
)

# ==============================
# RoPE based modification
# Apply the modification only to the first five denoising steps.
ROPE_MODIFICATION_STEPS="1,2,3,4"

# Manual mode: training-free axis-wise scaling.
ROPE_MODIFICATION_MODE="manual"
ROPE_MODIFICATION_LAMBDA_F=1.00  # 0.75
ROPE_MODIFICATION_LAMBDA_H=0.45  # 0.50, 0.75
ROPE_MODIFICATION_LAMBDA_W=0.45

# Timestep-conditioned mode (need training):
# params to train: lambda & timestep conditions
# - `global`: one 3-vector for all heads at each timestep
# - `head_aware`: one 3-vector per head at each timestep
# ROPE_MODIFICATION_MODE="timestep_conditioned"

# ==============================
# Spatial-temporal reweight: apply post-RoPE channel reweight before
# self-attention. Temporal channels receive sqrt(alpha); spatial channels
# receive sqrt(1-alpha).
# ROPE_MODIFICATION_MODE="spatial_temporal_reweight"
ROPE_MODIFICATION_SPATIAL_TEMPORAL_REWEIGHT_ALPHA=0.80
ROPE_MODIFICATION_TIMESTEP_CONDITIONED_RESOLUTION="global"
ROPE_MODIFICATION_TIMESTEP_CONDITIONED_HIDDEN_DIM=128
ROPE_MODIFICATION_TIMESTEP_CONDITIONED_CHECKPOINT=""

# ============================================================
# Semantic Residual Self-Attention: add a cross-frame semantic logit on top of the RoPE logit.
ROPE_MODIFICATION_SEMANTIC_RESIDUAL_ENABLED=False
ROPE_MODIFICATION_SEMANTIC_RESIDUAL_ALPHA=0.5
ROPE_MODIFICATION_SEMANTIC_RESIDUAL_STEPS="1,2,3,4,5"
ROPE_MODIFICATION_SEMANTIC_RESIDUAL_QUERY_CHUNK_SIZE=512

# Optional timestep-conditioned semantic residual alpha:
# - `global`: one alpha for all heads at each timestep
# - `head_aware`: one alpha per head at each timestep
ROPE_MODIFICATION_SEMANTIC_RESIDUAL_TIMESTEP_CONDITIONED=False
ROPE_MODIFICATION_SEMANTIC_RESIDUAL_TIMESTEP_CONDITIONED_RESOLUTION="global"
ROPE_MODIFICATION_SEMANTIC_RESIDUAL_TIMESTEP_CONDITIONED_HIDDEN_DIM=128
ROPE_MODIFICATION_SEMANTIC_RESIDUAL_TIMESTEP_CONDITIONED_CHECKPOINT=""


# ==============================
for PROMPT in "${PROMPTS[@]}"; do
for SEED in "${SEEDS[@]}"; do
    echo "=================================================================================="
    echo "Running rope_modification | prompt: $PROMPT | seed: $SEED | mode: $ROPE_MODIFICATION_MODE"
    echo "=================================================================================="

    PROMPT_TAG=$(build_prompt_tag "$PROMPT")
    EXP_NAME="rope_modification"
    SAVE_DIR="$WORK_DIR/outputs_wan_2_1_${task}/${EXP_NAME}/${PROMPT_TAG}/seed_${SEED}_shift_${SAMPLE_SHIFT}_guide_${SAMPLE_GUIDE_SCALE}"
    OUTPUT_MP4=$(build_rope_output_filename)
    OUTPUT_MP4_PATH="$SAVE_DIR/$OUTPUT_MP4"
    if [ -f "$OUTPUT_MP4_PATH" ]; then
        echo "Output already exists: $OUTPUT_MP4_PATH"
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
        --rope_modification_spatial_temporal_reweight_alpha $ROPE_MODIFICATION_SPATIAL_TEMPORAL_REWEIGHT_ALPHA \
        --rope_modification_steps "$ROPE_MODIFICATION_STEPS" \
        --rope_modification_timestep_conditioned_resolution $ROPE_MODIFICATION_TIMESTEP_CONDITIONED_RESOLUTION \
        --rope_modification_timestep_conditioned_hidden_dim $ROPE_MODIFICATION_TIMESTEP_CONDITIONED_HIDDEN_DIM \
        --rope_modification_timestep_conditioned_checkpoint "$ROPE_MODIFICATION_TIMESTEP_CONDITIONED_CHECKPOINT" \
        --rope_modification_semantic_residual_enabled $ROPE_MODIFICATION_SEMANTIC_RESIDUAL_ENABLED \
        --rope_modification_semantic_residual_alpha $ROPE_MODIFICATION_SEMANTIC_RESIDUAL_ALPHA \
        --rope_modification_semantic_residual_steps "$ROPE_MODIFICATION_SEMANTIC_RESIDUAL_STEPS" \
        --rope_modification_semantic_residual_query_chunk_size $ROPE_MODIFICATION_SEMANTIC_RESIDUAL_QUERY_CHUNK_SIZE \
        --rope_modification_semantic_residual_timestep_conditioned $ROPE_MODIFICATION_SEMANTIC_RESIDUAL_TIMESTEP_CONDITIONED \
        --rope_modification_semantic_residual_timestep_conditioned_resolution $ROPE_MODIFICATION_SEMANTIC_RESIDUAL_TIMESTEP_CONDITIONED_RESOLUTION \
        --rope_modification_semantic_residual_timestep_conditioned_hidden_dim $ROPE_MODIFICATION_SEMANTIC_RESIDUAL_TIMESTEP_CONDITIONED_HIDDEN_DIM \
        --rope_modification_semantic_residual_timestep_conditioned_checkpoint "$ROPE_MODIFICATION_SEMANTIC_RESIDUAL_TIMESTEP_CONDITIONED_CHECKPOINT"

    echo "Finished rope_modification | prompt: $PROMPT | seed: $SEED | mode: $ROPE_MODIFICATION_MODE"
    echo ""
done
done
