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

TARGET_OBJECT_WORDS="basketball"
TARGET_VERB_WORDS="falls,bounces,up"

SELF_ATTENTION_VIZ_STEPS=$(seq -s, 1 $SAMPLE_STEPS)
SELF_ATTENTION_VIZ_QUERY_VIDEO_FRAME_INDICES="1,17,33,41,57,81"  # "1,17,33,41,57,81", "1,33,41,81"
SELF_ATTENTION_VIZ_OBJECT_QUERY_TOKEN_LIMIT_PER_FRAME=0  # ! 64, 32
SELF_ATTENTION_VIZ_NUM_VIZ_FRAMES=10
SELF_ATTENTION_VIZ_VIZ_FRAME_INDICES=""
SELF_ATTENTION_VIZ_SAVE_ATTENTION_PDFS=True
SELF_ATTENTION_VIZ_ATTENTION_PDF_SHARE_COLOR_SCALE=True  # !
SELF_ATTENTION_VIZ_SKIP_EXISTING_PDFS=True
SELF_ATTENTION_VIZ_STOP_AFTER_LAST_PROBE_STEP=True
DRAW_SELF_ATTENTION_MAPS_ONLY=True  # !
DRAW_SELF_ATTENTION_MAPS_PATH="$WORK_DIR/outputs_wan_2_1_t2v-1.3B/self_attention_viz/Against_a_pure_white_background,_a_basketball_falls_vertically_from_mid-air_onto_a_wooden_floor_and_bounces_up_several_times./seed_26_shift_5.0_guide_5.0/self_attention_viz_maps.pt"  # ! path to the map.pt file
SELF_ATTENTION_VIZ_VISUALIZATION_OUTPUT_DIR=""
SELF_ATTENTION_VIZ_NUM_WORKERS=16
SAVE_VIDEO=True

for PROMPT in "${PROMPTS[@]}"; do
for SEED in "${SEEDS[@]}"; do
    echo "=================================================================================="
    echo "Running self_attention_viz | prompt: $PROMPT | seed: $SEED"
    echo "=================================================================================="

    PROMPT_TAG=$(build_prompt_tag "$PROMPT")

    # For object area extraction
    CROSS_ATTN_DIR="$WORK_DIR/outputs_wan_2_1_${task}/cross_attention_token_viz/${PROMPT_TAG}/seed_${SEED}_shift_${SAMPLE_SHIFT}_guide_${SAMPLE_GUIDE_SCALE}"

    if [ "$SELF_ATTENTION_VIZ_ATTENTION_PDF_SHARE_COLOR_SCALE" = "True" ]; then
        EXP_NAME="self_attention_viz_shared_color_scale"
    else
        EXP_NAME="self_attention_viz"
    fi

    SAVE_DIR="$WORK_DIR/outputs_wan_2_1_${task}/${EXP_NAME}/${PROMPT_TAG}/seed_${SEED}_shift_${SAMPLE_SHIFT}_guide_${SAMPLE_GUIDE_SCALE}"
    SUMMARY_FILE="$SAVE_DIR/self_attention_viz_summary.json"
    if [ -f "$SUMMARY_FILE" ] && [ "${DRAW_SELF_ATTENTION_MAPS_ONLY,,}" = "false" ]; then
        echo "Summary already exists: $SUMMARY_FILE"
        echo "Skip."
        echo ""
        continue
    fi
    mkdir -p "$SAVE_DIR"

    python $ROOT_DIR/wan21_t2v_experiments/run_wan21_t2v_experiments.py \
        --experiment self_attention_viz \
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
        --reuse_cross_attention_dir "$CROSS_ATTN_DIR" \
        --self_attention_viz_steps "$SELF_ATTENTION_VIZ_STEPS" \
        --self_attention_viz_query_video_frame_indices "$SELF_ATTENTION_VIZ_QUERY_VIDEO_FRAME_INDICES" \
        --self_attention_viz_object_query_token_limit_per_frame $SELF_ATTENTION_VIZ_OBJECT_QUERY_TOKEN_LIMIT_PER_FRAME \
        --self_attention_viz_num_viz_frames $SELF_ATTENTION_VIZ_NUM_VIZ_FRAMES \
        --self_attention_viz_viz_frame_indices "$SELF_ATTENTION_VIZ_VIZ_FRAME_INDICES" \
        --self_attention_viz_save_attention_pdfs $SELF_ATTENTION_VIZ_SAVE_ATTENTION_PDFS \
        --self_attention_viz_attention_pdf_share_color_scale $SELF_ATTENTION_VIZ_ATTENTION_PDF_SHARE_COLOR_SCALE \
        --self_attention_viz_skip_existing_pdfs $SELF_ATTENTION_VIZ_SKIP_EXISTING_PDFS \
        --self_attention_viz_stop_after_last_probe_step $SELF_ATTENTION_VIZ_STOP_AFTER_LAST_PROBE_STEP \
        --draw_self_attention_maps_only $DRAW_SELF_ATTENTION_MAPS_ONLY \
        --draw_self_attention_maps_path "$DRAW_SELF_ATTENTION_MAPS_PATH" \
        --self_attention_viz_visualization_output_dir "$SELF_ATTENTION_VIZ_VISUALIZATION_OUTPUT_DIR" \
        --self_attention_viz_num_workers $SELF_ATTENTION_VIZ_NUM_WORKERS \
        --save_video $SAVE_VIDEO

    echo "Finished self_attention_viz | prompt: $PROMPT | seed: $SEED"
    echo ""
done
done
