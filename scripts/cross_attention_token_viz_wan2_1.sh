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
    # "Against a pure white background, a basketball falls vertically from mid-air onto a wooden floor."
    "Against a pure white background, a basketball falls vertically from mid-air onto a wooden floor and bounces up several times."
    # "Against a pure white background, a basketball falls vertically from mid-air onto a wooden floor and bounces up several times, with the height of each bounce gradually decreasing until it comes to a stop."
    # "Against a pure white background, there is a wooden horizontal surface, with one single wooden slope attached to its left end. One small green ball starts from rest at the top of the slope, slides straight along the slope the entire time with its speed increasing steadily and uniformly, then rolls rightward along the wooden horizontal surface after reaching it."
    # "Against a pure white background, a wooden cube block at the top of a smooth slope slides straight down the slope with steadily and uniformly increasing speed."
)

TARGET_OBJECT_WORDS="basketball"
TARGET_VERB_WORDS="falls,bounces,up"
# TARGET_OBJECT_WORDS="cube"  # "basketball"
# TARGET_VERB_WORDS="slope,slides"  # "falls,bounces,up"

CROSS_ATTN_STEPS=$(seq -s, 1 $SAMPLE_STEPS)  # $(seq -s, 1 $SAMPLE_STEPS)  "1,2,3"
VIZ_NUM_FRAMES=10
VIZ_FRAME_INDICES=""  # e.g. "0,8,16,24,32", non-empty overrides VIZ_NUM_FRAMES
TRAJ_STYLE="glow_arrow"
TRAJ_NUM_FRAMES=0  # 21
TRAJ_SMOOTH_RADIUS=2
TRAJ_POWER=1.5
TRAJ_QUANTILE=0.8
TRAJ_ARROW_STRIDE=4


SKIP_EXISTING_PDFS=True  # resume mode: if target pdf already exists (map/trajectory/timeline), skip drawing
SAVE_ATTENTION_PDFS=True  # save attention maps for each timestep -> layer -> head
SAVE_TRAJECTORY_PDFS=True  # save trajectory (in one picture) for each timestep -> layer -> head
SAVE_TRAJECTORY_TIMELINE_PDFS=True  # save trajectory timeline for each timestep -> layer -> head
TRAJECTORY_TIMELINE_NUM_FRAMES=10  # number of frames for trajectory timeline (default: fps=2)

ATTENTION_PDF_PER_FRAME_NORMALIZE=False  # whether to normalize each frame over H*W before drawing attention PDFs
ATTENTION_PDF_SHARE_COLOR_SCALE=True  # if True, all frames in one attention PDF share one vmin/vmax

DRAW_ATTENTION_MAP_ONLY=True  # whether to only visualize attention & trajectory via the saved attention maps
DRAW_ATTENTION_MAPS_PATH=""  # the path to the saved attention maps (.pt) for re-drawing; if empty, use the attention maps in the standard output dir
DRAW_ATTENTION_MAPS_PATH="/work/liyueyan/Interpretability/physics/outputs_wan_2_1_t2v-1.3B/cross_attention_token_viz/Against_a_pure_white_background,_a_basketball_falls_vertically_from_mid-air_onto_a_wooden_floor_and_bounces_up_several_times./seed_26_shift_5.0_guide_5.0/cross_attention_maps.pt"

# VISUALIZATION_OUTPUT_DIR="/work/liyueyan/Interpretability/physics/outputs_wan_2_1_t2v-1.3B/cross_attention_token_viz/Against_a_pure_white_background,_a_basketball_falls_vertically_from_mid-air_onto_a_wooden_floor_and_bounces_up_several_times./seed_26_shift_5.0_guide_5.0/re-draw"  # "/work/liyueyan/Interpretability/physics/outputs_wan_2_1_t2v-1.3B/cross_attention_token_viz/Against_a_pure_white_background,_a_basketball_falls_vertically_from_mid-air_onto_a_wooden_floor_and_bounces_up_several_times./seed_26_shift_5.0_guide_5.0/re-draw1"  # for re-drawing
VISUALIZATION_OUTPUT_DIR=""

STREAM_FLUSH_PER_STEP=False
PLOT_DURING_SAMPLING=False

# ==============================
for PROMPT in "${PROMPTS[@]}"; do
for SEED in "${SEEDS[@]}"; do
    echo "=================================================================================="
    echo "Running cross_attention_token_viz | prompt: $PROMPT | seed: $SEED"
    echo "=================================================================================="
    
    PROMPT_TAG=$(build_prompt_tag "$PROMPT")

    if [ "$ATTENTION_PDF_PER_FRAME_NORMALIZE" = "True" ]; then
        EXP_NAME="cross_attention_token_viz_per_frame_norm"
    else
        EXP_NAME="cross_attention_token_viz"
    fi
    if [ "$ATTENTION_PDF_SHARE_COLOR_SCALE" = "True" ]; then
        EXP_NAME="${EXP_NAME}_shared_color_scale"
    fi
    SAVE_DIR="$WORK_DIR/outputs_wan_2_1_${task}/${EXP_NAME}/${PROMPT_TAG}/seed_${SEED}_shift_${SAMPLE_SHIFT}_guide_${SAMPLE_GUIDE_SCALE}"
    SUMMARY_FILE="$SAVE_DIR/cross_attention_token_viz_summary.json"
    if [ -f "$SUMMARY_FILE" ] && [ "${DRAW_ATTENTION_MAP_ONLY,,}" = "false" ]; then
        echo "Summary already exists: $SUMMARY_FILE"
        echo "Skip."
        echo ""
        continue
    fi
    mkdir -p "$SAVE_DIR"

    python $ROOT_DIR/wan21_t2v_experiments/run_wan21_t2v_experiments.py \
        --experiment cross_attention_token_viz \
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
        --cross_attn_steps $CROSS_ATTN_STEPS \
        --viz_num_frames $VIZ_NUM_FRAMES \
        --viz_frame_indices "$VIZ_FRAME_INDICES" \
        --traj_enable True \
        --traj_style $TRAJ_STYLE \
        --traj_num_frames $TRAJ_NUM_FRAMES \
        --traj_smooth_radius $TRAJ_SMOOTH_RADIUS \
        --traj_power $TRAJ_POWER \
        --traj_quantile $TRAJ_QUANTILE \
        --traj_arrow_stride $TRAJ_ARROW_STRIDE \
        --traj_include_head_mean True \
        --save_attention_pdfs $SAVE_ATTENTION_PDFS \
        --attention_pdf_per_frame_normalize $ATTENTION_PDF_PER_FRAME_NORMALIZE \
        --attention_pdf_share_color_scale $ATTENTION_PDF_SHARE_COLOR_SCALE \
        --skip_existing_pdfs $SKIP_EXISTING_PDFS \
        --save_trajectory_pdfs $SAVE_TRAJECTORY_PDFS \
        --save_trajectory_timeline_pdfs $SAVE_TRAJECTORY_TIMELINE_PDFS \
        --trajectory_timeline_num_frames $TRAJECTORY_TIMELINE_NUM_FRAMES \
        --save_video True \
        --stream_flush_per_step $STREAM_FLUSH_PER_STEP \
        --plot_during_sampling $PLOT_DURING_SAMPLING \
        --draw_attention_map_only $DRAW_ATTENTION_MAP_ONLY \
        --draw_attention_maps_path "$DRAW_ATTENTION_MAPS_PATH" \
        --visualization_output_dir "$VISUALIZATION_OUTPUT_DIR"

    echo "Finished cross_attention_token_viz | prompt: $PROMPT | seed: $SEED"
    echo ""
done
done
