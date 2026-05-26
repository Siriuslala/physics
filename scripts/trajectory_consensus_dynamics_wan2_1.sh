source ./env.sh
cd $ROOT_DIR

source ~/miniforge3/etc/profile.d/conda.sh
conda activate video

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

CLEAR_TRAJ_CROSS_ATTN_HEADS="L0H0,L0H1,L0H6,L0H7,L0H8,L0H9,L0H11,L1H1,L1H2,L1H3,L1H4,L1H5,L1H8,L1H10,L1H11,L2H1,L2H2,L2H4,L2H6,L2H7,L2H9,L2H10,L2H11,L3H2,L3H5,L3H7,L3H8,L3H9,L3H11,L4H4,L4H5,L4H6,L4H7,L4H10,L5H1,L5H4,L5H5,L5H6,L5H9,L6H0,L6H2,L6H4,L6H5,L6H6,L6H7,L6H8,L6H9,L6H11,L7H0,L7H1,L7H2,L7H3,L7H5,L7H6,L7H9,L7H11,L8H0,L8H2,L8H3,L8H5,L8H6,L8H7,L8H8,L8H9,L8H10,L8H11,L9H1,L9H2,L9H6,L9H7,L9H9,L9H10,L10H2,L10H3,L10H4,L10H5,L10H7,L11H0,L11H2,L11H3,L11H5,L11H6,L11H10,L11H11,L12H1,L12H5,L12H9,L12H10,L12H11,L13H0,L13H1,L13H3,L13H4,L13H6,L13H8,L13H9,L13H11,L14H0,L14H2,L14H4,L14H6,L15H0,L15H2,L15H5,L15H6,L15H8,L15H9,L15H10,L15H11,L16H2,L16H3,L16H4,L16H5,L16H7,L16H8,L16H9,L16H10,L16H11,L17H0,L17H3,L17H4,L17H5,L17H7,L17H11,L18H3,L18H11,L19H0,L19H1,L19H3,L19H6,L19H9,L19H10,L21H1,L21H2,L21H8,L21H9,L21H10,L21H11,L22H0,L22H2,L22H4,L22H11,L23H1,L23H4,L23H5,L23H7,L23H11,L24H0,L24H1,L24H2,L24H9,L24H10,L25H7,L25H8,L25H9,L26H0,L26H2,L26H6,L27H0,L27H1,L27H3,L27H4,L27H5,L27H6,L27H8,L27H9,L27H10,L27H11,L28H1,L28H3,L29H1,L29H2,L29H3,L29H4,L29H5,L29H7,L29H8,L29H10"
VAGUE_TRAJ_CROSS_ATTN_HEADS="L0H2,L0H4,L0H10,L1H6,L1H7,L3H3,L4H1,L4H2,L4H9,L4H11,L5H0,L5H7,L5H11,L6H1,L7H7,L7H8,L8H1,L8H4,L9H11,L10H0,L10H6,L10H8,L10H9,L10H10,L10H11,L11H1,L11H4,L12H0,L12H2,L13H2,L13H7,L14H3,L14H5,L14H8,L14H9,L14H10,L15H4,L15H7,L16H1,L16H6,L17H6,L17H9,L18H4,L18H5,L18H8,L18H9,L19H2,L19H4,L19H7,L19H8,L20H0,L20H9,L20H10,L21H3,L22H1,L22H10,L23H0,L23H10,L24H3,L24H4,L24H5,L24H6,L25H1,L25H2,L25H6,L25H11,L26H1,L26H7,L26H8,L26H9,L27H2,L28H5,L28H6,L28H7,L28H11,L29H6"


# ============================================================
# Stage guide
# - candidate_consensus:
#   offline stage, reuses cross_attention_token_viz maps only.
# - head_contribution:
#   runtime patch stage, runs exact zero-ablation for selected heads.
# - self_attention_coupling:
#   runtime probe stage, collects self-attention candidate coupling and
#   aggregates winner-versus-loser features plus temporal precedence.
# - plot_candidate / plot_head:
#   plot-only redraw modes from saved outputs.
# - plot_self:
#   plot-only redraw for self_attention_coupling from saved CSV files.
# - all:
#   sequentially run candidate_consensus, self_attention_coupling, then
#   head_contribution,
#   using the same SAVE_DIR so all outputs stay together.
#
# Dependency notes
# 1. candidate_consensus and head_contribution are not strict runtime
#    dependencies of each other.
# 2. We still recommend reusing the same SAVE_DIR for one prompt/seed so that
#    all stage outputs and later planned caches stay in one place.
# 3. plot_candidate requires the saved candidate CSV/PT files in SAVE_DIR.
#    If candidate-region PDFs are redrawn, the script still passes
#    REUSE_CROSS_ATTENTION_DIR because the first row uses original head maps.
# 4. plot_head requires the mode-specific head contribution CSV under:
#    SAVE_DIR/trajectory_consensus_head_contribution/[mode_tag]/trajectory_consensus_head_contribution.csv
#    where [mode_tag] is:
#    - exact_ablation
#    - taylor_approx
#    - direct_proxy
#    Ablation and direct_proxy are stored separately even if both are computed
#    in one run. The rendered figures are split under
#    SAVE_DIR/trajectory_consensus_head_contribution_plots/heatmaps/ and
#    SAVE_DIR/trajectory_consensus_head_contribution_plots/scatter_<distance_metric>/.
# 5. Early-alignment scatter plots additionally require
#    REUSE_HEAD_TRAJECTORY_DYNAMICS_DIR.
# ============================================================
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

# ==============================
# candidate_consensus | head_contribution | self_attention_coupling | plot_candidate | plot_head | plot_self| all
# RUN_MODE="candidate_consensus"  # candidate_consensus | plot_candidate
# RUN_MODE="head_contribution"  # head_contribution | plot_head
RUN_MODE="plot_self"  # self_attention_coupling | plot_self

# For candidate region extraction & head contribution. Empty -> all heads
CROSS_ATTN_TRAJ_TYPE="all"  # ! all, null, traj_clear, traj_vague, traj_clear_vague
if [ "$CROSS_ATTN_TRAJ_TYPE" == "traj_clear" ]; then
    CROSS_ATTN_HEADS="$CLEAR_TRAJ_CROSS_ATTN_HEADS"
elif [ "$CROSS_ATTN_TRAJ_TYPE" == "traj_vague" ]; then
    CROSS_ATTN_HEADS="$VAGUE_TRAJ_CROSS_ATTN_HEADS"
elif [ "$CROSS_ATTN_TRAJ_TYPE" == "traj_clear_vague" ]; then
    CROSS_ATTN_HEADS="$CLEAR_TRAJ_CROSS_ATTN_HEADS,$VAGUE_TRAJ_CROSS_ATTN_HEADS"
elif [ "$CROSS_ATTN_TRAJ_TYPE" == "all" ]; then
    CROSS_ATTN_HEADS=""
elif [ "$CROSS_ATTN_TRAJ_TYPE" == "null" ]; then
    CROSS_ATTN_HEADS="None"
else
    echo "Unknown traj type: $CROSS_ATTN_TRAJ_TYPE"
    exit 1
fi

# Optional scope controls
TRAJECTORY_CONSENSUS_STEPS="1,2,3,4,5,6,7,8,9,10"  # "1,2,3,4,5"；empty -> all available reused-map steps; "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
TRAJECTORY_CONSENSUS_LAYERS=""  # empty -> all available reused-map layers
TRAJECTORY_CONSENSUS_CROSS_HEADS="$CROSS_ATTN_HEADS"  # "" -> all cross-attention heads in selected layers; None -> no cross-attention head
TRAJECTORY_CONSENSUS_SELF_HEADS=""  # "" -> all self-attention heads in selected layers; None -> no self-attention head
TRAJECTORY_CONSENSUS_MODULES="cross"  # enable `self` only when TRAJECTORY_CONSENSUS_SELF_HEADS is not None
TRAJECTORY_CONSENSUS_BRANCH="cond"

# Candidate extraction
TRAJECTORY_CONSENSUS_CANDIDATE_BASE_QUANTILE=0.85
TRAJECTORY_CONSENSUS_CANDIDATE_SPLIT_QUANTILES="0.92,0.95,0.97"
TRAJECTORY_CONSENSUS_CANDIDATE_SMOOTH_RADIUS=1
TRAJECTORY_CONSENSUS_CANDIDATE_STABLE_PEAK_MIN_LEVELS=2
TRAJECTORY_CONSENSUS_CANDIDATE_PEAK_MERGE_DISTANCE=2.0
TRAJECTORY_CONSENSUS_CANDIDATE_PREPROCESS_WINSORIZE_QUANTILE=0.995
TRAJECTORY_CONSENSUS_CANDIDATE_PREPROCESS_DESPIKE_QUANTILE=0.98
TRAJECTORY_CONSENSUS_CANDIDATE_MIN_COMPONENT_AREA=4
TRAJECTORY_CONSENSUS_CANDIDATE_VIZ_NUM_FRAMES=10

# Head contribution
TRAJECTORY_CONSENSUS_OBJECT_MASK_REFERENCE_STEP=50
TRAJECTORY_CONSENSUS_OBJECT_MASK_REFERENCE_LAYER=27
TRAJECTORY_CONSENSUS_DO_ABLATION=True  # !
TRAJECTORY_CONSENSUS_CONTRIBUTION_METHOD="taylor_approx"  # ! exact_ablation | taylor_approx
TRAJECTORY_CONSENSUS_DO_DIRECT_PROXY=False  # !
TRAJECTORY_CONSENSUS_ABLATE_POSITION="pre_o"  # ! pre_o | post_o
TRAJECTORY_CONSENSUS_TAYLOR_OBJECT_ONLY=True  # ! only ablate object area
TRAJECTORY_CONSENSUS_TAYLOR_METRIC_SCOPE="obj"  # ! patching metric: obj | global
TRAJECTORY_CONSENSUS_TAYLOR_NUM_LATENT_FRAMES=10  # !
TRAJECTORY_CONSENSUS_TAYLOR_USE_GRADIENT_CHECKPOINTING=True
TRAJECTORY_CONSENSUS_REFERENCE_DISTANCE_METRIC="center_l2,support_overlap,js,hellinger,wasserstein_map"  # !
TRAJECTORY_CONSENSUS_SCATTER_OUTLIER_HEADS="L0H4,"  # !

# Self-attention candidate coupling
TRAJECTORY_CONSENSUS_SA_ANCHOR_STEP=49
TRAJECTORY_CONSENSUS_SA_ANCHOR_LAYER=27
TRAJECTORY_CONSENSUS_SA_COVERED_MASS_MIN=0.0
TRAJECTORY_CONSENSUS_SA_PRECEDENCE_PERSISTENCE=2  # whether a winner keeps leading for a long time

# Engineering controls
TRAJECTORY_CONSENSUS_SKIP_EXISTING_PLOTS=True  # if False, redraw existing plots
TRAJECTORY_CONSENSUS_NUM_WORKERS=16

run_once() {
    local prompt="$1"
    local seed="$2"
    local save_dir="$3"
    local stage_csv="$4"
    local plot_only_flag="$5"
    local reuse_cross_attention_dir="$6"
    local reuse_head_trajectory_dir="$7"

    mkdir -p "$save_dir"

    local cmd=(
        python $ROOT_DIR/wan21_t2v_experiments/run_wan21_t2v_experiments.py
        --experiment trajectory_consensus_dynamics \
        --wan21_root $ROOT_DIR/projects/Wan2_1 \
        --ckpt_dir $CKPT_DIR \
        --output_dir $save_dir \
        --task $task \
        --prompt "$prompt" \
        --size $SIZE \
        --frame_num $FRAME_NUM \
        --sample_solver unipc \
        --sampling_steps $SAMPLE_STEPS \
        --shift $SAMPLE_SHIFT \
        --guide_scale $SAMPLE_GUIDE_SCALE \
        --seed $seed \
        --offload_model True \
        --target_object_words "$TARGET_OBJECT_WORDS" \
        --target_verb_words "$TARGET_VERB_WORDS" \
        --reuse_cross_attention_dir "$reuse_cross_attention_dir" \
        --reuse_head_trajectory_dynamics_dir "$reuse_head_trajectory_dir" \
        --trajectory_consensus_stages "$stage_csv" \
        --trajectory_consensus_steps "$TRAJECTORY_CONSENSUS_STEPS" \
        --trajectory_consensus_layers "$TRAJECTORY_CONSENSUS_LAYERS" \
        --trajectory_consensus_plot_only_from_csv $plot_only_flag \
        --trajectory_consensus_skip_existing_plots $TRAJECTORY_CONSENSUS_SKIP_EXISTING_PLOTS \
        --trajectory_consensus_num_workers $TRAJECTORY_CONSENSUS_NUM_WORKERS
    )

    if [[ "$stage_csv" == *"candidate_consensus"* || "$stage_csv" == *"head_contribution"* ]]; then
        cmd+=(
            --trajectory_consensus_cross_heads "$TRAJECTORY_CONSENSUS_CROSS_HEADS"
        )
    fi

    if [[ "$stage_csv" == *"candidate_consensus"* ]]; then
        cmd+=(
            --trajectory_consensus_candidate_base_quantile $TRAJECTORY_CONSENSUS_CANDIDATE_BASE_QUANTILE
            --trajectory_consensus_candidate_split_quantiles "$TRAJECTORY_CONSENSUS_CANDIDATE_SPLIT_QUANTILES"
            --trajectory_consensus_candidate_smooth_radius $TRAJECTORY_CONSENSUS_CANDIDATE_SMOOTH_RADIUS
            --trajectory_consensus_candidate_stable_peak_min_levels $TRAJECTORY_CONSENSUS_CANDIDATE_STABLE_PEAK_MIN_LEVELS
            --trajectory_consensus_candidate_peak_merge_distance $TRAJECTORY_CONSENSUS_CANDIDATE_PEAK_MERGE_DISTANCE
            --trajectory_consensus_candidate_preprocess_winsorize_quantile $TRAJECTORY_CONSENSUS_CANDIDATE_PREPROCESS_WINSORIZE_QUANTILE
            --trajectory_consensus_candidate_preprocess_despike_quantile $TRAJECTORY_CONSENSUS_CANDIDATE_PREPROCESS_DESPIKE_QUANTILE
            --trajectory_consensus_candidate_min_component_area $TRAJECTORY_CONSENSUS_CANDIDATE_MIN_COMPONENT_AREA
            --trajectory_consensus_candidate_viz_num_frames $TRAJECTORY_CONSENSUS_CANDIDATE_VIZ_NUM_FRAMES
        )
    fi

    if [[ "$stage_csv" == *"head_contribution"* ]]; then
        cmd+=(
            --trajectory_consensus_self_heads "$TRAJECTORY_CONSENSUS_SELF_HEADS"
            --trajectory_consensus_modules "$TRAJECTORY_CONSENSUS_MODULES"
            --trajectory_consensus_branch $TRAJECTORY_CONSENSUS_BRANCH
            --trajectory_consensus_reference_distance_metric $TRAJECTORY_CONSENSUS_REFERENCE_DISTANCE_METRIC
            --trajectory_consensus_scatter_outlier_heads "$TRAJECTORY_CONSENSUS_SCATTER_OUTLIER_HEADS"
            --trajectory_consensus_do_ablation $TRAJECTORY_CONSENSUS_DO_ABLATION
            --trajectory_consensus_contribution_method $TRAJECTORY_CONSENSUS_CONTRIBUTION_METHOD
            --trajectory_consensus_ablate_position $TRAJECTORY_CONSENSUS_ABLATE_POSITION
            --trajectory_consensus_do_direct_proxy $TRAJECTORY_CONSENSUS_DO_DIRECT_PROXY
            --trajectory_consensus_object_mask_reference_step $TRAJECTORY_CONSENSUS_OBJECT_MASK_REFERENCE_STEP
            --trajectory_consensus_object_mask_reference_layer $TRAJECTORY_CONSENSUS_OBJECT_MASK_REFERENCE_LAYER
            --trajectory_consensus_taylor_object_only $TRAJECTORY_CONSENSUS_TAYLOR_OBJECT_ONLY
            --trajectory_consensus_taylor_num_latent_frames $TRAJECTORY_CONSENSUS_TAYLOR_NUM_LATENT_FRAMES
            --trajectory_consensus_taylor_metric_scope $TRAJECTORY_CONSENSUS_TAYLOR_METRIC_SCOPE
            --trajectory_consensus_taylor_use_gradient_checkpointing $TRAJECTORY_CONSENSUS_TAYLOR_USE_GRADIENT_CHECKPOINTING
        )
    fi

    if [[ "$stage_csv" == *"self_attention_coupling"* ]]; then
        cmd+=(
            --trajectory_consensus_self_heads "$TRAJECTORY_CONSENSUS_SELF_HEADS"
            --trajectory_consensus_branch $TRAJECTORY_CONSENSUS_BRANCH
            --trajectory_consensus_sa_anchor_step $TRAJECTORY_CONSENSUS_SA_ANCHOR_STEP
            --trajectory_consensus_sa_anchor_layer $TRAJECTORY_CONSENSUS_SA_ANCHOR_LAYER
            --trajectory_consensus_sa_covered_mass_min $TRAJECTORY_CONSENSUS_SA_COVERED_MASS_MIN
            --trajectory_consensus_sa_precedence_persistence $TRAJECTORY_CONSENSUS_SA_PRECEDENCE_PERSISTENCE
        )
    fi

    "${cmd[@]}"
}

for PROMPT in "${PROMPTS[@]}"; do
for SEED in "${SEEDS[@]}"; do
    echo "=================================================================================="
    echo "Running trajectory_consensus_dynamics | mode: $RUN_MODE | prompt: $PROMPT | seed: $SEED"
    echo "=================================================================================="

    PROMPT_TAG=$(build_prompt_tag "$PROMPT")
    SAVE_DIR="$WORK_DIR/outputs_wan_2_1_${task}/trajectory_consensus_dynamics/${PROMPT_TAG}/seed_${SEED}_shift_${SAMPLE_SHIFT}_guide_${SAMPLE_GUIDE_SCALE}"
    REUSE_CROSS_ATTENTION_DIR="$WORK_DIR/outputs_wan_2_1_${task}/cross_attention_token_viz/${PROMPT_TAG}/seed_${SEED}_shift_${SAMPLE_SHIFT}_guide_${SAMPLE_GUIDE_SCALE}"
    # REUSE_HEAD_TRAJECTORY_DYNAMICS_DIR="$WORK_DIR/outputs_wan_2_1_${task}/head_trajectory_dynamics/${PROMPT_TAG}/seed_${SEED}_shift_${SAMPLE_SHIFT}_guide_${SAMPLE_GUIDE_SCALE}/traj_clear_vague/head_trajectory_dynamics_metrics_hypothesis_attractor_motion_planning_region_on_preprocessed_on_center_mode_geometric_center"
    REUSE_HEAD_TRAJECTORY_DYNAMICS_DIR="$WORK_DIR/outputs_wan_2_1_${task}/head_trajectory_dynamics/${PROMPT_TAG}/seed_${SEED}_shift_${SAMPLE_SHIFT}_guide_${SAMPLE_GUIDE_SCALE}/all/hypothesis_attractor_motion_planning_region_on"

    case "$RUN_MODE" in
        candidate_consensus)
            run_once "$PROMPT" "$SEED" "$SAVE_DIR" "candidate_consensus" "False" "$REUSE_CROSS_ATTENTION_DIR" "$REUSE_HEAD_TRAJECTORY_DYNAMICS_DIR"
            ;;
        self_attention_coupling)
            run_once "$PROMPT" "$SEED" "$SAVE_DIR" "self_attention_coupling" "False" "$REUSE_CROSS_ATTENTION_DIR" "$REUSE_HEAD_TRAJECTORY_DYNAMICS_DIR"
            ;;
        head_contribution)
            run_once "$PROMPT" "$SEED" "$SAVE_DIR" "head_contribution" "False" "$REUSE_CROSS_ATTENTION_DIR" "$REUSE_HEAD_TRAJECTORY_DYNAMICS_DIR"
            ;;
        plot_candidate)
            run_once "$PROMPT" "$SEED" "$SAVE_DIR" "candidate_consensus" "True" "$REUSE_CROSS_ATTENTION_DIR" "$REUSE_HEAD_TRAJECTORY_DYNAMICS_DIR"
            ;;
        plot_self)
            run_once "$PROMPT" "$SEED" "$SAVE_DIR" "self_attention_coupling" "True" "$REUSE_CROSS_ATTENTION_DIR" "$REUSE_HEAD_TRAJECTORY_DYNAMICS_DIR"
            ;;
        plot_head)
            run_once "$PROMPT" "$SEED" "$SAVE_DIR" "head_contribution" "True" "$REUSE_CROSS_ATTENTION_DIR" "$REUSE_HEAD_TRAJECTORY_DYNAMICS_DIR"
            ;;
        all)
            run_once "$PROMPT" "$SEED" "$SAVE_DIR" "candidate_consensus" "False" "$REUSE_CROSS_ATTENTION_DIR" "$REUSE_HEAD_TRAJECTORY_DYNAMICS_DIR"
            run_once "$PROMPT" "$SEED" "$SAVE_DIR" "self_attention_coupling" "False" "$REUSE_CROSS_ATTENTION_DIR" "$REUSE_HEAD_TRAJECTORY_DYNAMICS_DIR"
            run_once "$PROMPT" "$SEED" "$SAVE_DIR" "head_contribution" "False" "$REUSE_CROSS_ATTENTION_DIR" "$REUSE_HEAD_TRAJECTORY_DYNAMICS_DIR"
            ;;
        *)
            echo "Unknown RUN_MODE: $RUN_MODE"
            exit 1
            ;;
    esac

    echo "Finished trajectory_consensus_dynamics | mode: $RUN_MODE | prompt: $PROMPT | seed: $SEED"
    echo ""
done
done
