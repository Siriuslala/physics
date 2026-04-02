source ../env.sh
cd $ROOT_DIR/projects/Wan2_1


NPROC_PER_NODE=2
export CUDA_VISIBLE_DEVICES=0,1

task="t2v-14B"
SIZE="832*480"
CKPT_DIR="$MODEL_DIR/Wan2.1-T2V-14B"
FRAME_NUM=81
SAMPLE_STEPS=50
SAMPLE_SHIFT=5.0
SAMPLE_GUIDE_SCALE=5.0

SEEDS=(14)
SEEDS=($(seq 1 32))

PROMPT="An egg falls to the ground."
PROMPTS=(
    # "A basketball falls to the ground."
    # "A basketball falls to the ground and bounces."
    # "A basketball falls to the ground and bounces up several times, with the height of each bounce gradually decreasing until it comes to a stop."
    # "A basketball falls to the ground from the air."
    # "A basketball falls to the ground from the air and bounces up several times."
    # "A basketball falls to the ground from the air and bounces up several times, with the height of each bounce gradually decreasing until it comes to a stop."
    "Against a pure white background, a basketball falls vertically from mid-air onto a wooden floor."
    "Against a pure white background, a basketball falls vertically from mid-air onto a wooden floor and bounces up several times."
    "Against a pure white background, a basketball falls vertically from mid-air onto a wooden floor and bounces up several times, with the height of each bounce gradually decreasing until it comes to a stop."
)


# prompts
# "A basketball falls to the ground and bounces up"
# "A basketball falls to the ground and bounces up several times, with the height of each bounce gradually decreasing until it comes to a stop."
# "A ball moving to the left squeezes a spring on the wall, and is then bounced to the right by the spring."
# "The cannon fired a shell into the air, and after reaching its highest point, the shell began to fall."

# multi gpus
for PROMPT in "${PROMPTS[@]}"; do
for SEED in "${SEEDS[@]}"; do
    echo "=================================================================================="
    echo "Starting inference for prompt: $PROMPT, seed: $SEED"
    echo "=================================================================================="
    
    # check if the output file already exists. If it does, skip the inference for this prompt and seed.
    SAVE_DIR="$WORK_DIR/outputs_wan_2_1_${task}/general/$(echo $PROMPT | tr ' ' '_')"
    OUTPUT_FILE="$SAVE_DIR/${task}_${SIZE}_frame_num_${FRAME_NUM}_seed_${SEED}_shift_${SAMPLE_SHIFT}_guide_${SAMPLE_GUIDE_SCALE}.mp4"
    if [ -f "$OUTPUT_FILE" ]; then
        echo "Output file already exists for prompt: $PROMPT, seed: $SEED. Skipping inference."
        echo ""
        continue
    fi
    mkdir -p "$SAVE_DIR"

    torchrun --nproc_per_node=$NPROC_PER_NODE generate.py \
        --task $task \
        --size $SIZE \
        --ckpt_dir $CKPT_DIR \
        --dit_fsdp \
        --t5_cpu \
        --ulysses_size $NPROC_PER_NODE \
        --offload_model True \
        --sample_steps $SAMPLE_STEPS \
        --frame_num $FRAME_NUM \
        --base_seed $SEED \
        --sample_shift $SAMPLE_SHIFT \
        --sample_guide_scale $SAMPLE_GUIDE_SCALE \
        --save_diffusion_steps False \
        --diffusion_output_dir $SAVE_DIR/${task}_${SIZE}_diffusion_steps/"${PROMPT}" \
        --diffusion_sample_count 5 \
        --diffusion_summary_every 10 \
        --prompt "$PROMPT" \
        --save_file $OUTPUT_FILE

    echo "Finished inference for prompt: $PROMPT, seed: $SEED"
    echo "" 
done
done

# mem (81 frames)
# 14B: 4gpus: 4*22000M, 2gpus: 2*37000M
# (129 frames)
# 14B: 2gpus: 2*39052M