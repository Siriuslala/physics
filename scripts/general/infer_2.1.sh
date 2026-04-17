# hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B
# huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./Wan2.1-T2V-14B
# modelscope download Wan-AI/Wan2.1-T2V-1.3B --local_dir ./Wan2.1-T2V-1.3B
# modelscope download Wan-AI/Wan2.1-T2V-14B --local_dir ./Wan2.1-T2V-14B

source ../env.sh
cd $ROOT_DIR/projects/Wan2_1

# size
# t2v-A14B: '720*1280', '1280*720', '480*832', '832*480'
# t2v-1.3b: '480*832', '832*480'

export CUDA_VISIBLE_DEVICES=3
GPU_TAG="a800"

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

# t2v-1.3B
task="t2v-1.3B"  
SIZE="832*480"  #
CKPT_DIR="$MODEL_DIR/Wan2.1-T2V-1.3B"
FRAME_NUM=81
SAMPLE_STEPS=50
SAMPLE_SHIFT=5.0
SAMPLE_GUIDE_SCALE=5.0

# t2v-A14B
# task="t2v-14B"
# SIZE="832*480"  # 
# CKPT_DIR="$MODEL_DIR/Wan2.1-T2V-14B"
# FRAME_NUM=81
# SAMPLE_STEPS=50
# SAMPLE_SHIFT=5.0
# SAMPLE_GUIDE_SCALE=5.0

# SEEDS=(26)
SEEDS=($(seq 1 32))
# SEEDS=($(seq 33 1024))

PROMPT="A basketball falls to the ground and bounces up several times, with the height of each bounce gradually decreasing until it comes to a stop."
PROMPTS=(
    # "A basketball falls to the ground."
    # "A basketball falls to the ground and bounces."
    # "A basketball falls to the ground and bounces up several times, with the height of each bounce gradually decreasing until it comes to a stop."
    # "A basketball falls to the ground from the air."
    # "A basketball falls to the ground from the air and bounces up several times."
    # "A basketball falls to the ground from the air and bounces up several times, with the height of each bounce gradually decreasing until it comes to a stop."
    # "Against a pure white background, a basketball falls vertically from mid-air onto a wooden floor."
    # "Against a pure white background, a basketball falls vertically from mid-air onto a wooden floor and bounces up several times."
    # "Against a pure white background, a basketball falls vertically from mid-air onto a wooden floor and bounces up several times, with the height of each bounce gradually decreasing until it comes to a stop."

    # "Against a pure white background, two green balls of the same size and mass move toward each other from both ends of a wooden table at the same speed and rebound after collision."
    # "Against a pure white background, two green balls with identical size and mass are launched toward each other from the two ends of a wooden desktop at equal initial speeds, and undergo rebound following collision."

    # "Against a pure white background, a small green ball flies off a horizontal table with initial velocity and then falls onto the floor."
    # "Against a pure white background, a little green ball slides off a level table with some initial speed and then drops onto the floor."

    # "Against a pure white background, a small green ball rolls rightward across a wooden horizontal surface and rebounds after colliding with a vertical wall."
    # "Against a pure white background, a little green ball rolls to the right on a wooden flat surface and bounces back when it hits a vertical wall."
    # "Against a pure white background, there is a horizontal wooden tabletop with a wall leaning against its right side. A small green ball starts rolling uniformly and horizontally to the right from the left end of the tabletop, rebounds after colliding with the wall, and then rolls to the left."
    # "Against a pure white background, there is a flat wooden table with a wall next to its right edge. A little green ball begins rolling steadily to the right from the left end of the table, bounces back when it hits the wall, and then rolls leftwards."
    # "Against a pure white background, a wooden horizontal surface lies at the bottom of the frame, a vertical wall stands on the right side, and a small green ball is placed on the middle of the surface. The ball starts rolling slowly to the right from the center of the surface; upon colliding with the right wall, it rebounds and rolls leftward at a constant speed."

    # "Against a pure white background, a small green ball is thrown vertically upward from below, reaches its highest point, and then falls vertically back down to the ground."
    # "Against a pure white background, a small green ball is thrown from the bottom left corner toward the upper right, rises to its highest point, and then falls back down."

    "Against a pure white background, a wooden cube block at the top of a smooth slope slides straight down the slope with steadily and uniformly increasing speed."
    # "Against a pure white background, a wooden cube block starts from rest at the top of a smooth slope (higher on the left, lower on the right), slides straight down the slope with steadily and uniformly increasing speed, and never tips over or flips throughout the entire movement."
    # "Against a pure white background, a small green ball starts from rest at the top of a smooth slope (higher on the left, lower on the right), slides straight along the slope the entire time, with its speed increasing steadily and uniformly."
    # "Against a pure white background, there is one single wooden slope. A small green ball starts from rest at the top of the slope, slides straight along the slope the entire time, with its speed increasing steadily and uniformly."
    # "Against a pure white background, there is a wooden horizontal surface, with one single wooden slope attached to its left end. One small green ball starts from rest at the top of the slope, slides straight along the slope the entire time with its speed increasing steadily and uniformly, then rolls rightward along the wooden horizontal surface after reaching it."

    # "Against a pure white background, a lightweight, non-stretching string is fixed at its upper end, with a small metal ball tied to its lower end. The ball is pulled to one side at an angle from the vertical, released from rest, and swings back and forth repeatedly in a vertical plane with a constant swing range."
)   

# prompts
# "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
# "A basketball falls to the ground."
# "A basketball falls to the ground and bounces."
# "A basketball falls to the ground and bounces up"
# "A basketball falls to the ground and bounces up several times, with the height of each bounce gradually decreasing until it comes to a stop."
# "A ping pong ball falls to the ground and bounces up"
# "A ball moving to the left squeezes a spring on the wall, and is then bounced to the right by the spring."
# "The cannon fired a shell into the air, and after reaching its highest point, the shell began to fall."

# single gpu
for PROMPT in "${PROMPTS[@]}"; do
for SEED in "${SEEDS[@]}"; do
    echo "=================================================================================="
    echo "Starting inference for prompt: $PROMPT, seed: $SEED"
    echo "=================================================================================="
    
    # check if the output file already exists. If it does, skip the inference for this prompt and seed.
    PROMPT_TAG=$(build_prompt_tag "$PROMPT")
    SAVE_DIR="$WORK_DIR/outputs_wan_2_1_${task}/general_${GPU_TAG}/${PROMPT_TAG}"
    OUTPUT_FILE="$SAVE_DIR/${task}_${SIZE}_frame_num_${FRAME_NUM}_seed_${SEED}_shift_${SAMPLE_SHIFT}_guide_${SAMPLE_GUIDE_SCALE}.mp4"
    if [ -f "$OUTPUT_FILE" ]; then
        echo "Output file already exists for prompt: $PROMPT, seed: $SEED. Skipping inference."
        echo ""
        continue
    fi
    mkdir -p "$SAVE_DIR"

    python generate.py \
        --task $task \
        --size $SIZE \
        --ckpt_dir $CKPT_DIR \
        --offload_model True \
        --sample_steps $SAMPLE_STEPS \
        --frame_num $FRAME_NUM \
        --base_seed $SEED \
        --sample_shift $SAMPLE_SHIFT \
        --sample_guide_scale $SAMPLE_GUIDE_SCALE \
        --save_diffusion_steps False \
        --diffusion_output_dir $SAVE_DIR/${task}_${SIZE}_diffusion_steps \
        --diffusion_sample_count 5 \
        --diffusion_summary_every 10 \
        --prompt "$PROMPT" \
        --save_file $OUTPUT_FILE
    
    echo "Finished inference for prompt: $PROMPT, seed: $SEED"
    echo ""
done
done

# mem (81 frames)
# 1.3B: 14000M, 18000M
# 14B: 71874M, 63390M, 

# (129 frames)
# 14B: 76778M
