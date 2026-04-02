# huggingface-cli download Wan-AI/Wan2.2-T2V-A14B
# huggingface-cli download Wan-AI/Wan2.2-TI2V-5B
# modelscope download Wan-AI/Wan2.2-TI2V-5B --local_dir ./Wan2.2-TI2V-5B
# modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-TI2V-A14B

cd ./projects/Wan2_2

# size
# t2v-A14B: 720*1280, 1280*720, 480*832, 832*480, 704*1280, 1280*704, 1024*704, 704*1024
# ti2v-5b: 704*1280, 1280*704

export CUDA_VISIBLE_DEVICES=0

# task="t2v-A14B"  # t2v-A14B ti2v-5B
# SIZE="832*480"  # 
# CKPT_DIR="/datacache/modelscope/Wan2.2-T2V-A14B"  # "/datacache/modelscope/Wan2.2-T2V-A14B", "/datacache/modelscope/Wan2.2-TI2V-5B"
# PROMPT="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
# SAVE_DIR="/home/liyueyan/Interpretability/physics"

# ti2v-5B
# task="ti2v-5B"  
# SIZE="1280*704"  #
# CKPT_DIR="/datacache/modelscope/Wan2.2-TI2V-5B"
# SAMPLE_STEPS=50

# t2v-A14B
task="t2v-A14B"
SIZE="832*480"  # 
CKPT_DIR="/datacache/modelscope/Wan2.2-T2V-A14B"
SAMPLE_STEPS=40

SEED=41


PROMPT="A ball falls to the ground and bounces up"
SAVE_DIR="/home/liyueyan/Interpretability/physics/outputs_wan_2_2"

# prompts
# "A basketball falls to the ground and bounces up"
# "A basketball falls to the ground and bounces up several times, with the height of each bounce gradually decreasing until it comes to a stop."
# "A ping pong ball falls to the ground and bounces up"
# "A ball moving to the left squeezes a spring on the wall, and is then bounced to the right by the spring."
# "The cannon fired a shell into the air, and after reaching its highest point, the shell began to fall."

# 5s, 16fps,

# single
python generate.py \
    --task $task \
    --size $SIZE \
    --ckpt_dir $CKPT_DIR \
    --offload_model True \
    --convert_model_dtype \
    --sample_steps $SAMPLE_STEPS \
    --frame_num 81 \
    --base_seed $SEED \
    --save_diffusion_steps False \
    --diffusion_output_dir $SAVE_DIR/${task}_${SIZE}_diffusion_steps/"${PROMPT}" \
    --diffusion_sample_count 5 \
    --diffusion_summary_every 10 \
    --prompt "$PROMPT" \
    --save_file $SAVE_DIR/${task}_${SIZE}_"${PROMPT}"_sample_shift_${SAMPLE_SHIFT}_guide_scale_${SAMPLE_GUIDE_SCALE}_seed_${SEED}.mp4

# mem
# 5b: 24000M
# 14B: 50000M