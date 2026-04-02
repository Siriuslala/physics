# huggingface-cli download Wan-AI/Wan2.2-T2V-A14B
# huggingface-cli download Wan-AI/Wan2.2-TI2V-5B
# modelscope download Wan-AI/Wan2.2-TI2V-5B --local_dir ./Wan2.2-TI2V-5B
# modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-TI2V-A14B

cd /home/liyueyan/resources/Wan2_2

# size
# t2v-A14B: 720*1280, 1280*720, 480*832, 832*480, 704*1280, 1280*704, 1024*704, 704*1024
# ti2v-5b: 704*1280, 1280*704

NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
task="t2v-A14B"  # t2v-A14B ti2v-5B
SIZE="832*480"  # 
CKPT_DIR="/datacache/modelscope/Wan2.2-T2V-A14B"  # "/datacache/modelscope/Wan2.2-T2V-A14B", "/datacache/modelscope/Wan2.2-TI2V-5B"
PROMPT="A basketball falls to the ground and bounces up"
SAVE_DIR="/home/liyueyan/Interpretability/physics/outputs"

# prompts
# "A basketball falls to the ground and bounces up"
# "A basketball falls to the ground and bounces up several times, with the height of each bounce gradually decreasing until it comes to a stop."
# "A ball moving to the left squeezes a spring on the wall, and is then bounced to the right by the spring."
# "The cannon fired a shell into the air, and after reaching its highest point, the shell began to fall."

# multi 
torchrun --nproc_per_node=$NPROC_PER_NODE generate.py \
    --task $task \
    --size $SIZE \
    --ckpt_dir $CKPT_DIR \
    --dit_fsdp \
    --t5_cpu \
    --ulysses_size 8 \
    --offload_model True \
    --convert_model_dtype \
    --sample_steps 50 \
    --frame_num 121 \
    --save_diffusion_steps False \
    --diffusion_output_dir $SAVE_DIR/${task}_${SIZE}_diffusion_steps/"${PROMPT}" \
    --diffusion_sample_count 5 \
    --diffusion_summary_every 10 \
    --prompt "$PROMPT" \
    --save_file $SAVE_DIR/${task}_${SIZE}_"${PROMPT}".mp4

# mem
