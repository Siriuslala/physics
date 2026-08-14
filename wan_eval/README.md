# Wan Eval

`wan_eval` contains lightweight evaluation-side utilities for Wan2.1 video models.

Current entrypoint:

- `infer_wan_t2v.py`: batch inference for official `wan2.1 t2v` checkpoints using a JSONL file with required fields `id` and `prompt`.

The script keeps the official Wan weights, DiT, T5, VAE, and sampling solvers unchanged. It only adds a batch wrapper around the official `WanT2V.generate()` logic so multiple prompts can be denoised in parallel.

Example:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate video

python wan_eval/infer_wan_t2v.py \
  --input_jsonl /path/to/prompts.jsonl \
  --output_dir /path/to/output_videos \
  --ckpt_dir /path/to/Wan2.1-T2V-1.3B \
  --task t2v-1.3B \
  --size 832*480 \
  --frame_num 81 \
  --sample_steps 50 \
  --sample_shift 5.0 \
  --sample_solver unipc \
  --sample_guide_scale 5.0 \
  --batch_size 4
```

Each JSONL row may additionally include optional fields `negative_prompt` and `seed`. Outputs are saved as `output_dir/{id}.mp4`.


Additional features:

- `--model_path`: optional evaluation checkpoint path or experiment directory.
- `--gpu_ids`: comma-separated GPU ids for one-model-per-GPU sharded inference.
- `--spatial_rope_lambda_*`: optional official-Wan inference patch for the lambda-based RoPE method trained in DiffSynth.
