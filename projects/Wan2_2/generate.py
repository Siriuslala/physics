# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image, ImageDraw, ImageFont

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import merge_video_audio, save_video, str2bool


EXAMPLE_PROMPT = {
    "t2v-A14B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "i2v-A14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
    "ti2v-5B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "animate-14B": {
        "prompt": "视频中的人在做动作",
        "video": "",
        "pose": "",
        "mask": "",
    },
    "s2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
        "audio":
            "examples/talk.wav",
        "tts_prompt_audio":
            "examples/zero_shot_prompt.wav",
        "tts_prompt_text":
            "希望你以后能够做的比我还好呦。",
        "tts_text":
            "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
    },
}


class DiffusionStepExporter:

    def __init__(self,
                 output_root,
                 sample_count,
                 video_fps,
                 summary_every,
                 summary_file=None,
                 cell_width=320):
        self.output_root = output_root
        self.sample_count = sample_count
        self.video_fps = video_fps
        self.summary_every = summary_every
        self.summary_file = summary_file
        self.cell_width = cell_width
        self.step_dirs = {}
        self.step_frame_counts = {}

        os.makedirs(self.output_root, exist_ok=True)

    def _summary_frame_indices(self, frame_num):
        if frame_num <= 0:
            return []

        if self.sample_count == 1:
            return [0]

        max_required = int(round((self.sample_count - 1) * self.video_fps))
        if self.video_fps > 0 and frame_num - 1 >= max_required:
            return [
                min(int(round(i * self.video_fps)), frame_num - 1)
                for i in range(self.sample_count)
            ]

        return torch.linspace(
            0, frame_num - 1,
            steps=self.sample_count).round().to(torch.int64).tolist()

    def _to_pil(self, frame):
        frame = frame.detach().float().cpu().clamp(-1, 1)
        frame = ((frame + 1.0) * 127.5).round().to(torch.uint8)
        frame = frame.permute(1, 2, 0).contiguous().numpy()
        return Image.fromarray(frame)

    @staticmethod
    def _frame_file_name(frame_idx):
        return f"frame_{frame_idx:05d}.png"

    def export_step(self, step_idx, total_steps, timestep, video_tensor):
        del timestep
        frame_num = int(video_tensor.shape[1])
        step_dir = os.path.join(self.output_root, f"nfe_{step_idx:03d}")
        os.makedirs(step_dir, exist_ok=True)

        # Save all frames for this NFE step; visualization sampling happens later.
        for frame_idx in range(frame_num):
            img = self._to_pil(video_tensor[:, frame_idx])
            img_name = self._frame_file_name(frame_idx)
            img_path = os.path.join(step_dir, img_name)
            img.save(img_path)

        self.step_dirs[step_idx] = step_dir
        self.step_frame_counts[step_idx] = frame_num
        if step_idx == total_steps:
            logging.info(
                f"Saved full-frame diffusion snapshots for {len(self.step_dirs)} NFE steps to {self.output_root}"
            )

    def build_summary_image(self):
        if not self.step_dirs:
            return None

        selected_steps = sorted(
            [step for step in self.step_dirs if step % self.summary_every == 0]
        )
        if not selected_steps:
            selected_steps = [max(self.step_dirs.keys())]

        first_step = selected_steps[0]
        frame_num = self.step_frame_counts.get(first_step, 0)
        sample_indices = self._summary_frame_indices(frame_num)
        if not sample_indices:
            return None

        first_image_path = os.path.join(
            self.step_dirs[first_step],
            self._frame_file_name(sample_indices[0]))
        with Image.open(first_image_path) as probe:
            cell_w, cell_h = probe.size

        if self.cell_width > 0 and cell_w > self.cell_width:
            scale = self.cell_width / float(cell_w)
            cell_w = self.cell_width
            cell_h = max(1, int(round(cell_h * scale)))

        rows = len(selected_steps)
        cols = len(sample_indices)
        pad = 8
        left_margin = 120
        top_margin = 70
        right_margin = 20
        bottom_margin = 20

        canvas_w = left_margin + cols * cell_w + (cols - 1) * pad + right_margin
        canvas_h = top_margin + rows * cell_h + (rows - 1) * pad + bottom_margin
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()

        draw.text((left_margin, 16),
                  "sampling timestep (NFE) vs time (s)",
                  fill=(0, 0, 0),
                  font=font)

        for col_idx, frame_idx in enumerate(sample_indices):
            time_sec = frame_idx / self.video_fps if self.video_fps > 0 else float(
                frame_idx)
            label = f"time={time_sec:.1f}s"
            x = left_margin + col_idx * (cell_w + pad)
            draw.text((x, 46), label, fill=(0, 0, 0), font=font)

        for row_idx, step in enumerate(selected_steps):
            row_label = f"NFE={step}"
            y_top = top_margin + row_idx * (cell_h + pad)
            draw.text((10, y_top + cell_h // 2 - 6),
                      row_label,
                      fill=(0, 0, 0),
                      font=font)
            step_dir = self.step_dirs[step]
            step_frame_num = self.step_frame_counts.get(step, 0)
            if step_frame_num <= 0:
                continue
            for col_idx, frame_idx in enumerate(sample_indices):
                safe_frame_idx = min(frame_idx, step_frame_num - 1)
                frame_path = os.path.join(step_dir,
                                          self._frame_file_name(safe_frame_idx))
                with Image.open(frame_path) as tile:
                    tile = tile.convert("RGB")
                    if tile.size != (cell_w, cell_h):
                        bicubic = Image.Resampling.BICUBIC if hasattr(
                            Image, "Resampling") else Image.BICUBIC
                        tile = tile.resize((cell_w, cell_h), bicubic)
                    x = left_margin + col_idx * (cell_w + pad)
                    y = top_margin + row_idx * (cell_h + pad)
                    canvas.paste(tile, (x, y))

        summary_path = self.summary_file
        if summary_path is None:
            summary_path = os.path.join(
                self.output_root, f"nfe_summary_every_{self.summary_every}.png")
        canvas.save(summary_path)
        return summary_path


def _default_save_file(args):
    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    formatted_prompt = args.prompt.replace(" ", "_").replace("/", "_")[:50]
    suffix = '.mp4'
    return (
        f"{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_"
        f"{args.ulysses_size}_{formatted_prompt}_{formatted_time}" + suffix)


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
        args.image = EXAMPLE_PROMPT[args.task]["image"]
    if args.audio is None and args.enable_tts is False and "audio" in EXAMPLE_PROMPT[args.task]:
        args.audio = EXAMPLE_PROMPT[args.task]["audio"]
    if (args.tts_prompt_audio is None or args.tts_text is None) and args.enable_tts is True and "audio" in EXAMPLE_PROMPT[args.task]:
        args.tts_prompt_audio = EXAMPLE_PROMPT[args.task]["tts_prompt_audio"]
        args.tts_prompt_text = EXAMPLE_PROMPT[args.task]["tts_prompt_text"]
        args.tts_text = EXAMPLE_PROMPT[args.task]["tts_text"]

    if args.task == "i2v-A14B":
        assert args.image is not None, "Please specify the image path for i2v."

    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    if not 's2v' in args.task:
        assert args.size in SUPPORTED_SIZES[
            args.
            task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"

    if args.save_diffusion_steps:
        assert ("ti2v" in args.task or "t2v" in args.task), \
            "--save_diffusion_steps currently supports t2v/ti2v tasks only."
        assert args.diffusion_sample_count > 0, "--diffusion_sample_count should be positive."
        assert args.diffusion_summary_every > 0, "--diffusion_summary_every should be positive."
        assert args.diffusion_cell_width > 0, "--diffusion_cell_width should be positive."


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-A14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames of video are generated. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=None,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model paramerters dtype.")
    parser.add_argument(
        "--save_diffusion_steps",
        type=str2bool,
        default=False,
        help="Whether to save sampled frames for each diffusion step.")
    parser.add_argument(
        "--diffusion_output_dir",
        type=str,
        default=None,
        help="Output folder for per-step diffusion frames.")
    parser.add_argument(
        "--diffusion_sample_count",
        type=int,
        default=5,
        help="How many frames to sample per diffusion step.")
    parser.add_argument(
        "--diffusion_summary_every",
        type=int,
        default=10,
        help="Build summary image using one NFE every N steps.")
    parser.add_argument(
        "--diffusion_summary_file",
        type=str,
        default=None,
        help="Output file for summary visualization image.")
    parser.add_argument(
        "--diffusion_cell_width",
        type=int,
        default=320,
        help="Cell width in summary visualization image.")

    # animate
    parser.add_argument(
        "--src_root_path",
        type=str,
        default=None,
        help="The file of the process output path. Default None.")
    parser.add_argument(
        "--refert_num",
        type=int,
        default=77,
        help="How many frames used for temporal guidance. Recommended to be 1 or 5."
    )
    parser.add_argument(
        "--replace_flag",
        action="store_true",
        default=False,
        help="Whether to use replace.")
    parser.add_argument(
        "--use_relighting_lora",
        action="store_true",
        default=False,
        help="Whether to use relighting lora.")
    
    # following args only works for s2v
    parser.add_argument(
        "--num_clip",
        type=int,
        default=None,
        help="Number of video clips to generate, the whole video will not exceed the length of audio."
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to the audio file, e.g. wav, mp3")
    parser.add_argument(
        "--enable_tts",
        action="store_true",
        default=False,
        help="Use CosyVoice to synthesis audio")
    parser.add_argument(
        "--tts_prompt_audio",
        type=str,
        default=None,
        help="Path to the tts prompt audio file, e.g. wav, mp3. Must be greater than 16khz, and between 5s to 15s.")
    parser.add_argument(
        "--tts_prompt_text",
        type=str,
        default=None,
        help="Content to the tts prompt audio. If provided, must exactly match tts_prompt_audio")
    parser.add_argument(
        "--tts_text",
        type=str,
        default=None,
        help="Text wish to synthesize")
    parser.add_argument(
        "--pose_video",
        type=str,
        default=None,
        help="Provide Dw-pose sequence to do Pose Driven")
    parser.add_argument(
        "--start_from_ref",
        action="store_true",
        default=False,
        help="whether set the reference image as the starting point for generation"
    )
    parser.add_argument(
        "--infer_frames",
        type=int,
        default=80,
        help="Number of frames per clip, 48 or 80 or others (must be multiple of 4) for 14B s2v"
    )
    args = parser.parse_args()
    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1
        ), f"sequence parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, f"The number of ulysses_size should be equal to the world size."
        init_distributed_group()

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=args.image is not None)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=args.image is not None,
                device=rank)
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info(f"Input prompt: {args.prompt}")
    img = None
    if args.image is not None:
        img = Image.open(args.image).convert("RGB")
        logging.info(f"Input image: {args.image}")

    # prompt extend
    if args.use_prompt_extend:
        logging.info("Extending prompt ...")
        if rank == 0:
            prompt_output = prompt_expander(
                args.prompt,
                image=img,
                tar_lang=args.prompt_extend_target_lang,
                seed=args.base_seed)
            if prompt_output.status == False:
                logging.info(
                    f"Extending prompt failed: {prompt_output.message}")
                logging.info("Falling back to original prompt.")
                input_prompt = args.prompt
            else:
                input_prompt = prompt_output.prompt
            input_prompt = [input_prompt]
        else:
            input_prompt = [None]
        if dist.is_initialized():
            dist.broadcast_object_list(input_prompt, src=0)
        args.prompt = input_prompt[0]
        logging.info(f"Extended prompt: {args.prompt}")

    if rank == 0 and args.save_file is None:
        args.save_file = _default_save_file(args)

    step_exporter = None
    if rank == 0 and args.save_diffusion_steps:
        output_root = args.diffusion_output_dir
        if output_root is None:
            output_root = os.path.splitext(args.save_file)[0] + "_diffusion_steps"
        step_exporter = DiffusionStepExporter(
            output_root=output_root,
            sample_count=args.diffusion_sample_count,
            video_fps=cfg.sample_fps,
            summary_every=args.diffusion_summary_every,
            summary_file=args.diffusion_summary_file,
            cell_width=args.diffusion_cell_width,
        )
        logging.info(
            f"Saving per-step diffusion snapshots to {step_exporter.output_root}")

    if "t2v" in args.task:
        logging.info("Creating WanT2V pipeline.")
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )

        logging.info(f"Generating video ...")
        video = wan_t2v.generate(
            args.prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
            step_callback=step_exporter.export_step if step_exporter else None)
    elif "ti2v" in args.task:
        logging.info("Creating WanTI2V pipeline.")
        wan_ti2v = wan.WanTI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )

        logging.info(f"Generating video ...")
        video = wan_ti2v.generate(
            args.prompt,
            img=img,
            size=SIZE_CONFIGS[args.size],
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
            step_callback=step_exporter.export_step if step_exporter else None)
    elif "animate" in args.task:
        logging.info("Creating Wan-Animate pipeline.")
        wan_animate = wan.WanAnimate(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
            use_relighting_lora=args.use_relighting_lora
        )

        logging.info(f"Generating video ...")
        video = wan_animate.generate(
            src_root_path=args.src_root_path,
            replace_flag=args.replace_flag,
            refert_num = args.refert_num,
            clip_len=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)
    elif "s2v" in args.task:
        logging.info("Creating WanS2V pipeline.")
        wan_s2v = wan.WanS2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )
        logging.info(f"Generating video ...")
        video = wan_s2v.generate(
            input_prompt=args.prompt,
            ref_image_path=args.image,
            audio_path=args.audio,
            enable_tts=args.enable_tts,
            tts_prompt_audio=args.tts_prompt_audio,
            tts_prompt_text=args.tts_prompt_text,
            tts_text=args.tts_text,
            num_repeat=args.num_clip,
            pose_video=args.pose_video,
            max_area=MAX_AREA_CONFIGS[args.size],
            infer_frames=args.infer_frames,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
            init_first_frame=args.start_from_ref,
        )
    else:
        logging.info("Creating WanI2V pipeline.")
        wan_i2v = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )
        logging.info("Generating video ...")
        video = wan_i2v.generate(
            args.prompt,
            img,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)

    if rank == 0:
        if args.save_file is None:
            args.save_file = _default_save_file(args)

        logging.info(f"Saving generated video to {args.save_file}")
        save_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
        if "s2v" in args.task:
            if args.enable_tts is False:
                merge_video_audio(video_path=args.save_file, audio_path=args.audio)
            else:
                merge_video_audio(video_path=args.save_file, audio_path="tts.wav")

        if step_exporter is not None:
            summary_file = step_exporter.build_summary_image()
            if summary_file is not None:
                logging.info(f"Saved diffusion summary image to {summary_file}")
    del video

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
