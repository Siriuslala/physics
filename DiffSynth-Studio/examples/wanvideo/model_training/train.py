import torch, os, argparse, accelerate, warnings, time, json, random
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.core import load_state_dict
from diffsynth.models.wan_video_dit import WanSpatialRopeLambda
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_reproducibility_seed(seed, deterministic=False):
    if seed is None:
        return
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_alpha=None, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        resume_from_checkpoint=None, remove_prefix_in_ckpt=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        compact_cache=False,
        cache_timing=False,
        vae_encode_batch_size=0,
        wan_spatial_rope_lambda_enabled=False,
        wan_spatial_rope_lambda_scope="layer",
        wan_spatial_rope_lambda_learn_f=False,
        wan_spatial_rope_lambda_timestep_conditioned=True,
        wan_spatial_rope_lambda_hidden_dim=128,
        wan_spatial_rope_lambda_lr=None,
        wan_spatial_rope_lambda_beta=0.0,
        wan_spatial_rope_lambda_checkpoint=None,
        timestep_sampling_strategy="uniform",
        timestep_mixture_early_boundary=0.12,
        timestep_mixture_early_prob=0.5,
    ):
        super().__init__()
        # Honor the user-provided gradient checkpointing setting.
        # Wan video training can OOM without checkpointing, but the script should
        # remain the source of truth for controlled experiments.

        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        audio_processor_config = self.parse_path_or_model_id(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, audio_processor_config=audio_processor_config, redirect_common_files=False)
        split_lora_base_model = lora_base_model
        if wan_spatial_rope_lambda_enabled and not task.endswith(":data_process"):
            split_lora_base_model = "dit" if split_lora_base_model is None else split_lora_base_model
        self.pipe = self.split_pipeline_units(
            task, self.pipe, trainable_models, split_lora_base_model,
            remove_unnecessary_params=task.endswith(":data_process"),
        )
        self.resume_from_checkpoint(resume_from_checkpoint, remove_prefix_in_ckpt)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_alpha, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        self.wan_spatial_rope_lambda_enabled = bool(wan_spatial_rope_lambda_enabled) and not task.endswith(":data_process")
        self.wan_spatial_rope_lambda_lr = wan_spatial_rope_lambda_lr
        self.wan_spatial_rope_lambda_beta = float(wan_spatial_rope_lambda_beta or 0.0)
        if self.wan_spatial_rope_lambda_enabled:
            self.attach_wan_spatial_rope_lambda(
                scope=wan_spatial_rope_lambda_scope,
                learn_f=wan_spatial_rope_lambda_learn_f,
                timestep_conditioned=wan_spatial_rope_lambda_timestep_conditioned,
                hidden_dim=wan_spatial_rope_lambda_hidden_dim,
                checkpoint=wan_spatial_rope_lambda_checkpoint,
            )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.compact_cache = compact_cache
        self.cache_timing = cache_timing
        self.vae_encode_batch_size = vae_encode_batch_size
        self.timestep_sampling_strategy = str(timestep_sampling_strategy)
        self.timestep_mixture_early_boundary = float(timestep_mixture_early_boundary)
        self.timestep_mixture_early_prob = float(timestep_mixture_early_prob)

    def attach_wan_spatial_rope_lambda(self, scope, learn_f, timestep_conditioned, hidden_dim, checkpoint=None):
        dit = self.pipe.dit
        module = WanSpatialRopeLambda(
            num_layers=len(dit.blocks),
            num_heads=dit.blocks[0].num_heads if len(dit.blocks) > 0 else dit.num_heads,
            freq_dim=dit.freq_dim,
            scope=scope,
            learn_f=learn_f,
            timestep_conditioned=timestep_conditioned,
            hidden_dim=hidden_dim,
        ).to(device=next(dit.parameters()).device)
        if checkpoint is not None and str(checkpoint).strip():
            state = load_state_dict(str(checkpoint).strip())
            cleaned = {}
            for key, value in state.items():
                if key.startswith("pipe.dit.spatial_rope_lambda."):
                    key = key[len("pipe.dit.spatial_rope_lambda."):]
                elif key.startswith("spatial_rope_lambda."):
                    key = key[len("spatial_rope_lambda."):]
                cleaned[key] = value
            missing, unexpected = module.load_state_dict(cleaned, strict=False)
            print(f"Wan spatial RoPE lambda checkpoint loaded: {checkpoint}. missing={missing}, unexpected={unexpected}")
        dit.spatial_rope_lambda = module
        print(
            "Wan spatial RoPE lambda enabled: "
            f"scope={scope}, learn_f={learn_f}, timestep_conditioned={timestep_conditioned}, "
            f"hidden_dim={hidden_dim}, beta={self.wan_spatial_rope_lambda_beta}."
        )

    def get_optimizer_param_groups(self, default_lr, default_weight_decay):
        if not self.wan_spatial_rope_lambda_enabled:
            return self.trainable_modules()
        lambda_params, other_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if ".spatial_rope_lambda." in name:
                lambda_params.append(param)
            else:
                other_params.append(param)
        groups = []
        if other_params:
            groups.append({"params": other_params, "lr": default_lr, "weight_decay": default_weight_decay, "name": "main"})
        if lambda_params:
            lambda_lr = float(self.wan_spatial_rope_lambda_lr or default_lr)
            groups.append({"params": lambda_params, "lr": lambda_lr, "weight_decay": 0.0, "name": "spatial_rope_lambda"})
        return groups

    def get_training_metrics(self):
        if not self.wan_spatial_rope_lambda_enabled:
            return {}
        module = getattr(self.pipe.dit, "spatial_rope_lambda", None)
        if module is None:
            return {}
        metrics = module.summary()
        metrics["lambda/beta"] = float(self.wan_spatial_rope_lambda_beta)
        return metrics

    def get_wan_spatial_rope_lambda_regularization(self):
        if not self.wan_spatial_rope_lambda_enabled or self.wan_spatial_rope_lambda_beta <= 0:
            return None
        module = getattr(self.pipe.dit, "spatial_rope_lambda", None)
        if module is None:
            return None
        return module.regularization_loss() * self.wan_spatial_rope_lambda_beta
        
    def inject_timestep_sampling_inputs(self, inputs):
        inputs_shared, inputs_posi, inputs_nega = inputs
        inputs_shared["max_timestep_boundary"] = self.max_timestep_boundary
        inputs_shared["min_timestep_boundary"] = self.min_timestep_boundary
        inputs_shared["timestep_sampling_strategy"] = self.timestep_sampling_strategy
        inputs_shared["timestep_mixture_early_boundary"] = self.timestep_mixture_early_boundary
        inputs_shared["timestep_mixture_early_prob"] = self.timestep_mixture_early_prob
        return inputs_shared, inputs_posi, inputs_nega

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        if inputs_shared.get("framewise_decoding", False):
            # WanToDance global model
            inputs_shared["num_frames"] = 4 * (len(data["video"]) - 1) + 1
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def get_batched_t2v_inputs(self, batch, include_noise=True):
        """Build one batched Wan T2V SFT input from a dataloader batch.

        This fast path is intentionally limited to the standard text-to-video SFT
        case used by our Wan2.1 training scripts: each sample has a prompt and a
        video, while extra control inputs such as image/video conditioning are not
        present. The VAE wrapper still encodes videos one by one internally, but
        UMT5 prompt encoding and the trainable DiT forward run as a real batch.
        """
        if isinstance(batch, dict):
            batch = [batch]
        if self.task not in ("sft", "sft:data_process"):
            raise ValueError("Batched Wan T2V fast path only supports task='sft' or task='sft:data_process'.")
        if self.extra_inputs:
            raise ValueError("Batched Wan T2V fast path does not support extra_inputs.")
        if not batch:
            raise ValueError("Empty batch received by batched Wan T2V fast path.")

        pipe = self.pipe
        prompts = [data["prompt"] for data in batch]
        videos = [data["video"] for data in batch]
        if torch.is_tensor(videos[0]):
            num_frames = videos[0].shape[1]
            height = videos[0].shape[2]
            width = videos[0].shape[3]
            for video in videos:
                if not torch.is_tensor(video) or tuple(video.shape) != tuple(videos[0].shape):
                    raise ValueError("Batched Wan T2V fast path requires identical video tensor shapes in one batch.")
        else:
            height = videos[0][0].size[1]
            width = videos[0][0].size[0]
            num_frames = len(videos[0])
            for video in videos:
                if len(video) != num_frames or video[0].size[0] != width or video[0].size[1] != height:
                    raise ValueError("Batched Wan T2V fast path requires identical video shapes in one batch.")

        noise = None
        if include_noise:
            latent_frames = (num_frames - 1) // 4 + 1
            noise_shape = (len(batch), pipe.vae.model.z_dim, latent_frames, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
            noise = pipe.generate_noise(noise_shape, rand_device=pipe.device)

        timing = {}
        timer = time.perf_counter()
        pipe.load_models_to_device(["text_encoder"])
        ids, mask = pipe.tokenizer(prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = pipe.text_encoder(ids, mask)
        for i, seq_len in enumerate(seq_lens):
            context[i, seq_len:] = 0
        context = context.to(dtype=pipe.torch_dtype, device=pipe.device)
        if pipe.device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        timing["text"] = time.perf_counter() - timer

        timer = time.perf_counter()
        pipe.load_models_to_device(["vae"])
        video_is_tensor = torch.is_tensor(videos[0])
        if video_is_tensor:
            input_video = torch.stack(videos, dim=0).contiguous()
        else:
            video_tensors = [pipe.preprocess_video(video, device="cpu") for video in videos]
            input_video = torch.cat(video_tensors, dim=0).contiguous()
        timing["video_to_tensor"] = time.perf_counter() - timer

        timer = time.perf_counter()
        # WanVideoVAE.encode loops over the batch dimension internally. We call the
        # underlying encoder directly, but only move the current VAE micro-batch to
        # GPU. This keeps cache batches large without pinning the whole video batch
        # in GPU memory.
        vae_batch_size = self.vae_encode_batch_size if self.vae_encode_batch_size and self.vae_encode_batch_size > 0 else input_video.shape[0]
        input_latents = []
        cache_data_process = self.task == "sft:data_process"
        for start in range(0, input_video.shape[0], vae_batch_size):
            video_chunk = input_video[start:start + vae_batch_size].to(dtype=pipe.torch_dtype, device=pipe.device)
            if video_is_tensor:
                video_chunk = video_chunk * (2.0 / 255.0) - 1.0
            hidden_states = pipe.vae.single_encode(video_chunk, pipe.device).to(dtype=pipe.torch_dtype)
            if cache_data_process:
                input_latents.append(hidden_states.detach().cpu().contiguous())
                del hidden_states, video_chunk
                if pipe.device != "cpu" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                input_latents.append(hidden_states)
        input_latents = torch.cat(input_latents, dim=0)
        if not cache_data_process:
            input_latents = input_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        if pipe.device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        timing["vae"] = time.perf_counter() - timer
        if self.cache_timing and self.task == "sft:data_process":
            print(
                f"[cache timing] batch={len(batch)} text={timing['text']:.2f}s "
                f"video_to_tensor={timing['video_to_tensor']:.2f}s vae={timing['vae']:.2f}s",
                flush=True,
            )

        inputs = {
            "input_latents": input_latents,
            "context": context,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        if noise is not None:
            inputs["latents"] = noise
        return inputs

    def split_batch_by_video_shape(self, batch):
        """Group samples with identical video tensor shapes for batched Wan T2V SFT."""
        if isinstance(batch, dict):
            return [[batch]]
        groups = {}
        ordered_keys = []
        for data in batch:
            video = data["video"]
            key = tuple(video.shape) if torch.is_tensor(video) else (len(video), video[0].size[1], video[0].size[0])
            if key not in groups:
                groups[key] = []
                ordered_keys.append(key)
            groups[key].append(data)
        return [groups[key] for key in ordered_keys]

    def split_indexed_batch_by_video_shape(self, batch):
        """Group samples with identical video tensor shapes while preserving original batch indices."""
        if isinstance(batch, dict):
            return [([0], [batch])]
        groups = {}
        ordered_keys = []
        for data_id, data in enumerate(batch):
            video = data["video"]
            key = tuple(video.shape) if torch.is_tensor(video) else (len(video), video[0].size[1], video[0].size[0])
            if key not in groups:
                groups[key] = ([], [])
                ordered_keys.append(key)
            groups[key][0].append(data_id)
            groups[key][1].append(data)
        return [groups[key] for key in ordered_keys]

    def forward_batched_t2v_sft(self, batch):
        groups = self.split_batch_by_video_shape(batch)
        total_size = sum(len(group) for group in groups)
        loss = None
        for group in groups:
            inputs = self.get_batched_t2v_inputs(group)
            self.pipe.load_models_to_device(self.pipe.in_iteration_models)
            group_loss = FlowMatchSFTLoss(self.pipe, **inputs)
            group_loss = group_loss * (len(group) / total_size)
            loss = group_loss if loss is None else loss + group_loss
        return loss

    def is_cached_training_inputs(self, inputs):
        return (
            isinstance(inputs, tuple)
            and len(inputs) == 3
            and all(isinstance(item, dict) for item in inputs)
        )

    def is_cached_training_batch(self, batch):
        return (
            isinstance(batch, list)
            and len(batch) > 0
            and all(self.is_cached_training_inputs(item) for item in batch)
        )

    def merge_cached_training_batch(self, batch):
        """Merge per-sample compact Wan T2V cache tuples into one real batch."""
        if self.is_cached_training_inputs(batch):
            return batch
        if not self.is_cached_training_batch(batch):
            raise ValueError("Batched cached Wan T2V SFT expects a list of cached input tuples.")

        merged_parts = []
        for part_id in range(3):
            first_keys = set(batch[0][part_id].keys())
            for item in batch[1:]:
                item_keys = set(item[part_id].keys())
                if item_keys != first_keys:
                    raise ValueError(
                        "Batched cached Wan T2V SFT requires identical cache fields "
                        f"in every sample. Expected {sorted(first_keys)}, got {sorted(item_keys)}."
                    )

            merged = {}
            for key in batch[0][part_id]:
                values = [item[part_id][key] for item in batch]
                if torch.is_tensor(values[0]):
                    shape_tail = tuple(values[0].shape[1:])
                    for value in values:
                        if not torch.is_tensor(value):
                            raise ValueError(f"Cache field `{key}` mixes tensor and non-tensor values.")
                        if tuple(value.shape[1:]) != shape_tail:
                            raise ValueError(
                                f"Cache field `{key}` has mismatched non-batch shapes: "
                                f"{tuple(value.shape)} vs {tuple(values[0].shape)}."
                            )
                    merged[key] = torch.cat(values, dim=0).contiguous()
                else:
                    first_value = values[0]
                    if any(value != first_value for value in values[1:]):
                        raise ValueError(f"Cache field `{key}` must be identical across a batched cache input.")
                    merged[key] = first_value
            merged_parts.append(merged)
        return tuple(merged_parts)

    def forward_batched_cached_t2v_sft(self, batch):
        inputs = self.merge_cached_training_batch(batch)
        return self.forward({}, inputs=inputs)

    def override_runtime_training_flags(self, inputs):
        if not self.is_cached_training_inputs(inputs):
            return inputs
        inputs_shared, inputs_posi, inputs_nega = inputs
        inputs_shared = dict(inputs_shared)
        inputs_shared["use_gradient_checkpointing"] = self.use_gradient_checkpointing
        inputs_shared["use_gradient_checkpointing_offload"] = self.use_gradient_checkpointing_offload
        return inputs_shared, inputs_posi, inputs_nega

    def split_batched_t2v_cache_inputs(self, inputs, group_size):
        """Split batched Wan T2V inputs into per-sample compact cache tuples."""
        cache_items = []
        for data_id in range(group_size):
            item = (
                {
                    "input_latents": inputs["input_latents"][data_id:data_id + 1],
                    "max_timestep_boundary": inputs["max_timestep_boundary"],
                    "min_timestep_boundary": inputs["min_timestep_boundary"],
                    "use_gradient_checkpointing": inputs["use_gradient_checkpointing"],
                    "use_gradient_checkpointing_offload": inputs["use_gradient_checkpointing_offload"],
                },
                {"context": inputs["context"][data_id:data_id + 1]},
                {},
            )
            cache_items.append(self.compact_wan_t2v_sft_cache(item))
        return cache_items

    def forward_batched_t2v_sft_data_process(self, batch):
        """Compute compact Wan T2V SFT cache for a dataloader batch."""
        indexed_groups = self.split_indexed_batch_by_video_shape(batch)
        total_size = sum(len(group) for _, group in indexed_groups)
        cache_items = [None] * total_size
        for group_indices, group in indexed_groups:
            inputs = self.get_batched_t2v_inputs(group, include_noise=False)
            group_cache_items = self.split_batched_t2v_cache_inputs(inputs, len(group))
            for local_id, data_id in enumerate(group_indices):
                cache_items[data_id] = group_cache_items[local_id]
        return cache_items

    def _cache_tensor(self, value):
        if torch.is_tensor(value):
            if value.is_floating_point():
                value = value.to(dtype=torch.bfloat16)
            return value.detach().cpu().contiguous()
        return value

    def compact_wan_t2v_sft_cache(self, inputs):
        """Keep only fields needed by Wan T2V SFT training from cached data."""
        inputs_shared, inputs_posi, inputs_nega = inputs
        shared_keys = (
            "input_latents",
            "max_timestep_boundary",
            "min_timestep_boundary",
            "use_gradient_checkpointing",
            "use_gradient_checkpointing_offload",
        )
        posi_keys = ("context",)
        nega_keys = tuple()
        compact_shared = {key: self._cache_tensor(inputs_shared[key]) for key in shared_keys if key in inputs_shared and inputs_shared[key] is not None}
        compact_posi = {key: self._cache_tensor(inputs_posi[key]) for key in posi_keys if key in inputs_posi and inputs_posi[key] is not None}
        compact_nega = {key: self._cache_tensor(inputs_nega[key]) for key in nega_keys if key in inputs_nega and inputs_nega[key] is not None}
        return compact_shared, compact_posi, compact_nega

    def forward(self, data, inputs=None):
        if inputs is None and self.is_cached_training_inputs(data):
            return self.forward({}, inputs=data)
        if inputs is None and self.is_cached_training_batch(data):
            return self.forward_batched_cached_t2v_sft(data)
        if inputs is None and isinstance(data, list):
            if self.task == "sft:data_process":
                if self.compact_cache:
                    return self.forward_batched_t2v_sft_data_process(data)
                return [self.forward(item) for item in data]
            return self.forward_batched_t2v_sft(data)
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.override_runtime_training_flags(inputs)
        inputs = self.inject_timestep_sampling_inputs(inputs)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        reg_loss = self.get_wan_spatial_rope_lambda_regularization()
        if reg_loss is not None:
            loss = loss + reg_loss
        if self.compact_cache and self.task == "sft:data_process":
            loss = self.compact_wan_t2v_sft_cache(loss)
        return loss


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--timestep_sampling_strategy", type=str, default="uniform", choices=["uniform", "early_rest_mixture"], help="How to sample flow-match training timesteps.")
    parser.add_argument("--timestep_mixture_early_boundary", type=float, default=0.12, help="Boundary fraction rho for --timestep_sampling_strategy early_rest_mixture.")
    parser.add_argument("--timestep_mixture_early_prob", type=float, default=0.5, help="Probability of sampling the early interval for --timestep_sampling_strategy early_rest_mixture.")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--framewise_decoding", default=False, action="store_true", help="Enable it if this model is a WanToDance global model.")
    parser.add_argument(
        "--video_sampling_mode",
        type=str,
        default="prefix",
        choices=["prefix", "first_seconds_uniform"],
        help="How to sample video frames before resizing. 'prefix' keeps the original DiffSynth behavior; 'first_seconds_uniform' samples uniformly from the first --video_clip_seconds seconds.",
    )
    parser.add_argument("--video_clip_seconds", type=float, default=None, help="Temporal window in seconds for --video_sampling_mode first_seconds_uniform.")
    parser.add_argument("--compact_cache", default=False, action="store_true", help="For Wan T2V SFT data_process, save only compact cache fields required for training.")
    parser.add_argument("--cache_timing", default=False, action="store_true", help="Print timing breakdown for Wan T2V cache batches.")
    parser.add_argument("--video_output_format", type=str, default="pil", choices=["pil", "tensor_uint8"], help="Video format returned by the dataset loader.")
    parser.add_argument("--vae_encode_batch_size", type=int, default=0, help="Micro-batch size for Wan VAE encode. 0 means encode the whole batch at once.")
    parser.add_argument("--wan_spatial_rope_lambda_enabled", default=False, action="store_true", help="Enable learnable spatial RoPE lambda scaling in Wan self-attention.")
    parser.add_argument("--wan_spatial_rope_lambda_scope", type=str, default="layer", choices=["model", "layer", "head"], help="Sharing scope for spatial RoPE lambda parameters.")
    parser.add_argument("--wan_spatial_rope_lambda_learn_f", default=False, action="store_true", help="Also learn frame-axis lambda. Reserved for later ablations; current implementation keeps lambda_f fixed.")
    parser.add_argument("--wan_spatial_rope_lambda_timestep_conditioned", default=False, action="store_true", help="Enable timestep-conditioned lambda MLP g(e_t).")
    parser.add_argument("--wan_spatial_rope_lambda_hidden_dim", type=int, default=128, help="Hidden dimension of the timestep-conditioned lambda MLP.")
    parser.add_argument("--wan_spatial_rope_lambda_lr", type=float, default=None, help="Optional learning rate for spatial RoPE lambda parameters.")
    parser.add_argument("--wan_spatial_rope_lambda_beta", type=float, default=0.0, help="L2 regularization coefficient on effective log lambda.")
    parser.add_argument("--wan_spatial_rope_lambda_checkpoint", type=str, default=None, help="Optional checkpoint for the spatial RoPE lambda module.")
    parser.add_argument("--cache_prefetch_batches", type=int, default=0, help="Number of preprocessed cache batches to keep in a background prefetch queue during data_process. 0 disables it.")
    parser.add_argument("--cache_resume", default=False, action="store_true", help="Resume single-process cache generation by skipping existing continuous 0.pth..N.pth files.")
    parser.add_argument("--cache_skip_mismatched_shapes", default=False, action="store_true", help="During data_process, skip samples whose decoded video shape does not match --num_frames/--height/--width and log them to jsonl.")
    parser.add_argument("--cache_filter_invalid_latents", default=True, action=argparse.BooleanOptionalAction, help="During cache training, filter cached samples whose input_latents shape does not match --num_frames/--height/--width before building the DataLoader.")
    parser.add_argument("--cache_filter_invalid_latents_rebuild_index", default=False, action="store_true", help="Force rebuilding the cache input_latents validity index before training.")
    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    set_reproducibility_seed(args.seed, deterministic=args.deterministic)
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4 if not args.framewise_decoding else 1,
            time_division_remainder=1 if not args.framewise_decoding else 0,
            video_sampling_mode=args.video_sampling_mode,
            video_clip_seconds=args.video_clip_seconds,
            video_output_format=args.video_output_format,
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
            "wantodance_music_path": ToAbsolutePath(args.dataset_base_path),
        }
    )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        task=args.task,
        device="cpu" if (args.initialize_model_on_cpu or args.enable_model_cpu_offload) else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        compact_cache=args.compact_cache,
        cache_timing=args.cache_timing,
        vae_encode_batch_size=args.vae_encode_batch_size,
        wan_spatial_rope_lambda_enabled=args.wan_spatial_rope_lambda_enabled,
        wan_spatial_rope_lambda_scope=args.wan_spatial_rope_lambda_scope,
        wan_spatial_rope_lambda_learn_f=args.wan_spatial_rope_lambda_learn_f,
        wan_spatial_rope_lambda_timestep_conditioned=args.wan_spatial_rope_lambda_timestep_conditioned,
        wan_spatial_rope_lambda_hidden_dim=args.wan_spatial_rope_lambda_hidden_dim,
        wan_spatial_rope_lambda_lr=args.wan_spatial_rope_lambda_lr,
        wan_spatial_rope_lambda_beta=args.wan_spatial_rope_lambda_beta,
        wan_spatial_rope_lambda_checkpoint=args.wan_spatial_rope_lambda_checkpoint,
        timestep_sampling_strategy=args.timestep_sampling_strategy,
        timestep_mixture_early_boundary=args.timestep_mixture_early_boundary,
        timestep_mixture_early_prob=args.timestep_mixture_early_prob,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_swanlab_log=args.enable_swanlab_log,
        swanlab_project=args.swanlab_project,
        enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project,
    )
    if accelerator.is_main_process:
        os.makedirs(args.output_path, exist_ok=True)
        with open(os.path.join(args.output_path, "training_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True)
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
