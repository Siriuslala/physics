import os, math, json, torch, importlib, time, threading, queue
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger
from diffsynth.core import OffloadTrainingManager


def get_optimizer_class(customized_optimizer=None):
    if customized_optimizer is None:
        return torch.optim.AdamW
    else:
        module_name, class_name = customized_optimizer.rsplit(".", 1)
        module = importlib.import_module(module_name)
        print(f"Customized opimizer `{customized_optimizer}` imported.")
        return getattr(module, class_name)


def get_scheduler(optimizer, scheduler_type="constant", warmup_steps=0, total_steps=None):
    if total_steps is None or total_steps <= 0:
        total_steps = 1
    warmup_steps = max(0, int(warmup_steps))

    def lr_lambda(current_step):
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        if scheduler_type in (None, "constant", "constant_with_warmup"):
            return 1.0
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        if scheduler_type == "linear":
            return max(0.0, 1.0 - progress)
        if scheduler_type == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError(f"Unsupported lr scheduler: {scheduler_type}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def collate_training_batch(batch):
    return batch[0] if len(batch) == 1 else batch


def iter_micro_batches(batch):
    if isinstance(batch, list):
        return batch
    return [batch]


def compute_total_update_steps(dataset, dataset_batch_size, gradient_accumulation_steps, num_epochs, max_train_steps, num_processes=1):
    global_micro_batch = max(1, dataset_batch_size) * max(1, num_processes)
    global_update_batch = global_micro_batch * max(1, gradient_accumulation_steps)
    updates_per_epoch = math.ceil(len(dataset) / global_update_batch)
    total_steps = updates_per_epoch * max(1, num_epochs)
    if max_train_steps is not None and max_train_steps > 0:
        total_steps = min(total_steps, max_train_steps)
    return max(1, total_steps)


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    enable_model_cpu_offload: bool = False,
    enable_optimizer_cpu_offload: bool = False,
    cpu_offload_split_threshold: int = None,
    customized_optimizer: str = None,
    args = None,
    **kwargs,
):
    dataset_batch_size = 1
    gradient_accumulation_steps = accelerator.gradient_accumulation_steps
    max_train_steps = None
    lr_scheduler = "constant"
    warmup_ratio = 0.0
    warmup_steps = 0
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_epsilon = 1e-8
    max_grad_norm = 1.0
    enable_batched_sft = False
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        dataset_batch_size = args.dataset_batch_size
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        max_train_steps = args.max_train_steps
        lr_scheduler = args.lr_scheduler
        warmup_ratio = args.warmup_ratio
        warmup_steps = args.warmup_steps
        adam_beta1 = args.adam_beta1
        adam_beta2 = args.adam_beta2
        adam_epsilon = args.adam_epsilon
        max_grad_norm = args.max_grad_norm
        enable_batched_sft = args.enable_batched_sft
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold
        customized_optimizer = args.customized_optimizer

    total_update_steps = compute_total_update_steps(
        dataset, dataset_batch_size, gradient_accumulation_steps, num_epochs, max_train_steps,
        num_processes=accelerator.num_processes,
    )
    if warmup_steps <= 0 and warmup_ratio > 0:
        warmup_steps = int(total_update_steps * warmup_ratio)

    optimizer_class = get_optimizer_class(customized_optimizer)
    if hasattr(model, "get_optimizer_param_groups"):
        optimizer_params = model.get_optimizer_param_groups(learning_rate, weight_decay)
    else:
        optimizer_params = model.trainable_modules()
    optimizer = optimizer_class(
        optimizer_params,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(adam_beta1, adam_beta2),
        eps=adam_epsilon,
    )
    scheduler = get_scheduler(optimizer, lr_scheduler, warmup_steps, total_update_steps)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=dataset_batch_size, collate_fn=collate_training_batch, num_workers=num_workers)

    if enable_model_cpu_offload:
        optimizer, dataloader, scheduler = accelerator.prepare(optimizer, dataloader, scheduler)
        model.pipe.device = accelerator.device
        offload_manager = OffloadTrainingManager(model, accelerator.device, enable_optimizer_cpu_offload, cpu_offload_split_threshold)
    else:
        model.to(device=accelerator.device)
        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    initialize_deepspeed_gradient_checkpointing(accelerator)
    stop_training = False
    progress_bar = tqdm(
        total=total_update_steps,
        initial=model_logger.num_steps,
        disable=not accelerator.is_main_process,
        desc="Train steps",
    )
    for epoch_id in range(num_epochs):
        for batch in dataloader:
            with accelerator.accumulate(model):
                loss = None
                if enable_batched_sft:
                    loss = model(batch)
                    accelerator.backward(loss)
                    loss = loss.detach()
                else:
                    micro_batches = iter_micro_batches(batch)
                    micro_batch_count = max(1, len(micro_batches))
                    for data in micro_batches:
                        if dataset.load_from_cache:
                            micro_loss = model({}, inputs=data)
                        else:
                            micro_loss = model(data)
                        micro_loss = micro_loss / micro_batch_count
                        accelerator.backward(micro_loss)
                        loss = micro_loss.detach() if loss is None else loss + micro_loss.detach()
                if enable_model_cpu_offload:
                    offload_manager.after_backward()
                if accelerator.sync_gradients and max_grad_norm is not None and max_grad_norm > 0:
                    accelerator.clip_grad_norm_(accelerator.unwrap_model(model).trainable_modules(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if accelerator.sync_gradients:
                    metrics = {"lr": float(optimizer.param_groups[0]["lr"])}
                    for group_id, group in enumerate(optimizer.param_groups):
                        group_name = group.get("name", f"group_{group_id}")
                        metrics[f"lr/{group_name}"] = float(group["lr"])
                    unwrapped_model = accelerator.unwrap_model(model)
                    if hasattr(unwrapped_model, "get_training_metrics"):
                        metrics.update(unwrapped_model.get_training_metrics())
                    model_logger.on_step_end(accelerator, model, save_steps, loss=loss, metrics=metrics)
                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=float(loss.detach().float().cpu()) if loss is not None else None)
                    if max_train_steps is not None and max_train_steps > 0 and model_logger.num_steps >= max_train_steps:
                        stop_training = True
                        break
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
        if stop_training:
            break

    progress_bar.close()
    model_logger.on_training_end(accelerator, model, save_steps)



def find_cache_resume_start_id(folder):
    start_id = 0
    while os.path.exists(os.path.join(folder, f"{start_id}.pth")):
        start_id += 1
    numeric_ids = []
    if os.path.isdir(folder):
        for file_name in os.listdir(folder):
            stem, ext = os.path.splitext(file_name)
            if ext == ".pth" and stem.isdigit():
                numeric_ids.append(int(stem))
    if numeric_ids:
        max_id = max(numeric_ids)
        if max_id + 1 != start_id:
            print(
                f"Warning: cache resume found a gap in {folder}. "
                f"Continuous prefix ends at {start_id - 1}, but max cached id is {max_id}. "
                "Only the continuous prefix will be skipped.",
                flush=True,
            )
    return start_id


def get_cache_sample_video_shape(data):
    video = data.get("video") if isinstance(data, dict) else None
    if torch.is_tensor(video):
        return tuple(int(dim) for dim in video.shape)
    if isinstance(video, list) and len(video) > 0:
        first_frame = video[0]
        if hasattr(first_frame, "size"):
            return (len(video), int(first_frame.size[1]), int(first_frame.size[0]))
    return None


def get_expected_cache_video_shape(args):
    if args is None:
        return None
    if getattr(args, "height", None) is None or getattr(args, "width", None) is None or getattr(args, "num_frames", None) is None:
        return None
    video_output_format = getattr(args, "video_output_format", "pil")
    if video_output_format == "tensor_uint8":
        return (3, int(args.num_frames), int(args.height), int(args.width))
    return (int(args.num_frames), int(args.height), int(args.width))


def get_cache_sample_data_id(data):
    if isinstance(data, dict) and "__data_id__" in data:
        try:
            return int(data["__data_id__"])
        except Exception:
            return None
    return None


def make_json_safe(value):
    if isinstance(value, dict):
        return {str(key): make_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if torch.is_tensor(value):
        return {"tensor_shape": [int(dim) for dim in value.shape], "tensor_dtype": str(value.dtype)}
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if hasattr(value, "item"):
        try:
            return make_json_safe(value.item())
        except Exception:
            pass
    try:
        json.dumps(value, allow_nan=False)
        return value
    except Exception:
        return str(value)


def make_skipped_cache_record(data, expected_shape, actual_shape, reason):
    metadata = data.get("__metadata__") if isinstance(data, dict) else None
    if metadata is None and isinstance(data, dict):
        metadata = {key: value for key, value in data.items() if key != "video" and not key.startswith("__")}
    return {
        "data_id": get_cache_sample_data_id(data),
        "reason": reason,
        "expected_video_shape": list(expected_shape) if expected_shape is not None else None,
        "actual_video_shape": list(actual_shape) if actual_shape is not None else None,
        "metadata": make_json_safe(metadata),
    }


def get_cache_resume_cursor_path(folder):
    return os.path.join(folder, "_cache_resume_cursor.json")


def read_cache_resume_cursor(folder, expected_save_id):
    cursor_path = get_cache_resume_cursor_path(folder)
    if not os.path.exists(cursor_path):
        return None
    try:
        with open(cursor_path, "r") as f:
            cursor = json.load(f)
    except Exception as exc:
        print(f"Warning: failed to read cache resume cursor {cursor_path}: {exc}", flush=True)
        return None
    cursor_save_id = cursor.get("save_id")
    next_data_id = cursor.get("next_data_id")
    if cursor_save_id != expected_save_id or next_data_id is None:
        print(
            f"Warning: cache resume cursor {cursor_path} does not match existing cache files "
            f"(cursor save_id={cursor_save_id}, existing save_id={expected_save_id}). "
            "Falling back to cache-file count.",
            flush=True,
        )
        return None
    return int(next_data_id)


def write_cache_resume_cursor(folder, save_id, next_data_id):
    cursor_path = get_cache_resume_cursor_path(folder)
    tmp_path = f"{cursor_path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump({"save_id": int(save_id), "next_data_id": int(next_data_id)}, f)
    os.replace(tmp_path, cursor_path)

def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
    **kwargs,
):
    dataset_batch_size = 1
    cache_prefetch_batches = 0
    cache_resume = False
    cache_skip_mismatched_shapes = False
    enable_model_cpu_offload = False
    enable_optimizer_cpu_offload = False
    cpu_offload_split_threshold = None
    if args is not None:
        num_workers = args.dataset_num_workers
        dataset_batch_size = args.dataset_batch_size
        cache_prefetch_batches = args.cache_prefetch_batches
        cache_resume = args.cache_resume
        cache_skip_mismatched_shapes = args.cache_skip_mismatched_shapes
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold

    folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
    os.makedirs(folder, exist_ok=True)
    skipped_folder = os.path.dirname(model_logger.output_path.rstrip(os.sep))
    os.makedirs(skipped_folder, exist_ok=True)
    skipped_jsonl_path = os.path.join(skipped_folder, f"{os.path.basename(model_logger.output_path.rstrip(os.sep))}_skipped_cache_samples.jsonl")
    expected_video_shape = get_expected_cache_video_shape(args) if cache_skip_mismatched_shapes else None
    skipped_count = 0
    save_id = 0
    if cache_resume:
        if accelerator.num_processes != 1:
            print("Warning: --cache_resume is only supported for single-process cache. Ignoring it.", flush=True)
        else:
            save_id = find_cache_resume_start_id(folder)
            resume_start_id = read_cache_resume_cursor(folder, save_id) if cache_skip_mismatched_shapes else None
            if resume_start_id is None:
                resume_start_id = save_id
            if resume_start_id > 0:
                print(
                    f"Resuming cache from metadata row {resume_start_id}; "
                    f"existing valid cache files: {save_id}; cache folder: {folder}.",
                    flush=True,
                )
                dataset = torch.utils.data.Subset(dataset, range(resume_start_id, len(dataset)))
    if cache_skip_mismatched_shapes and expected_video_shape is not None:
        print(f"Skipping cache samples whose video shape is not {expected_video_shape}.", flush=True)
        print(f"Skipped cache samples will be appended to {skipped_jsonl_path}.", flush=True)

    # Keep DataLoader batch_size=1 for video cache generation. A raw sample contains
    # dozens of frames; sending a DataLoader batch of many such samples through
    # multiprocessing queues can block for a long time. We accumulate samples after
    # DataLoader and still run the model with dataset_batch_size samples at once.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        collate_fn=lambda x: x[0],
        num_workers=num_workers,
    )
    if enable_model_cpu_offload:
        dataloader = accelerator.prepare(dataloader)
        offload_manager = OffloadTrainingManager(model, accelerator.device, enable_optimizer_cpu_offload, cpu_offload_split_threshold)
        model.pipe.device = accelerator.device
    else:
        model.to(device=accelerator.device)
        model, dataloader = accelerator.prepare(model, dataloader)

    cache_batch = []
    cache_load_time = 0.0
    cache_process_time = 0.0
    data_timer = time.perf_counter()

    try:
        progress_total = math.ceil(len(dataloader) / max(1, dataset_batch_size))
    except TypeError:
        progress_total = None
    progress_bar = tqdm(
        total=progress_total,
        disable=not accelerator.is_main_process,
        desc="Cache batches",
    )

    def append_skipped_cache_sample(data, actual_shape, reason):
        nonlocal skipped_count
        if not accelerator.is_main_process:
            return
        record = make_skipped_cache_record(data, expected_video_shape, actual_shape, reason)
        with open(skipped_jsonl_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        skipped_count += 1

    def accept_cache_sample(data):
        if not cache_skip_mismatched_shapes or expected_video_shape is None:
            return True
        actual_shape = get_cache_sample_video_shape(data)
        if actual_shape == expected_video_shape:
            return True
        append_skipped_cache_sample(data, actual_shape, "video_shape_mismatch")
        return False

    def save_processed_cache(data):
        nonlocal save_id
        if isinstance(data, list):
            for item in data:
                save_path = os.path.join(folder, f"{save_id}.pth")
                torch.save(item, save_path)
                save_id += 1
        else:
            save_path = os.path.join(folder, f"{save_id}.pth")
            torch.save(data, save_path)
            save_id += 1

    def process_cache_batch(batch, cursor_next_data_id=None):
        nonlocal cache_process_time
        process_timer = time.perf_counter()
        data = batch[0] if len(batch) == 1 else batch
        with accelerator.accumulate(model):
            with torch.no_grad():
                data = model(data)
                save_processed_cache(data)
                if cache_resume and accelerator.num_processes == 1 and cursor_next_data_id is not None:
                    write_cache_resume_cursor(folder, save_id, cursor_next_data_id)
                if enable_model_cpu_offload:
                    offload_manager.after_backward()
        cache_process_time += time.perf_counter() - process_timer
        progress_bar.update(1)
        progress_bar.set_postfix(
            load=f"{cache_load_time:.1f}s",
            process_save=f"{cache_process_time:.1f}s",
            skipped=skipped_count,
        )

    def iter_cache_batches_sync():
        nonlocal cache_load_time, data_timer
        batch_cursor_next_data_id = None
        for data in dataloader:
            cache_load_time += time.perf_counter() - data_timer
            data_id = get_cache_sample_data_id(data)
            if data_id is not None:
                batch_cursor_next_data_id = data_id + 1
            if accept_cache_sample(data):
                cache_batch.append(data)
            if len(cache_batch) >= dataset_batch_size:
                yield list(cache_batch), batch_cursor_next_data_id
                cache_batch.clear()
            data_timer = time.perf_counter()
        if cache_batch:
            yield list(cache_batch), batch_cursor_next_data_id

    def iter_cache_batches_prefetch():
        batch_queue = queue.Queue(maxsize=max(1, cache_prefetch_batches))
        stop_token = object()

        def producer():
            nonlocal cache_load_time, data_timer
            try:
                local_batch = []
                batch_cursor_next_data_id = None
                for data in dataloader:
                    cache_load_time += time.perf_counter() - data_timer
                    data_id = get_cache_sample_data_id(data)
                    if data_id is not None:
                        batch_cursor_next_data_id = data_id + 1
                    if accept_cache_sample(data):
                        local_batch.append(data)
                    if len(local_batch) >= dataset_batch_size:
                        batch_queue.put((list(local_batch), batch_cursor_next_data_id))
                        local_batch.clear()
                    data_timer = time.perf_counter()
                if local_batch:
                    batch_queue.put((list(local_batch), batch_cursor_next_data_id))
                batch_queue.put(stop_token)
            except BaseException as exc:
                batch_queue.put(exc)

        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()
        while True:
            item = batch_queue.get()
            if item is stop_token:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
        producer_thread.join()

    cache_batch_iter = iter_cache_batches_prefetch() if cache_prefetch_batches and cache_prefetch_batches > 0 else iter_cache_batches_sync()
    for batch, cursor_next_data_id in cache_batch_iter:
        process_cache_batch(batch, cursor_next_data_id)
    progress_bar.close()

def initialize_deepspeed_gradient_checkpointing(accelerator: Accelerator):
    if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        if "activation_checkpointing" in ds_config:
            import deepspeed
            act_config = ds_config["activation_checkpointing"]
            deepspeed.checkpointing.configure(
                mpu_=None, 
                partition_activations=act_config.get("partition_activations", False),
                checkpoint_in_cpu=act_config.get("cpu_checkpointing", False),
                contiguous_checkpointing=act_config.get("contiguous_memory_optimization", False)
            )
        else:
            print("Do not find activation_checkpointing config in deepspeed config, skip initializing deepspeed gradient checkpointing.")
