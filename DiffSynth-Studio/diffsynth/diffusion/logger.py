import json, os, signal, threading, torch
from accelerate import Accelerator


class TensorBoardLogger:
    def __init__(self, log_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard is enabled. Run `tensorboard --logdir={log_dir}` to visualize the training progress.")

    def log(self, key, value, step):
        self.writer.add_scalar(key, value, step)

    def close(self):
        if self.writer is not None:
            self.writer.close()


class SwanLabLogger:
    def __init__(self, project_name="DiffSynth-Studio", log_dir=None):
        import swanlab
        project_name = os.environ.get("SWANLAB_PROJECT", project_name)
        self.swanlab = swanlab
        self.swanlab.init(project=project_name, logdir=log_dir)
        print(f"SwanLab is enabled. Project: {project_name}")

    def log(self, key, value, step):
        self.swanlab.log({key: value}, step=step)

    def close(self):
        self.swanlab.finish()


class LocalJSONMetricsLogger:
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, "metrics.jsonl")
        self.file = open(self.path, "a", buffering=1)
        print(f"Local metrics logging is enabled: {self.path}", flush=True)

    def log(self, key, value, step):
        self.file.write(json.dumps({"step": int(step), key: value}, ensure_ascii=True) + "\n")

    def close(self):
        self.file.close()


class WandbLogger:
    def __init__(self, project_name="DiffSynth-Studio", log_dir=None):
        project_name = os.environ.get("WANDB_PROJECT", project_name)
        timeout = int(os.environ.get("WANDB_INIT_TIMEOUT", "15"))
        requested_mode = os.environ.get("WANDB_MODE", "offline").strip().lower()
        allow_online = os.environ.get("WANDB_ALLOW_ONLINE", "0") == "1"

        self.wandb = None
        self.run = None
        self.disabled = False
        self.local_logger = None
        self.log_dir = log_dir
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)

        if requested_mode == "online" and not allow_online:
            print(
                "Warning: WANDB_MODE=online is ignored because WANDB_ALLOW_ONLINE is not 1. "
                "Using offline mode to avoid blocking training.",
                flush=True,
            )
            requested_mode = "offline"

        if requested_mode == "disabled":
            self.disabled = True
            print("Wandb is disabled.", flush=True)
            return

        if requested_mode in ("offline", "dryrun"):
            os.environ["WANDB_MODE"] = "offline"
        elif requested_mode == "online":
            os.environ["WANDB_MODE"] = "online"
        else:
            print(f"Warning: unsupported WANDB_MODE={requested_mode}; using offline mode.", flush=True)
            requested_mode = "offline"
            os.environ["WANDB_MODE"] = "offline"

        self._clear_stale_wandb_service_env()

        try:
            import wandb
            self.wandb = wandb
            settings = self.wandb.Settings(init_timeout=timeout)
            self.run = self._call_with_hard_timeout(
                lambda: self.wandb.init(project=project_name, dir=log_dir, mode=requested_mode, settings=settings),
                timeout,
                f"wandb {requested_mode} init",
            )
            if requested_mode == "offline":
                print(
                    f"Wandb is enabled offline. Project: {project_name}. "
                    f"Sync later with: wandb sync {log_dir}/wandb/offline-run-*",
                    flush=True,
                )
            else:
                print(f"Wandb is enabled online. Project: {project_name}. Init timeout: {timeout}s", flush=True)
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            self._fallback_to_local_json(
                f"Warning: wandb {requested_mode} init failed or timed out after {timeout}s: {exc}. "
                "Falling back to local metrics.jsonl and continuing training."
            )

    def _clear_stale_wandb_service_env(self):
        for key in (
            "WANDB_SERVICE",
            "WANDB_SERVICE_HOST",
            "WANDB_SERVICE_PORT",
            "WANDB_SERVICE_TRANSPORT",
            "WANDB_SERVICE_PID",
        ):
            os.environ.pop(key, None)

    def _call_with_hard_timeout(self, fn, timeout, label):
        if timeout <= 0 or threading.current_thread() is not threading.main_thread() or not hasattr(signal, "SIGALRM"):
            return fn()

        class WandbTimeout(TimeoutError):
            pass

        def timeout_handler(signum, frame):
            raise WandbTimeout(f"{label} exceeded {timeout}s")

        old_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout)
        try:
            return fn()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)

    def _fallback_to_local_json(self, message):
        self.disabled = False
        self.run = None
        print(message, flush=True)
        if self.log_dir is not None:
            self.local_logger = LocalJSONMetricsLogger(self.log_dir)
        else:
            self.disabled = True

    def log(self, key, value, step):
        if self.local_logger is not None:
            self.local_logger.log(key, value, step)
            return
        if self.disabled or self.run is None:
            return
        timeout = int(os.environ.get("WANDB_LOG_TIMEOUT", os.environ.get("WANDB_INIT_TIMEOUT", "15")))
        try:
            self._call_with_hard_timeout(lambda: self.wandb.log({key: value}, step=step), timeout, "wandb log")
        except Exception as exc:
            self._fallback_to_local_json(f"Warning: wandb log failed or timed out at step {step}: {exc}. Switching to local metrics.jsonl.")
            if self.local_logger is not None:
                self.local_logger.log(key, value, step)

    def close(self):
        if self.local_logger is not None:
            self.local_logger.close()
            return
        if self.disabled or self.run is None:
            return
        timeout = int(os.environ.get("WANDB_FINISH_TIMEOUT", os.environ.get("WANDB_INIT_TIMEOUT", "15")))
        try:
            self._call_with_hard_timeout(lambda: self.wandb.finish(), timeout, "wandb finish")
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            print(f"Warning: wandb finish failed or timed out after {timeout}s: {exc}", flush=True)


class ModelLogger:
    def __init__(
        self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x: x,
        enable_tensorboard_log=False,
        enable_swanlab_log=False, swanlab_project="DiffSynth-Studio",
        enable_wandb_log=False, wandb_project="DiffSynth-Studio",
    ):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0
        # Loggers
        self.enable_tensorboard_log = enable_tensorboard_log
        self.enable_swanlab_log = enable_swanlab_log
        self.swanlab_project = swanlab_project
        self.enable_wandb_log = enable_wandb_log
        self.wandb_project = wandb_project
        self.loggers = []
        self.loggers_initialized = False

    def init_loggers(self):
        if self.enable_tensorboard_log:
            self.loggers.append(TensorBoardLogger(os.path.join(self.output_path, "tensorboard_log")))
        if self.enable_swanlab_log:
            self.loggers.append(SwanLabLogger(project_name=self.swanlab_project, log_dir=os.path.join(self.output_path, "swanlab_log")))
        if self.enable_wandb_log:
            self.loggers.append(WandbLogger(project_name=self.wandb_project, log_dir=os.path.join(self.output_path, "wandb_log")))
        self.loggers_initialized = True

    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, **kwargs):
        self.num_steps += 1
        if accelerator.is_main_process:
            if not self.loggers_initialized:
                self.init_loggers()
            loss = kwargs.get("loss")
            if loss is not None:
                loss_value = float(loss.detach().float().cpu()) if torch.is_tensor(loss) else float(loss)
                for logger in self.loggers:
                    logger.log("loss", loss_value, self.num_steps)
            metrics = kwargs.get("metrics") or {}
            for key, value in metrics.items():
                if torch.is_tensor(value):
                    if value.numel() != 1:
                        continue
                    value = float(value.detach().float().cpu())
                elif isinstance(value, (int, float)):
                    value = float(value)
                else:
                    continue
                for logger in self.loggers:
                    logger.log(key, value, self.num_steps)
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")

    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        self.save_model(accelerator, model, f"epoch-{epoch_id}.safetensors")

    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
        for logger in self.loggers:
            logger.close()

    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
            lambda_state_dict = {
                key: value for key, value in state_dict.items()
                if key.startswith("spatial_rope_lambda.") or ".spatial_rope_lambda." in key
            }
            if lambda_state_dict:
                stem, ext = os.path.splitext(file_name)
                lambda_path = os.path.join(self.output_path, f"{stem}_lambda{ext}")
                accelerator.save(lambda_state_dict, lambda_path, safe_serialization=True)
