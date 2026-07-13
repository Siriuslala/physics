import json, os, re, signal, threading, time, torch
from accelerate import Accelerator


def parse_wandb_offline_run_dir(path):
    """Return (timespec, run_id) for an offline-run directory name."""
    if not path:
        return None, None
    name = os.path.basename(os.path.normpath(path))
    match = re.match(r"^offline-run-(\d{8}_\d{6})-(.+?)(?:-\d+)?$", name)
    if match is None:
        return None, None
    return match.group(1), match.group(2)


def wandb_timespec_to_timestamp(timespec):
    if not timespec:
        return None
    try:
        return time.mktime(time.strptime(timespec, "%Y%m%d_%H%M%S"))
    except ValueError:
        return None


def read_wandb_max_logged_step(run_dir, run_id=None):
    if not run_dir or not os.path.isdir(run_dir):
        return -1
    files = []
    if run_id:
        candidate = os.path.join(run_dir, f"run-{run_id}.wandb")
        if os.path.exists(candidate):
            files.append(candidate)
    if not files:
        for name in os.listdir(run_dir):
            if name.startswith("run-") and name.endswith(".wandb"):
                files.append(os.path.join(run_dir, name))
    max_step = -1
    for file_path in files:
        try:
            from wandb.proto import wandb_internal_pb2 as pb
            from wandb.sdk.internal.datastore import DataStore
            store = DataStore()
            store.open_for_scan(file_path)
            while True:
                data = store.scan_data()
                if data is None:
                    break
                record = pb.Record()
                record.ParseFromString(data)
                if record.history.item:
                    max_step = max(max_step, int(record.history.step.num))
        except Exception as exc:
            print(f"Warning: failed to scan wandb history from {file_path}: {exc}", flush=True)
    return max_step


def _save_wan_spatial_rope_lambda_heatmaps(output_path, file_name, model):
    """Save lambda layer/head heatmaps next to checkpoints without making checkpointing fragile."""
    pipe = getattr(model, "pipe", None)
    dit = getattr(pipe, "dit", None)
    module = getattr(dit, "spatial_rope_lambda", None)
    if module is None or not hasattr(module, "heatmap_tensors"):
        return
    stem, _ = os.path.splitext(file_name)
    heatmap_dir = os.path.join(output_path, "lambda_heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)
    try:
        tensors = module.heatmap_tensors()
        payload = {
            "scope": getattr(module, "scope", "unknown"),
            "parametrization": getattr(module, "parametrization", "unknown"),
            "lambda_min": float(getattr(module, "lambda_min", 0.0)),
            "init_eps": float(getattr(module, "init_eps", 0.0)),
            "fixed_h": float(getattr(module, "fixed_h", 1.0)),
            "fixed_w": float(getattr(module, "fixed_w", 1.0)),
            "tensors": tensors,
        }
        torch.save(payload, os.path.join(heatmap_dir, f"{stem}_lambda_heatmaps.pt"))
        summary_path = os.path.join(heatmap_dir, f"{stem}_lambda_heatmaps.json")
        summary = {
            key: {
                "h_min": float(value[..., 0].min()),
                "h_max": float(value[..., 0].max()),
                "h_mean": float(value[..., 0].mean()),
                "w_min": float(value[..., 1].min()),
                "w_max": float(value[..., 1].max()),
                "w_mean": float(value[..., 1].mean()),
            }
            for key, value in tensors.items()
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"Warning: failed to import matplotlib for lambda heatmaps: {exc}", flush=True)
            return
        for timestep_key, value in tensors.items():
            for axis_id, axis_name in enumerate(("h", "w")):
                array = value[..., axis_id].numpy()
                fig_width = max(3.0, min(10.0, 0.35 * array.shape[1] + 2.0))
                fig_height = max(3.0, min(12.0, 0.22 * array.shape[0] + 2.0))
                fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)
                im = ax.imshow(array, aspect="auto", interpolation="nearest", vmin=0.0, vmax=max(1.0, float(array.max())))
                ax.set_title(f"lambda_{axis_name} {timestep_key}")
                ax.set_xlabel("head")
                ax.set_ylabel("layer")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                fig.savefig(os.path.join(heatmap_dir, f"{stem}_lambda_{timestep_key}_{axis_name}.png"))
                plt.close(fig)
    except Exception as exc:
        print(f"Warning: failed to save Wan spatial RoPE lambda heatmaps for {file_name}: {exc}", flush=True)


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
        self.offline_run_dir = os.environ.get("WANDB_OFFLINE_RUN_DIR") or None
        self.max_logged_step = -1
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
            offline_timespec, offline_run_id = parse_wandb_offline_run_dir(self.offline_run_dir)
            offline_start_time = wandb_timespec_to_timestamp(offline_timespec)
            raw_run_id = os.environ.get("WANDB_RUN_ID")
            if raw_run_id is not None and raw_run_id.strip() == "":
                os.environ.pop("WANDB_RUN_ID", None)
                raw_run_id = None
            run_id = raw_run_id or offline_run_id or None
            run_name = os.environ.get("WANDB_NAME") or None
            resume = os.environ.get("WANDB_RESUME", "allow") if run_id is not None else None
            init_fn = lambda: self.wandb.init(
                project=project_name,
                dir=log_dir,
                mode=requested_mode,
                settings=settings,
                id=run_id,
                name=run_name,
                resume=resume,
            )
            if requested_mode in ("offline", "dryrun") and self.offline_run_dir and offline_start_time is not None:
                self.run = self._call_with_hard_timeout(
                    lambda: self._init_existing_offline_run(init_fn, offline_start_time),
                    timeout,
                    f"wandb {requested_mode} init",
                )
            else:
                self.run = self._call_with_hard_timeout(
                    init_fn,
                    timeout,
                    f"wandb {requested_mode} init",
                )
            if self.run is not None and log_dir is not None:
                with open(os.path.join(log_dir, "wandb_run_id.txt"), "w") as f:
                    f.write(str(self.run.id))
                run_dir = getattr(getattr(self.run, "_settings", None), "sync_dir", None)
                if run_dir:
                    with open(os.path.join(log_dir, "wandb_run_dir.txt"), "w") as f:
                        f.write(str(run_dir))
                    if self.offline_run_dir:
                        self.max_logged_step = read_wandb_max_logged_step(run_dir, str(self.run.id))
                        if self.max_logged_step >= 0:
                            print(f"Wandb offline resume will skip existing logged steps <= {self.max_logged_step} in {run_dir}.", flush=True)
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


    def _init_existing_offline_run(self, init_fn, offline_start_time):
        """Force wandb offline resume to reuse the original offline-run directory.

        wandb 0.27 ignores resume=... in offline mode and appends a numeric
        suffix when the sync directory already exists. For checkpoint resume we
        want the local run directory to remain stable, so we temporarily fix the
        run start time and disable the suffix bump only for this init call.
        """
        from wandb.sdk import wandb_init

        original_time = wandb_init.time.time
        original_set_suffix = wandb_init._WandbInit.set_sync_dir_suffix

        def fixed_time():
            return offline_start_time

        def no_sync_dir_suffix(self, settings):
            return None

        try:
            wandb_init.time.time = fixed_time
            wandb_init._WandbInit.set_sync_dir_suffix = no_sync_dir_suffix
            return init_fn()
        finally:
            wandb_init.time.time = original_time
            wandb_init._WandbInit.set_sync_dir_suffix = original_set_suffix

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
        if int(step) <= self.max_logged_step:
            return
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
            _save_wan_spatial_rope_lambda_heatmaps(self.output_path, file_name, accelerator.unwrap_model(model))
