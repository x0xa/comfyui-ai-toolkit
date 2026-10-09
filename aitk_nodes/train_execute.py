"""Main training execution node. Assembles config, runs ai-toolkit subprocess.

When a FantasioTrainingContext is connected it also uploads each saved checkpoint
with its epoch samples to S3 and emits the comfy-api training contract events.
"""

import os
import re
import sys
import time
import glob
import yaml
import importlib.util
from urllib.parse import urlparse

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AITK_DIR = os.path.join(_PKG_ROOT, "ai-toolkit")

SAMPLE_POLL_INTERVAL_SECONDS = 5
LOOP_SLEEP_SECONDS = 0.5
HEARTBEAT_INTERVAL_SECONDS = 10
OUTPUT_TAIL_CHARS = 6000
EPOCH_UPLOAD_DEADLINE_SECONDS = 900
EPOCH_UPLOAD_RETRY_DELAY_SECONDS = 10
RESUME_DOWNLOAD_TIMEOUT_SECONDS = 300
RESUMED_OPTIMIZER_FILENAME = "optimizer.pt"
EGRESS_FAILURE_EXCEPTION_NAME = "InstanceEgressUnavailable"

# Both the sample images and the checkpoints carry the step they belong to in their
# file name, which is the only reliable way to pair them: the trainer writes the
# checkpoint first and renders that step's samples afterwards, so "whatever samples
# arrived since the last upload" always belongs to the previous checkpoint.
_SAMPLE_STEP_RE = re.compile(r"__(\d+)_\d+\.[A-Za-z0-9]+$")
_CHECKPOINT_STEP_RE = re.compile(r"_(\d+)\.safetensors$")


def _sample_step(path):
    match = _SAMPLE_STEP_RE.search(os.path.basename(path))
    return int(match.group(1)) if match else None


# The trainer's final save, made after the loop, carries no step in its name, while
# that save's samples are named with the last step, which equals the configured steps.
def _checkpoint_step(path, final_step):
    match = _CHECKPOINT_STEP_RE.search(os.path.basename(path))
    return int(match.group(1)) if match else final_step


# The trainer copies its optimizer state next to every step-named checkpoint, so the
# state of an epoch is found by the checkpoint's step. The final save has none.
def _epoch_state_path(checkpoint_path, step):
    state_path = os.path.join(os.path.dirname(checkpoint_path), f"optimizer_{str(step).zfill(9)}.pt")
    return state_path if os.path.isfile(state_path) else None


class EpochUploadDeadlineExceeded(RuntimeError):
    pass


# Raised in place of the original error when the failure belongs to the machine, so
# the exception type in ComfyUI's execution_error and history carries the same verdict
# as the training.failed event.
class TrainingMachineFault(RuntimeError):
    pass


GPU_FAULT_SIGNATURES = (
    "unspecified launch failure",
    "cudaerrorlaunchfailure",
    "an illegal memory access",
    "uncorrectable ecc error",
    "ecc error",
    "has fallen off the bus",
    "device-side assert triggered",
    "no cuda-capable device",
    "cuda-capable device(s) is/are busy",
)

GPU_LIBRARY_SIGNATURES = (
    "cublas",
    "cudnn",
    "cusolver",
)


def _classify_training_failure(error, output):
    """Return (message, machine_fault) for a failure raised while training.

    machine_fault marks a failure of the rented machine itself (card, driver or
    network), which comfy-api recovers by moving the task to another machine and
    continuing from the last uploaded epoch instead of failing the task.
    """
    text = str(error).strip()
    haystack = f"{text}\n{output or ''}".lower()

    if isinstance(error, EpochUploadDeadlineExceeded):
        return f"Machine could not upload an epoch checkpoint: {text}", True

    if type(error).__name__ == EGRESS_FAILURE_EXCEPTION_NAME:
        return f"Machine lost its network egress: {text}", True

    if any(signature in haystack for signature in GPU_FAULT_SIGNATURES):
        return (
            "GPU fault during training - the card stopped serving kernels mid-run. "
            f"This is a hardware/driver failure of the rented machine, not a training setting: {text}",
            True,
        )

    if "out of memory" in haystack:
        return (
            "GPU out of memory during training "
            f"(reduce batch size or resolution): {text}",
            False,
        )

    if any(signature in haystack for signature in GPU_LIBRARY_SIGNATURES):
        return f"GPU math library error during training (cuBLAS/cuDNN): {text}", False

    if "cuda error" in haystack:
        return f"CUDA error during training: {text}", False

    if "no images found" in haystack or "no such file" in haystack or "filenotfounderror" in haystack:
        return f"Dataset / file error during training: {text}", False

    return text, False


def _load_pkg_module(rel_path):
    """Load a module by file path relative to the package root.

    Avoids name collisions with other packages that have common names
    like 'utils' (e.g. comfy.utils).
    """
    mod_name = f"comfyui_aitk.{rel_path}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    parts = rel_path.replace(".", os.sep)
    fpath = os.path.join(_PKG_ROOT, parts + ".py")
    spec = importlib.util.spec_from_file_location(mod_name, fpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


class AIToolkitTrainExecute:
    CATEGORY = "AI Toolkit"
    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "FLOAT")
    RETURN_NAMES = ("lora_path", "sample_images", "training_log", "final_loss")
    FUNCTION = "execute"
    OUTPUT_NODE = True

    DEVICES = ["cuda:0", "cuda:1", "cpu"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_config": ("AITK_MODEL_CONFIG",),
                "network_config": ("AITK_NETWORK_CONFIG",),
                "train_config": ("AITK_TRAIN_CONFIG",),
                "dataset_config": ("AITK_DATASET_CONFIG",),
                "save_config": ("AITK_SAVE_CONFIG",),
                "job_name": ("STRING", {
                    "default": "my_lora_v1",
                    "tooltip": "Name for this training run (used as folder/file name)",
                }),
                "training_folder": ("STRING", {
                    "default": "output",
                    "tooltip": "Root folder to save training output (relative to ai-toolkit or absolute)",
                }),
                "device": (cls.DEVICES, {
                    "default": "cuda:0",
                }),
            },
            "optional": {
                "sample_config": ("AITK_SAMPLE_CONFIG",),
                "embedding_config": ("AITK_EMBEDDING_CONFIG",),
                "caption_config": ("AITK_CAPTION_CONFIG",),
                "dataset_list": ("AITK_DATASET_LIST",),
                "fantasio_context": ("AITK_FANTASIO_CTX",),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-execute when queued
        return float("nan")

    def execute(
        self,
        model_config: dict,
        network_config: dict,
        train_config: dict,
        dataset_config: dict,
        save_config: dict,
        job_name: str,
        training_folder: str,
        device: str,
        sample_config: dict = None,
        embedding_config: dict = None,
        caption_config: dict = None,
        dataset_list: list = None,
        fantasio_context: dict = None,
    ):
        import torch

        # Lazy imports for ComfyUI compatibility
        try:
            import comfy.model_management
            import comfy.utils
            import folder_paths
            has_comfy = True
        except ImportError:
            has_comfy = False

        build_config = _load_pkg_module("utils.config_builder").build_config
        AIToolkitProcess = _load_pkg_module("utils.process_manager").AIToolkitProcess
        _sw = _load_pkg_module("utils.sample_watcher")
        SampleWatcher, load_images_as_tensor = _sw.SampleWatcher, _sw.load_images_as_tensor
        CheckpointWatcher = _load_pkg_module("utils.checkpoint_watcher").CheckpointWatcher
        epoch_events = _load_pkg_module("utils.epoch_events")
        EpochLedger = _load_pkg_module("utils.epoch_ledger").EpochLedger
        upload_epoch_artifacts = _load_pkg_module("utils.checkpoint_upload").upload_epoch_artifacts
        compute_adapter_metrics = _load_pkg_module("utils.adapter_metrics").compute_adapter_metrics

        client_id = fantasio_context["client_id"] if fantasio_context else None
        total_epochs = fantasio_context["total_epochs"] if fantasio_context else 0

        # Emit progress before any slow setup so the comfy-api stall monitor
        # sees activity from the moment the node starts executing.
        if fantasio_context:
            epoch_events.emit_message(client_id, "Preparing training")

        process = None

        try:
            # Free VRAM before training
            if has_comfy:
                comfy.model_management.soft_empty_cache()
                comfy.model_management.unload_all_models()

            # Determine datasets
            if dataset_list is not None:
                datasets = dataset_list
            else:
                datasets = [dataset_config]

            # Run auto-captioning if configured
            if caption_config and caption_config.get("enabled", False):
                from .caption_config import AIToolkitCaptionConfig
                for ds in datasets:
                    folder = ds.get("folder_path", "")
                    if folder:
                        if fantasio_context:
                            epoch_events.emit_message(client_id, "Captioning dataset")
                        success, msg = AIToolkitCaptionConfig.run_captioning(
                            caption_config, folder, AITK_DIR
                        )
                        if not success:
                            raise RuntimeError(f"Auto-captioning failed: {msg}")

            # Build config
            full_config = build_config(
                job_name=job_name,
                training_folder=training_folder,
                device=device,
                model_config=model_config,
                network_config=network_config,
                train_config=train_config,
                dataset_configs=datasets,
                save_config=save_config,
                sample_config=sample_config,
                embedding_config=embedding_config,
            )

            # Write config to YAML
            if os.path.isabs(training_folder):
                config_dir = training_folder
            else:
                config_dir = os.path.join(AITK_DIR, training_folder)

            os.makedirs(config_dir, exist_ok=True)
            config_path = os.path.join(config_dir, f"{job_name}_config.yaml")
            with open(config_path, "w") as f:
                yaml.dump(full_config, f, default_flow_style=False, allow_unicode=True)

            output_base = config_dir
            sample_watcher = SampleWatcher(output_base, job_name)
            checkpoint_watcher = CheckpointWatcher(output_base, job_name)

            save_every = save_config["save_every"]
            completed_epochs = 0
            fantasio_lib = None
            ledger = None
            if fantasio_context:
                fantasio_lib = _load_pkg_module("utils.fantasio_lib").load_fantasio_lib()
                ledger = EpochLedger(folder_paths.get_output_directory(), fantasio_context["task_id"])

                if fantasio_context["resume_checkpoint_url"]:
                    completed_epochs = self._restore_training_state(
                        fantasio_lib, fantasio_context, os.path.join(output_base, job_name), save_every, client_id
                    )
                    checkpoint_watcher.check_new_checkpoints()

            # Setup progress bar
            total_steps = train_config.get("steps", 2000)
            pbar = None
            if has_comfy:
                pbar = comfy.utils.ProgressBar(total_steps)

            # Launch training subprocess
            process = AIToolkitProcess(config_path, AITK_DIR, train_steps=total_steps)
            process.start()

            last_step = 0
            last_sample_check = 0
            last_progress_emit = 0
            samples_by_step = {}
            awaiting_upload = []
            expected_samples = len(sample_config.get("prompts", [])) if sample_config else 0

            def collect_samples():
                for sample_path in sample_watcher.check_new_samples():
                    samples_by_step.setdefault(_sample_step(sample_path), []).append(sample_path)

            def queue_new_checkpoints():
                for checkpoint_path in checkpoint_watcher.check_new_checkpoints():
                    step = _checkpoint_step(checkpoint_path, total_steps)
                    awaiting_upload.append({
                        "epoch": step // save_every,
                        "path": checkpoint_path,
                        "step": step,
                    })

            def flush_ready(force=False):
                nonlocal last_progress_emit, completed_epochs

                while awaiting_upload:
                    entry = awaiting_upload[0]
                    step = entry["step"]
                    own_samples = sorted(samples_by_step.get(step, []))
                    complete = expected_samples > 0 and len(own_samples) >= expected_samples
                    superseded = step is not None and any(
                        seen is not None and seen > step for seen in samples_by_step
                    )

                    if not (force or complete or superseded):
                        return

                    awaiting_upload.pop(0)
                    samples_by_step.pop(step, None)
                    epoch_events.emit_message(client_id, f"Uploading epoch {entry['epoch']} checkpoint")
                    self._handle_epoch(
                        epoch_events, upload_epoch_artifacts, compute_adapter_metrics,
                        fantasio_lib, fantasio_context, ledger,
                        client_id, entry, total_epochs, own_samples, process.progress,
                    )
                    completed_epochs = entry["epoch"]
                    last_progress_emit = time.time()
                    process.progress.reset_loss_window()

            try:
                while process.is_running():
                    process.get_new_lines()

                    progress = process.progress
                    now = time.time()
                    step_changed = progress.step > last_step

                    if step_changed and pbar:
                        pbar.update_absolute(progress.step, total_steps)

                    # Heartbeat: emit progress on every step and at least every
                    # HEARTBEAT_INTERVAL_SECONDS so silent phases (model load,
                    # validation, saving) never look like a stalled instance.
                    if fantasio_context and (step_changed or now - last_progress_emit >= HEARTBEAT_INTERVAL_SECONDS):
                        self._emit_progress(
                            epoch_events, client_id, progress, total_steps, completed_epochs, total_epochs
                        )
                        last_progress_emit = now

                    last_step = progress.step

                    if now - last_sample_check > SAMPLE_POLL_INTERVAL_SECONDS:
                        collect_samples()
                        last_sample_check = now

                    if fantasio_context:
                        queue_new_checkpoints()
                        collect_samples()
                        flush_ready()

                    time.sleep(LOOP_SLEEP_SECONDS)

            except KeyboardInterrupt:
                process.terminate()
                raise

            # Wait for process to finish
            exit_code = process.wait(timeout=30)

            if exit_code != 0:
                full = process.full_output or ""
                log_path = os.path.join(config_dir, f"{job_name}_error.log")
                try:
                    with open(log_path, "w") as f:
                        f.write(full)
                except Exception:
                    log_path = "(failed to write log)"
                tail = full[-OUTPUT_TAIL_CHARS:] if full else "(no output captured)"
                raise RuntimeError(
                    f"Training failed with exit code {exit_code}. Full log: {log_path}\n"
                    f"--- output tail ---\n{tail}"
                )

            # The last checkpoint and its samples land after the loop has already
            # left, so drain both once the trainer is done: without this the final
            # epoch never reaches S3.
            if fantasio_context:
                queue_new_checkpoints()
                collect_samples()
                flush_ready(force=True)

            # Find the final LoRA checkpoint
            lora_path = self._find_latest_checkpoint(output_base, job_name)

            # Load sample images for output
            all_samples = sample_watcher.get_latest_samples(count=20)
            sample_tensor = load_images_as_tensor(all_samples)
            if sample_tensor is None:
                sample_tensor = torch.zeros(1, 64, 64, 3)

            training_log = process.full_output
            final_loss = process.progress.loss

            return (lora_path, sample_tensor, training_log, final_loss)

        except BaseException as e:
            # Any failure — config errors, captioning, a crashed subprocess or an
            # interrupt — is reported to comfy-api so the task fails fast instead
            # of waiting for the stall monitor.
            if fantasio_context:
                output = process.full_output if process is not None else ""
                message, machine_fault = _classify_training_failure(e, output)
                epoch_events.emit_training_failed(client_id, message, machine_fault)
                if machine_fault:
                    raise TrainingMachineFault(message) from e
            raise

    def _emit_progress(self, epoch_events, client_id, progress, total_steps, completed_epochs, total_epochs):
        steps_total = progress.total_steps or total_steps
        percentage = round((progress.step / steps_total) * 100, 2) if steps_total else 0.0
        progress_data = {
            "completedSteps": progress.step,
            "totalSteps": steps_total,
            "completedEpochs": completed_epochs,
            "totalEpochs": total_epochs or 0,
            "progressPercentage": percentage,
            "phase": progress.phase,
            "avgLoss": progress.avg_loss,
        }
        if progress.phase != "Training":
            progress_data["message"] = progress.phase
        epoch_events.emit_progress(client_id, progress_data)

    def _handle_epoch(self, epoch_events, upload_epoch_artifacts, compute_adapter_metrics,
                      fantasio_lib, context, ledger,
                      client_id, entry, total_epochs, sample_paths, progress):
        epoch = entry["epoch"]
        step = entry["step"]
        checkpoint_path = entry["path"]

        metrics = None
        try:
            metrics = compute_adapter_metrics(checkpoint_path)
        except Exception as e:
            epoch_events.emit_message(
                client_id,
                f"Epoch {epoch} adapter metrics failed ({type(e).__name__}): {e}",
            )

        state_path = _epoch_state_path(checkpoint_path, step)

        with fantasio_lib.GpuActivityNotifier(f"Uploading epoch {epoch} checkpoint", client_id):
            lora_url, sample_urls, state_url = self._upload_epoch_until_deadline(
                epoch_events, upload_epoch_artifacts, fantasio_lib, context,
                client_id, epoch, checkpoint_path, sample_paths, state_path,
            )

        payload = epoch_events.build_epoch_payload(
            context["task_id"], epoch, progress.avg_loss, step, lora_url, sample_urls, state_url, metrics,
        )
        ledger.record(payload)
        epoch_events.emit_epoch_uploaded(client_id, payload)

        if state_path is not None:
            os.remove(state_path)

        if total_epochs and epoch >= total_epochs:
            epoch_events.emit_task_completed(client_id, context["task_id"], epoch)

    # A failed upload is retried while training goes on; only a machine that keeps
    # failing past the deadline is reported, so comfy-api moves the task elsewhere.
    def _upload_epoch_until_deadline(self, epoch_events, upload_epoch_artifacts, fantasio_lib, context,
                                     client_id, epoch, checkpoint_path, sample_paths, state_path):
        deadline = time.monotonic() + EPOCH_UPLOAD_DEADLINE_SECONDS

        while True:
            try:
                return upload_epoch_artifacts(
                    fantasio_lib, context, epoch, checkpoint_path, sample_paths, state_path
                )
            except Exception as e:
                if time.monotonic() >= deadline:
                    raise EpochUploadDeadlineExceeded(
                        f"epoch {epoch} not uploaded within {EPOCH_UPLOAD_DEADLINE_SECONDS}s "
                        f"({type(e).__name__}): {e}"
                    ) from e

                epoch_events.emit_message(
                    client_id,
                    f"Epoch {epoch} checkpoint upload failed, retrying ({type(e).__name__}): {e}",
                )
                time.sleep(EPOCH_UPLOAD_RETRY_DELAY_SECONDS)

    def _restore_training_state(self, fantasio_lib, context, job_dir, save_every, client_id):
        checkpoint_url = context["resume_checkpoint_url"]
        checkpoint_path = os.path.join(job_dir, os.path.basename(urlparse(checkpoint_url).path))
        os.makedirs(job_dir, exist_ok=True)

        with fantasio_lib.GpuActivityNotifier("Restoring training state", client_id):
            fantasio_lib.download_to_file(
                checkpoint_url, checkpoint_path, timeout_seconds=RESUME_DOWNLOAD_TIMEOUT_SECONDS
            )

            if context["resume_state_url"]:
                fantasio_lib.download_to_file(
                    context["resume_state_url"],
                    os.path.join(job_dir, RESUMED_OPTIMIZER_FILENAME),
                    timeout_seconds=RESUME_DOWNLOAD_TIMEOUT_SECONDS,
                )

        return _checkpoint_step(checkpoint_path, None) // save_every

    def _find_latest_checkpoint(self, output_base: str, job_name: str) -> str:
        """Find the most recent checkpoint file in the output directory."""
        job_dir = os.path.join(output_base, job_name)

        patterns = [
            os.path.join(job_dir, "*.safetensors"),
            os.path.join(job_dir, "**", "*.safetensors"),
        ]

        all_checkpoints = []
        for pattern in patterns:
            all_checkpoints.extend(glob.glob(pattern, recursive=True))

        if not all_checkpoints:
            diffusers_dirs = glob.glob(os.path.join(job_dir, "*", "model_index.json"))
            if diffusers_dirs:
                return os.path.dirname(diffusers_dirs[-1])
            return ""

        all_checkpoints.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return all_checkpoints[0]
