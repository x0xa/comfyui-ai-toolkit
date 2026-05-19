"""Main training execution node. Assembles config, runs ai-toolkit subprocess.

When a FantasioTrainingContext is connected it also uploads each saved checkpoint
with its epoch samples to S3 and emits the comfy-api training contract events.
"""

import os
import sys
import time
import glob
import yaml
import importlib.util

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AITK_DIR = os.path.join(_PKG_ROOT, "ai-toolkit")

SAMPLE_POLL_INTERVAL_SECONDS = 5
LOOP_SLEEP_SECONDS = 0.5
HEARTBEAT_INTERVAL_SECONDS = 10


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
            has_comfy = True
        except ImportError:
            has_comfy = False

        build_config = _load_pkg_module("utils.config_builder").build_config
        AIToolkitProcess = _load_pkg_module("utils.process_manager").AIToolkitProcess
        _sw = _load_pkg_module("utils.sample_watcher")
        SampleWatcher, load_images_as_tensor = _sw.SampleWatcher, _sw.load_images_as_tensor
        CheckpointWatcher = _load_pkg_module("utils.checkpoint_watcher").CheckpointWatcher
        epoch_events = _load_pkg_module("utils.epoch_events")
        upload_epoch_artifacts = _load_pkg_module("utils.checkpoint_upload").upload_epoch_artifacts

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

        client_id = fantasio_context["client_id"] if fantasio_context else None
        total_epochs = fantasio_context["total_epochs"] if fantasio_context else 0
        fantasio_lib = None
        if fantasio_context:
            fantasio_lib = _load_pkg_module("utils.fantasio_lib").load_fantasio_lib()

        # Setup progress bar
        total_steps = train_config.get("steps", 2000)
        pbar = None
        if has_comfy:
            pbar = comfy.utils.ProgressBar(total_steps)

        # Launch training subprocess
        process = AIToolkitProcess(config_path, AITK_DIR)
        process.start()

        last_step = 0
        last_sample_check = 0
        last_progress_emit = 0
        epoch_counter = 0
        pending_samples = []

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
                        epoch_events, client_id, progress, total_steps, epoch_counter, total_epochs
                    )
                    last_progress_emit = now

                last_step = progress.step

                if now - last_sample_check > SAMPLE_POLL_INTERVAL_SECONDS:
                    pending_samples.extend(sample_watcher.check_new_samples())
                    last_sample_check = now

                if fantasio_context:
                    for checkpoint_path in checkpoint_watcher.check_new_checkpoints():
                        epoch_counter += 1
                        pending_samples.extend(sample_watcher.check_new_samples())
                        epoch_events.emit_message(client_id, f"Uploading epoch {epoch_counter} checkpoint")
                        self._handle_epoch(
                            epoch_events, upload_epoch_artifacts, fantasio_lib, fantasio_context,
                            client_id, epoch_counter, total_epochs, checkpoint_path,
                            pending_samples, process.progress,
                        )
                        pending_samples = []
                        last_progress_emit = time.time()

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
            if len(full) > 5000:
                error_msg = full[:2500] + "\n\n... [truncated, full log: " + log_path + "] ...\n\n" + full[-2500:]
            else:
                error_msg = full
            failure = (
                f"Training failed with exit code {exit_code}.\n"
                f"Full log saved to: {log_path}\n"
                f"Output:\n{error_msg}"
            )
            if fantasio_context:
                epoch_events.emit_training_failed(client_id, f"Training failed with exit code {exit_code}")
            raise RuntimeError(failure)

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

    def _emit_progress(self, epoch_events, client_id, progress, total_steps, completed_epochs, total_epochs):
        steps_total = progress.total_steps or total_steps
        percentage = round((progress.step / steps_total) * 100, 2) if steps_total else 0.0
        progress_data = {
            "completedSteps": progress.step,
            "totalSteps": steps_total,
            "completedEpochs": completed_epochs,
            "totalEpochs": total_epochs or 0,
            "progressPercentage": percentage,
        }
        # Non-training phases (load/save/sample) carry a phase message so the UI
        # shows the activity; the training phase keeps the numeric progress bar.
        if progress.phase != "Training":
            progress_data["message"] = progress.phase
        epoch_events.emit_progress(client_id, progress_data)

    def _handle_epoch(self, epoch_events, upload_epoch_artifacts, fantasio_lib, context,
                      client_id, epoch, total_epochs, checkpoint_path, sample_paths, progress):
        try:
            lora_url, sample_urls = upload_epoch_artifacts(
                fantasio_lib, context, epoch, checkpoint_path, sample_paths
            )
            epoch_events.emit_epoch_uploaded(
                client_id, context["task_id"], context["user_id"], epoch,
                progress.loss, progress.step, lora_url, sample_urls,
            )
            if total_epochs and epoch >= total_epochs:
                epoch_events.emit_task_completed(
                    client_id, context["task_id"], context["user_id"], epoch
                )
        except Exception as e:
            epoch_events.emit_message(client_id, f"Epoch {epoch} upload failed: {e}")

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
