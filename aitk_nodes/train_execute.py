"""Main training execution node. Assembles config, runs ai-toolkit subprocess."""

import os
import sys
import time
import glob
import yaml

AITK_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ai-toolkit")


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
    ):
        import torch

        # Lazy imports for ComfyUI compatibility
        try:
            import comfy.model_management
            import comfy.utils
            from server import PromptServer
            has_comfy = True
        except ImportError:
            has_comfy = False

        from utils.config_builder import build_config
        from utils.process_manager import AIToolkitProcess
        from utils.sample_watcher import SampleWatcher, load_images_as_tensor

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
            from aitk_nodes.caption_config import AIToolkitCaptionConfig
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

        # Determine output/sample directories
        output_base = config_dir
        sample_watcher = SampleWatcher(output_base, job_name)

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
        latest_samples = []

        try:
            while process.is_running():
                # Read new output lines
                new_lines = process.get_new_lines()

                for line in new_lines:
                    # Send log lines to frontend
                    if has_comfy and line.strip():
                        try:
                            PromptServer.instance.send_sync(
                                "aitoolkit.log",
                                {"message": line},
                            )
                        except Exception:
                            pass

                # Update progress
                progress = process.progress
                if progress.step > last_step:
                    step_delta = progress.step - last_step
                    if pbar:
                        pbar.update_absolute(progress.step, total_steps)
                    if has_comfy:
                        try:
                            PromptServer.instance.send_sync(
                                "aitoolkit.progress",
                                {
                                    "step": progress.step,
                                    "total_steps": progress.total_steps or total_steps,
                                    "loss": progress.loss,
                                },
                            )
                        except Exception:
                            pass
                    last_step = progress.step

                # Check for new sample images periodically
                now = time.time()
                if now - last_sample_check > 5:
                    new_samples = sample_watcher.check_new_samples()
                    if new_samples:
                        latest_samples = new_samples
                        if has_comfy:
                            try:
                                PromptServer.instance.send_sync(
                                    "aitoolkit.samples",
                                    {
                                        "step": last_step,
                                        "count": len(new_samples),
                                        "paths": new_samples,
                                    },
                                )
                            except Exception:
                                pass
                    last_sample_check = now

                time.sleep(0.5)

        except KeyboardInterrupt:
            process.terminate()
            raise

        # Wait for process to finish
        exit_code = process.wait(timeout=30)

        if exit_code != 0:
            log_tail = process.full_output[-2000:] if process.full_output else ""
            raise RuntimeError(
                f"Training failed with exit code {exit_code}.\n"
                f"Last output:\n{log_tail}"
            )

        # Find the final LoRA checkpoint
        lora_path = self._find_latest_checkpoint(output_base, job_name)

        # Load sample images for output
        all_samples = sample_watcher.get_latest_samples(count=20)
        sample_tensor = load_images_as_tensor(all_samples)
        if sample_tensor is None:
            # Return a small placeholder image
            sample_tensor = torch.zeros(1, 64, 64, 3)

        training_log = process.full_output
        final_loss = process.progress.loss

        return (lora_path, sample_tensor, training_log, final_loss)

    def _find_latest_checkpoint(self, output_base: str, job_name: str) -> str:
        """Find the most recent checkpoint file in the output directory."""
        job_dir = os.path.join(output_base, job_name)

        # Look for .safetensors files
        patterns = [
            os.path.join(job_dir, "*.safetensors"),
            os.path.join(job_dir, "**", "*.safetensors"),
        ]

        all_checkpoints = []
        for pattern in patterns:
            all_checkpoints.extend(glob.glob(pattern, recursive=True))

        if not all_checkpoints:
            # Try looking for diffusers format
            diffusers_dirs = glob.glob(os.path.join(job_dir, "*", "model_index.json"))
            if diffusers_dirs:
                return os.path.dirname(diffusers_dirs[-1])
            return ""

        # Return the most recently modified one
        all_checkpoints.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return all_checkpoints[0]
