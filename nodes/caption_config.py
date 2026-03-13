"""Auto-captioning configuration node.

Runs captioning as a pre-step before training using ai-toolkit's
built-in captioning capabilities.
"""

import os
import sys
import subprocess


class AIToolkitCaptionConfig:
    CATEGORY = "AI Toolkit/Config"
    RETURN_TYPES = ("AITK_CAPTION_CONFIG",)
    RETURN_NAMES = ("caption_config",)
    FUNCTION = "build"

    CAPTION_MODELS = [
        "florence2",
        "joy-caption",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable auto-captioning before training",
                }),
                "model": (cls.CAPTION_MODELS, {
                    "default": "florence2",
                    "tooltip": "Captioning model to use",
                }),
            },
            "optional": {
                "prompt": ("STRING", {
                    "default": "Describe this image in detail",
                    "multiline": True,
                    "tooltip": "Prompt/instruction for the captioning model",
                }),
                "caption_ext": ("STRING", {
                    "default": "txt",
                    "tooltip": "File extension for generated captions",
                }),
                "overwrite": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Overwrite existing caption files",
                }),
                "prefix": ("STRING", {
                    "default": "",
                    "tooltip": "Prefix added to all generated captions",
                }),
                "suffix": ("STRING", {
                    "default": "",
                    "tooltip": "Suffix added to all generated captions",
                }),
            },
        }

    def build(
        self,
        enabled: bool,
        model: str,
        prompt: str = "Describe this image in detail",
        caption_ext: str = "txt",
        overwrite: bool = False,
        prefix: str = "",
        suffix: str = "",
    ):
        config = {
            "enabled": enabled,
            "model": model,
            "prompt": prompt,
            "caption_ext": caption_ext,
            "overwrite": overwrite,
        }
        if prefix:
            config["prefix"] = prefix
        if suffix:
            config["suffix"] = suffix

        return (config,)

    @staticmethod
    def run_captioning(
        caption_config: dict,
        dataset_folder: str,
        ai_toolkit_dir: str,
    ) -> tuple[bool, str]:
        """Run auto-captioning on the dataset folder.

        Returns (success, message).
        """
        if not caption_config.get("enabled", False):
            return True, "Captioning disabled, skipping"

        if not os.path.isdir(dataset_folder):
            return False, f"Dataset folder not found: {dataset_folder}"

        # Build captioning script command
        # ai-toolkit uses a captioning script that can be invoked
        caption_ext = caption_config.get("caption_ext", "txt")
        overwrite = caption_config.get("overwrite", False)
        prefix = caption_config.get("prefix", "")
        suffix = caption_config.get("suffix", "")
        model = caption_config.get("model", "florence2")
        prompt = caption_config.get("prompt", "Describe this image in detail")

        # Generate captions using a simple Python script that uses ai-toolkit internals
        script = f"""
import sys
import os
sys.path.insert(0, {ai_toolkit_dir!r})
os.chdir({ai_toolkit_dir!r})

from toolkit.captioning.caption_job import CaptionJob

config = {{
    "job": "caption",
    "config": {{
        "process": [{{
            "type": "caption",
            "model": {model!r},
            "folder_path": {dataset_folder!r},
            "caption_ext": {caption_ext!r},
            "overwrite": {overwrite!r},
            "prompt": {prompt!r},
            "prefix": {prefix!r},
            "suffix": {suffix!r},
        }}]
    }}
}}

job = CaptionJob(config)
job.run()
job.cleanup()
print("CAPTIONING_COMPLETE")
"""

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        try:
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=3600,
                env=env,
            )
            if result.returncode == 0 and "CAPTIONING_COMPLETE" in result.stdout:
                return True, "Captioning completed successfully"
            else:
                error = result.stderr or result.stdout
                return False, f"Captioning failed: {error[-500:]}"
        except subprocess.TimeoutExpired:
            return False, "Captioning timed out after 1 hour"
        except Exception as e:
            return False, f"Captioning error: {e}"
