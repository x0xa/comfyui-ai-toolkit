"""Checkpoint save configuration node."""


class AIToolkitSaveConfig:
    CATEGORY = "AI Toolkit/Config"
    RETURN_TYPES = ("AITK_SAVE_CONFIG",)
    RETURN_NAMES = ("save_config",)
    FUNCTION = "build"

    SAVE_DTYPES = ["float16", "bfloat16", "float32"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dtype": (cls.SAVE_DTYPES, {
                    "default": "float16",
                    "tooltip": "Precision for saved checkpoint",
                }),
                "save_every": ("INT", {
                    "default": 250,
                    "min": 1,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Save checkpoint every N steps",
                }),
                "max_step_saves_to_keep": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Maximum number of intermediate checkpoints to keep on disk",
                }),
            },
            "optional": {
                "push_to_hub": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Push trained model to HuggingFace Hub",
                }),
                "hf_repo_id": ("STRING", {
                    "default": "",
                    "tooltip": "HuggingFace repo ID (e.g. username/model-name)",
                }),
                "hf_private": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Make HuggingFace repo private",
                }),
                "save_format": ("STRING", {
                    "default": "",
                    "tooltip": "Save format override (e.g. 'diffusers'). Empty = default safetensors",
                }),
            },
        }

    def build(
        self,
        dtype: str,
        save_every: int,
        max_step_saves_to_keep: int,
        push_to_hub: bool = False,
        hf_repo_id: str = "",
        hf_private: bool = True,
        save_format: str = "",
    ):
        config = {
            "dtype": dtype,
            "save_every": save_every,
            "max_step_saves_to_keep": max_step_saves_to_keep,
            "push_to_hub": push_to_hub,
        }

        if push_to_hub and hf_repo_id:
            config["hf_repo_id"] = hf_repo_id
            config["hf_private"] = hf_private

        if save_format:
            config["save_format"] = save_format

        return (config,)
