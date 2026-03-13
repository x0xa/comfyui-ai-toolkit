"""Dataset configuration node."""


class AIToolkitDatasetConfig:
    CATEGORY = "AI Toolkit/Config"
    RETURN_TYPES = ("AITK_DATASET_CONFIG",)
    RETURN_NAMES = ("dataset_config",)
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute path to folder with training images (jpg/jpeg/png)",
                }),
                "caption_ext": ("STRING", {
                    "default": "txt",
                    "tooltip": "Extension for caption files (e.g. txt)",
                }),
                "resolution": ("STRING", {
                    "default": "512, 768, 1024",
                    "tooltip": "Comma-separated resolutions for multi-resolution training",
                }),
            },
            "optional": {
                "caption_dropout_rate": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Probability to drop caption during training (regularization)",
                }),
                "shuffle_tokens": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Shuffle caption tokens (comma-separated) for augmentation",
                }),
                "cache_latents_to_disk": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Cache encoded latents to disk (recommended for speed)",
                }),
                "control_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to control/source images folder (for Kontext-style editing)",
                }),
            },
        }

    def build(
        self,
        folder_path: str,
        caption_ext: str,
        resolution: str,
        caption_dropout_rate: float = 0.05,
        shuffle_tokens: bool = False,
        cache_latents_to_disk: bool = True,
        control_path: str = "",
    ):
        # Parse resolution string into list of ints
        try:
            res_list = [int(r.strip()) for r in resolution.split(",") if r.strip()]
        except ValueError:
            res_list = [1024]

        config = {
            "folder_path": folder_path,
            "caption_ext": caption_ext,
            "caption_dropout_rate": caption_dropout_rate,
            "shuffle_tokens": shuffle_tokens,
            "cache_latents_to_disk": cache_latents_to_disk,
            "resolution": res_list,
        }

        if control_path:
            config["control_path"] = control_path

        return (config,)
