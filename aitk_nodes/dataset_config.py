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
                "num_repeats": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Number of times to repeat the dataset's file list per epoch",
                }),
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
                "is_reg": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Treat as a regularization / prior-preservation dataset",
                }),
                "trigger_word": ("STRING", {
                    "default": "",
                    "tooltip": "Trigger word carried by this dataset. Required by diff_output_preservation, which swaps it for the class when building the preservation embeddings",
                }),
            },
        }

    def build(
        self,
        folder_path: str,
        caption_ext: str,
        resolution: str,
        num_repeats: int = 1,
        caption_dropout_rate: float = 0.05,
        shuffle_tokens: bool = False,
        cache_latents_to_disk: bool = True,
        control_path: str = "",
        is_reg: bool = False,
        trigger_word: str = "",
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
            "num_repeats": num_repeats,
            "is_reg": is_reg,
        }

        if control_path:
            config["control_path"] = control_path

        if trigger_word:
            config["trigger_word"] = trigger_word

        return (config,)
