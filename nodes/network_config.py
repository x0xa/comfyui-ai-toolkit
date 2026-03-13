"""Network (LoRA) configuration node."""

import json


class AIToolkitNetworkConfig:
    CATEGORY = "AI Toolkit/Config"
    RETURN_TYPES = ("AITK_NETWORK_CONFIG",)
    RETURN_NAMES = ("network_config",)
    FUNCTION = "build"

    NETWORK_TYPES = ["lora", "locon", "lokr", "loha", "ia3"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "type": (cls.NETWORK_TYPES, {
                    "default": "lora",
                    "tooltip": "Network type: lora, locon, lokr, loha, ia3",
                }),
                "linear": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 512,
                    "step": 1,
                    "tooltip": "Rank for linear layers. Higher = more capacity but larger file",
                }),
                "linear_alpha": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 512,
                    "step": 1,
                    "tooltip": "Alpha for linear layers. Typically set equal to rank",
                }),
            },
            "optional": {
                "conv": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 512,
                    "step": 1,
                    "tooltip": "Rank for conv layers (LoCoN/LoHA). 0 = disabled",
                }),
                "conv_alpha": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 512,
                    "step": 1,
                    "tooltip": "Alpha for conv layers",
                }),
                "dropout": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Dropout rate for regularization (0 = disabled)",
                }),
                "network_kwargs_json": ("STRING", {
                    "default": "",
                    "tooltip": "Extra network arguments as JSON dict (advanced)",
                    "multiline": True,
                }),
            },
        }

    def build(
        self,
        type: str,
        linear: int,
        linear_alpha: int,
        conv: int = 0,
        conv_alpha: int = 0,
        dropout: float = 0.0,
        network_kwargs_json: str = "",
    ):
        config = {
            "type": type,
            "linear": linear,
            "linear_alpha": linear_alpha,
        }

        if conv > 0:
            config["conv"] = conv
            config["conv_alpha"] = conv_alpha

        if dropout > 0:
            config["dropout"] = dropout

        if network_kwargs_json.strip():
            try:
                extra = json.loads(network_kwargs_json)
                if isinstance(extra, dict):
                    config.update(extra)
            except json.JSONDecodeError:
                pass

        return (config,)
