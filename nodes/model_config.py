"""Model configuration node for ai-toolkit training."""


class AIToolkitModelConfig:
    CATEGORY = "AI Toolkit/Config"
    RETURN_TYPES = ("AITK_MODEL_CONFIG",)
    RETURN_NAMES = ("model_config",)
    FUNCTION = "build"

    # Supported architectures for Flux.2 Klein 9B and ZImage Turbo
    ARCHITECTURES = [
        "flux2_klein_9b",
        "zimage",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name_or_path": ("STRING", {
                    "default": "",
                    "tooltip": "HuggingFace model name or local path to model weights",
                }),
                "arch": (cls.ARCHITECTURES, {
                    "default": "flux2_klein_9b",
                    "tooltip": "Model architecture: flux2_klein_9b or zimage",
                }),
                "quantize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Quantize base model to 8-bit for lower VRAM usage",
                }),
                "quantize_te": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Quantize text encoder separately",
                }),
                "low_vram": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable low VRAM mode (offloads to CPU, slower)",
                }),
            },
            "optional": {
                "qtype": ("STRING", {
                    "default": "qfloat8",
                    "tooltip": "Quantization type for transformer (qfloat8, qint8, qint4, etc.)",
                }),
                "qtype_te": ("STRING", {
                    "default": "qfloat8",
                    "tooltip": "Quantization type for text encoder",
                }),
                "vae_path": ("STRING", {
                    "default": "",
                    "tooltip": "Optional separate VAE path (leave empty to use default)",
                }),
                "extras_name_or_path": ("STRING", {
                    "default": "",
                    "tooltip": "Separate path for text encoder/tokenizer/VAE (ZImage). Defaults to name_or_path",
                }),
                "assistant_lora_path": ("STRING", {
                    "default": "",
                    "tooltip": "Optional assistant LoRA path for ZImage turbo training",
                }),
                "layer_offloading": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable layer-wise CPU/GPU offloading for memory savings",
                }),
                "layer_offloading_transformer_percent": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Percentage of transformer layers to keep on GPU (0-1)",
                }),
                "layer_offloading_text_encoder_percent": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Percentage of text encoder layers to keep on GPU (0-1)",
                }),
            },
        }

    def build(
        self,
        name_or_path: str,
        arch: str,
        quantize: bool,
        quantize_te: bool,
        low_vram: bool,
        qtype: str = "qfloat8",
        qtype_te: str = "qfloat8",
        vae_path: str = "",
        extras_name_or_path: str = "",
        assistant_lora_path: str = "",
        layer_offloading: bool = False,
        layer_offloading_transformer_percent: float = 0.5,
        layer_offloading_text_encoder_percent: float = 0.5,
    ):
        config = {
            "name_or_path": name_or_path,
            "arch": arch,
            "quantize": quantize,
        }

        if quantize_te:
            config["quantize_te"] = True

        if low_vram:
            config["low_vram"] = True

        if qtype and qtype != "qfloat8":
            config["qtype"] = qtype

        if qtype_te and qtype_te != "qfloat8":
            config["qtype_te"] = qtype_te

        if vae_path:
            config["vae_path"] = vae_path

        if extras_name_or_path:
            config["extras_name_or_path"] = extras_name_or_path

        if assistant_lora_path:
            config["assistant_lora_path"] = assistant_lora_path

        if layer_offloading:
            config["layer_offloading"] = True
            config["layer_offloading_transformer_percent"] = layer_offloading_transformer_percent
            config["layer_offloading_text_encoder_percent"] = layer_offloading_text_encoder_percent

        return (config,)
