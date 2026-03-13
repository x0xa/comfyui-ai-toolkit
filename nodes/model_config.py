"""Model configuration node for ai-toolkit training."""


class AIToolkitModelConfig:
    CATEGORY = "AI Toolkit/Config"
    RETURN_TYPES = ("AITK_MODEL_CONFIG",)
    RETURN_NAMES = ("model_config",)
    FUNCTION = "build"

    ARCHITECTURES = [
        "flux2_klein_9b",
        "flux2_klein_4b",
        "zimage",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name_or_path": ("STRING", {
                    "default": "",
                    "tooltip": "Local path to transformer weights (e.g. /models/flux-2-klein-base-9b.safetensors or folder containing it)",
                }),
                "arch": (cls.ARCHITECTURES, {
                    "default": "flux2_klein_9b",
                    "tooltip": "Model architecture",
                }),
                "te_name_or_path": ("STRING", {
                    "default": "",
                    "tooltip": "Local path to text encoder (e.g. /models/Qwen3-8B or /models/Qwen3-8B-FP8)",
                }),
                "vae_path": ("STRING", {
                    "default": "",
                    "tooltip": "Local path to VAE weights (e.g. /models/ae.safetensors)",
                }),
            },
            "optional": {
                "quantize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Quantize transformer to 8-bit for lower VRAM usage",
                }),
                "quantize_te": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Quantize text encoder (disable if already using fp8 weights)",
                }),
                "low_vram": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable low VRAM mode (offloads to CPU, slower)",
                }),
                "qtype": ("STRING", {
                    "default": "qfloat8",
                    "tooltip": "Quantization type for transformer (qfloat8, qint8, qint4, etc.)",
                }),
                "qtype_te": ("STRING", {
                    "default": "qfloat8",
                    "tooltip": "Quantization type for text encoder",
                }),
                "extras_name_or_path": ("STRING", {
                    "default": "",
                    "tooltip": "Separate path for auxiliary components (ZImage tokenizer/TE/VAE). Defaults to name_or_path",
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
        te_name_or_path: str,
        vae_path: str,
        quantize: bool = True,
        quantize_te: bool = False,
        low_vram: bool = False,
        qtype: str = "qfloat8",
        qtype_te: str = "qfloat8",
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

        if te_name_or_path:
            config["te_name_or_path"] = te_name_or_path

        if vae_path:
            config["vae_path"] = vae_path

        if quantize_te:
            config["quantize_te"] = True

        if low_vram:
            config["low_vram"] = True

        if qtype and qtype != "qfloat8":
            config["qtype"] = qtype

        if qtype_te and qtype_te != "qfloat8":
            config["qtype_te"] = qtype_te

        if extras_name_or_path:
            config["extras_name_or_path"] = extras_name_or_path

        if assistant_lora_path:
            config["assistant_lora_path"] = assistant_lora_path

        if layer_offloading:
            config["layer_offloading"] = True
            config["layer_offloading_transformer_percent"] = layer_offloading_transformer_percent
            config["layer_offloading_text_encoder_percent"] = layer_offloading_text_encoder_percent

        return (config,)
