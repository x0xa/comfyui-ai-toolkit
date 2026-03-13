"""Training hyperparameters configuration node."""

import json


class AIToolkitTrainConfig:
    CATEGORY = "AI Toolkit/Config"
    RETURN_TYPES = ("AITK_TRAIN_CONFIG",)
    RETURN_NAMES = ("train_config",)
    FUNCTION = "build"

    OPTIMIZERS = [
        "adamw8bit",
        "adamw",
        "prodigy",
        "adafactor",
        "lion8bit",
        "came",
        "dadaptation",
    ]

    LR_SCHEDULERS = [
        "constant",
        "constant_with_warmup",
        "cosine",
        "cosine_with_restarts",
        "linear",
        "polynomial",
    ]

    NOISE_SCHEDULERS = ["flowmatch", "ddpm"]

    TIMESTEP_TYPES = ["sigmoid", "linear", "weighted", "lumina2_shift"]

    DTYPES = ["bf16", "fp16", "fp32"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {
                    "default": 3000,
                    "min": 1,
                    "max": 100000,
                    "step": 1,
                    "tooltip": "Total number of training steps (500-4000 typical, 3000+ recommended for Klein 9B)",
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Batch size per step",
                }),
                "lr": ("FLOAT", {
                    "default": 1e-4,
                    "min": 1e-8,
                    "max": 1.0,
                    "step": 1e-6,
                    "tooltip": "Learning rate",
                }),
                "optimizer": (cls.OPTIMIZERS, {
                    "default": "adamw8bit",
                    "tooltip": "Optimizer algorithm",
                }),
                "noise_scheduler": (cls.NOISE_SCHEDULERS, {
                    "default": "flowmatch",
                    "tooltip": "Noise scheduler (flowmatch for Flux2/ZImage)",
                }),
                "dtype": (cls.DTYPES, {
                    "default": "bf16",
                    "tooltip": "Training precision",
                }),
                "gradient_checkpointing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Save VRAM by checkpointing gradients (slower but less memory)",
                }),
                "train_unet": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Train the transformer/unet",
                }),
                "train_text_encoder": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Train the text encoder (usually not recommended)",
                }),
            },
            "optional": {
                "gradient_accumulation": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 128,
                    "step": 1,
                    "tooltip": "Gradient accumulation steps (effective batch = batch_size * this)",
                }),
                "lr_scheduler": (cls.LR_SCHEDULERS, {
                    "default": "constant",
                    "tooltip": "Learning rate schedule",
                }),
                "timestep_type": (cls.TIMESTEP_TYPES, {
                    "default": "sigmoid",
                    "tooltip": "Timestep sampling distribution for flowmatch",
                }),
                "noise_offset": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Noise offset for darker/lighter images (0 = disabled)",
                }),
                "min_snr_gamma": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "Min SNR gamma for loss weighting (0 = disabled, 5.0 typical)",
                }),
                "skip_first_sample": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Skip generating sample before training starts",
                }),
                "disable_sampling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Completely disable sample generation during training",
                }),
                "linear_timesteps": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use bell-curved timestep weighting (experimental, may improve results)",
                }),
                "unload_text_encoder": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Unload text encoder during training to save VRAM (use with trigger word)",
                }),
                "optimizer_params_json": ("STRING", {
                    "default": "",
                    "tooltip": "Extra optimizer params as JSON (e.g. {\"weight_decay\": 1e-4})",
                    "multiline": True,
                }),
                "use_ema": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use Exponential Moving Average (smooths learning)",
                }),
                "ema_decay": ("FLOAT", {
                    "default": 0.99,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "tooltip": "EMA decay rate",
                }),
            },
        }

    def build(
        self,
        steps: int,
        batch_size: int,
        lr: float,
        optimizer: str,
        noise_scheduler: str,
        dtype: str,
        gradient_checkpointing: bool,
        train_unet: bool,
        train_text_encoder: bool,
        gradient_accumulation: int = 1,
        lr_scheduler: str = "constant",
        timestep_type: str = "sigmoid",
        noise_offset: float = 0.0,
        min_snr_gamma: float = 0.0,
        skip_first_sample: bool = False,
        disable_sampling: bool = False,
        linear_timesteps: bool = False,
        unload_text_encoder: bool = False,
        optimizer_params_json: str = "",
        use_ema: bool = True,
        ema_decay: float = 0.99,
    ):
        config = {
            "steps": steps,
            "batch_size": batch_size,
            "lr": lr,
            "optimizer": optimizer,
            "noise_scheduler": noise_scheduler,
            "dtype": dtype,
            "gradient_checkpointing": gradient_checkpointing,
            "train_unet": train_unet,
            "train_text_encoder": train_text_encoder,
            "gradient_accumulation": gradient_accumulation,
        }

        if lr_scheduler != "constant":
            config["lr_scheduler"] = lr_scheduler

        if timestep_type != "sigmoid":
            config["timestep_type"] = timestep_type

        if noise_offset > 0:
            config["noise_offset"] = noise_offset

        if min_snr_gamma > 0:
            config["min_snr_gamma"] = min_snr_gamma

        if skip_first_sample:
            config["skip_first_sample"] = True

        if disable_sampling:
            config["disable_sampling"] = True

        if linear_timesteps:
            config["linear_timesteps"] = True

        if unload_text_encoder:
            config["unload_text_encoder"] = True

        if optimizer_params_json.strip():
            try:
                params = json.loads(optimizer_params_json)
                if isinstance(params, dict):
                    config["optimizer_params"] = params
            except json.JSONDecodeError:
                pass

        if use_ema:
            config["ema_config"] = {
                "use_ema": True,
                "ema_decay": ema_decay,
            }

        return (config,)
