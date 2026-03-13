"""Sample generation configuration during training."""


class AIToolkitSampleConfig:
    CATEGORY = "AI Toolkit/Config"
    RETURN_TYPES = ("AITK_SAMPLE_CONFIG",)
    RETURN_NAMES = ("sample_config",)
    FUNCTION = "build"

    SAMPLERS = ["flowmatch", "ddpm"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": (cls.SAMPLERS, {
                    "default": "flowmatch",
                    "tooltip": "Sampler type (must match noise_scheduler in training config)",
                }),
                "sample_every": ("INT", {
                    "default": 250,
                    "min": 1,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Generate sample images every N training steps",
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 128,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Sample image width",
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 128,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Sample image height",
                }),
                "prompts": ("STRING", {
                    "default": "a woman holding a coffee cup, in a beanie, sitting at a cafe\na bear building a log cabin in the snow covered mountains",
                    "multiline": True,
                    "tooltip": "Sample prompts, one per line. Use [trigger] for trigger word substitution",
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**32 - 1,
                    "tooltip": "Random seed for reproducible samples",
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 4.0,
                    "min": 1.0,
                    "max": 32.0,
                    "step": 0.5,
                    "tooltip": "CFG guidance scale for sampling",
                }),
                "sample_steps": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Number of inference steps for sample generation (50 recommended for Klein Base)",
                }),
            },
            "optional": {
                "neg": ("STRING", {
                    "default": "",
                    "tooltip": "Negative prompt (not used for Flux/ZImage but available)",
                }),
                "walk_seed": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Increment seed for each prompt in a batch",
                }),
            },
        }

    def build(
        self,
        sampler: str,
        sample_every: int,
        width: int,
        height: int,
        prompts: str,
        seed: int,
        guidance_scale: float,
        sample_steps: int = 50,
        neg: str = "",
        walk_seed: bool = True,
    ):
        # Parse prompts: one per line, skip empty
        prompt_list = [p.strip() for p in prompts.split("\n") if p.strip()]

        config = {
            "sampler": sampler,
            "sample_every": sample_every,
            "width": width,
            "height": height,
            "prompts": prompt_list,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "sample_steps": sample_steps,
            "neg": neg,
            "walk_seed": walk_seed,
        }

        return (config,)
