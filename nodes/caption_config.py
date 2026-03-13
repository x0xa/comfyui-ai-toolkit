"""Auto-captioning configuration node.

Runs captioning as a pre-step before training using Florence-2,
the same model used in ai-toolkit's built-in UI.
"""

import os
import sys
import subprocess


class AIToolkitCaptionConfig:
    CATEGORY = "AI Toolkit/Config"
    RETURN_TYPES = ("AITK_CAPTION_CONFIG",)
    RETURN_NAMES = ("caption_config",)
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable auto-captioning with Florence-2 before training",
                }),
            },
            "optional": {
                "caption_ext": ("STRING", {
                    "default": "txt",
                    "tooltip": "File extension for generated captions",
                }),
                "overwrite": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Overwrite existing caption files",
                }),
                "append_trigger": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Append [trigger] token to generated captions",
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
        caption_ext: str = "txt",
        overwrite: bool = False,
        append_trigger: bool = True,
        prefix: str = "",
        suffix: str = "",
    ):
        config = {
            "enabled": enabled,
            "caption_ext": caption_ext,
            "overwrite": overwrite,
            "append_trigger": append_trigger,
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
        """Run Florence-2 auto-captioning on images in the dataset folder.

        Returns (success, message).
        """
        if not caption_config.get("enabled", False):
            return True, "Captioning disabled, skipping"

        if not os.path.isdir(dataset_folder):
            return False, f"Dataset folder not found: {dataset_folder}"

        caption_ext = caption_config.get("caption_ext", "txt")
        overwrite = caption_config.get("overwrite", False)
        append_trigger = caption_config.get("append_trigger", True)
        prefix = caption_config.get("prefix", "")
        suffix = caption_config.get("suffix", "")

        # Florence-2 captioning script based on ai-toolkit's flux_train_ui.py
        script = f"""
import os, glob, torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

dataset_folder = {dataset_folder!r}
caption_ext = {caption_ext!r}
overwrite = {overwrite!r}
append_trigger = {append_trigger!r}
prefix = {prefix!r}
suffix = {suffix!r}

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16

print("Loading Florence-2 model...")
model = AutoModelForCausalLM.from_pretrained(
    "multimodalart/Florence-2-large-no-flash-attn",
    torch_dtype=torch_dtype, trust_remote_code=True
).to(device)
processor = AutoProcessor.from_pretrained(
    "multimodalart/Florence-2-large-no-flash-attn",
    trust_remote_code=True
)

extensions = (".jpg", ".jpeg", ".png", ".webp")
images = []
for ext in extensions:
    images.extend(glob.glob(os.path.join(dataset_folder, f"*{{ext}}")))
images.sort()

print(f"Found {{len(images)}} images to caption")
captioned = 0

for img_path in images:
    base = os.path.splitext(img_path)[0]
    caption_path = f"{{base}}.{{caption_ext}}"

    if os.path.exists(caption_path) and not overwrite:
        print(f"  Skip (exists): {{os.path.basename(img_path)}}")
        continue

    image = Image.open(img_path).convert("RGB")
    prompt = "<DETAILED_CAPTION>"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        generated_text, task=prompt, image_size=(image.width, image.height)
    )
    caption_text = parsed["<DETAILED_CAPTION>"].replace("The image shows ", "")

    if prefix:
        caption_text = f"{{prefix}} {{caption_text}}"
    if suffix:
        caption_text = f"{{caption_text}} {{suffix}}"
    if append_trigger:
        caption_text = f"{{caption_text}} [trigger]"

    with open(caption_path, "w", encoding="utf-8") as f:
        f.write(caption_text)

    captioned += 1
    print(f"  Captioned: {{os.path.basename(img_path)}}")

model.to("cpu")
del model, processor
torch.cuda.empty_cache()

print(f"CAPTIONING_COMPLETE: {{captioned}} images captioned")
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
