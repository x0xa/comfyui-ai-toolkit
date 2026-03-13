"""ComfyUI AI Toolkit - LoRA Training Nodes

Custom node package wrapping ostris/ai-toolkit for training LoRA models
directly from ComfyUI. Currently supports FLUX.2 Klein 9B and ZImage Turbo.
"""

from .nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
