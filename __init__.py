"""ComfyUI AI Toolkit - LoRA Training Nodes

Custom node package wrapping ostris/ai-toolkit for training LoRA models
directly from ComfyUI. Currently supports FLUX.2 Klein 9B and ZImage Turbo.
"""

import os
import sys
import importlib

# Ensure our package root is on sys.path so submodule imports work
_pkg_root = os.path.dirname(__file__)
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

# Import node classes directly to avoid conflicts with ComfyUI's own 'nodes' module
from nodes.model_config import AIToolkitModelConfig
from nodes.network_config import AIToolkitNetworkConfig
from nodes.train_config import AIToolkitTrainConfig
from nodes.dataset_config import AIToolkitDatasetConfig
from nodes.dataset_list import AIToolkitDatasetList
from nodes.sample_config import AIToolkitSampleConfig
from nodes.save_config import AIToolkitSaveConfig
from nodes.embedding_config import AIToolkitEmbeddingConfig
from nodes.caption_config import AIToolkitCaptionConfig
from nodes.train_execute import AIToolkitTrainExecute

NODE_CLASS_MAPPINGS = {
    "AIToolkitModelConfig": AIToolkitModelConfig,
    "AIToolkitNetworkConfig": AIToolkitNetworkConfig,
    "AIToolkitTrainConfig": AIToolkitTrainConfig,
    "AIToolkitDatasetConfig": AIToolkitDatasetConfig,
    "AIToolkitDatasetList": AIToolkitDatasetList,
    "AIToolkitSampleConfig": AIToolkitSampleConfig,
    "AIToolkitSaveConfig": AIToolkitSaveConfig,
    "AIToolkitEmbeddingConfig": AIToolkitEmbeddingConfig,
    "AIToolkitCaptionConfig": AIToolkitCaptionConfig,
    "AIToolkitTrainExecute": AIToolkitTrainExecute,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIToolkitModelConfig": "AI Toolkit - Model Config",
    "AIToolkitNetworkConfig": "AI Toolkit - Network Config",
    "AIToolkitTrainConfig": "AI Toolkit - Training Config",
    "AIToolkitDatasetConfig": "AI Toolkit - Dataset Config",
    "AIToolkitDatasetList": "AI Toolkit - Dataset List",
    "AIToolkitSampleConfig": "AI Toolkit - Sample Config",
    "AIToolkitSaveConfig": "AI Toolkit - Save Config",
    "AIToolkitEmbeddingConfig": "AI Toolkit - Embedding Config",
    "AIToolkitCaptionConfig": "AI Toolkit - Caption Config",
    "AIToolkitTrainExecute": "AI Toolkit - Train LoRA",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
