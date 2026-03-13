"""ComfyUI AI Toolkit - LoRA Training Nodes"""

from aitk_nodes.model_config import AIToolkitModelConfig
from aitk_nodes.network_config import AIToolkitNetworkConfig
from aitk_nodes.train_config import AIToolkitTrainConfig
from aitk_nodes.dataset_config import AIToolkitDatasetConfig
from aitk_nodes.dataset_list import AIToolkitDatasetList
from aitk_nodes.sample_config import AIToolkitSampleConfig
from aitk_nodes.save_config import AIToolkitSaveConfig
from aitk_nodes.embedding_config import AIToolkitEmbeddingConfig
from aitk_nodes.caption_config import AIToolkitCaptionConfig
from aitk_nodes.train_execute import AIToolkitTrainExecute

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
