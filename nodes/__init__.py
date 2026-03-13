from .model_config import AIToolkitModelConfig
from .network_config import AIToolkitNetworkConfig
from .train_config import AIToolkitTrainConfig
from .dataset_config import AIToolkitDatasetConfig
from .dataset_list import AIToolkitDatasetList
from .sample_config import AIToolkitSampleConfig
from .save_config import AIToolkitSaveConfig
from .embedding_config import AIToolkitEmbeddingConfig
from .caption_config import AIToolkitCaptionConfig
from .train_execute import AIToolkitTrainExecute

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
