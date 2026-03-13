"""Embedding / trigger word configuration node."""


class AIToolkitEmbeddingConfig:
    CATEGORY = "AI Toolkit/Config"
    RETURN_TYPES = ("AITK_EMBEDDING_CONFIG",)
    RETURN_NAMES = ("embedding_config",)
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger_word": ("STRING", {
                    "default": "",
                    "tooltip": "Trigger word added to captions (e.g. 'ohwx'). [trigger] in prompts will be replaced with this",
                }),
            },
        }

    def build(self, trigger_word: str):
        config = {}
        if trigger_word.strip():
            config["trigger_word"] = trigger_word.strip()
        return (config,)
