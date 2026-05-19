"""Fantasio training context node.

Carries the comfy-api task identity and S3 destination into AIToolkitTrainExecute
so per-epoch checkpoint uploads and contract events can be produced during training.
"""


class FantasioTrainingContext:
    CATEGORY = "AI Toolkit/Fantasio"
    RETURN_TYPES = ("AITK_FANTASIO_CTX",)
    RETURN_NAMES = ("fantasio_context",)
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_id": ("INT", {"default": 0, "min": 0}),
                "user_id": ("INT", {"default": 0, "min": 0}),
                "trigger_word": ("STRING", {"default": ""}),
                "total_epochs": ("INT", {"default": 0, "min": 0}),
                "s3_endpoint": ("STRING", {"default": ""}),
                "s3_access_key": ("STRING", {"default": ""}),
                "s3_secret_key": ("STRING", {"default": ""}),
                "s3_bucket": ("STRING", {"default": ""}),
                "s3_public_url": ("STRING", {"default": ""}),
                "s3_key_prefix": ("STRING", {"default": ""}),
            },
            "hidden": {
                "client_id": ("STRING",),
            },
        }

    def build(self, task_id, user_id, trigger_word, total_epochs,
              s3_endpoint, s3_access_key, s3_secret_key, s3_bucket,
              s3_public_url, s3_key_prefix, client_id=""):
        context = {
            "task_id": int(task_id),
            "user_id": int(user_id),
            "trigger_word": trigger_word,
            "total_epochs": int(total_epochs),
            "s3_endpoint": s3_endpoint,
            "s3_access_key": s3_access_key,
            "s3_secret_key": s3_secret_key,
            "s3_bucket": s3_bucket,
            "s3_public_url": s3_public_url,
            "s3_key_prefix": s3_key_prefix,
            "client_id": client_id,
        }

        return (context,)
