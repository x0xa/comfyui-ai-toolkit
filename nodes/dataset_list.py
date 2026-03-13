"""Combines multiple dataset configs into a list for multi-dataset training."""


class AIToolkitDatasetList:
    CATEGORY = "AI Toolkit/Config"
    RETURN_TYPES = ("AITK_DATASET_LIST",)
    RETURN_NAMES = ("dataset_list",)
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_1": ("AITK_DATASET_CONFIG",),
            },
            "optional": {
                "dataset_2": ("AITK_DATASET_CONFIG",),
                "dataset_3": ("AITK_DATASET_CONFIG",),
                "dataset_4": ("AITK_DATASET_CONFIG",),
            },
        }

    def build(
        self,
        dataset_1,
        dataset_2=None,
        dataset_3=None,
        dataset_4=None,
    ):
        datasets = [dataset_1]
        for ds in [dataset_2, dataset_3, dataset_4]:
            if ds is not None:
                datasets.append(ds)
        return (datasets,)
