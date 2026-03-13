"""Assembles node outputs into ai-toolkit YAML config dict."""

import copy
from typing import Optional


def build_config(
    job_name: str,
    training_folder: str,
    device: str,
    model_config: dict,
    network_config: dict,
    train_config: dict,
    dataset_configs: list,
    save_config: dict,
    sample_config: Optional[dict] = None,
    embedding_config: Optional[dict] = None,
) -> dict:
    """Build the full ai-toolkit YAML config dict from node outputs."""

    process = {
        "type": "sd_trainer",
        "training_folder": training_folder,
        "device": device,
        "network": copy.deepcopy(network_config),
        "save": copy.deepcopy(save_config),
        "datasets": [copy.deepcopy(d) for d in dataset_configs],
        "train": copy.deepcopy(train_config),
        "model": copy.deepcopy(model_config),
    }

    if sample_config is not None:
        process["sample"] = copy.deepcopy(sample_config)

    if embedding_config is not None:
        trigger = embedding_config.get("trigger_word", "")
        if trigger:
            process["trigger_word"] = trigger

    config = {
        "job": "extension",
        "config": {
            "name": job_name,
            "process": [process],
        },
        "meta": {
            "name": "[name]",
            "version": "1.0",
        },
    }

    return config
