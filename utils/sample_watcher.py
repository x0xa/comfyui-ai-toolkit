"""Watches output folder for new sample images during training."""

import os
import glob
from typing import Optional

import numpy as np
from PIL import Image


class SampleWatcher:
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

    def __init__(self, output_dir: str, job_name: str):
        self.sample_dir = os.path.join(output_dir, job_name, "samples")
        self._seen_files: set[str] = set()

    def check_new_samples(self) -> list[str]:
        """Return list of new image file paths since last check."""
        if not os.path.isdir(self.sample_dir):
            return []

        new_files = []
        for entry in os.scandir(self.sample_dir):
            if not entry.is_file():
                continue
            ext = os.path.splitext(entry.name)[1].lower()
            if ext not in self.IMAGE_EXTENSIONS:
                continue
            if entry.path not in self._seen_files:
                self._seen_files.add(entry.path)
                new_files.append(entry.path)

        new_files.sort()
        return new_files

    def get_latest_samples(self, count: int = 10) -> list[str]:
        """Get the most recent sample images by modification time."""
        if not os.path.isdir(self.sample_dir):
            return []

        all_images = []
        for entry in os.scandir(self.sample_dir):
            if not entry.is_file():
                continue
            ext = os.path.splitext(entry.name)[1].lower()
            if ext in self.IMAGE_EXTENSIONS:
                all_images.append((entry.path, entry.stat().st_mtime))

        all_images.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in all_images[:count]]


def load_images_as_tensor(image_paths: list[str]) -> Optional["torch.Tensor"]:
    """Load image files and return as ComfyUI IMAGE tensor [N, H, W, 3]."""
    import torch

    if not image_paths:
        return None

    tensors = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            arr = np.array(img).astype(np.float32) / 255.0
            tensors.append(torch.from_numpy(arr))
        except Exception:
            continue

    if not tensors:
        return None

    # Resize all to the same size as the first image for batching
    h, w = tensors[0].shape[:2]
    result = []
    for t in tensors:
        if t.shape[0] != h or t.shape[1] != w:
            t = torch.nn.functional.interpolate(
                t.permute(2, 0, 1).unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).permute(1, 2, 0)
        result.append(t)

    return torch.stack(result)
