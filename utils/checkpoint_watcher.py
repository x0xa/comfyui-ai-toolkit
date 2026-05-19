"""Watches the training job folder for new LoRA checkpoints."""

import os


class CheckpointWatcher:
    def __init__(self, output_dir, job_name):
        self.job_dir = os.path.join(output_dir, job_name)
        self._seen = set()

    def check_new_checkpoints(self):
        if not os.path.isdir(self.job_dir):
            return []

        new_checkpoints = []
        for entry in os.scandir(self.job_dir):
            if not entry.is_file():
                continue
            if not entry.name.lower().endswith(".safetensors"):
                continue
            if entry.path in self._seen:
                continue
            self._seen.add(entry.path)
            new_checkpoints.append(entry.path)

        new_checkpoints.sort(key=lambda path: os.path.getmtime(path))
        return new_checkpoints
