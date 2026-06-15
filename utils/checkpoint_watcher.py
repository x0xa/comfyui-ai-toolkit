"""Watches the training job folder for new LoRA checkpoints."""

import json
import os
import struct

# safetensors layout: 8-byte little-endian header length, then that many bytes
# of JSON header, then the tensor data buffer. A checkpoint is only complete
# once the file covers every tensor's declared byte range; the trainer writes
# the file in place (non-atomically), so a freshly appeared file may still be
# mid-write and would fail to deserialize.
_HEADER_LEN_BYTES = 8


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
            if not _is_complete_safetensors(entry.path):
                continue
            self._seen.add(entry.path)
            new_checkpoints.append(entry.path)

        new_checkpoints.sort(key=lambda path: os.path.getmtime(path))
        return new_checkpoints


def _is_complete_safetensors(path):
    try:
        file_size = os.path.getsize(path)
        if file_size < _HEADER_LEN_BYTES:
            return False

        with open(path, "rb") as handle:
            header_len = struct.unpack("<Q", handle.read(_HEADER_LEN_BYTES))[0]
            if file_size < _HEADER_LEN_BYTES + header_len:
                return False
            header = json.loads(handle.read(header_len))

        data_end = 0
        for name, info in header.items():
            if name == "__metadata__":
                continue
            data_end = max(data_end, info["data_offsets"][1])

        return file_size >= _HEADER_LEN_BYTES + header_len + data_end
    except (OSError, ValueError, KeyError, struct.error):
        return False
