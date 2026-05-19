"""Downloads and extracts the training dataset archive.

Moved here from ComfyUI-Fantasio-Nodes: it is used only by the training workflow.
The download/extract implementation stays in the shared Fantasio library.
"""

import os
import importlib.util

from server import PromptServer

PROGRESS_REPORT_STEP_PERCENT = 10


def _load_fantasio_lib():
    here = os.path.dirname(os.path.abspath(__file__))
    custom_nodes_dir = os.path.dirname(os.path.dirname(here))
    lib_path = os.path.join(custom_nodes_dir, "fantasio", "lib.py")

    if not os.path.isfile(lib_path):
        raise RuntimeError(f"Fantasio shared library not found at {lib_path}")

    spec = importlib.util.spec_from_file_location("fantasio_shared_lib", lib_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FantasioDownloadAndExtractArchive:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"multiline": False}),
                "output_dir": ("STRING", {"multiline": False}),
            },
            "optional": {
                "archive_name": ("STRING", {"multiline": False, "default": ""}),
                "timeout_seconds": ("INT", {"default": 120, "min": 5, "max": 7200}),
                "clean_archive": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "client_id": ("STRING",),
                "task_id": ("INT",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_dir",)
    FUNCTION = "run"
    CATEGORY = "AI Toolkit/Fantasio"

    def run(self, url, output_dir, archive_name="", timeout_seconds=120, clean_archive=True, client_id="", task_id=0):
        sid = client_id if client_id else None
        fantasio_lib = _load_fantasio_lib()

        try:
            self._emit(sid, "Downloading training dataset")
            fantasio_lib.download_and_extract_archive(
                url,
                output_dir,
                archive_name=archive_name,
                timeout_seconds=timeout_seconds,
                progress_cb=self._download_progress_cb(sid),
            )
            self._emit(sid, "Training dataset ready")
            return (output_dir,)
        except Exception as e:
            self._emit(sid, f"Dataset download failed: {e}")
            raise

    def _download_progress_cb(self, sid):
        def report(percent):
            if percent % PROGRESS_REPORT_STEP_PERCENT == 0:
                self._emit(sid, f"Downloading training dataset: {percent}%")

        return report

    def _emit(self, sid, message):
        PromptServer.instance.send_sync("progress", {"message": message}, sid)
