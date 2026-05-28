"""Packs the captioned dataset folder and uploads it back to S3.

Runs in the training phase, after the captioning phase has written the .txt
sidecars to the instance disk. Overwrites the original dataset object so the
stored archive carries the generated captions. Packing and S3 upload reuse the
shared Fantasio library.
"""

import os
import shutil
import tempfile
import zipfile
import importlib.util

from server import PromptServer

# ai-toolkit writes these caches next to the images during training; they must
# never end up in the dataset archive that we re-upload over the original object.
EXCLUDED_DIR_NAMES = {"_latent_cache", "_clip_vision_cache", "_t_e_cache"}


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


class FantasioUploadDataset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"multiline": False}),
                "s3_key": ("STRING", {"multiline": False}),
                "s3_endpoint": ("STRING", {"default": ""}),
                "s3_access_key": ("STRING", {"default": ""}),
                "s3_secret_key": ("STRING", {"default": ""}),
                "s3_bucket": ("STRING", {"default": ""}),
            },
            "hidden": {
                "client_id": ("STRING",),
                "task_id": ("INT",),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "AI Toolkit/Fantasio"

    def run(self, folder_path, s3_key, s3_endpoint, s3_access_key, s3_secret_key, s3_bucket, client_id="", task_id=0):
        sid = client_id if client_id else None
        fantasio_lib = _load_fantasio_lib()

        temp_dir = tempfile.mkdtemp(prefix="fantasio_dataset_upload_")
        try:
            self._emit(sid, "Packing captioned dataset")
            archive_path = os.path.join(temp_dir, "dataset.zip")
            self._pack_dataset(folder_path, archive_path)

            self._emit(sid, "Uploading captioned dataset")
            s3 = fantasio_lib.create_s3_client(s3_endpoint, s3_access_key, s3_secret_key)
            fantasio_lib.upload_file_to_s3(s3, archive_path, s3_bucket, s3_key)

            self._emit(sid, "Captioned dataset uploaded")
            return {}
        except Exception as e:
            self._emit(sid, f"Dataset upload failed: {e}")
            raise
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _pack_dataset(self, folder_path, archive_path):
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for root, dirs, files in os.walk(folder_path):
                dirs[:] = [d for d in dirs if d not in EXCLUDED_DIR_NAMES and not d.startswith(".")]
                for file_name in files:
                    if file_name.startswith("."):
                        continue
                    file_path = os.path.join(root, file_name)
                    archive.write(file_path, os.path.relpath(file_path, folder_path))

    def _emit(self, sid, message):
        PromptServer.instance.send_sync("progress", {"message": message}, sid)
