"""Uploads a trained LoRA checkpoint and its epoch samples to S3."""

import os
import shutil
import tempfile

from PIL import Image

SAMPLE_WEBP_QUALITY = 90


def upload_epoch_artifacts(fantasio_lib, context, epoch, checkpoint_path, sample_paths):
    s3 = fantasio_lib.create_s3_client(
        context["s3_endpoint"],
        context["s3_access_key"],
        context["s3_secret_key"],
    )

    bucket = context["s3_bucket"]
    public_url = context["s3_public_url"]
    prefix = context["s3_key_prefix"].rstrip("/")

    lora_filename = os.path.basename(checkpoint_path)
    lora_key = f"{prefix}/checkpoints/{lora_filename}"
    fantasio_lib.upload_file_to_s3(s3, checkpoint_path, bucket, lora_key)
    lora_url = fantasio_lib.normalize_s3_public_url(public_url, lora_key)

    sample_urls = upload_epoch_samples(fantasio_lib, s3, bucket, public_url, prefix, epoch, sample_paths)

    return lora_url, sample_urls


def upload_epoch_samples(fantasio_lib, s3, bucket, public_url, prefix, epoch, sample_paths):
    if not sample_paths:
        return []

    sample_urls = []
    temp_dir = tempfile.mkdtemp(prefix="aitk_epoch_samples_")

    try:
        for index, sample_path in enumerate(sample_paths):
            webp_name = f"epoch_{epoch}_{index}.webp"
            webp_path = os.path.join(temp_dir, webp_name)

            with Image.open(sample_path) as image:
                image.convert("RGB").save(webp_path, format="WEBP", quality=SAMPLE_WEBP_QUALITY, method=4)

            sample_key = f"{prefix}/samples/epoch_{epoch}/{webp_name}"
            fantasio_lib.upload_file_to_s3(s3, webp_path, bucket, sample_key)
            sample_urls.append(fantasio_lib.normalize_s3_public_url(public_url, sample_key))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return sample_urls
