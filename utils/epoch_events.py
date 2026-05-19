"""Emits training contract events the comfy-api CharacterTrainingHandler consumes."""

from datetime import datetime, timedelta, timezone

SELECTION_WINDOW_DAYS = 3


def _send(client_id, event, payload):
    from server import PromptServer

    sid = client_id if client_id else None
    PromptServer.instance.send_sync(event, payload, sid)


def emit_progress(client_id, progress_data):
    _send(client_id, "progress", {"progressData": progress_data})


def emit_message(client_id, message):
    _send(client_id, "progress", {"message": message})


def emit_epoch_uploaded(client_id, task_id, user_id, epoch, avg_loss, step, lora_url, sample_urls):
    _send(client_id, "training.epoch.uploaded", {
        "task_id": int(task_id),
        "user_id": int(user_id),
        "epoch": int(epoch),
        "avg_loss": float(avg_loss),
        "step": int(step),
        "lora_url": lora_url,
        "sample_images": [{"url": url} for url in sample_urls],
    })


def emit_task_completed(client_id, task_id, user_id, epoch):
    selection_expires_at = (
        datetime.now(timezone.utc) + timedelta(days=SELECTION_WINDOW_DAYS)
    ).isoformat().replace("+00:00", "Z")

    _send(client_id, "training.task.completed", {
        "task_id": int(task_id),
        "user_id": int(user_id),
        "status": "TrainingCompleted",
        "epoch_number": int(epoch),
        "selection_expires_at": selection_expires_at,
    })


def emit_training_failed(client_id, message):
    _send(client_id, "training.failed", {"message": str(message)})
