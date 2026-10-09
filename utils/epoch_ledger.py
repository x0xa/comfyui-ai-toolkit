"""Keeps the uploaded epochs of a run in ComfyUI's output directory.

comfy-api reads this file through ComfyUI's /view endpoint after it reconnects to the
instance, so an epoch event lost while the socket was down still reaches the database.
"""

import json
import os

LEDGER_SUBFOLDER = "training-ledgers"
LEDGER_FILENAME = "epochs.json"


class EpochLedger:
    def __init__(self, output_directory, task_id):
        self.directory = os.path.join(output_directory, LEDGER_SUBFOLDER, str(task_id))
        self.path = os.path.join(self.directory, LEDGER_FILENAME)
        self.entries = []

    def record(self, payload):
        self.entries.append(payload)
        os.makedirs(self.directory, exist_ok=True)

        temporary_path = f"{self.path}.tmp"
        with open(temporary_path, "w") as handle:
            json.dump(self.entries, handle)

        os.replace(temporary_path, self.path)
