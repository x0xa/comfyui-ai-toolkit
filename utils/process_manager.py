"""Manages ai-toolkit training subprocess lifecycle."""

import os
import re
import sys
import signal
import subprocess
import threading
import queue
from typing import Optional


class ProgressInfo:
    __slots__ = ("step", "total_steps", "loss", "avg_loss", "message", "phase", "_loss_sum", "_loss_count")

    def __init__(self):
        self.step = 0
        self.total_steps = 0
        self.loss = 0.0
        self.avg_loss = 0.0
        self.message = ""
        self.phase = "Preparing training"
        self._loss_sum = 0.0
        self._loss_count = 0

    def record_loss(self, value: float):
        self.loss = value
        self._loss_sum += value
        self._loss_count += 1
        self.avg_loss = self._loss_sum / self._loss_count

    def reset_loss_window(self):
        self._loss_sum = 0.0
        self._loss_count = 0
        self.avg_loss = 0.0


class AIToolkitProcess:
    TQDM_PATTERN = re.compile(r"(\d+)\s*/\s*(\d+)\s*\[")
    LOSS_PATTERN = re.compile(r"[Ll]oss\s*[=:]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
    STEP_LOG_PATTERN = re.compile(r"[Ss]tep[:\s]+(\d+)(?:\s*/\s*(\d+))?")
    SAMPLE_PATTERN = re.compile(r"[Gg]enerating\s+sample|[Ss]ampling\s+prompts?")
    SAVE_PATTERN = re.compile(r"[Ss]aving\s+checkpoint|[Cc]heckpoint\s+saved")
    CACHE_PATTERN = re.compile(r"[Cc]aching\s+latents")
    QUANTIZE_PATTERN = re.compile(r"[Qq]uantiz")
    LOAD_PATTERN = re.compile(r"[Ll]oading\s+(?:model|Qwen|VAE|transformer)")

    def __init__(self, config_path: str, ai_toolkit_dir: str, train_steps: Optional[int] = None):
        self.config_path = config_path
        self.ai_toolkit_dir = ai_toolkit_dir
        self.train_steps = train_steps
        self.process: Optional[subprocess.Popen] = None
        self._output_queue: queue.Queue = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None
        self._all_output: list[str] = []
        self._latest_progress = ProgressInfo()

    def start(self):
        """Launch the ai-toolkit training subprocess."""
        env = os.environ.copy()
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        env["NO_ALBUMENTATIONS_UPDATE"] = "1"
        env["DISABLE_TELEMETRY"] = "YES"
        # Ensure unbuffered python output for real-time progress
        env["PYTHONUNBUFFERED"] = "1"

        self.process = subprocess.Popen(
            [sys.executable, "run.py", self.config_path],
            cwd=self.ai_toolkit_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        self._reader_thread = threading.Thread(
            target=self._read_output, daemon=True
        )
        self._reader_thread.start()

    def _read_output(self):
        """Background thread to read subprocess output."""
        try:
            for line in self.process.stdout:
                line = line.rstrip("\n\r")
                self._all_output.append(line)
                self._output_queue.put(line)
                self._parse_progress(line)
        except (ValueError, OSError):
            pass

    def _parse_progress(self, line: str):
        phase_from_marker = self._detect_phase_marker(line)
        if phase_from_marker:
            self._latest_progress.phase = phase_from_marker

        tqdm_match = self.TQDM_PATTERN.search(line)
        if tqdm_match:
            current = int(tqdm_match.group(1))
            total = int(tqdm_match.group(2))
            is_training_tqdm = (
                self.train_steps is not None and total == self.train_steps
            )
            if is_training_tqdm:
                self._latest_progress.step = current
                self._latest_progress.total_steps = total
                self._latest_progress.phase = "Training"

        log_step_match = self.STEP_LOG_PATTERN.search(line)
        if log_step_match and (
            self.train_steps is None
            or (log_step_match.group(2) and int(log_step_match.group(2)) == self.train_steps)
        ):
            self._latest_progress.step = int(log_step_match.group(1))
            if log_step_match.group(2):
                self._latest_progress.total_steps = int(log_step_match.group(2))
            self._latest_progress.phase = "Training"

        loss_match = self.LOSS_PATTERN.search(line)
        if loss_match:
            try:
                self._latest_progress.record_loss(float(loss_match.group(1)))
            except ValueError:
                pass

    def _detect_phase_marker(self, line: str) -> Optional[str]:
        if self.SAVE_PATTERN.search(line):
            return "Saving checkpoint"
        if self.SAMPLE_PATTERN.search(line):
            return "Generating samples"
        if self.CACHE_PATTERN.search(line):
            return "Caching latents"
        if self.QUANTIZE_PATTERN.search(line):
            return "Quantizing model"
        if self.LOAD_PATTERN.search(line):
            return "Loading model"
        return None

    def get_new_lines(self) -> list[str]:
        """Get all new output lines since last call (non-blocking)."""
        lines = []
        while True:
            try:
                lines.append(self._output_queue.get_nowait())
            except queue.Empty:
                break
        return lines

    @property
    def progress(self) -> ProgressInfo:
        return self._latest_progress

    @property
    def full_output(self) -> str:
        return "\n".join(self._all_output)

    def is_running(self) -> bool:
        if self.process is None:
            return False
        return self.process.poll() is None

    @property
    def return_code(self) -> Optional[int]:
        if self.process is None:
            return None
        return self.process.poll()

    def terminate(self):
        """Gracefully terminate the subprocess."""
        if self.process and self.is_running():
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)

    def wait(self, timeout=None) -> int:
        """Wait for the process to finish and return the exit code."""
        if self.process is None:
            return -1
        return self.process.wait(timeout=timeout)
