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
    __slots__ = ("step", "total_steps", "loss", "message")

    def __init__(self):
        self.step = 0
        self.total_steps = 0
        self.loss = 0.0
        self.message = ""


class AIToolkitProcess:
    # Patterns to parse ai-toolkit stdout
    # tqdm style: "  5%|█         | 100/2000 [02:30<47:00, 1.26s/it, loss=0.123]"
    TQDM_PATTERN = re.compile(
        r"(\d+)/(\d+)\s*\[.*?(?:loss[=:]\s*([\d.]+))?"
    )
    # Simple loss pattern: "loss: 0.1234" or "loss=0.1234"
    LOSS_PATTERN = re.compile(r"loss[=:]\s*([\d.]+)")
    # Step pattern from log lines: "step 100/2000" or "Step: 100"
    STEP_PATTERN = re.compile(r"[Ss]tep[:\s]+(\d+)(?:\s*/\s*(\d+))?")
    # Sampling indicator
    SAMPLE_PATTERN = re.compile(r"[Ss]ampl|[Gg]enerating\s+sample")
    # Save indicator
    SAVE_PATTERN = re.compile(r"[Ss]aving|[Cc]heckpoint\s+saved")

    def __init__(self, config_path: str, ai_toolkit_dir: str):
        self.config_path = config_path
        self.ai_toolkit_dir = ai_toolkit_dir
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
        """Parse a line for progress information."""
        # Try tqdm pattern first
        m = self.TQDM_PATTERN.search(line)
        if m:
            self._latest_progress.step = int(m.group(1))
            self._latest_progress.total_steps = int(m.group(2))
            if m.group(3):
                self._latest_progress.loss = float(m.group(3))
            return

        # Try step pattern
        m = self.STEP_PATTERN.search(line)
        if m:
            self._latest_progress.step = int(m.group(1))
            if m.group(2):
                self._latest_progress.total_steps = int(m.group(2))

        # Try loss pattern
        m = self.LOSS_PATTERN.search(line)
        if m:
            self._latest_progress.loss = float(m.group(1))

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
