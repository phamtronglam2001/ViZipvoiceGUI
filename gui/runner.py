"""Background task runner for ViZipVoice Slint GUI."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TaskHandle:
    thread: threading.Thread
    process: Optional[subprocess.Popen] = None


class CommandRunner:
    def __init__(self, on_log: Callable[[str], None], on_done: Callable[[int], None]) -> None:
        self._on_log = on_log
        self._on_done = on_done
        self._lock = threading.Lock()
        self._active: Optional[TaskHandle] = None

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._active is not None and self._active.thread.is_alive()

    def _set_active(self, handle: Optional[TaskHandle]) -> None:
        with self._lock:
            self._active = handle

    def _python_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def _run_subprocess(self, cmd: list[str], cwd: Path | None = None) -> int:
        self._on_log(f"$ {' '.join(cmd)}\n")
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd or REPO_ROOT),
            env=self._python_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        with self._lock:
            if self._active is not None:
                self._active.process = process

        assert process.stdout is not None
        for line in process.stdout:
            self._on_log(line)

        return int(process.wait())

    def run_async(self, cmd: list[str], cwd: Path | None = None) -> None:
        if self.busy:
            self._on_log("⚠ Task đang chạy, vui lòng đợi.\n")
            return

        def worker() -> None:
            code = 1
            try:
                code = self._run_subprocess(cmd, cwd=cwd)
            except Exception as exc:
                self._on_log(f"ERROR: {exc}\n")
            finally:
                self._set_active(None)
                self._on_done(code)

        thread = threading.Thread(target=worker, daemon=True)
        self._set_active(TaskHandle(thread=thread))
        thread.start()

    def python_module(self, module: str, *args: str) -> None:
        cmd = [sys.executable, "-m", module, *args]
        self.run_async(cmd)

    def huggingface_download(self, repo: str, local_dir: str) -> None:
        cmd = [
            sys.executable,
            "-m",
            "huggingface_hub.cli",
            "download",
            repo,
            "--local-dir",
            local_dir,
            "--local-dir-use-symlinks",
            "False",
        ]
        self.run_async(cmd)
