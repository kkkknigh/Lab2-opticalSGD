from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter

_LOG_FILE: Path | None = None


def configure_log_file(path: Path | None) -> None:
    """设置当前实验的日志文件；为 None 时只输出到控制台。"""

    global _LOG_FILE
    _LOG_FILE = None if path is None else Path(path)
    if _LOG_FILE is not None:
        _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        _LOG_FILE.write_text("", encoding="utf-8")


def log_step(message: str) -> None:
    """输出带时间戳的实验进度日志。"""

    line = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    if _LOG_FILE is not None:
        with _LOG_FILE.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class StepTimer:
    """记录单个实验步骤耗时的上下文管理器。"""

    def __init__(self, message: str):
        self.message = message
        self.elapsed_seconds = 0.0

    def __enter__(self):
        self._start = perf_counter()
        log_step(f"START {self.message}")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.elapsed_seconds = perf_counter() - self._start
        status = "FAILED" if exc_type else "DONE"
        log_step(f"{status} {self.message} ({self.elapsed_seconds:.2f}s)")
        return False
