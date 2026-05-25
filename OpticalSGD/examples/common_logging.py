from __future__ import annotations

from datetime import datetime
from time import perf_counter


def log_step(message: str) -> None:
    """输出带时间戳的实验进度日志。"""

    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


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
