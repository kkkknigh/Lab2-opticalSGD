"""运行时间统计工具。"""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter


@contextmanager
def measure_seconds(container: dict, key: str):
    """统计代码块运行时间

    Args:
        container: 用于保存耗时结果的字典。
        key: 耗时结果写入的键名。
    """

    start = perf_counter()
    yield
    container[key] = perf_counter() - start
