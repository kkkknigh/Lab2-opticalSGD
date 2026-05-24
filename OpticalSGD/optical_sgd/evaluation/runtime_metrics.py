"""Runtime metric helpers."""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter


@contextmanager
def measure_seconds(container: dict, key: str):
    start = perf_counter()
    yield
    container[key] = perf_counter() - start
