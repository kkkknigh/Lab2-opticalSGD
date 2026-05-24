"""Pattern saving helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_patterns_npz(path: str | Path, patterns: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, patterns=np.asarray(patterns, dtype=np.float32))
