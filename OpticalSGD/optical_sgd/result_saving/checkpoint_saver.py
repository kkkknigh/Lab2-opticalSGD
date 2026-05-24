"""Checkpoint saving helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_checkpoint(path: str | Path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
