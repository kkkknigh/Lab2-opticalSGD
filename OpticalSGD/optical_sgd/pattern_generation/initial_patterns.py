"""Initial projection pattern generators."""

from __future__ import annotations

import numpy as np


def create_initial_patterns(
    count: int,
    projector_width: int,
    method: str = "random",
    seed: int = 7,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if method == "constant":
        values = np.full((count, projector_width), 0.5, dtype=np.float32)
    elif method == "stripes":
        x = np.linspace(0.0, 2.0 * np.pi, projector_width, dtype=np.float32)
        values = np.stack([0.5 + 0.45 * np.sin(x * (i + 1)) for i in range(count)])
    else:
        values = rng.uniform(0.45, 0.55, (count, projector_width)).astype(np.float32)
    return np.clip(values, 0.0, 1.0).astype(np.float32)
