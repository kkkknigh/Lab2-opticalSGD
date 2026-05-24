"""Depth surface generators."""

from __future__ import annotations

import numpy as np


def make_depth_surface(height: int, camera_width: int, profile: str = "slanted_wave") -> np.ndarray:
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, height, dtype=np.float32),
        np.linspace(0.0, 1.0, camera_width, dtype=np.float32),
        indexing="ij",
    )
    if profile == "flat":
        depth = np.ones((height, camera_width), dtype=np.float32)
    elif profile == "bump":
        depth = 1.0 + 0.35 * np.exp(-((xx - 0.52) ** 2 + (yy - 0.45) ** 2) / 0.045)
    else:
        depth = 0.8 + 0.45 * xx + 0.08 * np.sin(yy * np.pi * 3.0)
    return depth.astype(np.float32)
