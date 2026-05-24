"""Shared material map container.

Every material module returns the same table-shaped fields so the renderer does
not need material-specific branches.  The maps are deliberately simple
approximations of optical effects that matter for structured light: albedo
changes signal strength, specular adds view-dependent highlights, scattering
mixes neighboring projector columns, and gamma models nonlinear device response.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MaterialMaps:
    name: str
    albedo: np.ndarray
    specular: np.ndarray
    scattering: np.ndarray
    projector_gamma: float
    camera_gamma: float
    description: str


def normalized_grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, height, dtype=np.float32),
        np.linspace(0.0, 1.0, width, dtype=np.float32),
        indexing="ij",
    )
    return yy, xx


def constant_map(height: int, width: int, value: float) -> np.ndarray:
    return np.full((height, width), float(value), dtype=np.float32)

