"""Synthetic scene container."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SceneDescription:
    height: int
    camera_width: int
    projector_width: int
    depth: np.ndarray
    albedo: np.ndarray
    specular: np.ndarray
    scattering: np.ndarray
    correspondence: np.ndarray
    valid_mask: np.ndarray
    material_name: str
    projector_gamma: float = 1.0
    camera_gamma: float = 1.0
    material_description: str = ""
