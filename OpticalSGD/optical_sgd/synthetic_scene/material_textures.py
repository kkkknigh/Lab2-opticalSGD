"""Compatibility helpers for procedural material maps."""

from __future__ import annotations

import numpy as np

from optical_sgd.synthetic_scene.materials import make_material_maps


def make_material_texture(height: int, camera_width: int, material: str = "diffuse") -> np.ndarray:
    return make_material_maps(height, camera_width, material).albedo
