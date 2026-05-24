"""Diffuse material.

This is the Lambertian baseline: a smooth albedo map, no specular highlight,
almost no subsurface/indirect mixing, and linear projector/camera response.
"""

from __future__ import annotations

import numpy as np

from optical_sgd.synthetic_scene.materials.base import MaterialMaps, constant_map, normalized_grid


def make_diffuse(height: int, camera_width: int) -> MaterialMaps:
    _, xx = normalized_grid(height, camera_width)
    albedo = 0.75 + 0.08 * np.sin(4.0 * np.pi * xx)
    return MaterialMaps(
        name="diffuse",
        albedo=np.clip(albedo, 0.05, 1.0).astype(np.float32),
        specular=constant_map(height, camera_width, 0.0),
        scattering=constant_map(height, camera_width, 0.02),
        projector_gamma=1.0,
        camera_gamma=1.0,
        description="Lambertian baseline with mild sinusoidal albedo.",
    )
