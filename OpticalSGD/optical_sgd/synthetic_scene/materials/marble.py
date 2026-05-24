"""Marble material.

Marble is represented by high-contrast sinusoidal veins.  It keeps moderate
specular and scattering terms so ZNCC must handle both textured albedo and mild
indirect light, which matches the assignment's required marble robustness case.
"""

from __future__ import annotations

import numpy as np

from optical_sgd.synthetic_scene.materials.base import MaterialMaps, normalized_grid


def make_marble(height: int, camera_width: int) -> MaterialMaps:
    yy, xx = normalized_grid(height, camera_width)
    veins = np.sin(18.0 * xx + 7.0 * np.sin(10.0 * yy))
    fine = 0.08 * np.sin(61.0 * xx + 13.0 * yy)
    albedo = 0.56 + 0.31 * veins + fine
    specular = 0.05 + 0.1 * (veins > 0.55).astype(np.float32)
    scattering = 0.08 + 0.06 * (1.0 - np.abs(veins))
    return MaterialMaps(
        name="marble",
        albedo=np.clip(albedo, 0.05, 1.0).astype(np.float32),
        specular=np.clip(specular, 0.0, 0.3).astype(np.float32),
        scattering=np.clip(scattering, 0.0, 0.25).astype(np.float32),
        projector_gamma=1.08,
        camera_gamma=1.04,
        description="Veined stone with contrast texture, mild specular response, and local scattering.",
    )
