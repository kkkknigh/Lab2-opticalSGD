"""Frosted-glass material.

This is a translucent approximation rather than a path-traced glass model.  The
high scattering map intentionally mixes neighboring projector columns before the
camera response, imitating blurred subsurface/indirect transport.
"""

from __future__ import annotations

import numpy as np

from optical_sgd.synthetic_scene.materials.base import MaterialMaps, normalized_grid


def make_frosted_glass(height: int, camera_width: int) -> MaterialMaps:
    yy, xx = normalized_grid(height, camera_width)
    albedo = 0.68 + 0.06 * np.sin(28.0 * xx) * np.sin(21.0 * yy)
    specular = 0.08 + 0.12 * np.exp(-((xx - 0.65) ** 2 + (yy - 0.35) ** 2) / 0.025)
    scattering = 0.22 + 0.12 * np.exp(-((xx - 0.45) ** 2 + (yy - 0.55) ** 2) / 0.08)
    return MaterialMaps(
        name="frosted_glass",
        albedo=np.clip(albedo, 0.05, 1.0).astype(np.float32),
        specular=np.clip(specular, 0.0, 0.35).astype(np.float32),
        scattering=np.clip(scattering, 0.0, 0.45).astype(np.float32),
        projector_gamma=1.15,
        camera_gamma=1.12,
        description="Translucent surface approximation with strong local projector-column mixing.",
    )
