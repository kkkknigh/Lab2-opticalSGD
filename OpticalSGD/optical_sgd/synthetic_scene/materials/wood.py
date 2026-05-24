"""Wood material.

Wood uses ring-like albedo variation and directional grain.  The material is
mostly diffuse but has low-frequency texture that can confuse correspondence
when projected patterns are also smooth.
"""

from __future__ import annotations

import numpy as np

from optical_sgd.synthetic_scene.materials.base import MaterialMaps, normalized_grid


def make_wood(height: int, camera_width: int) -> MaterialMaps:
    yy, xx = normalized_grid(height, camera_width)
    rings = np.sin(35.0 * np.sqrt((xx - 0.2) ** 2 + (yy - 0.4) ** 2))
    grain = 0.06 * np.sin(85.0 * xx + 4.0 * np.sin(12.0 * yy))
    albedo = 0.48 + 0.22 * rings + 0.13 * xx + grain
    specular = 0.03 + 0.04 * np.maximum(0.0, np.sin(9.0 * xx))
    scattering = 0.04 + 0.03 * np.abs(np.sin(18.0 * yy))
    return MaterialMaps(
        name="wood",
        albedo=np.clip(albedo, 0.05, 1.0).astype(np.float32),
        specular=np.clip(specular, 0.0, 0.2).astype(np.float32),
        scattering=np.clip(scattering, 0.0, 0.16).astype(np.float32),
        projector_gamma=1.03,
        camera_gamma=1.02,
        description="Ring and grain texture with weak anisotropic shine.",
    )
