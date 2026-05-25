"""磨砂玻璃材料。

半透明、强散射和柔化后的高光。
实现上通过较高的 scattering 将相邻投影列混合，模拟光在粗糙半透明表面内部扩散后导致的条纹模糊和对比度下降。
"""

from __future__ import annotations

import numpy as np

from optical_sgd.synthetic_scene.materials.base import MaterialMaps, normalized_grid


def make_frosted_glass(height: int, camera_width: int) -> MaterialMaps:
    yy, xx = normalized_grid(height, camera_width)
    # 反照率：整体偏亮且纹理很弱
    albedo = 0.70 + 0.04 * np.sin(26.0 * xx) * np.sin(19.0 * yy)
    # 镜面项：保留一块被磨砂表面扩散后的柔和高光。
    specular = 0.10 + 0.14 * np.exp(-((xx - 0.66) ** 2 + (yy - 0.34) ** 2) / 0.035)
    # 散射项：高，用来显著混合邻近投影列。
    scattering = 0.28 + 0.16 * np.exp(-((xx - 0.46) ** 2 + (yy - 0.56) ** 2) / 0.09)
    return MaterialMaps(
        name="frosted_glass",
        albedo=np.clip(albedo, 0.05, 1.0).astype(np.float32),
        specular=np.clip(specular, 0.0, 0.38).astype(np.float32),
        scattering=np.clip(scattering, 0.0, 0.55).astype(np.float32),
        # gamma：非线性最强，模拟半透明材料下更明显的亮度压缩。
        projector_gamma=1.18,
        camera_gamma=1.14,
        description="Translucent surface approximation with strong local projector-column mixing.",
    )
