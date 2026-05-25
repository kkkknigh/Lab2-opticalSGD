"""大理石材料。

高对比度的弯曲纹理和局部亮色脉络，镜面反射比较明显，同时石材内部会产生轻微
半透明感和局部散射。
"""

from __future__ import annotations

import numpy as np

from optical_sgd.synthetic_scene.materials.base import MaterialMaps, normalized_grid


def make_marble(height: int, camera_width: int) -> MaterialMaps:
    yy, xx = normalized_grid(height, camera_width)
    # 主纹理：低频弯曲条纹，形成大理石最强的亮暗结构。
    veins = np.sin(20.0 * xx + 8.0 * np.sin(9.0 * yy))
    # 细纹理：高频弱条纹，避免表面过于规则。
    fine = 0.07 * np.sin(68.0 * xx + 15.0 * yy)
    # 反照率：整体偏亮，但脉络对比度高。
    albedo = 0.60 + 0.30 * veins + fine
    # 镜面项：亮色石纹区域更容易产生高光。
    specular = 0.06 + 0.13 * (veins > 0.50).astype(np.float32)
    # 散射项：在主石纹亮暗过渡区域更强，模拟石材内部浅层光扩散。
    scattering = 0.07 + 0.08 * (1.0 - np.abs(veins))
    return MaterialMaps(
        name="marble",
        albedo=np.clip(albedo, 0.05, 1.0).astype(np.float32),
        specular=np.clip(specular, 0.0, 0.32).astype(np.float32),
        scattering=np.clip(scattering, 0.0, 0.24).astype(np.float32),
        # gamma：略高，体现抛光石材和相机响应的非线性。
        projector_gamma=1.10,
        camera_gamma=1.05,
        description="Veined stone with contrast texture, mild specular response, and local scattering.",
    )
