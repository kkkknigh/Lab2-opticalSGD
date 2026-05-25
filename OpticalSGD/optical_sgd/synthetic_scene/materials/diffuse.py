"""漫反射材料。

基准材料，整体亮度稳定、纹理很弱、没有镜面高光，几乎不产生次表面散射或间接光混合。
主要用于观察结构光算法在简单表面上的基本表现。
"""

from __future__ import annotations

import numpy as np

from optical_sgd.synthetic_scene.materials.base import MaterialMaps, constant_map, normalized_grid


def make_diffuse(height: int, camera_width: int) -> MaterialMaps:
    _, xx = normalized_grid(height, camera_width)
    # 反照：整体较高，有很轻的横向亮度起伏。
    albedo = 0.78 + 0.04 * np.sin(4.0 * np.pi * xx)
    return MaterialMaps(
        name="diffuse",
        albedo=np.clip(albedo, 0.05, 1.0).astype(np.float32),
        # 镜面：不产生高光，0
        specular=constant_map(height, camera_width, 0.0),
        # 散射：接近 0
        scattering=constant_map(height, camera_width, 0.01),
        # gamma：投影仪线性响应
        projector_gamma=1.0,
        camera_gamma=1.0,
        description="Lambertian baseline with mild sinusoidal albedo.",
    )
