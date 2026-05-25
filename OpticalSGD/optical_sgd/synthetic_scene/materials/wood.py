"""木材材料。

以年轮和顺纹方向纹理为主要特征，反照率存在明显的低频环状变化，
并叠加细密的横向木纹。整体以漫反射为主，只有弱镜面反射和少量散射。
"""

from __future__ import annotations

import numpy as np

from optical_sgd.synthetic_scene.materials.base import MaterialMaps, normalized_grid


def make_wood(height: int, camera_width: int) -> MaterialMaps:
    yy, xx = normalized_grid(height, camera_width)
    # 年轮：偏心圆环形成木材最明显的低频明暗变化。
    rings = np.sin(38.0 * np.sqrt((xx - 0.18) ** 2 + (yy - 0.42) ** 2))
    # 木纹：沿 x 方向拉长的细纹，幅度低于年轮但频率更高。
    grain = 0.07 * np.sin(95.0 * xx + 5.0 * np.sin(13.0 * yy))
    # 反照率：中等偏暗，叠加年轮、木纹和轻微横向渐变。
    albedo = 0.46 + 0.24 * rings + 0.12 * xx + grain
    # 镜面项：木材表面有弱高光，沿纹理方向缓慢变化。
    specular = 0.025 + 0.055 * np.maximum(0.0, np.sin(8.0 * xx))
    # 散射项：比漫反射略高，模拟木质纤维带来的轻微光扩散。
    scattering = 0.035 + 0.04 * np.abs(np.sin(16.0 * yy))
    return MaterialMaps(
        name="wood",
        albedo=np.clip(albedo, 0.05, 1.0).astype(np.float32),
        specular=np.clip(specular, 0.0, 0.18).astype(np.float32),
        scattering=np.clip(scattering, 0.0, 0.14).astype(np.float32),
        # gamma：加入很弱的非线性响应。
        projector_gamma=1.04,
        camera_gamma=1.02,
        description="Ring and grain texture with weak anisotropic shine.",
    )
