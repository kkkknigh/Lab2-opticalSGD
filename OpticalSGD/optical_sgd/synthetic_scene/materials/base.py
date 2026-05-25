"""材质二维贴图基类。

参数是对结构光成像中关键光学现象的简化建模：
- albedo 控制表面反射强度
- specular 表示视角相关的镜面高光
- scattering 表示相邻投影列之间的混合
- gamma 用来模拟投影仪和相机的非线性亮度响应。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MaterialMaps:
    name: str
    albedo: np.ndarray
    specular: np.ndarray
    scattering: np.ndarray
    projector_gamma: float
    camera_gamma: float
    description: str


def normalized_grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """生成归一化到 [0, 1] 的 yy/xx 坐标网格。"""
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, height, dtype=np.float32),
        np.linspace(0.0, 1.0, width, dtype=np.float32),
        indexing="ij",
    )
    return yy, xx


def constant_map(height: int, width: int, value: float) -> np.ndarray:
    """生成指定参数值的二维材质贴图。"""
    return np.full((height, width), float(value), dtype=np.float32)

