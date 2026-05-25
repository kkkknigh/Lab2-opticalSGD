"""初始投影图案生成"""

from __future__ import annotations

import numpy as np


def create_initial_patterns(
    count: int,
    projector_width: int,
    method: str = "random",
    seed: int = 7,
) -> np.ndarray:
    """生成优化开始前使用的一组一维投影图案。

    Args:
        count: 投影图案数量。
        projector_width: 每张图案的投影仪横向分辨率。
        method: 初始图案类型，支持 `constant`、`stripes` 和 `random`。
        seed: 随机种子，仅 `random` 方法使用。

    Returns:
        np.ndarray: 形状为 (count, projector_width)，取值范围为 [0, 1]。
    """

    rng = np.random.default_rng(seed)
    if method == "constant":
        # 常量图案用于渲染器自检，所有投影列亮度相同。
        values = np.full((count, projector_width), 0.5, dtype=np.float32)
    elif method == "stripes":
        # 条纹图案用于检查投影方向和相机-投影仪对应关系是否合理。
        x = np.linspace(0.0, 2.0 * np.pi, projector_width, dtype=np.float32)
        values = np.stack([0.5 + 0.45 * np.sin(x * (i + 1)) for i in range(count)])
    else:
        # 随机图案作为默认优化初值，围绕中等亮度做小幅扰动。
        values = rng.uniform(0.45, 0.55, (count, projector_width)).astype(np.float32)
    return np.clip(values, 0.0, 1.0).astype(np.float32)
