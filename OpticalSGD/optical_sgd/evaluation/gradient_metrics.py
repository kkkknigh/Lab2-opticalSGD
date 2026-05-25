"""梯度比较指标。"""

from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个梯度向量的余弦相似度。

    Args:
        a: 第一个梯度数组，任意形状。
        b: 第二个梯度数组，任意形状。

    Returns:
        float: 展平后两个向量的余弦相似度；若任一向量近似为零，则返回 0。
    """

    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float((a.reshape(-1) @ b.reshape(-1)) / denom)
