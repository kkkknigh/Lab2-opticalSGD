"""ZNCC 匹配使用的局部特征提取工具"""

from __future__ import annotations

import numpy as np


def normalize_features(features: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """沿最后一维做零均值 L2 归一化。

    Args:
        features: 任意形状的特征数组，最后一维是特征维。
        eps: 防止除零的最小范数。

    Returns:
        np.ndarray: 与输入同形状的归一化特征。
    """

    centered = features - features.mean(axis=-1, keepdims=True)
    norm = np.linalg.norm(centered, axis=-1, keepdims=True)
    return centered / np.maximum(norm, eps)


def projector_neighborhood_features(patterns: np.ndarray, radius: int) -> np.ndarray:
    """构造每个 projector 列的局部邻域特征。

    Args:
        patterns: 投影图案数组，(pattern_count, projector_width)。
        radius: 横向邻域半径。

    Returns:
        np.ndarray: (projector_width, pattern_count * (2 * radius + 1))。
    """

    patterns = np.asarray(patterns, dtype=np.float32)
    count, width = patterns.shape
    # 边界用 edge padding，保证第 0 列和最后一列也有同样长度的邻域特征。
    padded = np.pad(patterns, ((0, 0), (radius, radius)), mode="edge")
    features = []
    for col in range(width):
        patch = padded[:, col : col + 2 * radius + 1]
        features.append(patch.reshape(count * (2 * radius + 1)))
    return np.stack(features, axis=0).astype(np.float32)


def camera_neighborhood_features(images: np.ndarray, radius: int) -> np.ndarray:
    """构造每个 camera 像素的横向局部邻域特征。

    Args:
        images: 相机观测图像，(pattern_count, height, camera_width)。
        radius: 横向邻域半径。

    Returns:
        np.ndarray: (height, camera_width, pattern_count * (2 * radius + 1))。
    """

    images = np.asarray(images, dtype=np.float32)
    count, height, width = images.shape
    padded = np.pad(images, ((0, 0), (0, 0), (radius, radius)), mode="edge")
    features = []
    for col in range(width):
        patch = padded[:, :, col : col + 2 * radius + 1]
        features.append(patch.transpose(1, 0, 2).reshape(height, count * (2 * radius + 1)))
    return np.stack(features, axis=1).astype(np.float32)
