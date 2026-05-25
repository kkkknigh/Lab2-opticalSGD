"""correspondence 解码结果评估指标。"""

from __future__ import annotations

import numpy as np


def threshold_accuracy(predicted: np.ndarray, ground_truth: np.ndarray, valid_mask: np.ndarray, threshold: float) -> float:
    """计算有效像素中误差不超过阈值的比例。

    Args:
        predicted: decoder 预测的 projector 列坐标。
        ground_truth: 合成场景提供的 projector 列真值。
        valid_mask: 有效像素掩码。
        threshold: 判定为正确的最大绝对列误差。

    Returns:
        float: 有效像素上的阈值准确率。
    """

    error = np.abs(predicted - ground_truth)
    return float((error[valid_mask] <= threshold).mean())


def error_map(predicted: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    """生成逐像素绝对 correspondence 误差图。"""

    return np.abs(predicted - ground_truth).astype(np.float32)
