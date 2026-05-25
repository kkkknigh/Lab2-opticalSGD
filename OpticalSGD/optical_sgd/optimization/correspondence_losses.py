"""投影列对应关系的训练 loss 和评估误差。"""

from __future__ import annotations

import numpy as np


def correspondence_mae(predicted: np.ndarray, ground_truth: np.ndarray, valid_mask: np.ndarray) -> float:
    """计算有效像素上的平均绝对对应误差。

    Args:
        predicted: decoder 预测的 projector 列坐标，(height, camera_width)。
        ground_truth: 合成场景提供的 projector 列真值。
        valid_mask: 标记哪些像素的对应关系真值有效。

    Returns:
        float: 有效像素上的平均绝对误差。
    """

    error = np.abs(predicted - ground_truth)
    return float(error[valid_mask].mean())


def soft_expected_l1_loss(
    scores: np.ndarray,
    ground_truth: np.ndarray,
    valid_mask: np.ndarray,
    temperature: float = 25.0,
) -> float:
    """用 soft matching 分布计算期望 L1 correspondence loss。

    `scores[y, x, k]` 表示相机像素 (y, x) 匹配到投影列 k 的相似度。
    先通过 softmax 得到列概率，再计算该概率分布到真值列的期望 L1 距离。

    Args:
        scores: decoder 输出的匹配分数体，(height, camera_width, projector_width)。
        ground_truth: projector 列真值，(height, camera_width)。
        valid_mask: 有效像素掩码。
        temperature: softmax 温度系数

    Returns:
        float: 有效像素上的期望 L1 loss。
    """

    logits = scores * float(temperature)
    # 减去最大值提高 softmax 的数值稳定性。
    logits = logits - logits.max(axis=-1, keepdims=True)
    weights = np.exp(logits)
    weights = weights / np.maximum(weights.sum(axis=-1, keepdims=True), 1e-8)
    columns = np.arange(scores.shape[-1], dtype=np.float32)
    penalty = np.abs(columns[None, None, :] - ground_truth[:, :, None])
    loss_map = (weights * penalty).sum(axis=-1)
    return float(loss_map[valid_mask].mean())
