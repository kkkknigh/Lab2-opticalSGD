"""图案梯度估计器

包含两种基础梯度估计方式：
- 有限差分：只要求 loss_function 能接收 NumPy pattern 并返回标量 loss。
- autograd：基于 PyTorch， 可微 loss 直接反传到 pattern。
"""

from __future__ import annotations

import numpy as np


class FiniteDifferenceGradientEstimator:
    """使用中心差分估计 loss 对 pattern 的梯度。"""

    def __init__(self, epsilon: float = 0.03):
        """保存扰动步长。

        Args:
            epsilon: 每个 pattern 元素的正负扰动幅度。
        """

        self.epsilon = float(epsilon)

    def estimate(self, patterns: np.ndarray, loss_function) -> np.ndarray:
        """逐元素扰动 pattern，计算中心差分梯度。

        Args:
            patterns: 当前投影图案，形状为 (pattern_count, projector_width)。
            loss_function: 接收候选 pattern 并返回标量 loss 的函数。

        Returns:
            np.ndarray: 与 patterns 同形状的梯度数组。
        """

        gradient = np.zeros_like(patterns, dtype=np.float32)
        for index in np.ndindex(patterns.shape):
            plus = np.array(patterns, copy=True)
            minus = np.array(patterns, copy=True)
            plus[index] += self.epsilon
            minus[index] -= self.epsilon
            gradient[index] = (loss_function(plus) - loss_function(minus)) / (2.0 * self.epsilon)
        return gradient


class AutogradGradientEstimator:
    """使用 PyTorch autograd 计算 loss 对 pattern 的梯度。"""

    def __init__(self, epsilon: float = 0.02):
        """保存与有限差分接口一致的步长字段。"""

        self.epsilon = float(epsilon)

    def estimate(self, patterns: np.ndarray, loss_function, differentiable_loss_function=None) -> np.ndarray:
        """通过可微 loss 函数反传得到 pattern 梯度。

        Args:
            patterns: 当前投影图案，形状为 (pattern_count, projector_width)。
            loss_function: 占位参数，不使用。
            differentiable_loss_function: 接收 torch.Tensor pattern 并返回 torch 标量 loss 的函数。

        Returns:
            np.ndarray: 与 patterns 同形状的梯度数组。
        """

        try:
            import torch
        except Exception as exc:
            raise ImportError("AutogradGradientEstimator requires PyTorch.") from exc
        if differentiable_loss_function is None:
            raise ValueError("AutogradGradientEstimator requires a differentiable loss function.")

        pattern_tensor = torch.tensor(patterns, dtype=torch.float32, requires_grad=True)
        loss = differentiable_loss_function(pattern_tensor)
        loss.backward()
        if pattern_tensor.grad is None:
            raise RuntimeError("Autograd did not produce a pattern gradient.")
        return pattern_tensor.grad.detach().cpu().numpy().astype(np.float32)
