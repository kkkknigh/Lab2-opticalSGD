"""PyTorch autograd gradient estimator."""

from __future__ import annotations

import numpy as np


class AutogradGradientEstimator:
    def __init__(self, epsilon: float = 0.02):
        self.epsilon = float(epsilon)

    def estimate(self, patterns: np.ndarray, loss_function, differentiable_loss_function=None) -> np.ndarray:
        """Return d(loss)/d(patterns) using PyTorch autograd."""

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
