"""Finite-difference gradient estimation."""

from __future__ import annotations

import numpy as np


class FiniteDifferenceGradientEstimator:
    def __init__(self, epsilon: float = 0.03):
        self.epsilon = float(epsilon)

    def estimate(self, patterns: np.ndarray, loss_function) -> np.ndarray:
        gradient = np.zeros_like(patterns, dtype=np.float32)
        for index in np.ndindex(patterns.shape):
            plus = np.array(patterns, copy=True)
            minus = np.array(patterns, copy=True)
            plus[index] += self.epsilon
            minus[index] -= self.epsilon
            gradient[index] = (loss_function(plus) - loss_function(minus)) / (2.0 * self.epsilon)
        return gradient
