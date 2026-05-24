"""Correspondence evaluation metrics."""

from __future__ import annotations

import numpy as np


def threshold_accuracy(predicted: np.ndarray, ground_truth: np.ndarray, valid_mask: np.ndarray, threshold: float) -> float:
    error = np.abs(predicted - ground_truth)
    return float((error[valid_mask] <= threshold).mean())


def error_map(predicted: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    return np.abs(predicted - ground_truth).astype(np.float32)
