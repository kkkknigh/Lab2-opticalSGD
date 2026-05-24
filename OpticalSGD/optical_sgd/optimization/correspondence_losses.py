"""Losses for projector-column correspondence."""

from __future__ import annotations

import numpy as np


def correspondence_mae(predicted: np.ndarray, ground_truth: np.ndarray, valid_mask: np.ndarray) -> float:
    error = np.abs(predicted - ground_truth)
    return float(error[valid_mask].mean())


def soft_expected_l1_loss(
    scores: np.ndarray,
    ground_truth: np.ndarray,
    valid_mask: np.ndarray,
    temperature: float = 25.0,
) -> float:
    logits = scores * float(temperature)
    logits = logits - logits.max(axis=-1, keepdims=True)
    weights = np.exp(logits)
    weights = weights / np.maximum(weights.sum(axis=-1, keepdims=True), 1e-8)
    columns = np.arange(scores.shape[-1], dtype=np.float32)
    penalty = np.abs(columns[None, None, :] - ground_truth[:, :, None])
    loss_map = (weights * penalty).sum(axis=-1)
    return float(loss_map[valid_mask].mean())
