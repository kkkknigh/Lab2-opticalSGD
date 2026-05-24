"""Feature extraction for ZNCC matching."""

from __future__ import annotations

import numpy as np


def normalize_features(features: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    centered = features - features.mean(axis=-1, keepdims=True)
    norm = np.linalg.norm(centered, axis=-1, keepdims=True)
    return centered / np.maximum(norm, eps)


def projector_neighborhood_features(patterns: np.ndarray, radius: int) -> np.ndarray:
    patterns = np.asarray(patterns, dtype=np.float32)
    count, width = patterns.shape
    padded = np.pad(patterns, ((0, 0), (radius, radius)), mode="edge")
    features = []
    for col in range(width):
        patch = padded[:, col : col + 2 * radius + 1]
        features.append(patch.reshape(count * (2 * radius + 1)))
    return np.stack(features, axis=0).astype(np.float32)


def camera_neighborhood_features(images: np.ndarray, radius: int) -> np.ndarray:
    images = np.asarray(images, dtype=np.float32)
    count, height, width = images.shape
    padded = np.pad(images, ((0, 0), (0, 0), (radius, radius)), mode="edge")
    features = []
    for col in range(width):
        patch = padded[:, :, col : col + 2 * radius + 1]
        features.append(np.moveaxis(patch, 0, -1).reshape(height, count * (2 * radius + 1)))
    return np.stack(features, axis=1).astype(np.float32)
