from __future__ import annotations

import numpy as np

from optical_sgd.evaluation.correspondence_metrics import error_map, threshold_accuracy
from optical_sgd.evaluation.gradient_metrics import cosine_similarity
from optical_sgd.optimization.correspondence_losses import correspondence_mae
from optical_sgd.optimization.finite_difference_gradient import FiniteDifferenceGradientEstimator


def test_correspondence_metrics_respect_valid_mask_and_threshold():
    predicted = np.array([[0.0, 2.0], [5.0, 9.0]], dtype=np.float32)
    ground_truth = np.array([[0.0, 1.0], [3.0, 10.0]], dtype=np.float32)
    valid_mask = np.array([[True, True], [False, True]])

    assert threshold_accuracy(predicted, ground_truth, valid_mask, threshold=1.0) == 1.0
    assert np.isclose(correspondence_mae(predicted, ground_truth, valid_mask), np.mean([0.0, 1.0, 1.0]))
    assert np.array_equal(error_map(predicted, ground_truth), np.array([[0.0, 1.0], [2.0, 1.0]], dtype=np.float32))


def test_cosine_similarity_and_finite_difference_gradient():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 0.0
    assert np.isclose(cosine_similarity(np.array([1.0, 1.0]), np.array([2.0, 2.0])), 1.0)

    patterns = np.array([[1.0, -2.0]], dtype=np.float32)
    estimator = FiniteDifferenceGradientEstimator(epsilon=1e-3)

    gradient = estimator.estimate(patterns, lambda candidate: float((candidate**2).sum()))

    assert np.allclose(gradient, 2.0 * patterns, atol=1e-2)
