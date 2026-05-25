"""评估指标和基础梯度估计器单元测试。

覆盖 correspondence 阈值准确率、MAE、误差图、梯度余弦相似度、运行时间记录，
以及有限差分估计器在二次函数上的输入输出。
"""

from __future__ import annotations

import numpy as np
import pytest

from optical_sgd.evaluation.correspondence_metrics import error_map, threshold_accuracy
from optical_sgd.evaluation.gradient_metrics import cosine_similarity
from optical_sgd.evaluation.runtime_metrics import measure_seconds
from optical_sgd.optimization.correspondence_losses import correspondence_mae
from optical_sgd.correspondence_decoding.zncc_neural_decoder import ZNCCNeuralDecoder
from optical_sgd.optimization.gradient_estimators import FiniteDifferenceGradientEstimator
from optical_sgd.optimization.optical_sgd_optimizer import OpticalSGDOptimizer
from optical_sgd.rendering.torch_renderer import TorchRenderer
from optical_sgd.synthetic_scene.scene import create_scene


def test_threshold_accuracy_counts_only_valid_pixels():
    predicted = np.array([[0.0, 2.0], [5.0, 9.0]], dtype=np.float32)
    ground_truth = np.array([[0.0, 1.0], [3.0, 10.0]], dtype=np.float32)
    valid_mask = np.array([[True, True], [False, True]])

    assert threshold_accuracy(predicted, ground_truth, valid_mask, threshold=1.0) == 1.0


def test_correspondence_mae_counts_only_valid_pixels():
    predicted = np.array([[0.0, 2.0], [5.0, 9.0]], dtype=np.float32)
    ground_truth = np.array([[0.0, 1.0], [3.0, 10.0]], dtype=np.float32)
    valid_mask = np.array([[True, True], [False, True]])

    assert np.isclose(correspondence_mae(predicted, ground_truth, valid_mask), np.mean([0.0, 1.0, 1.0]))


def test_error_map_returns_absolute_float32_errors():
    predicted = np.array([[0.0, 2.0], [5.0, 9.0]], dtype=np.float32)
    ground_truth = np.array([[0.0, 1.0], [3.0, 10.0]], dtype=np.float32)

    errors = error_map(predicted, ground_truth)

    assert np.array_equal(error_map(predicted, ground_truth), np.array([[0.0, 1.0], [2.0, 1.0]], dtype=np.float32))
    assert errors.dtype == np.float32


def test_cosine_similarity_returns_zero_for_orthogonal_vectors():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 0.0


def test_cosine_similarity_returns_one_for_same_direction():
    assert np.isclose(cosine_similarity(np.array([1.0, 1.0]), np.array([2.0, 2.0])), 1.0)


def test_measure_seconds_writes_elapsed_time_to_container():
    metrics = {}

    with measure_seconds(metrics, "elapsed"):
        pass

    assert "elapsed" in metrics
    assert metrics["elapsed"] >= 0.0


def test_finite_difference_gradient_matches_quadratic_derivative():
    patterns = np.array([[1.0, -2.0]], dtype=np.float32)
    estimator = FiniteDifferenceGradientEstimator(epsilon=1e-3)

    gradient = estimator.estimate(patterns, lambda candidate: float((candidate**2).sum()))

    assert np.allclose(gradient, 2.0 * patterns, atol=1e-2)


def test_autograd_training_updates_neural_decoder_parameters_when_joint_enabled():
    pytest.importorskip("torch")
    config = {
        "renderer": {"scene_height": 2, "camera_width": 3, "projector_width": 3, "projector_baseline": 0.08},
        "scene": {"depth_profile": "flat", "material": "diffuse"},
    }
    scene = create_scene(config)
    decoder = ZNCCNeuralDecoder(neighborhood=1, seed=1)
    optimizer = OpticalSGDOptimizer(
        renderer=TorchRenderer(noise_std=0.0, ambient=0.02, device="cpu"),
        decoder=decoder,
        scene=scene,
        iterations=1,
        gradient_method="autograd",
        learning_rate=0.05,
        temperature=5.0,
        joint_optimize_decoder=True,
        decoder_learning_rate=0.01,
    )
    patterns = np.array([[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]], dtype=np.float32)
    optimizer.evaluate(patterns)
    before = decoder.parameter_array().copy()

    _, state = optimizer.train(patterns)
    after = decoder.parameter_array()

    assert state.decoder_gradient_norms[0] > 0.0
    assert not np.allclose(after, before)


def test_finite_difference_pattern_training_uses_autograd_for_neural_decoder_parameters():
    pytest.importorskip("torch")
    config = {
        "renderer": {"scene_height": 2, "camera_width": 3, "projector_width": 3, "projector_baseline": 0.08},
        "scene": {"depth_profile": "flat", "material": "diffuse"},
    }
    scene = create_scene(config)
    decoder = ZNCCNeuralDecoder(neighborhood=1, seed=1)
    optimizer = OpticalSGDOptimizer(
        renderer=TorchRenderer(noise_std=0.0, ambient=0.02, device="cpu"),
        decoder=decoder,
        scene=scene,
        iterations=1,
        gradient_method="finite_difference",
        finite_difference_epsilon=0.02,
        learning_rate=0.05,
        temperature=5.0,
        joint_optimize_decoder=True,
        decoder_learning_rate=0.01,
    )
    patterns = np.array([[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]], dtype=np.float32)
    optimizer.evaluate(patterns)
    before = decoder.parameter_array().copy()

    _, state = optimizer.train(patterns)
    after = decoder.parameter_array()

    assert state.decoder_gradient_norms[0] > 0.0
    assert not np.allclose(after, before)
