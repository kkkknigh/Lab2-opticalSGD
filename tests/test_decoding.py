from __future__ import annotations

import numpy as np

from optical_sgd.correspondence_decoding.feature_extraction import (
    camera_neighborhood_features,
    normalize_features,
    projector_neighborhood_features,
)
from optical_sgd.correspondence_decoding.decoder_protocol import (
    TorchFeatureTransformProtocol,
    TrainableDecoderProtocol,
)
from optical_sgd.correspondence_decoding.zncc_decoder import ZNCCDecoder
from optical_sgd.correspondence_decoding.zncc_neural_decoder import ZNCCNeuralDecoder


def test_normalize_features_centers_last_axis_and_handles_constant_vectors():
    features = np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]], dtype=np.float32)

    normalized = normalize_features(features)

    assert np.allclose(normalized[0].mean(), 0.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(normalized[0]), 1.0)
    assert np.allclose(normalized[1], 0.0)


def test_neighborhood_feature_shapes():
    patterns = np.arange(8, dtype=np.float32).reshape(2, 4)
    images = np.arange(16, dtype=np.float32).reshape(2, 2, 4)

    projector_features = projector_neighborhood_features(patterns, radius=1)
    camera_features = camera_neighborhood_features(images, radius=1)

    assert projector_features.shape == (4, 6)
    assert camera_features.shape == (2, 4, 6)


def test_zncc_decoder_recovers_known_projector_columns():
    patterns = np.array(
        [
            [0.0, 0.2, 0.8, 1.0],
            [1.0, 0.8, 0.2, 0.0],
            [0.1, 0.9, 0.4, 0.7],
        ],
        dtype=np.float32,
    )
    captured_images = patterns[:, None, :]

    decoded = ZNCCDecoder().decode(captured_images, patterns)

    assert decoded.scores.shape == (1, 4, 4)
    assert np.array_equal(decoded.predicted_correspondence, np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32))


def test_decoders_expose_protocol_capabilities_without_class_name_checks():
    assert ZNCCDecoder().feature_radius == 0
    assert ZNCCDecoder(neighborhood=3).feature_radius == 1

    neural = ZNCCNeuralDecoder(neighborhood=3, seed=1)
    assert neural.feature_radius == 1
    assert isinstance(neural, TrainableDecoderProtocol)
    assert isinstance(neural, TorchFeatureTransformProtocol)
