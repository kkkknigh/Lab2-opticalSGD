"""Decoder protocols used by optimization code."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from optical_sgd.correspondence_decoding.zncc_decoder import DecoderOutput


class DecoderProtocol(Protocol):
    @property
    def feature_radius(self) -> int:
        """Horizontal neighborhood radius used to build matching features."""

    def decode(self, captured_images: np.ndarray, patterns: np.ndarray) -> DecoderOutput:
        """Decode projector-column correspondences from captured images."""


@runtime_checkable
class TrainableDecoderProtocol(Protocol):
    def parameter_vector(self) -> np.ndarray:
        """Return decoder parameters as one flat vector."""

    def set_parameter_vector(self, vector: np.ndarray) -> None:
        """Restore decoder parameters from one flat vector."""


@runtime_checkable
class TorchFeatureTransformProtocol(Protocol):
    def transform_torch_features(self, image_features, projector_features, device):
        """Apply differentiable decoder-specific feature transforms."""
