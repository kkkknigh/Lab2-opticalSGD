"""ZNCC decoder with learnable neural transforms.

This decoder wraps standard ZNCC matching with three learnable components:
- A 32-bin piecewise-linear projector response curve g()
- A camera residual MLP that refines per-pixel intensity features
- A projector residual MLP that refines column features after response mapping

The combined transforms are diag([g, MLP_cam, MLP_proj]) so the ZNCC
inner product is still a valid similarity score on the transformed features.
When parameter_vector() / set_parameter_vector() are used, the OpticalSGD
optimizer jointly updates patterns and decoder parameters.
"""

from __future__ import annotations

import numpy as np

from optical_sgd.correspondence_decoding.feature_extraction import (
    camera_neighborhood_features,
    normalize_features,
    projector_neighborhood_features,
)
from optical_sgd.correspondence_decoding.zncc_decoder import DecoderOutput


def _mlp_relu(x, w1, b1, w2, b2):
    hidden = np.maximum(0.0, x @ w1 + b1)
    return hidden @ w2 + b2


def _piecewise_response(values, curve):
    values = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    positions = values * (curve.size - 1)
    left = np.floor(positions).astype(np.int64)
    right = np.clip(left + 1, 0, curve.size - 1)
    frac = (positions - left).astype(np.float32)
    return curve[left] * (1.0 - frac) + curve[right] * frac


class ZNCCNeuralDecoder:
    def __init__(self, neighborhood: int = 3, seed: int = 7):
        self.neighborhood = max(1, int(neighborhood))
        self.seed = int(seed)
        self.residual_scale = 0.08
        self._response_bins = 32
        self._mlp_hidden = 16
        self._initialized = False

    @property
    def feature_radius(self) -> int:
        return max(0, self.neighborhood // 2)

    def _ensure_parameters(self, feature_dim: int) -> None:
        if self._initialized:
            return
        self._initialized = True
        rng = np.random.default_rng(self.seed)
        self._response_curve = np.linspace(0.0, 1.0, self._response_bins, dtype=np.float32)
        self._response_curve += rng.normal(0.0, 0.008, self._response_bins).astype(np.float32)
        self._response_curve = np.clip(self._response_curve, 0.0, 1.0)

        d = feature_dim
        h = self._mlp_hidden
        self._camera_w1 = (rng.normal(0.0, 0.15, (d, h)) / np.sqrt(d)).astype(np.float32)
        self._camera_b1 = np.zeros(h, dtype=np.float32)
        self._camera_w2 = (rng.normal(0.0, 0.15, (h, d)) / np.sqrt(h)).astype(np.float32)
        self._camera_b2 = np.zeros(d, dtype=np.float32)
        self._projector_w1 = (rng.normal(0.0, 0.15, (d, h)) / np.sqrt(d)).astype(np.float32)
        self._projector_b1 = np.zeros(h, dtype=np.float32)
        self._projector_w2 = (rng.normal(0.0, 0.15, (h, d)) / np.sqrt(h)).astype(np.float32)
        self._projector_b2 = np.zeros(d, dtype=np.float32)

    def decode(self, captured_images: np.ndarray, patterns: np.ndarray) -> DecoderOutput:
        radius = self.feature_radius
        image_features = camera_neighborhood_features(captured_images, radius)
        projector_features = projector_neighborhood_features(patterns, radius)
        feature_dim = projector_features.shape[-1]
        self._ensure_parameters(feature_dim)

        projector_response = _piecewise_response(projector_features, self._response_curve)
        image_residual = _mlp_relu(
            image_features, self._camera_w1, self._camera_b1, self._camera_w2, self._camera_b2
        )
        projector_residual = _mlp_relu(
            projector_response, self._projector_w1, self._projector_b1, self._projector_w2, self._projector_b2
        )
        transformed_image = image_features + self.residual_scale * image_residual
        transformed_projector = projector_response + self.residual_scale * projector_residual

        image_norm = normalize_features(transformed_image)
        projector_norm = normalize_features(transformed_projector)
        scores = image_norm @ projector_norm.T
        predicted = np.argmax(scores, axis=-1).astype(np.float32)
        return DecoderOutput(scores=scores.astype(np.float32), predicted_correspondence=predicted)

    def transform_torch_features(self, image_features, projector_features, device):
        import torch

        self._ensure_parameters(projector_features.shape[-1])
        response_curve = torch.as_tensor(self._response_curve, dtype=torch.float32, device=device)
        projector_response = self._torch_piecewise_response(projector_features, response_curve)

        def tensor(name: str):
            return torch.as_tensor(getattr(self, name), dtype=torch.float32, device=device)

        image_residual = self._torch_mlp(
            image_features,
            tensor("_camera_w1"),
            tensor("_camera_b1"),
            tensor("_camera_w2"),
            tensor("_camera_b2"),
        )
        projector_residual = self._torch_mlp(
            projector_response,
            tensor("_projector_w1"),
            tensor("_projector_b1"),
            tensor("_projector_w2"),
            tensor("_projector_b2"),
        )
        return (
            image_features + self.residual_scale * image_residual,
            projector_response + self.residual_scale * projector_residual,
        )

    @staticmethod
    def _torch_piecewise_response(values, response_curve):
        import torch

        values = torch.clamp(values, 0.0, 1.0)
        positions = values * float(response_curve.numel() - 1)
        left = torch.floor(positions).long()
        right = torch.clamp(left + 1, 0, response_curve.numel() - 1)
        fraction = positions - left.to(values.dtype)
        return response_curve[left] * (1.0 - fraction) + response_curve[right] * fraction

    @staticmethod
    def _torch_mlp(values, w1, b1, w2, b2):
        import torch

        hidden = torch.relu(values @ w1 + b1)
        return hidden @ w2 + b2

    def parameter_vector(self) -> np.ndarray:
        self._ensure_parameters(4)
        return np.concatenate(
            [
                self._response_curve.ravel(),
                self._camera_w1.ravel(),
                self._camera_b1.ravel(),
                self._camera_w2.ravel(),
                self._camera_b2.ravel(),
                self._projector_w1.ravel(),
                self._projector_b1.ravel(),
                self._projector_w2.ravel(),
                self._projector_b2.ravel(),
            ]
        ).astype(np.float32)

    def set_parameter_vector(self, vector: np.ndarray) -> None:
        self._ensure_parameters(4)
        v = np.asarray(vector, dtype=np.float32)
        offset = 0

        def take(shape):
            nonlocal offset
            size = int(np.prod(shape))
            chunk = v[offset : offset + size].reshape(shape)
            offset += size
            return chunk

        self._response_curve = take(self._response_curve.shape)
        self._camera_w1 = take(self._camera_w1.shape)
        self._camera_b1 = take(self._camera_b1.shape)
        self._camera_w2 = take(self._camera_w2.shape)
        self._camera_b2 = take(self._camera_b2.shape)
        self._projector_w1 = take(self._projector_w1.shape)
        self._projector_b1 = take(self._projector_b1.shape)
        self._projector_w2 = take(self._projector_w2.shape)
        self._projector_b2 = take(self._projector_b2.shape)
