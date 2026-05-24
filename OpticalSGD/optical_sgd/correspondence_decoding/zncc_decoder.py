"""Standard ZNCC decoder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optical_sgd.correspondence_decoding.feature_extraction import (
    camera_neighborhood_features,
    normalize_features,
    projector_neighborhood_features,
)


@dataclass(frozen=True)
class DecoderOutput:
    scores: np.ndarray
    predicted_correspondence: np.ndarray


@dataclass
class ZNCCDecoder:
    neighborhood: int = 1

    @property
    def feature_radius(self) -> int:
        return max(0, int(self.neighborhood) // 2)

    def decode(self, captured_images: np.ndarray, patterns: np.ndarray) -> DecoderOutput:
        radius = self.feature_radius
        if radius == 0:
            image_features = np.moveaxis(captured_images, 0, -1)
            projector_features = patterns.T
        else:
            image_features = camera_neighborhood_features(captured_images, radius)
            projector_features = projector_neighborhood_features(patterns, radius)
        image_norm = normalize_features(image_features)
        projector_norm = normalize_features(projector_features)
        scores = image_norm @ projector_norm.T
        predicted = np.argmax(scores, axis=-1).astype(np.float32)
        return DecoderOutput(scores=scores.astype(np.float32), predicted_correspondence=predicted)
