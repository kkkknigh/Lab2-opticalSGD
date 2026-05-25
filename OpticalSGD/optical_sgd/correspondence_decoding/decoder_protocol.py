"""correspondence decoder 协议"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from optical_sgd.correspondence_decoding.zncc_decoder import DecoderOutput


class DecoderProtocol(Protocol):
    """基础 decoder 接口，用于从相机观测恢复 projector 列坐标。"""

    @property
    def feature_radius(self) -> int:
        """构造横向局部匹配特征时使用的半径。"""

    def decode(self, captured_images: np.ndarray, patterns: np.ndarray) -> DecoderOutput:
        """根据相机观测图像和投影图案估计 projector 列对应关系。"""


@runtime_checkable
class TrainableDecoderProtocol(Protocol):
    """可学习 decoder 参数读写接口。"""

    def parameter_vector(self) -> np.ndarray:
        """把 decoder 参数展平成一维向量。"""

    def set_parameter_vector(self, vector: np.ndarray) -> None:
        """从一维向量恢复 decoder 参数。"""


@runtime_checkable
class TorchFeatureTransformProtocol(Protocol):
    """decoder 的可微 Torch 特征变换接口。"""

    def transform_torch_features(self, image_features, projector_features, device):
        """对 Torch 版 image/projector 特征应用 decoder 特定的可微变换。"""
