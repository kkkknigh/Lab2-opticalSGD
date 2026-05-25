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
class TorchFeatureTransformProtocol(Protocol):
    """decoder 的可微 Torch 特征变换接口。"""

    def transform_torch_features(self, image_features, projector_features, device, trainable_parameters=None):
        """对 Torch 版 image/projector 特征应用 decoder 特定的可微变换。"""


@runtime_checkable
class AutogradTrainableDecoderProtocol(Protocol):
    """支持 autograd 参数更新的 decoder 接口。"""

    def torch_parameter_tensors(self, feature_dim: int, device):
        """返回带 requires_grad 的 Torch 参数张量。"""

    def apply_torch_parameter_update(self, named_parameters, learning_rate: float) -> float:
        """把 Torch 参数梯度更新回 decoder 内部 NumPy 参数，并返回梯度范数。"""
