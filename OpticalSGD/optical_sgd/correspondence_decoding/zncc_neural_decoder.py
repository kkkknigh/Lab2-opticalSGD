"""带可学习特征变换的 ZNCC-NN decoder。

该 decoder 在标准 ZNCC 匹配前增加三个可学习模块：
- projector 响应曲线：用 32 个控制点近似投影仪/相机亮度响应。
- camera residual MLP：修正 camera 像素局部特征。
- projector residual MLP：修正 projector 列局部特征。

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
    """两层 ReLU MLP，用作局部特征的 residual 修正。"""

    hidden = np.maximum(0.0, x @ w1 + b1)
    return hidden @ w2 + b2


def _piecewise_response(values, curve):
    """对输入亮度应用一维分段线性响应曲线。"""

    values = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    positions = values * (curve.size - 1)
    left = np.floor(positions).astype(np.int64)
    right = np.clip(left + 1, 0, curve.size - 1)
    frac = (positions - left).astype(np.float32)
    return curve[left] * (1.0 - frac) + curve[right] * frac


class ZNCCNeuralDecoder:
    """在 ZNCC 特征前加入可学习响应曲线和 residual MLP 的 decoder。"""

    def __init__(self, neighborhood: int = 3, seed: int = 7):
        self.neighborhood = max(1, int(neighborhood))
        self.seed = int(seed)
        self.residual_scale = 0.08
        self._response_bins = 32
        self._mlp_hidden = 16
        self._initialized = False
        self._feature_dim: int | None = None

    @property
    def feature_radius(self) -> int:
        """返回横向局部特征半径。"""

        return max(0, self.neighborhood // 2)

    def _ensure_parameters(self, feature_dim: int) -> None:
        """按特征维度惰性初始化可学习参数。"""

        if self._initialized:
            if self._feature_dim != int(feature_dim):
                raise ValueError(
                    f"ZNCCNeuralDecoder was initialized for feature_dim={self._feature_dim}, "
                    f"but got feature_dim={int(feature_dim)}."
                )
            return
        self._initialized = True
        self._feature_dim = int(feature_dim)
        rng = np.random.default_rng(self.seed)
        # 响应曲线从接近恒等映射开始，少量噪声打破完全对称。
        self._response_curve = np.linspace(0.0, 1.0, self._response_bins, dtype=np.float32)
        self._response_curve += rng.normal(0.0, 0.008, self._response_bins).astype(np.float32)
        self._response_curve = np.clip(self._response_curve, 0.0, 1.0)

        d = feature_dim
        h = self._mlp_hidden
        # residual MLP 初始权重较小，避免一开始完全破坏 ZNCC 基线特征。
        self._camera_w1 = (rng.normal(0.0, 0.15, (d, h)) / np.sqrt(d)).astype(np.float32)
        self._camera_b1 = np.zeros(h, dtype=np.float32)
        self._camera_w2 = (rng.normal(0.0, 0.15, (h, d)) / np.sqrt(h)).astype(np.float32)
        self._camera_b2 = np.zeros(d, dtype=np.float32)
        self._projector_w1 = (rng.normal(0.0, 0.15, (d, h)) / np.sqrt(d)).astype(np.float32)
        self._projector_b1 = np.zeros(h, dtype=np.float32)
        self._projector_w2 = (rng.normal(0.0, 0.15, (h, d)) / np.sqrt(h)).astype(np.float32)
        self._projector_b2 = np.zeros(d, dtype=np.float32)

    def decode(self, captured_images: np.ndarray, patterns: np.ndarray) -> DecoderOutput:
        """解码 camera pixel 到 projector 列的对应关系。

        Args:
            captured_images: 相机观测图像，(pattern_count, height, camera_width)。
            patterns: 投影图案，(pattern_count, projector_width)。

        Returns:
            DecoderOutput: 匹配分数体和预测 projector 列坐标。
        """

        radius = self.feature_radius
        image_features = camera_neighborhood_features(captured_images, radius)
        projector_features = projector_neighborhood_features(patterns, radius)
        feature_dim = projector_features.shape[-1]
        self._ensure_parameters(feature_dim)

        # projector 特征先经过亮度响应曲线，再进入 residual MLP。
        projector_response = _piecewise_response(projector_features, self._response_curve)
        image_residual = _mlp_relu(
            image_features, self._camera_w1, self._camera_b1, self._camera_w2, self._camera_b2
        )
        projector_residual = _mlp_relu(
            projector_response, self._projector_w1, self._projector_b1, self._projector_w2, self._projector_b2
        )
        transformed_image = image_features + self.residual_scale * image_residual
        transformed_projector = projector_response + self.residual_scale * projector_residual

        # 后续匹配仍然是标准 ZNCC：归一化特征内积 + argmax。
        image_norm = normalize_features(transformed_image)
        projector_norm = normalize_features(transformed_projector)
        scores = image_norm @ projector_norm.T
        predicted = np.argmax(scores, axis=-1).astype(np.float32)
        return DecoderOutput(scores=scores.astype(np.float32), predicted_correspondence=predicted)

    def transform_torch_features(self, image_features, projector_features, device, trainable_parameters=None):
        """Torch 版可微特征变换，用于 autograd 优化路径。

        Args:
            image_features: Torch 版 camera 特征。
            projector_features: Torch 版 projector 特征。
            device: 参数所在的 Torch 设备。
            trainable_parameters: 可选的 `(name, tensor)` 参数列表。传入时这些
                tensor 会保留 `requires_grad`，用于真正更新 decoder 参数。
        """

        import torch

        self._ensure_parameters(projector_features.shape[-1])
        parameters = dict(trainable_parameters or [])
        response_curve = parameters.get(
            "_response_curve",
            torch.as_tensor(self._response_curve, dtype=torch.float32, device=device),
        )
        projector_response = self._torch_piecewise_response(projector_features, response_curve)

        def tensor(name: str):
            return parameters.get(name, torch.as_tensor(getattr(self, name), dtype=torch.float32, device=device))

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

    def torch_parameter_tensors(self, feature_dim: int, device):
        """创建参与 autograd 的 decoder 参数张量。"""

        import torch

        self._ensure_parameters(feature_dim)
        names = [
            "_response_curve",
            "_camera_w1",
            "_camera_b1",
            "_camera_w2",
            "_camera_b2",
            "_projector_w1",
            "_projector_b1",
            "_projector_w2",
            "_projector_b2",
        ]
        return [
            (
                name,
                torch.tensor(getattr(self, name), dtype=torch.float32, device=device, requires_grad=True),
            )
            for name in names
        ]

    def apply_torch_parameter_update(self, named_parameters, learning_rate: float) -> float:
        """用 autograd 梯度更新 decoder 参数，并返回参数梯度范数。"""

        squared_norm = 0.0
        for name, tensor in named_parameters:
            if tensor.grad is None:
                continue
            gradient = tensor.grad.detach()
            squared_norm += float((gradient * gradient).sum().cpu())
            updated = tensor.detach() - float(learning_rate) * gradient
            array = updated.cpu().numpy().astype(np.float32)
            if name == "_response_curve":
                array = np.clip(array, 0.0, 1.0)
            setattr(self, name, array)
        return float(np.sqrt(squared_norm))

    def parameter_array(self) -> np.ndarray:
        """返回当前 decoder 参数数组，仅用于测试和结果检查。"""

        if not self._initialized:
            raise RuntimeError("Call decode() or transform_torch_features() before reading decoder parameters.")
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

    @staticmethod
    def _torch_piecewise_response(values, response_curve):
        """Torch 版分段线性响应曲线。"""

        import torch

        values = torch.clamp(values, 0.0, 1.0)
        positions = values * float(response_curve.numel() - 1)
        left = torch.floor(positions).long()
        right = torch.clamp(left + 1, 0, response_curve.numel() - 1)
        fraction = positions - left.to(values.dtype)
        return response_curve[left] * (1.0 - fraction) + response_curve[right] * fraction

    @staticmethod
    def _torch_mlp(values, w1, b1, w2, b2):
        """Torch 版两层 ReLU MLP。"""

        import torch

        hidden = torch.relu(values @ w1 + b1)
        return hidden @ w2 + b2
