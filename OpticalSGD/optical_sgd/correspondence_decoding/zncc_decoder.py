"""标准 ZNCC correspondence decoder。"""

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
    """decoder 输出结果。"""

    # 每个 camera 像素对每个 projector 列的匹配分数。
    scores: np.ndarray

    # 对 scores 做 argmax 后得到的 projector 列坐标预测。
    predicted_correspondence: np.ndarray


@dataclass
class ZNCCDecoder:
    """基于零均值归一化互相关的 projector 列匹配器。"""

    neighborhood: int = 1

    @property
    def feature_radius(self) -> int:
        """把邻域宽度转换为左右半径。"""

        return max(0, int(self.neighborhood) // 2)

    def decode(self, captured_images: np.ndarray, patterns: np.ndarray) -> DecoderOutput:
        """从相机观测图像中解码每个像素对应的 projector 列。

        Args:
            captured_images: 渲染得到的相机图像，(pattern_count, height, camera_width)。
            patterns: 投影图案，(pattern_count, projector_width)。

        Returns:
            DecoderOutput: 匹配分数体和 argmax 得到的 projector 列坐标。
        """

        radius = self.feature_radius
        if radius == 0:
            # 无邻域时，每个像素/列的特征就是多张 pattern 下的亮度序列。
            image_features = np.moveaxis(captured_images, 0, -1)
            projector_features = patterns.T
        else:
            # 有邻域时，把横向邻域内的多张 pattern 亮度拼成一个特征向量。
            image_features = camera_neighborhood_features(captured_images, radius)
            projector_features = projector_neighborhood_features(patterns, radius)
        image_norm = normalize_features(image_features)
        projector_norm = normalize_features(projector_features)
        # ZNCC 等价于归一化特征的内积，最后一维和 projector 列特征做矩阵乘法。
        scores = image_norm @ projector_norm.T
        predicted = np.argmax(scores, axis=-1).astype(np.float32)
        return DecoderOutput(scores=scores.astype(np.float32), predicted_correspondence=predicted)
