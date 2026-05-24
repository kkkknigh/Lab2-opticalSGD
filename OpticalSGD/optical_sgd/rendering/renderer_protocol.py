"""渲染器接口协议"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from optical_sgd.rendering.render_result import RenderResult
from optical_sgd.synthetic_scene.scene_description import SceneDescription


class RendererProtocol(Protocol):
    def render(self, patterns: np.ndarray, scene: SceneDescription) -> RenderResult:
        """把投影图案渲染成相机观测图像。"""


@runtime_checkable
class DifferentiableRendererProtocol(RendererProtocol, Protocol):
    def render_torch(self, patterns, scene: SceneDescription) -> dict:
        """使用可微渲染，可以通过 autograd 反传梯度。"""
