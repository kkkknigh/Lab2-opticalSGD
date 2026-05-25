"""渲染器接口协议"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from optical_sgd.rendering.render_result import RenderResult
from optical_sgd.synthetic_scene import SceneDescription


@runtime_checkable
class RendererProtocol(Protocol):
    def render(self, patterns: np.ndarray, scene: SceneDescription) -> RenderResult:
        """黑盒渲染接口，返回 NumPy 格式的完整渲染结果。"""


@runtime_checkable
class DifferentiableRendererProtocol(Protocol):
    def render_torch(self, patterns, scene: SceneDescription) -> dict:
        """可微渲染接口，返回 torch.Tensor 结果并保留梯度路径。"""
