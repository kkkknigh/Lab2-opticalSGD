"""TorchRenderer 内部张量函数单元测试。

覆盖 projector 列线性采样、相邻列散射混合、程序化高光范围，
以及设备选择函数的基本输入输出。
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from optical_sgd.rendering.torch_renderer import TorchRenderer


def test_sample_projector_columns_torch_interpolates_float_columns():
    patterns = torch.tensor([[0.0, 10.0, 20.0, 30.0]], dtype=torch.float32)
    columns = torch.tensor([[0.0, 0.5, 1.5, 3.0, 4.0]], dtype=torch.float32)

    sampled = TorchRenderer._sample_projector_columns_torch(patterns, columns)

    assert sampled.shape == (1, 1, 5)
    assert torch.allclose(sampled[0, 0], torch.tensor([0.0, 5.0, 15.0, 30.0, 30.0]))


def test_mix_neighbor_columns_blends_original_and_blurred_values():
    sampled = torch.tensor([[[10.0, 20.0, 30.0, 40.0]]], dtype=torch.float32)
    scattering = torch.full((1, 4), 0.5, dtype=torch.float32)

    mixed = TorchRenderer._mix_neighbor_columns(sampled, scattering)

    expected_blurred = torch.tensor([[[12.5, 20.0, 30.0, 37.5]]], dtype=torch.float32)
    expected = sampled * 0.5 + expected_blurred * 0.5
    assert torch.allclose(mixed, expected)


def test_specular_highlight_returns_unit_interval_map():
    depth = torch.ones((2, 3), dtype=torch.float32)
    correspondence = torch.tensor([[0.0, 1.0, 2.0], [0.5, 1.5, 2.0]], dtype=torch.float32)

    highlight = TorchRenderer._specular_highlight(depth, correspondence, projector_width=3)

    assert highlight.shape == (2, 3)
    assert float(highlight.min()) >= 0.0
    assert float(highlight.max()) <= 1.0


def test_resolve_device_returns_requested_cpu():
    renderer = TorchRenderer(device="auto")

    assert renderer._resolve_device(torch, "cpu") == "cpu"
