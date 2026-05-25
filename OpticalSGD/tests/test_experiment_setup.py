"""实验组件构建函数单元测试。

覆盖 `build_renderer()`、`build_decoder()`、`build_scene()`、`build_initial_patterns()`
和 `build_optimizer()` 是否根据配置返回正确类型和关键字段。
"""

from __future__ import annotations

import numpy as np

from optical_sgd.correspondence_decoding.zncc_decoder import ZNCCDecoder
from optical_sgd.correspondence_decoding.zncc_neural_decoder import ZNCCNeuralDecoder
from optical_sgd.experiments.experiment_setup import (
    build_decoder,
    build_initial_patterns,
    build_optimizer,
    build_renderer,
    build_scene,
)
from optical_sgd.optimization.optical_sgd_optimizer import OpticalSGDOptimizer
from optical_sgd.rendering.mitsuba_renderer import MitsubaRenderer
from optical_sgd.rendering.torch_renderer import TorchRenderer


def minimal_config() -> dict:
    return {
        "renderer": {
            "backend": "torch",
            "device": "cpu",
            "scene_height": 2,
            "camera_width": 3,
            "projector_width": 4,
            "noise_std": 0.0,
            "ambient": 0.1,
            "camera_fov": 42.0,
            "projector_fov": 38.0,
            "projector_baseline": 0.08,
        },
        "scene": {"depth_profile": "flat", "material": "diffuse"},
        "patterns": {"count": 2, "initial_method": "constant", "seed": 1, "lowpass_fraction": 0.5},
        "decoder": {"type": "zncc", "neighborhood": 1},
        "optimization": {
            "learning_rate": 0.1,
            "iterations": 1,
            "gradient_method": "finite_difference",
            "epsilon": 0.01,
            "temperature": 10.0,
            "decoder_learning_rate": 0.02,
        },
    }


def test_build_renderer_returns_torch_renderer_for_torch_backend():
    renderer = build_renderer(minimal_config())

    assert isinstance(renderer, TorchRenderer)
    assert renderer.device == "cpu"


def test_build_renderer_returns_mitsuba_renderer_for_mitsuba_backend():
    config = minimal_config()
    config["renderer"]["backend"] = "mitsuba"

    renderer = build_renderer(config)

    assert isinstance(renderer, MitsubaRenderer)


def test_build_decoder_returns_zncc_decoder():
    decoder = build_decoder(minimal_config())

    assert isinstance(decoder, ZNCCDecoder)


def test_build_decoder_returns_neural_decoder_for_zncc_nn():
    config = minimal_config()
    config["decoder"]["type"] = "zncc_nn"

    decoder = build_decoder(config)

    assert isinstance(decoder, ZNCCNeuralDecoder)


def test_build_initial_patterns_uses_configured_shape_and_method():
    patterns = build_initial_patterns(minimal_config())

    assert patterns.shape == (2, 4)
    assert np.allclose(patterns, 0.5)


def test_build_scene_uses_configured_dimensions():
    scene = build_scene(minimal_config())

    assert scene.height == 2
    assert scene.camera_width == 3
    assert scene.projector_width == 4


def test_build_optimizer_copies_optimization_config():
    config = minimal_config()
    renderer = build_renderer(config)
    decoder = build_decoder(config)
    scene = build_scene(config)

    optimizer = build_optimizer(config, renderer, decoder, scene)

    assert isinstance(optimizer, OpticalSGDOptimizer)
    assert optimizer.learning_rate == 0.1
    assert optimizer.iterations == 1
    assert optimizer.temperature == 10.0
