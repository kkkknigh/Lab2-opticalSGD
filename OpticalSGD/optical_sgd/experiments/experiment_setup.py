"""对外实验启动接口"""

from __future__ import annotations

import numpy as np

from optical_sgd.correspondence_decoding.zncc_decoder import ZNCCDecoder
from optical_sgd.correspondence_decoding.zncc_neural_decoder import ZNCCNeuralDecoder
from optical_sgd.optimization.optical_sgd_optimizer import OpticalSGDOptimizer
from optical_sgd.pattern_generation.initial_patterns import create_initial_patterns
from optical_sgd.rendering.mitsuba_renderer import MitsubaRenderer
from optical_sgd.rendering.torch_renderer import TorchRenderer
from optical_sgd.synthetic_scene.scene_factory import create_scene


def build_renderer(config: dict):
    """根据配置创建渲染器。

    Args:
        config: 实验配置。

    Returns:
        TorchRenderer 或 MitsubaRenderer 实例。
    """
    renderer_cfg = config["renderer"]
    kwargs = {
        "noise_std": float(renderer_cfg["noise_std"]),
        "ambient": float(renderer_cfg["ambient"]),
        "seed": int(config["patterns"]["seed"]),
    }
    if renderer_cfg.get("backend") == "mitsuba":
        kwargs.update(
            {
                "variant": str(renderer_cfg.get("mitsuba_variant", "auto")),
                "spp": int(renderer_cfg.get("spp", 16)),
                "camera_fov": float(renderer_cfg.get("camera_fov", 42.0)),
                "projector_fov": float(renderer_cfg.get("projector_fov", 38.0)),
                "projector_scale": float(renderer_cfg.get("projector_scale", 6.0)),
                "pattern_texture_height": int(renderer_cfg.get("pattern_texture_height", 16)),
                "exposure": float(renderer_cfg.get("exposure", 1.0)),
            }
        )
        return MitsubaRenderer(**kwargs)
    kwargs["device"] = str(renderer_cfg.get("device", "auto"))
    return TorchRenderer(**kwargs)


def build_decoder(config: dict):
    """根据配置创建 correspondence decoder。

    Args:
        config: 实验配置。

    Returns:
        ZNCCDecoder 或 ZNCCNeuralDecoder 实例。
    """
    decoder_cfg = config["decoder"]
    decoder_type = str(decoder_cfg["type"])
    neighborhood = int(decoder_cfg.get("neighborhood", 1))
    if decoder_type == "zncc_nn":
        return ZNCCNeuralDecoder(neighborhood=neighborhood, seed=int(config["patterns"]["seed"]))
    return ZNCCDecoder(neighborhood=neighborhood)


def build_scene(config: dict):
    """根据配置创建合成场景。

    Args:
        config: 实验配置。

    Returns:
        `SceneDescription`，包含深度、材质、对应关系和有效 mask。
    """
    return create_scene(config)


def build_initial_patterns(config: dict) -> np.ndarray:
    """根据配置创建初始投影 pattern。

    Args:
        config: 实验配置。

    Returns:
        形状为 (pattern_count, projector_width) 的 float32 pattern 数组。
    """
    return create_initial_patterns(
        int(config["patterns"]["count"]),
        int(config["renderer"]["projector_width"]),
        str(config["patterns"]["initial_method"]),
        int(config["patterns"]["seed"]),
    )


def build_optimizer(config: dict, renderer, decoder, scene) -> OpticalSGDOptimizer:
    """根据配置和已构建组件创建 OpticalSGD 优化器。

    Args:
        config: 实验配置。
        renderer: 渲染器实例。
        decoder: correspondence decoder 实例。
        scene: 合成场景描述。

    Returns:
        配置好的 OpticalSGDOptimizer。
    """
    return OpticalSGDOptimizer(
        renderer=renderer,
        decoder=decoder,
        scene=scene,
        learning_rate=float(config["optimization"]["learning_rate"]),
        iterations=int(config["optimization"]["iterations"]),
        gradient_method=str(config["optimization"]["gradient_method"]),
        finite_difference_epsilon=float(config["optimization"]["epsilon"]),
        lowpass_fraction=float(config["patterns"]["lowpass_fraction"]),
        temperature=float(config["optimization"]["temperature"]),
        decoder_learning_rate=float(config["optimization"].get("decoder_learning_rate", 0.02)),
    )
