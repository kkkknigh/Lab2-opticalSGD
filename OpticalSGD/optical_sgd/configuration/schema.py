"""定义配置对象并提供校验接口"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:

    data: dict[str, Any]
    config_path: Path | None = None

    @property
    def experiment_name(self) -> str:
        return str(self.data["experiment"]["name"])

    @property
    def output_dir(self) -> Path:
        raw = Path(str(self.data["experiment"]["output_dir"]))
        if raw.is_absolute() or self.config_path is None:
            return raw
        return self.config_path.parent / raw


def validate_config(data: dict[str, Any]) -> None:
    """校验合并后的实验配置是否合法。

    Args:
        data: 完整配置字典。

    Raises:
        ValueError: 当必需配置段缺失、配置段类型错误，或关键数值明显不合法时抛出。
    """
    required_sections = [
        "experiment",
        "renderer",
        "scene",
        "patterns",
        "decoder",
        "optimization",
    ]
    for section in required_sections:
        if section not in data or not isinstance(data[section], dict):
            raise ValueError(f"Missing configuration section: {section}")

    required_fields = {
        "experiment": ["name", "output_dir"],
        "renderer": ["backend", "camera_width", "scene_height", "projector_width", "noise_std", "ambient"],
        "scene": ["depth_profile", "material"],
        "patterns": ["count", "initial_method", "seed", "lowpass_fraction"],
        "decoder": ["type"],
        "optimization": ["gradient_method", "iterations", "learning_rate", "epsilon", "temperature"],
    }
    for section_name, keys in required_fields.items():
        for key in keys:
            if key not in data[section_name]:
                raise ValueError(f"Missing configuration field: {section_name}.{key}")

    if not str(data["experiment"]["name"]):
        raise ValueError("experiment.name cannot be empty")
    if not str(data["experiment"]["output_dir"]):
        raise ValueError("experiment.output_dir cannot be empty")
    if int(data["renderer"]["camera_width"]) <= 1:
        raise ValueError("renderer.camera_width must be greater than 1")
    if int(data["renderer"]["scene_height"]) <= 0:
        raise ValueError("renderer.scene_height must be positive")
    if int(data["renderer"]["projector_width"]) <= 1:
        raise ValueError("renderer.projector_width must be greater than 1")
    if float(data["renderer"]["noise_std"]) < 0.0:
        raise ValueError("renderer.noise_std cannot be negative")
    if float(data["renderer"]["ambient"]) < 0.0:
        raise ValueError("renderer.ambient cannot be negative")
    if not str(data["scene"]["depth_profile"]):
        raise ValueError("scene.depth_profile cannot be empty")
    if not str(data["scene"]["material"]):
        raise ValueError("scene.material cannot be empty")
    if int(data["patterns"]["count"]) <= 0:
        raise ValueError("patterns.count must be positive")
    if not str(data["patterns"]["initial_method"]):
        raise ValueError("patterns.initial_method cannot be empty")
    lowpass_fraction = float(data["patterns"]["lowpass_fraction"])
    if not 0.0 <= lowpass_fraction <= 1.0:
        raise ValueError("patterns.lowpass_fraction must be in [0, 1]")
    if not str(data["decoder"]["type"]):
        raise ValueError("decoder.type cannot be empty")
    if int(data["optimization"]["iterations"]) < 0:
        raise ValueError("optimization.iterations cannot be negative")
    if float(data["optimization"]["learning_rate"]) <= 0.0:
        raise ValueError("optimization.learning_rate must be positive")
    if float(data["optimization"]["epsilon"]) <= 0.0:
        raise ValueError("optimization.epsilon must be positive")
    if float(data["optimization"]["temperature"]) <= 0.0:
        raise ValueError("optimization.temperature must be positive")
