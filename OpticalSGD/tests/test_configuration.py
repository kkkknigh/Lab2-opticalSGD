"""配置模块单元测试。

覆盖 `load_config()` 的默认配置合并和相对输出路径解析，以及 `validate_config()`
对非法数值、缺失必需字段的输入校验。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from optical_sgd.configuration.loader import load_config
from optical_sgd.configuration.schema import validate_config


def test_load_config_deep_merges_defaults_and_resolves_relative_output(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
experiment:
  name: unit_test
  output_dir: analysis/output
renderer:
  projector_width: 17
patterns:
  count: 3
optimization:
  iterations: 2
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.experiment_name == "unit_test"
    assert config.output_dir == tmp_path / "analysis" / "output"
    assert config.data["renderer"]["projector_width"] == 17
    assert config.data["renderer"]["backend"] == "torch"
    assert config.data["scene"]["material"] == "diffuse"
    assert config.data["patterns"]["count"] == 3


def test_validate_config_rejects_invalid_numeric_values():
    data = {
        "experiment": {"name": "bad", "output_dir": "output"},
        "renderer": {
            "backend": "torch",
            "camera_width": 16,
            "scene_height": 8,
            "projector_width": 1,
            "noise_std": 0.0,
            "ambient": 0.1,
        },
        "scene": {"depth_profile": "flat", "material": "diffuse"},
        "patterns": {"count": 1, "initial_method": "random", "seed": 1, "lowpass_fraction": 0.5},
        "decoder": {"type": "zncc"},
        "optimization": {
            "gradient_method": "finite_difference",
            "iterations": 1,
            "learning_rate": 0.1,
            "epsilon": 0.01,
            "temperature": 1.0,
        },
    }

    with pytest.raises(ValueError, match="projector_width"):
        validate_config(data)


def test_validate_config_rejects_missing_required_fields():
    data = {
        "experiment": {"name": "bad", "output_dir": "output"},
        "renderer": {"projector_width": 16},
        "scene": {},
        "patterns": {"count": 1},
        "decoder": {},
        "optimization": {"iterations": 1},
    }

    with pytest.raises(ValueError, match="renderer.backend"):
        validate_config(data)
