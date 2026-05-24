"""YAML 配置加载"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from optical_sgd.configuration.schema import ExperimentConfig, validate_config


DEFAULT_CONFIG_PATH = Path(__file__).with_name("default.yaml")


def _load_yaml(path: Path) -> dict[str, Any]:
    "读取给定路径的 yaml 配置文件"
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration file must contain a YAML mapping: {path}")
    return loaded


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    "递归合并默认配置和覆写配置"
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path) -> ExperimentConfig:
    """加载特定实验配置。

    Args:
        config_path: 实验 YAML 配置文件路径。

    Returns:
        合并基础配置并校验后的 `ExperimentConfig`。
    """
    path = Path(config_path).resolve()
    defaults = _load_yaml(DEFAULT_CONFIG_PATH)
    loaded = _load_yaml(path)
    data = _deep_merge(defaults, loaded)
    validate_config(data)
    return ExperimentConfig(data=data, config_path=path)
