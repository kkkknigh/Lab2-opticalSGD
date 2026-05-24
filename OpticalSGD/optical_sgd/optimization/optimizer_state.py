"""Optimizer state containers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OptimizerState:
    losses: list[float] = field(default_factory=list)
    maes: list[float] = field(default_factory=list)
    gradient_norms: list[float] = field(default_factory=list)
    decoder_gradient_norms: list[float] = field(default_factory=list)
