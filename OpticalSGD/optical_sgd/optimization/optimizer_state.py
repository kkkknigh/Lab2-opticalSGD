"""优化过程状态记录"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OptimizerState:
    """保存每轮训练后的曲线数据"""

    # 每轮优化后的 correspondence loss。
    losses: list[float] = field(default_factory=list)

    # 每轮优化后的有效像素平均绝对 correspondence 误差。
    maes: list[float] = field(default_factory=list)

    # 每轮 pattern 梯度的 L2 范数。
    gradient_norms: list[float] = field(default_factory=list)

    # 若 decoder 可学习，记录 decoder 参数梯度范数；否则为 0。
    decoder_gradient_norms: list[float] = field(default_factory=list)
