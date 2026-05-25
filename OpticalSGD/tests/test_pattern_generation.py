"""pattern 生成和频率约束单元测试。

覆盖初始 pattern 的 deterministic/random/constant/stripes 输出，亮度裁剪、
FFT 低通约束、频谱形状和带外能量占比。
"""

from __future__ import annotations

import numpy as np

from optical_sgd.pattern_generation.frequency_constraints import (
    apply_frequency_constraint,
    clamp_patterns,
    out_of_band_energy_ratio,
    spectrum_magnitude,
)
from optical_sgd.pattern_generation.initial_patterns import create_initial_patterns


def test_initial_patterns_are_deterministic_and_bounded():
    first = create_initial_patterns(2, 8, method="random", seed=123)
    second = create_initial_patterns(2, 8, method="random", seed=123)

    assert first.shape == (2, 8)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert float(first.min()) >= 0.0
    assert float(first.max()) <= 1.0


def test_create_initial_patterns_returns_constant_mid_gray():
    constant = create_initial_patterns(3, 5, method="constant", seed=1)

    assert constant.shape == (3, 5)
    assert constant.dtype == np.float32
    assert np.allclose(constant, 0.5)


def test_create_initial_patterns_returns_bounded_stripes():
    stripes = create_initial_patterns(3, 16, method="stripes", seed=1)

    assert stripes.shape == (3, 16)
    assert float(stripes.min()) >= 0.0
    assert float(stripes.max()) <= 1.0


def test_clamp_patterns_clips_values_to_projector_range():
    patterns = np.array([[1.5, -0.5, 1.5, -0.5, 1.5, -0.5, 1.5, -0.5]], dtype=np.float32)

    clamped = clamp_patterns(patterns)

    assert clamped.dtype == np.float32
    assert float(clamped.min()) >= 0.0
    assert float(clamped.max()) <= 1.0


def test_apply_frequency_constraint_preserves_shape_and_reduces_high_frequency_energy():
    patterns = np.array([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]], dtype=np.float32)

    filtered = apply_frequency_constraint(patterns, lowpass_fraction=0.25)

    assert filtered.shape == patterns.shape
    assert out_of_band_energy_ratio(filtered, 0.25) <= out_of_band_energy_ratio(patterns, 0.25)


def test_spectrum_magnitude_uses_rfft_width():
    patterns = np.ones((2, 8), dtype=np.float32)

    magnitude = spectrum_magnitude(patterns)

    assert magnitude.shape == (2, 5)


def test_out_of_band_energy_ratio_returns_zero_for_zero_signal():
    patterns = np.zeros((2, 8), dtype=np.float32)

    assert out_of_band_energy_ratio(patterns, 0.25) == 0.0
