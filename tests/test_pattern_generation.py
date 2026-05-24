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


def test_stripes_and_constant_initializers_have_expected_shape_and_range():
    constant = create_initial_patterns(3, 5, method="constant", seed=1)
    stripes = create_initial_patterns(3, 16, method="stripes", seed=1)

    assert np.allclose(constant, 0.5)
    assert stripes.shape == (3, 16)
    assert float(stripes.min()) >= 0.0
    assert float(stripes.max()) <= 1.0


def test_clamp_and_frequency_constraint_reduce_out_of_band_energy():
    patterns = np.array([[1.5, -0.5, 1.5, -0.5, 1.5, -0.5, 1.5, -0.5]], dtype=np.float32)

    clamped = clamp_patterns(patterns)
    filtered = apply_frequency_constraint(clamped, lowpass_fraction=0.25)

    assert clamped.dtype == np.float32
    assert float(clamped.min()) >= 0.0
    assert float(clamped.max()) <= 1.0
    assert filtered.shape == clamped.shape
    assert out_of_band_energy_ratio(filtered, 0.25) <= out_of_band_energy_ratio(clamped, 0.25)
    assert spectrum_magnitude(filtered).shape[1] == filtered.shape[1] // 2 + 1
