"""Pattern value and frequency constraints."""

from __future__ import annotations

import numpy as np


def clamp_patterns(patterns: np.ndarray) -> np.ndarray:
    return np.clip(patterns, 0.0, 1.0).astype(np.float32)


def apply_frequency_constraint(patterns: np.ndarray, lowpass_fraction: float = 0.5) -> np.ndarray:
    patterns = np.asarray(patterns, dtype=np.float32)
    lowpass_fraction = float(np.clip(lowpass_fraction, 0.0, 1.0))
    spectrum = np.fft.rfft(patterns, axis=1)
    cutoff = max(1, int((spectrum.shape[1] - 1) * lowpass_fraction))
    spectrum[:, cutoff + 1 :] = 0.0
    filtered = np.fft.irfft(spectrum, n=patterns.shape[1], axis=1)
    return clamp_patterns(filtered)


def spectrum_magnitude(patterns: np.ndarray) -> np.ndarray:
    spectrum = np.fft.rfft(np.asarray(patterns, dtype=np.float32), axis=1)
    return np.abs(spectrum).astype(np.float32)


def out_of_band_energy_ratio(patterns: np.ndarray, lowpass_fraction: float = 0.5) -> float:
    magnitude = spectrum_magnitude(patterns)
    cutoff = max(1, int((magnitude.shape[1] - 1) * float(np.clip(lowpass_fraction, 0.0, 1.0))))
    total = float((magnitude ** 2).sum())
    if total <= 1e-12:
        return 0.0
    return float((magnitude[:, cutoff + 1 :] ** 2).sum() / total)
