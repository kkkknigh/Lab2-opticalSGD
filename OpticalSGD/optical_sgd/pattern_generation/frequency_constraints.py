"""投影图案的取值约束、频率约束和频谱统计"""

from __future__ import annotations

import numpy as np


def clamp_patterns(patterns: np.ndarray) -> np.ndarray:
    """把投影图案亮度限制在 [0, 1] 范围内。"""

    return np.clip(patterns, 0.0, 1.0).astype(np.float32)


def apply_frequency_constraint(patterns: np.ndarray, lowpass_fraction: float = 0.5) -> np.ndarray:
    """对投影图案做一维 FFT 低通约束，抑制超过设定比例的高频成分。

    Args:
        patterns: 投影图案数组，形状为 (pattern_count, projector_width)。
        lowpass_fraction: 保留的低频比例，范围会被裁剪到 [0, 1]。

    Returns:
        np.ndarray: 低通滤波并裁剪到 [0, 1] 的投影图案。
    """

    patterns = np.asarray(patterns, dtype=np.float32)
    lowpass_fraction = float(np.clip(lowpass_fraction, 0.0, 1.0))
    spectrum = np.fft.rfft(patterns, axis=1)
    # rfft 只保留非负频率；cutoff 之后的频率直接置零。
    cutoff = max(1, int((spectrum.shape[1] - 1) * lowpass_fraction))
    spectrum[:, cutoff + 1 :] = 0.0
    filtered = np.fft.irfft(spectrum, n=patterns.shape[1], axis=1)
    return clamp_patterns(filtered)


def spectrum_magnitude(patterns: np.ndarray) -> np.ndarray:
    """计算每张投影图案的一维频谱幅值。"""

    spectrum = np.fft.rfft(np.asarray(patterns, dtype=np.float32), axis=1)
    return np.abs(spectrum).astype(np.float32)


def out_of_band_energy_ratio(patterns: np.ndarray, lowpass_fraction: float = 0.5) -> float:
    """统计设定低通范围之外的频谱能量占比。"""

    magnitude = spectrum_magnitude(patterns)
    cutoff = max(1, int((magnitude.shape[1] - 1) * float(np.clip(lowpass_fraction, 0.0, 1.0))))
    total = float((magnitude ** 2).sum())
    if total <= 1e-12:
        return 0.0
    return float((magnitude[:, cutoff + 1 :] ** 2).sum() / total)
