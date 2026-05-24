"""渲染结果的数据类"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RenderResult:
    captured_images: np.ndarray                 # (pattern_count, height, camera_width)
    ground_truth_correspondence: np.ndarray     # (height, camera_width)
    valid_mask: np.ndarray                      # (height, camera_width)
    albedo: np.ndarray
    depth: np.ndarray
