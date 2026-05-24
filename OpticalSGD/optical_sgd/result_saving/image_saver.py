"""Image saving helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_image(path: str | Path, image: np.ndarray, cmap: str = "viridis") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, np.asarray(image), cmap=cmap)
