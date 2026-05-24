"""Plot saving helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_line_plot(path: str | Path, values: list[float], ylabel: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(values)
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
