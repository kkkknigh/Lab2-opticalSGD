"""实验结果保存工具。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def prepare_output_directory(path: str | Path) -> Path:
    """创建并返回实验输出目录。"""

    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_image(path: str | Path, image: np.ndarray, cmap: str = "viridis") -> None:
    """把二维数组保存为图片。

    Args:
        path: 输出图片路径。
        image: 待保存图像数组。
        cmap: matplotlib colormap 名称。
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, np.asarray(image), cmap=cmap)


def save_line_plot(path: str | Path, values: list[float], ylabel: str) -> None:
    """把一维数值序列保存为训练曲线图。"""

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


def save_metrics_json(path: str | Path, metrics: dict) -> None:
    """把实验指标保存为 UTF-8 JSON 文件。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def save_rows_csv(path: str | Path, rows: list[dict]) -> None:
    """把字典列表保存为 CSV 表格。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path: str | Path, **arrays) -> None:
    """把数组保存为压缩 npz checkpoint。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
