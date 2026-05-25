"""实验结果保存函数单元测试。

覆盖输出目录创建、JSON/CSV 保存、空表格处理、npz checkpoint 保存、
图像保存和曲线图保存这些独立输入输出行为。
"""

from __future__ import annotations

import json

import numpy as np

from optical_sgd.result_saving.savers import (
    prepare_output_directory,
    save_checkpoint,
    save_image,
    save_line_plot,
    save_metrics_json,
    save_rows_csv,
)


def test_prepare_output_directory_creates_path(tmp_path):
    output_dir = prepare_output_directory(tmp_path / "nested" / "output")

    assert output_dir.exists()
    assert output_dir.is_dir()


def test_save_metrics_json_writes_dictionary(tmp_path):
    path = tmp_path / "metrics.json"

    save_metrics_json(path, {"loss": 1.25})

    assert json.loads(path.read_text(encoding="utf-8")) == {"loss": 1.25}


def test_save_rows_csv_writes_header_and_rows(tmp_path):
    path = tmp_path / "rows.csv"

    save_rows_csv(path, [{"a": 1, "b": 2}, {"a": 3, "b": 4}])

    assert path.read_text(encoding="utf-8").splitlines() == ["a,b", "1,2", "3,4"]


def test_save_rows_csv_keeps_columns_from_later_rows(tmp_path):
    path = tmp_path / "rows.csv"

    save_rows_csv(path, [{"a": 1}, {"a": 2, "b": 3}])

    assert path.read_text(encoding="utf-8").splitlines() == ["a,b", "1,", "2,3"]


def test_save_rows_csv_with_empty_rows_removes_stale_file(tmp_path):
    path = tmp_path / "tables" / "empty.csv"
    path.parent.mkdir(parents=True)
    path.write_text("stale,data\n1,2\n", encoding="utf-8")

    save_rows_csv(path, [])

    assert path.parent.exists()
    assert not path.exists()


def test_save_checkpoint_writes_npz_arrays(tmp_path):
    path = tmp_path / "checkpoint.npz"

    save_checkpoint(path, patterns=np.array([[1.0, 2.0]], dtype=np.float32))

    loaded = np.load(path)
    assert np.array_equal(loaded["patterns"], np.array([[1.0, 2.0]], dtype=np.float32))


def test_save_image_writes_file(tmp_path):
    path = tmp_path / "image.png"

    save_image(path, np.zeros((2, 2), dtype=np.float32), cmap="gray")

    assert path.exists()
    assert path.stat().st_size > 0


def test_save_line_plot_writes_file(tmp_path):
    path = tmp_path / "curve.png"

    save_line_plot(path, [1.0, 0.5, 0.25], "loss")

    assert path.exists()
    assert path.stat().st_size > 0
