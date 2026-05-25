"""投影仪-相机几何函数单元测试。

覆盖 `sample_projector_columns()` 的线性插值与边界裁剪，以及
`make_correspondence_map()` 的输出形状、列范围、深度视差和 valid_mask 行为。
"""

from __future__ import annotations

import numpy as np

from optical_sgd.rendering.projector_camera_model import (
    make_correspondence_map,
    sample_projector_columns,
)


def test_sample_projector_columns_linearly_interpolates_and_clips():
    patterns = np.array([[0.0, 10.0, 20.0, 30.0]], dtype=np.float32)
    columns = np.array([[0.0, 0.5, 1.5, 3.0, 4.0]], dtype=np.float32)

    sampled = sample_projector_columns(patterns, columns)

    assert sampled.shape == (1, 1, 5)
    assert np.allclose(sampled[0, 0], [0.0, 5.0, 15.0, 30.0, 30.0])


def test_correspondence_map_shape_and_bounds_follow_projector_width():
    depth = np.ones((2, 4), dtype=np.float32)

    correspondence, valid_mask = make_correspondence_map(height=2, camera_width=4, projector_width=8, depth=depth)

    assert correspondence.shape == (2, 4)
    assert valid_mask.shape == (2, 4)
    assert float(correspondence.min()) >= 0.0
    assert float(correspondence.max()) <= 7.0
    assert np.all(np.diff(correspondence[0]) >= 0.0)


def test_pinhole_correspondence_has_larger_disparity_for_nearer_points():
    far_depth = np.full((1, 5), 2.0, dtype=np.float32)
    near_depth = np.full((1, 5), 0.5, dtype=np.float32)

    far, _ = make_correspondence_map(height=1, camera_width=5, projector_width=64, depth=far_depth)
    near, _ = make_correspondence_map(height=1, camera_width=5, projector_width=64, depth=near_depth)

    center_col = 2
    camera_center_projector_col = 0.5 * 63.0
    far_disparity = abs(float(far[0, center_col]) - camera_center_projector_col)
    near_disparity = abs(float(near[0, center_col]) - camera_center_projector_col)
    assert near_disparity > far_disparity


def test_projector_valid_mask_rejects_points_outside_projector_fov():
    depth = np.ones((1, 7), dtype=np.float32)

    _, valid = make_correspondence_map(
        height=1,
        camera_width=7,
        projector_width=64,
        depth=depth,
        camera_fov=90.0,
        projector_fov=20.0,
        projector_baseline=0.0,
    )

    assert valid.shape == (1, 7)
    assert valid[0, 3]
    assert not valid[0, 0]
    assert not valid[0, -1]
