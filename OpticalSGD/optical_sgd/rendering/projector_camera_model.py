"""简化的投影仪-相机几何模型，用于合成实验中的 projector-column correspondence 真值生成。

理想化建模：
1. 物体三维几何用相机视角下的 depth map 表示，每个相机像素对应一个表面点。
2. 相机成像平面用 height x camera_width 像素网格表示。
3. 相机和投影仪都使用理想 pinhole 模型，并由各自的水平 FOV 决定视锥。
4. 相机作为原点，光轴为z轴。
5. 相机成像平面上的像素通过 depth 反投影到相机坐标系中的 3D 点。
6. 投影仪相对相机沿水平 x 方向平移 projector_baseline，光轴保持平行。
7. 同一个 3D 点再投影到投影仪成像平面，得到一维 projector column。
8. 用 valid_mask 标记该 3D 点是否落在投影仪水平视野内，也即是否合法。

"""

from __future__ import annotations

import numpy as np


def make_correspondence_map(
    height: int,
    camera_width: int,
    projector_width: int,
    depth: np.ndarray,
    camera_fov: float = 42.0,
    projector_fov: float = 38.0,
    projector_baseline: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    """生成相机像素到投影仪列坐标的真值对应关系和有效区域。

    Args:
        height: 相机成像平面的高度
        camera_width: 相机成像平面的宽度
        projector_width: 投影仪一维 pattern 的宽度
        depth: 形状为(height, camera_width)的深度图（相机坐标系下每个相机像素对应表面点的 z 距离）。
        camera_fov: 相机水平视场角。
        projector_fov: 投影仪水平视场角。
        projector_baseline: 投影仪相对相机沿 x 方向的平移距离。

    Returns:
        correspondence:
            形状为(height, camera_width)的浮点数组，每个元素是对应相机像素看到的投影仪列坐标。
            越过投影仪水平视野的坐标会被裁剪到投影仪边界，保证后续采样不越界。
        valid_mask:
            形状为(height, camera_width)的 bool 数组。True 表示该像素对应的 3D 点落在
            投影仪水平视野内，可以参与 loss 和评价指标；False 表示应该被忽略。

    """

    # 获得[0, 1] 范围内的投影仪归一化横坐标
    projector_x = _projector_normalized_x(
        height,
        camera_width,
        depth,
        camera_fov,
        projector_fov,
        projector_baseline,
    )

    # 归一化投影坐标在 [0, 1] 内，说明落在投影仪水平视野内。
    valid_mask = ((projector_x >= 0.0) & (projector_x <= 1.0)).astype(bool)

    # correspondence clip 限制到投影仪范围内，保证后续采样 pattern 不越界。
    correspondence = np.clip(projector_x, 0.0, 1.0) * float(projector_width - 1)

    return correspondence, valid_mask


def _projector_normalized_x(
    height: int,
    camera_width: int,
    depth: np.ndarray,
    camera_fov: float,
    projector_fov: float,
    projector_baseline: float,
) -> np.ndarray:
    """ 用一维 pinhole 几何计算投影仪归一化横坐标 """

    depth = np.asarray(depth, dtype=np.float32)
    if depth.shape != (height, camera_width):
        raise ValueError("depth shape must match (height, camera_width)")

    # z=1 成像平面上的半宽
    camera_half_width = np.tan(np.deg2rad(float(camera_fov)) * 0.5)
    projector_half_width = np.tan(np.deg2rad(float(projector_fov)) * 0.5)

    # 把相机像素列坐标变成相机光线方向 camera_ray_x：
    # 给每个相机列分配一条从相机原点出发的光线方向 (x, z=1)。
    camera_ray_x = np.linspace(-camera_half_width, camera_half_width, camera_width, dtype=np.float32)[None, :]

    # 2. 用 depth 把相机光线上的点反投影成相机坐标系下的 3D 点 (x, z)
    point_x_camera = camera_ray_x * depth

    # 3. 根据 projector_baseline，把该 3D 点平移到投影仪坐标系。
    point_x_projector = point_x_camera - float(projector_baseline)

    # 4. 用 pinhole 投影公式 x / z 得到该点在投影仪成像平面的横向位置。
    projector_ray_x = point_x_projector / np.maximum(depth, 1e-6)

    # 5. 根据 projector_fov，把投影仪成像平面横坐标转成 [0, 1] 归一化列坐标
    return 0.5 + 0.5 * projector_ray_x / max(projector_half_width, 1e-6)


def sample_projector_columns(patterns: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """按照列坐标，从一维投影图案中线性插值采样亮度。

    Args:
        patterns: 形状为(pattern_count, projector_width) 的投影图案，每一行是一张一维 projector-column pattern。
        columns: 形状为(height, camera_width) 的每像素对应投影仪列坐标。

    Returns:
        形状为 (pattern_count, height, camera_width) 的数组，表示每张 pattern
        被投影到场景后，相机每个像素看到的投影亮度。
。
    """

    patterns = np.asarray(patterns, dtype=np.float32)
    columns = np.asarray(columns, dtype=np.float32)
    max_col = patterns.shape[1] - 1

    # 防止列坐标越过 pattern 数组边界。
    columns = np.clip(columns, 0.0, float(max_col))

    # 找到浮点列坐标左右两侧的整数列。
    left = np.floor(columns).astype(np.int64)
    right = np.clip(left + 1, 0, max_col)

    frac = columns - left
    sampled = []
    for pattern in patterns:
        # 浮点坐标用左右两列做线性插值。
        values = pattern[left] * (1.0 - frac) + pattern[right] * frac
        sampled.append(values)
    return np.stack(sampled, axis=0).astype(np.float32)
