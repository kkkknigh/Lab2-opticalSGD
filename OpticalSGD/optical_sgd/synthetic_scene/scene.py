"""合成场景的数据类"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optical_sgd.rendering.projector_camera_model import make_correspondence_map
from optical_sgd.synthetic_scene.materials import make_material_maps


@dataclass(frozen=True)
class SceneDescription:
    """合成场景描述。

    Attributes:
        height: 图像高度
        camera_width: 相机图像宽度
        projector_width: 投影仪图像宽度
        depth: 相机视角下的深度图，(height, camera_width)
        albedo: 表面漫反射率贴图，(height, camera_width)
        specular: 表面镜面反射强度贴图，(height, camera_width)
        scattering: 相邻投影列之间的光散射混合强度，(height, camera_width)
        correspondence: 相机像素到投影仪横向坐标的对应关系
        valid_mask: correspondence 是否有效
        material_name: 当前场景使用的材质名称
        projector_gamma: 投影仪亮度响应的 gamma 参数，默认1
        camera_gamma: 相机亮度响应的 gamma 参数，默认1
        material_description: 材质参数的说明
    """

    height: int
    camera_width: int
    projector_width: int
    depth: np.ndarray
    albedo: np.ndarray
    specular: np.ndarray
    scattering: np.ndarray
    correspondence: np.ndarray
    valid_mask: np.ndarray
    material_name: str
    projector_gamma: float = 1.0
    camera_gamma: float = 1.0
    material_description: str = ""


def make_depth_surface(height: int, camera_width: int, profile: str = "slanted_wave") -> np.ndarray:
    """生成相机坐标系下场景的简化深度图。

    三种 profile：
    - flat：平面，所有像素深度相同。
    - bump：带一个局部凸起的表面。
    - slanted_wave：默认值，带轻微波纹的倾斜表面。
    """

    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, height, dtype=np.float32),
        np.linspace(0.0, 1.0, camera_width, dtype=np.float32),
        indexing="ij",
    )
    if profile == "flat":
        depth = np.ones((height, camera_width), dtype=np.float32)
    elif profile == "bump":
        depth = 1.0 + 0.35 * np.exp(-((xx - 0.52) ** 2 + (yy - 0.45) ** 2) / 0.045)
    else:
        depth = 0.8 + 0.45 * xx + 0.08 * np.sin(yy * np.pi * 3.0)
    return depth.astype(np.float32)


def create_scene(config: dict) -> SceneDescription:
    """根据实验配置创建完整合成场景。

    1. 根据 scene.depth_profile 生成深度面。
    2. 根据 scene.material 生成 albedo/specular/scattering/gamma 材质参数。
    3. 根据投影仪-相机几何生成 correspondence 和 valid_mask 真值。
    """

    renderer_cfg = config["renderer"]
    scene_cfg = config["scene"]
    height = int(renderer_cfg["scene_height"])
    camera_width = int(renderer_cfg["camera_width"])
    projector_width = int(renderer_cfg["projector_width"])
    depth = make_depth_surface(height, camera_width, str(scene_cfg["depth_profile"]))
    material_maps = make_material_maps(height, camera_width, str(scene_cfg["material"]))
    geometry_kwargs = {
        "camera_fov": float(renderer_cfg.get("camera_fov", 42.0)),
        "projector_fov": float(renderer_cfg.get("projector_fov", 38.0)),
        "projector_baseline": float(renderer_cfg.get("projector_baseline", 0.08)),
    }
    correspondence, valid_mask = make_correspondence_map(
        height,
        camera_width,
        projector_width,
        depth,
        **geometry_kwargs,
    )
    return SceneDescription(
        height=height,
        camera_width=camera_width,
        projector_width=projector_width,
        depth=depth,
        albedo=material_maps.albedo,
        specular=material_maps.specular,
        scattering=material_maps.scattering,
        correspondence=correspondence,
        valid_mask=valid_mask,
        material_name=material_maps.name,
        projector_gamma=material_maps.projector_gamma,
        camera_gamma=material_maps.camera_gamma,
        material_description=material_maps.description,
    )
