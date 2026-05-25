"""合成场景和材质贴图单元测试。

覆盖归一化网格、常量材质贴图、不同深度 profile、材质注册表输出，
以及 `create_scene()` 返回的场景描述字段形状。
"""

from __future__ import annotations

import numpy as np

from optical_sgd.synthetic_scene import SceneDescription, create_scene, make_depth_surface, make_material_maps
from optical_sgd.synthetic_scene.materials.base import constant_map, normalized_grid


def test_normalized_grid_returns_height_width_arrays_in_unit_range():
    yy, xx = normalized_grid(3, 4)

    assert yy.shape == (3, 4)
    assert xx.shape == (3, 4)
    assert np.isclose(yy[0, 0], 0.0)
    assert np.isclose(yy[-1, 0], 1.0)
    assert np.isclose(xx[0, 0], 0.0)
    assert np.isclose(xx[0, -1], 1.0)


def test_constant_map_fills_float32_array():
    material_map = constant_map(2, 3, 0.25)

    assert material_map.shape == (2, 3)
    assert material_map.dtype == np.float32
    assert np.allclose(material_map, 0.25)


def test_make_depth_surface_flat_returns_unit_depth():
    flat = make_depth_surface(3, 4, "flat")

    assert flat.shape == (3, 4)
    assert flat.dtype == np.float32
    assert np.allclose(flat, 1.0)


def test_make_depth_surface_bump_has_spatial_variation():
    bump = make_depth_surface(8, 8, "bump")

    assert bump.shape == (8, 8)
    assert bump.max() > bump.min()


def test_make_depth_surface_slanted_wave_has_expected_shape():
    wave = make_depth_surface(4, 5, "slanted_wave")

    assert wave.shape == (4, 5)
    assert wave.dtype == np.float32


def test_make_material_maps_returns_requested_material_name_and_shapes():
    maps = make_material_maps(5, 6, "marble")

    assert maps.name == "marble"
    assert maps.albedo.shape == (5, 6)
    assert maps.specular.shape == (5, 6)
    assert maps.scattering.shape == (5, 6)
    assert maps.albedo.dtype == np.float32
    assert maps.projector_gamma > 0.0
    assert maps.camera_gamma > 0.0


def test_make_material_maps_falls_back_to_diffuse_for_unknown_material():
    maps = make_material_maps(2, 3, "unknown")

    assert maps.name == "diffuse"


def test_create_scene_returns_scene_description_with_expected_shapes():
    config = {
        "renderer": {
            "scene_height": 4,
            "camera_width": 5,
            "projector_width": 8,
            "camera_fov": 42.0,
            "projector_fov": 38.0,
            "projector_baseline": 0.08,
        },
        "scene": {
            "depth_profile": "bump",
            "material": "marble",
        },
    }

    scene = create_scene(config)

    assert isinstance(scene, SceneDescription)
    assert scene.depth.shape == (4, 5)
    assert scene.correspondence.shape == (4, 5)
    assert scene.valid_mask.shape == (4, 5)
    assert scene.material_name == "marble"
