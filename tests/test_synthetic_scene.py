from __future__ import annotations

import numpy as np

from optical_sgd.synthetic_scene import SceneDescription, create_scene, make_depth_surface, make_material_maps


def test_depth_surfaces_have_expected_shape_and_variation():
    flat = make_depth_surface(3, 4, "flat")
    bump = make_depth_surface(8, 8, "bump")
    wave = make_depth_surface(4, 5, "slanted_wave")

    assert flat.shape == (3, 4)
    assert np.allclose(flat, 1.0)
    assert bump.max() > bump.min()
    assert wave.shape == (4, 5)
    assert wave.dtype == np.float32


def test_material_registry_returns_complete_maps_for_all_named_materials():
    for material in ["diffuse", "marble", "wood", "frosted_glass"]:
        maps = make_material_maps(5, 6, material)

        assert maps.name == material
        assert maps.albedo.shape == (5, 6)
        assert maps.specular.shape == (5, 6)
        assert maps.scattering.shape == (5, 6)
        assert maps.albedo.dtype == np.float32
        assert maps.projector_gamma > 0.0
        assert maps.camera_gamma > 0.0


def test_synthetic_scene_package_exposes_complete_scene_builder():
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
