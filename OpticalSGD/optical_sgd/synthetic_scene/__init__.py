"""Public API for synthetic scene construction."""

from __future__ import annotations

from optical_sgd.synthetic_scene.materials import MATERIAL_FACTORIES, MaterialMaps, make_material_maps
from optical_sgd.synthetic_scene.scene import (
    SceneDescription,
    create_scene,
    make_depth_surface,
)

__all__ = [
    "MATERIAL_FACTORIES",
    "MaterialMaps",
    "SceneDescription",
    "create_scene",
    "make_depth_surface",
    "make_material_maps",
]
