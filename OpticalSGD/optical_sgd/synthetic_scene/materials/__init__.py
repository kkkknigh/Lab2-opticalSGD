"""Material registry for synthetic projector-camera scenes."""

from __future__ import annotations

from optical_sgd.synthetic_scene.materials.base import MaterialMaps
from optical_sgd.synthetic_scene.materials.diffuse import make_diffuse
from optical_sgd.synthetic_scene.materials.frosted_glass import make_frosted_glass
from optical_sgd.synthetic_scene.materials.marble import make_marble
from optical_sgd.synthetic_scene.materials.wood import make_wood

MATERIAL_FACTORIES = {
    "diffuse": make_diffuse,
    "marble": make_marble,
    "wood": make_wood,
    "frosted_glass": make_frosted_glass,
}


def make_material_maps(height: int, camera_width: int, material: str = "diffuse") -> MaterialMaps:
    """Create all renderer-facing maps for one named material."""

    factory = MATERIAL_FACTORIES.get(str(material), make_diffuse)
    return factory(height, camera_width)
