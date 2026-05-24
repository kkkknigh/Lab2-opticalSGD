"""Factory for configured synthetic scenes."""

from __future__ import annotations

from optical_sgd.rendering.projector_camera_model import make_correspondence_map
from optical_sgd.synthetic_scene.depth_surfaces import make_depth_surface
from optical_sgd.synthetic_scene.materials import make_material_maps
from optical_sgd.synthetic_scene.scene_description import SceneDescription


def create_scene(config: dict) -> SceneDescription:
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
