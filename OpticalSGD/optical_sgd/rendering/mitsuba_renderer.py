"""Official Mitsuba 3 black-box renderer for finite-difference OpticalSGD.

This backend intentionally does not expose a PyTorch autograd path.  It uses
Mitsuba's projector emitter to cast each 1D structured-light pattern as a 2D
bitmap texture into a small physically rendered scene, then returns the rendered
camera image to the same decoder/loss pipeline used by the Torch backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optical_sgd.rendering.render_result import RenderResult
from optical_sgd.synthetic_scene.scene_description import SceneDescription

_ACTIVE_VARIANT: str | None = None


@dataclass
class MitsubaRenderer:
    noise_std: float = 0.01
    ambient: float = 0.02
    seed: int = 7
    variant: str = "auto"
    spp: int = 16
    camera_fov: float = 42.0
    projector_fov: float = 38.0
    projector_scale: float = 6.0
    pattern_texture_height: int = 16
    exposure: float = 1.0

    def render(self, patterns: np.ndarray, scene: SceneDescription) -> RenderResult:
        mi = self._load_mitsuba()
        patterns = np.asarray(patterns, dtype=np.float32)
        captured = []
        for shot_index, pattern in enumerate(patterns):
            mitsuba_scene = self._build_scene(mi, pattern, scene)
            image = mi.render(mitsuba_scene, spp=int(self.spp), seed=int(self.seed) + shot_index)
            rgb = np.asarray(image, dtype=np.float32)
            grayscale = rgb.mean(axis=-1) if rgb.ndim == 3 else rgb
            grayscale = grayscale * float(self.exposure) + float(self.ambient)
            captured.append(grayscale)
        captured_images = np.stack(captured, axis=0).astype(np.float32)
        if self.noise_std > 0:
            rng = np.random.default_rng(int(self.seed))
            captured_images = captured_images + rng.normal(0.0, float(self.noise_std), captured_images.shape)
        captured_images = np.clip(captured_images, 0.0, 1.0).astype(np.float32)
        return RenderResult(
            captured_images=captured_images,
            ground_truth_correspondence=scene.correspondence.astype(np.float32),
            valid_mask=scene.valid_mask.astype(bool),
            albedo=scene.albedo.astype(np.float32),
            depth=scene.depth.astype(np.float32),
        )

    def _load_mitsuba(self):
        global _ACTIVE_VARIANT
        try:
            import mitsuba as mi
        except Exception as exc:
            raise ImportError(
                "renderer.backend='mitsuba' requires the official Mitsuba 3 package. "
                "Install it with: pip install mitsuba"
            ) from exc

        variant = self._choose_variant(mi)
        if _ACTIVE_VARIANT != variant:
            mi.set_variant(variant)
            _ACTIVE_VARIANT = variant
        return mi

    def _choose_variant(self, mi) -> str:
        if self.variant != "auto":
            return str(self.variant)
        variants = set(mi.variants())
        if "cuda_ad_rgb" in variants:
            return "cuda_ad_rgb"
        if "llvm_ad_rgb" in variants:
            return "llvm_ad_rgb"
        return "scalar_rgb"

    def _build_scene(self, mi, pattern: np.ndarray, scene: SceneDescription):
        transform = self._transform(mi)
        return mi.load_dict(
            {
                "type": "scene",
                "integrator": {"type": "path", "max_depth": 4},
                "sensor": self._camera_dict(mi, transform, scene),
                "projector": self._projector_dict(mi, transform, pattern),
                "ground": self._ground_dict(mi, transform, scene),
                **self._extra_geometry_dict(transform, scene),
            }
        )

    def _camera_dict(self, mi, transform, scene: SceneDescription) -> dict:
        return {
            "type": "perspective",
            "fov": float(self.camera_fov),
            "to_world": self._look_at(transform, [0.0, -3.2, 2.1], [0.0, 0.0, 0.05], [0.0, 0.0, 1.0]),
            "sampler": {
                "type": "independent",
                "sample_count": int(self.spp),
            },
            "film": {
                "type": "hdrfilm",
                "width": int(scene.camera_width),
                "height": int(scene.height),
                "pixel_format": "rgb",
                "component_format": "float32",
                "rfilter": {"type": "box"},
            },
        }

    def _projector_dict(self, mi, transform, pattern: np.ndarray) -> dict:
        return {
            "type": "projector",
            "irradiance": self._pattern_texture(mi, pattern),
            "scale": float(self.projector_scale),
            "fov": float(self.projector_fov),
            "to_world": self._look_at(transform, [0.65, -2.7, 1.85], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        }

    def _ground_dict(self, mi, transform, scene: SceneDescription) -> dict:
        return {
            "type": "rectangle",
            "to_world": self._scale(transform, [1.8, 1.25, 1.0]),
            "bsdf": self._material_bsdf(mi, scene),
        }

    def _extra_geometry_dict(self, transform, scene: SceneDescription) -> dict:
        if scene.material_name == "frosted_glass":
            bsdf = {
                "type": "twosided",
                "bsdf": {
                    "type": "roughdielectric",
                    "distribution": "ggx",
                    "alpha": 0.35,
                    "int_ior": "bk7",
                    "ext_ior": "air",
                },
            }
        else:
            bsdf = {
                "type": "roughplastic",
                "diffuse_reflectance": self._rgb(0.55),
                "specular_reflectance": self._rgb(0.12),
                "alpha": 0.25,
            }
        if scene.depth.max() - scene.depth.min() < 0.05:
            return {}
        if scene.depth.max() - scene.depth.min() < 0.25:
            return {
                "bump_sphere": {
                    "type": "sphere",
                    "center": [0.18, -0.05, 0.28],
                    "radius": 0.28,
                    "bsdf": bsdf,
                }
            }
        return {
            "left_sphere": {
                "type": "sphere",
                "center": [-0.34, -0.08, 0.22],
                "radius": 0.22,
                "bsdf": bsdf,
            },
            "right_sphere": {
                "type": "sphere",
                "center": [0.34, 0.12, 0.34],
                "radius": 0.34,
                "bsdf": bsdf,
            },
        }

    def _material_bsdf(self, mi, scene: SceneDescription) -> dict:
        reflectance = self._albedo_texture(mi, scene.albedo)
        if scene.material_name == "frosted_glass":
            return {
                "type": "twosided",
                "bsdf": {
                    "type": "roughplastic",
                    "diffuse_reflectance": reflectance,
                    "specular_reflectance": self._rgb(0.22),
                    "alpha": 0.65,
                },
            }
        if scene.material_name in {"marble", "wood"}:
            return {
                "type": "twosided",
                "bsdf": {
                    "type": "roughplastic",
                    "diffuse_reflectance": reflectance,
                    "specular_reflectance": self._rgb(0.14),
                    "alpha": 0.22 if scene.material_name == "marble" else 0.34,
                },
            }
        return {
            "type": "twosided",
            "bsdf": {
                "type": "diffuse",
                "reflectance": reflectance,
            },
        }

    def _pattern_texture(self, mi, pattern: np.ndarray) -> dict:
        pattern = np.clip(np.asarray(pattern, dtype=np.float32), 0.0, 1.0)
        image = np.repeat(pattern[None, :, None], int(self.pattern_texture_height), axis=0)
        image = np.repeat(image, 3, axis=2)
        return {
            "type": "bitmap",
            "bitmap": mi.Bitmap(np.ascontiguousarray(image.astype(np.float32))),
            "raw": True,
            "filter_type": "bilinear",
            "wrap_mode": "clamp",
        }

    @staticmethod
    def _albedo_texture(mi, albedo: np.ndarray) -> dict:
        image = np.repeat(np.asarray(albedo, dtype=np.float32)[:, :, None], 3, axis=2)
        return {
            "type": "bitmap",
            "bitmap": mi.Bitmap(np.ascontiguousarray(np.clip(image, 0.0, 1.0))),
            "raw": True,
            "filter_type": "bilinear",
            "wrap_mode": "clamp",
        }

    @staticmethod
    def _transform(mi):
        if hasattr(mi, "ScalarAffineTransform4f"):
            return mi.ScalarAffineTransform4f()
        return mi.ScalarTransform4f()

    @staticmethod
    def _rgb(value: float) -> dict:
        return {"type": "rgb", "value": [float(value), float(value), float(value)]}

    @staticmethod
    def _look_at(transform, origin, target, up):
        if hasattr(transform, "look_at"):
            return transform.look_at(origin=origin, target=target, up=up)
        return transform.__class__.look_at(origin=origin, target=target, up=up)

    @staticmethod
    def _scale(transform, values):
        if hasattr(transform, "scale"):
            return transform.scale(values)
        return transform.__class__.scale(values)
