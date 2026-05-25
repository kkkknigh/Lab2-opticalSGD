"""基于 Mitsuba 3 的结构光物理渲染器

该实现不提供 autograd 梯度回传，主要用于生成更接近物理光照效果的观测图像，并搭配有限差分实现优化。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optical_sgd.rendering.render_result import RenderResult
from optical_sgd.synthetic_scene import SceneDescription

_ACTIVE_VARIANT: str | None = None


@dataclass
class MitsubaRenderer:
    """使用 Mitsuba 3 路径追踪模拟结构光成像的黑盒渲染器。"""

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
        """渲染投影图案，并返回 NumPy 格式的相机观测结果。

        Args:
            patterns: 投影仪图案数组，形状为 (pattern_count, projector_width)。
            scene: 合成场景描述，提供图像尺寸、材质和几何真值。

        Returns:
            RenderResult: Mitsuba 渲染图像和与场景绑定的真值数据。
        """

        mi = self._load_mitsuba()
        patterns = np.asarray(patterns, dtype=np.float32)
        captured = []
        for shot_index, pattern in enumerate(patterns):
            # 每个 pattern 单独作为 projector 纹理渲染，seed 偏移保证噪声可复现。
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
        """动态导入 Mitsuba"""

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
        """ Mitsuba 设备选择"""

        if self.variant != "auto":
            return str(self.variant)
        variants = set(mi.variants())
        if "cuda_ad_rgb" in variants:
            return "cuda_ad_rgb"
        if "llvm_ad_rgb" in variants:
            return "llvm_ad_rgb"
        return "scalar_rgb"

    def _build_scene(self, mi, pattern: np.ndarray, scene: SceneDescription):
        """把单个投影图案和合成场景组装成 Mitsuba scene 字典。"""

        transform = self._transform(mi)
        return mi.load_dict(
            {
                "type": "scene",
                "integrator": {"type": "path", "max_depth": 4},
                "sensor": self._camera_dict(transform, scene),
                "projector": self._projector_dict(mi, transform, pattern),
                "ground": self._ground_dict(mi, transform, scene),
                **self._extra_geometry_dict(scene),
            }
        )

    def _camera_dict(self, transform, scene: SceneDescription) -> dict:
        """创建 Mitsuba perspective sensor 配置。"""

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
        """创建 Mitsuba projector emitter 配置。"""

        return {
            "type": "projector",
            "irradiance": self._pattern_texture(mi, pattern),
            "scale": float(self.projector_scale),
            "fov": float(self.projector_fov),
            "to_world": self._look_at(transform, [0.65, -2.7, 1.85], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        }

    def _ground_dict(self, mi, transform, scene: SceneDescription) -> dict:
        """创建承接投影的主平面几何体配置。"""

        return {
            "type": "rectangle",
            "to_world": self._scale(transform, [1.8, 1.25, 1.0]),
            "bsdf": self._material_bsdf(mi, scene),
        }

    def _extra_geometry_dict(self, scene: SceneDescription) -> dict:
        """根据深度变化添加辅助几何，近似 bump 或起伏场景。"""

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
        # 平面深度不添加额外几何；深度变化越大，添加的球体起伏越明显。
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
        """根据场景材质名称生成 Mitsuba BSDF 配置。"""

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
        """把一维投影图案扩展为 Mitsuba projector 使用的 RGB bitmap。"""

        pattern = np.clip(np.asarray(pattern, dtype=np.float32), 0.0, 1.0)
        # projector 需要二维纹理；这里沿高度复制同一条 1D pattern。
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
        """把单通道 albedo 贴图转换为 Mitsuba RGB bitmap texture。"""

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
        """获取 Mitsuba 标量仿射变换类实例。"""

        return mi.ScalarAffineTransform4f()

    @staticmethod
    def _rgb(value: float) -> dict:
        """构造 Mitsuba RGB 常量纹理配置。"""

        return {"type": "rgb", "value": [float(value), float(value), float(value)]}

    @staticmethod
    def _look_at(transform, origin, target, up):
        """生成相机或投影仪的 look_at 变换。"""

        return transform.look_at(origin=origin, target=target, up=up)

    @staticmethod
    def _scale(transform, values):
        """生成几何缩放变换。"""

        return transform.scale(values)
