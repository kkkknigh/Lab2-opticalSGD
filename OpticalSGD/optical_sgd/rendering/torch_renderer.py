"""基于 PyTorch 的结构光可微渲染器。

模块中直接使用 SceneDescription 中的深度、材质和相机-投影仪对应真值，
生成模拟相机拍到的结果，并支持梯度反传。
"""

from __future__ import annotations

from dataclasses import dataclass

from optical_sgd.synthetic_scene import SceneDescription


@dataclass
class TorchRenderer:
    """使用 PyTorch 张量运算模拟结构光成像的可微渲染器。"""

    noise_std: float = 0.01    # 相机噪声标准差
    ambient: float = 0.02      # 环境光亮度
    seed: int = 7              # 随机种子
    device: str = "auto"

    def render_torch(self, patterns, scene: SceneDescription, device: str | None = None) -> dict:
        """使用 PyTorch 运算渲染图案，保留对 `patterns` 的梯度路径。

        Args:
            patterns: NumPy 数组或 torch.Tensor， (pattern_count, projector_width)。
                当传入 tensor 且 `requires_grad=True` 时，输出图像可对它反传梯度。
            scene: 合成场景描述。
            device: 可选 auto / cpu / gpu

        Returns:
            dict: 包含 `captured_images`、`ground_truth_correspondence`、
            `valid_mask` 的张量结果。
        """

        try:
            import torch
        except Exception as exc:
            raise ImportError("TorchRenderer requires PyTorch.") from exc

        resolved_device = self._resolve_device(torch, device)
        # 输入统一为 torch.Tensor
        if not torch.is_tensor(patterns):
            patterns = torch.as_tensor(patterns, dtype=torch.float32, device=resolved_device)
        else:
            patterns = patterns.to(dtype=torch.float32)
            patterns = patterns.to(resolved_device)

        # 场景数值统一到同一设备
        tensor_device = patterns.device
        correspondence = torch.as_tensor(scene.correspondence, dtype=torch.float32, device=tensor_device)
        albedo = torch.as_tensor(scene.albedo, dtype=torch.float32, device=tensor_device)
        specular = torch.as_tensor(scene.specular, dtype=torch.float32, device=tensor_device)
        scattering = torch.as_tensor(scene.scattering, dtype=torch.float32, device=tensor_device)
        depth = torch.as_tensor(scene.depth, dtype=torch.float32, device=tensor_device)
        valid_mask = torch.as_tensor(scene.valid_mask, dtype=torch.bool, device=tensor_device)

        projector_gamma = max(float(scene.projector_gamma), 1e-4)
        camera_gamma = max(float(scene.camera_gamma), 1e-4)

        # projector_gamma 模拟投影仪输入灰度到实际出射亮度之间的非线性响应。
        projected = torch.pow(torch.clamp(patterns, 0.0, 1.0), projector_gamma)

        # 用 correspondence[y, x] 把一维 projector pattern 重采样成相机视角下的亮度图。
        sampled = self._sample_projector_columns_torch(projected, correspondence)

        # scattering 控制相邻投影列之间的混合强度，用来近似串扰。
        scattered = self._mix_neighbor_columns(sampled, scattering)

        # 根据深度图生成简化 shading：近处更亮，远处更暗。
        shading = 0.65 + 0.35 * (1.0 - depth / torch.clamp(depth.max(), min=1e-6))

        # 根据深度和投影坐标生成简化高光分布 highlight 。
        highlight = self._specular_highlight(depth, correspondence, projected.shape[1])

        # 漫反射项 = 投影到相机像素的亮度 * 表面 albedo * 深度 shading。
        direct = scattered * albedo.unsqueeze(0) * shading.unsqueeze(0)

        # 相机接收到的线性亮度 = 漫反射 + 镜面高光 + 环境光
        captured = direct + specular.unsqueeze(0) * highlight.unsqueeze(0) + float(self.ambient)

        # 亮度裁剪到相机有效范围，并用 camera_gamma 模拟相机输出响应。
        captured = torch.pow(torch.clamp(captured, 0.0, 1.0), 1.0 / camera_gamma)
        if self.noise_std > 0:
            generator = torch.Generator(device=tensor_device)
            generator.manual_seed(int(self.seed))
            noise = torch.randn(captured.shape, generator=generator, device=tensor_device) * float(self.noise_std)
            captured = captured + noise
        captured = torch.clamp(captured, 0.0, 1.0)
        return {
            "captured_images": captured,
            "ground_truth_correspondence": correspondence,
            "valid_mask": valid_mask,
        }

    @staticmethod
    def _sample_projector_columns_torch(patterns, columns):
        """按相机-投影仪对应关系对一维投影图案做线性采样。"""

        import torch

        max_col = patterns.shape[1] - 1
        columns = torch.clamp(columns, 0.0, float(max_col))
        left = torch.floor(columns).long()
        right = torch.clamp(left + 1, 0, max_col)
        frac = columns - left.to(columns.dtype)
        sampled = []
        for pattern in patterns:
            # 浮点坐标，用左右列线性插值保持可导近似。
            values = pattern[left] * (1.0 - frac) + pattern[right] * frac
            sampled.append(values)
        return torch.stack(sampled, dim=0)

    def _resolve_device(self, torch, requested: str | None):
        """PyTorch 实际运行设备"""

        selected = self.device if requested is None else requested
        if selected == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return selected

    @staticmethod
    def _mix_neighbor_columns(sampled, scattering):
        """用相邻列卷积近似投影列之间的光学串扰。"""

        import torch
        import torch.nn.functional as functional

        # 1D [1, 2, 1] 核沿投影列方向模糊
        padded = functional.pad(sampled.unsqueeze(0), (1, 1, 0, 0), mode="replicate").squeeze(0)
        blurred = (padded[:, :, :-2] + 2.0 * padded[:, :, 1:-1] + padded[:, :, 2:]) * 0.25
        amount = torch.clamp(scattering, 0.0, 0.75).unsqueeze(0)
        return sampled * (1.0 - amount) + blurred * amount

    @staticmethod
    def _specular_highlight(depth, correspondence, projector_width: int):
        """根据深度和投影坐标生成一个简化的镜面高光分布。"""

        import torch

        x = correspondence / max(float(projector_width - 1), 1.0)
        normalized_depth = depth / torch.clamp(depth.max(), min=1e-6)
        ridge = torch.exp(-((x - 0.68) ** 2) / 0.018)  # 一维 Gaussian-like 高光带假设
        facing = torch.clamp(1.15 - normalized_depth, 0.0, 1.0) 
        return torch.clamp(ridge * facing, 0.0, 1.0)
