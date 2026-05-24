"""PyTorch structured-light renderer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optical_sgd.rendering.render_result import RenderResult
from optical_sgd.synthetic_scene.scene_description import SceneDescription


@dataclass
class TorchRenderer:
    noise_std: float = 0.01
    ambient: float = 0.02
    seed: int = 7
    device: str = "auto"

    def render(self, patterns: np.ndarray, scene: SceneDescription) -> RenderResult:
        """Render with PyTorch and return NumPy arrays for downstream modules."""

        patterns = np.asarray(patterns, dtype=np.float32)
        torch_result = self.render_torch(patterns, scene)
        captured = torch_result["captured_images"].detach().cpu().numpy()
        return RenderResult(
            captured_images=captured,
            ground_truth_correspondence=scene.correspondence.astype(np.float32),
            valid_mask=scene.valid_mask.astype(bool),
            albedo=scene.albedo.astype(np.float32),
            depth=scene.depth.astype(np.float32),
        )

    def render_torch(self, patterns, scene: SceneDescription, device: str | None = None) -> dict:
        """Render with torch operations so loss can backpropagate to patterns."""

        try:
            import torch
        except Exception as exc:
            raise ImportError("TorchRenderer.render_torch requires PyTorch.") from exc

        resolved_device = self._resolve_device(torch, device)
        if not torch.is_tensor(patterns):
            patterns = torch.as_tensor(patterns, dtype=torch.float32, device=resolved_device)
        else:
            patterns = patterns.to(dtype=torch.float32)
            patterns = patterns.to(resolved_device)

        tensor_device = patterns.device
        correspondence = torch.as_tensor(scene.correspondence, dtype=torch.float32, device=tensor_device)
        albedo = torch.as_tensor(scene.albedo, dtype=torch.float32, device=tensor_device)
        specular = torch.as_tensor(scene.specular, dtype=torch.float32, device=tensor_device)
        scattering = torch.as_tensor(scene.scattering, dtype=torch.float32, device=tensor_device)
        depth = torch.as_tensor(scene.depth, dtype=torch.float32, device=tensor_device)
        valid_mask = torch.as_tensor(scene.valid_mask, dtype=torch.bool, device=tensor_device)

        projector_gamma = max(float(scene.projector_gamma), 1e-4)
        camera_gamma = max(float(scene.camera_gamma), 1e-4)
        projected = torch.pow(torch.clamp(patterns, 0.0, 1.0), projector_gamma)
        sampled = self._sample_projector_columns_torch(projected, correspondence)
        scattered = self._mix_neighbor_columns(sampled, scattering)
        shading = 0.65 + 0.35 * (1.0 - depth / torch.clamp(depth.max(), min=1e-6))
        highlight = self._specular_highlight(depth, correspondence, projected.shape[1])
        direct = scattered * albedo.unsqueeze(0) * shading.unsqueeze(0)
        captured = direct + specular.unsqueeze(0) * highlight.unsqueeze(0) + float(self.ambient)
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
            "albedo": albedo,
            "depth": depth,
        }

    @staticmethod
    def _sample_projector_columns_torch(patterns, columns):
        import torch

        max_col = patterns.shape[1] - 1
        columns = torch.clamp(columns, 0.0, float(max_col))
        left = torch.floor(columns).long()
        right = torch.clamp(left + 1, 0, max_col)
        frac = columns - left.to(columns.dtype)
        sampled = []
        for pattern in patterns:
            values = pattern[left] * (1.0 - frac) + pattern[right] * frac
            sampled.append(values)
        return torch.stack(sampled, dim=0)

    def _resolve_device(self, torch, requested: str | None):
        selected = self.device if requested is None else requested
        if selected == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return selected

    @staticmethod
    def _mix_neighbor_columns(sampled, scattering):
        import torch
        import torch.nn.functional as functional

        padded = functional.pad(sampled.unsqueeze(0), (1, 1, 0, 0), mode="replicate").squeeze(0)
        blurred = (padded[:, :, :-2] + 2.0 * padded[:, :, 1:-1] + padded[:, :, 2:]) * 0.25
        amount = torch.clamp(scattering, 0.0, 0.75).unsqueeze(0)
        return sampled * (1.0 - amount) + blurred * amount

    @staticmethod
    def _specular_highlight(depth, correspondence, projector_width: int):
        import torch

        x = correspondence / max(float(projector_width - 1), 1.0)
        normalized_depth = depth / torch.clamp(depth.max(), min=1e-6)
        ridge = torch.exp(-((x - 0.68) ** 2) / 0.018)
        facing = torch.clamp(1.15 - normalized_depth, 0.0, 1.0)
        return torch.clamp(ridge * facing, 0.0, 1.0)
