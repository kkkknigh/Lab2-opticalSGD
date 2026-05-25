"""OpticalSGD training loop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optical_sgd.correspondence_decoding.decoder_protocol import (
    DecoderProtocol,
    TorchFeatureTransformProtocol,
    TrainableDecoderProtocol,
)
from optical_sgd.optimization.autograd_gradient import AutogradGradientEstimator
from optical_sgd.optimization.correspondence_losses import correspondence_mae, soft_expected_l1_loss
from optical_sgd.optimization.finite_difference_gradient import FiniteDifferenceGradientEstimator
from optical_sgd.optimization.optimizer_state import OptimizerState
from optical_sgd.pattern_generation.frequency_constraints import apply_frequency_constraint, clamp_patterns
from optical_sgd.rendering.render_result import RenderResult
from optical_sgd.rendering.renderer_protocol import DifferentiableRendererProtocol, RendererProtocol
from optical_sgd.synthetic_scene import SceneDescription


@dataclass
class OpticalSGDOptimizer:
    renderer: RendererProtocol | DifferentiableRendererProtocol
    decoder: DecoderProtocol
    scene: SceneDescription
    learning_rate: float = 0.1
    iterations: int = 8
    gradient_method: str = "finite_difference"
    finite_difference_epsilon: float = 0.03
    lowpass_fraction: float = 0.5
    temperature: float = 25.0
    decoder_learning_rate: float = 0.02

    def _estimator(self):
        if self.gradient_method == "autograd":
            return AutogradGradientEstimator(self.finite_difference_epsilon)
        return FiniteDifferenceGradientEstimator(self.finite_difference_epsilon)

    def evaluate(self, patterns: np.ndarray) -> dict[str, float | np.ndarray]:
        result = self._render_numpy(patterns)
        decoded = self.decoder.decode(result.captured_images, patterns)
        loss = soft_expected_l1_loss(
            decoded.scores,
            result.ground_truth_correspondence,
            result.valid_mask,
            self.temperature,
        )
        mae = correspondence_mae(
            decoded.predicted_correspondence,
            result.ground_truth_correspondence,
            result.valid_mask,
        )
        return {
            "loss": loss,
            "mae": mae,
            "predicted": decoded.predicted_correspondence,
            "render_result": result,
        }

    def train(self, initial_patterns: np.ndarray) -> tuple[np.ndarray, OptimizerState]:
        patterns = apply_frequency_constraint(initial_patterns, self.lowpass_fraction)
        state = OptimizerState()
        estimator = self._estimator()

        for _ in range(int(self.iterations)):
            def loss_function(candidate: np.ndarray) -> float:
                constrained = apply_frequency_constraint(clamp_patterns(candidate), self.lowpass_fraction)
                return float(self.evaluate(constrained)["loss"])

            decoder_gradient = self._decoder_parameter_gradient(patterns)
            if self.gradient_method == "autograd":
                gradient = estimator.estimate(patterns, loss_function, self._torch_soft_zncc_loss)
            else:
                gradient = self._optical_finite_difference_gradient(patterns, loss_function)
            patterns = patterns - float(self.learning_rate) * gradient
            self._apply_decoder_update(decoder_gradient)
            patterns = apply_frequency_constraint(clamp_patterns(patterns), self.lowpass_fraction)
            metrics = self.evaluate(patterns)
            state.losses.append(float(metrics["loss"]))
            state.maes.append(float(metrics["mae"]))
            state.gradient_norms.append(float(np.linalg.norm(gradient)))
            state.decoder_gradient_norms.append(float(np.linalg.norm(decoder_gradient)) if decoder_gradient.size else 0.0)
        return patterns.astype(np.float32), state

    def _optical_finite_difference_gradient(self, patterns: np.ndarray, fallback_loss_function) -> np.ndarray:
        """Estimate pattern gradients with an image-Jacobian split.

        This mirrors the OpticalSGD chain rule: perturb a projector control
        value, render plus/minus images to estimate d image / d pattern, multiply
        by d loss / d image, then add the direct decoder-codebook dependency.
        If differentiating the image loss is unavailable, it falls back to the
        older full-loss central difference so the public method remains usable.
        """

        try:
            base_result = self._render_numpy(patterns)
            image_loss_gradient = self._captured_loss_gradient(patterns, base_result.captured_images)
        except Exception:
            return self._estimator().estimate(patterns, fallback_loss_function)

        gradient = np.zeros_like(patterns, dtype=np.float32)
        epsilon = float(self.finite_difference_epsilon)
        for index in np.ndindex(patterns.shape):
            plus = np.array(patterns, copy=True)
            minus = np.array(patterns, copy=True)
            plus[index] += epsilon
            minus[index] -= epsilon
            plus = apply_frequency_constraint(clamp_patterns(plus), self.lowpass_fraction)
            minus = apply_frequency_constraint(clamp_patterns(minus), self.lowpass_fraction)

            plus_result = self._render_numpy(plus)
            minus_result = self._render_numpy(minus)
            image_jacobian_column = (plus_result.captured_images - minus_result.captured_images) / (2.0 * epsilon)
            image_path = float(np.sum(image_loss_gradient * image_jacobian_column))

            direct_plus = self._loss_with_captured(base_result.captured_images, plus, base_result)
            direct_minus = self._loss_with_captured(base_result.captured_images, minus, base_result)
            decoder_path = (direct_plus - direct_minus) / (2.0 * epsilon)
            gradient[index] = image_path + decoder_path
        return gradient

    def _render_numpy(self, patterns: np.ndarray) -> RenderResult:
        """按渲染器类型生成 NumPy 结果，Torch 后端不经过 render() 包装。"""

        if isinstance(self.renderer, DifferentiableRendererProtocol):
            torch_result = self.renderer.render_torch(patterns, self.scene)
            captured = torch_result["captured_images"].detach().cpu().numpy().astype(np.float32)
            return RenderResult(
                captured_images=captured,
                ground_truth_correspondence=self.scene.correspondence.astype(np.float32),
                valid_mask=self.scene.valid_mask.astype(bool),
                albedo=self.scene.albedo.astype(np.float32),
                depth=self.scene.depth.astype(np.float32),
            )
        if isinstance(self.renderer, RendererProtocol):
            return self.renderer.render(patterns, self.scene)
        raise TypeError("Renderer must provide render_torch() or render().")

    def _decoder_parameter_gradient(self, patterns: np.ndarray) -> np.ndarray:
        if not isinstance(self.decoder, TrainableDecoderProtocol):
            return np.zeros(0, dtype=np.float32)
        self.evaluate(patterns)
        base_parameters = self.decoder.parameter_vector()
        epsilon = max(float(self.finite_difference_epsilon) * 0.25, 1e-3)

        def parameter_loss(candidate: np.ndarray) -> float:
            self.decoder.set_parameter_vector(candidate)
            return float(self.evaluate(patterns)["loss"])

        gradient = FiniteDifferenceGradientEstimator(epsilon=epsilon).estimate(base_parameters, parameter_loss)
        self.decoder.set_parameter_vector(base_parameters)
        return gradient

    def _apply_decoder_update(self, gradient: np.ndarray) -> None:
        if gradient.size == 0:
            return
        if not isinstance(self.decoder, TrainableDecoderProtocol):
            return
        parameters = self.decoder.parameter_vector()
        updated = parameters - float(self.decoder_learning_rate) * gradient
        self.decoder.set_parameter_vector(updated)

    def _loss_with_captured(self, captured_images: np.ndarray, patterns: np.ndarray, render_result) -> float:
        decoded = self.decoder.decode(captured_images, patterns)
        return soft_expected_l1_loss(
            decoded.scores,
            render_result.ground_truth_correspondence,
            render_result.valid_mask,
            self.temperature,
        )

    def _torch_soft_zncc_loss(self, pattern_tensor):
        """Differentiable renderer + ZNCC soft correspondence loss."""

        import torch
        if not isinstance(self.renderer, DifferentiableRendererProtocol):
            raise RuntimeError("Autograd gradient requires a renderer with render_torch().")

        patterns = torch.clamp(pattern_tensor.to(self._torch_device(torch)), 0.0, 1.0)
        render_result = self.renderer.render_torch(patterns, self.scene)
        captured = render_result["captured_images"]
        ground_truth = render_result["ground_truth_correspondence"]
        valid_mask = render_result["valid_mask"]

        scores = self._torch_decoder_scores(captured, patterns)
        weights = torch.softmax(scores * float(self.temperature), dim=-1)
        columns = torch.arange(patterns.shape[1], dtype=torch.float32, device=patterns.device)
        penalty = torch.abs(columns.view(1, 1, -1) - ground_truth.unsqueeze(-1))
        loss_map = (weights * penalty).sum(dim=-1)
        return loss_map[valid_mask].mean()

    def _torch_device(self, torch):
        if hasattr(self.renderer, "_resolve_device"):
            return self.renderer._resolve_device(torch, None)
        return "cpu"

    def _captured_loss_gradient(self, patterns: np.ndarray, captured_images: np.ndarray) -> np.ndarray:
        import torch

        captured = torch.tensor(captured_images, dtype=torch.float32, requires_grad=True)
        pattern_tensor = torch.tensor(patterns, dtype=torch.float32)
        ground_truth = torch.as_tensor(self.scene.correspondence, dtype=torch.float32)
        valid_mask = torch.as_tensor(self.scene.valid_mask, dtype=torch.bool)
        scores = self._torch_decoder_scores(captured, pattern_tensor)
        weights = torch.softmax(scores * float(self.temperature), dim=-1)
        columns = torch.arange(pattern_tensor.shape[1], dtype=torch.float32)
        penalty = torch.abs(columns.view(1, 1, -1) - ground_truth.unsqueeze(-1))
        loss = (weights * penalty).sum(dim=-1)[valid_mask].mean()
        loss.backward()
        if captured.grad is None:
            raise RuntimeError("Captured-image loss gradient was not produced.")
        return captured.grad.detach().cpu().numpy().astype(np.float32)

    def _torch_decoder_scores(self, captured, patterns):
        radius = self.decoder.feature_radius
        image_features = self._torch_camera_features(captured, radius)
        projector_features = self._torch_projector_features(patterns, radius)
        if isinstance(self.decoder, TorchFeatureTransformProtocol):
            image_features, projector_features = self.decoder.transform_torch_features(
                image_features,
                projector_features,
                patterns.device,
            )
        image_features = self._torch_normalize(image_features)
        projector_features = self._torch_normalize(projector_features)
        return image_features @ projector_features.T

    @staticmethod
    def _torch_normalize(features):
        import torch

        centered = features - features.mean(dim=-1, keepdim=True)
        norm = torch.linalg.norm(centered, dim=-1, keepdim=True)
        return centered / torch.clamp(norm, min=1e-6)

    @staticmethod
    def _torch_projector_features(patterns, radius: int):
        if radius == 0:
            return patterns.T
        padded = functional_pad_1d(patterns, radius)
        width = patterns.shape[1]
        features = []
        for col in range(width):
            patch = padded[:, col : col + 2 * radius + 1]
            features.append(patch.reshape(-1))
        return torch_stack_column(features)

    @staticmethod
    def _torch_camera_features(captured, radius: int):
        if radius == 0:
            return captured.permute(1, 2, 0)
        padded = functional_pad_2d_width(captured, radius)
        _, height, width = captured.shape
        features = []
        for col in range(width):
            patch = padded[:, :, col : col + 2 * radius + 1]
            features.append(patch.permute(1, 0, 2).reshape(height, -1))
        return torch_stack_width(features)


def functional_pad_1d(patterns, radius: int):
    import torch.nn.functional as functional

    return functional.pad(patterns.unsqueeze(0), (radius, radius), mode="replicate").squeeze(0)


def functional_pad_2d_width(captured, radius: int):
    import torch.nn.functional as functional

    return functional.pad(captured.unsqueeze(0), (radius, radius, 0, 0), mode="replicate").squeeze(0)


def torch_stack_width(features):
    import torch

    return torch.stack(features, dim=1)


def torch_stack_column(features):
    import torch

    return torch.stack(features, dim=0)
