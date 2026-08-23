"""Semantic feature projection for semantic-carrying Gaussians."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SemanticFeatureProjector(nn.Module):
    """Deterministically compress dense teacher features without collapse."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 16,
        seed: int = 3407,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        if input_dim < output_dim or output_dim < 1:
            raise ValueError("semantic dimensions must satisfy input >= output >= 1")
        self.projection = nn.Conv2d(input_dim, output_dim, 1, bias=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            nn.init.orthogonal_(self.projection.weight.flatten(1))
        self.projection.weight.requires_grad = trainable

    def forward(self, features: Tensor, output_size: tuple[int, int]) -> Tensor:
        if features.ndim != 4 or features.shape[1] != self.projection.in_channels:
            raise ValueError("teacher features must have shape [BV, C, H, W]")
        projected = self.projection(features.float())
        projected = F.interpolate(
            projected,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        return F.normalize(projected, dim=1, eps=1e-6)


def apply_semantic_state_residual(
    features: Tensor, residual: Tensor, gain: float = 0.1
) -> Tensor:
    """Apply a bounded FP32 residual and return unit semantic embeddings."""
    if features.shape != residual.shape or features.ndim != 3:
        raise ValueError("features and residual must share shape [B, G, D]")
    if gain < 0:
        raise ValueError("semantic state residual gain must be non-negative")
    updated = features.float() + gain * torch.tanh(residual.float())
    return F.normalize(updated, dim=-1, eps=1e-6)


def semantic_render_residual_features(
    rendered: Tensor,
    teacher: Tensor,
    alpha: Tensor,
    alpha_floor: float = 0.1,
) -> Tensor:
    """Build visibility-gated semantic feedback in image space.

    Args:
        rendered: Alpha-normalized rendered features ``[B, V, D, H, W]``.
        teacher: Frozen teacher features with the same shape.
        alpha: Accumulated rendering alpha ``[B, V, H, W]``.
        alpha_floor: Visibility below this value contributes no residual.

    Returns:
        ``[B, V, D + 2, H, W]`` containing the directional feature
        residual, cosine error, and visibility confidence.
    """
    if rendered.shape != teacher.shape or rendered.ndim != 5:
        raise ValueError("rendered and teacher must share shape [B, V, D, H, W]")
    if alpha.shape != rendered.shape[:2] + rendered.shape[-2:]:
        raise ValueError("alpha must have shape [B, V, H, W]")
    if not 0 <= alpha_floor < 1:
        raise ValueError("alpha_floor must satisfy 0 <= alpha_floor < 1")

    rendered = F.normalize(rendered.float(), dim=2, eps=1e-6)
    teacher = F.normalize(teacher.float(), dim=2, eps=1e-6)
    visibility = ((alpha.float() - alpha_floor) / (1 - alpha_floor)).clamp(0, 1)
    visibility = visibility.unsqueeze(2)
    directional_residual = (teacher - rendered) * visibility
    cosine_error = (
        1.0 - (teacher * rendered).sum(dim=2, keepdim=True)
    ).clamp(0, 2) * visibility
    return torch.cat((directional_residual, cosine_error, visibility), dim=2)


def semantic_gradient_feedback_features(
    gradient: Tensor,
    relative_scale: float = 4.0,
    eps: float = 1e-8,
) -> Tensor:
    """Convert a semantic-rendering VJP into bounded per-Gaussian feedback.

    The negative normalized gradient is the local correction direction. Its
    magnitude is normalized by the per-scene mean so the representation is
    insensitive to image size and the loss reduction convention.
    """
    if gradient.ndim != 3:
        raise ValueError("gradient must have shape [B, G, D]")
    if relative_scale <= 0:
        raise ValueError("relative_scale must be positive")
    gradient = gradient.float()
    magnitude = gradient.norm(dim=-1, keepdim=True)
    mean_magnitude = magnitude.mean(dim=1, keepdim=True).clamp_min(eps)
    relative_magnitude = (magnitude / (relative_scale * mean_magnitude)).clamp(0, 1)
    direction = -gradient / magnitude.clamp_min(eps)
    direction = torch.where(magnitude > eps, direction, torch.zeros_like(direction))
    active = (magnitude > eps).to(gradient.dtype)
    return torch.cat(
        (direction * relative_magnitude, relative_magnitude, active), dim=-1
    )


def apply_semantic_geometry_vjp(
    means: Tensor,
    scales: Tensor,
    mean_feedback: Tensor | None,
    scale_feedback: Tensor | None,
    mean_gain: float = 0.0,
    scale_gain: float = 0.0,
    min_scale: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Apply bounded, scene-scale-aware semantic VJP geometry corrections."""
    if means.ndim != 3 or scales.shape != means.shape:
        raise ValueError("means and scales must share shape [B, G, 3]")
    if mean_feedback is not None and mean_feedback.shape != means.shape:
        raise ValueError("mean_feedback must match means")
    if scale_feedback is not None and scale_feedback.shape != scales.shape:
        raise ValueError("scale_feedback must match scales")
    if mean_gain < 0 or scale_gain < 0:
        raise ValueError("semantic geometry gains must be non-negative")
    if min_scale <= 0:
        raise ValueError("min_scale must be positive")

    updated_means = means.float()
    updated_scales = scales.float()
    if mean_feedback is not None and mean_gain > 0:
        local_scale = updated_scales.mean(dim=-1, keepdim=True)
        updated_means = updated_means + (
            mean_gain * local_scale * torch.tanh(mean_feedback.float())
        )
    if scale_feedback is not None and scale_gain > 0:
        log_scale_delta = scale_gain * torch.tanh(scale_feedback.float())
        updated_scales = updated_scales * torch.exp(log_scale_delta)
    return updated_means, updated_scales.clamp_min(min_scale)


def confidence_gate_semantic_gradient_feedback(
    feedback: Tensor,
    confidence_floor: float,
) -> Tensor:
    """Return the directional channels after smooth confidence thresholding.

    ``feedback`` is expected to contain directional channels followed by the
    relative gradient magnitude and active flag produced by
    :func:`semantic_gradient_feedback_features`. A zero floor preserves the
    original directional residual exactly.
    """
    if feedback.ndim != 3 or feedback.shape[-1] < 3:
        raise ValueError("feedback must have shape [B, G, D + 2]")
    if not 0 <= confidence_floor < 1:
        raise ValueError("confidence_floor must satisfy 0 <= value < 1")
    directional = feedback[..., :-2]
    confidence = feedback[..., -2:-1]
    gated_confidence = (
        (confidence - confidence_floor) / (1 - confidence_floor)
    ).clamp(0, 1)
    confidence_ratio = torch.where(
        confidence > 0,
        gated_confidence / confidence.clamp_min(1e-8),
        torch.zeros_like(confidence),
    )
    return directional * confidence_ratio


def project_geometry_feedback_to_rays(
    feedback: Tensor,
    ray_directions: Tensor,
) -> Tensor:
    """Restrict a world-space geometry correction to source-camera rays."""
    if feedback.ndim != 3 or feedback.shape[-1] != 3:
        raise ValueError("feedback must have shape [B, G, 3]")
    if ray_directions.shape != feedback.shape:
        raise ValueError("ray_directions must match feedback")
    rays = F.normalize(ray_directions.float(), dim=-1, eps=1e-8)
    return (feedback.float() * rays).sum(dim=-1, keepdim=True) * rays
