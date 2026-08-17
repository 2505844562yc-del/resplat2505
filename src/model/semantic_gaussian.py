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
