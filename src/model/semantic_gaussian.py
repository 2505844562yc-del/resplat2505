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


def normalize_gaussian_visibility_support(
    support_gradient: Tensor,
    relative_scale: float = 4.0,
    eps: float = 1e-8,
) -> Tensor:
    """Normalize per-Gaussian alpha contribution into a bounded support score."""
    if support_gradient.ndim == 2:
        support_gradient = support_gradient.unsqueeze(-1)
    if support_gradient.ndim != 3 or support_gradient.shape[-1] != 1:
        raise ValueError("support_gradient must have shape [B, G] or [B, G, 1]")
    if relative_scale <= 0:
        raise ValueError("relative_scale must be positive")
    support = support_gradient.float().abs()
    mean_support = support.mean(dim=1, keepdim=True).clamp_min(eps)
    return (support / (relative_scale * mean_support)).clamp(0, 1)


def semantic_uncertainty_components(
    semantic_feedback: Tensor,
    visibility_support: Tensor,
    need_floor: float = 0.25,
    reliability_floor: float = 0.05,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build need, reliability, uncertainty, and correction-priority signals.

    Semantic VJP magnitude says *where* a correction is needed.  Alpha-support
    VJP says whether that Gaussian was sufficiently visible for the observation
    to be trusted.  Keeping these concepts separate avoids treating every
    non-zero semantic gradient as equally reliable.
    """
    if semantic_feedback.ndim != 3 or semantic_feedback.shape[-1] < 3:
        raise ValueError("semantic_feedback must have shape [B, G, D + 2]")
    expected_shape = semantic_feedback.shape[:2] + (1,)
    if visibility_support.shape != expected_shape:
        raise ValueError("visibility_support must have shape [B, G, 1]")
    if not 0 <= need_floor < 1:
        raise ValueError("need_floor must satisfy 0 <= value < 1")
    if not 0 <= reliability_floor < 1:
        raise ValueError("reliability_floor must satisfy 0 <= value < 1")

    relative_magnitude = semantic_feedback[..., -2:-1].float().clamp(0, 1)
    active = semantic_feedback[..., -1:].float().clamp(0, 1)
    need = (
        (relative_magnitude - need_floor) / (1 - need_floor)
    ).clamp(0, 1) * active
    reliability = (
        (visibility_support.float() - reliability_floor)
        / (1 - reliability_floor)
    ).clamp(0, 1) * active
    priority = need * reliability

    # Uncertainty requires observed semantic mismatch. Poor reliability can
    # amplify an existing mismatch, but must not label every invisible Gaussian
    # as uncertain when there is no semantic evidence for it.
    uncertainty = need * (2.0 - reliability)
    return need, reliability, uncertainty.clamp(0, 1), priority


def apply_semantic_uncertainty_residual(
    base_uncertainty: Tensor,
    raw_residual: Tensor,
    reliability: Tensor,
    gain: float = 0.25,
) -> Tensor:
    """Apply a bounded learned calibration to deterministic uncertainty."""
    if base_uncertainty.shape != raw_residual.shape:
        raise ValueError("base_uncertainty and raw_residual must share shape")
    if reliability.shape != base_uncertainty.shape:
        raise ValueError("reliability must match base_uncertainty")
    if base_uncertainty.ndim != 3 or base_uncertainty.shape[-1] != 1:
        raise ValueError("uncertainty tensors must have shape [B, G, 1]")
    if gain < 0:
        raise ValueError("gain must be non-negative")
    return (
        base_uncertainty.float()
        + gain * reliability.float() * torch.tanh(raw_residual.float())
    ).clamp(0, 1)


def apply_semantic_support_residual(
    scales: Tensor,
    opacity_logits: Tensor,
    raw_residual: Tensor,
    priority: Tensor,
    opacity_gain: float = 0.1,
    scale_gain: float = 0.05,
    min_scale: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Apply bounded semantic-conditioned opacity and log-scale corrections.

    The semantic priority selects where an update is allowed.  The learned head,
    supervised through RGB/perceptual rendering losses, chooses its sign.
    Returns updated scales/logits and the applied log-scale/opacity residuals.
    """
    if scales.ndim != 3 or scales.shape[-1] != 3:
        raise ValueError("scales must have shape [B, G, 3]")
    expected_scalar_shape = scales.shape[:2] + (1,)
    if opacity_logits.shape != expected_scalar_shape:
        raise ValueError("opacity_logits must have shape [B, G, 1]")
    if raw_residual.shape != scales.shape[:2] + (4,):
        raise ValueError("raw_residual must have shape [B, G, 4]")
    if priority.shape != expected_scalar_shape:
        raise ValueError("priority must have shape [B, G, 1]")
    if opacity_gain < 0 or scale_gain < 0:
        raise ValueError("semantic support gains must be non-negative")
    if min_scale <= 0:
        raise ValueError("min_scale must be positive")

    gate = priority.float().clamp(0, 1)
    opacity_delta = opacity_gain * torch.tanh(raw_residual[..., :1].float()) * gate
    log_scale_delta = (
        scale_gain * torch.tanh(raw_residual[..., 1:].float()) * gate
    )
    updated_logits = opacity_logits.float() + opacity_delta
    updated_scales = scales.float() * torch.exp(log_scale_delta)
    return (
        updated_scales.clamp_min(min_scale),
        updated_logits,
        log_scale_delta,
        opacity_delta,
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


def apply_learned_ray_depth_residual(
    means: Tensor,
    scales: Tensor,
    ray_directions: Tensor,
    raw_depth_residual: Tensor,
    confidence: Tensor,
    gain: float,
    confidence_floor: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Apply a learned, confidence-gated displacement along source rays.

    Returns the updated means and the signed world-space depth displacement.
    """
    if means.ndim != 3 or means.shape[-1] != 3 or scales.shape != means.shape:
        raise ValueError("means and scales must share shape [B, G, 3]")
    if ray_directions.shape != means.shape:
        raise ValueError("ray_directions must match means")
    expected_scalar_shape = means.shape[:-1] + (1,)
    if raw_depth_residual.shape != expected_scalar_shape:
        raise ValueError("raw_depth_residual must have shape [B, G, 1]")
    if confidence.shape != expected_scalar_shape:
        raise ValueError("confidence must have shape [B, G, 1]")
    if gain < 0:
        raise ValueError("gain must be non-negative")
    if not 0 <= confidence_floor < 1:
        raise ValueError("confidence_floor must satisfy 0 <= value < 1")

    rays = F.normalize(ray_directions.float(), dim=-1, eps=1e-8)
    confidence_gate = (
        (confidence.float() - confidence_floor) / (1 - confidence_floor)
    ).clamp(0, 1)
    local_scale = scales.float().mean(dim=-1, keepdim=True)
    depth_delta = (
        gain
        * local_scale
        * torch.tanh(raw_depth_residual.float())
        * confidence_gate
    )
    return means.float() + depth_delta * rays, depth_delta


def apply_priority_gated_ray_depth_residual(
    means: Tensor,
    scales: Tensor,
    ray_directions: Tensor,
    raw_depth_residual: Tensor,
    priority: Tensor,
    gain: float = 0.1,
) -> tuple[Tensor, Tensor]:
    """Move Gaussians along source rays only where semantic evidence is trusted.

    ``priority`` is the Stage-4 product of semantic correction need and
    visibility reliability.  The displacement is expressed relative to each
    Gaussian's local scale, bounded by ``tanh``, and exactly zero when either
    the head output or priority is zero.
    """
    if means.ndim != 3 or means.shape[-1] != 3 or scales.shape != means.shape:
        raise ValueError("means and scales must share shape [B, G, 3]")
    if ray_directions.shape != means.shape:
        raise ValueError("ray_directions must match means")
    expected_scalar_shape = means.shape[:-1] + (1,)
    if raw_depth_residual.shape != expected_scalar_shape:
        raise ValueError("raw_depth_residual must have shape [B, G, 1]")
    if priority.shape != expected_scalar_shape:
        raise ValueError("priority must have shape [B, G, 1]")
    if gain < 0:
        raise ValueError("gain must be non-negative")

    rays = F.normalize(ray_directions.float(), dim=-1, eps=1e-8)
    local_scale = scales.float().mean(dim=-1, keepdim=True)
    depth_delta = (
        gain
        * local_scale
        * torch.tanh(raw_depth_residual.float())
        * priority.float().clamp(0, 1)
    )
    return means.float() + depth_delta * rays, depth_delta
