"""Spatially selective semantic residuals for Gaussian geometry updates."""

import torch
from torch import Tensor


def apply_selective_geometry_residual(
    delta_means: Tensor,
    delta_scales: Tensor,
    corrections: Tensor,
    gate: Tensor,
    active_mask: Tensor,
    gain: float = 0.5,
    reference_floor: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Add bounded, per-axis mean/scale residuals at selected Gaussian tokens.

    Corrections are expressed relative to the detached magnitude of the base
    update, which preserves parameter units without uniformly scaling the base
    update. A zero correction is an exact identity mapping.
    """
    expected_prefix = delta_means.shape[:-1]
    if delta_scales.shape[:-1] != expected_prefix:
        raise ValueError("mean and scale deltas must share their leading shape")
    if corrections.shape != (*expected_prefix, 6):
        raise ValueError("corrections must have shape [B, N, 6]")
    if gate.shape != (*expected_prefix, 1):
        raise ValueError("gate must have shape [B, N, 1]")
    if active_mask.shape != gate.shape:
        raise ValueError("active_mask must have the same shape as gate")
    if gain < 0 or reference_floor < 0:
        raise ValueError("gain and reference_floor must be non-negative")

    selection = gate.clamp(0, 1) * active_mask.to(delta_means.dtype)
    mean_reference = (
        delta_means.detach().abs().mean(dim=-1, keepdim=True) + reference_floor
    )
    scale_reference = (
        delta_scales.detach().abs().mean(dim=-1, keepdim=True) + reference_floor
    )
    mean_correction, scale_correction = corrections.split((3, 3), dim=-1)
    return (
        delta_means
        + gain * selection * torch.tanh(mean_correction) * mean_reference,
        delta_scales
        + gain * selection * torch.tanh(scale_correction) * scale_reference,
    )
