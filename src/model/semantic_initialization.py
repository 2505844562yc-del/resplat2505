"""Semantic structure inputs for Version-2 Gaussian initialization."""

import torch
import torch.nn.functional as F
from torch import Tensor


def boundary_proximity_features(
    boundary: Tensor,
    confidence: Tensor,
    confidence_floor: float = 0.5,
    max_radius: int = 4,
) -> Tensor:
    """Return boundary, trusted confidence, and finite-radius proximity.

    The proximity channel is one on a trusted boundary and decreases linearly
    with Chebyshev distance to zero outside ``max_radius``. Computing it after
    dataset augmentation guarantees pixel alignment with context RGB.
    """
    if boundary.shape != confidence.shape or boundary.shape[-3] != 1:
        raise ValueError("boundary and confidence must share shape [..., 1, H, W]")
    if not 0 <= confidence_floor <= 1:
        raise ValueError("confidence_floor must be in [0, 1]")
    if max_radius < 1:
        raise ValueError("max_radius must be positive")

    trusted = (boundary >= 0.5) & (confidence >= confidence_floor)
    flat_trusted = trusted.reshape(-1, 1, boundary.shape[-2], boundary.shape[-1])
    covered = flat_trusted
    proximity = flat_trusted.to(boundary.dtype)

    for distance in range(1, max_radius + 1):
        dilated = F.max_pool2d(
            covered.to(boundary.dtype), kernel_size=3, stride=1, padding=1
        ) > 0
        ring = dilated & ~covered
        weight = 1.0 - distance / (max_radius + 1.0)
        proximity = proximity + ring.to(boundary.dtype) * weight
        covered = dilated

    proximity = proximity.reshape_as(boundary)
    trusted_confidence = confidence * trusted.to(confidence.dtype)
    return torch.cat((boundary, trusted_confidence, proximity), dim=-3)
