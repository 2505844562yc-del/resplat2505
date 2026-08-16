"""Semantic structure inputs for Version-2 Gaussian initialization."""

import torch
import torch.nn.functional as F
from torch import nn
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


class SemanticGeometryAdapter(nn.Module):
    """Zero-initialized semantic residual adapter for initializer features."""

    def __init__(
        self,
        feature_channels: int,
        hidden_channels: int = 32,
        gate_bias: float = -2.0,
    ) -> None:
        super().__init__()
        if feature_channels < 1 or hidden_channels < 1:
            raise ValueError("adapter channel counts must be positive")
        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, feature_channels, 3, padding=1),
            nn.GELU(),
        )
        self.projection = nn.Conv2d(feature_channels, feature_channels, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(2 * feature_channels, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, gate_bias)

    def forward(
        self, base_features: Tensor, semantic_features: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if base_features.ndim != 4:
            raise ValueError("base_features must have shape [BV, C, H, W]")
        if semantic_features.ndim != 4 or semantic_features.shape[1] != 3:
            raise ValueError("semantic_features must have shape [BV, 3, H, W]")
        semantic_features = F.interpolate(
            semantic_features,
            size=base_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        encoded = self.encoder(semantic_features.to(base_features.dtype))
        gate = torch.sigmoid(self.gate(torch.cat((base_features, encoded), dim=1)))
        residual = self.projection(encoded)
        return base_features + gate * residual, gate, residual
