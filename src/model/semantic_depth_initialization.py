import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_depth_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Return per-pixel categorical entropy in [0, 1]."""
    if logits.ndim != 4:
        raise ValueError("depth logits must have shape [N, D, H, W]")
    if logits.shape[1] < 2:
        raise ValueError("depth logits must contain at least two candidates")

    probabilities = logits.float().softmax(dim=1)
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(
        dim=1, keepdim=True
    )
    return entropy / math.log(logits.shape[1])


class SemanticDepthLogitAdapter(nn.Module):
    """Predict a bounded semantic residual for the pre-Gaussian depth logits.

    The residual head is zero initialized, so enabling this module is an exact
    identity until its parameters are trained.
    """

    def __init__(
        self,
        semantic_channels: int,
        depth_feature_channels: int,
        num_depth_candidates: int,
        hidden_channels: int = 64,
        max_logit_residual: float = 0.25,
        confidence_floor: float = 0.05,
        gate_bias: float = -2.0,
    ) -> None:
        super().__init__()
        if min(
            semantic_channels,
            depth_feature_channels,
            num_depth_candidates,
            hidden_channels,
        ) <= 0:
            raise ValueError("all channel counts must be positive")
        if max_logit_residual < 0:
            raise ValueError("max_logit_residual must be non-negative")
        if not 0 <= confidence_floor <= 1:
            raise ValueError("confidence_floor must lie in [0, 1]")

        semantic_hidden = hidden_channels // 2
        depth_hidden = hidden_channels - semantic_hidden
        if semantic_hidden == 0:
            raise ValueError("hidden_channels must be at least 2")

        self.max_logit_residual = float(max_logit_residual)
        self.confidence_floor = float(confidence_floor)
        self.semantic_projection = nn.Sequential(
            nn.Conv2d(semantic_channels, semantic_hidden, 1),
            nn.GroupNorm(1, semantic_hidden),
            nn.GELU(),
        )
        self.depth_projection = nn.Sequential(
            nn.Conv2d(depth_feature_channels, depth_hidden, 1),
            nn.GroupNorm(1, depth_hidden),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
        )
        self.residual_head = nn.Conv2d(
            hidden_channels, num_depth_candidates, 3, padding=1
        )
        self.gate_head = nn.Conv2d(hidden_channels, 1, 3, padding=1)

        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, gate_bias)

    def forward(
        self,
        base_logits: torch.Tensor,
        depth_features: torch.Tensor,
        semantic_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if base_logits.ndim != 4 or depth_features.ndim != 4:
            raise ValueError("base logits and depth features must be 4D tensors")
        if semantic_features.ndim != 4:
            raise ValueError("semantic features must be a 4D tensor")
        if base_logits.shape[0] != depth_features.shape[0]:
            raise ValueError("base logits and depth features must share a batch size")
        if base_logits.shape[-2:] != depth_features.shape[-2:]:
            raise ValueError("base logits and depth features must share a resolution")
        if semantic_features.shape[0] != base_logits.shape[0]:
            raise ValueError("semantic features and depth logits must share a batch size")
        if base_logits.shape[1] != self.residual_head.out_channels:
            raise ValueError("unexpected number of depth candidates")

        semantic_features = F.interpolate(
            semantic_features.float(),
            size=base_logits.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        encoded = self.fusion(
            torch.cat(
                (
                    self.semantic_projection(semantic_features),
                    self.depth_projection(depth_features.float()),
                ),
                dim=1,
            )
        )

        residual = self.max_logit_residual * torch.tanh(
            self.residual_head(encoded)
        )
        depth_confidence = 1.0 - normalized_depth_entropy(base_logits).detach()
        reliability = self.confidence_floor + (
            1.0 - self.confidence_floor
        ) * depth_confidence
        gate = torch.sigmoid(self.gate_head(encoded)) * reliability
        fused_logits = base_logits + gate.to(base_logits.dtype) * residual.to(
            base_logits.dtype
        )
        return fused_logits, gate, residual, encoded
