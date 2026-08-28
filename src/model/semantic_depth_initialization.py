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


def boundary_weighted_semantic_feature_loss(
    cosine_similarity: torch.Tensor,
    alpha: torch.Tensor,
    boundary: torch.Tensor,
    confidence: torch.Tensor,
    boundary_weight: float,
    confidence_floor: float = 0.5,
    alpha_floor: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Emphasize trusted semantic boundaries without changing loss scale.

    Inputs use image layout [B, V, H, W]. Boundary and confidence may include
    an additional singleton channel. The returned loss is normalized by the
    applied weights, so enabling boundary emphasis does not simply multiply the
    global semantic feature loss.
    """
    if cosine_similarity.ndim != 4 or alpha.shape != cosine_similarity.shape:
        raise ValueError("cosine similarity and alpha must share [B, V, H, W]")
    if boundary.ndim == 5 and boundary.shape[2] == 1:
        boundary = boundary.squeeze(2)
    if confidence.ndim == 5 and confidence.shape[2] == 1:
        confidence = confidence.squeeze(2)
    if boundary.shape != cosine_similarity.shape:
        raise ValueError("boundary must align with the semantic render")
    if confidence.shape != cosine_similarity.shape:
        raise ValueError("boundary confidence must align with the semantic render")
    if boundary_weight < 0:
        raise ValueError("boundary_weight must be non-negative")
    if not 0 <= confidence_floor <= 1:
        raise ValueError("confidence_floor must lie in [0, 1]")
    if not 0 <= alpha_floor <= 1:
        raise ValueError("alpha_floor must lie in [0, 1]")

    cosine_similarity = cosine_similarity.float()
    alpha = alpha.float()
    boundary = boundary.float().clamp(0, 1)
    confidence = confidence.float().clamp(0, 1)
    trusted_boundary = torch.where(
        confidence >= confidence_floor,
        boundary * confidence,
        torch.zeros_like(boundary),
    )
    weights = 1.0 + boundary_weight * trusted_boundary
    valid = alpha >= alpha_floor
    if not torch.any(valid):
        valid = torch.ones_like(valid)
    valid_weight = weights * valid.to(weights.dtype)
    semantic_error = 1.0 - cosine_similarity
    loss = (semantic_error * valid_weight).sum() / valid_weight.sum().clamp_min(1)

    trusted_valid = valid & (trusted_boundary > 0)
    if torch.any(trusted_valid):
        boundary_error = semantic_error.masked_select(trusted_valid).mean()
    else:
        boundary_error = semantic_error.new_zeros(())
    return loss, weights, trusted_boundary, boundary_error


class SemanticDepthFeatureAdapter(nn.Module):
    """Apply a bounded semantic residual before the pretrained depth head.

    The residual head is zero initialized, making the adapter an exact identity
    at construction. Residual magnitude is expressed relative to the local RMS
    of the frozen depth feature, which avoids a scene-dependent absolute scale.
    """

    def __init__(
        self,
        semantic_channels: int,
        depth_feature_channels: int,
        hidden_channels: int = 64,
        max_relative_residual: float = 0.1,
        confidence_floor: float = 0.05,
        gate_bias: float = -2.0,
    ) -> None:
        super().__init__()
        if min(semantic_channels, depth_feature_channels, hidden_channels) <= 0:
            raise ValueError("all channel counts must be positive")
        if hidden_channels < 2:
            raise ValueError("hidden_channels must be at least 2")
        if max_relative_residual < 0:
            raise ValueError("max_relative_residual must be non-negative")
        if not 0 <= confidence_floor <= 1:
            raise ValueError("confidence_floor must lie in [0, 1]")

        semantic_hidden = hidden_channels // 2
        depth_hidden = hidden_channels - semantic_hidden
        self.max_relative_residual = float(max_relative_residual)
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
            hidden_channels, depth_feature_channels, 3, padding=1
        )
        self.gate_head = nn.Conv2d(hidden_channels, 1, 3, padding=1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, gate_bias)

    def forward(
        self,
        depth_features: torch.Tensor,
        semantic_features: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if depth_features.ndim != 4 or semantic_features.ndim != 4:
            raise ValueError("depth and semantic features must be 4D tensors")
        if base_logits.ndim != 4:
            raise ValueError("base logits must be a 4D tensor")
        if depth_features.shape[0] != base_logits.shape[0]:
            raise ValueError("depth features and logits must share a batch size")
        if depth_features.shape[-2:] != base_logits.shape[-2:]:
            raise ValueError("depth features and logits must share a resolution")
        if semantic_features.shape[0] != depth_features.shape[0]:
            raise ValueError("semantic and depth features must share a batch size")
        if depth_features.shape[1] != self.residual_head.out_channels:
            raise ValueError("unexpected number of depth feature channels")

        semantic_features = F.interpolate(
            semantic_features.float(),
            size=depth_features.shape[-2:],
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
        local_scale = (
            depth_features.detach().float().square().mean(dim=1, keepdim=True)
            .sqrt().clamp_min(1e-3)
        )
        residual = (
            self.max_relative_residual
            * local_scale
            * torch.tanh(self.residual_head(encoded))
        )
        depth_confidence = 1.0 - normalized_depth_entropy(base_logits).detach()
        reliability = self.confidence_floor + (
            1.0 - self.confidence_floor
        ) * depth_confidence
        gate = torch.sigmoid(self.gate_head(encoded)) * reliability
        conditioned = depth_features + gate.to(depth_features.dtype) * residual.to(
            depth_features.dtype
        )
        return conditioned, gate, residual


class SemanticDepthResidualInjection(nn.Module):
    """Screenshot Situation A: ``F_depth' = F_c + gamma * A(F_s)``.

    ``F_c`` is the frozen ReSplat depth-regressor feature and ``F_s`` is the
    shared low-dimensional semantic field. The learnable channel-wise gate
    ``gamma`` is initialized to exactly zero, so loading a pretrained ReSplat
    checkpoint and enabling this module cannot change its initial depth
    prediction. Channel-wise gating avoids destructive gradient cancellation
    between unrelated depth-feature channels.

    The semantic adapter itself is deliberately *not* zero initialized.  This
    lets ``gamma`` receive a gradient on the first optimization step; once
    ``gamma`` moves away from zero, gradients also reach the semantic adapter.
    """

    def __init__(
        self,
        semantic_channels: int,
        depth_feature_channels: int,
        hidden_channels: int = 64,
        max_relative_residual: float = 0.1,
    ) -> None:
        super().__init__()
        if min(semantic_channels, depth_feature_channels, hidden_channels) <= 0:
            raise ValueError("all channel counts must be positive")
        if max_relative_residual < 0:
            raise ValueError("max_relative_residual must be non-negative")

        self.depth_feature_channels = depth_feature_channels
        self.max_relative_residual = float(max_relative_residual)
        self.semantic_adapter = nn.Sequential(
            nn.Conv2d(semantic_channels, hidden_channels, 1),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, depth_feature_channels, 3, padding=1),
        )
        self.gamma = nn.Parameter(
            torch.zeros(1, depth_feature_channels, 1, 1)
        )

    def forward(
        self,
        depth_features: torch.Tensor,
        semantic_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if depth_features.ndim != 4 or semantic_features.ndim != 4:
            raise ValueError("depth and semantic features must be 4D tensors")
        if depth_features.shape[0] != semantic_features.shape[0]:
            raise ValueError("semantic and depth features must share a batch size")
        if depth_features.shape[1] != self.depth_feature_channels:
            raise ValueError("unexpected number of depth feature channels")

        semantic_features = F.interpolate(
            semantic_features.float(),
            size=depth_features.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        adapter_output = torch.tanh(self.semantic_adapter(semantic_features))
        local_scale = (
            depth_features.detach().float().square().mean(dim=1, keepdim=True)
            .sqrt()
            .clamp_min(1e-3)
        )
        gamma = torch.tanh(self.gamma)
        applied_residual = (
            gamma
            * self.max_relative_residual
            * local_scale
            * adapter_output
        )
        conditioned = depth_features + applied_residual.to(depth_features.dtype)
        return conditioned, gamma, applied_residual


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
