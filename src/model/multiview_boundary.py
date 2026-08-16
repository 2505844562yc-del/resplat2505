"""Geometry-aware confidence for semantic boundaries across camera views."""

import torch
import torch.nn.functional as F
from torch import Tensor


def signed_boundary_consensus_feature(
    boundary: Tensor, confidence: Tensor, consensus: Tensor
) -> Tensor:
    """Encode supported boundaries as positive and unsupported ones as negative."""
    if confidence.shape != boundary.shape or consensus.shape != boundary.shape:
        raise ValueError("boundary, confidence, and consensus must have equal shapes")
    return (2.0 * consensus - 1.0) * boundary * confidence


def boundary_consensus_reliability_feature(
    boundary: Tensor, confidence: Tensor, consensus: Tensor
) -> Tensor:
    """Return non-negative cross-view reliability at teacher boundaries."""
    if confidence.shape != boundary.shape or consensus.shape != boundary.shape:
        raise ValueError("boundary, confidence, and consensus must have equal shapes")
    return consensus * boundary * confidence


def _validate_inputs(
    boundary: Tensor,
    confidence: Tensor,
    depth: Tensor,
    extrinsics: Tensor,
    intrinsics: Tensor,
) -> tuple[int, int, int, int]:
    if boundary.ndim != 5 or boundary.shape[2] != 1:
        raise ValueError("boundary must have shape [B, V, 1, H, W]")
    if confidence.shape != boundary.shape:
        raise ValueError("confidence must match boundary")
    if depth.ndim == 5 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if depth.shape != (boundary.shape[0], boundary.shape[1], *boundary.shape[-2:]):
        raise ValueError("depth must have shape [B, V, H, W]")
    if extrinsics.shape != (boundary.shape[0], boundary.shape[1], 4, 4):
        raise ValueError("extrinsics must have shape [B, V, 4, 4]")
    if intrinsics.shape != (boundary.shape[0], boundary.shape[1], 3, 3):
        raise ValueError("intrinsics must have shape [B, V, 3, 3]")
    return boundary.shape[0], boundary.shape[1], boundary.shape[-2], boundary.shape[-1]


def multiview_boundary_consensus_confidence(
    boundary: Tensor,
    confidence: Tensor,
    depth: Tensor,
    extrinsics: Tensor,
    intrinsics: Tensor,
    radius: int = 2,
    depth_relative_tolerance: float = 0.05,
    min_support_views: float = 1.0,
    blend: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Return confidence gated by depth-aware cross-view boundary agreement.

    Camera extrinsics are camera-to-world and intrinsics use normalized image
    coordinates. Only positive teacher boundaries are attenuated. Confidence on
    non-boundary pixels is preserved so false rendered edges remain penalized.

    Returns ``(effective_confidence, consensus)`` with input boundary shape.
    Consensus is detached by callers when it is used as an uncertainty gate.
    """
    batch, views, height, width = _validate_inputs(
        boundary, confidence, depth, extrinsics, intrinsics
    )
    if views < 2:
        return confidence, torch.ones_like(confidence)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if depth_relative_tolerance <= 0:
        raise ValueError("depth_relative_tolerance must be positive")
    if min_support_views <= 0:
        raise ValueError("min_support_views must be positive")
    if not 0 <= blend <= 1:
        raise ValueError("blend must be in [0, 1]")

    if depth.ndim == 5:
        depth = depth[:, :, 0]
    dtype = depth.dtype
    device = depth.device
    source_boundary = boundary.to(dtype)
    source_confidence = confidence.to(dtype)
    target_boundary = F.max_pool2d(
        source_boundary.flatten(0, 1),
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    ).reshape(batch, views, 1, height, width)

    x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    pixels = torch.stack((grid_x, grid_y, torch.ones_like(grid_x)), dim=-1)
    pixels = pixels.reshape(1, height * width, 3).expand(batch, -1, -1)

    support_sum = torch.zeros_like(source_boundary)
    valid_sum = torch.zeros_like(source_boundary)
    world_to_camera = torch.linalg.inv(extrinsics.to(dtype))

    for source_index in range(views):
        source_depth = depth[:, source_index].reshape(batch, height * width, 1)
        source_rays = pixels @ torch.linalg.inv(
            intrinsics[:, source_index].to(dtype)
        ).transpose(-1, -2)
        source_camera = source_rays * source_depth
        source_h = torch.cat(
            (source_camera, torch.ones_like(source_camera[..., :1])), dim=-1
        )
        source_world = source_h @ extrinsics[:, source_index].to(dtype).transpose(-1, -2)

        source_support = torch.zeros(batch, 1, height, width, device=device, dtype=dtype)
        source_valid = torch.zeros_like(source_support)
        for target_index in range(views):
            if target_index == source_index:
                continue
            target_camera = source_world @ world_to_camera[:, target_index].transpose(-1, -2)
            target_xyz = target_camera[..., :3]
            target_z = target_xyz[..., 2:3]
            normalized_xy = target_xyz[..., :2] / target_z.clamp_min(1e-6)
            normalized_h = torch.cat(
                (normalized_xy, torch.ones_like(target_z)), dim=-1
            )
            projected = normalized_h @ intrinsics[:, target_index].to(dtype).transpose(-1, -2)
            sample_grid = projected[..., :2].reshape(batch, height, width, 2) * 2 - 1

            inside = (
                (target_z[..., 0] > 0)
                & (projected[..., 0] >= 0)
                & (projected[..., 0] <= 1)
                & (projected[..., 1] >= 0)
                & (projected[..., 1] <= 1)
                & (source_depth[..., 0] > 0)
            ).reshape(batch, 1, height, width)
            sampled_boundary = F.grid_sample(
                target_boundary[:, target_index], sample_grid,
                mode="bilinear", padding_mode="zeros", align_corners=False,
            )
            sampled_confidence = F.grid_sample(
                source_confidence[:, target_index], sample_grid,
                mode="bilinear", padding_mode="zeros", align_corners=False,
            )
            sampled_depth = F.grid_sample(
                depth[:, target_index].unsqueeze(1), sample_grid,
                mode="bilinear", padding_mode="zeros", align_corners=False,
            )
            projected_depth = target_z.reshape(batch, 1, height, width)
            depth_scale = torch.maximum(
                sampled_depth.abs(), projected_depth.abs()
            ).clamp_min(1e-3)
            relative_error = (sampled_depth - projected_depth).abs() / depth_scale
            depth_agreement = torch.exp(-relative_error / depth_relative_tolerance)
            valid = inside & (sampled_depth > 0)
            source_support += (
                sampled_boundary * sampled_confidence * depth_agreement * valid
            )
            source_valid += valid.to(dtype)

        support_sum[:, source_index] = source_support
        valid_sum[:, source_index] = source_valid

    available_support = torch.minimum(
        valid_sum, torch.full_like(valid_sum, float(min_support_views))
    )
    consensus = torch.where(
        available_support > 0,
        support_sum / available_support.clamp_min(1e-6),
        torch.zeros_like(support_sum),
    ).clamp(0, 1)
    positive = source_boundary >= 0.5
    positive_float = positive.to(consensus.dtype)
    mean_consensus = (consensus * positive_float).sum(
        dim=(1, 2, 3, 4), keepdim=True
    ) / positive_float.sum(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1.0)
    # Redistribute a fixed semantic budget instead of globally weakening it:
    # supported boundaries are strengthened and unsupported ones are reduced.
    normalized_consensus = consensus / mean_consensus.clamp_min(1e-6)
    positive_scale = ((1.0 - blend) + blend * normalized_consensus).clamp(0, 2)
    effective = torch.where(
        positive, source_confidence * positive_scale, source_confidence
    )
    return effective.to(confidence.dtype), consensus.to(confidence.dtype)
