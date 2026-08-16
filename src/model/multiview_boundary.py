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


def semantic_boundary_source_mask(
    rendered_boundary: Tensor,
    teacher_boundary: Tensor,
    teacher_confidence: Tensor,
    radius: int = 4,
    confidence_floor: float = 0.5,
    rendered_threshold: float = 0.1,
) -> Tensor:
    """Keep rendered edges only near trusted same-view semantic boundaries."""
    if not (
        rendered_boundary.shape
        == teacher_boundary.shape
        == teacher_confidence.shape
    ):
        raise ValueError("rendered and teacher boundary tensors must match")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    trusted = teacher_boundary * (teacher_confidence >= confidence_floor)
    nearby = F.max_pool2d(
        trusted.flatten(0, 1),
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    ).reshape_as(trusted)
    return rendered_boundary * (rendered_boundary >= rendered_threshold) * (nearby > 0)


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


def _nearest_boundary_offset_maps(
    boundary: Tensor, confidence: Tensor, radius: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Return nearest trusted-boundary dx, dy, and confidence for every pixel."""
    batch, _, height, width = boundary.shape
    reliable = boundary * confidence
    padded = F.pad(reliable, (radius, radius, radius, radius))
    best_distance = torch.full_like(reliable, float("inf"))
    best_confidence = torch.zeros_like(reliable)
    best_dx = torch.zeros_like(reliable)
    best_dy = torch.zeros_like(reliable)
    for offset_y in range(-radius, radius + 1):
        for offset_x in range(-radius, radius + 1):
            candidate = padded[
                ...,
                radius + offset_y : radius + offset_y + height,
                radius + offset_x : radius + offset_x + width,
            ]
            distance = float(offset_x * offset_x + offset_y * offset_y)
            closer = (candidate > 0) & (
                (distance < best_distance)
                | ((distance == best_distance) & (candidate > best_confidence))
            )
            best_distance = torch.where(
                closer, torch.full_like(best_distance, distance), best_distance
            )
            best_confidence = torch.where(closer, candidate, best_confidence)
            best_dx = torch.where(
                closer, torch.full_like(best_dx, float(offset_x)), best_dx
            )
            best_dy = torch.where(
                closer, torch.full_like(best_dy, float(offset_y)), best_dy
            )
    return best_dx, best_dy, best_confidence


def multiview_boundary_displacement_candidates(
    rendered_boundary: Tensor,
    teacher_boundary: Tensor,
    teacher_confidence: Tensor,
    depth: Tensor,
    extrinsics: Tensor,
    intrinsics: Tensor,
    radius: int = 4,
    depth_relative_tolerance: float = 0.05,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Estimate source-view boundary correction from other camera views.

    A rendered source-boundary pixel is back-projected with rendered depth and
    reprojected into every other view. The closest teacher boundary is selected
    there, back-projected with its target-view depth, and projected back into
    the source view. The returned dx/dy are therefore measured in source-view
    pixels, rather than mixing directions from different camera frames.

    Returns ``(dx, dy, confidence, visibility)`` with shape ``[B,V,1,H,W]``.
    """
    batch, views, height, width = _validate_inputs(
        teacher_boundary,
        teacher_confidence,
        depth,
        extrinsics,
        intrinsics,
    )
    if rendered_boundary.shape != teacher_boundary.shape:
        raise ValueError("rendered_boundary must match teacher_boundary")
    if radius < 1:
        raise ValueError("radius must be positive")
    if depth_relative_tolerance <= 0:
        raise ValueError("depth_relative_tolerance must be positive")
    if views < 2:
        zeros = torch.zeros_like(rendered_boundary)
        return zeros, zeros.clone(), zeros.clone(), zeros.clone()
    if depth.ndim == 5:
        depth = depth[:, :, 0]

    dtype = depth.dtype
    device = depth.device
    rendered = rendered_boundary.to(dtype)
    teacher = teacher_boundary.to(dtype)
    confidence = teacher_confidence.to(dtype)
    extrinsics = extrinsics.to(dtype)
    intrinsics = intrinsics.to(dtype)
    world_to_camera = torch.linalg.inv(extrinsics)

    x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    pixels = torch.stack((grid_x, grid_y, torch.ones_like(grid_x)), dim=-1)
    pixels = pixels.reshape(1, height * width, 3).expand(batch, -1, -1)
    pixel_scale = depth.new_tensor((width, height)).reshape(1, 1, 2)

    offset_maps = [
        _nearest_boundary_offset_maps(
            teacher[:, target_index], confidence[:, target_index], radius
        )
        for target_index in range(views)
    ]
    displacement_x = torch.zeros_like(rendered)
    displacement_y = torch.zeros_like(rendered)
    consensus = torch.zeros_like(rendered)
    visibility = torch.zeros_like(rendered)

    for source_index in range(views):
        source_depth = depth[:, source_index].reshape(batch, height * width, 1)
        source_rays = pixels @ torch.linalg.inv(
            intrinsics[:, source_index]
        ).transpose(-1, -2)
        source_camera = source_rays * source_depth
        source_h = torch.cat(
            (source_camera, torch.ones_like(source_camera[..., :1])), dim=-1
        )
        source_world = source_h @ extrinsics[:, source_index].transpose(-1, -2)
        weighted_dx = torch.zeros(batch, 1, height, width, device=device, dtype=dtype)
        weighted_dy = torch.zeros_like(weighted_dx)
        weighted_magnitude = torch.zeros_like(weighted_dx)
        support_sum = torch.zeros_like(weighted_dx)
        visible_sum = torch.zeros_like(weighted_dx)

        for target_index in range(views):
            if target_index == source_index:
                continue
            target_camera = source_world @ world_to_camera[:, target_index].transpose(
                -1, -2
            )
            target_xyz = target_camera[..., :3]
            target_z = target_xyz[..., 2:3]
            target_xy = target_xyz[..., :2] / target_z.clamp_min(1e-6)
            target_h = torch.cat((target_xy, torch.ones_like(target_z)), dim=-1)
            projected_target = target_h @ intrinsics[:, target_index].transpose(
                -1, -2
            )
            target_grid = (
                projected_target[..., :2].reshape(batch, height, width, 2) * 2 - 1
            )
            map_dx, map_dy, map_confidence = offset_maps[target_index]
            sampled_dx = F.grid_sample(
                map_dx, target_grid, mode="nearest", padding_mode="zeros",
                align_corners=False,
            )
            sampled_dy = F.grid_sample(
                map_dy, target_grid, mode="nearest", padding_mode="zeros",
                align_corners=False,
            )
            sampled_confidence = F.grid_sample(
                map_confidence, target_grid, mode="nearest", padding_mode="zeros",
                align_corners=False,
            )
            matched_target = projected_target[..., :2].reshape(
                batch, height, width, 2
            ).clone()
            matched_target[..., 0] += sampled_dx[:, 0] / width
            matched_target[..., 1] += sampled_dy[:, 0] / height
            matched_grid = matched_target * 2 - 1
            matched_depth = F.grid_sample(
                depth[:, target_index].unsqueeze(1), matched_grid,
                mode="bilinear", padding_mode="zeros", align_corners=False,
            )
            projected_depth = target_z.reshape(batch, 1, height, width)
            depth_scale = torch.maximum(
                matched_depth.abs(), projected_depth.abs()
            ).clamp_min(1e-3)
            relative_error = (matched_depth - projected_depth).abs() / depth_scale
            depth_agreement = torch.exp(-relative_error / depth_relative_tolerance)
            projected_inside = (
                (target_z[..., 0] > 0)
                & (projected_target[..., 0] >= 0)
                & (projected_target[..., 0] <= 1)
                & (projected_target[..., 1] >= 0)
                & (projected_target[..., 1] <= 1)
                & (source_depth[..., 0] > 0)
            ).reshape(batch, 1, height, width)
            matched_inside = (
                (matched_target[..., 0] >= 0)
                & (matched_target[..., 0] <= 1)
                & (matched_target[..., 1] >= 0)
                & (matched_target[..., 1] <= 1)
            ).unsqueeze(1)
            geometrically_visible = projected_inside & matched_inside & (matched_depth > 0)

            matched_h = torch.cat(
                (
                    matched_target.reshape(batch, height * width, 2),
                    torch.ones(batch, height * width, 1, device=device, dtype=dtype),
                ),
                dim=-1,
            )
            matched_rays = matched_h @ torch.linalg.inv(
                intrinsics[:, target_index]
            ).transpose(-1, -2)
            matched_camera = matched_rays * matched_depth.reshape(
                batch, height * width, 1
            )
            matched_camera_h = torch.cat(
                (matched_camera, torch.ones_like(matched_camera[..., :1])), dim=-1
            )
            matched_world = matched_camera_h @ extrinsics[:, target_index].transpose(
                -1, -2
            )
            matched_source = matched_world @ world_to_camera[:, source_index].transpose(
                -1, -2
            )
            matched_source_xyz = matched_source[..., :3]
            matched_source_xy = matched_source_xyz[..., :2] / matched_source_xyz[
                ..., 2:3
            ].clamp_min(1e-6)
            matched_source_h = torch.cat(
                (matched_source_xy, torch.ones_like(matched_source_xyz[..., 2:3])),
                dim=-1,
            )
            projected_source = matched_source_h @ intrinsics[:, source_index].transpose(
                -1, -2
            )
            source_displacement = (
                projected_source[..., :2] - pixels[..., :2]
            ) * pixel_scale
            source_displacement = source_displacement.reshape(
                batch, height, width, 2
            ).permute(0, 3, 1, 2)
            weight = (
                sampled_confidence
                * depth_agreement
                * geometrically_visible.to(dtype)
            )
            weighted_dx += weight * source_displacement[:, 0:1]
            weighted_dy += weight * source_displacement[:, 1:2]
            weighted_magnitude += weight * torch.sqrt(
                source_displacement[:, 0:1].square()
                + source_displacement[:, 1:2].square()
            )
            support_sum += weight
            visible_sum += geometrically_visible.to(dtype)

        source_mask = rendered[:, source_index]
        displacement_x[:, source_index] = source_mask * torch.where(
            support_sum > 0, weighted_dx / support_sum.clamp_min(1e-6), 0.0
        )
        displacement_y[:, source_index] = source_mask * torch.where(
            support_sum > 0, weighted_dy / support_sum.clamp_min(1e-6), 0.0
        )
        vector_sum_magnitude = torch.sqrt(
            weighted_dx.square() + weighted_dy.square()
        )
        directional_coherence = torch.where(
            weighted_magnitude > 1e-6,
            vector_sum_magnitude / weighted_magnitude.clamp_min(1e-6),
            (support_sum > 0).to(dtype),
        ).clamp(0, 1)
        consensus[:, source_index] = source_mask * (
            (support_sum / visible_sum.clamp_min(1.0)) * directional_coherence
        ).clamp(0, 1)
        visibility[:, source_index] = source_mask * (
            visible_sum / float(views - 1)
        ).clamp(0, 1)

    return (
        displacement_x.to(rendered_boundary.dtype),
        displacement_y.to(rendered_boundary.dtype),
        consensus.to(rendered_boundary.dtype),
        visibility.to(rendered_boundary.dtype),
    )


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
