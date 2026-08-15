"""Differentiable boundary operators shared by feedback and supervision."""

import torch
import torch.nn.functional as F
from torch import Tensor


def rgb_to_soft_boundary(image: Tensor, gain: float = 5.0) -> Tensor:
    if image.shape[-3] != 3:
        raise ValueError(f"Expected RGB input, got shape {tuple(image.shape)}")
    flat = image.reshape(-1, 3, image.shape[-2], image.shape[-1])
    weights = flat.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
    gray = (flat * weights).sum(dim=1, keepdim=True)
    sobel_x = gray.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3) / 8.0
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_x.transpose(-1, -2), padding=1)
    magnitude = torch.sqrt(grad_x.square() + grad_y.square() + 1e-12)
    boundary = 1.0 - torch.exp(-gain * magnitude)
    return boundary.reshape(*image.shape[:-3], 1, image.shape[-2], image.shape[-1])


def confidence_gated_boundary_residual(
    rendered_rgb: Tensor,
    target_boundary: Tensor,
    confidence: Tensor,
    gain: float = 5.0,
    confidence_floor: float = 0.0,
) -> Tensor:
    predicted = rgb_to_soft_boundary(rendered_rgb, gain)
    if predicted.shape != target_boundary.shape or predicted.shape != confidence.shape:
        raise ValueError(
            f"Boundary shape mismatch: {predicted.shape}, "
            f"{target_boundary.shape}, {confidence.shape}"
        )
    gate = torch.where(
        confidence >= confidence_floor, confidence, torch.zeros_like(confidence)
    )
    return (predicted - target_boundary) * gate


def balanced_boundary_l1(predicted: Tensor, target: Tensor, weight: Tensor) -> Tensor:
    positive = (target >= 0.5).to(predicted.dtype)
    negative = 1.0 - positive
    pos_weight = weight * positive
    neg_weight = weight * negative
    pos_loss = (predicted - target).abs().mul(pos_weight).sum() / pos_weight.sum().clamp_min(1.0)
    neg_loss = (predicted - target).abs().mul(neg_weight).sum() / neg_weight.sum().clamp_min(1.0)
    return 0.5 * (pos_loss + neg_loss)
