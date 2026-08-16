"""Parameter-level routing for semantic Gaussian refinement."""

import torch
from torch import Tensor


def apply_semantic_parameter_routes(
    delta_means: Tensor,
    delta_scales: Tensor,
    delta_rotations: Tensor,
    delta_opacities: Tensor,
    delta_shs: Tensor,
    routes: Tensor,
    gain: float = 0.5,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Route geometry and appearance deltas with bounded residual factors.

    ``routes[..., 0]`` controls geometry.  An optional second channel controls
    SH appearance; with one channel, appearance is kept exactly unchanged.
    Zero routes are an exact identity mapping.
    """
    if routes.shape[:-1] != delta_means.shape[:-1] or routes.shape[-1] not in (1, 2):
        raise ValueError("routes must have shape [B, N, 1] or [B, N, 2]")
    if gain < 0:
        raise ValueError("gain must be non-negative")
    geometry_factor = 1.0 + gain * torch.tanh(routes[..., 0:1])
    appearance_factor = (
        1.0 + gain * torch.tanh(routes[..., 1:2])
        if routes.shape[-1] == 2
        else 1.0
    )
    return (
        delta_means * geometry_factor,
        delta_scales * geometry_factor,
        delta_rotations * geometry_factor,
        delta_opacities * geometry_factor,
        delta_shs * appearance_factor,
    )
