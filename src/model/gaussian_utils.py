"""Utilities that preserve optional fields of Gaussian collections."""

from collections.abc import Sequence

import torch

from .types import Gaussians


def _concatenate_optional(
    gaussians: Sequence[Gaussians], name: str
) -> torch.Tensor | None:
    values = [getattr(item, name) for item in gaussians]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"cannot merge mixed Gaussian field: {name}")
    return torch.cat(values, dim=1)


def _shared_optional(
    gaussians: Sequence[Gaussians], name: str
) -> torch.Tensor | None:
    values = [getattr(item, name) for item in gaussians]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"cannot merge mixed scene field: {name}")
    first = values[0]
    if any(not torch.equal(first, value) for value in values[1:]):
        raise ValueError(f"scene field differs across Gaussian windows: {name}")
    return first


def merge_gaussians(gaussians: Sequence[Gaussians]) -> Gaussians:
    """Merge windowed Gaussians without silently dropping semantic state."""
    if not gaussians:
        raise ValueError("at least one Gaussian collection is required")
    return Gaussians(
        means=torch.cat([item.means for item in gaussians], dim=1),
        covariances=_concatenate_optional(gaussians, "covariances"),
        harmonics=torch.cat([item.harmonics for item in gaussians], dim=1),
        opacities=torch.cat([item.opacities for item in gaussians], dim=1),
        scales=_concatenate_optional(gaussians, "scales"),
        rotations=_concatenate_optional(gaussians, "rotations"),
        probabilities=_concatenate_optional(gaussians, "probabilities"),
        mask=_concatenate_optional(gaussians, "mask"),
        filter_3D=_concatenate_optional(gaussians, "filter_3D"),
        rotations_unnorm=_concatenate_optional(gaussians, "rotations_unnorm"),
        scale_factor=_shared_optional(gaussians, "scale_factor"),
        shift=_shared_optional(gaussians, "shift"),
        semantic_features=_concatenate_optional(gaussians, "semantic_features"),
        semantic_uncertainty=_concatenate_optional(
            gaussians, "semantic_uncertainty"
        ),
    )
