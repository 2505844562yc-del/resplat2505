from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from ...geometry.projection import get_world_rays, sample_image_grid
from ..types import Gaussians
from .types import CandidateVerification, GaussianAssembly, RayHypotheses


@dataclass
class GaussianAssemblerCfg:
    primary_keep_floor: float = 0.5
    secondary_keep_floor: float = 0.0
    opacity_delta_scale: float = 0.5
    scale_ratio_min: float = 0.5
    scale_ratio_max: float = 2.0
    mask_threshold: float = 1e-4
    eps: float = 1e-6


class HypothesisGaussianAssembler(nn.Module):
    """Turn verified ray hypotheses into renderer-ready Gaussians.

    Geometry topology changes here: each compact ReSplat parent Gaussian receives
    two fixed candidate slots. The primary is protected by a conservative floor;
    the secondary starts effectively transparent unless both the ray mixture and
    cross-view verifier support it. Hard zero-opacity gating is reserved for inference; physical compaction is an export step.
    """

    def __init__(self, cfg: Optional[GaussianAssemblerCfg] = None) -> None:
        super().__init__()
        self.cfg = cfg or GaussianAssemblerCfg()

    @staticmethod
    def _reshape_parent(value: Tensor, b: int, v: int, h: int, w: int) -> Tensor:
        if value.shape[0] != b or value.shape[1] != v * h * w:
            raise ValueError(
                f"parent Gaussian count {value.shape[1]} does not equal V*H*W={v*h*w}"
            )
        return value.reshape(b, v, h, w, *value.shape[2:])

    @staticmethod
    def _duplicate_slots(value: Tensor, k: int) -> Tensor:
        # [B,V,H,W,...] -> [B,V,K,H,W,...]
        return value.unsqueeze(2).expand(
            value.shape[0], value.shape[1], k, *value.shape[2:]
        )

    def forward(
        self,
        parent_gaussians: Gaussians,
        hypotheses: RayHypotheses,
        verification: CandidateVerification,
        intrinsics: Tensor,
        extrinsics: Tensor,
        hard_gate: bool = False,
    ) -> GaussianAssembly:
        depths = hypotheses.depths.float()
        b, v, k, h, w = depths.shape
        expected_shape = (b, v, k, h, w)
        for name, value in (
            ("mixture_weights", hypotheses.mixture_weights),
            ("keep_probability", verification.keep_probability),
            ("opacity_delta", verification.opacity_delta),
        ):
            if value.shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}, got {value.shape}")
        if k != 2:
            raise ValueError("current semantic mixture design requires exactly two slots")
        if intrinsics.shape[:2] != (b, v) or extrinsics.shape[:2] != (b, v):
            raise ValueError("camera dimensions do not match hypotheses")
        if parent_gaussians.covariances is None:
            raise ValueError("parent Gaussian covariances are required")

        xy, _ = sample_image_grid((h, w), depths.device)
        xy = xy.to(depths.dtype).view(1, 1, 1, h, w, 2)
        origins, directions = get_world_rays(
            xy,
            extrinsics[:, :, None, None, None],
            intrinsics[:, :, None, None, None],
        )
        means = origins + directions * depths[..., None]

        parent_covariances = self._reshape_parent(
            parent_gaussians.covariances, b, v, h, w
        )
        parent_harmonics = self._reshape_parent(
            parent_gaussians.harmonics, b, v, h, w
        )
        parent_opacities = self._reshape_parent(
            parent_gaussians.opacities, b, v, h, w
        )
        covariances = self._duplicate_slots(parent_covariances, k)
        harmonics = self._duplicate_slots(parent_harmonics, k)
        base_opacity = self._duplicate_slots(parent_opacities, k)

        primary_depth = depths[:, :, :1]
        scale_ratio = (depths / primary_depth.clamp_min(self.cfg.eps)).clamp(
            self.cfg.scale_ratio_min,
            self.cfg.scale_ratio_max,
        )
        covariances = covariances * scale_ratio[..., None, None].square()

        floors = depths.new_tensor(
            [self.cfg.primary_keep_floor, self.cfg.secondary_keep_floor]
        ).view(1, 1, k, 1, 1)
        keep_factor = floors + (1.0 - floors) * verification.keep_probability
        opacity_factor = torch.exp(
            self.cfg.opacity_delta_scale * verification.opacity_delta
        )
        candidate_opacity = (
            base_opacity
            * hypotheses.mixture_weights
            * keep_factor
            * opacity_factor
        ).clamp(0, 1)
        candidate_mask = candidate_opacity > self.cfg.mask_threshold
        if hard_gate:
            candidate_opacity = candidate_opacity * candidate_mask

        parent_scales = None
        if parent_gaussians.scales is not None:
            parent_scales = self._reshape_parent(parent_gaussians.scales, b, v, h, w)
            parent_scales = self._duplicate_slots(parent_scales, k)
            parent_scales = parent_scales * scale_ratio[..., None]

        parent_rotations = None
        if parent_gaussians.rotations is not None:
            parent_rotations = self._duplicate_slots(
                self._reshape_parent(parent_gaussians.rotations, b, v, h, w), k
            )
        parent_rotations_unnorm = None
        if parent_gaussians.rotations_unnorm is not None:
            parent_rotations_unnorm = self._duplicate_slots(
                self._reshape_parent(parent_gaussians.rotations_unnorm, b, v, h, w), k
            )

        def flatten(value: Tensor) -> Tensor:
            return value.reshape(b, v * k * h * w, *value.shape[5:])

        gaussians = Gaussians(
            means=flatten(means),
            covariances=flatten(covariances),
            harmonics=flatten(harmonics),
            opacities=flatten(candidate_opacity),
            scales=None if parent_scales is None else flatten(parent_scales),
            rotations=None if parent_rotations is None else flatten(parent_rotations),
            rotations_unnorm=(
                None
                if parent_rotations_unnorm is None
                else flatten(parent_rotations_unnorm)
            ),
            mask=flatten(candidate_mask) if hard_gate else None,
        )

        source_view = torch.arange(v, device=depths.device).view(1, v, 1, 1, 1)
        source_view = source_view.expand(b, v, k, h, w).reshape(b, -1)
        slot_index = torch.arange(k, device=depths.device).view(1, 1, k, 1, 1)
        slot_index = slot_index.expand(b, v, k, h, w).reshape(b, -1)
        return GaussianAssembly(
            gaussians=gaussians,
            candidate_opacity=candidate_opacity,
            candidate_mask=candidate_mask,
            source_view=source_view,
            slot_index=slot_index,
        )
