from dataclasses import dataclass
from typing import Optional

import torch
from einops import rearrange
from torch import Tensor, nn
import torch.nn.functional as F

from ...geometry.projection import get_world_rays, sample_image_grid
from .types import CandidateVerification, RayHypotheses, SemanticPredictions


@dataclass
class CrossViewVerifierCfg:
    hidden_channels: int = 64
    depth_tolerance_relative: float = 0.03
    depth_tolerance_absolute: float = 0.02
    visibility_temperature_relative: float = 0.02
    keep_bias: float = -3.0
    geometry_prior_strength: float = 4.0
    semantic_prior_strength: float = 1.5
    visibility_prior_strength: float = 1.5
    eps: float = 1e-6


class CrossViewHypothesisVerifier(nn.Module):
    """Validate candidate surfaces against all other context views.

    This module operates before Gaussian construction. Each candidate 3D point is
    reprojected into neighboring views and scored using their depth posterior,
    semantic class, instance identity, semantic confidence, visibility, and camera
    geometry. The learned network predicts only a residual over an interpretable
    support prior, so useful behavior exists before large-scale training.
    """

    evidence_channels = 7

    def __init__(self, cfg: Optional[CrossViewVerifierCfg] = None) -> None:
        super().__init__()
        self.cfg = cfg or CrossViewVerifierCfg()
        hidden = self.cfg.hidden_channels
        self.evidence_encoder = nn.Sequential(
            nn.Linear(self.evidence_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.view_attention = nn.Linear(hidden, 1)
        self.keep_residual = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.opacity_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.keep_residual[-1].weight)
        nn.init.zeros_(self.keep_residual[-1].bias)
        nn.init.zeros_(self.opacity_head[-1].weight)
        nn.init.zeros_(self.opacity_head[-1].bias)

    @staticmethod
    def _resize_view_maps(maps: Tensor, size: tuple[int, int]) -> Tensor:
        b, v, c = maps.shape[:3]
        flat = maps.reshape(b * v, c, *maps.shape[-2:])
        if flat.shape[-2:] != size:
            flat = F.interpolate(flat, size=size, mode="bilinear", align_corners=False)
        return flat.reshape(b, v, c, *size)

    @staticmethod
    def _sample_target_maps(maps: Tensor, target_grid: Tensor) -> Tensor:
        """Sample [B,Vt,C,Hm,Wm] at [B,Vr,Vt,K,H,W,2] normalized grids."""
        b, vt, c, hm, wm = maps.shape
        _, vr, vt_grid, k, h, w, _ = target_grid.shape
        if vt != vt_grid:
            raise ValueError("target grid view count does not match maps")
        # Keep each target-view feature map only once. Reference-view and
        # hypothesis queries are packed into the output-grid height instead of
        # expanding the input map B*V_ref*K times.
        flat_maps = maps.reshape(b * vt, c, hm, wm)
        packed_grid = target_grid.permute(0, 2, 1, 3, 4, 5, 6)
        packed_grid = packed_grid.reshape(b * vt, vr * k * h, w, 2)
        sampled = F.grid_sample(
            flat_maps,
            packed_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.reshape(b, vt, c, vr, k, h, w)
        return sampled.permute(0, 3, 1, 4, 2, 5, 6).contiguous()

    def _project_hypotheses(
        self,
        depths: Tensor,
        intrinsics: Tensor,
        extrinsics: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        b, vr, k, h, w = depths.shape
        xy, _ = sample_image_grid((h, w), depths.device)
        xy = xy.to(depths.dtype).view(1, 1, 1, h, w, 2)
        xy = xy.expand(b, vr, k, h, w, 2)
        origins, directions = get_world_rays(
            xy,
            extrinsics[:, :, None, None, None],
            intrinsics[:, :, None, None, None],
        )
        world_points = origins + directions * depths[..., None]

        world_h = torch.cat([world_points, torch.ones_like(world_points[..., :1])], dim=-1)
        world_to_camera = torch.linalg.inv(extrinsics.float()).to(depths.dtype)
        camera_points = torch.einsum(
            "btij,brkhwj->brtkhwi",
            world_to_camera,
            world_h,
        )[..., :3]
        projected = torch.einsum(
            "btij,brtkhwj->brtkhwi",
            intrinsics,
            camera_points,
        )
        projected_depth = camera_points[..., 2]
        uv = projected[..., :2] / projected[..., 2:].clamp_min(self.cfg.eps)
        valid = (
            (projected_depth > self.cfg.eps)
            & (uv[..., 0] >= 0)
            & (uv[..., 0] <= 1)
            & (uv[..., 1] >= 0)
            & (uv[..., 1] <= 1)
        )
        eye = torch.eye(vr, device=depths.device, dtype=torch.bool)
        valid = valid & (~eye.view(1, vr, vr, 1, 1, 1))
        grid = uv * 2.0 - 1.0

        camera_centers = extrinsics[..., :3, 3]
        ref_vectors = camera_centers[:, :, None, None, None] - world_points
        target_vectors = (
            camera_centers[:, None, :, None, None, None]
            - world_points[:, :, None]
        )
        ref_vectors = F.normalize(ref_vectors, dim=-1, eps=self.cfg.eps)
        target_vectors = F.normalize(target_vectors, dim=-1, eps=self.cfg.eps)
        view_cosine = (
            ref_vectors[:, :, None] * target_vectors
        ).sum(dim=-1).clamp(-1, 1)
        view_cosine = (view_cosine + 1.0) * 0.5

        baseline = camera_centers[:, None] - camera_centers[:, :, None]
        baseline = baseline.norm(dim=-1)[:, :, :, None, None, None]
        baseline_quality = torch.tanh(
            baseline / projected_depth.abs().clamp_min(self.cfg.eps)
        )
        return world_points, grid, projected_depth, valid, (view_cosine + baseline_quality) * 0.5

    def forward(
        self,
        hypotheses: RayHypotheses,
        depth_probabilities: Tensor,
        depth_candidates: Tensor,
        semantics: SemanticPredictions,
        intrinsics: Tensor,
        extrinsics: Tensor,
    ) -> CandidateVerification:
        depths = hypotheses.depths.float()
        if depths.ndim != 5:
            raise ValueError("hypothesis depths must have shape [B,V,K,H,W]")
        if depth_probabilities.shape != depth_candidates.shape:
            raise ValueError("depth probabilities and candidates must have equal shape")
        b, v, k, h, w = depths.shape
        if intrinsics.shape[:2] != (b, v) or extrinsics.shape[:2] != (b, v):
            raise ValueError("camera batch/view dimensions must match hypotheses")

        _, grid, projected_depth, valid, view_geometry = self._project_hypotheses(
            depths, intrinsics.float(), extrinsics.float()
        )

        sampled_probabilities = self._sample_target_maps(
            depth_probabilities.float(), grid
        )
        sampled_candidates = self._sample_target_maps(
            depth_candidates.float(), grid
        )
        probability_norm = sampled_probabilities.sum(dim=4, keepdim=True).clamp_min(
            self.cfg.eps
        )
        sampled_probabilities = sampled_probabilities / probability_norm
        tolerance = (
            self.cfg.depth_tolerance_absolute
            + self.cfg.depth_tolerance_relative * projected_depth.abs()
        )
        depth_kernel = torch.exp(
            -0.5
            * (
                (sampled_candidates - projected_depth.unsqueeze(4))
                / tolerance.unsqueeze(4).clamp_min(self.cfg.eps)
            )
            ** 2
        )
        depth_support = (sampled_probabilities * depth_kernel).sum(dim=4)
        expected_target_depth = (
            sampled_probabilities * sampled_candidates
        ).sum(dim=4)
        visibility_temperature = (
            self.cfg.depth_tolerance_absolute
            + self.cfg.visibility_temperature_relative * projected_depth.abs()
        )
        visibility = torch.sigmoid(
            (expected_target_depth + tolerance - projected_depth)
            / visibility_temperature.clamp_min(self.cfg.eps)
        )

        class_probabilities = semantics.class_logits.float().softmax(dim=2)
        instance_embeddings = F.normalize(
            semantics.instance_embeddings.float(), dim=2, eps=self.cfg.eps
        )
        semantic_confidence = semantics.confidence_logits.float().sigmoid()
        ref_classes = self._resize_view_maps(class_probabilities, (h, w))
        ref_instances = self._resize_view_maps(instance_embeddings, (h, w))
        ref_confidence = self._resize_view_maps(semantic_confidence, (h, w))
        target_classes = self._sample_target_maps(class_probabilities, grid)
        target_instances = self._sample_target_maps(instance_embeddings, grid)
        target_confidence = self._sample_target_maps(semantic_confidence, grid).squeeze(4)

        class_similarity = (
            ref_classes[:, :, None, None] * target_classes
        ).sum(dim=4).clamp(0, 1)
        instance_similarity = (
            ref_instances[:, :, None, None] * target_instances
        ).sum(dim=4).clamp(-1, 1)
        instance_similarity = (instance_similarity + 1.0) * 0.5
        confidence_pair = (
            ref_confidence[:, :, None, None, 0] * target_confidence
        ).clamp(0, 1)

        raw_evidence = torch.stack(
            [
                depth_support,
                class_similarity,
                instance_similarity,
                confidence_pair,
                visibility,
                view_geometry,
                hypotheses.mixture_weights[:, :, None].expand(-1, -1, v, -1, -1, -1),
            ],
            dim=-1,
        )
        encoded = self.evidence_encoder(raw_evidence)
        attention_logits = self.view_attention(encoded).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~valid, -1e4)
        attention = attention_logits.softmax(dim=2) * valid.float()
        attention = attention / attention.sum(dim=2, keepdim=True).clamp_min(self.cfg.eps)

        aggregated_hidden = (attention[..., None] * encoded).sum(dim=2)
        aggregated_raw = (attention[..., None] * raw_evidence).sum(dim=2)
        valid_count = valid.float().sum(dim=2)
        has_support = valid_count > 0

        geometric_support = aggregated_raw[..., 0]
        semantic_support = (
            aggregated_raw[..., 1]
            + aggregated_raw[..., 2]
            + aggregated_raw[..., 3]
        ) / 3.0
        visibility_support = aggregated_raw[..., 4]
        prior_logit = (
            self.cfg.keep_bias
            + self.cfg.geometry_prior_strength * geometric_support
            + self.cfg.semantic_prior_strength * semantic_support
            + self.cfg.visibility_prior_strength * visibility_support
        )
        keep_residual = self.keep_residual(aggregated_hidden).squeeze(-1)
        keep_probability = torch.sigmoid(prior_logit + keep_residual)
        keep_probability = keep_probability * has_support.float()
        opacity_delta = torch.tanh(self.opacity_head(aggregated_hidden).squeeze(-1))
        opacity_delta = opacity_delta * has_support.float()

        return CandidateVerification(
            keep_probability=keep_probability,
            opacity_delta=opacity_delta,
            geometric_support=geometric_support,
            semantic_support=semantic_support,
            visibility_support=visibility_support,
            valid_view_count=valid_count,
            view_attention=attention,
        )
