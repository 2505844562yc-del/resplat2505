from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .types import SemanticPredictions


def _group_count(channels: int) -> int:
    for groups in (16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _conv_block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
        nn.GroupNorm(_group_count(out_channels), out_channels),
        nn.GELU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.GroupNorm(_group_count(out_channels), out_channels),
        nn.GELU(),
    )


@dataclass
class SemanticSceneEncoderCfg:
    backbone_channels: int = 768
    decoder_channels: int = 128
    edge_channels: int = 64
    num_backbone_levels: int = 4
    num_classes: int = 100
    instance_dim: int = 16


class SemanticSceneEncoder(nn.Module):
    """Decode frozen DINO features while preserving high-resolution boundaries.

    ReSplat already computes DINO features for depth. Reusing those frozen tensors
    avoids a second foundation-model forward pass. A trainable RGB edge pathway
    restores fine contours that cannot be represented on the stride-14 token grid.
    """

    def __init__(self, cfg: SemanticSceneEncoderCfg) -> None:
        super().__init__()
        self.cfg = cfg
        edge_half = cfg.edge_channels // 2
        self.rgb_half = _conv_block(3, edge_half, stride=2)
        self.rgb_quarter = _conv_block(edge_half, cfg.edge_channels, stride=2)

        self.backbone_projections = nn.ModuleList(
            nn.Conv2d(cfg.backbone_channels, cfg.decoder_channels, 1)
            for _ in range(cfg.num_backbone_levels)
        )
        self.fusion = _conv_block(
            cfg.decoder_channels + cfg.edge_channels,
            cfg.decoder_channels,
        )

        self.class_head = nn.Conv2d(cfg.decoder_channels, cfg.num_classes, 1)
        self.instance_head = nn.Conv2d(cfg.decoder_channels, cfg.instance_dim, 1)
        self.confidence_head = nn.Conv2d(cfg.decoder_channels, 1, 1)

        # Boundary prediction receives a half-resolution RGB skip connection.
        self.boundary_refine = _conv_block(
            cfg.decoder_channels + edge_half,
            cfg.edge_channels,
        )
        self.boundary_head = nn.Conv2d(cfg.edge_channels, 1, 1)

    def forward(
        self,
        images: Tensor,
        backbone_features: tuple[Tensor, ...] | list[Tensor],
    ) -> SemanticPredictions:
        if images.ndim != 5:
            raise ValueError("images must have shape [B,V,3,H,W]")
        if len(backbone_features) != self.cfg.num_backbone_levels:
            raise ValueError(
                f"expected {self.cfg.num_backbone_levels} backbone levels, "
                f"got {len(backbone_features)}"
            )

        b, v, _, h, w = images.shape
        flat_images = images.reshape(b * v, 3, h, w)
        rgb_half = self.rgb_half(flat_images)
        rgb_quarter = self.rgb_quarter(rgb_half)
        quarter_size = rgb_quarter.shape[-2:]

        projected = []
        for projection, feature in zip(self.backbone_projections, backbone_features):
            if feature.shape[0] != b * v:
                raise ValueError("backbone feature batch must equal B*V")
            feature = projection(feature.float())
            feature = F.interpolate(
                feature,
                size=quarter_size,
                mode="bilinear",
                align_corners=False,
            )
            projected.append(feature)
        semantic_tokens = torch.stack(projected, dim=0).mean(dim=0)
        decoder_features = self.fusion(torch.cat([semantic_tokens, rgb_quarter], dim=1))

        class_logits = self.class_head(decoder_features)
        instance_embeddings = F.normalize(
            self.instance_head(decoder_features), dim=1, eps=1e-6
        )
        confidence_logits = self.confidence_head(decoder_features)

        boundary_context = F.interpolate(
            decoder_features,
            size=rgb_half.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        boundary_features = self.boundary_refine(
            torch.cat([boundary_context, rgb_half], dim=1)
        )
        boundary_logits = self.boundary_head(boundary_features)

        def full_resolution(x: Tensor) -> Tensor:
            return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)

        class_logits = full_resolution(class_logits)
        instance_embeddings = F.normalize(full_resolution(instance_embeddings), dim=1, eps=1e-6)
        confidence_logits = full_resolution(confidence_logits)
        boundary_logits = full_resolution(boundary_logits)

        def restore(x: Tensor) -> Tensor:
            return x.reshape(b, v, *x.shape[1:])

        return SemanticPredictions(
            class_logits=restore(class_logits),
            instance_embeddings=restore(instance_embeddings),
            boundary_logits=restore(boundary_logits),
            confidence_logits=restore(confidence_logits),
            decoder_features=restore(decoder_features),
        )
