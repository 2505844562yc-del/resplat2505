from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor
import math

from ...dataset import DatasetCfg
from ..types import Gaussians
from .decoder import DepthRenderingMode
from gsplat.rendering import rasterization
from .decoder import Decoder, DecoderOutput


@dataclass
class GSplatDecoderSplattingCUDACfg:
    name: Literal["gsplat"]
    scale_invariant: bool
    use_covariances: bool | None = True


class GSplatDecoderSplattingCUDA(Decoder[GSplatDecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: GSplatDecoderSplattingCUDACfg,
        dataset_cfg: DatasetCfg,
    ) -> None:
        super().__init__(cfg, dataset_cfg)
        self.register_buffer(
            "background_color",
            torch.tensor(dataset_cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
        return_radii: bool = False,
    ) -> DecoderOutput:
        height, width = image_shape
        means = gaussians.means  # [B, G, 3]
        # NOTE: rasterization does normalization internally
        quats = gaussians.rotations_unnorm  # [B, G, 4]
        scales = gaussians.scales  # [B, G, 3]
        if self.cfg.use_covariances:
            covars = gaussians.covariances  # [B, G, 3, 3]
        else:
            covars = None
        opacities = gaussians.opacities  # [B, G]
        colors = gaussians.harmonics.permute(0, 1, 3, 2)  # [B, G, d_sh, 3]
        sh_degree = int(math.sqrt(colors.shape[-2])) - 1  # d_sh = (degree + 1) ** 2
        viewmats = extrinsics.inverse()  # [B, V, 4, 4]
        intrinsics = intrinsics.clone()  # [B, V, 3, 3]
        # scale to the image shape
        intrinsics[:, :, 0] *= width
        intrinsics[:, :, 1] *= height

        render_colors, render_alphas, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            sh_degree=sh_degree,
            viewmats=viewmats,
            Ks=intrinsics,
            width=width,
            height=height,
            near_plane=near[0, 0].item(),  # expect float
            far_plane=far[0, 0].item(),
            eps2d=0.1,
            rasterize_mode="antialiased",
            packed=True,
            absgrad=False,
            sparse_grad=False,
            render_mode="RGB+ED",
            covars=covars,
        )

        color = render_colors[..., :3].permute(0, 1, 4, 2, 3)  # [B, V, 3, H, W]
        depth = render_colors[..., -1]  # [B, V, H, W]

        return DecoderOutput(
            color,
            depth=depth,
            accumulated_alpha=render_alphas.squeeze(-1)  # [B, V, H, W]
        )

    def forward_features(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        features: Tensor | None = None,
        normalize: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Render arbitrary per-Gaussian features with RGB visibility weights."""
        if features is None:
            features = gaussians.semantic_features
        if features is None:
            raise ValueError("semantic features are required for feature rendering")
        if features.ndim != 3 or features.shape[:2] != gaussians.means.shape[:2]:
            raise ValueError("features must have shape [B, G, D]")

        height, width = image_shape
        viewmats = extrinsics.inverse()
        scaled_intrinsics = intrinsics.clone()
        scaled_intrinsics[:, :, 0] *= width
        scaled_intrinsics[:, :, 1] *= height
        covars = gaussians.covariances if self.cfg.use_covariances else None

        rendered, alphas, _ = rasterization(
            means=gaussians.means,
            quats=gaussians.rotations_unnorm,
            scales=gaussians.scales,
            opacities=gaussians.opacities,
            colors=features.float(),
            sh_degree=None,
            viewmats=viewmats,
            Ks=scaled_intrinsics,
            width=width,
            height=height,
            near_plane=near[0, 0].item(),
            far_plane=far[0, 0].item(),
            eps2d=0.1,
            rasterize_mode="antialiased",
            packed=True,
            absgrad=False,
            sparse_grad=False,
            render_mode="RGB",
            channel_chunk=32,
            covars=covars,
        )
        if normalize:
            rendered = rendered / alphas.clamp_min(1e-6)
        return rendered.permute(0, 1, 4, 2, 3), alphas.squeeze(-1)
