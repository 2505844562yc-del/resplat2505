from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn
import numpy as np
import torch.utils.checkpoint

from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid, get_world_rays
from ..types import Gaussians
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, build_covariance
from .encoder import Encoder
from .visualization.encoder_visualizer_resplat_cfg import EncoderVisualizerReSplatCfg

import torchvision.transforms as T
import torch.nn.functional as F

from .unimatch.mv_unimatch import MultiViewUniMatch

from .point_transformer.layer import PlainPointTransformer, PointLinearWrapper, MultViewLowresAttn





from .layer import ResNetFeatureWarpper
from ..semantic_boundary import (
    confidence_gated_boundary_features,
    rgb_to_soft_boundary,
)
from ..multiview_boundary import (
    boundary_consensus_reliability_feature,
    multiview_boundary_consensus_confidence,
    multiview_boundary_displacement_candidates,
    semantic_boundary_source_mask,
    signed_boundary_consensus_feature,
)
from ..semantic_routing import apply_semantic_parameter_routes
from ..semantic_selective_refinement import apply_selective_geometry_residual
from ..semantic_initialization import (
    SemanticGeometryAdapter,
    boundary_proximity_features,
)

@dataclass
class EncoderReSplatCfg:
    # Core settings
    name: Literal["resplat"]
    num_depth_candidates: int
    visualizer: EncoderVisualizerReSplatCfg
    gaussian_adapter: GaussianAdapterCfg

    # UniMatch / cost volume
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int

    # Multi-view stereo
    num_scales: int
    upsample_factor: int
    lowest_feature_resolution: int
    depth_unet_channels: int
    grid_sample_disable_cudnn: bool
    local_mv_match: int

    # Color branch
    gaussian_regressor_channels: int

    # Depth supervision and output
    supervise_intermediate_depth: bool
    return_depth: bool
    sample_log_depth: bool
    bilinear_upsample_depth: bool
    no_upsample_depth: bool

    # Monodepth backbone
    monodepth_vit_type: str

    # Point Transformer (init)
    attn_proj_channels: int | None
    knn_samples: int
    num_blocks: int
    init_use_local_knn: bool
    init_local_knn_spatial_radius: int
    init_local_knn_num_neighbor_views: int
    init_local_knn_cross_view_radius: int

    # Latent Gaussians
    latent_downsample: int
    fixed_latent_size: bool

    # Multiple Gaussians per point
    init_gaussian_multiple: int
    refine_same_num_points: bool

    # Handle high resolution images
    depth_pred_half_res: bool
    no_crop_image: bool

    # Iterative refinement
    num_refine: int
    train_min_refine: int
    train_max_refine: int
    num_basic_refine_blocks: int

    # Refinement state
    state_channels: int

    # Refinement update module
    update_attn_proj_channels: int | None
    refine_knn_samples: int
    refine_use_local_knn: bool
    refine_local_knn_spatial_radius: int
    refine_local_knn_num_neighbor_views: int
    refine_local_knn_cross_view_radius: int

    # Render error multi-view attention
    render_error_mv_attn_blocks: int
    use_semantic_boundary_feedback: bool
    semantic_boundary_gain: float
    semantic_boundary_confidence_floor: float
    semantic_boundary_feedback_scale: float
    semantic_boundary_feature_mode: Literal[
        "residual",
        "residual_gradient",
        "residual_alignment",
        "residual_alignment_consensus",
        "residual_alignment_consensus_gated",
        "residual_alignment_consensus_dual",
        "residual_alignment_consensus_dual_warmup",
        "residual_alignment_displacement",
    ]
    semantic_boundary_alignment_radius: int
    semantic_boundary_alignment_sigma: float
    use_multiview_boundary_consensus: bool
    multiview_boundary_radius: int
    multiview_boundary_depth_relative_tolerance: float
    multiview_boundary_min_support_views: float
    multiview_boundary_blend: float
    semantic_structure_hidden_channels: int
    semantic_structure_gate_bias: float
    semantic_structure_warmup_steps: int
    semantic_structure_ramp_steps: int
    multiview_boundary_displacement_radius: int
    multiview_boundary_displacement_source_radius: int
    multiview_boundary_displacement_diagnostic_path: Optional[str]
    use_semantic_parameter_routing: bool
    semantic_parameter_route_hidden_channels: int
    semantic_parameter_route_gain: float
    semantic_parameter_route_appearance: bool
    use_semantic_selective_refinement: bool
    semantic_selective_hidden_channels: int
    semantic_selective_gate_bias: float
    semantic_selective_gain: float
    semantic_selective_radius: int
    use_semantic_gaussian_init: bool
    semantic_init_hidden_channels: int
    semantic_init_gate_bias: float
    semantic_init_proximity_radius: int

    # AMP (automatic mixed precision)
    use_amp: bool
    pt_head_amp: bool
    pt_update_amp: bool

    # Checkpointing
    use_checkpointing: bool
    init_use_checkpointing: bool
    recurrent_use_checkpointing: bool



def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def create_init_point_transformer(cfg, channels):
    """Create KNN-based PlainPointTransformer for initial Gaussian prediction."""
    return PlainPointTransformer(
        channels, cfg.knn_samples,
        num_blocks=cfg.num_blocks,
        attn_proj_channels=cfg.attn_proj_channels,
        init_use_checkpointing=cfg.init_use_checkpointing,
        with_mv_attn=True,
        with_mv_attn_lowres=True,
        use_local_knn=cfg.init_use_local_knn,
        local_knn_spatial_radius=cfg.init_local_knn_spatial_radius,
        local_knn_num_neighbor_views=cfg.init_local_knn_num_neighbor_views,
        local_knn_cross_view_radius=cfg.init_local_knn_cross_view_radius,
    )


class EncoderReSplat(Encoder[EncoderReSplatCfg]):
    def __init__(self, cfg: EncoderReSplatCfg) -> None:
        super().__init__(cfg)

        self.depth_predictor = MultiViewUniMatch(
            num_scales=cfg.num_scales,
            upsample_factor=cfg.upsample_factor,
            lowest_feature_resolution=cfg.lowest_feature_resolution,
            num_depth_candidates=cfg.num_depth_candidates,
            vit_type=cfg.monodepth_vit_type,
            unet_channels=cfg.depth_unet_channels,
            grid_sample_disable_cudnn=cfg.grid_sample_disable_cudnn,
            sample_log_depth=self.cfg.sample_log_depth,
            bilinear_upsample_depth=self.cfg.bilinear_upsample_depth,
            no_upsample_depth=self.cfg.no_upsample_depth,
            use_amp=self.cfg.use_amp,
            return_raw_mono_features=True,
        )

        # upsample features to the original resolution
        model_configs = {
            'vits': {'in_channels': 384, 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'in_channels': 768, 'features': 96, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'in_channels': 1024, 'features': 128, 'out_channels': [128, 256, 512, 1024]},
        }

        # simply concat all the features
        if self.cfg.latent_downsample == 4:
            feature_upsampler_channels = model_configs[self.cfg.monodepth_vit_type]['in_channels'] // 4 + 128 + 64 + 96 + 128
        elif self.cfg.latent_downsample == 2:
            if self.cfg.lowest_feature_resolution == 8:
                feature_upsampler_channels = model_configs[self.cfg.monodepth_vit_type]['in_channels'] // 64 * 4 + 128 // 16 + 64 + 96 // 4 + 128 // 16
            else:
                feature_upsampler_channels = model_configs[self.cfg.monodepth_vit_type]['in_channels'] // 64 * 4 + 128 // 4 + 64 + 96 + 128 // 4
        elif self.cfg.latent_downsample == 8:
            # align feature channels for both downsample 4 and 8 such that the model weights can be shared
            if self.cfg.fixed_latent_size:
                feature_upsampler_channels = model_configs[self.cfg.monodepth_vit_type]['in_channels'] // 4 + 128 + 64 + 96 + 128
            else:
                feature_upsampler_channels = model_configs[self.cfg.monodepth_vit_type]['in_channels'] + 128 + 64 + 96 + 128
        else:
            raise NotImplementedError
        
        # gaussians adapter
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        # concat(img, depth, match_prob, features)
        in_channels = 3 + 1 + 1 + feature_upsampler_channels
        channels = self.cfg.gaussian_regressor_channels

        # image unshuffle
        if self.cfg.fixed_latent_size:
            # fixed patch size 4
            in_channels = in_channels - 3 + 3 * (4 ** 2)
        else:
            in_channels = in_channels - 3 + 3 * (self.cfg.latent_downsample ** 2)

        # gaussian regressor
        modules = [
                    nn.Conv2d(in_channels, channels, 3, 1, 1),
                    nn.GELU(),
                    nn.Conv2d(channels, channels, 3, 1, 1),
                ]

        self.gaussian_regressor = nn.Sequential(*modules)

        # predict gaussian parameters: scale, q, sh, offset, opacity
        # d_in: (scale, q, sh)
        num_gaussian_parameters = self.gaussian_adapter.d_in + 2 + 1

        # gaussian head input channels
        # concat(img, features, regressor_out, match_prob)
        in_channels = 3 + feature_upsampler_channels + channels + 1

        # image unshuffle
        if self.cfg.fixed_latent_size:
            in_channels = in_channels - 3 + 3 * (4 ** 2)
        else:
            in_channels = in_channels - 3 + 3 * (self.cfg.latent_downsample ** 2)

        channels = self.cfg.gaussian_regressor_channels

        self.proj = nn.Linear(in_channels, channels)

        if self.cfg.use_semantic_gaussian_init:
            # Preserve controlled-ablation data sampling when the identity
            # adapter is enabled.
            with torch.random.fork_rng(devices=[]):
                self.semantic_init_adapter = SemanticGeometryAdapter(
                    channels,
                    hidden_channels=self.cfg.semantic_init_hidden_channels,
                    gate_bias=self.cfg.semantic_init_gate_bias,
                )

        # Create PT head using factory function (KNN attention)
        self.pt = create_init_point_transformer(self.cfg, channels)

        out_channels = channels

        # multiple gaussians per latent
        if self.cfg.init_gaussian_multiple > 1:
            num_gaussian_parameters *= self.cfg.init_gaussian_multiple

        self.gaussian_head = nn.Sequential(
            nn.Linear(out_channels, num_gaussian_parameters),
            nn.GELU(),
            nn.Linear(num_gaussian_parameters, num_gaussian_parameters)
        )

        # random initialize rotations: first part
        num_rotation_params = 4 * self.cfg.init_gaussian_multiple

        # zero init other remaining params
        nn.init.zeros_(self.gaussian_head[-1].weight[num_rotation_params:])
        nn.init.zeros_(self.gaussian_head[-1].bias[num_rotation_params:])

        if self.cfg.num_refine > 0:
            self.state_channels = 256
            if self.cfg.state_channels > 0:
                self.state_channels = self.cfg.state_channels

            in_channels = self.cfg.gaussian_regressor_channels

            self.update_proj = nn.Conv2d(in_channels, self.state_channels, 1)

            if self.cfg.init_gaussian_multiple > 1:
                num_gaussian_parameters = num_gaussian_parameters // self.cfg.init_gaussian_multiple

            # no pixel offset
            num_gaussian_parameters -= 2

            # concat(prev_gaussians, point cloud, state, rendering error)
            if self.cfg.fixed_latent_size:
                in_channels = (num_gaussian_parameters + 3) * self.cfg.init_gaussian_multiple + self.state_channels + 3 * 4 ** 2
            else:
                in_channels = (num_gaussian_parameters + 3) * self.cfg.init_gaussian_multiple + self.state_channels + 3 * self.cfg.latent_downsample ** 2
            # ResNet-18 feature channels: 3 scales (1/2, 1/4, 1/8) → 64+64+128=256
            resnet_channels = 64 + 64 + 128  # ResNet-18

            if self.cfg.refine_same_num_points:
                in_channels = (num_gaussian_parameters + 3) + self.state_channels + resnet_channels
            else:
                in_channels = (num_gaussian_parameters + 3) * self.cfg.init_gaussian_multiple + self.state_channels + resnet_channels

            if self.cfg.init_gaussian_multiple == 4:  # re10k
                self.update_rgb_error_proj = nn.Sequential(
                    nn.Linear(3, resnet_channels),
                    nn.LayerNorm(resnet_channels)
                )
            else:
                self.update_rgb_error_proj = nn.Sequential(
                    nn.Linear(3 * self.cfg.latent_downsample ** 2, resnet_channels),
                    nn.LayerNorm(resnet_channels)
                )

            if self.cfg.use_semantic_boundary_feedback:
                semantic_channels = {
                    "residual": 1,
                    "residual_gradient": 3,
                    "residual_alignment": 3,
                    "residual_alignment_consensus": 4,
                    "residual_alignment_consensus_gated": 4,
                    "residual_alignment_consensus_dual": 4,
                    "residual_alignment_consensus_dual_warmup": 4,
                    "residual_alignment_displacement": 7,
                }[self.cfg.semantic_boundary_feature_mode]
                boundary_channels = (
                    semantic_channels
                    if self.cfg.init_gaussian_multiple == 4
                    else semantic_channels * self.cfg.latent_downsample ** 2
                )
                # Keep ablation data sampling identical: module construction must
                # not advance the global RNG used by the iterable dataset.
                gated_structure = (
                    self.cfg.semantic_boundary_feature_mode
                    in {
                        "residual_alignment_consensus_gated",
                        "residual_alignment_consensus_dual",
                        "residual_alignment_consensus_dual_warmup",
                        "residual_alignment_displacement",
                    }
                )
                dual_structure = (
                    self.cfg.semantic_boundary_feature_mode
                    in {
                        "residual_alignment_consensus_dual",
                        "residual_alignment_consensus_dual_warmup",
                        "residual_alignment_displacement",
                    }
                )
                with torch.random.fork_rng(devices=[]):
                    if gated_structure:
                        hidden = self.cfg.semantic_structure_hidden_channels
                        self.update_boundary_structure_encoder = nn.Sequential(
                            nn.Linear(boundary_channels, hidden),
                            nn.GELU(),
                            nn.Linear(hidden, resnet_channels),
                            nn.LayerNorm(resnet_channels),
                        )
                        self.update_boundary_gate = nn.Sequential(
                            nn.Linear(2 * resnet_channels, hidden),
                            nn.GELU(),
                            nn.Linear(hidden, 1),
                        )
                        if dual_structure:
                            base_boundary_channels = (
                                3
                                if self.cfg.init_gaussian_multiple == 4
                                else 3 * self.cfg.latent_downsample ** 2
                            )
                            self.update_boundary_error_proj = nn.Sequential(
                                nn.Linear(base_boundary_channels, resnet_channels),
                                nn.LayerNorm(resnet_channels),
                            )
                    else:
                        self.update_boundary_error_proj = nn.Sequential(
                            nn.Linear(boundary_channels, resnet_channels),
                            nn.LayerNorm(resnet_channels),
                        )
                if gated_structure:
                    nn.init.zeros_(self.update_boundary_structure_encoder[2].weight)
                    nn.init.zeros_(self.update_boundary_structure_encoder[2].bias)
                    nn.init.zeros_(self.update_boundary_gate[2].weight)
                    nn.init.constant_(
                        self.update_boundary_gate[2].bias,
                        self.cfg.semantic_structure_gate_bias,
                    )
                    if dual_structure:
                        nn.init.zeros_(self.update_boundary_error_proj[0].weight)
                        nn.init.zeros_(self.update_boundary_error_proj[0].bias)
                else:
                    nn.init.zeros_(self.update_boundary_error_proj[0].weight)
                    nn.init.zeros_(self.update_boundary_error_proj[0].bias)

                if self.cfg.use_semantic_parameter_routing:
                    route_hidden = self.cfg.semantic_parameter_route_hidden_channels
                    route_channels = (
                        2 if self.cfg.semantic_parameter_route_appearance else 1
                    )
                    with torch.random.fork_rng(devices=[]):
                        self.update_semantic_parameter_route_head = nn.Sequential(
                            nn.Linear(resnet_channels, route_hidden),
                            nn.GELU(),
                            nn.Linear(route_hidden, route_channels),
                        )
                    nn.init.zeros_(
                        self.update_semantic_parameter_route_head[-1].weight
                    )
                    nn.init.zeros_(
                        self.update_semantic_parameter_route_head[-1].bias
                    )

            out_channels = num_gaussian_parameters + 3

            channels = self.state_channels

            if self.cfg.use_semantic_selective_refinement:
                selective_hidden = self.cfg.semantic_selective_hidden_channels
                selective_repeat = (
                    self.cfg.init_gaussian_multiple
                    if not self.cfg.refine_same_num_points
                    else 1
                )
                with torch.random.fork_rng(devices=[]):
                    self.update_semantic_selective_head = nn.Sequential(
                        nn.Linear(channels + resnet_channels, selective_hidden),
                        nn.GELU(),
                        nn.Linear(selective_hidden, 7 * selective_repeat),
                    )
                nn.init.zeros_(self.update_semantic_selective_head[-1].weight)
                nn.init.zeros_(self.update_semantic_selective_head[-1].bias)
                with torch.no_grad():
                    self.update_semantic_selective_head[-1].bias[
                        :selective_repeat
                    ].fill_(self.cfg.semantic_selective_gate_bias)

            # Update module (kNN attention)
            self.update_module = nn.Sequential(
                PointLinearWrapper(in_channels, channels),
                PlainPointTransformer(channels, self.cfg.refine_knn_samples,
                    num_blocks=self.cfg.num_basic_refine_blocks,
                    attn_proj_channels=self.cfg.update_attn_proj_channels or self.cfg.attn_proj_channels,
                    use_checkpointing=self.cfg.recurrent_use_checkpointing,
                    use_local_knn=self.cfg.refine_use_local_knn,
                    local_knn_spatial_radius=self.cfg.refine_local_knn_spatial_radius,
                    local_knn_num_neighbor_views=self.cfg.refine_local_knn_num_neighbor_views,
                    local_knn_cross_view_radius=self.cfg.refine_local_knn_cross_view_radius,
                )
            )

            if not self.cfg.refine_same_num_points:
                out_channels = out_channels * self.cfg.init_gaussian_multiple

            # Update head
            self.update_head = nn.Sequential(
                nn.Linear(channels, channels),
                nn.GELU(),
                nn.Linear(channels, channels),
                nn.GELU(),
                nn.Linear(channels, channels),
                nn.GELU(),
                nn.Linear(channels, out_channels)
            )

            # init the delta as 0
            nn.init.zeros_(self.update_head[-1].weight)
            nn.init.zeros_(self.update_head[-1].bias)

            # ResNet-18 feature extractor (cached)
            self.update_feature = ResNetFeatureWarpper(
                shallow_resnet_feature=False,
                resnet_layers=18,
            )

            # freeze resnet
            self.update_feature.eval()
            for params in self.update_feature.parameters():
                params.requires_grad = False

            # Multi-view attention on render error
            in_channels = resnet_channels

            self.update_error_attn = nn.ModuleList([
                MultViewLowresAttn(in_channels)
                for _ in range(self.cfg.render_error_mv_attn_blocks)
            ])


    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
        renderer=None,
    ):
        self._semantic_global_step = global_step
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        if v > 3:
            with torch.no_grad():
                xyzs = context["extrinsics"][:, :, :3, -1].detach()
                cameras_dist_matrix = torch.cdist(xyzs, xyzs, p=2)
                cameras_dist_index = torch.argsort(cameras_dist_matrix)

                cameras_dist_index = cameras_dist_index[:, :, :(self.cfg.local_mv_match + 1)]
        else:
            cameras_dist_index = None

        # depth prediction
        if self.cfg.depth_pred_half_res:
            half_img = rearrange(context["image"], "b v c h w -> (b v) c h w")
            half_img = F.interpolate(half_img, scale_factor=0.5, mode='bilinear', align_corners=True)
            half_img = rearrange(half_img, "(b v) c h w -> b v c h w", b=b, v=v)

            results_dict = self.depth_predictor(
                half_img,
                attn_splits_list=[2],
                min_depth=1. / context["far"],
                max_depth=1. / context["near"],
                intrinsics=context["intrinsics"],
                extrinsics=context["extrinsics"],
                nn_matrix=cameras_dist_index,
            )

            # upsample depth to the original resolution
            for key in results_dict.keys():
                # NOTE: no need to upsample depth since depth later is in the low resolution
                if key != 'depth_preds':
                    for i in range(len(results_dict[key])):
                        results_dict[key][i] = F.interpolate(results_dict[key][i], scale_factor=2, mode='bilinear', align_corners=True)

        else:
            results_dict = self.depth_predictor(
                context["image"],
                attn_splits_list=[2],
                min_depth=1. / context["far"],
                max_depth=1. / context["near"],
                intrinsics=context["intrinsics"],
                extrinsics=context["extrinsics"],
                nn_matrix=cameras_dist_index,
            )

        # list of [B, V, H, W], with all the intermediate depths
        depth_preds = results_dict['depth_preds']

        # [B, V, H, W]
        depth = depth_preds[-1]

        # features [BV, C, H, W]
        # concat all features
        assert self.cfg.num_scales == 1

        # use pixelshuffle and pixelunshuffle to align all feature resolutions
        # first resize the mono features to 1/16
        mono_features = [F.interpolate(x, size=(h // 16, w // 16), mode='bilinear', align_corners=True) for x in results_dict['raw_mono_features']]
        if self.cfg.fixed_latent_size:
            scale_factor = 4
            mono_features = [F.pixel_shuffle(x, upscale_factor=scale_factor) for x in mono_features]
            mono_features = torch.cat(mono_features, dim=1)  # channel: 384 / 16 * 4

            if self.cfg.latent_downsample == 8:
                mono_features = F.interpolate(mono_features, scale_factor=0.5, mode='bilinear', align_corners=True)
        else:
            if self.cfg.latent_downsample == 4:
                scale_factor = 4
                mono_features = [F.pixel_shuffle(x, upscale_factor=scale_factor) for x in mono_features]
                mono_features = torch.cat(mono_features, dim=1)  # channel: 384 / 16 * 4
            elif self.cfg.latent_downsample == 2:
                scale_factor = 8
                mono_features = [F.pixel_shuffle(x, upscale_factor=scale_factor) for x in mono_features]
                mono_features = torch.cat(mono_features, dim=1)  # channel: 384 / 64 * 4
            elif self.cfg.latent_downsample == 8:
                scale_factor = 2
                mono_features = [F.pixel_shuffle(x, upscale_factor=scale_factor) for x in mono_features]
                mono_features = torch.cat(mono_features, dim=1)  # channel: 384 / 4 * 4
            else:
                raise NotImplementedError

        cnn_features = results_dict["features_cnn_all_scales"][::-1]

        if self.cfg.latent_downsample == 2:
            # use pixel shuffle to save channels
            if self.cfg.lowest_feature_resolution == 8:
                # 1/2, 1/4, 1/8
                cnn_features[1] = F.pixel_shuffle(cnn_features[1], upscale_factor=2)
                cnn_features[2] = F.pixel_shuffle(cnn_features[2], upscale_factor=4)
                # 64 + 96 // 4 + 128 // 16 = 96
                cnn_features = torch.cat(cnn_features, dim=1)

                # 128 // 16 = 8
                mv_features = results_dict["features_mv"][0]
                mv_features = F.pixel_shuffle(mv_features, upscale_factor=4)
            else:
                # 1/2, 1/2, 1/4
                cnn_features[2] = F.pixel_shuffle(cnn_features[2], upscale_factor=2)
                # 64 + 96 + 128 // 4
                cnn_features = torch.cat(cnn_features, dim=1)

                # 128 // 4
                mv_features = results_dict["features_mv"][0]
                mv_features = F.pixel_shuffle(mv_features, upscale_factor=2)
        else:
            # resize all cnn features to the latent resolution
            target_h, target_w = h // self.cfg.latent_downsample, w // self.cfg.latent_downsample
            for i in range(len(cnn_features)):
                cnn_features[i] = F.interpolate(cnn_features[i], size=(target_h, target_w), mode='bilinear', align_corners=True)
            cnn_features = torch.cat(cnn_features, dim=1)

            mv_features = results_dict["features_mv"][0]

            if mv_features.shape[-2] != target_h or mv_features.shape[-1] != target_w:
                mv_features = F.interpolate(mv_features, size=(target_h, target_w), mode='bilinear', align_corners=True)

        features = torch.cat((mono_features, cnn_features, mv_features), dim=1)

        # match prob from softmax
        # [BV, D, H, W] in feature resolution
        match_prob = results_dict['match_probs'][-1]
        match_prob = torch.max(match_prob, dim=1, keepdim=True)[
            0]  # [BV, 1, H, W]

        # unet input
        img_unshuffle = rearrange(context["image"], "b v c h w -> (b v) c h w")
        if self.cfg.fixed_latent_size:
            if self.cfg.latent_downsample == 8:
                img_unshuffle = F.interpolate(img_unshuffle, scale_factor=0.5, mode='area')

            img_unshuffle = F.pixel_unshuffle(img_unshuffle, downscale_factor=4)
        else:
            img_unshuffle = F.pixel_unshuffle(img_unshuffle, downscale_factor=self.cfg.latent_downsample)
        # depth is in the full resolution, downsample to latent depth
        if self.cfg.depth_pred_half_res:
            latent_depth = F.interpolate(depth, scale_factor=1./(self.cfg.latent_downsample // 2), mode='bilinear', align_corners=True)
        else:
            if self.cfg.no_upsample_depth:
                assert self.cfg.latent_downsample == 8 or self.cfg.latent_downsample == 4
                if self.cfg.latent_downsample == 8:
                    latent_depth = depth
                else:
                    # 1/8 depth to 1/4
                    latent_depth = F.interpolate(depth, scale_factor=2, mode='bilinear', align_corners=True)
            else:
                latent_depth = F.interpolate(depth, scale_factor=1./self.cfg.latent_downsample, mode='bilinear', align_corners=True)

        if match_prob.shape[-2:] != latent_depth.shape[-2]:
            match_prob = F.interpolate(
                match_prob, size=latent_depth.shape[-2:], mode='nearest')

        concat = torch.cat((
            img_unshuffle,
            rearrange(latent_depth, "b v h w -> (b v) () h w"),
            match_prob,
            features,
        ), dim=1)

        with torch.amp.autocast(device_type='cuda', enabled=self.cfg.use_amp, dtype=torch.bfloat16):
            out = self.gaussian_regressor(concat)

        concat = [out, img_unshuffle, features, match_prob]

        out = torch.cat(concat, dim=1)

        if self.cfg.num_refine > 0:
            # [BV, C, H, W]
            condition_features = out

        h, w = latent_depth.shape[-2:]
        with torch.amp.autocast(device_type='cuda', enabled=self.cfg.pt_head_amp, dtype=torch.bfloat16):
            tmp_feature = self.proj(rearrange(out, "bv c h w -> (bv h w) c"))
            if self.cfg.use_semantic_gaussian_init:
                if "boundary" not in context or "boundary_confidence" not in context:
                    raise KeyError(
                        "semantic Gaussian initialization requires context "
                        "boundary and boundary_confidence"
                    )
                semantic_features = boundary_proximity_features(
                    context["boundary"],
                    context["boundary_confidence"],
                    confidence_floor=self.cfg.semantic_boundary_confidence_floor,
                    max_radius=self.cfg.semantic_init_proximity_radius,
                )
                semantic_features = rearrange(
                    semantic_features, "b v c h w -> (b v) c h w"
                )
                tmp_feature_grid = rearrange(
                    tmp_feature, "(bv h w) c -> bv c h w", h=h, w=w
                )
                tmp_feature_grid, semantic_gate, semantic_residual = (
                    self.semantic_init_adapter(tmp_feature_grid, semantic_features)
                )
                tmp_feature = rearrange(
                    tmp_feature_grid, "bv c h w -> (bv h w) c"
                )
                if not hasattr(self, "_logged_semantic_init_stats"):
                    print(
                        "semantic Gaussian init adapter: "
                        f"gate={semantic_gate.mean().item():.4f}, "
                        f"residual={semantic_residual.abs().mean().item():.6f}"
                    )
                    self._logged_semantic_init_stats = True
        # get point cloud
        xy_ray, _ = sample_image_grid((h, w), out.device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")

        # [B, V, H*W, 1, 2]
        tmp_coords = xy_ray.unsqueeze(0).unsqueeze(0).repeat(b, v, 1, 1, 1)

        # [B, V, H*W, 1, 1]
        tmp_depth = rearrange(latent_depth, "b v h w -> b v (h w) () ()")

        # [B, V, 1, 1, 4, 4]
        tmp_extrinsics = context["extrinsics"].unsqueeze(2).unsqueeze(2)
        # [B, V, 1, 1, 3, 3]
        tmp_intrinsics = context["intrinsics"].unsqueeze(2).unsqueeze(2)

        # [B, V, H*W, 1, 3]
        origins, directions = get_world_rays(tmp_coords, tmp_extrinsics, tmp_intrinsics)
        point_cloud = origins + directions * tmp_depth

        offset = torch.tensor([v * h * w * i for i in range(1, b + 1)]).to(depth.device)
        point_cloud = rearrange(point_cloud, "b v h w c -> (b v h w) c")

        with torch.amp.autocast(device_type='cuda', enabled=self.cfg.pt_head_amp, dtype=torch.bfloat16):
            pt_kwargs = dict(b=b, v=v, h=h, w=w)
            if self.cfg.init_use_local_knn:
                pt_kwargs["extrinsics"] = context["extrinsics"][0]  # [V, 4, 4]
                pt_kwargs["intrinsics"] = context["intrinsics"][0]  # [V, 3, 3]
            pt_output = self.pt((point_cloud, tmp_feature, offset), **pt_kwargs)
            out = tmp_feature + pt_output

            condition_features = rearrange(out, "(bv h w) c -> bv c h w", h=h, w=w)

        with torch.amp.autocast(device_type='cuda', enabled=self.cfg.pt_head_amp, dtype=torch.bfloat16):
            out = self.gaussian_head(out)

        point_cloud = rearrange(point_cloud, "(b v h w) c -> b v (h w) () () c", b=b, v=v, h=h, w=w)

        gaussians = rearrange(out, "(b v h w) c -> (b v) c h w", b=b, h=h, w=w)

        # [BV, C, H, W]
        gaussians = gaussians.float()

        if self.cfg.init_gaussian_multiple > 1:
            # hard coded for now
            if self.cfg.init_gaussian_multiple == 4:
                if self.cfg.latent_downsample == 4:
                    # resize full resolution depth
                    depths = F.interpolate(depth, scale_factor=0.5, mode='bilinear', align_corners=True)
                elif self.cfg.latent_downsample == 8:
                    depths = F.interpolate(depth, scale_factor=0.25, mode='bilinear', align_corners=True)
                elif self.cfg.latent_downsample == 2:
                    depths = depth
                else:
                    raise NotImplementedError
            elif self.cfg.init_gaussian_multiple == 16:
                if self.cfg.latent_downsample == 4:
                    depths = depth
                elif self.cfg.latent_downsample == 8:
                    depths = F.interpolate(depth, scale_factor=0.5, mode='bilinear', align_corners=True)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

            depths = rearrange(depths, "b v h w -> b v (h w) () ()")
        else:
            depths = rearrange(latent_depth, "b v h w -> b v (h w) () ()")

        gaussians = rearrange(gaussians, "(b v) c h w -> b v c h w", b=b, v=v)

        # [B, V, H*W, 84]
        raw_gaussians = rearrange(
            gaussians, "b v c h w -> b v (h w) c")

        if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:

            # supervise all the intermediate depth predictions
            num_depths = len(depth_preds)

            # [B, V, H*W, 1, 1]
            intermediate_depths = torch.cat(
                depth_preds[:(num_depths - 1)], dim=0)

            intermediate_depths = rearrange(
                intermediate_depths, "b v h w -> b v (h w) () ()")

            # concat in the batch dim
            depths = torch.cat((intermediate_depths, depths), dim=0)

            # shared color head
            raw_gaussians = torch.cat(
                [raw_gaussians] * num_depths, dim=0)

            b *= num_depths

        # [B, V, H*W, C]
        repeat = self.cfg.init_gaussian_multiple
        num_sh = self.gaussian_adapter.d_sh

        rotations_unnorm, scales, opacities_raw, offset, sh = raw_gaussians.split(
            [4 * repeat, 3 * repeat, 1 * repeat, 2 * repeat, 3 * num_sh * repeat],
            dim=-1,
        )

        latent_h, latent_w = gaussians.shape[-2:]

        if repeat > 1:
            # reshape all the gaussian parameters
            r = int(np.sqrt(repeat))
            rotations_unnorm = rearrange(rotations_unnorm, "b v (h w) (c x y) -> b v (h x w y) c", h=latent_h, w=latent_w, x=r, y=r)
            scales = rearrange(scales, "b v (h w) (c x y) -> b v (h x w y) c", h=latent_h, w=latent_w, x=r, y=r)
            opacities_raw = rearrange(opacities_raw, "b v (h w) (c x y) -> b v (h x w y) c", h=latent_h, w=latent_w, x=r, y=r)
            offset = rearrange(offset, "b v (h w) (c x y) -> b v (h x w y) c", h=latent_h, w=latent_w, x=r, y=r)
            sh = rearrange(sh, "b v (h w) (c x y) -> b v (h x w y) c", h=latent_h, w=latent_w, x=r, y=r)

        opacities = opacities_raw.sigmoid()  # [B, V, H*W*K, 1]

        if self.cfg.latent_downsample == 4 and self.cfg.init_gaussian_multiple == 4:
            scale_factor = 2
        elif self.cfg.latent_downsample == 2 and self.cfg.init_gaussian_multiple == 4:
            scale_factor = 2
        elif self.cfg.latent_downsample == 4 and self.cfg.init_gaussian_multiple == 16:
            scale_factor = 4
        elif self.cfg.latent_downsample == 8 and self.cfg.init_gaussian_multiple == 4:
            scale_factor = 2
        elif self.cfg.latent_downsample == 8 and self.cfg.init_gaussian_multiple == 16:
            scale_factor = 4
        else:
            scale_factor = 1

        h, w = latent_h * scale_factor, latent_w * scale_factor

        # unproject depth
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")

        offset_xy = offset.sigmoid().unsqueeze(-2)  # [B, V, H*W, 1, 2]

        pixel_size = 1 / \
            torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size

        sh_input_images = rearrange(context["image"], "b v c h w -> (b v) c h w")
        if self.cfg.latent_downsample == 4 and self.cfg.init_gaussian_multiple == 4:
            sh_input_images = F.interpolate(sh_input_images, scale_factor=0.5, mode='area')
        elif self.cfg.latent_downsample == 4 and self.cfg.init_gaussian_multiple == 16:
            pass
        elif self.cfg.latent_downsample == 2 and self.cfg.init_gaussian_multiple == 4:
            pass
        elif self.cfg.latent_downsample == 8 and self.cfg.init_gaussian_multiple == 4:
            sh_input_images = F.interpolate(sh_input_images, scale_factor=0.25, mode='area')
        elif self.cfg.latent_downsample == 8 and self.cfg.init_gaussian_multiple == 16:
            sh_input_images = F.interpolate(sh_input_images, scale_factor=0.5, mode='area')
        else:
            sh_input_images = F.interpolate(sh_input_images, scale_factor=1./self.cfg.latent_downsample, mode='area')

        sh_input_images = rearrange(sh_input_images, "(b v) c h w -> b v c h w", b=b, v=v)

        if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:
            raise NotImplementedError

        # build gaussians
        # scale
        scales = torch.clamp(F.softplus(scales - self.cfg.gaussian_adapter.exp_scale_bias),
            min=self.cfg.gaussian_adapter.clamp_min_scale,
            max=self.cfg.gaussian_adapter.gaussian_scale_max
            )

        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations_unnorm / (rotations_unnorm.norm(dim=-1, keepdim=True) + 1e-8)

        # Create world-space covariance matrices.
        covariances = build_covariance(scales, rotations)
        c2w_rotations = context["extrinsics"][..., :3, :3].unsqueeze(2)  # [B, V, 1, 3, 3]
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)

        # means
        origins, directions = get_world_rays(xy_ray,
            context["extrinsics"].unsqueeze(2).unsqueeze(2),
            context["intrinsics"].unsqueeze(2).unsqueeze(2))
        means = origins + directions * depths

        # sh: [B, V, HW, 3, SH]
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3).clone()

        # [B, V, H*W, 3]
        sh_input_images = rearrange(sh_input_images, "b v c h w -> b v (h w) c")
        # init sh with input images
        sh[..., 0] = sh[..., 0] + RGB2SH(sh_input_images)

        gaussians = Gaussians(
            rearrange(means, "b v r spp xyz -> b (v r spp) xyz"),
            rearrange(covariances, "b v r i j -> b (v r) i j"),
            rearrange(sh, "b v r c d_sh -> b (v r) c d_sh"),
            rearrange(opacities, "b v r spp -> b (v r spp)"),
            scales=rearrange(scales, "b v r xyz -> b (v r) xyz"),
            rotations=rearrange(rotations, "b v r wxyz -> b (v r) wxyz"),
            rotations_unnorm=rearrange(rotations_unnorm, "b v r wxyz -> b (v r) wxyz")
        )

        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )
            # Also store full-resolution depth (before latent downsampling)
            fullres_depth = depth_preds[-1]
            if fullres_depth.shape[-2:] != context["image"].shape[-2:]:
                fullres_depth = F.interpolate(
                    fullres_depth, size=context["image"].shape[-2:],
                    mode='bilinear', align_corners=True,
                )
            visualization_dump["depth_fullres"] = fullres_depth

        if self.cfg.return_depth:
            # original depth predictions from the depth model
            if len(depth_preds) > 1:
                depths = torch.cat(depth_preds, dim=0)  # concat in the batch dim
            else:
                depths = depth_preds[-1]

            if depths.shape[-2:] != context["image"].shape[-2:]:
                # depths can be at lower resolution since we predict latent
                depths = F.interpolate(
                    depths, size=context["image"].shape[-2:], mode='bilinear', align_corners=True)

            results = {
                "gaussians": gaussians,
                "depths": depths
            }

            if self.cfg.num_refine > 0:
                results.update({
                    "condition_features": condition_features
                })

            return results

        return gaussians


    def forward_update(self,
        context,
        target,
        condition_features,
        init_gaussians,
        renderer,
        context_remain=None,
        ):
        render_output = []
        render_input_views = []
        gaussian_output = []
        # check if the delta gaussian means and scales are becoming smaller over time
        delta_means_all = []
        delta_scales_all = []

        b, v, _, h, w = context["image"].shape

        prev_gaussians = init_gaussians

        prev_means = prev_gaussians.means.detach()  # [B, N, 3]
        prev_scales = prev_gaussians.scales.detach()  # [B, N, 3]
        # use unnormalized rotations since we are going to refine the unnormed rotations
        prev_rotations_unnorm = prev_gaussians.rotations_unnorm.detach()  # [B, N, 4]
        # before sigmoid, epe is necessary, otherwise might be nan
        prev_opacities_raw = torch.logit(prev_gaussians.opacities.detach(), eps=1e-6)  # [B, N]
        prev_shs = prev_gaussians.harmonics.detach()  # [B, N, 3, 9]

        prev_opacities_raw = prev_opacities_raw.unsqueeze(-1)  # [B, N, 1]
        prev_shs = rearrange(prev_shs, "b n c x -> b n (c x)")  # [B, N, C]

        with torch.amp.autocast(device_type='cuda', enabled=self.cfg.pt_update_amp, dtype=torch.bfloat16):
            state = self.update_proj(condition_features.detach())

        if self.cfg.init_gaussian_multiple == 4 and self.cfg.refine_same_num_points:
            state = F.interpolate(state, scale_factor=2, mode='bilinear', align_corners=True)
        elif self.cfg.init_gaussian_multiple == 16 and self.cfg.refine_same_num_points:
            state = F.interpolate(state, scale_factor=4, mode='bilinear', align_corners=True)
        else:
            pass

        # [B, N, C]
        state = rearrange(state, "(b v) c h w -> b (v h w) c", b=b, v=v)

        num_refine = self.cfg.num_refine
        if self.training and self.cfg.train_min_refine > 0 and self.cfg.train_max_refine > 0:
            num_refine = np.random.randint(self.cfg.train_min_refine, self.cfg.train_max_refine + 1)

        tmp_state = rearrange(state, "b n c -> (b n) c")
        init_state = tmp_state

        # render input views
        input_render = renderer.forward(
                prev_gaussians,
                context["extrinsics"],
                context["intrinsics"],
                context["near"],
                context["far"],
                (h, w),
            )

        input_view_features = None
        cached_knn_idx = None

        for i in range(num_refine):
            semantic_parameter_routes = None
            semantic_selective_features = None
            semantic_selective_active = None
            input0 = rearrange(input_render.color, "b v c h w -> (b v) c h w")
            gt_input = context["image"]
            input1 = rearrange(gt_input, "b v c h w -> (b v) c h w")

            transform = T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )

            if input_view_features is None:
                assert i == 0
                # first time: extract all features
                concat = torch.cat((input0, input1), dim=0)

                input_tensor = transform(concat)
                with torch.amp.autocast(device_type='cuda', enabled=self.cfg.pt_update_amp, dtype=torch.bfloat16):
                    # Extract features
                    with torch.no_grad():
                        features = self.update_feature(input_tensor)

                # align to the latent resolution
                latent_h, latent_w = h // self.cfg.latent_downsample, w // self.cfg.latent_downsample
                if self.cfg.init_gaussian_multiple == 4 and self.cfg.refine_same_num_points:
                    latent_h *= 2
                    latent_w *= 2
                elif self.cfg.init_gaussian_multiple == 16 and self.cfg.refine_same_num_points:
                    latent_h *= 4
                    latent_w *= 4
                else:
                    pass

                out = []
                for feat in features:
                    if feat.shape[-2:] != (latent_h, latent_w):
                        feat = F.interpolate(feat, size=(latent_h, latent_w), mode='bilinear', align_corners=True)

                    out.append(feat)

                all_features = torch.cat(out, dim=1)

                render_view_features = all_features[:input0.shape[0]]
                input_view_features = all_features[input0.shape[0]:]

            else:
                # only extract render view features
                with torch.amp.autocast(device_type='cuda', enabled=self.cfg.pt_update_amp, dtype=torch.bfloat16):
                    # Extract features
                    with torch.no_grad():
                        features = self.update_feature(transform(input0))

                # align to the latent resolution
                latent_h, latent_w = h // self.cfg.latent_downsample, w // self.cfg.latent_downsample
                if self.cfg.init_gaussian_multiple == 4 and self.cfg.refine_same_num_points:
                    latent_h *= 2
                    latent_w *= 2
                elif self.cfg.init_gaussian_multiple == 16 and self.cfg.refine_same_num_points:
                    latent_h *= 4
                    latent_w *= 4
                else:
                    pass

                out = []
                for feat in features:
                    if feat.shape[-2:] != (latent_h, latent_w):
                        feat = F.interpolate(feat, size=(latent_h, latent_w), mode='bilinear', align_corners=True)

                    out.append(feat)

                render_view_features = torch.cat(out, dim=1)

            corr = render_view_features - input_view_features

            input_render_error = rearrange(corr, "(b v) c h w -> b (v h w) c", b=b, v=v)

            # include both feature error and image error
            rgb_render_error = input_render.color - context["image"]
            rgb_render_error = rearrange(rgb_render_error, "b v c h w -> (b v) c h w")

            if self.cfg.init_gaussian_multiple != 4:  # re10k 256x256, no downsample
                rgb_render_error = F.pixel_unshuffle(rgb_render_error, downscale_factor=self.cfg.latent_downsample)

            rgb_render_error = rearrange(rgb_render_error, "(b v) c h w -> b (v h w) c", b=b, v=v)  # [B, N, C]

            rgb_render_error = self.update_rgb_error_proj(rgb_render_error)
            input_render_error = input_render_error + rgb_render_error

            if self.cfg.use_semantic_boundary_feedback:
                if "boundary" not in context or "boundary_confidence" not in context:
                    raise KeyError("Semantic feedback requires context boundary maps")
                boundary_confidence = context["boundary_confidence"]
                explicit_consensus_channel = (
                    self.cfg.semantic_boundary_feature_mode
                    in {
                        "residual_alignment_consensus",
                        "residual_alignment_consensus_gated",
                        "residual_alignment_consensus_dual",
                        "residual_alignment_consensus_dual_warmup",
                    }
                )
                displacement_mode = (
                    self.cfg.semantic_boundary_feature_mode
                    == "residual_alignment_displacement"
                )
                gated_structure = (
                    self.cfg.semantic_boundary_feature_mode
                    in {
                        "residual_alignment_consensus_gated",
                        "residual_alignment_consensus_dual",
                        "residual_alignment_consensus_dual_warmup",
                        "residual_alignment_displacement",
                    }
                )
                dual_structure = (
                    self.cfg.semantic_boundary_feature_mode
                    in {
                        "residual_alignment_consensus_dual",
                        "residual_alignment_consensus_dual_warmup",
                        "residual_alignment_displacement",
                    }
                )
                warmup_structure = (
                    self.cfg.semantic_boundary_feature_mode
                    == "residual_alignment_consensus_dual_warmup"
                )
                if (
                    self.cfg.use_multiview_boundary_consensus
                    or explicit_consensus_channel
                ):
                    with torch.no_grad():
                        consensus_confidence, boundary_consensus = (
                            multiview_boundary_consensus_confidence(
                                context["boundary"],
                                boundary_confidence,
                                input_render.depth,
                                context["extrinsics"],
                                context["intrinsics"],
                                radius=self.cfg.multiview_boundary_radius,
                                depth_relative_tolerance=(
                                    self.cfg.multiview_boundary_depth_relative_tolerance
                                ),
                                min_support_views=(
                                    self.cfg.multiview_boundary_min_support_views
                                ),
                                blend=self.cfg.multiview_boundary_blend,
                            )
                        )
                    # Stage 6 uses consensus to gate the original feedback.
                    # Stage 7A instead preserves that feedback and exposes
                    # consensus as an independent structural observation.
                    if not explicit_consensus_channel:
                        boundary_confidence = consensus_confidence
                    if self.training and not hasattr(
                        self, "_logged_multiview_boundary_stats"
                    ):
                        positive = context["boundary"] >= 0.5
                        supported = boundary_consensus[positive]
                        mean_support = (
                            supported.mean().item() if supported.numel() else 0.0
                        )
                        supported_fraction = (
                            (supported >= 0.5).float().mean().item()
                            if supported.numel()
                            else 0.0
                        )
                        print(
                            "multi-view boundary consensus: "
                            f"mean={mean_support:.4f}, "
                            f">=0.5={supported_fraction:.4f}"
                        )
                        self._logged_multiview_boundary_stats = True
                boundary_error = confidence_gated_boundary_features(
                    input_render.color,
                    context["boundary"],
                    boundary_confidence,
                    gain=self.cfg.semantic_boundary_gain,
                    confidence_floor=self.cfg.semantic_boundary_confidence_floor,
                    mode=(
                        "residual_alignment"
                        if explicit_consensus_channel or displacement_mode
                        else self.cfg.semantic_boundary_feature_mode
                    ),
                    alignment_radius=self.cfg.semantic_boundary_alignment_radius,
                    alignment_sigma=self.cfg.semantic_boundary_alignment_sigma,
                )
                if explicit_consensus_channel:
                    if gated_structure:
                        consensus_feature = boundary_consensus_reliability_feature(
                            context["boundary"],
                            context["boundary_confidence"],
                            boundary_consensus,
                        )
                    else:
                        consensus_feature = signed_boundary_consensus_feature(
                            context["boundary"],
                            context["boundary_confidence"],
                            boundary_consensus,
                        )
                    boundary_error = torch.cat(
                        (boundary_error, consensus_feature), dim=2
                    )
                elif displacement_mode:
                    with torch.no_grad():
                        rendered_boundary = rgb_to_soft_boundary(
                            input_render.color,
                            gain=self.cfg.semantic_boundary_gain,
                        )
                        rendered_boundary = semantic_boundary_source_mask(
                            rendered_boundary,
                            context["boundary"],
                            context["boundary_confidence"],
                            radius=(
                                self.cfg.multiview_boundary_displacement_source_radius
                            ),
                            confidence_floor=(
                                self.cfg.semantic_boundary_confidence_floor
                            ),
                        )
                        mv_dx, mv_dy, mv_confidence, mv_visibility = (
                            multiview_boundary_displacement_candidates(
                                rendered_boundary,
                                context["boundary"],
                                context["boundary_confidence"],
                                input_render.depth,
                                context["extrinsics"],
                                context["intrinsics"],
                                radius=(
                                    self.cfg.multiview_boundary_displacement_radius
                                ),
                                depth_relative_tolerance=(
                                    self.cfg.multiview_boundary_depth_relative_tolerance
                                ),
                            )
                        )
                    displacement_radius = float(
                        self.cfg.multiview_boundary_displacement_radius
                    )
                    displacement_features = torch.cat(
                        (
                            (mv_dx / displacement_radius).clamp(-1, 1),
                            (mv_dy / displacement_radius).clamp(-1, 1),
                            mv_confidence,
                            mv_visibility,
                        ),
                        dim=2,
                    )
                    boundary_error = torch.cat(
                        (boundary_error, displacement_features), dim=2
                    )
                    if not hasattr(self, "_logged_multiview_displacement_stats"):
                        active = rendered_boundary >= 0.1
                        supported = active & (mv_confidence > 0)
                        mean_magnitude = torch.sqrt(
                            mv_dx.square() + mv_dy.square()
                        )[supported]
                        print(
                            "multi-view boundary displacement: "
                            f"active={active.float().mean().item():.4f}, "
                            f"supported={supported.float().sum().item() / active.float().sum().clamp_min(1).item():.4f}, "
                            f"mean_px={mean_magnitude.mean().item() if mean_magnitude.numel() else 0.0:.4f}, "
                            f"confidence={mv_confidence[supported].mean().item() if supported.any() else 0.0:.4f}, "
                            f"visibility={mv_visibility[active].mean().item() if active.any() else 0.0:.4f}"
                        )
                        diagnostic_path = (
                            self.cfg.multiview_boundary_displacement_diagnostic_path
                        )
                        if diagnostic_path:
                            path = Path(diagnostic_path)
                            path.parent.mkdir(parents=True, exist_ok=True)
                            torch.save(
                                {
                                    "rendered_boundary": rendered_boundary[0, 0].cpu(),
                                    "teacher_boundary": context["boundary"][0, 0].cpu(),
                                    "dx": mv_dx[0, 0].cpu(),
                                    "dy": mv_dy[0, 0].cpu(),
                                    "confidence": mv_confidence[0, 0].cpu(),
                                    "visibility": mv_visibility[0, 0].cpu(),
                                },
                                path,
                            )
                        self._logged_multiview_displacement_stats = True
                if self.cfg.use_semantic_selective_refinement:
                    trusted_boundary = (
                        (context["boundary"] >= 0.5)
                        & (
                            context["boundary_confidence"]
                            >= self.cfg.semantic_boundary_confidence_floor
                        )
                    ).to(boundary_error.dtype)
                    trusted_boundary = rearrange(
                        trusted_boundary, "b v c h w -> (b v) c h w"
                    )
                    radius = self.cfg.semantic_selective_radius
                    semantic_selective_active = F.max_pool2d(
                        trusted_boundary,
                        kernel_size=2 * radius + 1,
                        stride=1,
                        padding=radius,
                    )
                    if self.cfg.init_gaussian_multiple != 4:
                        semantic_selective_active = F.pixel_unshuffle(
                            semantic_selective_active,
                            downscale_factor=self.cfg.latent_downsample,
                        ).amax(dim=1, keepdim=True)
                    semantic_selective_active = rearrange(
                        semantic_selective_active,
                        "(b v) 1 h w -> b (v h w) 1",
                        b=b,
                        v=v,
                    ) > 0
                    if not hasattr(self, "_logged_semantic_selective_stats"):
                        print(
                            "semantic selective refinement: "
                            f"active={semantic_selective_active.float().mean().item():.4f}"
                        )
                        self._logged_semantic_selective_stats = True

                boundary_error = rearrange(
                    boundary_error, "b v c h w -> (b v) c h w"
                )
                if self.cfg.init_gaussian_multiple != 4:
                    boundary_error = F.pixel_unshuffle(
                        boundary_error, downscale_factor=self.cfg.latent_downsample
                    )
                boundary_error = rearrange(
                    boundary_error, "(b v) c h w -> b (v h w) c", b=b, v=v
                )
                if gated_structure:
                    structure_error = self.update_boundary_structure_encoder(
                        boundary_error
                    )
                    structure_gate = torch.sigmoid(
                        self.update_boundary_gate(
                            torch.cat(
                                (input_render_error, structure_error), dim=-1
                            )
                        )
                    )
                    structure_schedule = 1.0
                    if warmup_structure and self.training:
                        ramp_step = (
                            getattr(self, "_semantic_global_step", 0)
                            - self.cfg.semantic_structure_warmup_steps
                        )
                        structure_schedule = max(
                            0.0,
                            min(
                                1.0,
                                ramp_step
                                / max(1, self.cfg.semantic_structure_ramp_steps),
                            ),
                        )
                    structure_feedback = (
                        structure_schedule * structure_gate * structure_error
                    )
                    if dual_structure:
                        base_channels = (
                            3
                            if self.cfg.init_gaussian_multiple == 4
                            else 3 * self.cfg.latent_downsample ** 2
                        )
                        structure_feedback = structure_feedback + (
                            self.update_boundary_error_proj(
                                boundary_error[..., :base_channels]
                            )
                        )
                    input_render_error = input_render_error + (
                        self.cfg.semantic_boundary_feedback_scale
                        * structure_feedback
                    )
                else:
                    boundary_feedback = self.update_boundary_error_proj(
                        boundary_error
                    )
                    input_render_error = input_render_error + (
                        self.cfg.semantic_boundary_feedback_scale
                        * boundary_feedback
                    )
                    if self.cfg.use_semantic_parameter_routing:
                        semantic_parameter_routes = (
                            self.update_semantic_parameter_route_head(
                                boundary_feedback
                            )
                        )
                    if self.cfg.use_semantic_selective_refinement:
                        semantic_selective_features = boundary_feedback

            # stop gradient for last predictions
            prev_gaussians_concat = torch.cat((
                prev_means.detach(),
                prev_scales.detach(),
                prev_rotations_unnorm.detach(),
                prev_opacities_raw.detach(),
                prev_shs.detach(),
            ), dim=-1)  # [B, N, C]

            # detach previous gaussians
            prev_means = prev_means.detach()
            prev_scales = prev_scales.detach()
            prev_rotations_unnorm = prev_rotations_unnorm.detach()
            prev_opacities_raw = prev_opacities_raw.detach()
            prev_shs = prev_shs.detach()

            latent_h = h // self.cfg.latent_downsample
            latent_w = w // self.cfg.latent_downsample

            # prepare pt input
            if self.cfg.init_gaussian_multiple == 4 and not self.cfg.refine_same_num_points:
                point_cloud = rearrange(prev_means, "b (v h w) c -> b v h w c",
                v=v, h=latent_h * 2, w=latent_w * 2,
                )
                tmp_batch_size = v * latent_h * latent_w
                # simply use uniform grid subsample of point cloud to reduce points
                point_cloud = point_cloud[:, :, ::2, ::2]
                point_cloud = rearrange(point_cloud, "b v h w c -> (b v h w) c")
            elif self.cfg.init_gaussian_multiple == 16 and not self.cfg.refine_same_num_points:
                point_cloud = rearrange(prev_means, "b (v h w) c -> b v h w c",
                v=v, h=latent_h * 4, w=latent_w * 4,
                )
                tmp_batch_size = v * latent_h * latent_w
                # simply use uniform grid subsample of point cloud to reduce points
                point_cloud = point_cloud[:, :, ::4, ::4]
                point_cloud = rearrange(point_cloud, "b v h w c -> (b v h w) c")
            else:
                point_cloud = rearrange(prev_means, "b n c -> (b n) c")
                tmp_batch_size = prev_means.shape[1]

            offset = torch.tensor([k * tmp_batch_size for k in range(1, b + 1)]).to(state.device)

            # reshape
            if self.cfg.init_gaussian_multiple == 4 and not self.cfg.refine_same_num_points:
                # gaussians are with more points, reshape
                tmp_gaussian = rearrange(prev_gaussians_concat, "b (v h x w y) c -> (b v h w) (c x y)", 
                    v=v, h=latent_h, w=latent_w, x=2, y=2)
            elif self.cfg.init_gaussian_multiple == 16 and not self.cfg.refine_same_num_points:
                tmp_gaussian = rearrange(prev_gaussians_concat, "b (v h x w y) c -> (b v h w) (c x y)", 
                    v=v, h=latent_h, w=latent_w, x=4, y=4)
            else:
                tmp_gaussian = rearrange(prev_gaussians_concat, "b n c -> (b n) c")

            # add global attention to exchange info across views
            with torch.amp.autocast(device_type='cuda', enabled=self.cfg.use_amp, dtype=torch.bfloat16):
                for blk in self.update_error_attn:
                    if self.cfg.refine_same_num_points:
                        # no downsample, for re10k 256
                        input_render_error = blk(input_render_error, v=v, h=h, w=w)
                    else:
                        input_render_error = blk(input_render_error,
                            v=v,
                            h=latent_h, w=latent_w)

            tmp_render_error = rearrange(input_render_error, "b n c -> (b n) c")

            with torch.amp.autocast(device_type='cuda', enabled=self.cfg.pt_update_amp, dtype=torch.bfloat16):
                concat = torch.cat((tmp_gaussian, tmp_state, tmp_render_error), dim=-1)

                refine_pt_kwargs = dict(b=b, v=v, h=latent_h, w=latent_w)
                if self.cfg.refine_use_local_knn:
                    refine_pt_kwargs["extrinsics"] = context["extrinsics"][0]
                    refine_pt_kwargs["intrinsics"] = context["intrinsics"][0]

                if cached_knn_idx is not None:
                    refine_pt_kwargs["cached_knn_idx"] = cached_knn_idx

                if self.cfg.use_checkpointing or self.cfg.recurrent_use_checkpointing:
                    def recurrent_chunk(tmp_state, tmp_gaussian, tmp_render_error, point_cloud, offset,
                                        *extra_args):
                        concat = torch.cat((tmp_gaussian, tmp_state, tmp_render_error), dim=-1)
                        pxo = self.update_module[0]([point_cloud, concat, offset])
                        tmp_state = self.update_module[1](pxo, **refine_pt_kwargs) + tmp_state
                        return tmp_state

                    tmp_state = torch.utils.checkpoint.checkpoint(
                        recurrent_chunk,
                        tmp_state, tmp_gaussian, tmp_render_error, point_cloud, offset,
                        use_reentrant=False,
                    )

                else:
                    pxo = self.update_module[0]([point_cloud, concat, offset])
                    tmp_state = self.update_module[1](pxo, **refine_pt_kwargs) + tmp_state

                # delta gaussian head
                delta_gaussians = self.update_head(tmp_state)
                semantic_selective_output = None
                if semantic_selective_features is not None:
                    flat_semantic_features = rearrange(
                        semantic_selective_features, "b n c -> (b n) c"
                    )
                    semantic_selective_output = (
                        self.update_semantic_selective_head(
                            torch.cat((tmp_state, flat_semantic_features), dim=-1)
                        )
                    )

            # update gaussian parameters
            delta_gaussians = rearrange(delta_gaussians, "(b n) c -> b n c", b=b)

            if self.cfg.init_gaussian_multiple > 1 and not self.cfg.refine_same_num_points:
                repeat = self.cfg.init_gaussian_multiple
            else:
                repeat = 1

            sh_dim = 3 * self.gaussian_adapter.d_sh

            delta_means, delta_scales, delta_rotations, delta_opacities, delta_shs = delta_gaussians.split(
                (3 * repeat, 3 * repeat, 4 * repeat, 1 * repeat, sh_dim * repeat), dim=-1
            )

            semantic_selective_gate = None
            semantic_selective_corrections = None
            if semantic_selective_output is not None:
                semantic_selective_output = rearrange(
                    semantic_selective_output, "(b n) c -> b n c", b=b
                )
                semantic_selective_gate, semantic_selective_corrections = (
                    semantic_selective_output.split(
                        (repeat, 6 * repeat), dim=-1
                    )
                )

            if self.cfg.init_gaussian_multiple > 1 and not self.cfg.refine_same_num_points:
                delta_means = rearrange(delta_means, "b n (c k) -> b (n k) c", k=repeat)
                delta_scales = rearrange(delta_scales, "b n (c k) -> b (n k) c", k=repeat)
                delta_rotations = rearrange(delta_rotations, "b n (c k) -> b (n k) c", k=repeat)
                delta_opacities = rearrange(delta_opacities, "b n (c k) -> b (n k) c", k=repeat)
                delta_shs = rearrange(delta_shs, "b n (c k) -> b (n k) c", k=repeat)
                if semantic_parameter_routes is not None:
                    semantic_parameter_routes = (
                        semantic_parameter_routes.repeat_interleave(
                            repeat, dim=1
                        )
                    )
                if semantic_selective_corrections is not None:
                    semantic_selective_gate = rearrange(
                        semantic_selective_gate,
                        "b n k -> b (n k) 1",
                        k=repeat,
                    )
                    semantic_selective_corrections = rearrange(
                        semantic_selective_corrections,
                        "b n (c k) -> b (n k) c",
                        c=6,
                        k=repeat,
                    )
                    semantic_selective_active = (
                        semantic_selective_active.repeat_interleave(
                            repeat, dim=1
                        )
                    )

            if semantic_selective_corrections is not None:
                delta_means, delta_scales = apply_selective_geometry_residual(
                    delta_means,
                    delta_scales,
                    semantic_selective_corrections,
                    torch.sigmoid(semantic_selective_gate),
                    semantic_selective_active,
                    gain=self.cfg.semantic_selective_gain,
                )

            if semantic_parameter_routes is not None:
                (
                    delta_means,
                    delta_scales,
                    delta_rotations,
                    delta_opacities,
                    delta_shs,
                ) = apply_semantic_parameter_routes(
                    delta_means,
                    delta_scales,
                    delta_rotations,
                    delta_opacities,
                    delta_shs,
                    semantic_parameter_routes,
                    gain=self.cfg.semantic_parameter_route_gain,
                )

            prev_means = (prev_means + delta_means)

            # clamp the scale
            prev_scales = (prev_scales + delta_scales).clamp(min=self.cfg.gaussian_adapter.clamp_min_scale)

            prev_rotations_unnorm = prev_rotations_unnorm + delta_rotations
            # normalize
            prev_rotations = prev_rotations_unnorm / (prev_rotations_unnorm.norm(dim=-1, keepdim=True) + 1e-8)

            # Create world-space covariance matrices.
            covariances = build_covariance(prev_scales, prev_rotations)
            covariances = rearrange(covariances, "b (v hw) x y -> b v hw x y", v=v)
            c2w_rotations = context["extrinsics"][..., :3, :3].unsqueeze(2)  # [B, V, 1, 3, 3]
            covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)  # [B, V, HW, 3, 3]
            covariances = rearrange(covariances, "b v hw x y -> b (v hw) x y")

            prev_opacities_raw = prev_opacities_raw + delta_opacities

            prev_shs = prev_shs + delta_shs

            # update gaussians
            prev_gaussians = Gaussians(
                prev_means,
                covariances,
                rearrange(prev_shs, "b n (x y) -> b n x y", x=3),
                prev_opacities_raw.squeeze(-1).sigmoid(),
                prev_scales,
                rotations=prev_rotations,
                rotations_unnorm=prev_rotations_unnorm,
                scale_factor=prev_gaussians.scale_factor,
                shift=prev_gaussians.shift,
            )

            delta_means_all.append(delta_means)
            delta_scales_all.append(delta_scales)

            gaussian_output.append(prev_gaussians)

            # render target images
            if target is not None:
                render_img = renderer.forward(
                    prev_gaussians,
                    target["extrinsics"],
                    target["intrinsics"],
                    target["near"],
                    target["far"],
                    (h, w),
                )
                render_output.append(render_img)

            # render input views for next iteration's error computation
            input_render = renderer.forward(
                    prev_gaussians,
                    context["extrinsics"],
                    context["intrinsics"],
                    context["near"],
                    context["far"],
                    (h, w),
                )
            render_input_views.append(input_render)

        return {
            'render': render_output,
            'gaussian': gaussian_output,
            'render_input': render_input_views,
            'delta_means': delta_means_all,
            'delta_scales': delta_scales_all,
            'final_state': tmp_state,  # [BVHW, C], for sliding window global fusion
        }
    
    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=1 if self.cfg.no_crop_image else (self.cfg.shim_patch_size
                * self.cfg.downscale_factor),
            )

            return batch

        return data_shim

    @property
    def sampler(self):
        return None



def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0
