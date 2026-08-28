import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CNNEncoder
from .vit_fpn import ViTFeaturePyramid
from .mv_transformer import (
    MultiViewFeatureTransformer,
    batch_features_camera_parameters,
)
from .matching import warp_with_pose_depth_candidates
from .utils import mv_feature_add_position
from .dpt_head import DPTHead
from .ldm_unet.unet import UNetModel, AttentionBlock
from einops import rearrange
from .dinov2.dinov2 import DINOv2
from ...semantic_depth_initialization import (
    SemanticDepthFeatureAdapter,
    SemanticDepthLogitAdapter,
    SemanticDepthResidualInjection,
)


class MultiViewUniMatch(nn.Module):
    def __init__(
        self,
        num_scales=1,
        feature_channels=128,
        upsample_factor=8,
        lowest_feature_resolution=8,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        num_depth_candidates=128,
        vit_type="vits",
        unet_channels=128,
        unet_channel_mult=[1, 1, 1],
        unet_num_res_blocks=1,
        unet_attn_resolutions=[4],
        grid_sample_disable_cudnn=False,
        only_features=False,
        sample_log_depth=False,
        bilinear_upsample_depth=False,
        no_upsample_depth=False,
        use_amp=False,
        return_raw_mono_features=False,
        max_mono_vit_input_size=560,  # constrain the input resolution to vit
        use_checkpointing=False,
        use_semantic_depth_logit_adapter=False,
        use_semantic_depth_feature_adapter=False,
        use_semantic_depth_residual_injection=False,
        semantic_feature_dim=16,
        semantic_depth_teacher_layer=-1,
        semantic_depth_hidden_channels=64,
        semantic_depth_max_logit_residual=0.25,
        semantic_depth_confidence_floor=0.05,
        semantic_depth_gate_bias=-2.0,
        semantic_depth_feature_hidden_channels=64,
        semantic_depth_max_feature_residual=0.1,
        semantic_depth_feature_confidence_floor=0.05,
        semantic_depth_feature_gate_bias=-2.0,
        semantic_depth_residual_hidden_channels=64,
        semantic_depth_residual_max_relative_gain=0.1,
        **kwargs,
    ):
        super(MultiViewUniMatch, self).__init__()

        # CNN
        self.feature_channels = feature_channels
        self.num_scales = num_scales
        self.lowest_feature_resolution = lowest_feature_resolution
        self.upsample_factor = upsample_factor
        self.only_features = only_features
        self.bilinear_upsample_depth = bilinear_upsample_depth
        self.no_upsample_depth = no_upsample_depth
        self.return_raw_mono_features = return_raw_mono_features
        self.max_mono_vit_input_size = max_mono_vit_input_size
        self.use_semantic_depth_logit_adapter = use_semantic_depth_logit_adapter
        self.use_semantic_depth_feature_adapter = (
            use_semantic_depth_feature_adapter
        )
        self.use_semantic_depth_residual_injection = (
            use_semantic_depth_residual_injection
        )
        if (
            self.use_semantic_depth_feature_adapter
            and self.use_semantic_depth_residual_injection
        ):
            raise ValueError(
                "legacy feature adapter and screenshot Situation A residual "
                "injection must not be enabled together"
            )
        self.semantic_depth_teacher_layer = semantic_depth_teacher_layer
        if (
            self.use_semantic_depth_logit_adapter
            or self.use_semantic_depth_feature_adapter
            or self.use_semantic_depth_residual_injection
        ) and not return_raw_mono_features:
            raise ValueError(
                "semantic depth initialization requires raw DINO features"
            )

        self.use_amp = use_amp

        # sample depth in the log scale instead of the inverse depth
        self.sample_log_depth = sample_log_depth

        # monocular backbones: final
        self.vit_type = vit_type

        # cost volume
        self.num_depth_candidates = num_depth_candidates

        # upsampler
        vit_feature_channel_dict = {"vits": 384, "vitb": 768, "vitl": 1024}

        vit_feature_channel = vit_feature_channel_dict[vit_type]

        # CNN
        self.backbone = CNNEncoder(
            output_dim=feature_channels,
            num_output_scales=num_scales,
            downsample_factor=upsample_factor,
            lowest_scale=lowest_feature_resolution,
            return_all_scales=True,
        )

        # Transformer
        self.transformer = MultiViewFeatureTransformer(
            num_layers=num_transformer_layers,
            d_model=feature_channels,
            nhead=num_head,
            ffn_dim_expansion=ffn_dim_expansion,
            use_checkpointing=use_checkpointing,
        )

        if self.num_scales > 1:
            # generate multi-scale features
            self.mv_pyramid = ViTFeaturePyramid(
                in_channels=128, scale_factors=[2**i for i in range(self.num_scales)]
            )

        # monodepth
        encoder = vit_type
        # local load dinov2
        self.pretrained = DINOv2(encoder,
            use_checkpointing=use_checkpointing,
            )

        del self.pretrained.mask_token  # unused

        if self.num_scales > 1:
            # generate multi-scale features
            self.mono_pyramid = ViTFeaturePyramid(
                in_channels=vit_feature_channel,
                scale_factors=[2**i for i in range(self.num_scales)],
            )

        if self.only_features:
            return

        # UNet regressor
        self.regressor = nn.ModuleList()
        self.regressor_residual = nn.ModuleList()
        self.depth_head = nn.ModuleList()
        self.semantic_depth_adapters = nn.ModuleList()
        self.semantic_depth_feature_adapters = nn.ModuleList()
        self.semantic_depth_residual_injections = nn.ModuleList()

        for i in range(self.num_scales):
            curr_depth_candidates = num_depth_candidates // (4**i)
            cnn_feature_channels = 128 - (32 * i)
            mv_transformer_feature_channels = 128 // (2**i)

            mono_feature_channels = vit_feature_channel // (2**i)

            # concat(cost volume, cnn feature, mv feature, mono feature)
            in_channels = (
                curr_depth_candidates
                + cnn_feature_channels
                + mv_transformer_feature_channels
                + mono_feature_channels
            )

            # unet channels
            channels = unet_channels // (2**i)

            # unet channel mult & unet_attn_resolutions
            if i > 0:
                unet_channel_mult = unet_channel_mult + [1]
                unet_attn_resolutions = [x * 2 for x in unet_attn_resolutions]

            # unet
            modules = [
                nn.Conv2d(in_channels, channels, 3, 1, 1),
                nn.GroupNorm(8, channels),
                nn.GELU(),
            ]

            modules.append(
                UNetModel(
                    image_size=None,
                    in_channels=channels,
                    model_channels=channels,
                    out_channels=channels,
                    num_res_blocks=unet_num_res_blocks,
                    attention_resolutions=unet_attn_resolutions,
                    channel_mult=unet_channel_mult,
                    num_head_channels=32,
                    dims=2,
                    postnorm=False,
                    num_frames=2,
                    use_cross_view_self_attn=True,
                )
            )

            modules.append(nn.Conv2d(channels, channels, 3, 1, 1))

            self.regressor.append(nn.Sequential(*modules))

            # regressor residual
            self.regressor_residual.append(nn.Conv2d(in_channels, channels, 1))

            # depth head
            self.depth_head.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels, channels * 2, 3, 1, 1, padding_mode="replicate"
                    ),
                    nn.GELU(),
                    nn.Conv2d(
                        channels * 2,
                        curr_depth_candidates,
                        3,
                        1,
                        1,
                        padding_mode="replicate",
                    ),
                )
            )
            if self.use_semantic_depth_logit_adapter:
                self.semantic_depth_adapters.append(
                    SemanticDepthLogitAdapter(
                        semantic_channels=semantic_feature_dim,
                        depth_feature_channels=channels,
                        num_depth_candidates=curr_depth_candidates,
                        hidden_channels=semantic_depth_hidden_channels,
                        max_logit_residual=semantic_depth_max_logit_residual,
                        confidence_floor=semantic_depth_confidence_floor,
                        gate_bias=semantic_depth_gate_bias,
                    )
                )
            if self.use_semantic_depth_feature_adapter:
                # Preserve the global RNG stream so enabling a zero-initialized
                # ablation does not change the data sampler's view sequence.
                with torch.random.fork_rng(devices=[]):
                    self.semantic_depth_feature_adapters.append(
                        SemanticDepthFeatureAdapter(
                            semantic_channels=semantic_feature_dim,
                            depth_feature_channels=channels,
                            hidden_channels=(
                                semantic_depth_feature_hidden_channels
                            ),
                            max_relative_residual=(
                                semantic_depth_max_feature_residual
                            ),
                            confidence_floor=(
                                semantic_depth_feature_confidence_floor
                            ),
                            gate_bias=semantic_depth_feature_gate_bias,
                        )
                    )
            if self.use_semantic_depth_residual_injection:
                # Construction must not perturb ReSplat's data-sampling RNG.
                with torch.random.fork_rng(devices=[]):
                    self.semantic_depth_residual_injections.append(
                        SemanticDepthResidualInjection(
                            semantic_channels=semantic_feature_dim,
                            depth_feature_channels=channels,
                            hidden_channels=semantic_depth_residual_hidden_channels,
                            max_relative_residual=(
                                semantic_depth_residual_max_relative_gain
                            ),
                        )
                    )

        # upsampler
        # concat(lowres_depth, cnn feature, mv feature, mono feature)
        in_channels = (
            1
            + cnn_feature_channels
            + mv_transformer_feature_channels
            + mono_feature_channels
        )

        model_configs = {
            "vits": {
                "in_channels": 384,
                "features": 32,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "in_channels": 768,
                "features": 48,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "in_channels": 1024,
                "features": 64,
                "out_channels": [128, 256, 512, 1024],
            },
        }

        if not self.bilinear_upsample_depth and not self.no_upsample_depth:
            self.upsampler = DPTHead(
                **model_configs[vit_type],
                downsample_factor=upsample_factor,
                num_scales=num_scales,
            )

        self.grid_sample_disable_cudnn = grid_sample_disable_cudnn

    def normalize_images(self, images):
        """Normalize image to match the pretrained UniMatch model.
        images: (B, V, C, H, W)
        """
        shape = [*[1] * (images.dim() - 3), 3, 1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(*shape).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(*shape).to(images.device)

        return (images - mean) / std

    def extract_feature(self, images):
        # images: [B, V, C, H, W]
        b, v = images.shape[:2]
        concat = rearrange(images, "b v c h w -> (b v) c h w")
        # list of [BV, C, H, W], resolution from high to low
        features = self.backbone(concat)
        # reverse: resolution from low to high
        features = features[::-1]

        return features

    def forward(
        self,
        images,
        attn_splits_list=None,
        intrinsics=None,
        min_depth=1.0 / 0.5,  # inverse depth range
        max_depth=1.0 / 100,
        num_depth_candidates=128,
        extrinsics=None,
        nn_matrix=None,
        **kwargs,
    ):

        results_dict = {}
        depth_preds = []
        match_probs = []
        semantic_depth_gates = []
        semantic_depth_logit_residuals = []
        semantic_depth_candidate_deltas = []
        semantic_depth_kl_maps = []
        semantic_depth_feature_gates = []
        semantic_depth_feature_residuals = []
        situation_a_gammas = []
        situation_a_relative_residuals = []

        # first normalize images
        images = self.normalize_images(images)
        b, v, _, ori_h, ori_w = images.shape

        # update the num_views in unet attention, useful for random input views
        if not self.only_features:
            set_num_views(self.regressor, num_views=v)

        # NOTE: in this codebase, intrinsics are normalized by image width and height
        # in unimatch's codebase: https://github.com/autonomousvision/unimatch, no normalization
        intrinsics = intrinsics.clone()
        intrinsics[:, :, 0] *= ori_w
        intrinsics[:, :, 1] *= ori_h

        # max_depth, min_depth: [B, V] -> [BV]
        max_depth = max_depth.view(-1)
        min_depth = min_depth.view(-1)

        if self.sample_log_depth:
            # inverse depth to depth
            min_depth, max_depth = 1. / max_depth, 1. / min_depth
            min_depth, max_depth = torch.log(min_depth), torch.log(max_depth)

        # list of features, resolution low to high
        # list of [BV, C, H, W]
        with torch.amp.autocast(device_type='cuda', enabled=self.use_amp, dtype=torch.bfloat16):
            features_list_cnn = self.extract_feature(images)
        features_list_cnn_all_scales = features_list_cnn
        features_list_cnn = features_list_cnn[: self.num_scales]
        results_dict.update({"features_cnn_all_scales": features_list_cnn_all_scales})
        results_dict.update({"features_cnn": features_list_cnn})

        # mv transformer features
        # add position to features
        attn_splits = attn_splits_list[0]

        # [BV, C, H, W]
        features_cnn_pos = mv_feature_add_position(
            features_list_cnn[0], attn_splits, self.feature_channels
        )

        # list of [B, C, H, W]
        features_list = list(
            torch.unbind(
                rearrange(features_cnn_pos, "(b v) c h w -> b v c h w", b=b, v=v), dim=1
            )
        )
        with torch.amp.autocast(device_type='cuda', enabled=self.use_amp, dtype=torch.bfloat16):
            if features_list[0].shape[-1] > 96:
                attn_splits = 4

            if features_list[0].shape[-1] > 192:
                attn_splits = 8

            features_list_mv = self.transformer(
                features_list,
                attn_num_splits=attn_splits,
                nn_matrix=nn_matrix,
            )

        features_mv = rearrange(
            torch.stack(features_list_mv, dim=1), "b v c h w -> (b v) c h w"
        )  # [BV, C, H, W]

        if self.num_scales > 1:
            # multi-scale mv features: resolution from low to high
            # list of [BV, C, H, W]
            with torch.amp.autocast(device_type='cuda', enabled=self.use_amp, dtype=torch.bfloat16):
                features_list_mv = self.mv_pyramid(features_mv)
        else:
            features_list_mv = [features_mv]

        results_dict.update({"features_mv": features_list_mv})

        # mono feature
        ori_h, ori_w = images.shape[-2:]

        assert ori_h <= ori_w
        if ori_w > self.max_mono_vit_input_size:
            resize_w = self.max_mono_vit_input_size // 14 * 14
            resize_h = int((ori_h / ori_w) * self.max_mono_vit_input_size) // 14 * 14
        else:
            resize_h, resize_w = ori_h // 14 * 14, ori_w // 14 * 14

        concat = rearrange(images, "b v c h w -> (b v) c h w")
        concat = F.interpolate(
            concat, (resize_h, resize_w), mode="bilinear", align_corners=True
        )

        # get intermediate features
        intermediate_layer_idx = {
            "vits": [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            "vitl": [4, 11, 17, 23],
        }

        with torch.amp.autocast(device_type='cuda', enabled=self.use_amp, dtype=torch.bfloat16):
            mono_intermediate_features = list(
                self.pretrained.get_intermediate_layers(
                    concat, intermediate_layer_idx[self.vit_type], return_class_token=False
                )
            )

        if self.return_raw_mono_features:
            raw_mono_features = []

        for i in range(len(mono_intermediate_features)):
            curr_features = (
                mono_intermediate_features[i]
                .reshape(concat.shape[0], resize_h // 14, resize_w // 14, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            if self.return_raw_mono_features:
                raw_mono_features.append(curr_features)
            # resize to 1/8 resolution
            curr_features = F.interpolate(
                curr_features,
                (ori_h // 8, ori_w // 8),
                mode="bilinear",
                align_corners=True,
            )
            mono_intermediate_features[i] = curr_features

        results_dict.update({"features_mono_intermediate": mono_intermediate_features})

        if self.return_raw_mono_features:
            results_dict.update({"raw_mono_features": raw_mono_features})

        semantic_feature_source = None
        semantic_feature_projector = kwargs.get("semantic_feature_projector")
        if (
            self.use_semantic_depth_logit_adapter
            or self.use_semantic_depth_feature_adapter
            or self.use_semantic_depth_residual_injection
        ):
            if semantic_feature_projector is None:
                raise ValueError(
                    "semantic depth initialization requires a shared semantic "
                    "feature projector"
                )
            try:
                semantic_feature_source = raw_mono_features[
                    self.semantic_depth_teacher_layer
                ].detach()
            except IndexError as error:
                raise ValueError(
                    "semantic depth teacher layer is outside the DINO feature list"
                ) from error

        # last mono feature
        mono_features = mono_intermediate_features[-1]

        if self.lowest_feature_resolution == 4:
            mono_features = F.interpolate(
                mono_features, scale_factor=2, mode="bilinear", align_corners=True
            )

        if self.num_scales > 1:
            # multi-scale mono features, resolution from low to high
            # list of [BV, C, H, W]
            with torch.amp.autocast(device_type='cuda', enabled=self.use_amp, dtype=torch.bfloat16):
                features_list_mono = self.mono_pyramid(mono_features)
        else:
            features_list_mono = [mono_features]

        results_dict.update({"features_mono": features_list_mono})

        if self.only_features:
            return results_dict

        depth = None

        for scale_idx in range(self.num_scales):
            downsample_factor = self.upsample_factor * (
                2 ** (self.num_scales - 1 - scale_idx)
            )

            # scale intrinsics
            intrinsics_curr = intrinsics.clone()  # [B, V, 3, 3]
            intrinsics_curr[:, :, :2] = intrinsics_curr[:, :, :2] / downsample_factor

            # build cost volume
            features_mv = features_list_mv[scale_idx]  # [BV, C, H, W]

            # list of [B, C, H, W]
            features_mv_curr = list(
                torch.unbind(
                    rearrange(features_mv, "(b v) c h w -> b v c h w", b=b, v=v), dim=1
                )
            )

            intrinsics_curr = list(
                torch.unbind(intrinsics_curr, dim=1)
            )  # list of [B, 3, 3]
            extrinsics_curr = list(torch.unbind(extrinsics, dim=1))  # list of [B, 4, 4]

            # ref: [BV, C, H, W], [BV, 3, 3], [BV, 4, 4]
            # tgt: [BV, V-1, C, H, W], [BV, V-1, 3, 3], [BV, V-1, 4, 4]
            (
                ref_features,
                ref_intrinsics,
                ref_extrinsics,
                tgt_features,
                tgt_intrinsics,
                tgt_extrinsics,
            ) = batch_features_camera_parameters(
                features_mv_curr,
                intrinsics_curr,
                extrinsics_curr,
                nn_matrix=nn_matrix,
            )

            b_new, _, c, h, w = tgt_features.size()

            # relative pose
            # extrinsics: c2w
            pose_curr = torch.matmul(
                tgt_extrinsics.inverse(), ref_extrinsics.unsqueeze(1)
            )  # [BV, V-1, 4, 4]

            if scale_idx > 0:
                # 2x upsample depth
                assert depth is not None
                depth = F.interpolate(
                    depth, scale_factor=2, mode="bilinear", align_corners=True
                ).detach()

            num_depth_candidates = self.num_depth_candidates // (4**scale_idx)

            # generate depth candidates
            if scale_idx == 0:
                # min_depth, max_depth: [BV]
                depth_interval = (max_depth - min_depth) / (
                    self.num_depth_candidates - 1
                )  # [BV]

                linear_space = (
                    torch.linspace(0, 1, num_depth_candidates)
                    .type_as(features_list_cnn[0])
                    .view(1, num_depth_candidates, 1, 1)
                )  # [1, D, 1, 1]

                depth_candidates = min_depth.view(-1, 1, 1, 1) + linear_space * (
                    max_depth - min_depth
                ).view(
                    -1, 1, 1, 1
                )  # [BV, D, 1, 1]
            else:
                # half interval each scale
                depth_interval = (
                    (max_depth - min_depth)
                    / (self.num_depth_candidates - 1)
                    / (2**scale_idx)
                )  # [BV]
                # [BV, 1, 1, 1]
                depth_interval = depth_interval.view(-1, 1, 1, 1)

                # [BV, 1, H, W]
                depth_range_min = (
                    depth - depth_interval * (num_depth_candidates // 2)
                ).clamp(min=min_depth.view(-1, 1, 1, 1))
                depth_range_max = (
                    depth + depth_interval * (num_depth_candidates // 2 - 1)
                ).clamp(max=max_depth.view(-1, 1, 1, 1))

                linear_space = (
                    torch.linspace(0, 1, num_depth_candidates)
                    .type_as(features_list_cnn[0])
                    .view(1, num_depth_candidates, 1, 1)
                )  # [1, D, 1, 1]
                depth_candidates = depth_range_min + linear_space * (
                    depth_range_max - depth_range_min
                )  # [BV, D, H, W]

            if scale_idx == 0:
                # [BV*(V-1), D, H, W]
                depth_candidates_curr = (
                    depth_candidates.unsqueeze(1)
                    .repeat(1, tgt_features.size(1), 1, h, w)
                    .view(-1, num_depth_candidates, h, w)
                )
            else:
                depth_candidates_curr = (
                    depth_candidates.unsqueeze(1)
                    .repeat(1, tgt_features.size(1), 1, 1, 1)
                    .view(-1, num_depth_candidates, h, w)
                )

            intrinsics_input = torch.stack(intrinsics_curr, dim=1).view(
                -1, 3, 3
            )  # [BV, 3, 3]
            intrinsics_input = intrinsics_input.unsqueeze(1).repeat(
                1, tgt_features.size(1), 1, 1
            )  # [BV, V-1, 3, 3]

            ref_features = ref_features.float()
            tgt_features = tgt_features.float()
            depth_candidates_curr = depth_candidates_curr.float()

            warped_tgt_features = warp_with_pose_depth_candidates(
                rearrange(tgt_features, "b v ... -> (b v) ..."),
                rearrange(intrinsics_input, "b v ... -> (b v) ..."),
                rearrange(pose_curr, "b v ... -> (b v) ..."),
                torch.exp(depth_candidates_curr) if self.sample_log_depth else 1.0 / depth_candidates_curr,  # convert inverse/log depth to depth
                grid_sample_disable_cudnn=self.grid_sample_disable_cudnn,
            )  # [BV*(V-1), C, D, H, W]

            # ref: [BV, C, H, W]
            # warped: [BV*(V-1), C, D, H, W] -> [BV, V-1, C, D, H, W]
            warped_tgt_features = rearrange(
                warped_tgt_features,
                "(b v) ... -> b v ...",
                b=b_new,
                v=tgt_features.size(1),
            )
            # [BV, V-1, D, H, W] -> [BV, D, H, W]
            # average cross other views
            cost_volume = (
                (ref_features.unsqueeze(-3).unsqueeze(1) * warped_tgt_features).sum(2)
                / (c**0.5)
            ).mean(1)

            # regressor
            features_cnn = features_list_cnn[scale_idx]  # [BV, C, H, W]

            features_mono = features_list_mono[scale_idx]  # [BV, C, H, W]

            concat = torch.cat(
                (cost_volume, features_cnn, features_mv, features_mono), dim=1
            )

            with torch.amp.autocast(device_type='cuda', enabled=self.use_amp, dtype=torch.bfloat16):
                out = self.regressor[scale_idx](concat) + self.regressor_residual[
                    scale_idx
                ](concat)

            out = out.float()

            # Depth prediction. Screenshot Situation A injects semantics into
            # the frozen ReSplat feature immediately before the pretrained
            # depth head: F_depth' = F_c + gamma * A(F_s).  The same projected
            # field later becomes every Gaussian's semantic attribute z.
            base_logits = self.depth_head[scale_idx](out)
            base_match_prob = F.softmax(base_logits, dim=1)
            depth_features = out
            if self.use_semantic_depth_residual_injection:
                projected_semantics = semantic_feature_projector(
                    semantic_feature_source,
                    output_size=out.shape[-2:],
                )
                (
                    depth_features,
                    situation_a_gamma,
                    situation_a_residual,
                ) = self.semantic_depth_residual_injections[scale_idx](
                    out,
                    projected_semantics,
                )
                situation_a_gammas.append(situation_a_gamma)
                situation_a_relative_residuals.append(
                    situation_a_residual.abs().mean(dim=1, keepdim=True)
                    / out.detach().float().abs().mean(dim=1, keepdim=True)
                    .clamp_min(1e-6)
                )
            if self.use_semantic_depth_feature_adapter:
                projected_semantics = semantic_feature_projector(
                    semantic_feature_source,
                    output_size=out.shape[-2:],
                )
                (
                    depth_features,
                    semantic_feature_gate,
                    semantic_feature_residual,
                ) = self.semantic_depth_feature_adapters[scale_idx](
                    out,
                    projected_semantics,
                    base_logits,
                )
                semantic_depth_feature_gates.append(semantic_feature_gate)
                semantic_depth_feature_residuals.append(
                    semantic_feature_residual.abs().mean(dim=1, keepdim=True)
                    / out.detach().float().abs().mean(dim=1, keepdim=True)
                    .clamp_min(1e-6)
                )
            feature_logits = (
                self.depth_head[scale_idx](depth_features)
                if (
                    self.use_semantic_depth_feature_adapter
                    or self.use_semantic_depth_residual_injection
                )
                else base_logits
            )
            if self.use_semantic_depth_logit_adapter:
                projected_semantics = semantic_feature_projector(
                    semantic_feature_source,
                    output_size=out.shape[-2:],
                )
                (
                    depth_logits,
                    semantic_gate,
                    semantic_logit_residual,
                    _,
                ) = self.semantic_depth_adapters[scale_idx](
                    feature_logits,
                    depth_features,
                    projected_semantics,
                )
                match_prob = F.softmax(depth_logits, dim=1)
                semantic_depth_gates.append(semantic_gate)
                semantic_depth_logit_residuals.append(
                    semantic_logit_residual.abs().mean(dim=1, keepdim=True)
                )
                semantic_depth_kl_maps.append(
                    (
                        match_prob
                        * (
                            match_prob.clamp_min(1e-8).log()
                            - base_match_prob.detach().clamp_min(1e-8).log()
                        )
                    ).sum(dim=1, keepdim=True).clamp_min(0)
                )
            else:
                match_prob = F.softmax(feature_logits, dim=1)
            match_probs.append(match_prob)

            base_candidate_depth = (
                base_match_prob.detach() * depth_candidates
            ).sum(dim=1, keepdim=True)

            if scale_idx == 0:
                # [BV, D, H, W]
                depth_candidates = depth_candidates.repeat(1, 1, h, w)
            depth = (match_prob * depth_candidates).sum(
                dim=1, keepdim=True
            )  # [BV, 1, H, W]
            if (
                self.use_semantic_depth_logit_adapter
                or self.use_semantic_depth_feature_adapter
                or self.use_semantic_depth_residual_injection
            ):
                semantic_depth_candidate_deltas.append(
                    (depth - base_candidate_depth).abs()
                    / base_candidate_depth.abs().clamp_min(1e-6)
                )

            # upsample to the original resolution for supervison at training time only
            if self.training and scale_idx < self.num_scales - 1:
                depth_bilinear = F.interpolate(
                    depth,
                    scale_factor=downsample_factor,
                    mode="bilinear",
                    align_corners=True,
                )
                depth_preds.append(depth_bilinear)

            # final output, learned upsampler
            if scale_idx == self.num_scales - 1:
                if self.bilinear_upsample_depth or self.no_upsample_depth:
                    residual_depth = 0
                else:
                    with torch.amp.autocast(device_type='cuda', enabled=self.use_amp, dtype=torch.bfloat16):
                        residual_depth = self.upsampler(
                            mono_intermediate_features,
                            # resolution high to low
                            cnn_features=features_list_cnn_all_scales[::-1],
                            mv_features=(
                                features_mv if self.num_scales == 1 else features_list_mv[::-1]
                            ),
                            depth=depth,
                        )

                if self.no_upsample_depth:
                    depth_preds.append(depth)
                else:
                    depth_bilinear = F.interpolate(
                        depth,
                        scale_factor=self.upsample_factor,
                        mode="bilinear",
                        align_corners=True,
                    )
                    depth = (depth_bilinear + residual_depth).clamp(
                        min=min_depth.view(-1, 1, 1, 1), max=max_depth.view(-1, 1, 1, 1)
                    )

                    depth_preds.append(depth)

        for i in range(len(depth_preds)):
            if self.sample_log_depth:
                # log depth to depth
                depth_pred = torch.exp(depth_preds[i].squeeze(1))
            else:
                # convert inverse depth to depth
                depth_pred = 1.0 / depth_preds[i].squeeze(1)  # [BV, H, W]
            depth_preds[i] = rearrange(
                depth_pred, "(b v) ... -> b v ...", b=b, v=v
            )  # [B, V, H, W]

        results_dict.update({"depth_preds": depth_preds})
        results_dict.update({"match_probs": match_probs})
        if self.use_semantic_depth_logit_adapter:
            results_dict.update(
                {
                    "semantic_depth_gates": semantic_depth_gates,
                    "semantic_depth_logit_residuals": (
                        semantic_depth_logit_residuals
                    ),
                    "semantic_depth_candidate_deltas": (
                        semantic_depth_candidate_deltas
                    ),
                    "semantic_depth_kl_maps": semantic_depth_kl_maps,
                }
            )
        if self.use_semantic_depth_feature_adapter:
            results_dict.update(
                {
                    "semantic_depth_feature_gates": semantic_depth_feature_gates,
                    "semantic_depth_feature_residuals": (
                        semantic_depth_feature_residuals
                    ),
                }
            )
        if self.use_semantic_depth_residual_injection:
            results_dict.update(
                {
                    "situation_a_gammas": situation_a_gammas,
                    "situation_a_relative_residuals": (
                        situation_a_relative_residuals
                    ),
                    "semantic_depth_candidate_deltas": (
                        semantic_depth_candidate_deltas
                    ),
                }
            )

        return results_dict


def set_num_views(module, num_views):
    if isinstance(module, AttentionBlock):
        module.attention.n_frames = num_views
    elif (
        isinstance(module, nn.ModuleList)
        or isinstance(module, nn.Sequential)
        or isinstance(module, nn.Module)
    ):
        for submodule in module.children():
            set_num_views(submodule, num_views)
