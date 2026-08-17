from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

try:
    import moviepy.editor as mpy
except:
    import moviepy as mpy
import torch
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn, optim
import numpy as np
import json
import os
import time
from tqdm import tqdm
import torch.nn.functional as F
import math
from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..dataset import DatasetCfg
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DecoderOutput
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from src.visualization.vis_depth import viz_depth_tensor
from PIL import Image
from ..misc.stablize_camera import render_stabilization_path
from .ply_export import save_gaussian_ply
from ..visualization.export_point_cloud import export_to_point_cloud, transform_points
from ..evaluation.depth_metrics import compute_depth_errors
from ..loss.loss_depth_smooth import get_smooth_loss

from .types import Gaussians
from .gaussian_utils import merge_gaussians
from .semantic_boundary import balanced_boundary_l1, rgb_to_soft_boundary

try:
    from bitsandbytes.optim import AdamW8bit
except:
    pass

slurm_id_logged = False


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    lr_monodepth: float
    lr_depth: float
    weight_decay: float
    warm_up_ratio: float
    adamw_8bit: bool


@dataclass
class TestCfg:
    output_path: Path
    compute_scores: bool
    save_image: bool
    save_video: bool
    eval_time_skip_steps: int
    save_gt_image: bool
    save_input_images: bool
    save_depth: bool
    save_depth_concat_img: bool
    save_depth_npy: bool
    save_gaussian: bool
    save_gaussian_npz: bool
    no_align_to_view: bool
    save_point_cloud: bool
    render_chunk_size: int | None
    stablize_camera: bool
    stab_camera_kernel: int
    render_input_views: bool
    inference_window_size: int | None
    profile_model: bool
    test_zero_order_sh_only: bool


@dataclass
class TrainCfg:
    extended_visualization: bool
    print_log_every_n_steps: int
    eval_model_every_n_val: int
    eval_data_length: int
    eval_deterministic: bool
    eval_time_skip_steps: int
    eval_save_model: bool
    intermediate_loss_weight: float
    no_viz_video: bool
    eval_depth: bool
    train_ignore_large_loss: float
    no_log_projections: bool

    log_depth_loss: bool
    depth_smooth_loss_weight: float
    depth_smooth_loss_nonorm: bool

    no_log_video: bool

    # when doing refinement, supervise input view or not since we also render input views
    loss_on_input_views: bool

    # half res lpips loss to save memory
    half_res_lpips_loss: bool

    # local window training
    train_window_size: int | None


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None
    eval_data_cfg: Optional[DatasetCfg | None]

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
        eval_data_cfg: Optional[DatasetCfg | None] = None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker
        self.eval_data_cfg = eval_data_cfg

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        # This is used for testing.
        self.benchmarker = Benchmarker()
        self.eval_cnt = 0

        if self.test_cfg.compute_scores:
            self.test_step_outputs = {}
            self.time_skip_steps_dict = {"encoder": 0, "decoder": 0}

    def training_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["context"]["image"].shape
        # Run the model.
        if self.train_cfg.train_window_size is not None:
            assert self.train_cfg.train_window_size > 0
            window = self.train_cfg.train_window_size
            total_views = batch["context"]["image"].size(1)

            window_indices = sliding_window_indices(total_views, window, 0)

            all_gaussians = []
            all_states = []  # for global refinement

            # sliding window inference
            for indices in window_indices:
                start, end = indices
                curr_window_input = {
                    "extrinsics": batch["context"]["extrinsics"][:, start:end],
                    "intrinsics": batch["context"]["intrinsics"][:, start:end],
                    "image": batch["context"]["image"][:, start:end],
                    "near": batch["context"]["near"][:, start:end],
                    "far": batch["context"]["far"][:, start:end],
                    "index": batch["context"]["index"][:, start:end],
                }
                if "boundary" in batch["context"]:
                    curr_window_input["boundary"] = (
                        batch["context"]["boundary"][:, start:end]
                    )
                    curr_window_input["boundary_confidence"] = (
                        batch["context"]["boundary_confidence"][:, start:end]
                    )

                if self.encoder.cfg.num_refine > 0:
                    with torch.no_grad():
                        curr_gaussians = self.encoder(
                            curr_window_input,
                            self.global_step,
                            deterministic=False,
                        )
                else:
                    curr_gaussians = self.encoder(
                        curr_window_input,
                        self.global_step,
                        deterministic=False,
                    )

                if isinstance(curr_gaussians, dict):
                    pred_depths = curr_gaussians["depths"]
                    if "depth" in batch["context"]:
                        depth_gt = batch["context"]["depth"]
                    if "condition_features" in curr_gaussians:
                        condition_features = curr_gaussians["condition_features"]  # [BV, C, H, W]
                    else:
                        condition_features = None
                    curr_gaussians = curr_gaussians["gaussians"]

                all_gaussians.append(curr_gaussians)
                all_states.append(condition_features)

            # merge all gaussians
            # ['means', 'covariances', 'harmonics', 'opacities']
            gaussians = merge_gaussians(all_gaussians)

            # global refine after simply combining local window gaussians
            if self.encoder.cfg.num_refine > 0:
                # combine all condition features
                out = []
                is_ori_feature = True
                for i in range(len(all_states)):
                    curr = all_states[i]
                    if curr.dim() == 4:  
                        # [BV, C, H, W]
                        curr = rearrange(curr, "(b v) c h w -> b v c h w", b=b)
                        is_ori_feature = True
                    elif curr.dim() == 2:
                        # [BVHW, C]
                        curr = rearrange(curr, "(b v h w) c -> b v h w c", b=b, 
                        h=h // self.encoder.cfg.latent_downsample,
                        w=w // self.encoder.cfg.latent_downsample,
                        )
                        is_ori_feature = False
                    else:
                        raise NotImplementedError

                    out.append(curr)

                # concat
                if is_ori_feature:
                    concat = torch.cat(out, dim=1)  # [B, V*K, C, H, W]
                    concat = rearrange(concat, "b v c h w -> (b v) c h w")
                else:
                    concat = torch.cat(out, dim=1)  # [B, V*K, H, W, C]
                    concat = rearrange(concat, "b v h w c -> (b v) c h w")

                condition_features = concat

        else:
            if (
                self.encoder.cfg.num_refine > 0
                and not self.encoder.cfg.use_semantic_gaussian_init
            ):
                with torch.no_grad():
                    gaussians = self.encoder(
                        batch["context"], self.global_step, False, scene_names=batch["scene"]
                    )
            else:
                gaussians = self.encoder(
                    batch["context"], self.global_step, False, scene_names=batch["scene"]
                )

            if isinstance(gaussians, dict):
                pred_depths = gaussians["depths"]
                if self.encoder.cfg.num_refine > 0:
                    condition_features = gaussians["condition_features"]
                gaussians = gaussians["gaussians"]

        supervise_intermediate_depth = False
        semantic_init_output = None

        if self.encoder.cfg.num_refine > 0:
            if (
                self.encoder.cfg.use_semantic_gaussian_init
                and self.encoder.cfg.semantic_init_auxiliary_loss_weight > 0
            ):
                semantic_init_output = self.decoder.forward(
                    gaussians,
                    batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                    depth_mode=None,
                )
            refine_output = self.encoder.forward_update(
                batch["context"],
                batch["target"],
                condition_features,
                gaussians,
                self.decoder,
                batch["context_remain"] if "context_remain" in batch.keys() else None,
            )

            render_output = refine_output['render']
            gaussian_output = refine_output['gaussian']

            render_input_views = refine_output['render_input']

            delta_means = refine_output['delta_means']
            delta_scales = refine_output['delta_scales']

            # the last output
            output = render_output[-1]
            gaussians = gaussian_output[-1]

        else:
            if gaussians.means.size(0) != batch["target"]["extrinsics"].size(0):
                supervise_intermediate_depth = True
                assert gaussians.means.size(0) % batch["target"]["extrinsics"].size(0) == 0
                num_depths = gaussians.means.size(0) // batch["target"]["extrinsics"].size(
                    0
                )
                # add loss to intermediate depth predictions
                target_extrinsics = torch.cat(
                    [batch["target"]["extrinsics"]] * num_depths, dim=0
                )
                target_intrinsics = torch.cat(
                    [batch["target"]["intrinsics"]] * num_depths, dim=0
                )
                target_near = torch.cat([batch["target"]["near"]] * num_depths, dim=0)
                target_far = torch.cat([batch["target"]["far"]] * num_depths, dim=0)

                output_all = self.decoder.forward(
                    gaussians,
                    target_extrinsics,
                    target_intrinsics,
                    target_near,
                    target_far,
                    (h, w),
                    depth_mode=None,
                )
                # split
                batch_size = batch["target"]["extrinsics"].size(0)
                # order: intermediate depth, final depth
                output_intermediate = DecoderOutput(
                    color=output_all.color[:-batch_size],
                    depth=(
                        output_all.depth[:-batch_size]
                        if output_all.depth is not None
                        else None
                    ),
                )
                output = DecoderOutput(
                    color=output_all.color[-batch_size:],
                    depth=(
                        output_all.depth[-batch_size:]
                        if output_all.depth is not None
                        else None
                    ),
                )
            else:
                output = self.decoder.forward(
                    gaussians,
                    batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                    depth_mode=None,
                )

        target_gt = batch["target"]["image"]

        # Compute and log loss.
        total_loss = 0

        valid_depth_mask = None

        # loss with refinement
        if self.encoder.cfg.num_refine > 0:
            num_output = len(render_output)
            for i in range(num_output):
                # Compute training metrics.
                psnr_probabilistic = compute_psnr(
                    rearrange(target_gt, "b v c h w -> (b v) c h w"),
                    rearrange(render_output[i].color, "b v c h w -> (b v) c h w"),
                )
                self.log(f"train/psnr_{i}", psnr_probabilistic.mean())

                # compute losses
                for loss_fn in self.losses:
                    if loss_fn.name == "mse":
                        loss = loss_fn.forward(
                            render_output[i],
                            batch,
                            None,
                            self.global_step,

                            clamp_large_error=self.train_cfg.train_ignore_large_loss,
                            valid_depth_mask=valid_depth_mask,
                        )
                    else:
                        loss = loss_fn.forward(
                            render_output[i],
                            batch,
                            None,
                            self.global_step,
                            valid_depth_mask=valid_depth_mask,
                            half_res_lpips=self.train_cfg.half_res_lpips_loss,
                        )
                    self.log(f"loss/{loss_fn.name}_{i}", loss)

                    curr_loss_weight = self.train_cfg.intermediate_loss_weight ** (
                                num_output - 1 - i
                            )

                    total_loss = total_loss + curr_loss_weight * loss

            # V2 semantic initialization needs a direct learning signal. Without
            # this term its gradient is filtered through the recurrent updater,
            # which the 20-step diagnostic showed can almost entirely compensate
            # for the new initial geometry. The target view is used only here as
            # supervision; it is never fed to the initializer.
            if semantic_init_output is not None:
                init_weight = self.encoder.cfg.semantic_init_auxiliary_loss_weight
                init_psnr = compute_psnr(
                    rearrange(target_gt, "b v c h w -> (b v) c h w"),
                    rearrange(
                        semantic_init_output.color, "b v c h w -> (b v) c h w"
                    ),
                )
                self.log("train/semantic_init_psnr", init_psnr.mean())
                for loss_fn in self.losses:
                    if loss_fn.name == "mse":
                        init_loss = loss_fn.forward(
                            semantic_init_output,
                            batch,
                            None,
                            self.global_step,
                            clamp_large_error=self.train_cfg.train_ignore_large_loss,
                            valid_depth_mask=valid_depth_mask,
                        )
                    else:
                        init_loss = loss_fn.forward(
                            semantic_init_output,
                            batch,
                            None,
                            self.global_step,
                            valid_depth_mask=valid_depth_mask,
                            half_res_lpips=self.train_cfg.half_res_lpips_loss,
                        )
                    self.log(f"loss/semantic_init_{loss_fn.name}", init_loss)
                    total_loss = total_loss + init_weight * init_loss

            # loss on input views
            if self.train_cfg.loss_on_input_views:
                for i in range(num_output):
                    # Compute training metrics.
                    psnr_probabilistic = compute_psnr(
                        rearrange(batch["context"]["image"], "b v c h w -> (b v) c h w"),
                        rearrange(render_input_views[i].color, "b v c h w -> (b v) c h w"),
                    )
                    self.log(f"train/input_psnr_{i}", psnr_probabilistic.mean())

                    # compute losses
                    for loss_fn in self.losses:
                        if loss_fn.name == "mse":
                            loss = loss_fn.forward(
                                render_input_views[i],
                                batch,
                                gaussian_output[i],
                                self.global_step,
    
                                clamp_large_error=self.train_cfg.train_ignore_large_loss,
                                valid_depth_mask=valid_depth_mask,
                                loss_on_input_views=True,
                            )
                        else:
                            loss = loss_fn.forward(
                                render_input_views[i],
                                batch,
                                gaussian_output[i],
                                self.global_step,
                                valid_depth_mask=valid_depth_mask,
                                loss_on_input_views=True,
                                half_res_lpips=self.train_cfg.half_res_lpips_loss,
                            )
                        self.log(f"loss/input_{loss_fn.name}_{i}", loss)

                        curr_loss_weight = self.train_cfg.intermediate_loss_weight ** (
                                    num_output - 1 - i
                                )

                        total_loss = total_loss + curr_loss_weight * loss

        else:
            # Compute metrics.
            psnr_probabilistic = compute_psnr(
                rearrange(target_gt, "b v c h w -> (b v) c h w"),
                rearrange(output.color, "b v c h w -> (b v) c h w"),
            )
            self.log("train/psnr", psnr_probabilistic.mean())

            for loss_fn in self.losses:
                if loss_fn.name == "mse":
                    loss = loss_fn.forward(
                        output,
                        batch,
                        gaussians,
                        self.global_step,
                        clamp_large_error=self.train_cfg.train_ignore_large_loss,
                        valid_depth_mask=valid_depth_mask,
                    )
                else:
                    loss = loss_fn.forward(
                        output,
                        batch,
                        gaussians,
                        self.global_step,
                        valid_depth_mask=valid_depth_mask,
                        half_res_lpips=self.train_cfg.half_res_lpips_loss,
                    )
                self.log(f"loss/{loss_fn.name}", loss)
                total_loss = total_loss + loss

        # color loss on intermediate output
        if supervise_intermediate_depth:
            for loss_fn in self.losses:
                if output_intermediate.color.size(0) != batch_size:
                    assert output_intermediate.color.size(0) % batch_size == 0
                    num_intermediate = output_intermediate.color.size(0) // batch_size
                    intermediate_loss = 0
                    for i in range(num_intermediate):
                        curr_output = DecoderOutput(
                            color=output_intermediate.color[
                                (batch_size * i) : (batch_size * (i + 1))
                            ],
                            depth=(
                                output_intermediate.depth[
                                    (batch_size * i) : (batch_size * (i + 1))
                                ]
                                if output_intermediate.depth is not None
                                else None
                            ),
                        )
                        curr_loss_weight = self.train_cfg.intermediate_loss_weight ** (
                            num_intermediate - i
                        )

                        if loss_fn.name == "mse":
                            loss = loss_fn.forward(
                                curr_output,
                                batch,
                                gaussians,
                                self.global_step,
    
                                clamp_large_error=self.train_cfg.train_ignore_large_loss,
                                valid_depth_mask=valid_depth_mask,
                            )
                        else:
                            loss = loss_fn.forward(
                                curr_output,
                                batch,
                                gaussians,
                                self.global_step,
                                valid_depth_mask=valid_depth_mask,
                            )

                        intermediate_loss = intermediate_loss + curr_loss_weight * loss

                    self.log(f"loss/{loss_fn.name}_intermediate", intermediate_loss)
                    total_loss = total_loss + intermediate_loss
                else:
                    if loss_fn.name == "mse":
                        loss = loss_fn.forward(
                            output_intermediate,
                            batch,
                            gaussians,
                            self.global_step,

                            clamp_large_error=self.train_cfg.train_ignore_large_loss,
                            valid_depth_mask=valid_depth_mask,
                        )
                    else:
                        loss = loss_fn.forward(
                            output_intermediate,
                            batch,
                            gaussians,
                            self.global_step,
                            valid_depth_mask=valid_depth_mask,
                        )
                    self.log(f"loss/{loss_fn.name}_intermediate", loss)
                    total_loss = (
                        total_loss + self.train_cfg.intermediate_loss_weight * loss
                    )

        # depth smooth loss
        if self.train_cfg.depth_smooth_loss_weight > 0:
            imgs = batch["context"]["image"].flatten(0, 1)  # [BV, 3, H, W]
            
            if supervise_intermediate_depth:
                assert pred_depths.size(0) % batch_size == 0
                num_depths = pred_depths.size(0) // batch_size
                depth_smooth_loss = 0
                for i in range(num_depths):
                    curr_loss_weight = self.train_cfg.intermediate_loss_weight ** (
                        num_depths - i - 1)

                    depth = pred_depths[(batch_size * i):(
                            batch_size * (i + 1))].flatten(0, 1).unsqueeze(1)  # [BV, 1, H, W]
                    disp = 1. / depth
                    if self.train_cfg.depth_smooth_loss_nonorm:
                        norm_disp = disp
                    else:
                        mean_disp = disp.mean(2, True).mean(3, True)
                        norm_disp = disp / (mean_disp + 1e-7)

                    # resize to depth's resolution
                    if imgs.shape[-2:] != norm_disp.shape[-2:]:
                        imgs = F.interpolate(imgs, size=norm_disp.shape[-2:], mode='bilinear', align_corners=True)

                    depth_smooth_loss = get_smooth_loss(norm_disp, imgs)

                    depth_smooth_loss = depth_smooth_loss + curr_loss_weight * depth_smooth_loss

            else:
                depth = pred_depths.flatten(0, 1).unsqueeze(1)

                disp = 1. / depth
                if self.train_cfg.depth_smooth_loss_nonorm:
                    norm_disp = disp
                else:
                    mean_disp = disp.mean(2, True).mean(3, True)
                    norm_disp = disp / (mean_disp + 1e-7)

                # resize to depth's resolution
                if imgs.shape[-2:] != norm_disp.shape[-2:]:
                    imgs = F.interpolate(imgs, size=norm_disp.shape[-2:], mode='bilinear', align_corners=True)

                depth_smooth_loss = get_smooth_loss(norm_disp, imgs)

            depth_smooth_loss = self.train_cfg.depth_smooth_loss_weight * depth_smooth_loss

            self.log(f"loss/depth_smooth", depth_smooth_loss)
            total_loss = total_loss + depth_smooth_loss

        if self.encoder.cfg.use_semantic_state_refinement:
            semantic_render, semantic_alpha = self.decoder.forward_features(
                gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
            )
            with torch.no_grad():
                semantic_teacher = self.encoder.extract_semantic_teacher_features(
                    batch["target"]["image"], (h, w)
                )
            semantic_cosine = F.cosine_similarity(
                semantic_render, semantic_teacher, dim=2
            )
            semantic_valid = semantic_alpha >= 0.1
            semantic_feature_loss = (
                1.0 - semantic_cosine.masked_select(semantic_valid).mean()
            )
            semantic_render_std = semantic_render.permute(
                0, 1, 3, 4, 2
            ).reshape(-1, semantic_render.shape[2]).std(dim=0).mean()
            self.log("loss/semantic_feature", semantic_feature_loss)
            self.log("semantic/target_cosine", 1.0 - semantic_feature_loss)
            self.log("semantic/render_feature_std", semantic_render_std)
            total_loss = total_loss + (
                self.encoder.cfg.semantic_feature_loss_weight
                * semantic_feature_loss
            )

        self.log("loss/total", total_loss)

        if hasattr(self.encoder, "semantic_init_diagnostics"):
            for name, value in self.encoder.semantic_init_diagnostics.items():
                self.log(f"semantic_init/{name}", value)

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            print(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"bound = [{batch['context']['near'].detach().cpu().numpy().mean()} "
                f"{batch['context']['far'].detach().cpu().numpy().mean()}]; "
                f"loss = {total_loss:.6f}"
            )
        self.log("info/near", batch["context"]["near"].detach().cpu().numpy().mean())
        self.log("info/far", batch["context"]["far"].detach().cpu().numpy().mean())
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # log gaussians scales
        if self.encoder.cfg.num_refine > 0:
            num_output = len(delta_means)
            # delta means
            if isinstance(delta_means[0], torch.Tensor):
                for i in range(num_output):
                    self.log(f"update{i}/delta_means_min", delta_means[i].abs().min().item())
                    self.log(f"update{i}/delta_means_mean", delta_means[i].abs().mean().item())
                    self.log(f"update{i}/delta_means_max", delta_means[i].abs().max().item())

            # delta scales
            if isinstance(delta_scales[0], torch.Tensor):
                for i in range(num_output):
                    self.log(f"update{i}/delta_scales_min", delta_scales[i].abs().min().item())
                    self.log(f"update{i}/delta_scales_mean", delta_scales[i].abs().mean().item())
                    self.log(f"update{i}/delta_scales_max", delta_scales[i].abs().max().item())

        self.log("info/gaussian_scale_min", gaussians.scales.min().item())
        self.log("info/gaussian_scale_max", gaussians.scales.max().item())
        self.log("info/gaussian_scale_mean", gaussians.scales.mean().item())

        # log gaussians opacities
        self.log("info/gaussian_opacity_min", gaussians.opacities.min().item())
        self.log("info/gaussian_opacity_max", gaussians.opacities.max().item())
        self.log("info/gaussian_opacity_mean", gaussians.opacities.mean().item())

        # log gaussians opacities raw
        self.log("info/gaussian_opacity_raw_min", torch.logit(gaussians.opacities, eps=1e-6).min().item())
        self.log("info/gaussian_opacity_raw_max", torch.logit(gaussians.opacities, eps=1e-6).max().item())
        self.log("info/gaussian_opacity_raw_mean", torch.logit(gaussians.opacities, eps=1e-6).abs().mean().item())

        # log gaussians mean
        self.log("info/gaussian_mean_min", gaussians.means.min().item())
        self.log("info/gaussian_mean_max", gaussians.means.max().item())
        self.log("info/gaussian_mean_mean", gaussians.means.abs().mean().item())

        # log gaussians rotation unnorm
        self.log("info/gaussian_rotation_unnorm_min", gaussians.rotations_unnorm.min().item())
        self.log("info/gaussian_rotation_unnorm_max", gaussians.rotations_unnorm.max().item())
        self.log("info/gaussian_rotation_unnorm_mean", gaussians.rotations_unnorm.abs().mean().item())

        # log gaussians sh
        self.log("info/gaussian_sh_min", gaussians.harmonics.min().item())
        self.log("info/gaussian_sh_max", gaussians.harmonics.max().item())
        self.log("info/gaussian_sh_mean", gaussians.harmonics.abs().mean().item())

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        if self.global_step == 5 and self.global_rank == 0:
            os.system("nvidia-smi")

        global slurm_id_logged
        if self.global_rank == 0 and not slurm_id_logged:
            print('slurm id:', os.environ.get('SLURM_JOB_ID'))
            slurm_id_logged = True

        return total_loss

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        if self.test_cfg.render_input_views:
            # to see how good the model performs on the input views
            b, v, _, h, w = batch["context"]["image"].shape
        else:
            b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1

        pred_depths = None
        depth_gt = None
        initial_gaussians_for_eval = None
        initial_output = None

        # save input views for visualization
        if self.test_cfg.save_input_images:
            (scene,) = batch["scene"]
            self.test_cfg.output_path = os.path.join(get_cfg()["output_dir"], "metrics")
            path = Path(get_cfg()["output_dir"])

            input_images = batch["context"]["image"][0]  # [V, 3, H, W]
            index = batch["context"]["index"][0]
            for idx, color in zip(index, input_images):
                save_image(color, path / "images" / scene / f"color/input_{idx:0>6}.png")

        # save depth vis
        if self.test_cfg.save_depth or self.test_cfg.save_gaussian:
            visualization_dump = {}
        else:
            visualization_dump = None

        # Render Gaussians.
        with self.benchmarker.time("encoder"):
            if self.test_cfg.inference_window_size is not None:
                assert self.test_cfg.inference_window_size > 0
                window = self.test_cfg.inference_window_size
                total_views = batch["context"]["image"].size(1)

                window_indices = sliding_window_indices(total_views, window, 0)

                all_gaussians = []
                all_states = []  # for global refinement

                # sliding window inference
                for indices in window_indices:
                    start, end = indices
                    curr_window_input = {
                        "extrinsics": batch["context"]["extrinsics"][:, start:end],
                        "intrinsics": batch["context"]["intrinsics"][:, start:end],
                        "image": batch["context"]["image"][:, start:end],
                        "near": batch["context"]["near"][:, start:end],
                        "far": batch["context"]["far"][:, start:end],
                        "index": batch["context"]["index"][:, start:end],
                    }

                    curr_gaussians = self.encoder(
                        curr_window_input,
                        self.global_step,
                        deterministic=False,
                    )

                    if isinstance(curr_gaussians, dict):
                        pred_depths = curr_gaussians["depths"]
                        if "depth" in batch["context"]:
                            depth_gt = batch["context"]["depth"]
                        if "condition_features" in curr_gaussians:
                            condition_features = curr_gaussians["condition_features"]  # [BV, C, H, W]
                        else:
                            condition_features = None
                        curr_gaussians = curr_gaussians["gaussians"]

                    all_gaussians.append(curr_gaussians)
                    all_states.append(condition_features)

                # merge all gaussians
                # ['means', 'covariances', 'harmonics', 'opacities']
                gaussians = merge_gaussians(all_gaussians)
                if self.test_cfg.compute_scores and self.encoder.cfg.num_refine > 0:
                    initial_gaussians_for_eval = gaussians

                # global refine after simply combining local window gaussians
                if self.encoder.cfg.num_refine > 0:
                    # combine all condition features
                    out = []
                    is_ori_feature = False
                    for i in range(len(all_states)):
                        curr = all_states[i]
                        if curr.dim() == 4:  
                            # [BV, C, H, W]
                            curr = rearrange(curr, "(b v) c h w -> b v c h w", b=b)
                            is_ori_feature = True
                        elif curr.dim() == 2:
                            # [BVHW, C]
                            curr = rearrange(curr, "(b v h w) c -> b v h w c", b=b, 
                            h=h // self.encoder.cfg.latent_downsample,
                            w=w // self.encoder.cfg.latent_downsample,
                            )
                            is_ori_feature = False
                        else:
                            raise NotImplementedError

                        out.append(curr)

                    # concat
                    if is_ori_feature:
                        concat = torch.cat(out, dim=1)  # [B, V*K, C, H, W]
                        concat = rearrange(concat, "b v c h w -> (b v) c h w")
                    else:
                        concat = torch.cat(out, dim=1)  # [B, V*K, H, W, C]
                        concat = rearrange(concat, "b v h w c -> (b v) c h w")

                    refine_output = self.encoder.forward_update(
                            batch["context"],
                            batch["target"],
                            concat,
                            gaussians,
                            self.decoder,
                            batch["context_remain"] if "context_remain" in batch.keys() else None,
                        )
                        
                    render_output = refine_output['render']
                    gaussians = refine_output['gaussian'][-1]

                    output = render_output[-1]

            else:
                if self.encoder.cfg.no_crop_image:
                    # resize the context image
                    num_view, _, ori_h, ori_w = batch["context"]["image"].shape[1:]
                    resize_h, resize_w = int(np.ceil(ori_h / 64)) * 64, int(np.ceil(ori_w / 64)) * 64
                    if ori_h != resize_h or ori_w != resize_w:
                        batch["context"]["image"] = F.interpolate(
                            batch["context"]["image"].flatten(0, 1), size=(resize_h, resize_w), mode='bilinear', align_corners=True
                        ).view(1, num_view, 3, resize_h, resize_w)

                gaussians = self.encoder(
                    batch["context"],
                    self.global_step,
                    deterministic=False,
                    visualization_dump=visualization_dump,
                )

                if isinstance(gaussians, dict):
                    pred_depths = gaussians["depths"]
                    if "depth" in batch["context"]:
                        depth_gt = batch["context"]["depth"]
                    if "condition_features" in gaussians:
                        condition_features = gaussians["condition_features"]
                    gaussians = gaussians["gaussians"]

                if self.test_cfg.compute_scores and self.encoder.cfg.num_refine > 0:
                    initial_gaussians_for_eval = gaussians

                # refine
                if self.encoder.cfg.num_refine > 0:
                    refine_output = self.encoder.forward_update(
                        batch["context"],
                        batch["target"],
                        condition_features,
                        gaussians,
                        self.decoder,
                        batch["context_remain"] if "context_remain" in batch.keys() else None,
                    )
                    render_output = refine_output['render']
                    gaussians = refine_output['gaussian'][-1]

                    output = render_output[-1]

        # save gaussians
        if self.test_cfg.save_gaussian:
            scene = batch["scene"][0]
            save_path = Path(get_cfg()['output_dir']) / 'gaussians' / (scene + '.ply')
            save_gaussian_ply(gaussians, visualization_dump, batch, save_path, no_align_to_view=self.test_cfg.no_align_to_view,
                              save_gaussian_npz=self.test_cfg.save_gaussian_npz,
                              )

        # test zero-order sh only
        if self.test_cfg.test_zero_order_sh_only:
            gaussians.harmonics[:, :, :, 1:] = 0.

        # save point cloud
        if self.test_cfg.save_point_cloud:
            point_cloud = gaussians.means.reshape(-1, 3).detach()  # [N, 3]

            point_cloud = point_cloud.cpu().numpy()

            colors = batch["context"]["image"][0].permute(0, 2, 3, 1).reshape(-1, 3).detach().cpu().numpy()

            scene = batch["scene"][0]
            save_path = get_cfg()['output_dir'] + '/pointcloud/' + f"{scene}.ply"

            export_to_point_cloud(point_cloud, colors,
                save_path=save_path
                )

        with self.benchmarker.time("decoder", num_calls=v):

            if self.test_cfg.render_input_views:
                camera_poses = batch["context"]["extrinsics"]
            else:
                camera_poses = batch["target"]["extrinsics"]

            if self.test_cfg.stablize_camera:
                stable_poses = render_stabilization_path(
                    camera_poses[0].detach().cpu().numpy(),
                    k_size=self.test_cfg.stab_camera_kernel,
                )

                stable_poses = list(
                    map(
                        lambda x: np.concatenate(
                            (x, np.array([[0.0, 0.0, 0.0, 1.0]])), axis=0
                        ),
                        stable_poses,
                    )
                )
                stable_poses = torch.from_numpy(np.stack(stable_poses, axis=0)).to(
                    camera_poses
                )
                camera_poses = stable_poses.unsqueeze(0)

            if self.test_cfg.render_chunk_size is not None:
                chunk_size = self.test_cfg.render_chunk_size
                num_chunks = math.ceil(camera_poses.shape[1] / chunk_size)

                output = None
                for i in range(num_chunks):
                    start = chunk_size * i
                    end = chunk_size * (i + 1)

                    if self.test_cfg.render_input_views:
                        render_intrinsics = batch["context"]["intrinsics"]
                        render_near = batch["context"]["near"]
                        render_far = batch["context"]["far"]
                    else:
                        render_intrinsics = batch["target"]["intrinsics"]
                        render_near = batch["target"]["near"]
                        render_far = batch["target"]["far"]

                    curr_output = self.decoder.forward(
                        gaussians,
                        camera_poses[:, start:end],
                        render_intrinsics[:, start:end],
                        render_near[:, start:end],
                        render_far[:, start:end],
                        (h, w),
                        depth_mode=None,
                    )

                    if i == 0:
                        output = curr_output
                    else:
                        # ignore depth
                        output.color = torch.cat(
                            (output.color, curr_output.color), dim=1
                        )

            else:
                if self.test_cfg.render_input_views:
                    output = self.decoder.forward(
                        gaussians,
                        camera_poses,
                        batch["context"]["intrinsics"],
                        batch["context"]["near"],
                        batch["context"]["far"],
                        (h, w),
                        depth_mode=None,
                    )

                else:
                    output = self.decoder.forward(
                        gaussians,
                        camera_poses,
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h, w),
                        depth_mode=None,
                    )

        semantic_render_output = None
        if gaussians.semantic_features is not None:
            semantic_render_output, semantic_render_alpha = (
                self.decoder.forward_features(
                    gaussians,
                    camera_poses,
                    (
                        batch["context"]["intrinsics"]
                        if self.test_cfg.render_input_views
                        else batch["target"]["intrinsics"]
                    ),
                    (
                        batch["context"]["near"]
                        if self.test_cfg.render_input_views
                        else batch["target"]["near"]
                    ),
                    (
                        batch["context"]["far"]
                        if self.test_cfg.render_input_views
                        else batch["target"]["far"]
                    ),
                    (h, w),
                )
            )
            if not hasattr(self, "_logged_semantic_gaussian_render"):
                print(
                    "semantic Gaussian render: "
                    f"gaussians={gaussians.semantic_features.shape[1]}, "
                    f"dim={gaussians.semantic_features.shape[2]}, "
                    f"render={tuple(semantic_render_output.shape)}, "
                    f"feature_norm="
                    f"{gaussians.semantic_features.norm(dim=-1).mean().item():.4f}, "
                    f"alpha={semantic_render_alpha.mean().item():.4f}"
                )
                self._logged_semantic_gaussian_render = True
            if not self.test_cfg.render_input_views:
                target_semantic_features = (
                    self.encoder.extract_semantic_teacher_features(
                        batch["target"]["image"], (h, w)
                    )
                )
                semantic_cosine = F.cosine_similarity(
                    semantic_render_output,
                    target_semantic_features,
                    dim=2,
                )
                semantic_valid = semantic_render_alpha >= 0.1
                semantic_cosine_score = semantic_cosine.masked_select(
                    semantic_valid
                ).mean()
                rendered_feature_std = semantic_render_output.permute(
                    0, 1, 3, 4, 2
                ).reshape(-1, semantic_render_output.shape[2]).std(dim=0).mean()
                self.test_step_outputs.setdefault("semantic_cosine", []).append(
                    semantic_cosine_score.item()
                )
                self.test_step_outputs.setdefault(
                    "semantic_render_feature_std", []
                ).append(rendered_feature_std.item())

        # Render the pre-refinement Gaussians with exactly the same cameras.  This
        # is deliberately outside the benchmarked decoder block: it is a V2
        # diagnostic and must not change the reported inference time.
        if initial_gaussians_for_eval is not None:
            if self.test_cfg.render_input_views:
                initial_intrinsics = batch["context"]["intrinsics"]
                initial_near = batch["context"]["near"]
                initial_far = batch["context"]["far"]
            else:
                initial_intrinsics = batch["target"]["intrinsics"]
                initial_near = batch["target"]["near"]
                initial_far = batch["target"]["far"]
            initial_output = self.decoder.forward(
                initial_gaussians_for_eval,
                camera_poses,
                initial_intrinsics,
                initial_near,
                initial_far,
                (h, w),
                depth_mode=None,
            )

        (scene,) = batch["scene"]
        self.test_cfg.output_path = os.path.join(get_cfg()["output_dir"], "metrics")
        path = Path(get_cfg()["output_dir"])

        # save depth
        if self.test_cfg.save_depth:
            depth = (
                visualization_dump["depth"][0, :, :, :, 0, 0].cpu().detach()
            )  # [V, H, W]

            index = batch["context"]["index"][0]

            if self.test_cfg.save_depth_concat_img:
                # concat (img0, img1, depth0, depth1)
                image = batch['context']['image'][0]  # [V, 3, H, W] in [0,1]
                image = rearrange(image, "b c h w -> h (b w) c")  # [H, VW, 3]
                image_concat = (image.detach().cpu().numpy() * 255).astype(np.uint8)  # [H, VW, 3]

                depth_concat = []

            for idx, depth_i in zip(index, depth):
                depth_viz = viz_depth_tensor(
                    1.0 / depth_i, return_numpy=True
                )  # [H, W, 3]

                if self.test_cfg.save_depth_concat_img:
                    depth_concat.append(depth_viz)

                save_path = path / "images" / scene / "depth" / f"{idx:0>6}.png"
                save_dir = os.path.dirname(save_path)
                os.makedirs(save_dir, exist_ok=True)
                Image.fromarray(depth_viz).save(save_path)

                # save depth as npy
                if self.test_cfg.save_depth_npy:
                    depth_npy = depth_i.detach().cpu().numpy()
                    save_path = path / "images" / scene / "depth" / f"{idx:0>6}.npy"
                    save_dir = os.path.dirname(save_path)
                    os.makedirs(save_dir, exist_ok=True)
                    np.save(save_path, depth_npy)

            if self.test_cfg.save_depth_concat_img:
                depth_concat = np.concatenate(depth_concat, axis=1)  # [H, VW, 3]
                concat = np.concatenate((image_concat, depth_concat), axis=0)  # [2H, VW, 3]

                save_path = path / "images" / scene / "depth" /  f"img_depth_{scene}.png"
                save_dir = os.path.dirname(save_path)
                os.makedirs(save_dir, exist_ok=True)
                Image.fromarray(concat).save(save_path)

        images_prob = output.color[0]
        if self.test_cfg.render_input_views:
            rgb_gt = batch["context"]["image"][0]
        else:
            rgb_gt = batch["target"]["image"][0]

        # Save images.
        if self.test_cfg.save_image:
            if self.test_cfg.save_gt_image:
                for index, color, gt in zip(
                    batch["target"]["index"][0], images_prob, rgb_gt
                ):
                    save_image(color, path / "images" / scene / f"color/{index:0>6}.png")
                    save_image(gt, path / "images" / scene / f"color/{index:0>6}_gt.png")
            else:
                for index, color in zip(batch["target"]["index"][0], images_prob):
                    save_image(color, path / "images" / scene / f"color/{index:0>6}.png")

        # save video
        if self.test_cfg.save_video:
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])[:20]
            save_video(
                [a for a in images_prob],
                path / "videos" / f"{scene}_frame_{frame_str}.mp4",
            )

        # compute scores
        if self.test_cfg.compute_scores:
            if batch_idx < self.test_cfg.eval_time_skip_steps:
                self.time_skip_steps_dict["encoder"] += 1
                self.time_skip_steps_dict["decoder"] += v

            rgb = images_prob

            if f"psnr" not in self.test_step_outputs:
                self.test_step_outputs[f"psnr"] = []
            if f"ssim" not in self.test_step_outputs:
                self.test_step_outputs[f"ssim"] = []
            if f"lpips" not in self.test_step_outputs:
                self.test_step_outputs[f"lpips"] = []

            self.test_step_outputs[f"psnr"].append(
                compute_psnr(rgb_gt, rgb).mean().item()
            )
            self.test_step_outputs[f"ssim"].append(
                compute_ssim(rgb_gt, rgb).mean().item()
            )
            self.test_step_outputs[f"lpips"].append(
                compute_lpips(rgb_gt, rgb).mean().item()
            )

            if initial_output is not None:
                init_rgb = initial_output.color[0]
                init_psnr = compute_psnr(rgb_gt, init_rgb).mean().item()
                init_ssim = compute_ssim(rgb_gt, init_rgb).mean().item()
                init_lpips = compute_lpips(rgb_gt, init_rgb).mean().item()
                self.test_step_outputs.setdefault("init_psnr", []).append(init_psnr)
                self.test_step_outputs.setdefault("init_ssim", []).append(init_ssim)
                self.test_step_outputs.setdefault("init_lpips", []).append(init_lpips)
                self.test_step_outputs.setdefault("refine_psnr_gain", []).append(
                    self.test_step_outputs["psnr"][-1] - init_psnr
                )
                self.test_step_outputs.setdefault("refine_ssim_gain", []).append(
                    self.test_step_outputs["ssim"][-1] - init_ssim
                )
                self.test_step_outputs.setdefault("refine_lpips_reduction", []).append(
                    init_lpips - self.test_step_outputs["lpips"][-1]
                )

            if "boundary" in batch["target"]:
                target_boundary = batch["target"]["boundary"].to(rgb.dtype)
                confidence = batch["target"]["boundary_confidence"].to(rgb.dtype)
                gate = torch.where(
                    confidence >= self.encoder.cfg.semantic_boundary_confidence_floor,
                    confidence,
                    torch.zeros_like(confidence),
                )
                predicted_boundary = rgb_to_soft_boundary(
                    rgb, self.encoder.cfg.semantic_boundary_gain
                )
                boundary_l1 = balanced_boundary_l1(
                    predicted_boundary, target_boundary, gate
                )
                trusted = gate > 0
                predicted_positive = predicted_boundary >= 0.5
                target_positive = target_boundary >= 0.5
                true_positive = (
                    predicted_positive & target_positive & trusted
                ).sum().float()
                false_positive = (
                    predicted_positive & ~target_positive & trusted
                ).sum().float()
                false_negative = (
                    ~predicted_positive & target_positive & trusted
                ).sum().float()
                boundary_f1 = (2 * true_positive) / (
                    2 * true_positive + false_positive + false_negative
                ).clamp_min(1)
                self.test_step_outputs.setdefault("boundary_l1", []).append(
                    boundary_l1.item()
                )
                self.test_step_outputs.setdefault("boundary_f1", []).append(
                    boundary_f1.item()
                )

                if initial_output is not None:
                    initial_boundary = rgb_to_soft_boundary(
                        initial_output.color[0],
                        self.encoder.cfg.semantic_boundary_gain,
                    )
                    init_boundary_l1 = balanced_boundary_l1(
                        initial_boundary, target_boundary, gate
                    )
                    init_positive = initial_boundary >= 0.5
                    init_true_positive = (
                        init_positive & target_positive & trusted
                    ).sum().float()
                    init_false_positive = (
                        init_positive & ~target_positive & trusted
                    ).sum().float()
                    init_false_negative = (
                        ~init_positive & target_positive & trusted
                    ).sum().float()
                    init_boundary_f1 = (2 * init_true_positive) / (
                        2 * init_true_positive
                        + init_false_positive
                        + init_false_negative
                    ).clamp_min(1)
                    self.test_step_outputs.setdefault("init_boundary_l1", []).append(
                        init_boundary_l1.item()
                    )
                    self.test_step_outputs.setdefault("init_boundary_f1", []).append(
                        init_boundary_f1.item()
                    )

            # compute depth metrics
            if pred_depths is not None and depth_gt is not None:
                if f"abs_rel" not in self.test_step_outputs:
                    self.test_step_outputs[f"abs_rel"] = []
                if f"rmse" not in self.test_step_outputs:
                    self.test_step_outputs[f"rmse"] = []
                if f"a1" not in self.test_step_outputs:
                    self.test_step_outputs[f"a1"] = []

                pred_depths = pred_depths[0]  # [V, H, W]
                depth_gt = depth_gt[0]  # [V, H, W]

                near = batch["context"]["near"][...,
                                                None, None][0]  # [V, 1, 1]
                far = batch["context"]["far"][..., None, None][0]  # [V, 1, 1]

                valid = (depth_gt >= near) & (depth_gt <= far)

                all_metrics = compute_depth_errors(depth_gt[valid].detach().cpu().numpy(),
                                                   pred_depths[valid].detach().cpu().numpy())
                print(all_metrics)

                self.test_step_outputs[f"abs_rel"].append(
                    float(all_metrics[0]))
                self.test_step_outputs[f"rmse"].append(float(all_metrics[2]))
                self.test_step_outputs[f"a1"].append(float(all_metrics[4]))

    def on_test_end(self) -> None:
        out_dir = Path(self.test_cfg.output_path)
        saved_scores = {}
        if self.test_cfg.compute_scores:
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.dump(out_dir / "benchmark.json")

            for metric_name, metric_scores in self.test_step_outputs.items():
                avg_scores = sum(metric_scores) / len(metric_scores)
                saved_scores[metric_name] = avg_scores
                print(metric_name, avg_scores)
                with (out_dir / f"scores_{metric_name}_all.json").open("w") as f:
                    json.dump(metric_scores, f)
                metric_scores.clear()

            for tag, times in self.benchmarker.execution_times.items():
                times = times[int(self.time_skip_steps_dict[tag]) :]
                saved_scores[tag] = [len(times), np.mean(times)]
                print(
                    f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call"
                )
                self.time_skip_steps_dict[tag] = 0

            with (out_dir / f"scores_all_avg.json").open("w") as f:
                json.dump(saved_scores, f)
            self.benchmarker.clear_history()
        else:
            self.benchmarker.dump(out_dir / "benchmark.json")
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.summarize()


    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {[a[:20] for a in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1

        pred_depths = None

        if self.test_cfg.inference_window_size is not None:
            assert self.test_cfg.inference_window_size > 0
            window = self.test_cfg.inference_window_size
            total_views = batch["context"]["image"].size(1)

            window_indices = sliding_window_indices(total_views, window, 0)

            all_gaussians = []
            all_states = []  # for global refinement

            # sliding window inference
            for indices in window_indices:
                start, end = indices
                curr_window_input = {
                    "extrinsics": batch["context"]["extrinsics"][:, start:end],
                    "intrinsics": batch["context"]["intrinsics"][:, start:end],
                    "image": batch["context"]["image"][:, start:end],
                    "near": batch["context"]["near"][:, start:end],
                    "far": batch["context"]["far"][:, start:end],
                    "index": batch["context"]["index"][:, start:end],
                }

                curr_gaussians = self.encoder(
                    curr_window_input,
                    self.global_step,
                    deterministic=False,
                )

                if isinstance(curr_gaussians, dict):
                    pred_depths = curr_gaussians["depths"]
                    if "depth" in batch["context"]:
                        depth_gt = batch["context"]["depth"]
                    if "condition_features" in curr_gaussians:
                        condition_features = curr_gaussians["condition_features"]  # [BV, C, H, W]
                    curr_gaussians = curr_gaussians["gaussians"]

                all_gaussians.append(curr_gaussians)
                all_states.append(condition_features)

            # merge all gaussians
            gaussians = merge_gaussians(all_gaussians)

            # global refine after simply combining local window gaussians
            if self.encoder.cfg.num_refine > 0:
                # combine all condition features
                out = []
                is_ori_feature = True
                for i in range(len(all_states)):
                    curr = all_states[i]
                    if curr.dim() == 4:
                        # [BV, C, H, W]
                        curr = rearrange(curr, "(b v) c h w -> b v c h w", b=b)
                        is_ori_feature = True
                    elif curr.dim() == 2:
                        # [BVHW, C]
                        curr = rearrange(curr, "(b v h w) c -> b v h w c", b=b,
                        h=h // self.encoder.cfg.latent_downsample,
                        w=w // self.encoder.cfg.latent_downsample,
                        )
                        is_ori_feature = False
                    else:
                        raise NotImplementedError

                    out.append(curr)

                if is_ori_feature:
                    concat = torch.cat(out, dim=1)  # [B, V*K, C, H, W]
                    concat = rearrange(concat, "b v c h w -> (b v) c h w")
                else:
                    concat = torch.cat(out, dim=1)  # [B, V*K, H, W, C]
                    concat = rearrange(concat, "b v h w c -> (b v) c h w")

                condition_features = concat

            gaussians_softmax = gaussians

        else:
            gaussians_softmax = self.encoder(
                batch["context"],
                self.global_step,
                deterministic=False,
            )

            if isinstance(gaussians_softmax, dict):
                pred_depths = gaussians_softmax["depths"]
            if "depth" in batch["context"]:
                depth_gt = batch["context"]["depth"]  # [B, V, H, W]
            if "condition_features" in gaussians_softmax:
                condition_features = gaussians_softmax["condition_features"]
            gaussians_softmax = gaussians_softmax["gaussians"]

        output_softmax = self.decoder.forward(
            gaussians_softmax,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            depth_mode=None,
        )

        # refine
        if self.encoder.cfg.num_refine > 0:
            refine_output = self.encoder.forward_update(
                batch["context"],
                batch["target"],
                condition_features,
                gaussians_softmax,
                self.decoder,
                batch["context_remain"] if "context_remain" in batch.keys() else None,
            )

            render_output = refine_output['render']

            output_softmax = render_output[-1]

        rgb_softmax = output_softmax.color[0]

        # Compute validation metrics.
        rgb_gt = batch["target"]["image"][0]
        for tag, rgb in zip(("val",), (rgb_softmax,)):
            psnr = compute_psnr(rgb_gt, rgb).mean()
            self.log(f"val/psnr_{tag}", psnr)
            lpips = compute_lpips(rgb_gt, rgb).mean()
            self.log(f"val/lpips_{tag}", lpips)
            ssim = compute_ssim(rgb_gt, rgb).mean()
            self.log(f"val/ssim_{tag}", ssim)

        # viz depth
        if pred_depths is not None:
            # only visualize predicted depth
            pred_depths = pred_depths[0]  # [V, H, W]

            # gaussian downsample
            # downsample image to depth resolution
            if pred_depths.shape[1:] != batch["context"]["image"].shape[-2:]:
                input_images = F.interpolate(
                    batch["context"]["image"][0],
                    size=pred_depths.shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                ).squeeze(1)
            else:
                input_images = batch["context"]["image"][0]  # [N, 3, H, W]

            inverse_depth_pred = 1.0 / pred_depths

            concat = []
            for i in range(inverse_depth_pred.size(0)):
                concat.append(inverse_depth_pred[i])

            concat = torch.cat(concat, dim=1)  # [H, W*N]

            depth_viz = viz_depth_tensor(concat.cpu().detach())  # [3, H, W*N]

            # also concat images
            concat_img = [img for img in input_images]
            concat_img = torch.cat(concat_img, dim=-1) * 255  # [3, H, W*N]

            concat = torch.cat(
                (concat_img.cpu().detach(), depth_viz), dim=1
            )  # [3, H*2, W*N]

            # reshape when the number of input images is too large
            # otherwise the image will be too wide
            num_inputs = input_images.shape[0]
            width = input_images.shape[-1]
            if num_inputs > 8:
                rows = 4
                assert num_inputs % rows == 0
                stride = num_inputs // rows
                out = []
                for i in range(rows):
                    out.append(concat[:, :, width * stride * i : width * stride * (i + 1)])

                concat = torch.cat(out, dim=1)  # [3, H*2*R, W*N/R]

                # resize to half resolution to save space
                concat = F.interpolate(concat.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=True).squeeze(0)

            # viz gt depth
            if False:
                tmp_gt = batch["context"]["depth"][0]  # [V, H, W]
                tmp_gt[tmp_gt == 0] = 999999.
                tmp_inv_gt = 1. / tmp_gt
                tmp_concat = [tmp for tmp in tmp_inv_gt]
                tmp_concat = torch.cat(tmp_concat, dim=1)  # [H, W*N]
                tmp_concat = viz_depth_tensor(tmp_concat.cpu().detach())
                concat = torch.cat((concat, tmp_concat), dim=1)

            self.logger.log_image(
                "depth",
                [concat],
                step=self.global_step,
                caption=batch["scene"],
            )

        if batch["context"]["image"][0].shape[0] > 16:
            # when the number of input images is too large
            # subsample input images to save space
            viz_input = batch["context"]["image"][0][::4]
            tag = "Context (1/4)"
        elif batch["context"]["image"][0].shape[0] > 8:
            # when the number of input images is too large
            # subsample input images to save space
            viz_input = batch["context"]["image"][0][::2]
            tag = "Context (1/2)"
        else:
            viz_input = batch["context"]["image"][0]
            tag = "Context"

        # Construct comparison image.
        comparison = hcat(
            add_label(vcat(*viz_input), tag),
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            add_label(vcat(*rgb_softmax), "Target (Prediction)"),
        )

        self.logger.log_image(
            "comparison",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )

        if not self.train_cfg.no_log_projections:
            # Render projections and construct projection image.
            projections = hcat(
                *render_projections(
                    gaussians_softmax,
                    256,
                    extra_label="(Prediction)",
                )[0]
            )
            self.logger.log_image(
                "projection",
                [prep_image(add_border(projections))],
                step=self.global_step,
            )

        if self.encoder_visualizer is not None:
            for k, image in self.encoder_visualizer.visualize(
                batch["context"], self.global_step
            ).items():
                self.logger.log_image(k, [prep_image(image)], step=self.global_step)

        # Run video validation step.
        if not self.train_cfg.no_viz_video:
            self.render_video_interpolation(batch)
            if self.train_cfg.extended_visualization:
                self.render_video_interpolation_exaggerated(batch)

    def on_validation_epoch_end(self) -> None:
        """hack to run the full validation"""
        if self.trainer.sanity_checking and self.global_rank == 0:
            print(self.encoder)  # log the model to wandb log files

        if (not self.trainer.sanity_checking) and (self.eval_data_cfg is not None):
            self.eval_cnt = self.eval_cnt + 1
            if self.eval_cnt % self.train_cfg.eval_model_every_n_val == 0:
                # backup current ckpt before running full test sets eval
                if self.train_cfg.eval_save_model:
                    ckpt_saved_path = (
                        self.trainer.checkpoint_callback.format_checkpoint_name(
                            dict(
                                epoch=self.trainer.current_epoch,
                                step=self.trainer.global_step,
                            )
                        )
                    )
                    backup_dir = str(
                        Path(ckpt_saved_path).parent.parent / "checkpoints_backups"
                    )
                    if self.global_rank == 0:
                        os.makedirs(backup_dir, exist_ok=True)
                    ckpt_saved_path = os.path.join(
                        backup_dir, os.path.basename(ckpt_saved_path)
                    )
                    # call save_checkpoint on ALL process as suggested by pytorch_lightning
                    self.trainer.save_checkpoint(
                        ckpt_saved_path,
                        weights_only=True,
                    )
                    if self.global_rank == 0:
                        print(f"backup model to {ckpt_saved_path}.")

                # run full test sets eval on rank=0 device
                self.run_full_test_sets_eval()

    @rank_zero_only
    def run_full_test_sets_eval(self) -> None:
        start_t = time.time()

        pred_depths = None
        depth_gt = None

        full_testsets = self.trainer.datamodule.test_dataloader(
            dataset_cfg=self.eval_data_cfg
        )
        scores_dict = {}

        for score_tag in ("psnr", "ssim", "lpips"):
            scores_dict[score_tag] = {}
            for method_tag in ("deterministic", "probabilistic"):
                scores_dict[score_tag][method_tag] = []

        # evaluate input view depth
        if self.train_cfg.eval_depth:
            for score_tag in ("abs_rel", "rmse", "a1"):
                scores_dict[score_tag] = {}
                for method_tag in ("deterministic", "probabilistic"):
                    scores_dict[score_tag][method_tag] = []

        self.benchmarker.clear_history()
        time_skip_first_n_steps = min(
            self.train_cfg.eval_time_skip_steps, len(full_testsets)
        )
        time_skip_steps_dict = {"encoder": 0, "decoder": 0}
        for batch_idx, batch in tqdm(
            enumerate(full_testsets),
            total=min(len(full_testsets), self.train_cfg.eval_data_length),
        ):
            if batch_idx >= self.train_cfg.eval_data_length:
                break

            batch = self.data_shim(batch)
            batch = self.transfer_batch_to_device(batch, "cuda", dataloader_idx=0)

            # Render Gaussians.
            b, v, _, h, w = batch["target"]["image"].shape
            assert b == 1
            if batch_idx < time_skip_first_n_steps:
                time_skip_steps_dict["encoder"] += 1
                time_skip_steps_dict["decoder"] += v

            with self.benchmarker.time("encoder"):
                if self.test_cfg.inference_window_size is not None:
                    assert self.test_cfg.inference_window_size > 0
                    window = self.test_cfg.inference_window_size
                    total_views = batch["context"]["image"].size(1)

                    window_indices = sliding_window_indices(total_views, window, 0)

                    all_gaussians = []
                    all_states = []  # for global refinement

                    # sliding window inference
                    for indices in window_indices:
                        start, end = indices
                        curr_window_input = {
                            "extrinsics": batch["context"]["extrinsics"][:, start:end],
                            "intrinsics": batch["context"]["intrinsics"][:, start:end],
                            "image": batch["context"]["image"][:, start:end],
                            "near": batch["context"]["near"][:, start:end],
                            "far": batch["context"]["far"][:, start:end],
                            "index": batch["context"]["index"][:, start:end],
                        }

                        curr_gaussians = self.encoder(
                            curr_window_input,
                            self.global_step,
                            deterministic=False,
                        )

                        if isinstance(curr_gaussians, dict):
                            pred_depths = curr_gaussians["depths"]
                            if "depth" in batch["context"]:
                                depth_gt = batch["context"]["depth"]
                            if "condition_features" in curr_gaussians:
                                condition_features = curr_gaussians["condition_features"]  # [BV, C, H, W]
                            else:
                                condition_features = None
                            curr_gaussians = curr_gaussians["gaussians"]

                        all_gaussians.append(curr_gaussians)
                        all_states.append(condition_features)

                    # merge all gaussians
                    # ['means', 'covariances', 'harmonics', 'opacities']
                    gaussians = merge_gaussians(all_gaussians)

                    gaussians_probabilistic = gaussians

                    # global refine after simply combining local window gaussians
                    if self.encoder.cfg.num_refine > 0:
                        # combine all condition features
                        out = []
                        is_ori_feature = False
                        for i in range(len(all_states)):
                            curr = all_states[i]
                            if curr.dim() == 4:  
                                # [BV, C, H, W]
                                curr = rearrange(curr, "(b v) c h w -> b v c h w", b=b)
                                is_ori_feature = True
                            elif curr.dim() == 2:
                                # [BVHW, C]
                                curr = rearrange(curr, "(b v h w) c -> b v h w c", b=b, 
                                h=h // self.encoder.cfg.latent_downsample,
                                w=w // self.encoder.cfg.latent_downsample,
                                )
                                is_ori_feature = False
                            else:
                                raise NotImplementedError

                            out.append(curr)

                        # concat
                        if is_ori_feature:
                            concat = torch.cat(out, dim=1)  # [B, V*K, C, H, W]
                            concat = rearrange(concat, "b v c h w -> (b v) c h w")
                        else:
                            concat = torch.cat(out, dim=1)  # [B, V*K, H, W, C]
                            concat = rearrange(concat, "b v h w c -> (b v) c h w")

                        condition_features = concat

                else:
                    gaussians_probabilistic = self.encoder(
                        batch["context"],
                        self.global_step,
                        deterministic=False,
                    )

                    if isinstance(gaussians_probabilistic, dict):
                        pred_depths = gaussians_probabilistic["depths"]
                        if "depth" in batch["context"]:
                            depth_gt = batch["context"]["depth"]
                        if "condition_features" in gaussians_probabilistic:
                            condition_features = gaussians_probabilistic["condition_features"]
                        gaussians_probabilistic = gaussians_probabilistic["gaussians"]

            with self.benchmarker.time("decoder", num_calls=v):
                output_probabilistic = self.decoder.forward(
                    gaussians_probabilistic,
                    batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                    depth_mode=None,
                )

                # refine
                if self.encoder.cfg.num_refine > 0:
                    refine_output = self.encoder.forward_update(
                        batch["context"],
                        batch["target"],
                        condition_features,
                        gaussians_probabilistic,
                        self.decoder,
                        batch["context_remain"] if "context_remain" in batch.keys() else None,
                    )

                    render_output = refine_output['render']

                    output_probabilistic = render_output[-1]

            rgbs = [output_probabilistic.color[0]]
            tags = ["probabilistic"]

            if self.train_cfg.eval_deterministic:
                gaussians_deterministic = self.encoder(
                    batch["context"],
                    self.global_step,
                    deterministic=True,
                )
                output_deterministic = self.decoder.forward(
                    gaussians_deterministic,
                    batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                )
                rgbs.append(output_deterministic.color[0])
                tags.append("deterministic")

            # Compute validation metrics.
            rgb_gt = batch["target"]["image"][0]
            for tag, rgb in zip(tags, rgbs):
                scores_dict["psnr"][tag].append(
                    compute_psnr(rgb_gt, rgb).mean().item()
                )
                scores_dict["lpips"][tag].append(
                    compute_lpips(rgb_gt, rgb).mean().item()
                )
                scores_dict["ssim"][tag].append(
                    compute_ssim(rgb_gt, rgb).mean().item()
                )

            # compute depth metrics
            if pred_depths is not None and depth_gt is not None and depth_gt.max() > 0:
                assert pred_depths is not None and depth_gt is not None

                pred_depths = pred_depths[0]  # [V, H, W]

                # gaussian downsample
                if pred_depths.shape[1:] != batch["context"]["image"].shape[-2:]:
                    pred_depths = F.interpolate(
                        pred_depths.unsqueeze(1),
                        size=batch["context"]["image"].shape[-2:],
                        mode="bilinear",
                        align_corners=True,
                    ).squeeze(1)

                depth_gt = depth_gt[0]  # [V, H, W]

                near = batch["context"]["near"][...,
                                                None, None][0]  # [V, 1, 1]
                far = batch["context"]["far"][..., None, None][0]  # [V, 1, 1]

                valid = (depth_gt >= near) & (depth_gt <= far)

                all_metrics = compute_depth_errors(depth_gt[valid].detach().cpu().numpy(),
                                                   pred_depths[valid].detach().cpu().numpy())
                scores_dict["abs_rel"]["probabilistic"].append(all_metrics[0])
                scores_dict["rmse"]["probabilistic"].append(all_metrics[2])
                scores_dict["a1"]["probabilistic"].append(all_metrics[4])

        # summarise scores and log to logger
        for score_tag, methods in scores_dict.items():
            for method_tag, cur_scores in methods.items():
                if len(cur_scores) > 0:
                    cur_mean = sum(cur_scores) / len(cur_scores)
                    self.log(f"test/{score_tag}", cur_mean)
        # summarise run time
        for tag, times in self.benchmarker.execution_times.items():
            times = times[int(time_skip_steps_dict[tag]) :]
            print(f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call")
            self.log(f"test/runtime_avg_{tag}", np.mean(times))
        self.benchmarker.clear_history()

        overall_eval_time = time.time() - start_t
        print(f"Eval total time cost: {overall_eval_time:.3f}s")
        self.log("test/runtime_all", overall_eval_time)

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        if self.train_cfg.no_log_video:
            return
        # Render probabilistic estimate of scene.
        gaussians_prob = self.encoder(batch["context"], self.global_step, False)

        if isinstance(gaussians_prob, dict):
            gaussians_prob = gaussians_prob["gaussians"]

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # Color-map the result.
        def depth_map(result):
            near = result[result > 0][:16_000_000].quantile(0.01).log()
            far = result.reshape(-1)[:16_000_000].quantile(0.99).log()
            result = result.log()
            result = 1 - (result - near) / (far - near)
            return apply_color_map_to_image(result, "turbo")

        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output_prob = self.decoder.forward(
            gaussians_prob, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images_prob = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_prob.color[0], depth_map(output_prob.depth[0]))
        ]

        images = [
            add_border(
                hcat(
                    add_label(image_prob, "Prediction"),
                )
            )
            for image_prob, _ in zip(images_prob, images_prob)
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]

        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    def configure_optimizers(self):
        if self.optimizer_cfg.lr_depth > 0:
            pretrained_params = []
            new_params = []

            for name, param in self.named_parameters():
                if "depth_predictor" in name:
                    pretrained_params.append(param)
                else:
                    new_params.append(param)

            if self.optimizer_cfg.adamw_8bit:
                optimizer = AdamW8bit(
                    [
                        {
                            "params": pretrained_params,
                            "lr": self.optimizer_cfg.lr_depth,
                        },
                        {"params": new_params, "lr": self.optimizer_cfg.lr},
                    ],
                    weight_decay=self.optimizer_cfg.weight_decay,
                )
            else:
                optimizer = torch.optim.AdamW(
                    [
                        {
                            "params": pretrained_params,
                            "lr": self.optimizer_cfg.lr_depth,
                        },
                        {"params": new_params, "lr": self.optimizer_cfg.lr},
                    ],
                    weight_decay=self.optimizer_cfg.weight_decay,
                )

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                [self.optimizer_cfg.lr_monodepth, self.optimizer_cfg.lr],
                self.trainer.max_steps + 10,
                pct_start=self.optimizer_cfg.warm_up_ratio,
                cycle_momentum=False,
                anneal_strategy="cos",
            )

        elif self.optimizer_cfg.lr_monodepth > 0:
            pretrained_params = []
            new_params = []

            for name, param in self.named_parameters():
                if "pretrained" in name:
                    pretrained_params.append(param)
                else:
                    new_params.append(param)

            if self.optimizer_cfg.adamw_8bit:
                optimizer = AdamW8bit(
                    [
                        {
                            "params": pretrained_params,
                            "lr": self.optimizer_cfg.lr_monodepth,
                        },
                        {"params": new_params, "lr": self.optimizer_cfg.lr},
                    ],
                    weight_decay=self.optimizer_cfg.weight_decay,
                )
            else:
                optimizer = torch.optim.AdamW(
                    [
                        {
                            "params": pretrained_params,
                            "lr": self.optimizer_cfg.lr_monodepth,
                        },
                        {"params": new_params, "lr": self.optimizer_cfg.lr},
                    ],
                    weight_decay=self.optimizer_cfg.weight_decay,
                )

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                [self.optimizer_cfg.lr_monodepth, self.optimizer_cfg.lr],
                self.trainer.max_steps + 10,
                pct_start=self.optimizer_cfg.warm_up_ratio,
                cycle_momentum=False,
                anneal_strategy="cos",
            )

        else:
            if self.optimizer_cfg.adamw_8bit:
                optimizer = AdamW8bit(
                    self.parameters(),
                    lr=self.optimizer_cfg.lr,
                    weight_decay=self.optimizer_cfg.weight_decay,
                )
            else:
                optimizer = optim.AdamW(
                    self.parameters(),
                    lr=self.optimizer_cfg.lr,
                    weight_decay=self.optimizer_cfg.weight_decay,
                )

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                self.optimizer_cfg.lr,
                self.trainer.max_steps + 10,
                pct_start=self.optimizer_cfg.warm_up_ratio,
                cycle_momentum=False,
                anneal_strategy="cos",
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def sliding_window_indices(N, x, y):
    indices = []
    start = 0
    while start + x < N:  # Ensure the last window is not processed here
        end = min(start + x, N)
        indices.append([start, end])
        start += (x - y)  # Move the start by the window size minus overlap
    
    # Append the last window [N-x, N]
    indices.append([N - x, N])
    
    return indices
