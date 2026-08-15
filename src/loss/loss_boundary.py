from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.semantic_boundary import balanced_boundary_l1, rgb_to_soft_boundary
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossBoundaryCfg:
    weight: float
    gain: float
    confidence_floor: float


@dataclass
class LossBoundaryCfgWrapper:
    boundary: LossBoundaryCfg


class LossBoundary(Loss[LossBoundaryCfg, LossBoundaryCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians | None,
        global_step: int,
        valid_depth_mask: Tensor | None = None,
        loss_on_input_views: bool = False,
        half_res_lpips: bool = False,
    ) -> Float[Tensor, ""]:
        del gaussians, global_step, valid_depth_mask, half_res_lpips
        views = batch["context"] if loss_on_input_views else batch["target"]
        if "boundary" not in views or "boundary_confidence" not in views:
            raise KeyError("Boundary loss requires boundary and boundary_confidence")
        target = views["boundary"].to(prediction.color.dtype)
        confidence = views["boundary_confidence"].to(prediction.color.dtype)
        gate = torch.where(
            confidence >= self.cfg.confidence_floor,
            confidence,
            torch.zeros_like(confidence),
        )
        predicted = rgb_to_soft_boundary(prediction.color, self.cfg.gain)
        return self.cfg.weight * balanced_boundary_l1(predicted, target, gate)
