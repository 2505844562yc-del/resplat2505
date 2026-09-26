from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .types import RayHypotheses


@dataclass
class RayMixtureCfg:
    primary_bandwidth_bins: float = 2.0
    secondary_bandwidth_bins: float = 2.0
    suppression_width_bins: float = 3.0
    gate_hidden_channels: int = 16
    boundary_prior_strength: float = 6.0
    entropy_prior_strength: float = 1.5
    mass_prior_strength: float = 2.0
    separation_prior_strength: float = 1.5
    gate_bias: float = -4.0
    eps: float = 1e-8


class BoundaryAwareRayMixtureDecoder(nn.Module):
    """Convert a depth posterior into one or two semantic-conditioned modes.

    Mode centers are soft probability-weighted estimates. Argmax is used only to
    choose detached basin centers; gradients still flow through probability mass,
    metric candidate depths, and the learned gate residual.
    """

    def __init__(self, cfg: Optional[RayMixtureCfg] = None) -> None:
        super().__init__()
        self.cfg = cfg or RayMixtureCfg()
        cfg = self.cfg
        hidden = cfg.gate_hidden_channels
        self.gate_residual = nn.Sequential(
            nn.Conv2d(5, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.zeros_(self.gate_residual[-1].weight)
        nn.init.zeros_(self.gate_residual[-1].bias)

    @staticmethod
    def _resize_signal(signal: Tensor, size: tuple[int, int]) -> Tensor:
        b, v = signal.shape[:2]
        signal = signal.reshape(b * v, *signal.shape[2:])
        if signal.shape[-2:] != size:
            signal = F.interpolate(signal, size=size, mode="bilinear", align_corners=False)
        return signal

    def _soft_mode(
        self,
        probabilities: Tensor,
        metric_depths: Tensor,
        center_index: Tensor,
        bandwidth: float,
        extra_weight: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        d = probabilities.shape[1]
        bins = torch.arange(d, device=probabilities.device, dtype=probabilities.dtype)
        bins = bins.view(1, d, 1, 1)
        kernel = torch.exp(-0.5 * ((bins - center_index) / bandwidth) ** 2)
        weights = probabilities * kernel
        if extra_weight is not None:
            weights = weights * extra_weight
        mass = weights.sum(dim=1, keepdim=True)
        center = (weights * metric_depths).sum(dim=1, keepdim=True) / mass.clamp_min(
            self.cfg.eps
        )
        return center, mass

    def forward(
        self,
        depth_probabilities: Tensor,
        depth_candidates: Tensor,
        boundary_probability: Tensor,
        semantic_confidence: Tensor,
    ) -> RayHypotheses:
        if depth_probabilities.ndim != 5:
            raise ValueError("depth_probabilities must have shape [B,V,D,H,W]")
        if depth_candidates.shape != depth_probabilities.shape:
            raise ValueError("depth_candidates must match depth_probabilities")

        b, v, d, h, w = depth_probabilities.shape
        probabilities = depth_probabilities.reshape(b * v, d, h, w).float()
        metric_depths = depth_candidates.reshape(b * v, d, h, w).float()
        probabilities = probabilities.clamp_min(0)
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(
            self.cfg.eps
        )

        primary_index = probabilities.argmax(dim=1, keepdim=True).detach()
        primary_depth, primary_mass = self._soft_mode(
            probabilities,
            metric_depths,
            primary_index,
            self.cfg.primary_bandwidth_bins,
        )

        bins = torch.arange(d, device=probabilities.device, dtype=probabilities.dtype)
        bins = bins.view(1, d, 1, 1)
        suppression = 1.0 - torch.exp(
            -0.5 * ((bins - primary_index) / self.cfg.suppression_width_bins) ** 2
        )
        secondary_distribution = probabilities * suppression
        secondary_index = secondary_distribution.argmax(dim=1, keepdim=True).detach()
        secondary_depth, secondary_mass = self._soft_mode(
            probabilities,
            metric_depths,
            secondary_index,
            self.cfg.secondary_bandwidth_bins,
            extra_weight=suppression,
        )

        entropy = -(probabilities * probabilities.clamp_min(self.cfg.eps).log()).sum(
            dim=1, keepdim=True
        ) / torch.log(torch.tensor(float(d), device=probabilities.device))
        expected_depth = (probabilities * metric_depths).sum(dim=1, keepdim=True)
        separation = (secondary_depth - primary_depth).abs() / expected_depth.clamp_min(
            self.cfg.eps
        )
        separation = separation.clamp(max=2.0)

        boundary = self._resize_signal(boundary_probability.float(), (h, w)).clamp(0, 1)
        confidence = self._resize_signal(semantic_confidence.float(), (h, w)).clamp(0, 1)
        mode_mass = secondary_mass / (primary_mass + secondary_mass).clamp_min(self.cfg.eps)

        gate_features = torch.cat(
            [boundary, confidence, entropy, mode_mass, separation], dim=1
        )
        prior_logit = (
            self.cfg.gate_bias
            + self.cfg.boundary_prior_strength * boundary * confidence
            + self.cfg.entropy_prior_strength * entropy
            + self.cfg.mass_prior_strength * mode_mass
            + self.cfg.separation_prior_strength * separation
        )
        second_active = torch.sigmoid(prior_logit + self.gate_residual(gate_features))
        second_weight = (second_active * mode_mass).clamp(0, 1)
        mixture_weights = torch.cat([1.0 - second_weight, second_weight], dim=1)
        hypothesis_depths = torch.cat([primary_depth, secondary_depth], dim=1)
        masses = torch.cat([primary_mass, secondary_mass], dim=1)

        def restore(x: Tensor) -> Tensor:
            return x.reshape(b, v, *x.shape[1:])

        return RayHypotheses(
            depths=restore(hypothesis_depths),
            mixture_weights=restore(mixture_weights),
            second_active_probability=restore(second_active),
            posterior_entropy=restore(entropy),
            normalized_separation=restore(separation),
            mode_masses=restore(masses),
        )
