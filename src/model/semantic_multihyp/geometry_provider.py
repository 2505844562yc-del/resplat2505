from typing import Any

import torch
from torch import nn

from .types import GeometryEvidence, GeometryProviderOutput


class FrozenGeometryProvider(nn.Module):
    """Expose ReSplat initialization as a frozen, replaceable geometry service.

    The wrapper deliberately rejects recurrent configurations. This keeps the new
    method's computational graph independent from ReSplat's recurrent updater.
    """

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        num_refine = getattr(getattr(encoder, "cfg", None), "num_refine", None)
        if num_refine != 0:
            raise ValueError("FrozenGeometryProvider requires encoder.cfg.num_refine == 0")
        if not getattr(getattr(encoder, "cfg", None), "return_geometry_evidence", False):
            raise ValueError(
                "Set encoder.cfg.return_geometry_evidence=true before wrapping the encoder"
            )

        self.encoder = encoder
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # The provider remains frozen even while the surrounding model trains.
        self.encoder.eval()
        return self

    def forward(
        self,
        context: dict[str, torch.Tensor],
        global_step: int = 0,
        deterministic: bool = True,
        **kwargs: Any,
    ) -> GeometryProviderOutput:
        with torch.no_grad():
            result = self.encoder(
                context,
                global_step=global_step,
                deterministic=deterministic,
                **kwargs,
            )

        if not isinstance(result, dict) or "geometry_evidence" not in result:
            raise RuntimeError("Geometry encoder did not return geometry_evidence")

        raw = result["geometry_evidence"]
        evidence = GeometryEvidence(
            depth_probabilities=raw["depth_probabilities"],
            depth_candidates=raw["depth_candidates"],
            features=raw["features"],
            semantic_backbone_features=tuple(
                feature.detach()
                for feature in raw["semantic_backbone_features"]
            ),
        )
        return GeometryProviderOutput(
            gaussians=result["gaussians"],
            depths=result["depths"],
            evidence=evidence,
        )
