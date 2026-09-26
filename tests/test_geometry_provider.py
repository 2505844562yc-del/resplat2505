from dataclasses import dataclass

import torch
from torch import nn

from src.model.semantic_multihyp import FrozenGeometryProvider
from src.model.types import Gaussians


@dataclass
class _Cfg:
    num_refine: int = 0
    return_geometry_evidence: bool = True


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cfg = _Cfg()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, context, global_step, deterministic, **kwargs):
        batch = context["image"].shape[0]
        gaussians = Gaussians(
            means=torch.zeros(batch, 1, 3),
            covariances=torch.eye(3).view(1, 1, 3, 3).repeat(batch, 1, 1, 1),
            harmonics=torch.zeros(batch, 1, 3, 1),
            opacities=torch.ones(batch, 1),
        )
        return {
            "gaussians": gaussians,
            "depths": torch.ones(batch, 2, 8, 8),
            "geometry_evidence": {
                "depth_probabilities": torch.full((batch, 2, 4, 2, 2), 0.25),
                "depth_candidates": torch.ones(batch, 2, 4, 2, 2),
                "features": torch.zeros(batch, 2, 8, 2, 2),
                "semantic_backbone_features": tuple(
                    torch.zeros(batch * 2, 8, 1, 1) for _ in range(4)
                ),
            },
        }


def test_geometry_provider_freezes_encoder_and_exposes_evidence() -> None:
    encoder = _Encoder()
    provider = FrozenGeometryProvider(encoder)
    provider.train()
    result = provider({"image": torch.zeros(1, 2, 3, 8, 8)})

    assert not encoder.training
    assert not encoder.weight.requires_grad
    assert result.evidence.depth_probabilities.shape == (1, 2, 4, 2, 2)
    assert len(result.evidence.semantic_backbone_features) == 4
    assert result.gaussians.means.shape == (1, 1, 3)


def test_geometry_provider_rejects_recurrent_encoder() -> None:
    encoder = _Encoder()
    encoder.cfg.num_refine = 1
    try:
        FrozenGeometryProvider(encoder)
    except ValueError as error:
        assert "num_refine" in str(error)
    else:
        raise AssertionError("recurrent encoder should be rejected")
