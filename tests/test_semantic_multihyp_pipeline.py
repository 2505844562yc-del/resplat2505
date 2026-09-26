import torch
from torch import nn

from src.model.semantic_multihyp import (
    BoundaryAwareRayMixtureDecoder,
    CrossViewHypothesisVerifier,
    GaussianAssemblerCfg,
    HypothesisGaussianAssembler,
    SemanticMultiHypothesisInitializer,
    SemanticSceneEncoder,
    SemanticSceneEncoderCfg,
)
from src.model.semantic_multihyp.types import GeometryEvidence, GeometryProviderOutput
from src.model.types import Gaussians


class _GeometryProvider(nn.Module):
    def __init__(self, views: int, latent_h: int, latent_w: int, channels: int) -> None:
        super().__init__()
        self.views = views
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.channels = channels

    def forward(self, context, global_step=0, deterministic=False):
        b, v = context["image"].shape[:2]
        assert v == self.views
        d, hd, wd = 24, self.latent_h // 2, self.latent_w // 2
        line = torch.linspace(1.0, 6.0, d, device=context["image"].device)
        candidates = line.view(1, 1, d, 1, 1).expand(b, v, d, hd, wd)
        probabilities = (
            torch.exp(-0.5 * ((candidates - 2.0) / 0.15) ** 2)
            + 0.7 * torch.exp(-0.5 * ((candidates - 4.5) / 0.18) ** 2)
        )
        probabilities = probabilities / probabilities.sum(dim=2, keepdim=True)
        n = v * self.latent_h * self.latent_w
        gaussians = Gaussians(
            means=torch.zeros(b, n, 3, device=context["image"].device),
            covariances=torch.eye(3, device=context["image"].device).view(1, 1, 3, 3).repeat(b, n, 1, 1),
            harmonics=torch.zeros(b, n, 3, 4, device=context["image"].device),
            opacities=torch.full((b, n), 0.7, device=context["image"].device),
            scales=torch.ones(b, n, 3, device=context["image"].device),
            rotations=torch.tensor([1.0, 0.0, 0.0, 0.0], device=context["image"].device).view(1, 1, 4).repeat(b, n, 1),
            rotations_unnorm=torch.tensor([1.0, 0.0, 0.0, 0.0], device=context["image"].device).view(1, 1, 4).repeat(b, n, 1),
        )
        backbone = tuple(
            torch.rand(b * v, self.channels, 3, 5, device=context["image"].device)
            for _ in range(4)
        )
        return GeometryProviderOutput(
            gaussians=gaussians,
            depths=torch.full((b, v, 32, 48), 2.0, device=context["image"].device),
            evidence=GeometryEvidence(
                depth_probabilities=probabilities,
                depth_candidates=candidates,
                features=torch.zeros(b, v, 32, self.latent_h, self.latent_w, device=context["image"].device),
                semantic_backbone_features=backbone,
            ),
        )


def test_complete_initialization_pipeline_is_differentiable() -> None:
    b, v, h, w = 1, 2, 32, 48
    latent_h, latent_w, channels = 8, 12, 32
    context = {
        "image": torch.rand(b, v, 3, h, w),
        "intrinsics": torch.eye(3).view(1, 1, 3, 3).repeat(b, v, 1, 1),
        "extrinsics": torch.eye(4).view(1, 1, 4, 4).repeat(b, v, 1, 1),
    }
    context["intrinsics"][:, :, 0, 0] = 1.2
    context["intrinsics"][:, :, 1, 1] = 1.2
    context["intrinsics"][:, :, 0, 2] = 0.5
    context["intrinsics"][:, :, 1, 2] = 0.5

    semantic_encoder = SemanticSceneEncoder(
        SemanticSceneEncoderCfg(
            backbone_channels=channels,
            decoder_channels=24,
            edge_channels=16,
            num_classes=5,
            instance_dim=8,
        )
    )
    model = SemanticMultiHypothesisInitializer(
        geometry_provider=_GeometryProvider(v, latent_h, latent_w, channels),
        semantic_encoder=semantic_encoder,
        ray_decoder=BoundaryAwareRayMixtureDecoder(),
        verifier=CrossViewHypothesisVerifier(),
        assembler=HypothesisGaussianAssembler(
            GaussianAssemblerCfg(primary_keep_floor=0.5)
        ),
    )
    output = model(context)

    assert output.assembly.gaussians.means.shape == (b, v * 2 * latent_h * latent_w, 3)
    assert output.hypotheses.depths.shape == (b, v, 2, latent_h, latent_w)
    assert output.semantics.class_logits.shape == (b, v, 5, h, w)
    assert output.verification.keep_probability.shape == (b, v, 2, latent_h, latent_w)

    loss = (
        output.assembly.gaussians.opacities.mean()
        + output.hypotheses.depths.mean()
        + output.semantics.class_logits.mean()
    )
    loss.backward()
    assert semantic_encoder.class_head.weight.grad is not None
    assert semantic_encoder.boundary_head.weight.grad is not None
    assert model.ray_decoder.gate_residual[-1].weight.grad is not None
    assert model.verifier.keep_residual[-1].weight.grad is not None
