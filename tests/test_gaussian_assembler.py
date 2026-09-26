import torch

from src.model.semantic_multihyp import HypothesisGaussianAssembler, RayHypotheses
from src.model.semantic_multihyp.types import CandidateVerification
from src.model.types import Gaussians


def _inputs(h: int = 2, w: int = 3):
    b, v, k = 1, 2, 2
    n = v * h * w
    parent = Gaussians(
        means=torch.zeros(b, n, 3),
        covariances=torch.eye(3).view(1, 1, 3, 3).repeat(b, n, 1, 1),
        harmonics=torch.zeros(b, n, 3, 4),
        opacities=torch.full((b, n), 0.8),
        scales=torch.ones(b, n, 3),
        rotations=torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 4).repeat(b, n, 1),
        rotations_unnorm=torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 4).repeat(b, n, 1),
    )
    depths = torch.stack(
        [torch.full((b, v, h, w), 2.0), torch.full((b, v, h, w), 4.0)],
        dim=2,
    )
    weights = torch.empty_like(depths)
    weights[:, :, 0] = 0.75
    weights[:, :, 1] = 0.25
    hypotheses = RayHypotheses(
        depths=depths,
        mixture_weights=weights,
        second_active_probability=weights[:, :, 1:2],
        posterior_entropy=torch.zeros(b, v, 1, h, w),
        normalized_separation=torch.ones(b, v, 1, h, w),
        mode_masses=weights,
    )
    keep = torch.ones_like(depths, requires_grad=True)
    verification = CandidateVerification(
        keep_probability=keep,
        opacity_delta=torch.zeros_like(depths),
        geometric_support=torch.ones_like(depths),
        semantic_support=torch.ones_like(depths),
        visibility_support=torch.ones_like(depths),
        valid_view_count=torch.ones_like(depths),
        view_attention=torch.full((b, v, v, k, h, w), 0.5),
    )
    intrinsics = torch.eye(3).view(1, 1, 3, 3).repeat(b, v, 1, 1)
    intrinsics[:, :, 0, 0] = 1.2
    intrinsics[:, :, 1, 1] = 1.2
    intrinsics[:, :, 0, 2] = 0.5
    intrinsics[:, :, 1, 2] = 0.5
    extrinsics = torch.eye(4).view(1, 1, 4, 4).repeat(b, v, 1, 1)
    return parent, hypotheses, verification, intrinsics, extrinsics


def test_assembler_creates_two_renderer_ready_slots() -> None:
    parent, hypotheses, verification, intrinsics, extrinsics = _inputs()
    assembler = HypothesisGaussianAssembler()
    output = assembler(parent, hypotheses, verification, intrinsics, extrinsics)

    assert output.gaussians.means.shape == (1, 24, 3)
    assert output.gaussians.covariances.shape == (1, 24, 3, 3)
    assert output.gaussians.harmonics.shape == (1, 24, 3, 4)
    assert output.gaussians.opacities.shape == (1, 24)
    means = output.gaussians.means.reshape(1, 2, 2, 2, 3, 3)
    assert torch.allclose(means[:, :, 0, ..., 2], torch.full((1, 2, 2, 3), 2.0))
    assert torch.allclose(means[:, :, 1, ..., 2], torch.full((1, 2, 2, 3), 4.0))

    scales = output.gaussians.scales.reshape(1, 2, 2, 2, 3, 3)
    assert torch.allclose(scales[:, :, 0], torch.ones_like(scales[:, :, 0]))
    assert torch.allclose(scales[:, :, 1], torch.full_like(scales[:, :, 1], 2.0))
    slots = output.slot_index.reshape(1, 2, 2, 2, 3)
    assert torch.all(slots[:, :, 0] == 0)
    assert torch.all(slots[:, :, 1] == 1)


def test_assembler_keeps_training_path_differentiable() -> None:
    parent, hypotheses, verification, intrinsics, extrinsics = _inputs()
    hypotheses.depths.requires_grad_(True)
    output = HypothesisGaussianAssembler()(
        parent, hypotheses, verification, intrinsics, extrinsics
    )
    loss = output.gaussians.means.mean() + output.gaussians.opacities.mean()
    loss.backward()

    assert hypotheses.depths.grad is not None
    assert torch.isfinite(hypotheses.depths.grad).all()
    assert verification.keep_probability.grad is not None
