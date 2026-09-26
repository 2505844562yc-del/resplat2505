import torch

from src.model.semantic_multihyp import (
    CrossViewHypothesisVerifier,
    RayHypotheses,
    SemanticPredictions,
)


def _hypotheses(depth: float, h: int, w: int) -> RayHypotheses:
    depths = torch.full((1, 2, 1, h, w), depth)
    ones = torch.ones(1, 2, 1, h, w)
    zeros = torch.zeros_like(ones)
    return RayHypotheses(
        depths=depths,
        mixture_weights=ones,
        second_active_probability=zeros,
        posterior_entropy=zeros,
        normalized_separation=zeros,
        mode_masses=ones,
    )


def _scene(h: int = 4, w: int = 6, bins: int = 32):
    candidate_line = torch.linspace(1.0, 6.0, bins)
    candidates = candidate_line.view(1, 1, bins, 1, 1).expand(1, 2, bins, h, w)
    probabilities = torch.exp(-0.5 * ((candidates - 2.0) / 0.08) ** 2)
    probabilities = probabilities / probabilities.sum(dim=2, keepdim=True)

    class_logits = torch.full((1, 2, 3, h, w), -8.0)
    class_logits[:, :, 0] = 8.0
    instances = torch.zeros(1, 2, 4, h, w)
    instances[:, :, 0] = 1.0
    semantics = SemanticPredictions(
        class_logits=class_logits,
        instance_embeddings=instances,
        boundary_logits=torch.zeros(1, 2, 1, h, w),
        confidence_logits=torch.full((1, 2, 1, h, w), 8.0),
        decoder_features=torch.zeros(1, 2, 8, h, w),
    )

    intrinsics = torch.eye(3).view(1, 1, 3, 3).repeat(1, 2, 1, 1)
    intrinsics[:, :, 0, 0] = 1.2
    intrinsics[:, :, 1, 1] = 1.2
    intrinsics[:, :, 0, 2] = 0.5
    intrinsics[:, :, 1, 2] = 0.5
    extrinsics = torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1)
    return probabilities, candidates, semantics, intrinsics, extrinsics


def test_consistent_candidate_receives_more_support() -> None:
    h, w = 4, 6
    probabilities, candidates, semantics, intrinsics, extrinsics = _scene(h, w)
    verifier = CrossViewHypothesisVerifier()

    consistent = verifier(
        _hypotheses(2.0, h, w),
        probabilities,
        candidates,
        semantics,
        intrinsics,
        extrinsics,
    )
    inconsistent = verifier(
        _hypotheses(4.5, h, w),
        probabilities,
        candidates,
        semantics,
        intrinsics,
        extrinsics,
    )

    assert consistent.keep_probability.shape == (1, 2, 1, h, w)
    assert consistent.view_attention.shape == (1, 2, 2, 1, h, w)
    assert consistent.geometric_support.mean() > inconsistent.geometric_support.mean()
    assert consistent.keep_probability.mean() > inconsistent.keep_probability.mean()
    assert torch.all(consistent.valid_view_count == 1)


def test_cross_view_verifier_backpropagates_to_hypothesis_depth() -> None:
    h, w = 3, 5
    probabilities, candidates, semantics, intrinsics, extrinsics = _scene(h, w)
    hypotheses = _hypotheses(2.1, h, w)
    hypotheses.depths.requires_grad_(True)
    verifier = CrossViewHypothesisVerifier()

    output = verifier(
        hypotheses,
        probabilities,
        candidates,
        semantics,
        intrinsics,
        extrinsics,
    )
    output.keep_probability.mean().backward()

    assert hypotheses.depths.grad is not None
    assert torch.isfinite(hypotheses.depths.grad).all()
    assert verifier.keep_residual[-1].weight.grad is not None
