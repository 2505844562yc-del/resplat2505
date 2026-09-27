import torch

from src.model.semantic_multihyp.supervision import (
    SemanticSupervision,
    SemanticSupervisionCfg,
)
from src.model.semantic_multihyp.types import (
    CandidateVerification,
    RayHypotheses,
    SemanticMultiHypOutput,
    SemanticPredictions,
)


def _output() -> SemanticMultiHypOutput:
    b, v, h, w = 1, 2, 8, 10
    low_h, low_w = 4, 5
    semantics = SemanticPredictions(
        class_logits=torch.randn(b, v, 4, h, w, requires_grad=True),
        instance_embeddings=torch.nn.functional.normalize(
            torch.randn(b, v, 6, h, w, requires_grad=True), dim=2
        ),
        boundary_logits=torch.randn(b, v, 1, h, w, requires_grad=True),
        confidence_logits=torch.randn(b, v, 1, h, w, requires_grad=True),
        decoder_features=torch.randn(b, v, 8, 2, 3),
    )
    depths = torch.empty(b, v, 2, low_h, low_w).uniform_(1.0, 2.0)
    depths.requires_grad_()
    mixture = torch.softmax(
        torch.randn(b, v, 2, low_h, low_w, requires_grad=True), dim=2
    )
    second = torch.sigmoid(torch.randn(b, v, 1, low_h, low_w, requires_grad=True))
    hypotheses = RayHypotheses(
        depths=depths,
        mixture_weights=mixture,
        second_active_probability=second,
        posterior_entropy=torch.zeros(b, v, 1, low_h, low_w),
        normalized_separation=torch.zeros(b, v, 1, low_h, low_w),
        mode_masses=torch.ones(b, v, 2, low_h, low_w),
    )
    keep = torch.sigmoid(torch.randn(b, v, 2, low_h, low_w, requires_grad=True))
    verification = CandidateVerification(
        keep_probability=keep,
        opacity_delta=torch.zeros_like(keep),
        geometric_support=torch.zeros_like(keep),
        semantic_support=torch.zeros_like(keep),
        visibility_support=torch.zeros_like(keep),
        valid_view_count=torch.ones_like(keep),
        view_attention=torch.ones(b, v, v, 2, low_h, low_w),
    )
    return SemanticMultiHypOutput(
        assembly=None,
        semantics=semantics,
        hypotheses=hypotheses,
        verification=verification,
        base_depths=torch.ones(b, v, h, w),
        geometry_evidence=None,
    )


def test_all_semantic_supervision_terms_are_finite_and_differentiable() -> None:
    output = _output()
    semantic = torch.zeros(1, 2, 8, 10, dtype=torch.long)
    semantic[..., :, 5:] = 1
    instance = torch.ones(1, 2, 8, 10, dtype=torch.long)
    instance[..., :, 5:] = 2
    depth = torch.full((1, 2, 8, 10), 1.25)
    module = SemanticSupervision(SemanticSupervisionCfg())
    losses = module(
        output,
        {"semantic": semantic, "instance": instance, "depth": depth},
    )
    assert set(losses) == {
        "class",
        "confidence",
        "boundary",
        "instance",
        "second_gate",
        "hypothesis",
        "verification",
    }
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    assert output.semantics.class_logits.grad is not None
    assert output.hypotheses.depths.grad is not None


def test_rgb_only_context_skips_semantic_supervision() -> None:
    module = SemanticSupervision(SemanticSupervisionCfg())
    assert module(_output(), {}) == {}
