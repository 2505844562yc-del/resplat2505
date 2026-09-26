import torch

from src.model.semantic_multihyp import BoundaryAwareRayMixtureDecoder, RayMixtureCfg


def _posterior(peaks: list[tuple[int, float]], bins: int = 24) -> torch.Tensor:
    index = torch.arange(bins, dtype=torch.float32)
    logits = torch.full((bins,), -9.0)
    for center, amplitude in peaks:
        logits = torch.logaddexp(
            logits,
            torch.tensor(amplitude).log() - 0.5 * ((index - center) / 0.8) ** 2,
        )
    return logits.softmax(dim=0).view(1, 1, bins, 1, 1)


def _depth_grid(bins: int = 24) -> torch.Tensor:
    return torch.linspace(1.0, 8.0, bins).view(1, 1, bins, 1, 1)


def test_bimodal_boundary_activates_second_hypothesis() -> None:
    decoder = BoundaryAwareRayMixtureDecoder(RayMixtureCfg())
    candidates = _depth_grid()

    single = decoder(
        _posterior([(5, 1.0)]),
        candidates,
        boundary_probability=torch.zeros(1, 1, 1, 1, 1),
        semantic_confidence=torch.ones(1, 1, 1, 1, 1),
    )
    bimodal = decoder(
        _posterior([(5, 1.0), (18, 0.9)]),
        candidates,
        boundary_probability=torch.ones(1, 1, 1, 1, 1),
        semantic_confidence=torch.ones(1, 1, 1, 1, 1),
    )

    assert bimodal.depths.shape == (1, 1, 2, 1, 1)
    assert bimodal.mixture_weights.shape == (1, 1, 2, 1, 1)
    assert torch.allclose(
        bimodal.mixture_weights.sum(dim=2), torch.ones(1, 1, 1, 1), atol=1e-5
    )
    assert bimodal.second_active_probability.item() > single.second_active_probability.item()
    assert bimodal.normalized_separation.item() > single.normalized_separation.item()
    assert bimodal.depths[0, 0, 1].item() > bimodal.depths[0, 0, 0].item()


def test_ray_mixture_is_differentiable() -> None:
    decoder = BoundaryAwareRayMixtureDecoder(RayMixtureCfg())
    logits = torch.randn(1, 2, 24, 3, 4, requires_grad=True)
    probabilities = logits.softmax(dim=2)
    candidates = torch.linspace(0.5, 12.0, 24).view(1, 1, 24, 1, 1)
    candidates = candidates.expand_as(probabilities).clone().requires_grad_(True)
    boundary = torch.rand(1, 2, 1, 6, 8)
    confidence = torch.rand(1, 2, 1, 6, 8)

    output = decoder(probabilities, candidates, boundary, confidence)
    loss = output.depths.mean() + output.mixture_weights[:, :, 1].mean()
    loss.backward()

    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert candidates.grad is not None and torch.isfinite(candidates.grad).all()
    assert decoder.gate_residual[-1].weight.grad is not None
