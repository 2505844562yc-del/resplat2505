"""Small CUDA contract test for semantic feature rasterization."""

from pathlib import Path
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.decoder.gsplat_decoder_splatting_cuda import (
    GSplatDecoderSplattingCUDA,
    GSplatDecoderSplattingCUDACfg,
)
from src.model.types import Gaussians


def make_gaussians(semantic_features: torch.Tensor | None) -> Gaussians:
    device = torch.device("cuda")
    means = torch.tensor([[[-0.2, 0.0, 2.0], [0.2, 0.0, 2.0]]], device=device)
    rotations = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]], device=device
    )
    return Gaussians(
        means=means,
        covariances=None,
        harmonics=torch.zeros(1, 2, 3, 1, device=device),
        opacities=torch.full((1, 2), 0.8, device=device),
        scales=torch.full((1, 2, 3), 0.25, device=device),
        rotations=rotations,
        rotations_unnorm=rotations,
        semantic_features=semantic_features,
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test")

    decoder = GSplatDecoderSplattingCUDA(
        GSplatDecoderSplattingCUDACfg(
            name="gsplat", scale_invariant=False, use_covariances=False
        ),
        SimpleNamespace(background_color=[0.0, 0.0, 0.0]),
    ).cuda()
    features = torch.randn(1, 2, 16, device="cuda", requires_grad=True)
    semantic_gaussians = make_gaussians(features)
    baseline_gaussians = make_gaussians(None)
    extrinsics = torch.eye(4, device="cuda").reshape(1, 1, 4, 4)
    intrinsics = torch.tensor(
        [[[[0.8, 0.0, 0.5], [0.0, 0.8, 0.5], [0.0, 0.0, 1.0]]]],
        device="cuda",
    ).reshape(1, 1, 3, 3)
    near = torch.full((1, 1), 0.1, device="cuda")
    far = torch.full((1, 1), 10.0, device="cuda")

    rendered, alpha = decoder.forward_features(
        semantic_gaussians,
        extrinsics,
        intrinsics,
        near,
        far,
        (32, 32),
    )
    if rendered.shape != (1, 1, 16, 32, 32):
        raise AssertionError(f"unexpected semantic render shape: {rendered.shape}")
    loss = (rendered.square() * alpha[:, :, None]).mean()
    loss.backward()
    if features.grad is None or not torch.isfinite(features.grad).all():
        raise AssertionError("semantic feature gradient is missing or non-finite")
    if features.grad.abs().sum().item() == 0:
        raise AssertionError("semantic feature gradient is zero")

    rgb_with_semantics = decoder.forward(
        semantic_gaussians, extrinsics, intrinsics, near, far, (32, 32)
    ).color
    rgb_without_semantics = decoder.forward(
        baseline_gaussians, extrinsics, intrinsics, near, far, (32, 32)
    ).color
    max_rgb_difference = (
        rgb_with_semantics - rgb_without_semantics
    ).abs().max().item()
    if max_rgb_difference != 0.0:
        raise AssertionError(
            f"semantic field changed RGB rendering: {max_rgb_difference}"
        )

    print(
        "semantic render smoke passed: "
        f"shape={tuple(rendered.shape)}, "
        f"gradient_l1={features.grad.abs().sum().item():.6f}, "
        f"alpha_max={alpha.max().item():.6f}, "
        f"rgb_max_difference={max_rgb_difference:.1f}"
    )


if __name__ == "__main__":
    main()
