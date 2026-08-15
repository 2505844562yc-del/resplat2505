# Stage 3: Cross-scene validation and directional feedback

## Protocol

- Dataset: first five deterministic DL3DV test chunks, 8 context and 8 target views.
- Resolution: 256 x 448.
- Boundary source: SAM2.1 Hiera Small sidecars with confidence gating.
- Fixed Gaussian count and one recurrent refinement iteration.
- Stage 2 hyperparameters: boundary weight 0.01, feedback scale 0.25.

## Stage 2 cross-scene generalization (200 steps)

| Model | PSNR | SSIM | LPIPS | Boundary L1 | Boundary F1 |
|---|---:|---:|---:|---:|---:|
| Baseline | 27.222664 | 0.839730 | 0.141236 | 0.376504 | 0.141015 |
| Semantic residual | 27.222328 | 0.839703 | 0.141380 | 0.376123 | 0.142092 |

The semantic residual improves boundary F1 by about 0.764% and lowers boundary L1 by about 0.101%, while PSNR changes by only -0.00034 dB. The Stage 2 effect therefore survives a small cross-scene test and is not only a single-scene artifact.

## Directional residual prototype

The residual_gradient mode concatenates the confidence-gated signed residual with normalized Sobel x/y derivatives. It changes only the semantic projection input width; the Gaussian count, ReSplat backbone, recurrent state, and updated Gaussian attributes are unchanged.

### 20-step fair comparison

| Feature mode | PSNR | SSIM | LPIPS | Boundary L1 | Boundary F1 |
|---|---:|---:|---:|---:|---:|
| residual | 26.804490 | 0.831770 | 0.147176 | 0.377003 | 0.140453 |
| residual_gradient | 26.806017 | 0.831738 | 0.147200 | 0.376976 | 0.140565 |

### 50-step fair comparison

| Feature mode | PSNR | SSIM | LPIPS | Boundary L1 | Boundary F1 |
|---|---:|---:|---:|---:|---:|
| residual | 26.949325 | 0.835579 | 0.144352 | 0.376597 | 0.142044 |
| residual_gradient | 26.955647 | 0.835662 | 0.144275 | 0.376608 | 0.141757 |

At 50 steps the directional mode slightly improves all RGB metrics, but boundary F1 drops by about 0.202% and boundary L1 is effectively unchanged. This is not sufficient evidence to replace the single residual channel in the main method.

## Decision

- Keep residual as the main Stage 3/default representation.
- Keep residual_gradient as an ablation implementation, not a claimed contribution.
- Do not spend a 200-step run on this variant yet.
- The next higher-value experiment should address boundary alignment directly (for example, a local distance-transform residual) rather than adding more raw derivative channels.

## Reproduction

The overfit runner accepts feature mode as its fifth argument:

    bash scripts/stage2_overfit_ablation.sh joint 50 0.01 0.25 residual
    bash scripts/stage2_overfit_ablation.sh joint 50 0.01 0.25 residual_gradient
