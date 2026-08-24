# V3 Stage 4: Gaussian Semantic Uncertainty and Visibility State

## Objective

Turn the Stage-3 raster-VJP residual into a selective Gaussian state before any
semantic-conditioned opacity, scale, or position update is attempted. The stage
must distinguish whether a Gaussian needs correction from whether its observation
is reliable.

## Method

For every Gaussian, the implementation computes four bounded scalars:

- `need`: smooth-thresholded semantic raster-VJP magnitude;
- `reliable`: normalized VJP of accumulated render alpha with respect to that
  Gaussian's opacity, measuring its actual visibility/contribution support;
- `uncertainty = need * (2 - reliable)`: semantic mismatch amplified by poor
  reliability, without marking every invisible Gaussian as uncertain;
- `priority = need * reliable`: the safe region for later structural updates.

A 68.5K-parameter MLP reads the frozen ReSplat recurrent token, Gaussian-aligned
semantic feedback, and deterministic uncertainty signals. It predicts a bounded
calibration residual for uncertainty. Its final layer is zero initialized, and
only this MLP is trainable.

The calibrated value is stored in `Gaussians.semantic_uncertainty`. It can be
rendered into novel views through the existing N-D feature rasterizer. Target
DINO features are used only for training supervision: normalized target-view
semantic cosine error provides a detached uncertainty target. RGB, opacity,
scale, position, SH, DINO, and the original ReSplat updater remain frozen.

## Verification

- Python and shell syntax checks passed.
- Full unit suite: 81/81 passed (four new uncertainty tests).
- Two-step CUDA forward/backward smoke passed on RTX 4090D.
- Observed training memory remained about 6.0 GiB.
- Only 68.5K of about 224M parameters were trainable.
- The official-checkpoint zero-head control exactly reproduced the Stage-3B RGB
  and semantic metrics.

## Selectivity

The previous ray-depth prototype treated about 99.9% of nonzero semantic VJPs as
active at confidence floor zero. Across the fixed five-sample Stage-4 evaluation:

| Diagnostic | Mean |
|---|---:|
| Gaussian need | 0.099204 |
| Gaussian reliability | 0.145151 |
| Gaussian uncertainty (zero head) | 0.144060 |
| Gaussian priority | 0.033164 |
| Priority active fraction | **0.219064** |

The new gate therefore selects about 21.9% of Gaussians instead of acting as an
almost-all-on mask.

## Five-sample calibration result

| Variant | Uncertainty MAE (lower) | Correlation (higher) | Rendered mean | Rendered std |
|---|---:|---:|---:|---:|
| Official checkpoint, zero head | 0.229522 | 0.239202 | 0.331775 | 0.233839 |
| 20-step uncertainty head | **0.228664** | **0.239802** | 0.330146 | 0.233256 |
| Difference | **-0.000858** | **+0.000600** | -0.001629 | -0.000582 |

Both variants have identical reconstruction and semantic-state metrics:

| Semantic cosine | PSNR | SSIM | LPIPS |
|---:|---:|---:|---:|
| 0.932560015 | 27.844989395 | 0.852752054 | 0.131972204 |

## Decision

Stage 4 passes under the rebased development policy. It has a physically grounded
visibility signal, produces a selective rather than all-on gate, receives
trainable calibration gradients, slightly improves held-out uncertainty
calibration after a short run, and leaves the ReSplat/Stage-3B reconstruction
exactly unchanged.

Promote `semantic_uncertainty`, `need`, `reliable`, and `priority` as inputs to
Stage 5. The next module is a zero-initialized support head that predicts separate
bounded opacity-logit and log-scale residuals. Semantics controls where updates
are allowed; RGB and perceptual losses determine their direction.

