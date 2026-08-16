# Stage 9 — Semantic Parameter Routing

## Motivation

Stage 9 tests whether the validated Stage 4 semantic boundary residual should
directly route recurrent Gaussian parameter updates. The Gaussian count and the
base ReSplat updater remain unchanged.

Two bounded residual routes were implemented:

- geometry route: mean, scale, rotation, and opacity deltas;
- appearance route: spherical-harmonic (SH) deltas.

For a learned route value `r`, each selected update is multiplied by
`1 + 0.5 * tanh(r)`. The final layer is zero-initialized, so enabling the module
starts as an exact identity mapping.

## Correctness checks

- The route head is named as an updater module and is trainable under ReSplat's
  existing parameter-freezing policy.
- Zero route output is an exact identity.
- Geometry and appearance can be controlled independently.
- Geometry-only mode keeps SH updates bitwise unchanged.
- All 32 repository unit tests pass.

## Controlled results

All values use the same canonical overfit scene and deterministic five-scene
evaluation used for the earlier short-run gates.

| Variant | Steps | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Boundary L1 ↓ | Boundary F1 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Stage 4 reference | 20 | 26.808416 | 0.831748 | 0.147262 | 0.376912 | 0.140578 |
| geometry + appearance | 20 | 26.808886 | 0.831776 | 0.147260 | 0.376906 | 0.140703 |
| geometry + appearance | 50 | 26.957420 | 0.835727 | 0.144159 | 0.376538 | 0.142123 |
| Stage 4 reference | 50 | 26.955709 | 0.835787 | 0.144131 | 0.376539 | 0.142268 |
| geometry only | 20 | 26.806135 | 0.831705 | 0.147275 | 0.376902 | 0.140634 |

At 20 steps, the two-route version improved all five metrics by very small
amounts. At 50 steps, only PSNR and boundary L1 remained better; SSIM, LPIPS,
and boundary F1 regressed. Removing the appearance route did not remove the
trade-off: geometry-only routing improved the two boundary metrics slightly at
20 steps, but reduced all three image-quality metrics.

## Decision

Stage 9 is retained as a clean ablation but is **not promoted** into the main
method. The evidence does not support the claim that multiplicative parameter
routing improves reconstruction reliably. No 50-step geometry-only run is
performed because it failed the predeclared 20-step gate.

The main method remains the Stage 4 local-alignment semantic residual feedback.
The next architectural experiment should avoid uniformly scaling heterogeneous
Gaussian deltas. A better-supported direction is to alter how boundary evidence
is encoded or spatially assigned before the recurrent updater, while keeping a
strict identity path and the same short-run promotion gates.
