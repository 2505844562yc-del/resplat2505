# Stage 10 — Spatially Selective Semantic Geometry Refinement

## Motivation and design

This final Version-1 experiment tests whether the validated Stage 4 local
boundary-alignment signal should produce a separate Gaussian-geometry residual
only near trusted semantic boundaries.

The original ReSplat update path is preserved. A zero-initialized residual head
uses the recurrent state and Stage 4 boundary feature to predict independent
per-axis corrections for Gaussian means and scales. Rotation, opacity, and SH
appearance are unchanged. Corrections are bounded relative to the detached
magnitude of the corresponding base update. A learned gate is multiplied by a
hard spatial support mask: only tokens within two pixels of a confident SAM2
target boundary can receive the added residual.

This differs from Stage 9 parameter routing: it does not uniformly multiply an
existing group of Gaussian deltas, and it cannot modify non-boundary tokens.

## Correctness checks

- Zero correction is an exact identity mapping.
- Inactive tokens are exactly unchanged.
- Mean/scale and their three axes can be corrected independently.
- Rotation, opacity, and SH never enter this branch.
- The new updater head is trainable under ReSplat's freezing policy.
- All 36 repository unit tests pass.
- Real-data forward, backward, and checkpointing pass.
- The corrected boundary-neighborhood mask activates 13.95% of training tokens
  and 13.83% of tokens in the fixed evaluation.

An initial screening run exposed a mask bug: with confidence floor zero, a
product-based `>= 0` condition selected the whole image. That run is excluded.
The condition was corrected to require both a positive target boundary and
sufficient confidence before the controlled screening below.

## 20-step screening

The same canonical training scene and deterministic five-scene evaluation are
used as in prior gates.

| Variant | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Boundary L1 ↓ | Boundary F1 ↑ |
|---|---:|---:|---:|---:|---:|
| Stage 4 local alignment | 26.808416 | 0.831748 | 0.147262 | 0.376912 | 0.140578 |
| Stage 10 selective geometry | 26.806606 | 0.831713 | 0.147237 | 0.376932 | 0.140576 |
| Delta | -0.001810 | -0.000035 | -0.000025 | +0.000020 | -0.000002 |

LPIPS improves slightly, but PSNR, SSIM, and Boundary L1 regress, while
Boundary F1 is effectively unchanged. The module therefore fails the
predeclared 20-step promotion gate.

## Decision

Do not run 50 or 200 steps and do not tune radius or gain. Retain the module as
a controlled negative-result ablation, disabled by default.

Version 1 is now frozen at Stage 4:

```text
confidence-gated semantic boundary loss
+ local boundary alignment feedback [signed residual, dx, dy]
+ original fixed-count ReSplat recurrent Gaussian updater
```

Stages 6–10 demonstrate that additional multi-view correspondence, scalar
consensus, multiplicative parameter routing, and selective output residuals do
not produce stable gains under the current small-compute protocol. Full
Version-1 experiments should use Stage 4 as the main method and these later
modules only as ablations or documented negative explorations.
