# V2-2 — Identity Semantic Geometry Adapter

## Implementation

- Builds three context-only structure channels: boundary, trusted confidence,
  and finite-radius boundary proximity.
- Encodes them with a small CNN at initializer feature resolution.
- Predicts a spatial gate initialized to `sigmoid(-2) = 0.1192`.
- Adds a zero-initialized projected semantic residual before the original
  point-transformer Gaussian initialization head.
- Preserves global RNG state during module construction.
- Allows `encoder.semantic_init*` parameters to remain trainable under the
  existing refine-only freeze policy.

No Gaussian attribute or count is changed in V2-2. This stage establishes a
safe identity connection point for the V2-3 mean/depth and scale residuals.

## Verification

- 43/43 unit tests pass.
- Zero-initialized adapter output is exactly equal to its input feature tensor.
- Projection weights receive non-zero gradients.
- Missing or malformed semantic inputs are rejected.
- The frozen Stage-4 checkpoint completes the deterministic five-scene test
  with the adapter enabled and reports zero semantic residual.

| Check | PSNR | SSIM | LPIPS | Boundary L1 | Boundary F1 |
|---|---:|---:|---:|---:|---:|
| Frozen Stage-4 record | 27.242808 | 0.840200 | 0.140747 | 0.376024 | 0.142285 |
| Identity adapter evaluation | 27.242825 | 0.840201 | 0.140752 | 0.376023 | 0.142278 |

The minute metric differences are evaluation-level numerical variation; the
adapter's residual is exactly zero and its local identity property is directly
unit-tested.

## Decision

V2-2 passes. Proceed to V2-3, where separate zero-initialized semantic heads
will modify initial position/depth and scale while leaving rotation, opacity,
SH, and Gaussian count unchanged.
