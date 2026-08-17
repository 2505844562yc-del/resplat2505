# V3 Stage 2B — Isolated Recurrent Semantic-State Refinement

## Objective

Test the smallest possible recurrent update of the semantic attribute carried by
each Gaussian. Geometry, opacity, colour/SH coefficients, the pretrained ReSplat
model, and the DINOv2 teacher stay frozen. Only a newly added semantic-state head
is trainable.

This stage answers one narrow question: can the existing ReSplat recurrent state
predict a useful per-Gaussian semantic residual without changing RGB rendering?

## Implementation

- Each Gaussian already carries a normalized 16-D DINO semantic embedding from
  Stage 2A.
- A small MLP reads the existing recurrent token and predicts `delta_z`.
- The last layer is zero-initialized, so enabling the module initially preserves
  the Stage 2A result exactly.
- The update is bounded and renormalized:

  `z_next = normalize(z + 0.1 * tanh(delta_z))`

- Target-view semantic features are rendered with alpha normalization.
- A frozen DINOv2 teacher supplies target-view features.
- The only added supervision is alpha-masked cosine loss with weight `0.1`.
- Training code explicitly freezes every parameter except
  `encoder.semantic_state_head` (about 270K trainable parameters out of about
  223M total parameters).

## Verification

- Full unit-test suite: 58/58 passed.
- Valid two-step smoke test loaded the official ReSplat checkpoint and produced a
  non-zero semantic-head update.
- RGB metrics are exactly unchanged, confirming the isolated semantic update does
  not accidentally modify geometry or appearance.
- The rendered semantic feature standard deviation remains healthy, so there is
  no feature collapse.

## Fair five-sample comparison

Both rows use the same DL3DV evaluation indices, 8 context views, 8 target views,
image size 256x448, one recurrent refinement, and the same official ReSplat
checkpoint.

| Variant | Semantic cosine (higher is better) | Feature std | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage 2A baseline (no semantic-state update) | 0.927725983 | 0.186314359 | 27.844989 | 0.852752 | 0.131972 |
| Stage 2B, 20-step single-scene overfit | 0.927616787 | 0.186798877 | 27.844989 | 0.852752 | 0.131972 |
| Difference | -0.000109196 | +0.000484517 | 0 | 0 | 0 |

## Decision

Stage 2B does **not** pass the promotion gate. The isolated `delta_z` head is
technically valid and stable, but the 20-step result does not improve semantic
alignment. A 50-step run is therefore not justified at this point.

The likely structural reason is that the head only sees the original ReSplat
recurrent state, which was designed around RGB/geometry residuals. It is optimized
by a semantic loss after rendering, but it does not directly observe the rendered
semantic mismatch. In other words, supervision exists, but the updater lacks an
explicit semantic-residual input.

The code and scripts are retained as a reproducible negative ablation. The large
temporary checkpoints may be deleted after recording these results.

## Next stage

Build a closed semantic feedback loop:

1. Render the current Gaussian semantic map in each target view.
2. Compare it with the frozen target-view DINO teacher map.
3. Construct a compact, confidence/alpha-gated semantic residual.
4. Back-project or aggregate that residual to Gaussian/recurrent tokens.
5. Predict a bounded `delta_z` from both the existing recurrent state and the
   explicit semantic residual.
6. Keep geometry and RGB frozen for the first screening run, so any semantic gain
   can be attributed to the new feedback path.

Only after this closed-loop variant shows a stable semantic-cosine gain should
semantic information be allowed to influence Gaussian geometry or appearance.
