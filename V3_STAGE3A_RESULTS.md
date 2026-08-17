# V3 Stage 3A — Source-Grid Semantic Residual Feedback

## Goal

Close the semantic refinement loop for the first time: render the semantic
attributes carried by the current Gaussians, compare them with frozen DINO
features, and expose the mismatch to the recurrent semantic-state head.

Geometry, opacity, colour/SH, ReSplat, and DINO remain frozen. Only the
zero-initialized semantic-state head is trained.

## Feedback representation

At the recurrent grid resolution, the module constructs 18 feedback channels:

- 16-D normalized directional residual (`teacher - rendered`),
- 1-D cosine error,
- 1-D accumulated-alpha visibility confidence.

Pixels below alpha `0.1` are suppressed. The 18-D residual is concatenated with
the existing 256-D ReSplat recurrent token before predicting the bounded 16-D
Gaussian semantic update.

## Verification

- Full unit suite: 61/61 passed.
- Official ReSplat checkpoint loaded successfully.
- Only 280K parameters were trainable; about 223M parameters remained frozen.
- Peak observed GPU allocation was about 6.0 GB on an RTX 4090D.
- The zero-initialized head received a non-zero update after two steps.
- One-sample two-step semantic cosine changed from approximately `0.973077` to
  `0.973123`.
- RGB metrics remained exactly unchanged.

## Fair five-sample result after 20 steps

| Variant | Mean semantic cosine | Difference from Stage 2A |
| --- | ---: | ---: |
| Stage 2A baseline | 0.927725983 | — |
| Stage 2B, no explicit residual | 0.927616787 | -0.000109196 |
| Stage 3A, source-grid residual | 0.927749014 | +0.000023031 |

Per-sample Stage 3A minus Stage 2A differences are approximately:

`[-0.00000244, +0.00029492, +0.00005001, -0.00001574, -0.00021160]`

Only two of five samples improve. The mean gain is dominated by one sample, so
Stage 3A does not pass the consistency gate and should not simply be extended to
50 training steps.

## Diagnosis

The residual tensor is indexed by rendered context-view pixels, while recurrent
tokens are indexed by the source-view pixels that originally produced each
Gaussian. Those arrays have the same length in the current DL3DV configuration,
but they do not express the same correspondence. A rendered pixel blends many
Gaussians, potentially originating in different views. Assigning that pixel's
residual to the source-grid Gaussian at the same flat index is therefore only an
approximation and produces inconsistent corrections.

## Next experiment

Use the differentiable splatting renderer's vector-Jacobian product to
back-project the alpha-gated context semantic reconstruction loss to each
Gaussian semantic attribute. The negative per-Gaussian gradient gives a
visibility- and contribution-aware correction direction. Its normalized
direction and relative magnitude can replace the incorrectly aligned source-grid
residual while preserving the same lightweight semantic head and frozen RGB/
geometry backbone.
