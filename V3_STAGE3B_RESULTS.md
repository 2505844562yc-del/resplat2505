# V3 Stage 3B — Raster-VJP Gaussian Semantic Correction

## Motivation

Stage 3A rendered a semantic residual image but flattened that image onto the
source-grid Gaussian tokens. The tensor lengths matched, yet the correspondence
was not physically correct: a rendered pixel blends contributions from many
Gaussians and cannot be assigned to one source-view Gaussian by array index.

Stage 3B uses the differentiable Gaussian rasterizer itself to perform the
pixel-to-Gaussian aggregation.

## Method

For the context views, let the current Gaussian semantic attributes be `z`, the
alpha-normalized rendered feature map be `R(z)`, and the frozen source DINO map be
`T`. The module computes the alpha-gated cosine reconstruction loss

`L_ctx = weighted_mean(1 - cosine(R(z), T))`.

It then evaluates the vector-Jacobian product

`g_z = d L_ctx / d z`.

Because the derivative passes through splatting and alpha compositing, each
Gaussian receives the sum of the residual contributions from exactly the pixels
and views to which it was visible. The negative gradient is therefore a
Gaussian-aligned semantic correction direction rather than a source-grid
approximation.

The feedback representation contains:

- normalized negative VJP direction (16-D),
- per-Gaussian gradient magnitude relative to the scene mean (1-D),
- active/non-zero contribution indicator (1-D).

The relative magnitude is clipped to keep the correction bounded. Semantic
embeddings are renormalized after every update.

## Engineering properties

- Reuses DINO features already attached to the initial source Gaussians; no
  second DINO extraction is needed.
- Geometry, opacity, RGB/SH, ReSplat, and DINO remain unchanged.
- Works in both training and Lightning inference mode.
- Full unit suite: 64/64 passed.
- Observed GPU memory remained about 6.0 GB on an RTX 4090D.
- For the smoke scene, context semantic loss was about `0.0171`; 99.89% of
  Gaussians received non-zero VJP support.
- RGB PSNR, SSIM, and LPIPS are exactly unchanged, as intended for this isolated
  semantic-state experiment.

## Learned-head diagnostic

Feeding the VJP representation only to the zero-initialized semantic MLP and
training it for 20 steps produced mean semantic cosine `0.927749312`, essentially
the same as the Stage 3A source-grid result `0.927749014`. This shows that the
short-run learned head did not yet make meaningful use of the better alignment.

The VJP direction itself was then tested as a deterministic bounded correction,
with no training and no new checkpoint.

## Five-sample fixed-gain screen

Stage 2A baseline mean semantic cosine: `0.927725983`.

| Direct VJP gain | Mean semantic cosine | Gain over baseline |
| ---: | ---: | ---: |
| 0.01 | 0.928041863 | +0.000315881 |
| 0.05 | 0.929205358 | +0.001479375 |
| 0.10 | 0.930428052 | +0.002702069 |
| 0.20 | 0.932062030 | +0.004336047 |
| 0.30 | **0.932560015** | **+0.004834032** |
| 0.50 | 0.930014551 | +0.002288568 |

The decline at `0.50` demonstrates over-correction. The search was stopped after
the predeclared intermediate point `0.30` to avoid excessive tuning on five
samples.

## Twenty-sample confirmation

The same implementation path and official ReSplat checkpoint were evaluated
with direct gain `0.0` and `0.3`.

| Variant | Semantic cosine | Feature std | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Same-path baseline, gain 0.0 | 0.930260423 | 0.181367994 | 29.323123 | 0.896757 | 0.114895 |
| Raster-VJP correction, gain 0.3 | **0.935403997** | 0.180755904 | 29.323123 | 0.896757 | 0.114895 |
| Difference | **+0.005143574** | -0.000612091 | 0 | 0 | 0 |

All 20 of 20 samples improved. Per-sample semantic-cosine improvements ranged
from `+0.001194` to `+0.009795`. The feature standard deviation changed only
slightly and remains non-zero, so the result is not caused by feature collapse.

## Decision

Promote raster-VJP direct semantic correction with candidate gain `0.3` as the
current semantic feedback core. Keep the configuration default at gain `0.0` so
legacy experiments remain unchanged; the dedicated evaluation script defaults
to the promoted candidate.

The learned residual head remains an ablation, not a claimed contributor. Its
next use should be as a small confidence/gain predictor around the proven VJP
direction, rather than an unconstrained replacement for that direction.

## Scope and next step

This stage proves that semantic-carrying Gaussians can be corrected recurrently
and improve novel-view semantic alignment. It does **not** yet improve RGB
reconstruction because semantic updates are intentionally isolated from geometry
and appearance.

The next research step is to let the now-reliable Gaussian-aligned signal affect
reconstruction conservatively. The safest design is a bounded, confidence-gated
geometry residual driven by the VJP magnitude/direction, with RGB geometry
updates kept as the base path and exact identity initialization. This must be
screened against the original RGB/depth metrics before promotion.
