# V3 Stage 9 — Fixed-Candidate Semantic Gaussian Split

## Decision

**Engineering pass. Keep the fixed-candidate split in the semantic-Gaussian
mainline.**

Stage 9 changes Gaussian allocation rather than only correcting existing Gaussian
parameters. It adds a fixed candidate budget around the Gaussians with the
highest semantic correction priority. The implementation is differentiable,
bounded, compatible with one RTX 4090D, and does not require dynamic tensor sizes
or densification bookkeeping.

The current 5-sample, 20/50-step results establish engineering viability and
non-degradation only. They are not final paper claims.

## Method

For `G` parent Gaussians:

1. Rank detached `need × reliability` semantic priority per scene.
2. Select exactly `0.25 G` parents, giving a fixed output count of `1.25 G`.
3. Preserve the parent Gaussian and append one candidate child for each selected
   parent.
4. Inherit the parent's color, rotation, and 16-D semantic embedding.
5. Predict only a bounded source-ray depth residual and three bounded log-scale
   residuals for the child.
6. Give the child at most `0.1 × priority` of the parent's opacity and solve the
   new parent opacity analytically so coincident parent/child accumulated alpha is
   conserved.

At `256x448` with eight context views, the Gaussian count changes from 57,344 to
71,680. This count is fixed for every sample.

## Opacity design correction

The first alpha-conserving formula reduced parent opacity first and solved child
opacity afterwards. Near-opaque parents could then create children with opacity
near one (`0.992` observed), and raster sorting caused a `-0.0118 dB` PSNR shift.

The final formula bounds child opacity first and solves parent opacity afterwards.
With a 10% opacity fraction, the maximum observed child opacity fell to `0.0944`.
This corrected version is the promoted implementation. The failed formula remains
documented as a negative engineering ablation, not part of the method.

## Zero-displacement compatibility check

The Stage-7C checkpoint was loaded with a zero-initialized split head and evaluated
on five fixed samples.

| Metric | Stage 7C | Fixed split, zero displacement | Delta |
|---|---:|---:|---:|
| Semantic cosine | 0.932715034 | 0.932647693 | -0.000067341 |
| Uncertainty MAE ↓ | 0.206861413 | 0.206907296 | +0.000045884 |
| PSNR | 27.845243454 | 27.844886017 | -0.000357437 |
| SSIM | 0.852749324 | 0.852752435 | +0.000003111 |
| LPIPS ↓ | 0.131974377 | 0.131958644 | -0.000015733 |

The small non-zero change comes from raster ordering of coincident parent/child
Gaussians. It is far below the earlier `0.05 dB` warning threshold.

Additional diagnostics:

- candidate ratio: `0.25`;
- output count ratio: `1.25`;
- selected priority mean: `0.13266`;
- child opacity mean: `0.00531`;
- child opacity maximum: `0.09435`;
- relative depth and log-scale residuals: exactly zero before training;
- peak evaluation memory: about 5.31 GiB, only about 1.7 MB above the measured
  Stage-7C run because the frozen encoder dominates memory and decoder rendering
  is chunked.

## Split-only 50-step smoke test

Only the 137 K split-head parameters were trained from the Stage-7C checkpoint at
learning rate `5e-5`.

- mean relative child depth displacement: `1.29e-4`;
- maximum relative child depth displacement: `0.00280`;
- mean absolute child log-scale residual: `1.59e-4`;
- maximum absolute child log-scale residual: `0.00314`.

Relative to the zero-displacement split, PSNR changed by `-0.0000137 dB`, SSIM by
`+0.00000023`, LPIPS by `+0.00000090`, and semantic cosine by `+0.00000011`.
The head receives gradients and learns genuine geometric separation without a
material metric change at this short budget.

## Joint Stage 7–9 20-step smoke test

The semantic heads, Stage-8 adapter, last recurrent point-transformer block, and
split head were jointly trained from the same Stage-7C checkpoint at learning rate
`1e-5` (about 3.1 M trainable parameters).

| Metric | Stage 7C | Joint Stage 7–9 | Delta |
|---|---:|---:|---:|
| Semantic cosine | 0.932715034 | 0.932674074 | -0.000040960 |
| Uncertainty MAE ↓ | 0.206861413 | 0.206761730 | -0.000099683 |
| PSNR | 27.845243454 | 27.845988846 | +0.000745392 |
| SSIM | 0.852749324 | 0.852790272 | +0.000040948 |
| LPIPS ↓ | 0.131974377 | 0.131990047 | +0.000015670 |

The result is mixed but non-destructive. It supports retaining the architecture,
not claiming a stable improvement.

## Verification and storage

- All 99 unit tests passed.
- Hydra composition passed for split-only and joint modes.
- GPU memory stayed near 6.1 GiB during training.
- Short smoke-test checkpoints are deleted after evaluation; logs and all
  per-sample metric JSON files are retained.

## Current full model

```text
semantic-carrying Gaussians
  -> semantic rendering and raster-VJP residual
  -> uncertainty/visibility priority
  -> support and source-ray refinement
  -> semantic-conditioned recurrent updater
  -> fixed-priority child Gaussian candidates
  -> RGB/depth/semantic rendering
```

This is now a coherent first paper architecture. The next scientifically useful
step is a longer joint run and broader paired evaluation. A further optional
module would let parent and child semantic embeddings specialize separately, but
that should not be added before the current split has stronger evidence.
