# V2 Stage 4: controlled ablation and step-0 evaluation

## Goal

Make the four relevant methods directly comparable and distinguish a better
initial Gaussian set from improvements produced only by recurrent refinement.

## Four controlled modes

| Mode | Semantic initialization | V1 boundary loss | V1 boundary feedback |
|---|---:|---:|---:|
| `baseline` | no | no | no |
| `v1` | no | yes | yes |
| `init` | yes | no | no |
| `init_v1` | yes | yes | yes |

All modes keep the original ReSplat initializer and recurrent updater topology,
the same pretrained checkpoint, scene, views, optimizer schedule, and one refine
iteration. The `init` modes add only the V2 semantic conditioning and bounded
depth/scale residuals.

## New evaluation contract

When `num_refine > 0` and score computation is enabled, evaluation now renders
the Gaussian set twice with the same target cameras:

1. immediately after initial Gaussian construction;
2. after the recurrent refinement iteration.

The score file therefore contains `init_psnr`, `init_ssim`, `init_lpips`,
`init_boundary_l1`, and `init_boundary_f1`, alongside the existing final
metrics. It also reports `refine_psnr_gain`, `refine_ssim_gain`, and
`refine_lpips_reduction`.

The extra initial render is excluded from the decoder benchmark so published
runtime measurements still describe the actual inference path.

## Reproducible commands

```bash
bash scripts/v2_short_ablation.sh init_v1 20
bash scripts/v2_eval_ablation.sh init_v1 20 5
```

Replace `init_v1` with `baseline`, `v1`, or `init` for the controlled ablation.

## Smoke verification

The implementation passed syntax checks and all 46 unit tests. A two-step
`init_v1` train plus one-scene evaluation completed successfully. The diagnostic
reported distinct initial and final RGB and boundary scores. These two-step
values validate the pipeline only and are not evidence for or against the
method.

## First 20-step diagnostic

The first controlled run produced nearly identical results. Semantic init
changed initial PSNR by about `-0.0013 dB` and initial Boundary F1 by only about
`+0.00014`; the adapter residual was nonzero (`~0.0024`), so this was not a
broken-gradient failure. The final `init_v1` Boundary F1 was the best of the four
modes, but only by roughly `0.001`, which is too small to claim a useful trend.

This diagnosis exposed a supervision-path problem: the initial geometry was
optimized only through the recurrent updater and final render. The updater can
absorb or attenuate small initializer changes. The one permitted diagnostic
revision therefore adds a weighted, direct loss on the initial render
(`semantic_init_auxiliary_loss_weight=0.25`). It reuses the selected RGB and
boundary losses, while target boundaries remain supervision only and never
enter the initializer input. A second 20-step comparison is required before any
50-step promotion.

## Revised 20-step result

Five fixed evaluation samples were used. Higher is better for PSNR/SSIM/F1;
lower is better for LPIPS/L1.

| mode | init PSNR | init LPIPS | init boundary F1 | final PSNR | final boundary F1 |
|---|---:|---:|---:|---:|---:|
| baseline | 26.68553 | 0.151628 | 0.139394 | 26.80463 | 0.139705 |
| V1 | 26.68553 | 0.151628 | 0.139394 | 26.80455 | 0.140479 |
| init + direct supervision | 26.68487 | 0.151584 | 0.139468 | 26.80010 | 0.139742 |
| init + V1 + direct supervision | 26.68434 | 0.151623 | 0.139516 | 26.80554 | 0.140777 |

The direct loss increased the learned adapter residual from roughly `0.0024`
to `0.0035`, but did not create a meaningful step-0 improvement. The best full
model improves final Boundary F1 over V1 by only `0.00030`, while initial PSNR
is slightly worse. This fails the predefined 20-step promotion criterion.

Therefore this exact adapter/head design is **not promoted to 50 steps**. The
code and commits remain reproducible, but the next development step must inspect
actual depth/scale correction magnitude and spatial localization before deciding
whether to revise the parameterization or stop this V2 branch. Running longer
without that evidence would spend GPU time without answering the failure cause.
