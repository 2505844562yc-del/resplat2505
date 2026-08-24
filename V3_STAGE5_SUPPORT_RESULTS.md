# V3 Stage 5: Semantic-Conditioned Opacity and Scale Support Refinement

## Objective

Use the promoted Stage-4 Gaussian state to make the first conservative semantic-
conditioned structural update. Semantics selects which Gaussians may change;
photometric and perceptual reconstruction losses choose the update direction.

## Method

A zero-initialized 137K-parameter support head reads:

- frozen ReSplat recurrent state;
- Gaussian-aligned Stage-3 raster-VJP feedback;
- Stage-4 `need`, `reliable`, calibrated `uncertainty`, and `priority`.

It predicts four values per Gaussian:

```text
delta_opacity_logit: 1-D
delta_log_scale:     3-D
```

The applied residuals are

```text
opacity_delta   = 0.10 * tanh(raw_opacity) * priority
log_scale_delta = 0.05 * tanh(raw_scale)   * priority
```

Scale is updated multiplicatively and opacity is updated in logit space. A small
residual regularizer is included. The original ReSplat updater, Stage-3 semantic
correction, Stage-4 uncertainty head, DINO, RGB/SH, position, and rotation remain
frozen. Training loads the Stage-4 20-step checkpoint and optimizes only the new
support head using the existing RGB MSE and LPIPS losses.

## Verification

- Python and shell syntax checks passed.
- Full unit suite: 84/84 passed (three new bounded/identity/gating tests).
- Two-step CUDA forward/backward smoke passed on RTX 4090D.
- Only 137K of about 224M parameters were trainable.
- Observed memory remained about 6.0 GiB.
- Gain-zero evaluation on the trained checkpoint exactly reproduced Stage 4.
- After two steps both opacity and scale residuals became nonzero, confirming the
  RGB/perceptual gradient reaches the head through Gaussian rasterization.

## Twenty-step five-sample ablation

| Variant | Semantic cosine | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| Gain-zero control | 0.932560015 | 27.844989395 | 0.852752054 | 0.131972204 |
| Opacity only | 0.932560468 | 27.845205307 | 0.852750826 | 0.131973982 |
| Scale only | 0.932562113 | 27.845273209 | 0.852748203 | 0.131976506 |
| Opacity + scale | **0.932562518** | **27.845475769** | 0.852746856 | 0.131978250 |

Joint versus gain-zero:

- semantic cosine: `+0.000002503`;
- PSNR: `+0.000486374` dB;
- SSIM: `-0.000005198`;
- LPIPS: `+0.000006045`.

The opacity-only and scale-only PSNR gains are approximately additive, showing
that neither output is completely inactive.

## Fifty-step stability screen

| Variant | Semantic cosine | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| Gain-zero control | 0.932560015 | 27.844989395 | 0.852752054 | 0.131972204 |
| Opacity + scale | **0.932563388** | **27.845875931** | 0.852741206 | 0.131982391 |
| Difference | +0.000003374 | **+0.000886536** | -0.000010848 | +0.000010186 |

PSNR improves on all 5/5 samples. SSIM improves on 1/5 and LPIPS improves on 2/5;
their regressions remain around `1e-5`, far below the rebased warning thresholds.

Mean applied residuals at 50 steps are small:

- absolute opacity-logit delta: `0.0003125`;
- absolute log-scale delta: `0.0000670`;
- active Gaussian fraction: `0.2191`.

## Decision

Stage 5 passes under the rebased policy. It is identity initialized, selectively
gated, bounded, independently ablatable, trainable through RGB rendering, and
does not materially damage the ReSplat baseline. Both opacity and scale outputs
remain in the development mainline despite the small short-run effect.

Stage 6 should reuse the existing source-ray depth head but replace its old
almost-all-on semantic magnitude gate with the promoted Stage-4 `priority`. The
Stage-5 support checkpoint becomes the initialization for that experiment.

