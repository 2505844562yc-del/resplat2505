# Situation A Stage-6 boundary-calibrated short-screen report

## Scope

Stage 6 strengthens Situation A without changing the ReSplat depth backbone or
Gaussian initializer. The same 16-dimensional projected DINO semantic field is
used to predict a bounded residual on the depth-candidate logits before initial
Gaussians are constructed. The new supervision emphasizes semantic feature
errors at trusted SAM2 boundaries:

`weight = 1 + lambda_boundary * boundary * confidence`

Only pixels above the configured confidence floor receive boundary emphasis.
The weighted loss is normalized by the sum of valid weights, so enabling the
module does not merely multiply the global loss scale. The original Stage-5
configuration keeps `semantic_depth_boundary_weight: 0.0` and is unchanged.

## Calibrated settings

- Boundary weight: 4.0
- Semantic feature loss weight: 0.2
- Maximum depth-logit residual: 0.5
- Gate bias: -1.0
- Learning rate: 1e-4
- Trainable parameters: about 116 K
- Frozen parameters: about 224 M
- Training scene: `032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7`
- Training length: 50 steps
- Evaluation: one fixed held-out 8-context/8-target sample from the same scene

## Held-out results

| Variant | PSNR | SSIM | LPIPS | semantic cosine | init PSNR | init SSIM | init LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Identity | 34.609989 | 0.9655000 | 0.06316041 | 0.97425538 | 32.993423 | 0.95571911 | 0.08447147 |
| Stage-5, 50 steps | 34.608189 | 0.9655065 | 0.06308039 | 0.97425783 | 32.993408 | 0.95571506 | 0.08457163 |
| Stage-6 calibrated, 50 steps | 34.610844 | 0.9654894 | 0.06310438 | 0.97425318 | 32.992771 | 0.95568073 | 0.08454452 |

Stage 6 changes PSNR by +0.00085 dB and LPIPS by -0.000056 versus identity,
while SSIM changes by -0.000011. Compared with Stage 5, Stage 6 gains
+0.00266 dB PSNR but gives back 0.000017 SSIM and 0.000024 LPIPS. These
single-sample differences are too small to claim a quality improvement.

## Situation A diagnostics

| Variant | gate mean | gate max | logit residual L1 | candidate relative depth delta |
| --- | ---: | ---: | ---: | ---: |
| Identity | 0.079190 | 0.114791 | 0 | 0 |
| Stage-5, 50 steps | 0.078226 | 0.112659 | 4.4241e-5 | 9.6248e-7 |
| Stage-6 calibrated, 50 steps | 0.177425 | 0.259318 | 3.7009e-4 | 2.7041e-5 |

The calibrated Stage-6 adapter produces about 28.1 times the candidate-depth
change of the matched Stage-5 run. This confirms that trusted boundary
supervision reaches the pre-Gaussian depth distribution rather than acting only
as an auxiliary semantic output loss.

Checkpoint auditing compared 829 shared base tensors against the V3 checkpoint.
None changed in either run. Stage 6 changed only the 16 adapter tensors; the
maximum shared-base parameter delta was exactly zero.

## Decision

Keep Stage 6 as the current Situation-A implementation. It is logically aligned
with semantic geometry initialization, preserves the pretrained baseline, and
has a measurable effect on pre-Gaussian depth while remaining non-destructive.
The present 50-step, one-sample result is a screening result, not evidence of a
stable benchmark gain. Before a large training run, the next useful test is a
small multi-scene calibration screen that compares Stage 5 and Stage 6 and
reports both image metrics and boundary-region metrics.

