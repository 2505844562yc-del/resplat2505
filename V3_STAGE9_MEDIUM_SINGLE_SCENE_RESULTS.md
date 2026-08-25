# Stage 7–9 Medium Single-Scene Joint Validation

## Purpose

This experiment goes beyond the 20-step Stage-9 smoke test without retraining
the complete ReSplat model. It asks whether the complete Stage 7–9 semantic
chain can learn for 500 optimizer steps while preserving held-out RGB quality.

This is **scene-specific adaptation with disjoint held-out views**, not a final
multi-scene generalization experiment and not a train/test-identical overfit.

## Controlled split

- Scene: `dl3dv_970a5c674c27b504d592d0a70c496d0e35ab0dc76802fb6e1bf336a4c1fe150a`
- Available frames: 324
- Training-accessible frames: indices `0–119` only
- Held-out context: `[120, 126, 132, 138, 144, 150, 156, 159]`
- Held-out targets: `[121, 125, 130, 135, 140, 145, 151, 155]`
- Training log inspection confirmed that sampled training indices never exceeded
  119.

The held-out context is also unseen during adaptation. The evaluation therefore
tests whether the adapted network still reconstructs a disjoint segment of the
same scene from new context images.

## Training configuration

- Hardware: one RTX 4090D
- Initialization: Stage-7C 500-step checkpoint
- Schedule: 500 optimizer steps; checkpoints every 100 steps
- Batch size: 1
- Eight context and two target views per training sample
- Resolution: `256 × 448`
- Learning rate: `1e-5`
- Frozen parameters: most of the original ReSplat encoder
- Trainable parameters: 3.1 M of 224 M
- Trainable scope:
  - promoted semantic heads;
  - Stage-8 semantic updater adapter;
  - Stage-9 fixed-candidate split head;
  - only the last original recurrent updater block.
- Runtime: approximately 11 minutes for 500 steps
- Peak observed training memory: approximately 6.1 GiB

Parameter auditing found exactly 2,233,344 changed parameters in the final
recurrent block and 903,322 changed/new parameters in semantic heads. No
unexpected frozen parameter changed.

## Held-out reference and identity check

| Variant | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Semantic cosine ↑ | Uncertainty MAE ↓ |
|---|---:|---:|---:|---:|---:|
| Stage 7C | 29.635273 | 0.956034 | 0.056351 | 0.950728 | 0.202585 |
| Zero-init Stage 8–9 identity | 29.635212 | 0.956033 | 0.056343 | 0.950719 | 0.202579 |

The identity path changes PSNR by only `-0.000061 dB`. The fixed 1.25x Gaussian
allocation and conservative alpha redistribution therefore do not materially
alter the mature Stage-7C result before learning.

## Joint-training trajectory on held-out views

| Step | PSNR ↑ | ΔPSNR | SSIM ↑ | ΔSSIM | LPIPS ↓ | ΔLPIPS | Semantic cosine ↑ | Uncertainty MAE ↓ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 29.607080 | -0.028193 | 0.955832 | -0.000202 | 0.056668 | +0.000317 | 0.950840 | 0.202211 |
| 200 | **29.631794** | **-0.003479** | **0.956035** | **+0.000001** | **0.056653** | **+0.000302** | 0.950845 | 0.201847 |
| 300 | 29.618998 | -0.016275 | 0.955940 | -0.000094 | 0.056702 | +0.000351 | 0.950867 | 0.201691 |
| 400 | 29.619011 | -0.016262 | 0.955944 | -0.000090 | 0.056729 | +0.000378 | 0.950871 | 0.201605 |
| 500 | 29.617056 | -0.018217 | 0.955935 | -0.000099 | 0.056739 | +0.000388 | **0.950873** | **0.201596** |

All deltas are relative to Stage 7C on exactly the same held-out views.

Semantic cosine improves monotonically by up to `+0.000145`. Uncertainty MAE
improves monotonically by up to `-0.000989`, and uncertainty correlation rises
from `0.210749` to `0.214225`. RGB quality remains close to Stage 7C but does not
show a consistent improvement. Step 200 is the best RGB/semantic compromise in
this run.

## Does the split cause the RGB regression?

| Variant | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Semantic cosine ↑ |
|---|---:|---:|---:|---:|
| Joint 200, split disabled | 29.631811 | 0.956036 | 0.056658 | 0.950854 |
| Joint 200, split enabled | 29.631794 | 0.956035 | 0.056653 | 0.950845 |
| Joint 500, split disabled | 29.617008 | 0.955935 | 0.056743 | 0.950882 |
| Joint 500, split enabled | 29.617056 | 0.955935 | 0.056739 | 0.950873 |

Turning the trained split on or off changes PSNR by less than `0.00005 dB`.
Therefore the small RGB drift is not caused by appending child Gaussians. It is
mainly associated with joint adaptation of semantic heads and the last recurrent
updater block.

The split head is active rather than dormant. At step 500 its mean relative child
depth residual is `3.09e-5`, mean absolute log-scale residual is `9.13e-5`, and
the semantic updater adapter mean state residual is `4.13e-6`. These conservative
magnitudes explain why the method remains non-destructive.

## Conclusion

The experiment meets its engineering goal:

1. The complete Stage 7–9 chain trains stably for 500 steps.
2. Only intended parameters change.
3. Semantic prediction and uncertainty calibration improve on disjoint views.
4. RGB reconstruction remains very close to Stage 7C.
5. Fixed-candidate splitting is not responsible for the small RGB drift.

It does **not** establish a paper-level performance gain because it contains only
one adapted scene. The next evidence-building experiment should use three scenes
with the same 500-step controlled protocol, retain paired results, and use the
200-step checkpoint as the current conservative operating point. Architecture
changes are not justified before that comparison.

## Reproduction

```bash
bash scripts/v3_stage9_medium_eval.sh stage7c 500 500
bash scripts/v3_stage9_medium_train.sh 500 100
bash scripts/v3_stage9_medium_eval.sh joint 200 500
bash scripts/v3_stage9_medium_eval.sh joint 500 500
bash scripts/v3_stage9_medium_eval.sh joint_no_split 200 500
bash scripts/v3_stage9_medium_eval.sh joint_no_split 500 500
```

Retained checkpoints:

- `epoch_6-step_200.ckpt`: best held-out RGB/semantic compromise
- `epoch_16-step_500.ckpt`: final training state and semantic trend endpoint

