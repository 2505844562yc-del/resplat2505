# Situation A Stage-6 three-scene calibration screen

## Protocol

This screen compares the frozen identity checkpoint, the Stage-5 semantic
depth-logit adapter, and the Stage-6 trusted-boundary-calibrated adapter on three
DL3DV scenes. Each trained variant starts from the same V3 checkpoint and trains
only the approximately 116 K semantic depth-adapter parameters for 50 steps.
Within each scene, Stage 5 and Stage 6 see the same deterministic training view
sequence. Evaluation uses one fixed 8-context/8-target held-out sample per scene.

This is a calibration screen with three samples, not a publishable benchmark.

## Per-scene held-out results

| Scene | Variant | PSNR | SSIM | LPIPS | semantic cosine | candidate relative depth delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `032dee` | Identity | 34.609989 | 0.9655000 | 0.06316041 | 0.97425538 | 0 |
| `032dee` | Stage 5 | 34.608189 | 0.9655065 | 0.06308039 | 0.97425783 | 9.6248e-7 |
| `032dee` | Stage 6 | 34.610844 | 0.9654894 | 0.06310438 | 0.97425318 | 2.7041e-5 |
| `0569e8` | Identity | 26.194141 | 0.8560210 | 0.10458811 | 0.93418968 | 0 |
| `0569e8` | Stage 5 | 26.193459 | 0.8560271 | 0.10458352 | 0.93419176 | 4.1349e-6 |
| `0569e8` | Stage 6 | 26.196714 | 0.8561695 | 0.10461389 | 0.93424451 | 1.9917e-4 |
| `06da79` | Identity | 28.674873 | 0.9048898 | 0.10973900 | 0.94092304 | 0 |
| `06da79` | Stage 5 | 28.677143 | 0.9048869 | 0.10979136 | 0.94092852 | 7.6671e-7 |
| `06da79` | Stage 6 | 28.681103 | 0.9049310 | 0.10974789 | 0.94090176 | 3.8118e-5 |

## Three-scene means

| Variant | PSNR | SSIM | LPIPS | semantic cosine | candidate relative depth delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Identity | 29.826335 | 0.9088036 | 0.09249558 | 0.94978937 | 0 |
| Stage 5 | 29.826263 | 0.9088068 | 0.09248509 | 0.94979270 | 1.9547e-6 |
| Stage 6 | 29.829554 | 0.9088633 | 0.09248872 | 0.94979982 | 8.8109e-5 |

Stage 6 minus identity:

- PSNR: +0.003219 dB
- SSIM: +0.0000597
- LPIPS: -0.0000071 (lower is better)
- Semantic cosine: +0.0000105

Stage 6 minus Stage 5:

- PSNR: +0.003290 dB
- SSIM: +0.0000565
- LPIPS: +0.0000036
- Semantic cosine: +0.0000071

Stage 6 increases the candidate-depth change by 28.1x, 48.2x, and 49.7x over
the matched Stage-5 runs. PSNR is higher than both identity and Stage 5 in all
three scenes. The other metrics are mixed per scene, although the three-scene
mean remains essentially baseline-preserving.

## Safety audit and storage

All four newly trained extra-scene checkpoints passed the parameter audit:
829 shared base tensors were compared and zero changed. Only the 16 allowed
adapter tensors changed. After evaluation and auditing, the reproducible
extra-scene checkpoints were removed to recover disk space; their logs and all
metric JSON files remain. The primary `032dee` Stage-5 and Stage-6 checkpoints
remain available.

The Stage-6 runs report absolute RGB-derived boundary metrics, but identity and
Stage-5 were originally evaluated without loading sidecars, so this screen does
not claim a comparative boundary-F1 gain. A later formal evaluation must load
identical sidecars for every variant.

## Decision

Promote Stage 6 as the Situation-A default and stop tuning its strength on the
same three samples. The result is appropriately conservative: it makes semantic
information measurably alter the depth distribution before Gaussian creation,
keeps the pretrained reconstruction intact, and produces a consistent PSNR
direction without a meaningful average loss in SSIM or LPIPS.

The next development step should not be another scalar-weight sweep. It should
either begin a medium multi-scene Stage-6 training run for a stronger validity
check, or proceed to Situation B, where semantic evidence changes the internal
depth feature representation rather than only adding a residual to final depth
logits. Situation B is the larger architectural innovation and will require a
separate branch to avoid contaminating the stable Situation-A implementation.

