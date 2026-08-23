# V3 Stage 4 — Semantic Raster-VJP Geometry Screening

## Objective

Test whether the Stage 3B semantic reconstruction signal can directly improve
Gaussian geometry and therefore RGB reconstruction, while keeping the official
ReSplat model frozen.

This stage is intentionally training-free. Any RGB change is caused only by the
new semantic geometry correction.

## Implemented geometry paths

The alpha-gated context semantic cosine loss is differentiated not only with
respect to the Gaussian semantic attribute `z`, but also with respect to:

- Gaussian center `mu`,
- log Gaussian scale `log(s)`.

The resulting rasterizer VJPs are normalized per scene and converted to bounded
per-Gaussian feedback. Center updates are measured relative to each Gaussian's
local average scale; scale updates are multiplicative in log space. All gains
default to zero, giving exact identity behavior.

Three safeguards were implemented and tested:

1. **Full 3-D bounded center correction.**
2. **Relative-gradient confidence floor.** Only high-magnitude Gaussian VJPs
   survive a smooth threshold.
3. **Source-ray depth-only correction.** The center VJP is projected onto the
   original source-camera ray, preventing tangential image-plane drift.

Full unit suite: 74/74 passed. CUDA smoke tests confirmed gradients propagate
through both the mean and covariance/scale rasterization paths in Lightning
inference mode.

## Same-path five-sample baseline

Semantic direct gain is fixed at the promoted Stage 3B value `0.3`; geometry gains
are zero.

| Semantic cosine | PSNR | SSIM | LPIPS |
| ---: | ---: | ---: | ---: |
| 0.932560015 | 27.844989 | 0.852752054 | 0.131972204 |

## Unrestricted 3-D center VJP

| Mean gain | Semantic cosine | PSNR | SSIM | LPIPS |
| ---: | ---: | ---: | ---: | ---: |
| 0.005 | 0.932579672 | 27.846183 | 0.852753234 | 0.131962574 |
| 0.010 | 0.932598341 | 27.847066 | 0.852751517 | 0.131965224 |
| 0.020 | 0.932636070 | 27.847900 | 0.852739739 | 0.131960543 |

The average PSNR gain is misleading. At gain `0.02`, the first sample improves by
about `+0.0186 dB`, while three other samples decrease. SSIM improves only on the
first sample and decreases on the remaining four. Therefore the apparent mean
gain is not stable.

## Confidence gating

For mean gain `0.02`:

| Confidence floor | Approx. active Gaussians | Semantic cosine | PSNR | SSIM | LPIPS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 26.8% | 0.932619250 | 27.847635 | 0.852747750 | 0.131972611 |
| 0.50 | 11.9% | 0.932608950 | 27.847211 | 0.852749038 | 0.131980756 |

Sparsifying the update does not recover stable perceptual improvements. The
confidence floor reduces displacement but does not fix the direction problem.

## Source-ray depth-only correction

Projecting the center gradient onto the original camera ray is more stable:

| Ray-depth gain | Relative displacement | Semantic cosine | PSNR | SSIM | LPIPS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.02 | about 0.00023 | 0.932563865 | 27.844986 | 0.852754319 | 0.131964265 |
| 0.10 | about 0.00115 | 0.932577956 | 27.844556 | 0.852753663 | 0.131966738 |
| 0.20 | about 0.00230 | 0.932597339 | 27.844044 | 0.852755702 | 0.131964011 |

At gain `0.02`, semantic cosine and LPIPS improve on all five samples, but only by
roughly `1e-6` to `1e-5`; PSNR is mixed. Larger gains slightly amplify semantic
change but cause a systematic mean PSNR decline. These effects are too small and
conflicted to support a reconstruction claim.

Scale-only gain `0.01` was also screened on the smoke sample. It did not produce
a consistent RGB advantage and was not promoted to the five-sample sweep.

## Decision

Do **not** promote direct semantic geometry VJP into the main method. Keep all
paths behind disabled-by-default switches as reproducible negative ablations.

The diagnosis is important: DINO semantic reconstruction gradients reliably
indicate how the semantic attribute should change, but they are not calibrated as
photometric or geometric gradients. Directly moving centers to reduce a DINO
feature loss can preserve semantics while perturbing RGB alignment.

## Recommended next step

Retain the proven Stage 3B semantic-state correction as the main semantic module.
If geometry is revisited, use the Gaussian-aligned VJP only as an observation and
confidence signal for a small ray-depth residual predictor trained under RGB,
ResNet-feature, and geometric/depth consistency losses. The output must be
zero-initialized and ray-constrained. In other words, semantics should identify
*where geometry is uncertain*, while photometric/geometric supervision decides
*which direction geometry should move*.
