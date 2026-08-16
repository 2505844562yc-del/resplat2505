# Stage 8: Multi-View Boundary Displacement Candidates

## Method

For each rendered source-boundary pixel, rendered depth back-projects it to 3D.
The point is projected into every other context view, matched to the nearest
trusted SAM2 boundary within a four-pixel window, back-projected with target
depth, and projected back to the source view. This produces source-frame
displacement `(dx, dy)`, depth-aware match confidence, and visibility.

Opposing displacement proposals reduce directional confidence, preventing a
zero average caused by disagreement from being interpreted as a confident
zero correction. Stage 4 local alignment remains as a fallback branch. The
seven-channel structure input is `[residual, local_dx, local_dy, mv_dx, mv_dy,
confidence, visibility]` and enters through a zero-initialized gated residual
branch.

## Verification

- 27/27 tests pass, including direction, magnitude, depth rejection,
  single-view behavior, and opposing-view disagreement.
- Real diagnostic: 31.66% active rendered-boundary pixels, 36.89% supported,
  mean displacement 0.4662 pixels, coherent confidence 0.0356, visibility
  0.2481.
- Peak training memory is about 9.8 GB on RTX 4090D.

## Screening

| Variant | Steps | PSNR | SSIM | LPIPS | Boundary L1 | Boundary F1 |
|---|---:|---:|---:|---:|---:|---:|
| Stage 4 alignment | 20 | 26.808416 | 0.831748 | 0.147262 | 0.376912 | 0.140578 |
| Stage 8 displacement | 20 | 26.808581 | 0.831692 | 0.147316 | 0.376858 | 0.140675 |
| Stage 4 alignment | 50 | 26.955709 | 0.835787 | 0.144131 | 0.376539 | 0.142268 |
| Stage 8 displacement | 50 | 26.954998 | 0.835789 | 0.144147 | 0.376532 | 0.142380 |

Boundary F1 improves at both horizons while Boundary L1 does not regress and
appearance metrics remain effectively neutral. This is a weak but consistent
positive trend, so the method is promoted to the existing three-training-scene
200-step validation. It is not yet a validated contribution.

## 200-step multi-training-scene result

The coherent displacement model was trained independently on the same three
Stage 5 scenes and compared with the matched Stage 4 alignment checkpoints on
the same five test scenes.

Across all 15 train-model/test-scene pairs:

| Metric | Mean delta (Stage 8 - Stage 4) | Stage 8 wins |
|---|---:|---:|
| PSNR | -0.002526 | 10/15 |
| SSIM | -0.000100 | 11/15 |
| LPIPS | +0.000121 | 6/15 |
| Boundary L1 | +0.000036 | 5/15 |
| Boundary F1 | -0.000333 | 0/15 |

Boundary F1 decreases in every comparison, including all five test scenes for
each of the three independently trained models. The short-run improvement does
not survive longer training and the method must not advance to parameter-head
separation.

## Stage 8C semantic-source focus

Diagnostics showed that unfiltered rendered RGB edges activated 31.7% of the
canonical image and 66.9% in one added scene. Many were texture or illumination
edges rather than semantic structure. Stage 8C therefore keeps rendered edges
only within four pixels of a trusted same-view SAM2 boundary before cross-view
matching.

This reduced canonical activation from 31.7% to 7.6% and raised the supported
fraction from 36.9% to 95.4%, confirming that the filter behaves as intended.

| Variant | Steps | PSNR | SSIM | LPIPS | Boundary L1 | Boundary F1 |
|---|---:|---:|---:|---:|---:|---:|
| Stage 4 alignment | 20 | 26.808416 | 0.831748 | 0.147262 | 0.376912 | 0.140578 |
| Stage 8C focused | 20 | 26.804784 | 0.831632 | 0.147288 | 0.376914 | 0.140672 |
| Stage 4 alignment | 50 | 26.955709 | 0.835787 | 0.144131 | 0.376539 | 0.142268 |
| Stage 8C focused | 50 | 26.957026 | 0.835786 | 0.144107 | 0.376539 | 0.142144 |

The 20-step Boundary F1 gain reverses at 50 steps. Cleaner source selection is
not sufficient to make the displacement branch useful.

## Final decision

Stage 8 establishes a tested geometry implementation and a useful negative
result, but not an improving model. Do not run more radius/gate sweeps and do
not start the proposed Stage 9 geometry/appearance parameter heads on top of
this signal. The retained publication path remains Stage 4 semantic boundary
loss plus local alignment feedback, which already passed the three-scene
validation. Any next innovation should change the supervision representation
or Gaussian update target, rather than further processing this displacement.
