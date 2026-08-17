# V3 Stage S2A: Real DINO Semantics on Initial Gaussians

## Implementation

The existing ReSplat DINOv2 ViT-B path already returns four dense 768-D
intermediate feature maps. S2A reuses the selected layer and does not load a
second foundation model.

A deterministic orthogonal `1x1` projection compresses 768 dimensions to 16:

```text
context DINO [BV, 768, Hd, Wd]
  -> fixed orthogonal projection
  -> bilinear alignment to Gaussian grid
  -> L2 normalization
  -> semantic_features [B, G, 16]
```

The fixed projection prevents a trivial trainable collapse during this plumbing
stage. It has a stable seed, is checkpoint-independent, and can be made
trainable only in a later controlled experiment.

Target-view teachers use the same frozen DINO layer and fixed projection, but
are computed only for supervision/evaluation. Target features never enter the
initializer or recurrent state.

## Real-model CUDA verification

Official ReSplat weights, eight context views, eight target views, image size
`256x448`, and one recurrent refine iteration were used.

```text
semantic Gaussians:             57,344
semantic dimension:             16
mean per-Gaussian feature norm:  1.0000
rendered tensor:                 [1, 8, 16, 256, 448]
mean rendered alpha:             0.9878
target DINO cosine similarity:   0.973077
rendered feature std:            0.178580
```

The nonzero feature standard deviation rejects the all-constant collapse case.
RGB metrics remained identical to the semantic-field smoke:

```text
final PSNR:   34.606274
final SSIM:   0.965493
final LPIPS:  0.0631811
```

All 55 unit tests pass.

## Outcome and next gate

S2A passes: real DINO semantics can be attached to initial Gaussians, survive
recurrent refinement unchanged, render into novel views, and align strongly
with held-out target-view DINO features without affecting RGB.

S2B should not simply train the shared projection against itself. The next
learned component will be a zero-initialized semantic state residual on each
Gaussian, supervised against the fixed 16-D teacher space with an explicit
anti-collapse diagnostic. Geometry, opacity, and RGB remain frozen in S2B.
