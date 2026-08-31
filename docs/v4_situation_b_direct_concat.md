# V4 Situation B: Direct Semantic Feature Concatenation

## Strict definition

Situation B follows the naming in the design screenshot:

```text
F_depth' = Concat(F_c, F_s)
```

The concatenated feature is consumed before ReSplat's depth U-Net.  It is not
Situation A's post-U-Net residual injection, and the strict B experiment keeps
Situation A plus both legacy semantic-depth adapters disabled.

`F_c` is ReSplat's original input to the depth regressor: cost volume, CNN
features, multi-view transformer features, and its original DINO monocular
features.  `F_s` is the shared 16-D semantic field projected from a frozen DINO
teacher layer.  The same field is later sampled onto the initialized Gaussians
as their semantic attribute `z`.

## Checkpoint-compatible implementation

For a convolution, direct concatenation is exactly equivalent to

```text
Conv(Concat(F_c, F_s)) = Conv_c(F_c) + Conv_s(F_s).
```

The implementation therefore leaves ReSplat's pretrained 3x3 input stem and
1x1 residual/skip convolution unchanged, and adds only the two `Conv_s`
semantic slices.  Both new weights are initialized to zero and have no bias.
This has three practical advantages:

1. Existing ReSplat checkpoints load without changing any old tensor shape.
2. The initial RGB/depth output is exactly the pretrained baseline.
3. B1 can optimize only the new semantic slices without weight decay changing
   the old channel slices of a physically enlarged convolution tensor.

This split implementation is mathematically direct concatenation, not a
post-U-Net residual adapter.

## Training stages

### B1: new semantic channels only

Trainable parameters:

- `encoder.depth_predictor.semantic_depth_concat_projections.*`
- `encoder.semantic_feature_projector.*`

The complete pretrained ReSplat network remains frozen.  In the current
single-scale configuration this is 32,768 trainable parameters out of roughly
224 million.

### B2: conservative depth-tail tuning

After B1 is stable, optionally unfreeze only:

- the depth U-Net output projection (`regressor.*.3.out`);
- the enclosing regressor final convolution (`regressor.*.4`);
- the original depth head.

The DINO backbone, cost-volume path, multi-view transformer, earlier U-Net
layers, Gaussian updater, and renderer remain frozen.  A one-step audit found
918K trainable parameters, all 10 allowed depth-tail tensors changed, and zero
protected pretrained tensors changed.

## Verification on scene 032dee9...

The baseline and zero-initialized B identity runs used the same checkpoint and
held-out view index.  All reported reconstruction metrics were exactly equal:

| metric | baseline | B at initialization |
| --- | ---: | ---: |
| PSNR | 34.6062736511 | 34.6062736511 |
| SSIM | 0.9654933214 | 0.9654933214 |
| LPIPS | 0.0631811097 | 0.0631811097 |
| initial PSNR | 32.9934234619 | 32.9934234619 |
| initial SSIM | 0.9557191133 | 0.9557191133 |
| initial LPIPS | 0.0844714716 | 0.0844714716 |

All B semantic contributions and candidate-depth deltas were exactly zero.

After a 50-step B1 screen on the same scene:

| metric | baseline | 50-step B1 | delta |
| --- | ---: | ---: | ---: |
| PSNR | 34.6062736511 | 34.6011466980 | -0.0051269531 |
| SSIM | 0.9654933214 | 0.9654670954 | -0.0000262260 |
| LPIPS | 0.0631811097 | 0.0630870163 | -0.0000940934 |
| initial PSNR | 32.9934234619 | 33.0043792725 | +0.0109558105 |
| initial SSIM | 0.9557191133 | 0.9557232857 | +0.0000041723 |
| initial LPIPS | 0.0844714716 | 0.0848211348 | +0.0003496632 |

Additional diagnostics:

- semantic input-slice absolute mean: `5.58466360e-4`;
- semantic input-slice absolute maximum: `2.26345286e-3`;
- main-stem relative contribution: `3.65890388e-4`;
- skip relative contribution: `1.35984726e-3`;
- candidate-depth relative delta: `1.96860085e-4`;
- protected pretrained tensors changed: `0 / 812`.

This short screen establishes engineering validity and baseline preservation,
not a paper-level gain.

## 1,000-step multi-scene B1 validation

The medium run started again from the frozen ReSplat checkpoint rather than
the single-scene checkpoint. It sampled all 359 DL3DV training scenes and
trained only the 32,768 B1 parameters. The local boundary cache contains only
held-out test scenes, so training used the semantic feature losses without
boundary weighting. Boundary metrics were computed only on the eight fixed
held-out scenes for which the cache is complete.

The zero-initialized B identity run exactly reproduced every baseline RGB and
boundary metric on all eight scenes. After 1,000 steps:

| metric | baseline | B1 at 1,000 steps | delta |
| --- | ---: | ---: | ---: |
| PSNR | 28.90474868 | 28.90975142 | +0.00500274 |
| SSIM | 0.88268103 | 0.88277297 | +0.00009194 |
| LPIPS | 0.11558364 | 0.11558764 | +0.00000400 |
| initial PSNR | 27.62597966 | 27.62666607 | +0.00068641 |
| initial SSIM | 0.86239034 | 0.86242678 | +0.00003644 |
| initial LPIPS | 0.13440175 | 0.13447138 | +0.00006963 |
| boundary L1 | 0.36431200 | 0.36431973 | +0.00000773 |
| boundary F1 | 0.18359735 | 0.18373548 | +0.00013813 |
| initial boundary L1 | 0.36533679 | 0.36531610 | -0.00002069 |
| initial boundary F1 | 0.17899286 | 0.17906148 | +0.00006862 |

Per-scene consistency was mixed but non-destructive: PSNR improved on 5/8
scenes, SSIM on 6/8, LPIPS on 3/8, and boundary F1 on 4/8. The largest PSNR
gain was +0.02894 dB and the largest drop was -0.00616 dB.

The checkpoint audit compared 812 protected pretrained tensors and found zero
changes. The learned semantic slices had an absolute mean of 0.001468 and a
maximum of 0.007117. The measured relative contributions were 0.001007 for the
main stem, 0.002206 for the skip, and 0.000368 for candidate depth. Runtime was
459 seconds and peak steady-state memory was about 15.2 GiB on one RTX 4090 D.

This is evidence that strict Situation B is trainable and preserves the
baseline under multi-scene sampling. The gain is deliberately small and is not
yet a paper-level result; longer B1 training and a matched B2 comparison remain
the next scientific experiments.

## Commands

```bash
bash scripts/v4_situation_b_eval.sh baseline
bash scripts/v4_situation_b_eval.sh identity
bash scripts/v4_situation_b_train.sh 50
bash scripts/v4_situation_b_eval.sh trained 50

# Fixed eight-scene, multi-scene protocol.
bash scripts/v4_situation_b_multiscene_eval.sh baseline 1000
bash scripts/v4_situation_b_multiscene_eval.sh identity 1000
bash scripts/v4_situation_b_multiscene_train.sh 1000
bash scripts/v4_situation_b_multiscene_eval.sh trained 1000
```

Enable B2 with `V4_UNFREEZE_DEPTH_TAIL=true` and normally initialize it from a
stable B1 checkpoint through `V4_START_CKPT`.
