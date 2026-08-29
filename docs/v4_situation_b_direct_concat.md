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
not a paper-level gain.  Multi-scene medium/full training is still required to
compare A, B1, and B2 scientifically.

## Commands

```bash
bash scripts/v4_situation_b_eval.sh baseline
bash scripts/v4_situation_b_eval.sh identity
bash scripts/v4_situation_b_train.sh 50
bash scripts/v4_situation_b_eval.sh trained 50
```

Enable B2 with `V4_UNFREEZE_DEPTH_TAIL=true` and normally initialize it from a
stable B1 checkpoint through `V4_START_CKPT`.
