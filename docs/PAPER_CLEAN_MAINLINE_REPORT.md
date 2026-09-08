# Clean Semantic ReSplat Mainline Report

## Scope

This branch replaces the accumulated V3 semantic refinement heads with one
paper-facing path.  Historical V3 modules remain in the repository for
ablation, but the main experiment configuration keeps them disabled.

Branch: `paper-clean-semantic-refinement`

Main configuration: `paper_clean_semantic_refinement_dl3dv`

## Method

1. **Semantic-conditioned initialization (Situation B).**  A frozen DINO
   feature is projected to a shared 16-D semantic space.  The projected feature
   is concatenated with the original depth feature before the depth U-Net.  In
   implementation, only the mathematically equivalent semantic slices of the
   original 3x3 stem and 1x1 skip convolution are added, so all pretrained
   ReSplat tensors stay checkpoint-compatible.
2. **Semantic-carrying Gaussians.**  The same 16-D feature is attached to the
   resulting Gaussian as its semantic attribute `z`.  The paper mainline keeps
   `z` fixed during recurrent refinement.
3. **Gaussian-aligned semantic residual refinement.**  Context-view semantic
   feature maps are rendered from `z` and compared with context-view DINO
   features.  Raster-VJP aggregates the image-space semantic residual back to
   exactly the Gaussians that produced it.  One zero-initialized MLP maps
   `[z, VJP direction, VJP magnitude, VJP activity]` to an additive residual in
   the original recurrent hidden state.  ReSplat's original update head then
   updates position, covariance, opacity, and color.

Target-view DINO features are used only for the training loss and evaluation.
They are never updater input at training or inference time.

## Trainable parameters

The pretrained ReSplat/DINO backbone and the shared semantic projector are
frozen.  Only the following new parameters are trained:

| Component | Parameters |
|---|---:|
| Situation-B semantic convolution slices | 20,480 |
| Clean semantic residual adapter | 140,544 |
| **Total** | **161,024** |

The full instantiated model has about 224M parameters; Lightning reports 161K
trainable and 223M non-trainable parameters.

Fresh experiments start directly from the official
`pretrained/resplat-base-dl3dv-256x448-view8-1934a04c.pth` checkpoint.  The
fixed semantic projector is deterministically constructed, while Situation B
and the residual adapter are zero initialized.  The historical V3 checkpoint
is therefore not required by the clean mainline.

## Safety and functionality checks

- 125 unit tests passed.
- Hydra configuration composition passed.
- One real forward/backward/optimizer step passed.
- Peak observed GPU allocation was about 15.2--15.5 GiB on one RTX 4090D.
- Raster-VJP activity covered approximately 99.6%--99.9% of recurrent
  Gaussians in tested batches.
- At initialization, both Situation B and the residual adapter produce exactly
  zero residuals.
- On a fixed scene, the enabled identity model exactly matched baseline:
  PSNR 34.60627365, SSIM 0.96549332, LPIPS 0.06318111.

## Results

### Fixed-scene diagnostic

| Variant | PSNR | SSIM | LPIPS | Boundary F1 |
|---|---:|---:|---:|---:|
| Baseline | 34.606274 | 0.965493 | 0.063181 | 0.121591 |
| 200-step B only | 34.602718 | 0.965438 | 0.063120 | 0.121676 |
| 200-step adapter only | 34.608749 | 0.965502 | 0.063173 | 0.121631 |
| 200-step joint | 34.604847 | 0.965445 | 0.063106 | 0.121848 |

This small diagnostic is not a paper result.  It verifies that both trainable
paths leave zero, remain bounded, and do not damage the pretrained model.

### Fixed eight-scene held-out diagnostic after 1,000 multi-scene steps

| Variant | PSNR | Delta PSNR | SSIM | LPIPS | Boundary F1 |
|---|---:|---:|---:|---:|---:|
| Baseline | 28.904749 | -- | 0.882681 | 0.115584 | 0.183597 |
| B only | 28.909273 | +0.004525 | 0.882707 | 0.115603 | 0.183555 |
| Adapter only | 28.904785 | +0.000037 | 0.882684 | 0.115570 | 0.183586 |
| **Joint** | **28.909326** | **+0.004578** | **0.882708** | **0.115600** | **0.183572** |

The joint model's mean semantic cosine is 0.932996.  Mean measured encoder time
changed from 0.4061 s (baseline) to 0.4409 s (joint), approximately 8.6% in this
small run.  These values are development diagnostics, not final claims.

## Reproducibility

```bash
# Single-scene smoke/overfit
bash scripts/paper_clean_semantic_refinement_train.sh 200
bash scripts/paper_clean_semantic_refinement_eval.sh trained 200

# Multi-scene training
PAPER_SCENE=null PAPER_RUN_NAME=clean_joint_multiscene \
  bash scripts/paper_clean_semantic_refinement_train.sh 1000

# Eight-scene evaluation and ablations
bash scripts/paper_clean_semantic_refinement_multiscene_eval.sh baseline 1000
bash scripts/paper_clean_semantic_refinement_multiscene_eval.sh trained 1000
bash scripts/paper_clean_semantic_refinement_multiscene_eval.sh b_only 1000
bash scripts/paper_clean_semantic_refinement_multiscene_eval.sh adapter_only 1000
```

## Demo outputs

The evaluation path can save:

- context input images;
- target ground truth;
- initial and refined RGB renderings;
- rendered semantic PCA images;
- target DINO PCA images in the same PCA/color space;
- semantic cosine-error heatmaps;
- labeled horizontal comparison panels.

MP4 export is opt-in (`PAPER_SAVE_VIDEO=true`) because the current server image
does not expose an FFmpeg MP4 encoder.  Image and metric export do not depend on
FFmpeg.

## Next experiments

1. Run 5K--10K multi-scene steps with checkpoints every 1K and select by the
   fixed validation protocol.  At the measured speed, 10K steps are roughly
   1.5 hours on one 4090D.
2. Evaluate the selected checkpoint on the complete ReSplat test protocol, not
   only the eight-scene development subset.
3. Repeat baseline/B-only/adapter-only/joint for at least three seeds and report
   mean and standard deviation.
4. Report quality, semantic consistency, boundary quality, runtime, memory, and
   parameter overhead together.
5. Use the existing V3 uncertainty/support/ray-depth/split branches only as
   negative or supplementary ablations; do not re-enable them in the mainline.
