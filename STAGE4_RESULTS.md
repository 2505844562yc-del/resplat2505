# Stage 4: Local semantic boundary alignment

## Motivation

Stage 3 showed that raw residual gradients slightly improved RGB metrics but did not improve semantic boundary F1. The missing information was not edge orientation alone, but the direction from a rendered edge to a nearby trusted semantic edge.

## Method

The residual_alignment feature contains three channels:

1. confidence-gated signed boundary residual;
2. normalized horizontal displacement towards nearby trusted target boundaries;
3. normalized vertical displacement towards nearby trusted target boundaries.

For each predicted boundary pixel, target boundaries inside a local window are weighted by SAM2 confidence and a Gaussian spatial kernel. The expected x/y offset is multiplied by the predicted boundary strength. No Gaussian is added or removed. The ReSplat backbone, recurrent hidden state, Gaussian attributes, and number of refinement iterations remain unchanged.

Best configuration:

- alignment radius: 2 pixels;
- alignment sigma: 1.0;
- boundary loss weight: 0.01;
- semantic feedback scale: 0.35;
- training steps: 200;
- fixed Gaussian count.

## Fixed five-scene cross-scene evaluation

| Model | PSNR | SSIM | LPIPS | Boundary L1 | Boundary F1 |
|---|---:|---:|---:|---:|---:|
| Baseline, 200 steps | 27.222664 | 0.839730 | 0.141236 | 0.376504 | 0.141015 |
| Stage 2 residual, 200 steps | 27.222328 | 0.839703 | 0.141380 | 0.376123 | 0.142092 |
| Stage 4 alignment, 200 steps | 27.242808 | 0.840200 | 0.140747 | 0.376024 | 0.142285 |

Against the Stage 2 residual model, Stage 4 improves PSNR by 0.02048 dB, SSIM by 0.000497, LPIPS by 0.000632, Boundary L1 by 0.000099, and relative Boundary F1 by about 0.136%. Against the baseline, relative Boundary F1 improves by about 0.901% while all RGB metrics also improve.

## Scale ablation

| Configuration | PSNR | LPIPS | Boundary L1 | Boundary F1 |
|---|---:|---:|---:|---:|
| radius 4, sigma 2.0, 50 steps | 26.955163 | 0.144252 | 0.376576 | 0.141830 |
| radius 2, sigma 1.0, 50 steps | 26.958666 | 0.144220 | 0.376498 | 0.142035 |

A smaller neighborhood is clearly better. The wider window over-smooths local correspondence when several semantic edges lie nearby.

## Feedback-scale calibration

For the model trained with scale 0.25, inference scales 0.15, 0.20, 0.25, and 0.35 were evaluated. Scale 0.35 was the only setting that improved all five metrics over the Stage 2 residual. A final model was therefore trained and evaluated consistently at scale 0.35; its metrics are reported above.

## Resource cost

- Peak training memory: about 7.8 GB on an RTX 4090D.
- Trainable parameters remain about 14.0 M.
- The method keeps the Gaussian count fixed.
- Only one Stage 4 checkpoint is retained.

## Current interpretation

This is a stronger candidate than residual_gradient because it provides an explicit correction direction and improves both reconstruction and boundary metrics. However, the evidence is still a one-scene overfit training run evaluated on five test scenes. It is sufficient to promote the module to the main prototype, but not sufficient for a paper claim. The next stage must expand training scenes/seeds and report per-scene statistics.

## Reproduction

    bash scripts/stage2_overfit_ablation.sh joint 200 0.01 0.35 residual_alignment 2 1.0
