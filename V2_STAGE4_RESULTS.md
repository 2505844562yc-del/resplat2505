# V2 Stage 4: controlled ablation and step-0 evaluation

## Goal

Make the four relevant methods directly comparable and distinguish a better
initial Gaussian set from improvements produced only by recurrent refinement.

## Four controlled modes

| Mode | Semantic initialization | V1 boundary loss | V1 boundary feedback |
|---|---:|---:|---:|
| `baseline` | no | no | no |
| `v1` | no | yes | yes |
| `init` | yes | no | no |
| `init_v1` | yes | yes | yes |

All modes keep the original ReSplat initializer and recurrent updater topology,
the same pretrained checkpoint, scene, views, optimizer schedule, and one refine
iteration. The `init` modes add only the V2 semantic conditioning and bounded
depth/scale residuals.

## New evaluation contract

When `num_refine > 0` and score computation is enabled, evaluation now renders
the Gaussian set twice with the same target cameras:

1. immediately after initial Gaussian construction;
2. after the recurrent refinement iteration.

The score file therefore contains `init_psnr`, `init_ssim`, `init_lpips`,
`init_boundary_l1`, and `init_boundary_f1`, alongside the existing final
metrics. It also reports `refine_psnr_gain`, `refine_ssim_gain`, and
`refine_lpips_reduction`.

The extra initial render is excluded from the decoder benchmark so published
runtime measurements still describe the actual inference path.

## Reproducible commands

```bash
bash scripts/v2_short_ablation.sh init_v1 20
bash scripts/v2_eval_ablation.sh init_v1 20 5
```

Replace `init_v1` with `baseline`, `v1`, or `init` for the controlled ablation.

## Smoke verification

The implementation passed syntax checks and all 46 unit tests. A two-step
`init_v1` train plus one-scene evaluation completed successfully. The diagnostic
reported distinct initial and final RGB and boundary scores. These two-step
values validate the pipeline only and are not evidence for or against the
method.
