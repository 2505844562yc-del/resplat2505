# Stage 2 preliminary single-scene ablation

All variants use the same seed, context/target sequence, official ReSplat
checkpoint, one recurrent refinement, and 20 optimization steps. These numbers
are pipeline diagnostics, not paper results.

| Variant | PSNR | SSIM | LPIPS | Boundary L1 | Boundary F1 |
|---|---:|---:|---:|---:|---:|
| baseline | 33.49397 | 0.95957 | 0.07221 | 0.36837 | 0.12350 |
| loss_only | 33.48650 | 0.95944 | 0.07232 | 0.36786 | 0.12468 |
| feedback_only | 33.43749 | 0.95913 | 0.07402 | 0.36802 | 0.12437 |
| joint | 33.43179 | 0.95900 | 0.07388 | 0.36719 | 0.12655 |

The joint model improves boundary F1 by about 2.47% relative over baseline, but
slightly reduces RGB metrics. Next experiments should sweep the boundary-loss
weight and feedback scale, run longer than 20 steps, and evaluate more scenes.
