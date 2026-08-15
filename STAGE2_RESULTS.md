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
slightly reduces RGB metrics.

## 50-step strength sweep

| Variant | PSNR | SSIM | LPIPS | Boundary L1 | Boundary F1 |
|---|---:|---:|---:|---:|---:|
| baseline | 33.77695 | 0.96078 | 0.07207 | 0.36833 | 0.12352 |
| joint, weight 0.01, feedback 0.25 | 33.76070 | 0.96071 | 0.07211 | 0.36784 | 0.12496 |
| joint, weight 0.02, feedback 0.5 | 33.75076 | 0.96066 | 0.07225 | 0.36754 | 0.12505 |
| joint, weight 0.05, feedback 1.0 | 33.75141 | 0.96052 | 0.07304 | 0.36657 | 0.12728 |

The 0.01/0.25 setting is the most balanced and is selected for confirmation.

## 200-step confirmation

| Variant | PSNR | SSIM | LPIPS | Boundary L1 | Boundary F1 |
|---|---:|---:|---:|---:|---:|
| baseline | 34.21778 | 0.963106 | 0.068122 | 0.368228 | 0.124384 |
| joint, weight 0.01, feedback 0.25 | 34.20945 | 0.963050 | 0.068169 | 0.367731 | 0.125524 |

The selected joint setting improves boundary F1 by about 0.916% relative while
reducing PSNR by only 0.0083 dB. Stage 3 should evaluate more scenes and improve
the feedback representation before any paper-level quality claim is made.
