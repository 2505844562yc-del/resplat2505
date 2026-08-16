# Stage 6: Multi-View Boundary Consensus prototype

## Method

For every positive SAM2 boundary pixel in a context view, rendered expected depth and normalized camera intrinsics/extrinsics are used to back-project the pixel into 3D and reproject it into the other context views. Support combines a dilated target boundary match, SAM2 confidence, valid image projection, and soft relative-depth agreement.

The consensus score is detached before use so the network cannot game its uncertainty gate by changing depth. Non-boundary confidence is preserved, which keeps the penalty for false rendered edges. The retained prototype normalizes positive consensus to preserve the average semantic-feedback budget, strengthening supported boundaries while weakening unsupported ones.

The module is optional and disabled by default. Gaussian count, attributes, backbone, hidden state, and recurrent iteration count remain unchanged.

## Real-data diagnostic

On the canonical scene with eight context views, positive teacher boundaries had mean consensus 0.6081 and 59.72% had consensus at least 0.5. The projection therefore does not collapse to an all-zero gate.

## 20-step screening

| Variant | PSNR | SSIM | LPIPS | Boundary L1 | Boundary F1 |
|---|---:|---:|---:|---:|---:|
| Single-view alignment | 26.808416 | 0.831748 | 0.147262 | 0.376912 | 0.140578 |
| Attenuate feedback and loss | 26.812879 | 0.831844 | 0.147072 | 0.377101 | 0.140343 |
| Attenuate feedback only | 26.807752 | 0.831728 | 0.147223 | 0.377132 | 0.140323 |
| Normalized feedback only | 26.807616 | 0.831706 | 0.147232 | 0.376868 | 0.140555 |

Pure attenuation weakened semantic supervision. Preserving the mean feedback budget removed most of that regression, so only the normalized feedback design was promoted to 50 steps.

## 50-step fair comparison

| Variant | PSNR | SSIM | LPIPS | Boundary L1 | Boundary F1 |
|---|---:|---:|---:|---:|---:|
| Single-view alignment | 26.955709 | 0.835787 | 0.144131 | 0.376539 | 0.142268 |
| Normalized multi-view feedback | 26.954008 | 0.835709 | 0.144199 | 0.376490 | 0.142239 |

Against single-view alignment, normalized consensus improves Boundary L1 by 0.000049 but changes Boundary F1 by -0.000029 and PSNR by -0.00170 dB. These differences are effectively neutral and do not justify a 200-step run yet.

## Decision

The geometry implementation, depth visibility check, confidence redistribution, training path, and tests are complete. The current weighting rule is a viable prototype but not yet a validated contribution. Keep it behind configuration flags and do not claim an improvement.

The next useful revision should change how consensus enters refinement, rather than sweep tiny scalar weights. Candidates include adding consensus as an explicit feedback channel, selecting cross-view-supported displacement candidates, or using a warm-up schedule after depth becomes reliable.

## Reproduction

```bash
bash scripts/stage2_overfit_ablation.sh joint 50 0.01 0.35 residual_alignment 2 1.0 true 0.5 false
```
