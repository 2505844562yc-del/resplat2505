# V3 Stage 7 — Integrated Semantic-Structure Refinement

## Decision

**The engineering implementation is complete and retained.** Long training and
paper-scale matched ablations are still pending.

The integrated configuration jointly optimizes all promoted semantic heads while
keeping the original ReSplat backbone frozen. A 50-step screen over 50 distinct
DL3DV training scenes completed without NaN, out-of-memory errors, semantic
collapse, uncontrolled Gaussian motion, or material RGB regression.

## Integrated causal chain

The dedicated experiment activates:

```text
source DINO feature
  -> 16-D Gaussian semantic state z
  -> semantic feature rendering
  -> raster-VJP Gaussian residual
  -> semantic state correction
  -> need / reliability / uncertainty u
  -> priority-gated opacity and scale update
  -> priority-gated source-ray position update
```

The original ReSplat RGB/geometry updater remains the base branch. The base
`resplat.yaml` keeps all V3 modules off; only
`config/experiment/v3_semantic_joint_dl3dv.yaml` activates the integrated model.

## Optimization policy

Jointly trainable modules:

- `semantic_state_head`;
- `semantic_uncertainty_head`;
- `semantic_support_head`;
- `semantic_ray_depth_head`.

The frozen DINO projection, depth model, ReSplat updater, and decoder remain
unchanged. The integrated run has about **623 K trainable parameters** out of
224 M total. The measured evaluation peak was **5,700,930,048 bytes** (about
5.31 GiB).

The losses are:

- RGB MSE and LPIPS for reconstruction;
- semantic feature loss with weight 0.1;
- uncertainty calibration loss with weight 0.1;
- opacity/scale and ray-depth conservative regularizers with weight 0.01 each.

## Verification

- Hydra configuration composition: passed.
- Python compile and shell syntax: passed.
- `git diff --check`: passed.
- Full unit-test suite: **89/89 passed**.
- 2-step single-scene joint backward pass: passed.
- 20-step training: 20 distinct training scenes, passed.
- 50-step training: 50 distinct training scenes, passed.
- Fixed five-sample evaluation: passed.

## Five-sample results

The reference is the promoted Stage-6 50-step checkpoint. The joint checkpoints
start from that reference and optimize the four new heads at learning rate
`5e-5`.

### 20-scene / 20-step screen

| Metric | Stage 6 | Joint 20 | Delta |
|---|---:|---:|---:|
| Semantic cosine | 0.932563627 | 0.932522523 | -0.000041103 |
| Uncertainty MAE | 0.228940865 | 0.227749580 | -0.001191285 |
| PSNR | 27.845947266 | 27.845563507 | -0.000383759 |
| SSIM | 0.852740657 | 0.852745593 | +0.000004935 |
| LPIPS (lower is better) | 0.131977560 | 0.131973554 | -0.000004005 |

### 50-scene / 50-step screen

| Metric | Stage 6 | Joint 50 | Delta |
|---|---:|---:|---:|
| Semantic cosine | 0.932563627 | 0.932519829 | -0.000043798 |
| Uncertainty MAE | 0.228940865 | 0.225965485 | -0.002975380 |
| Uncertainty correlation | 0.239815190 | 0.240035990 | +0.000220799 |
| PSNR | 27.845947266 | 27.845166397 | -0.000780869 |
| SSIM | 0.852740657 | 0.852750897 | +0.000010240 |
| LPIPS (lower is better) | 0.131977560 | 0.131974103 | -0.000003457 |

Uncertainty MAE improved on 5/5 evaluation samples. SSIM improved on 4/5;
LPIPS improved on 2/5. PSNR decreased on 5/5, but the mean reduction is less than
0.001 dB and the largest sample reduction is about 0.003 dB, far below the
roadmap warning threshold.

## Update behavior after joint training

Joint training made support and position corrections more conservative:

- mean opacity-logit update: `3.12e-4 -> 1.11e-4`;
- mean log-scale update: `6.70e-5 -> 1.21e-5`;
- mean relative ray-depth update: `1.06e-4 -> 3.02e-5`;
- active priority fraction remains about 21.91% because it is determined by the
  deterministic Stage-4 gate.

This is a healthy numerical behavior: the learned heads do not exploit their
maximum allowed update. However, the slight semantic-cosine decrease means the
loss balance should be studied in longer validation rather than tuned from this
small screen.

## What this result does and does not prove

It proves that the full semantic-carrying Gaussian pipeline is executable,
jointly differentiable, memory-feasible on one RTX 4090D, and approximately
baseline preserving across multiple scenes.

It does not prove a publishable reconstruction gain. The next experimental phase
needs a longer matched training budget and the canonical ablation table. The
ReSplat backbone should remain frozen initially; unfreezing it now would obscure
which gains come from the semantic modules.
