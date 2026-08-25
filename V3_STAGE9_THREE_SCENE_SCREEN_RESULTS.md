# Stage 7–9 Three-Scene Controlled Screen and Architecture Freeze

## Decision

The Stage 7–9 innovation architecture passes the three-scene engineering
screen. The architecture is now frozen for the first full-training run: no new
mandatory module should be added before multi-scene training evidence is
collected.

This decision means the modules are implemented, connected, trainable, and
non-destructive at screening scale. It does not mean the paper experiment is
finished or that a statistically reliable reconstruction gain has been shown.

## Protocol

Three independent DL3DV training scenes were adapted from the same Stage-7C
checkpoint. Every scene contains more than 300 frames.

- Training-visible frames: `0–119`
- Held-out context and target segment: `120–159`
- Training: 500 optimizer steps per scene
- Decision checkpoints: 200 and 500 steps
- Resolution: `256 × 448`
- Trainable parameters: 3.1 M of 224 M
- Frozen: the mature ReSplat backbone except its final recurrent updater block
- GPU memory: approximately 6.1 GiB
- Runtime: approximately 11 minutes per 500-step scene

The same view indices, optimizer, learning rate, losses, module switches, and
starting checkpoint were used for all three scenes.

## Per-scene delta versus Stage 7C

### Conservative 200-step checkpoint

| Scene | ΔPSNR ↑ | ΔSSIM ↑ | ΔLPIPS ↓ | ΔSemantic cosine ↑ | ΔUncertainty MAE ↓ |
|---|---:|---:|---:|---:|---:|
| 1 | -0.003479 | +0.000001 | +0.000302 | +0.000117 | -0.000738 |
| 2 | +0.011589 | +0.000027 | +0.000364 | +0.000323 | -0.001429 |
| 3 | -0.006916 | -0.000335 | +0.000280 | +0.000427 | -0.001131 |
| **Mean** | **+0.000398** | **-0.000102** | **+0.000315** | **+0.000289** | **-0.001100** |

### Semantic endpoint at 500 steps

| Scene | ΔPSNR ↑ | ΔSSIM ↑ | ΔLPIPS ↓ | ΔSemantic cosine ↑ | ΔUncertainty MAE ↓ |
|---|---:|---:|---:|---:|---:|
| 1 | -0.018217 | -0.000099 | +0.000388 | +0.000145 | -0.000989 |
| 2 | +0.023981 | +0.000239 | +0.000320 | +0.000400 | -0.001820 |
| 3 | -0.020767 | -0.000418 | +0.000449 | +0.000537 | -0.001284 |
| **Mean** | **-0.005001** | **-0.000093** | **+0.000386** | **+0.000360** | **-0.001365** |

Semantic cosine and uncertainty MAE improve on all three scenes at both
checkpoints. RGB PSNR/SSIM remain near Stage 7C. LPIPS consistently regresses by
roughly `0.0003–0.00045`; this is small enough for the non-destructive engineering
criterion but must be monitored during full training and reported honestly.

## Fixed-candidate split control

At step 200, the PSNR change produced by enabling the trained split rather than
disabling it was:

- Scene 1: `-0.000017 dB`
- Scene 2: `-0.001501 dB`
- Scene 3: `-0.002430 dB`

The split changes LPIPS by less than `0.000015` on every scene. Therefore the
fixed 1.25x semantic candidate allocation is not the main source of RGB drift
and remains in the frozen architecture.

## Architecture-freeze gates

The following are engineering gates, not publication significance thresholds:

1. No NaN, divergence, memory growth, or uncontrolled Gaussian count.
2. Only intended semantic heads and the last recurrent updater block change.
3. Mean PSNR regression no worse than `-0.05 dB`; no scene worse than `-0.10 dB`.
4. Mean SSIM regression no worse than `-0.0005`; no scene worse than `-0.001`.
5. Mean LPIPS regression no worse than `+0.001`; no scene worse than `+0.002`.
6. Semantic cosine or uncertainty MAE improves on at least two of three scenes.
7. The trained split remains a small perturbation when toggled at inference.

All gates pass. At 200 steps, mean PSNR is effectively unchanged, worst PSNR is
`-0.0069 dB`, worst SSIM is `-0.000335`, and worst LPIPS is `+0.000364`.
Semantic cosine and uncertainty MAE improve on all three scenes.

## What is complete

The first-paper innovation chain is complete:

1. Low-dimensional semantic embedding attached to every Gaussian.
2. Semantic feature rendering with target-view supervision.
3. Raster-VJP semantic residual aligned back to Gaussian state.
4. Uncertainty and reliability gating.
5. Semantic-conditioned opacity and scale support refinement.
6. Priority-gated source-ray depth/position refinement.
7. Joint semantic residual recurrent refinement.
8. Semantic-conditioned recurrent updater adapter.
9. Fixed-budget, high-priority semantic Gaussian splitting.

Experimental ablations, longer training, and paper writing remain incomplete;
these are validation tasks rather than missing architecture modules.

## Full-training entry plan

The project is ready to start full new-module training from the Stage-7C
checkpoint. It does not need to retrain the original ReSplat backbone from
scratch.

Use the available 359-scene training subset and keep the current 3.1 M trainable
parameter scope. The recommended single continuous run is:

1. Save and evaluate at 1,000 steps as an early safety checkpoint.
2. Continue to 5,000 steps if the 30-scene fixed test evaluation remains within
   the safety gates above.
3. Continue to 10,000 steps only if semantic metrics are still improving and RGB
   metrics have not begun a consistent decline.
4. Retain checkpoints at 1k, 2k, 5k, and 10k only; evaluate paired per-scene
   metrics against Stage 7C.

The 1,000-step checkpoint is part of the full run, not another architecture
development phase. If it violates the gates, first tune semantic loss weights or
the last-block learning rate; do not immediately invent another module.

