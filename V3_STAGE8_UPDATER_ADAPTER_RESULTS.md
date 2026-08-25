# V3 Stage 8 — Semantic-Conditioned Recurrent Updater

## Decision

**Engineering pass. Promote the zero-initialized semantic updater adapter and
retain controlled last-block unfreezing in the Stage-8 training profile.**

This stage closes the remaining architectural gap in Stage 7: semantic Gaussian
state no longer only drives independent support and ray-depth heads. It now also
conditions the hidden state consumed by the original ReSplat Gaussian update
head.

The short experiments below validate implementation safety and gradient flow.
They are not paper-scale evidence of a reconstruction gain.

## Architecture

For every recurrent Gaussian token, the adapter receives:

- the current 16-D Gaussian semantic embedding `z`;
- the 18-D raster-VJP semantic feedback (direction, magnitude, activity);
- semantic need, reliability, deterministic uncertainty, and correction priority.

A two-layer MLP maps these 38 channels to the 512-D ReSplat updater state. Its
last layer is initialized to exactly zero. The injected residual is

```text
delta_h = 0.1 * priority * tanh(adapter(z, residual, uncertainty))
h_stage8 = h_resplat + delta_h
```

Only high-priority, sufficiently visible Gaussians can change the hidden state.
The bounded residual is applied after the recurrent point transformer and before
the original Gaussian update head and all semantic parameter heads.

## Training modes

- `adapter_only`: train 141,568 new adapter parameters; freeze every existing
  Stage-7C/ReSplat parameter.
- `adapter_last_block`: train the adapter plus only block 3, the last of four
  recurrent point-transformer blocks (about 2.4 M trainable parameters total).
- `joint`: train all semantic heads and the adapter while keeping the ReSplat
  updater frozen.

The base `resplat.yaml` keeps Stage 8 disabled. The dedicated Stage-8 experiment
profile enables the adapter and controlled last-block unfreezing.

## Zero-initialization identity check

The same Stage-7C checkpoint was evaluated on five fixed samples with the adapter
disabled and enabled before training.

| Metric | Stage 7C | Zero-init adapter | Difference |
|---|---:|---:|---:|
| Semantic cosine | 0.932715034 | 0.932715034 | 0 |
| PSNR | 27.845243454 | 27.845243454 | 0 |
| SSIM | 0.852749324 | 0.852749324 | 0 |
| LPIPS | 0.131974377 | 0.131974377 | 0 |
| Adapter state delta mean | — | 0 | — |
| Adapter state delta max | — | 0 | — |

The uncertainty-correlation reduction differed by about `1.2e-8` because of
floating-point reduction order. RGB outputs and reported reconstruction metrics
were exactly unchanged.

## Adapter-only 50-step smoke test

Starting point: Stage-7C full-joint 500-step checkpoint. Only the adapter was
trainable at learning rate `5e-5`.

- existing checkpoint parameters changed: **0**;
- adapter parameters: 141,568, all received non-zero updates;
- mean absolute hidden-state residual: `1.44e-5`;
- maximum absolute hidden-state residual: `0.00176`;
- PSNR: `27.845253372` (`+0.000009918`);
- SSIM: `0.852749300` (`-0.000000024`);
- LPIPS: `0.131975313` (`+0.000000936`);
- semantic cosine: `0.932715404` (`+0.000000370`).

The adapter learns a non-zero correction while leaving reconstruction effectively
unchanged.

## Adapter plus last block 20-step smoke test

Starting point: the same Stage-7C checkpoint. The adapter and last recurrent point
transformer block were trained at the smaller learning rate `1e-5`.

- trainable parameters: about 2.4 M;
- mean absolute adapter residual: `1.82e-6`;
- maximum absolute adapter residual: `0.000225`;
- PSNR: `27.847285843` (`+0.002042389`);
- SSIM: `0.852792454` (`+0.000043130`);
- LPIPS: `0.132013646` (`+0.000039269`, worse);
- semantic cosine: `0.932745004` (`+0.000029969`);
- uncertainty MAE: `0.206781194` versus `0.206861413`.

The result is mixed but non-destructive: PSNR, SSIM, semantic cosine, and
uncertainty improve slightly while LPIPS regresses slightly. Five samples and 20
steps are insufficient for a performance claim, but sufficient to retain the
mechanism for subsequent development.

## Verification

- Python compilation passed.
- Hydra composition passed for adapter-only and joint modes.
- All 94 unit tests passed.
- GPU memory remained about 6.1 GiB on one RTX 4090D.

## Storage policy

The 20/50-step checkpoints are temporary engineering artifacts. After metrics and
logs are retained, they may be deleted because the experiments are reproducible
from the Stage-7C checkpoint and the committed scripts.

## Next development step

The next architecture extension is fixed-candidate semantic splitting: allocate
additional candidate Gaussians near high-priority semantic boundaries while
keeping the maximum Gaussian count fixed and activation differentiable. Stage 8
provides the semantic-conditioned recurrent state that can score those candidates.
