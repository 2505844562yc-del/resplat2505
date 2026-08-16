# Minimal multi-training-scene validation

## Goal

Test whether the Stage 4 joint local-alignment prototype survives a change of overfit training scene before investing in multi-view boundary consensus.

## Protocol

- Three fixed DL3DV training scenes: `032dee9f`, `073f5a9b`, and `14eb48a5`.
- Baseline versus joint boundary loss plus `residual_alignment` feedback.
- 200 update steps, one recurrent refinement, fixed Gaussian count.
- Alignment radius 2, sigma 1.0, boundary weight 0.01, feedback scale 0.35.
- Every checkpoint is evaluated on the same deterministic five-scene test.

## Aggregate results by training scene

| Train scene | Delta PSNR | Delta SSIM | Delta LPIPS | Delta Boundary L1 | Delta Boundary F1 |
|---|---:|---:|---:|---:|---:|
| `032dee9f` | +0.020144 | +0.000469 | -0.000489 | -0.000480 | +0.001271 |
| `073f5a9b` | +0.003726 | -0.000089 | -0.000131 | -0.000758 | +0.002122 |
| `14eb48a5` | +0.000831 | +0.000045 | +0.000070 | -0.000207 | +0.000511 |
| Mean | +0.008234 | +0.000142 | -0.000183 | -0.000481 | +0.001301 |

Positive is better for PSNR, SSIM, and Boundary F1. Negative is better for LPIPS and Boundary L1.

## Granular wins over 15 train-model/test-scene pairs

| Metric | Joint wins | Mean delta | Delta range |
|---|---:|---:|---:|
| PSNR | 10/15 | +0.008234 | [-0.013649, +0.057329] |
| SSIM | 11/15 | +0.000142 | [-0.000415, +0.001832] |
| LPIPS | 9/15 | -0.000183 | [-0.002048, +0.000453] |
| Boundary L1 | 15/15 | -0.000481 | [-0.001153, -0.000009] |
| Boundary F1 | 14/15 | +0.001301 | [-0.000841, +0.003539] |

## Decision

The minimum validation gate passes:

- Boundary F1 improves for all three training-scene aggregates.
- Boundary L1 improves in every one of the 15 granular comparisons.
- Mean RGB metrics do not show a reconstruction-quality trade-off.

Proceed to the Multi-View Boundary Consensus prototype. These are short-run screening results, not paper-level final evidence.

## Storage

- Baseline checkpoints for the two added scenes were removed after evaluation; metrics and logs remain.
- Joint checkpoints for the two added scenes are retained for diagnostics and follow-up experiments.

## Reproduction

```bash
bash scripts/stage5_minimal_multiscene.sh 200
```

The canonical `032dee9f` pair was reused from Stage 4 during this run; the script can reproduce all three pairs from scratch.

Raw metrics are under `outputs/stage5_minival/eval` and the canonical pair remains in the Stage 3/4 evaluation directories.
