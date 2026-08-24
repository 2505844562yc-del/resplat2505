# V3 Stage 7B — 500-Step Joint Multi-Scene Pilot

## Decision

**Pass. Keep the current architecture and joint loss weights.**

The 500-step pilot produces a stable semantic/uncertainty improvement while RGB
reconstruction remains effectively unchanged. The earlier 20/50-step semantic
cosine decrease was a short-training transient, not evidence that the joint
architecture was invalid.

This remains a pilot on the currently available DL3DV subset. It is not a final
paper result.

## Protocol

- Starting point: promoted Stage-6 50-step checkpoint.
- Trainable parameters: the four semantic heads, about 623 K total.
- Frozen parameters: DINO projection, depth model, ReSplat updater, decoder.
- Training length: 500 steps at learning rate `5e-5`.
- Data coverage: one full pass over 359 training scenes plus 141 samples from the
  next epoch.
- Primary evaluation: 20 fixed DL3DV samples using the same camera index file for
  Stage 6 and Joint-500.
- Secondary compatibility screen: the original five fixed samples.
- Peak evaluation memory: about 5.31 GiB.

## Twenty-sample matched result

| Metric | Stage 6 | Joint 500 | Delta | Improved samples |
|---|---:|---:|---:|---:|
| Semantic cosine | 0.935406396 | 0.935512379 | +0.000105983 | 16/20 |
| Uncertainty MAE (lower is better) | 0.224397733 | 0.201628742 | -0.022768991 | 19/20 |
| Uncertainty correlation | 0.280196755 | 0.290357082 | +0.010160328 | 15/20 |
| PSNR | 29.323563290 | 29.323317051 | -0.000246239 | 10/20 |
| SSIM | 0.896748552 | 0.896757257 | +0.000008705 | 10/20 |
| LPIPS (lower is better) | 0.114897727 | 0.114892402 | -0.000005325 | 13/20 |

The uncertainty MAE reduction is about 10.1% relative to the Stage-6 reference.
The PSNR difference is four hundred times smaller than the roadmap's 0.05 dB
warning threshold.

## Five-sample continuity result

| Metric | Stage 6 | Joint 500 | Delta |
|---|---:|---:|---:|
| Semantic cosine | 0.932563627 | 0.932712615 | +0.000148988 |
| Uncertainty MAE | 0.228940865 | 0.206916738 | -0.022024128 |
| Uncertainty correlation | 0.239815190 | 0.243645543 | +0.003830352 |
| PSNR | 27.845947266 | 27.845238113 | -0.000709152 |
| SSIM | 0.852740657 | 0.852751946 | +0.000011289 |
| LPIPS | 0.131977560 | 0.131973629 | -0.000003931 |

## Gaussian update diagnostics

On the five-sample evaluation:

- priority-active Gaussians: about 21.91%;
- mean opacity-logit update: `1.14e-4`;
- mean log-scale update: `2.89e-5`;
- mean relative ray-depth update: `1.12e-4` local Gaussian scales;
- maximum relative ray-depth update: about `0.00902` local scales.

The heads remain conservative and far from the configured displacement bounds.
There is no evidence of opacity, scale, or geometry divergence.

## Interpretation

The main measurable gain at this budget is uncertainty calibration, accompanied
by a small but consistent semantic-feature improvement. RGB metrics remain at
baseline level. This matches the research objective: semantics becomes a real
Gaussian state and refinement signal without sacrificing the mature ReSplat RGB
path.

No loss-weight change is justified now. Tuning from the 20/50-step transient would
have been premature and would have hidden the fact that semantic cosine recovers
with sufficient training.

## Next experiment

Prepare storage-safe matched ablation configurations and a longer joint schedule.
Keep the ReSplat backbone frozen so that the contribution of the semantic modules
remains identifiable. Only consider unfreezing the final updater block after the
matched frozen-backbone experiments saturate.
