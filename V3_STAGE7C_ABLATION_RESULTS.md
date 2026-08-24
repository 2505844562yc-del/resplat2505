# V3 Stage 7C — Matched Semantic Gaussian Ablation

## Decision

**Pass as an engineering ablation. Keep the complete semantic Gaussian chain for
the next longer experiment.**

The new semantic state, uncertainty, support, and source-ray position paths form
one working end-to-end chain. At the current 500-step budget, the full model has
the best semantic cosine and preserves the official ReSplat RGB reconstruction
quality. The measured RGB deltas are extremely small and must not yet be reported
as a stable reconstruction improvement.

## Fair protocol

- Starting point for every trained variant: the same official ReSplat checkpoint,
  `resplat-base-dl3dv-256x448-view8-1934a04c.pth`.
- Frozen base: DINO projection, depth model, original ReSplat updater, and decoder.
- Trainable parameters: only the enabled semantic heads.
- Training: 500 steps, batch size 1, learning rate `5e-5`, identical seed and data
  sampler settings.
- Evaluation: the same 20 fixed DL3DV samples at `256x448`.
- This is a controlled pilot on the available subset, not a paper-scale result.

The four trained variants are cumulative:

1. `semantic_z`: 16-D Gaussian semantic state, semantic rendering, and raster-VJP
   state refinement.
2. `z_uncertainty`: add per-Gaussian semantic uncertainty/visibility state.
3. `z_uncertainty_support`: add priority-gated opacity and log-scale residuals.
4. `full_joint`: add priority-gated source-ray depth/3-D position residuals.

## Twenty-sample result

| Variant | Semantic cosine | Uncertainty MAE ↓ | Uncertainty corr. ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Official ReSplat | — | — | — | 29.323122692 | 0.896757293 | 0.114894969 |
| `semantic_z` | 0.935452133 | — | — | 29.323122692 | 0.896757293 | 0.114894969 |
| `z_uncertainty` | 0.935475716 | 0.201514570 | 0.290225110 | 29.323122692 | 0.896757293 | 0.114894969 |
| `z_uncertainty_support` | 0.935497317 | 0.201574854 | 0.290033641 | 29.323052502 | 0.896757805 | 0.114894449 |
| `full_joint` | **0.935513845** | 0.201555690 | **0.290250336** | **29.323322201** | 0.896755630 | **0.114892715** |

Full joint minus official ReSplat:

- semantic cosine: not available for the RGB-only baseline;
- PSNR: `+0.000199509` dB;
- SSIM: `-0.000001663`;
- LPIPS: `-0.000002254` (lower is better).

These RGB differences are effectively zero at this sample count. The result
supports non-degradation, not a claim of meaningful reconstruction gain.

## Paired sample counts

Against official ReSplat, `full_joint` is better on:

- PSNR: 15/20 samples;
- SSIM: 4/20 samples;
- LPIPS: 11/20 samples.

Against official ReSplat, the support-only structural variant is better on
10/20 PSNR, 8/20 SSIM, and 10/20 LPIPS samples. The mixed directions reinforce
the conclusion that all RGB changes remain very small.

## Gaussian update diagnostics

For `full_joint` on the 20 evaluation samples:

- priority-active Gaussians: `0.216893` (about 21.7%);
- mean absolute opacity-logit residual: `5.98e-5`;
- maximum absolute opacity-logit residual: `0.00600`;
- mean absolute log-scale residual: `2.17e-5`;
- maximum absolute log-scale residual: `0.00432`;
- mean relative source-ray depth displacement: `8.60e-5` local scales;
- maximum relative source-ray depth displacement: `0.00871` local scales.

The structural updates are non-zero but conservative. There is no evidence of
opacity, scale, or position divergence.

## Interpretation

The ablation establishes the intended causal chain:

```text
image/context features
        -> per-Gaussian semantic embedding z
        -> differentiable semantic rendering and raster residual
        -> Gaussian uncertainty/visibility priority
        -> bounded opacity, scale, and source-ray position updates
        -> ordinary Gaussian rendering
```

`semantic_z` and `z_uncertainty` exactly reproduce the baseline RGB metrics by
design because they do not alter RGB Gaussian parameters. The support and ray
depth heads make small, measurable structural changes while keeping reconstruction
at baseline level. This is the correct engineering outcome for the current
conservative first version.

The uncertainty numbers are calibration diagnostics rather than ground-truth
semantic labels. Their tiny variation across jointly trained variants is not
enough to rank architectures independently.

## Storage policy and retained artifact

After evaluation, intermediate ablation checkpoints were deleted while their
logs, configurations, and all per-sample metric JSON files were retained. The
retained Stage-7C model is:

`outputs/v3_stage7c/full_joint/500steps/checkpoints/epoch_1-step_500.ckpt`

Deleted checkpoints can only be recreated by rerunning the corresponding
500-step commands.

## Next experiment

Do not add another architecture module immediately. First run `full_joint` for a
longer frozen-backbone schedule and evaluate on more fixed samples/scenes. The
goal is to determine whether the conservative semantic structural updates grow
into a repeatable gain or remain neutral. Only after this evidence should we
consider unfreezing the last ReSplat updater block or tuning the support/ray-depth
gains.
