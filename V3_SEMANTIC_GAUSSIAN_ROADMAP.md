# V3 Semantic-Carrying Gaussian Mainline Roadmap (Rebased)

## 1. Research objective

The main method extends each ReSplat Gaussian from

```text
G = (mu, Sigma, alpha, SH)
```

to

```text
G_sem = (mu, Sigma, alpha, SH, z, u)
```

where `z` is a low-dimensional semantic embedding and `u` is semantic/visibility
uncertainty. The semantic feature map is rendered in novel views. Its residual is
back-projected through differentiable splatting to obtain Gaussian-aligned
feedback. That feedback first corrects `z`, then conditions conservative updates
of opacity/visibility, scale, and source-ray depth.

The objective is a coherent semantic-conditioned reconstruction pipeline, not a
large short-run RGB gain. A module remains on the development mainline when it is
well motivated, numerically stable, identity-initialized, and does not materially
damage the ReSplat baseline.

## 2. Revised promotion policy

Short overfit runs are engineering gates, not final paper verdicts.

A stage is retained when all of the following hold:

1. Enabling an untrained zero-initialized head is exactly or numerically identical
   to the current baseline.
2. The added tensor has a real information path and receives nonzero gradients.
3. The update is bounded and can be disabled by one explicit gain/flag.
4. No semantic collapse, NaN, uncontrolled Gaussian growth, or large RGB
   regression occurs.
5. In the small screen, approximately baseline-level RGB is acceptable. As a
   practical warning boundary, investigate rather than automatically discard when
   mean PSNR drops more than 0.05 dB, SSIM drops more than 0.001, or LPIPS rises
   more than 0.002. Final claims must still use matched full experiments.

Only structurally invalid signals are removed from the mainline. Small or neutral
effects are kept until the integrated model is trained for enough steps.

## 3. Existing module ledger

### Retained mainline core

| Module | Status | Reason |
|---|---|---|
| Optional `semantic_features` / `semantic_uncertainty` in `Gaussians` | Keep | Backward compatible; RGB path is unchanged |
| Separate N-D semantic rasterization | Keep | Differentiable, no CUDA modification, RGB max difference was 0 |
| Frozen DINOv2 768-to-16 projection and per-Gaussian `z` initialization | Keep | Real semantic state, no target leakage, no feature collapse |
| Raster-VJP pixel-to-Gaussian feedback | Keep | Correct visibility/contribution alignment |
| Direct bounded `z` correction at gain 0.3 | Keep | 20/20 semantic samples improved; RGB is exactly unchanged |
| Priority-gated source-ray depth head | Keep | Zero initialized; bounded by local scale and activated only by Stage-4 `need × reliable`; 50-step mean PSNR and LPIPS improve without material baseline damage |

### Retained for ablation, not active in the final main path

| Module | Status | Reason |
|---|---|---|
| Isolated learned `delta_z` without explicit residual | Ablation | Stable but lacks the information needed to beat direct raster-VJP correction |
| Source-grid flattened semantic residual | Negative ablation | Image pixels and source Gaussians do not have valid one-to-one correspondence |
| Direct DINO VJP update of full 3-D centers/scales | Negative ablation | Semantic gradient is not a calibrated photometric geometry direction |
| V1 boundary alignment | Optional auxiliary ablation | Useful earlier result, but not the semantic-carrying Gaussian core |
| Multi-view boundary consensus/displacement, global parameter routing, selective geometry | Negative/auxiliary ablations | Implemented and reproducible, but not needed in the clean V3 causal chain |
| V2 boundary-conditioned initial depth/scale adapter | Archived ablation | Different initialization hypothesis; must not be mixed into V3 before V3 is complete |

No module above should be deleted. Disabled defaults preserve official checkpoint
compatibility; dedicated V3 experiment configurations, not the base ReSplat YAML,
will turn on the mainline components.

## 4. Canonical eight-stage route

### Stage 0 — Baseline, repository, and identity contract

**Status: complete.**

- Freeze the official ReSplat reference and dataset split.
- Separate V1, V2, and V3 Git history.
- Require optional fields, zero-initialized heads, and gain-zero controls.
- Keep storage-safe scripts and matched evaluation indices.

### Stage 1 — Semantic-carrying Gaussian representation and rendering

**Status: complete.**

- Add optional `z` and `u` fields.
- Preserve them through recurrent refinement and sliding-window merge.
- Render arbitrary 16-D Gaussian features with a separate gsplat call.
- Verify semantic gradients and exact RGB identity.

### Stage 2 — Semantic initialization

**Status: complete.**

- Reuse frozen ReSplat DINOv2 dense context features.
- Compress 768-D features to normalized 16-D `z`.
- Align them to the Gaussian source grid and attach one embedding per Gaussian.
- Target DINO is supervision/evaluation only and never enters inference state.

### Stage 3 — Closed semantic residual refinement

**Status: complete and promoted.**

- Render current `z` in context/novel views.
- Compute alpha-masked cosine residual.
- Use rasterizer VJP to aggregate the pixel residual back to the Gaussians that
  actually contributed to each pixel.
- Apply the bounded direct semantic correction (`gain=0.3`).
- Keep the old learned semantic head only as an ablation until joint Stage 7.

This is the current validated semantic core.

### Stage 4 — Gaussian uncertainty and visibility state

**Status: complete and promoted.**

- Convert semantic mismatch magnitude, raster contribution support, alpha, and
  recurrent state into one bounded uncertainty value `u_i` per Gaussian.
- Distinguish two concepts:
  - `need_i`: high residual means this Gaussian needs correction;
  - `reliable_i`: sufficient multi-view visibility means its correction is
    trustworthy.
- Store `u_i` in `Gaussians.semantic_uncertainty` and render an uncertainty map for
  diagnostics.
- First version uses deterministic detached statistics plus a zero-initialized
  learned correction. It must not change RGB parameters yet.
- Check that the gate is selective. The promoted gate activates about 21.9% of
  Gaussians on the five-sample screen rather than the earlier nearly all-on gate.

### Stage 5 — Semantic-conditioned support update: opacity and scale

**Status: complete and promoted.**

- Add one small zero-initialized support head with separate outputs:
  - bounded opacity-logit residual `delta_alpha` for visibility/occlusion;
  - bounded log-scale residual `delta_log_s` for Gaussian footprint.
- Input: frozen recurrent state, raster-VJP semantic direction/magnitude, `need`,
  and `reliable` uncertainty gates.
- Semantics decides where and how strongly to update. RGB MSE, LPIPS, existing
  ReSplat feature residual, and conservative regularizers decide the parameter
  direction.
- Do not reuse raw DINO geometry gradients as the update direction.
- Train the support head alone first, then together with the semantic state.

### Stage 6 — Semantic-conditioned 3-D position refinement

**Status: complete and promoted.**

- Reuse the existing zero-initialized ray-depth head.
- Restrict displacement to each Gaussian's original source-camera ray.
- Bound displacement by local Gaussian scale and semantic uncertainty gates.
- The earlier almost-all-active magnitude gate has been replaced by Stage-4
  `need × reliable` priority.
- Train under photometric/perceptual reconstruction loss; semantics chooses the
  region, not the sign of the 3-D displacement.
- Keep direct full-3D semantic VJP and direct scale VJP disabled as negative
  ablations.

### Stage 7 — Joint recurrent semantic-structure refinement and paper ablations

**Status: engineering implementation, 500-step pilot, and matched 500-step
ablation complete; longer multi-scene training pending.**

Activate the coherent recurrent update

```text
(z, u, alpha, log-scale, ray-depth)
        <- recurrent state + raster-VJP semantic residual
```

while the original ReSplat geometry/appearance updater remains the base branch.
Use staged optimization:

1. Train only new heads on several scenes.
2. Jointly train all new semantic heads.
3. Optionally unfreeze only the final ReSplat updater block if the frozen-base
   model saturates; do not retrain the entire encoder initially.

The dedicated `v3_semantic_joint_dl3dv` configuration now activates the complete
chain while the base ReSplat configuration remains disabled. A 50-step screen
jointly optimized 623 K new-head parameters over 50 distinct training scenes with
no material RGB regression. This validates the implementation, not the final
paper performance.

A subsequent 500-step pilot covered one full pass over the available 359-scene
training subset plus 141 second-epoch samples. On 20 fixed evaluation samples it
improved semantic cosine on 16/20, uncertainty MAE on 19/20, uncertainty
correlation on 15/20, and LPIPS on 13/20. Mean PSNR changed by only -0.00025 dB.
The current joint loss weights are therefore retained for the next experiment.

Stage 7C then trained cumulative variants from the same official ReSplat
checkpoint for 500 steps each. The complete model achieved semantic cosine
`0.935513845`, PSNR `29.323322201`, SSIM `0.896755630`, and LPIPS `0.114892715`
on the same 20 fixed samples. Relative to official ReSplat, PSNR changed by only
`+0.00020` dB, SSIM by `-0.0000017`, and LPIPS by `-0.0000023`. Full-joint PSNR
improved on 15/20 samples, but SSIM improved on only 4/20. The justified conclusion
is that the full semantic Gaussian chain is active and non-destructive; this is
not yet evidence of a stable RGB reconstruction gain.

Required matched ablations:

1. Official ReSplat.
2. `+ semantic z/rendering` (Stages 1–2).
3. `+ raster-VJP z refinement` (Stage 3).
4. `+ uncertainty/visibility` (Stage 4).
5. `+ opacity and scale` (Stage 5).
6. `+ ray-depth position` (Stage 6).
7. Full joint model.

Report RGB metrics, semantic cosine, semantic-boundary metrics, uncertainty
selectivity, parameter-update magnitudes, memory, and inference time.

## 5. Current position and immediate execution order

The modular mainline is **complete through Stage 6**, and the Stage-7 integrated
training path is implemented and engineering-validated. Stages 1–6 form one
causal chain from semantic-carrying Gaussians to conservative semantic-conditioned
updates of semantic state, uncertainty, support, and source-ray position.

The correct execution order from the current commit is:

1. Run a longer full-joint schedule with the currently validated loss weights
   while keeping the ReSplat backbone frozen.
2. Evaluate on more fixed samples and scenes, retaining paired per-sample metrics.
3. Tune support/ray-depth gains only if the longer evidence shows a consistent
   RGB regression or vanishing structural updates.
4. Unfreeze the final ReSplat updater block only if the frozen-backbone model
   clearly saturates.

This route preserves all completed useful work and stops short-run noise from
continually changing the architecture.

### Stage 8 — Semantic-conditioned recurrent updater

**Status: engineering complete and promoted; longer validation deferred.**

- Concatenate Gaussian semantic state `z`, raster-VJP residual, semantic need,
  reliability, uncertainty, and priority.
- Map the semantic signals to the 512-D recurrent state with a two-layer adapter.
- Zero-initialize the final adapter layer and apply a bounded, priority-gated
  residual before the original Gaussian update head.
- Provide an isolated adapter-only mode and a controlled mode that additionally
  unfreezes only the last recurrent point-transformer block.
- Preserve the original ReSplat configuration with Stage 8 disabled.

The zero-initialized adapter exactly reproduced Stage 7C on five fixed samples.
After 50 adapter-only steps it learned a non-zero state correction while RGB
metrics remained numerically unchanged. A 20-step adapter-plus-last-block screen
improved PSNR by `0.00204` dB and SSIM by `0.000043`, while LPIPS regressed by
`0.000039`. This is a non-destructive engineering result, not a paper-level gain.

The next optional architecture stage is fixed-candidate semantic splitting at
high-priority semantic boundaries. Full multi-scene validation of Stages 7–8 is
deferred, not eliminated, and remains mandatory before final paper claims.

### Stage 9 — Fixed-candidate semantic Gaussian split

**Status: engineering complete and promoted; longer validation deferred.**

- Select a fixed 25% candidate budget using detached semantic correction priority.
- Append one child Gaussian per selected parent, producing a fixed 1.25x count.
- Inherit parent color, rotation, and semantic embedding.
- Predict bounded child source-ray depth and log-scale residuals.
- Conserve coincident parent/child accumulated alpha while limiting each child's
  opacity share to `0.1 × priority`.
- Apply the split only to the final refinement output, so recurrent token count
  and cached KNN state remain unchanged.

The split-only head learned non-zero child geometry in 50 steps while keeping RGB
metrics effectively unchanged. A 20-step joint Stage 7–9 screen changed PSNR by
`+0.00075` dB, SSIM by `+0.000041`, and LPIPS by `+0.000016` on five fixed samples.
Semantic cosine changed by `-0.000041`. These mixed, tiny changes justify keeping
the mechanism but not claiming a stable performance gain.

The full first-paper architecture is now coherent through semantic state,
rendered residual, uncertainty, semantic-conditioned recurrent refinement, and
fixed-budget Gaussian allocation. The next recommended work is longer joint
training and broader paired evaluation rather than another mandatory module.
