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
| Learned source-ray depth head | Keep as Stage-6 prototype | Zero initialized; 50-step PSNR improved on 5/5 and LPIPS on 4/5; effect is small but safe |

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
- Check that the gate is selective; the current Stage-5 prototype activates about
  99.9% of Gaussians at floor zero, so this stage must avoid a meaningless all-on
  gate.

### Stage 5 — Semantic-conditioned support update: opacity and scale

**Status: pending.**

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

**Status: prototype implemented early; retain and revise after Stages 4–5.**

- Reuse the existing zero-initialized ray-depth head.
- Restrict displacement to each Gaussian's original source-camera ray.
- Bound displacement by local Gaussian scale and semantic uncertainty gates.
- Replace the current almost-all-active magnitude gate with Stage-4 `need × reliable`.
- Train under photometric/perceptual reconstruction loss; semantics chooses the
  region, not the sign of the 3-D displacement.
- Keep direct full-3D semantic VJP and direct scale VJP disabled as negative
  ablations.

### Stage 7 — Joint recurrent semantic-structure refinement and paper ablations

**Status: pending.**

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

The mainline is **complete through Stage 3**, not failed at Stage 5. The earlier
Stage-4 direct geometry screen and Stage-5 learned ray-depth experiment were
useful out-of-order probes of the later geometry stages.

The correct execution order from the current commit is:

1. Implement Stage 5 opacity/scale support head with zero identity.
2. Reconnect the existing ray-depth head as Stage 6 using the new gate.
3. Build one dedicated V3 main experiment configuration that activates the
   retained components while base ReSplat defaults remain off.
4. Run a short multi-scene engineering validation of the full chain.
5. Only then start longer matched training and final ablations.

This route preserves all completed useful work and stops short-run noise from
continually changing the architecture.
