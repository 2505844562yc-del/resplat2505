# V3 Stage S0: Semantic-Carrying Gaussian Feasibility and Code Contract

## Decision

The semantic-Gaussian version can and should be developed on top of the current
repository. The archived V2 geometry residual remains available but is disabled
by default (`use_semantic_gaussian_init: false`). V1 boundary data, confidence
gating, local alignment, recurrent refinement, and initial/final evaluation are
reusable infrastructure rather than competing model changes.

Version-control boundary:

- archived tag: `v2-semantic-init-archived` at `75f36b3`;
- active branch: `v3-semantic-gaussians`;
- no V3 implementation is allowed on the archived branch;
- each stage receives an independent commit after tests pass.

## Rasterizer feasibility

The environment contains `gsplat 1.5.3`. Its public `rasterization` API supports
N-dimensional per-Gaussian features when:

```text
colors: [..., N, D]
sh_degree: None
```

It also exposes `channel_chunk=32`; therefore a 16-dimensional semantic feature
can be rendered without modifying or recompiling the CUDA rasterizer. Expected
depth can be requested together with features using `render_mode="RGB+ED"`
(where “RGB” means arbitrary feature channels when `sh_degree=None`).

The current RGB decoder uses SH coefficients with `sh_degree >= 0`, so semantic
features must be rendered by a separate decoder method/call. Concatenating SH
RGB and semantic embeddings into one call would be incorrect because SH and
arbitrary N-D features use different input contracts.

GPU execution cannot be validated in no-card mode. The first CUDA smoke test is
an explicit S1 gate, not an unresolved architectural risk.

## V3 Gaussian representation

The data structure will be extended from:

```text
G = (mean, covariance, scale, rotation, opacity, SH)
```

to:

```text
G_sem = (mean, covariance, scale, rotation, opacity, SH, z, uncertainty)
```

Initial implementation choices:

- `z`: 16-dimensional, L2-normalized semantic embedding;
- `uncertainty`: optional scalar, postponed until semantic rendering is stable;
- both new fields are optional so official ReSplat checkpoints and baseline
  code remain loadable;
- semantic fields do not alter Gaussian count or the RGB CUDA rendering path;
- missing semantics must produce `None`, never a fabricated all-zero semantic
  tensor that could be confused with a trained embedding.

## Required code seams

| Area | Stage-S1/S2 change | Compatibility rule |
|---|---|---|
| `src/model/types.py` | optional `semantic_features` and later uncertainty | existing positional constructors remain valid by appending fields |
| Gaussian initializer | predict/project one 16-D feature per generated Gaussian | zero/disabled path returns standard Gaussians |
| RGB decoder | no behavior change | official output must remain numerically identical |
| semantic decoder | new `forward_features` rasterization call | `sh_degree=None`, FP32 normalized output |
| sliding-window merge | concatenate semantic fields when present | reject mixed semantic/non-semantic windows |
| recurrent updater | initially carry `z` unchanged; later predict `delta_z` | geometry update is not enabled in S1/S2 |
| checkpoint loading | new keys loaded non-strictly | official checkpoint remains the initialization source |
| evaluation | semantic similarity and collapse diagnostics | existing RGB metrics unchanged |

There are four Gaussian reconstruction sites in `model_wrapper.py` for sliding
windows/evaluation. They must use one shared merge helper before semantic fields
are added; otherwise it is easy to silently drop `z` in one inference path.

## Semantic teacher choice

The first representation experiment uses a frozen DINOv2 dense feature as the
teacher and SAM2 only for trusted boundary/instance structure.

Rationale:

- DINOv2 provides denser spatial correspondence than global CLIP features;
- SAM2 supplies precise regions and boundaries but not a compact cross-scene
  semantic embedding by itself;
- CLIP is reserved for a later open-vocabulary experiment after the semantic
  Gaussian loop is proven.

To fit a single 4090D and the current 8.1-GB free disk budget, S1 must not create
full-resolution 768-D sidecars for the dataset. The initial options are:

1. reuse/expose already-computed frozen backbone features where possible;
2. teacher inference at reduced resolution with no gradient;
3. cache only compressed 16/32-D features for the small overfit scene.

Dataset-wide feature extraction is forbidden until the short-scene experiment
passes.

## Information-flow and leakage contract

During inference, only context RGB and context teacher/SAM features may initialize
the semantic Gaussians. Target semantic features are allowed only as supervision:

```text
context teacher feature -> Gaussian z initialization
target teacher feature  -> rendered-feature loss only
```

No target feature, target mask, or target boundary may enter the initializer or
recurrent state.

## Staged implementation gates

### S1: representation and rasterization plumbing

- append optional semantic fields to `Gaussians`;
- add a shared Gaussian merge helper;
- add `forward_features` to the gsplat decoder;
- carry a synthetic 16-D feature through merge and render;
- verify gradients reach the synthetic feature;
- verify RGB rendering is unchanged.

Pass gate: CPU structure tests plus one CUDA synthetic render/backward smoke when
a GPU is enabled.

### S2: semantic initialization only

- obtain reduced frozen DINOv2 context features;
- project to 16 dimensions;
- attach one embedding to every initial Gaussian;
- render target-view semantic features;
- train only the projection/semantic head;
- keep recurrent geometry updates disabled.

Pass gate: no feature collapse, multi-view semantic loss decreases, official RGB
baseline stays unchanged within numerical tolerance.

### S3: semantic state refinement

- append rendered semantic residual to recurrent input;
- update `z` and optionally opacity first;
- keep mean/scale semantic updates disabled until S3 is stable.

### S4: semantic-guided reconstruction

- uncertainty-gated semantic residual may update mean/scale near trusted
  boundaries;
- compare semantic-output-only against true semantic-guided geometry;
- reuse V1 boundary loss/local alignment as an independent ablation.

## Stage-S0 outcome

S0 passes in no-card mode:

- repository and disk are healthy;
- archived and active Git lines are separated and pushed;
- V3 can reuse the current modified code safely;
- installed gsplat already supports 16-D differentiable feature rasterization;
- no CUDA source change is required for the first semantic Gaussian prototype;
- the exact CUDA render/backward check is deferred to the first GPU session.
