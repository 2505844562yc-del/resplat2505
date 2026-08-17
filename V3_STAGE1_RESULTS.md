# V3 Stage S1: Semantic Gaussian Representation and Rendering Plumbing

## Implemented contract

`Gaussians` now has two optional trailing fields:

```text
semantic_features: [B, G, D]
semantic_uncertainty: [B, G, 1]
```

Appending optional fields preserves all existing positional constructors and
official checkpoints. Standard ReSplat Gaussians keep both fields as `None`.

The four duplicated sliding-window merge sites now call one shared
`merge_gaussians` helper. It concatenates semantic state when present and raises
an error for mixed semantic/non-semantic windows instead of silently dropping
features. Scene-level scale/shift metadata must agree across windows.

The recurrent updater carries semantic features and uncertainty unchanged while
it updates geometry and appearance. Predicting `delta_z` is intentionally
deferred until semantic rendering and supervision are validated.

## Feature rendering

`GSplatDecoderSplattingCUDA.forward_features` performs a separate gsplat call:

```text
colors = semantic_features [B, G, D]
sh_degree = None
render_mode = RGB
channel_chunk = 32
```

The composited feature is divided by accumulated alpha with a safe epsilon. RGB
SH rendering remains untouched and never reads the semantic fields.

## Verification

- Python compilation passed.
- All 51 CPU tests passed (47 existing plus 4 merge-contract tests).
- Synthetic CUDA feature render produced shape `[1, 1, 16, 32, 32]`.
- Semantic feature gradient L1 was `0.122853` (finite and nonzero).
- RGB maximum difference with versus without semantic fields was exactly `0.0`.
- Official ReSplat checkpoint loaded and completed the one-sample refine-0 smoke:
  PSNR `32.9934`, SSIM `0.955719`, LPIPS `0.0844715`.

## Outcome

S1 passes. The repository can store, merge, carry, and differentiably render
per-Gaussian 16-D semantics without changing RGB rendering or recompiling CUDA.
S2 may now connect a frozen dense teacher and initialize real Gaussian semantic
embeddings.
