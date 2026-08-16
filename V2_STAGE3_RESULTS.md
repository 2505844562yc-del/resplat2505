# V2-3 — Semantic Initial Depth and Scale Residuals

## Implemented geometry change

The Version-2 initializer now predicts, for every initial Gaussian candidate:

- one bounded relative depth correction;
- three independent raw-scale corrections.

Depth remains positive through an exponential factor bounded by
`depth_gain * tanh(correction)`. Raw scale logits receive independent bounded
additive residuals before the original softplus and clamp. The head is
zero-initialized, so an untrained module is an exact identity. Gaussian count,
rotation, opacity, pixel offset, and SH appearance are unchanged.

## ReSplat gradient-path audit

Two original ReSplat behaviors intentionally prevent initialization training
during refine-only training:

1. the complete initializer call is wrapped in `torch.no_grad()` when recurrent
   refinement is enabled;
2. initial means and scales are detached at the recurrent updater boundary.

Both paths are now conditionally preserved only for the original configuration.
When semantic Gaussian initialization is enabled, the graph is retained for
initial means/scales while the pretrained base parameters remain frozen. This
lets gradients reach only the new `encoder.semantic_init*` modules and keeps
the original training behavior unchanged when the feature is disabled.

## Verification

- 46/46 unit tests pass.
- Zero correction and closed gate are exact identities.
- Depth and the three scale axes are independently bounded.
- Real-data 2-step forward, backward, and checkpointing pass.
- After fixing both gradient barriers, formerly zero-initialized modules have
  non-zero learned weights after two steps:

```text
semantic projection weight norm: 0.064485
semantic gate output weight norm: 0.000340
initial geometry output weight norm: 0.001445
```

The earlier diagnostic checkpoints produced before both gradient barriers were
removed are invalid and must not be used for evaluation.

## Decision

V2-3 passes its engineering gate. Proceed to V2-4 with a controlled four-mode
experiment runner and refine-step-0 metrics before promoting any method based
on final-render metrics.
