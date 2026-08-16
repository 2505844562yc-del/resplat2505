# Version 2 frozen baseline

## Git boundary

- Version 1 tag: `v1-stage4-frozen`
- Frozen commit: `6f000f22144d9e99792c1eb2d01c380909f759b1`
- Version 2 integration branch: `v2-semantic-init`
- Later Version-1 experimental modules remain in the codebase but are disabled
  by default.

## Version 1 main-method configuration

```text
variant: joint
semantic feature: residual_alignment
boundary loss weight: 0.01
semantic feedback scale: 0.35
alignment radius: 2
alignment sigma: 1.0
Gaussian count: unchanged
recurrent refinements: 1 in the controlled validation
```

Disabled ablations:

```text
multi-view consensus: false
multi-view displacement: false
semantic parameter routing: false
semantic selective refinement: false
```

## Frozen controlled result

The canonical 200-step Stage-4 checkpoint evaluated on the deterministic
five-scene test produced:

| PSNR | SSIM | LPIPS | Boundary L1 | Boundary F1 |
|---:|---:|---:|---:|---:|
| 27.242808 | 0.840200 | 0.140747 | 0.376024 | 0.142285 |

This is the initial comparison point for Version 2. Version-2 experiments must
also report refine-step-0 metrics so that initialization quality is measured
separately from recurrent refinement.

## Data-access rule

Semantic initialization may consume only semantic maps belonging to context
views. Target-view boundary and confidence maps are restricted to supervision
and evaluation. They must never enter the encoder or initial Gaussian builder.

## Reproducibility gate

At branch creation:

- remote worktree clean;
- 36/36 unit tests passing;
- Stage-4 200-step checkpoint present;
- fixed five-scene evaluation JSON present;
- RTX 4090D data volume has approximately 8.1 GB free.
