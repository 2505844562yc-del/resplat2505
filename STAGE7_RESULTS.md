# Stage 7A: Explicit Multi-View Consensus Channel

## Design

Stage 7A preserves the Stage 4 three-channel local boundary-alignment feedback
and appends one structural channel. At positive teacher-boundary pixels the new
channel is

```text
(2 * multi_view_consensus - 1) * boundary * SAM2_confidence
```

so cross-view-supported boundaries are positive, unsupported boundaries are
negative, and non-boundary pixels are zero. Consensus is detached and the
projection layer is zero-initialized. Gaussian count and recurrent iterations
are unchanged.

## Verification

- 21/21 unit tests pass.
- A 2-step real-data smoke test completed forward, backward, and checkpointing.
- Positive teacher boundaries had mean consensus 0.6081; 59.72% had consensus
  at least 0.5, matching the Stage 6 diagnostic.

## 20-step screening

Both variants use the same canonical training scene, fixed five-scene test,
boundary-loss weight 0.01, feedback scale 0.35, alignment radius 2, and sigma
1.0.

| Variant | PSNR | SSIM | LPIPS | Boundary L1 | Boundary F1 |
|---|---:|---:|---:|---:|---:|
| Stage 4 single-view alignment | 26.808416 | 0.831748 | 0.147262 | 0.376912 | 0.140578 |
| Stage 7A explicit consensus | 26.790160 | 0.831408 | 0.147415 | 0.377092 | 0.140202 |
| Delta (7A - Stage 4) | -0.018256 | -0.000340 | +0.000153 | +0.000180 | -0.000376 |

Higher is better for PSNR, SSIM, and Boundary F1; lower is better for LPIPS
and Boundary L1. Stage 7A is worse on all five metrics and is therefore not
promoted to 50 or 200 steps.

## Interpretation and decision

Raw signed consensus is too abrupt for direct linear addition to the existing
appearance-error tensor. In particular, a low consensus score can mean a false
teacher boundary, occlusion, inaccurate early rendered depth, or simply that
the boundary is not visible in enough other views. Encoding all of these as a
negative correction conflates uncertainty with evidence against the boundary.

Retain this implementation as a negative-result ablation. The next revision
should not increase training length. It should replace the signed scalar with a
small structure encoder and a learned, initially closed gate, treating
consensus as reliability context rather than a direct negative residual.

## Reproduction

```bash
bash scripts/stage2_overfit_ablation.sh joint 20 0.01 0.35 \
  residual_alignment_consensus 2 1.0 false 0.5 false
```

## Stage 7B--7D follow-up

The direct signed channel was followed by three progressively safer dual-stream
variants:

- **7B gated structure:** a small MLP encodes three alignment channels plus
  non-negative consensus reliability, followed by a per-token learned gate
  initialized near closed.
- **7C residual dual branch:** restores the complete Stage 4 boundary projection
  as a fallback and adds 7B only as a zero-initialized gated residual branch.
- **7D warm-up:** keeps the Stage 4 branch active, disables the consensus branch
  for five steps, then linearly enables it over ten steps.

The new branch adds about 37K parameters and roughly 0.2 GB peak GPU memory.
All 22 unit tests and real-data smoke tests pass.

### 20-step comparison

| Variant | PSNR | SSIM | LPIPS | Boundary L1 | Boundary F1 |
|---|---:|---:|---:|---:|---:|
| Stage 4 single-view alignment | 26.808416 | 0.831748 | 0.147262 | 0.376912 | 0.140578 |
| 7A signed direct channel | 26.790160 | 0.831408 | 0.147415 | 0.377092 | 0.140202 |
| 7B gated structure only | 26.804214 | 0.831904 | 0.147080 | 0.377274 | 0.139828 |
| 7C fallback + gated residual | 26.808110 | 0.831654 | 0.147325 | 0.376877 | 0.140558 |
| 7D 5/10-step warm-up | 26.806266 | 0.831718 | 0.147233 | 0.376893 | 0.140471 |

7B improves SSIM and LPIPS but degrades both boundary metrics. 7C safely
returns to Stage 4 performance, showing that the fallback works, but its new
branch is effectively neutral. 7D does not create a meaningful trend.

### Final Stage 7 decision

Do not promote any Stage 7 variant to 50 or 200 steps. The repeated neutral or
negative results indicate that a scalar consensus derived from current rendered
depth is not informative enough for recurrent correction, rather than a simple
weighting or capacity problem. Keep 7C/7D as well-controlled ablations, but do
not claim them as improvements. A future multi-view module should provide a
geometric correspondence or displacement candidate, not only a confidence
scalar.
