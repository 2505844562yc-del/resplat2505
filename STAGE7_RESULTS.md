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
