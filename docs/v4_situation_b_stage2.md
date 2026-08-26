# Situation B Stage-2 feature-conditioned initialization report

## Architectural change

Situation B builds on the stable Situation-A branch but moves explicit semantic
conditioning earlier in the depth pipeline. ReSplat's depth regressor first
fuses the multi-view cost volume, CNN features, multi-view Transformer features,
and high-dimensional DINO monocular features. Situation B then inserts a small
adapter between the depth UNet output and the pretrained depth head:

`f_conditioned = f_depth + gate * bounded_residual(f_depth, z_semantic)`

Here `z_semantic` is the same frozen, projected 16-dimensional semantic field
that is stored on the initial Gaussians. The residual is bounded relative to the
local RMS of `f_depth`, and the gate is attenuated where the base depth
distribution is uncertain. The original depth head maps the conditioned feature
to candidate logits; Situation A's trusted-boundary-calibrated logit residual is
then applied. Thus the current B method is cumulative A+B, not a replacement for
A.

The feature residual head is zero initialized. Loading an old checkpoint is an
exact identity and requires no retraining of the 224 M-parameter base model.

## Verification

- Full unit suite: 115 tests passed.
- End-to-end identity evaluation matched the original identity PSNR, SSIM,
  initial PSNR, and initial SSIM exactly. LPIPS differed by 2.2e-8.
- RNG isolation makes the A6 and B2 training view sequences identical.
- Trainable parameters: about 232 K for the A and B adapters together.
- Frozen parameters: about 224 M.
- Peak GPU memory: about 13.4 GiB on RTX 4090 D.
- Checkpoint audit: 829 shared base tensors compared, zero changed.

## Matched 50-step held-out result

Scene: `032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7`.

| Variant | PSNR | SSIM | LPIPS | semantic cosine | init PSNR | init SSIM | init LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Identity | 34.609989 | 0.9655000 | 0.06316041 | 0.97425538 | 32.993423 | 0.95571911 | 0.08447147 |
| Situation A6 | 34.610844 | 0.9654894 | 0.06310438 | 0.97425318 | 32.992771 | 0.95568073 | 0.08454452 |
| Situation B2 | 34.602230 | 0.9654464 | 0.06339534 | 0.97423524 | 33.001190 | 0.95567095 | 0.08498620 |

Situation B2 versus identity:

- Initial PSNR: +0.00777 dB
- Final PSNR: -0.00776 dB
- Final SSIM: -0.0000536
- Final LPIPS: +0.0002349

## Adapter diagnostics

| Diagnostic | Situation A6 | Situation B2 |
| --- | ---: | ---: |
| Candidate relative depth delta | 2.7041e-5 | 2.6591e-4 |
| Logit residual L1 | 3.7009e-4 | 3.6089e-4 |
| Feature gate mean | n/a | 0.08598 |
| Feature gate max | n/a | 0.12884 |
| Feature relative residual L1 | n/a | 0.02005 |

The feature-conditioned path makes roughly 9.8 times the candidate-depth change
of Situation A6. Both the logit and feature residual heads learned non-zero
weights, while all shared base tensors remained unchanged.

## Interpretation and decision

Keep Situation B enabled. The early semantic path is active, checkpoint-safe,
and produces a small positive initial-PSNR signal. The slight final degradation
is not large enough to reject the architecture, but it reveals a compatibility
gap: the frozen recurrent updater was trained for the original initialization
distribution and does not fully exploit the changed initial Gaussians.

The next stage should address this interface rather than increase semantic
strength. A conservative B3 experiment should add a zero-initialized bridge from
the semantic-conditioned initialization diagnostics into the existing recurrent
updater state, train only that bridge together with the A/B adapters, and retain
the same parameter and checkpoint audits. This preserves the stable base while
allowing refinement to adapt to the new initialization.

