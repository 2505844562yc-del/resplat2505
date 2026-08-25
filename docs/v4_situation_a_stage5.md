# Situation A Stage-5 short-screen report

## Scope

The semantic depth-logit adapter was trained alone from the same frozen V3
checkpoint on one DL3DV training scene. The base depth network, Gaussian
initializer, recurrent updater, semantic heads, and decoder remained frozen.

## Reproducibility

- Base checkpoint: `outputs/v3_stage7c/full_joint/500steps/checkpoints/epoch_1-step_500.ckpt`
- Training script: `scripts/v4_semantic_depth_init_a_train.sh`
- Evaluation script: `scripts/v4_semantic_depth_init_a_eval.sh`
- Parameter audit: `scripts/audit_v4_semantic_depth_checkpoint.py`
- Trainable parameters: about 116 K
- Peak GPU memory: about 13.3 GiB on RTX 4090 D

## Held-out results

| Variant | PSNR | SSIM | LPIPS | semantic cosine | init PSNR | init SSIM | init LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Identity | 29.635212 | 0.9560330 | 0.05634323 | 0.95071876 | 27.911518 | 0.9410893 | 0.06983165 |
| 20 steps | 29.636074 | 0.9560298 | 0.05632246 | 0.95072246 | 27.911341 | 0.9410838 | 0.06990389 |
| 50 steps | 29.637554 | 0.9560311 | 0.05632611 | 0.95072919 | 27.911823 | 0.9410819 | 0.06988514 |

## Situation A diagnostics

| Variant | gate mean | gate max | logit residual L1 | candidate relative depth delta |
| --- | ---: | ---: | ---: | ---: |
| 20 steps | 0.085941 | 0.106935 | 1.9358e-4 | 1.4201e-6 |
| 50 steps | 0.089288 | 0.111718 | 5.1697e-5 | 1.3729e-6 |

Checkpoint auditing compared 829 shared floating-point tensors against the V3
checkpoint. None changed. The zero-initialized residual and gate heads became
non-zero in both trained checkpoints.

## Decision

The module is non-destructive, checkpoint-compatible, and receives a real
end-to-end learning signal. It remains in the main method. The present geometry
change is intentionally bounded but too small for the medium screen, so the next
experiment should strengthen the adapter signal before a multi-scene run. A
200-step run with the same settings is not yet justified because it would mainly
test duration rather than address the weak effective residual.
