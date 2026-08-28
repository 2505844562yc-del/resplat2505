# Screenshot Situation A: Residual Semantic Injection

## Fixed definition

Situation A now refers only to the residual feature-conditioning method shown
in the user's screenshot:

\[
F'_{\mathrm{depth}} = F_c + \gamma A(F_s), \qquad \gamma_0 = 0.
\]

- `F_c`: output of the original ReSplat depth U-Net/regressor.
- `F_s`: the shared 16-D semantic field projected from the frozen DINO teacher.
- `A`: a trainable semantic adapter that maps `F_s` to the depth-feature width.
- `gamma`: a learnable per-depth-channel gate initialized to exactly zero.
- `F'_depth`: input to ReSplat's original pretrained depth head.

The old candidate-logit residual remains available only as a legacy ablation
(`use_semantic_depth_logit_adapter`). It is not Situation A. The old gated
post-U-Net adapter also remains only for reproducing historical experiments
(`use_semantic_depth_feature_adapter`). The strict Situation A configuration
turns both legacy paths off.

## End-to-end flow

```text
multi-view images
  -> original ReSplat CNN / transformer / cost volume / DINO features
  -> original ReSplat depth U-Net -> F_c
  -> shared semantic projection -> F_s
  -> F'_depth = F_c + gamma * A(F_s)
  -> original ReSplat depth head -> candidate probabilities -> expected depth
  -> backprojection -> initial Gaussians carrying the same semantic z = F_s
  -> original recurrent refinement
```

Semantic information therefore acts before depth is converted into Gaussian
centres.  It is not a post-hoc correction of already-created Gaussian geometry.

## Stability choices

1. Every channel of `gamma` is exactly zero at initialization, giving exact
   pretrained identity while avoiding cross-channel gradient cancellation.
2. `A(F_s)` is not zero initialized. This avoids the deadlock where both the
   adapter and `gamma` receive zero gradients.
3. The applied residual is bounded relative to the local RMS magnitude of
   `F_c`; its maximum relative scale is 0.1 by default.
4. Adapter construction uses a forked RNG stream so enabling the identity
   module does not alter ReSplat's sampled training views.
5. The same semantic projection is used for depth conditioning and Gaussian
   attribute `z`; no duplicate semantic backbone is introduced.

## Training stages

### A1: adapter-only (default)

- Load the existing V3/ReSplat checkpoint.
- Freeze the complete original ReSplat backbone, depth U-Net/head, updater and
  decoder.
- Train only the semantic projector and Situation A residual adapter.
- Trainable parameters in the current configuration: about 124K of 224M.

Command:

```bash
V4_RUN_NAME=a1_50steps bash scripts/v4_situation_a_train.sh 50
```

### A2: optional depth-tail fine-tuning

Use only after A1 is stable. It additionally opens the U-Net output projection,
the enclosing final regressor projection and the original depth head. Earlier
depth layers, DINO, multi-view transformer and recurrent updater stay frozen.

```bash
V4_START_CKPT=<A1-checkpoint> \
V4_UNFREEZE_DEPTH_TAIL=true \
V4_DEPTH_TAIL_LR=1e-5 \
V4_LR=5e-5 \
V4_RUN_NAME=a2_tune_depth_tail \
bash scripts/v4_situation_a_train.sh 50
```

## Verification completed

### Unit and configuration tests

- 121 tests passed.
- Hydra composition confirms strict A is on and both legacy paths are off.
- Dedicated tests cover exact identity, first-step `gamma` gradient, later
  adapter gradient, residual bound, invalid input handling and freeze routing.

### Pretrained identity

On scene `032dee9f...`, with identical views and checkpoint, baseline and
Situation A at `gamma=0` produced exactly identical values:

| metric | baseline | Situation A identity |
|---|---:|---:|
| PSNR | 34.6062736511 | 34.6062736511 |
| SSIM | 0.9654933214 | 0.9654933214 |
| LPIPS | 0.0631811097 | 0.0631811097 |
| initial PSNR | 32.9934234619 | 32.9934234619 |
| initial SSIM | 0.9557191133 | 0.9557191133 |
| initial LPIPS | 0.0844714716 | 0.0844714716 |

Situation A diagnostics were all exactly zero before training.

### One-step learning and freeze audit

- `gamma`: `9.83472128e-05` after one step.
- relative applied feature residual: `5.12167162e-06`.
- relative candidate-depth change: `1.74326132e-07`.
- 812 shared floating-point base tensors compared.
- Changed pretrained base tensors: 0.
- Maximum pretrained parameter delta: 0.

This proves the path is both learnable and non-destructive. The one-step metric
values are only an engineering smoke test, not evidence of final quality.

### 50-step gate refinement

The first implementation used one global scalar `gamma`. After 50 steps it
returned from `9.83e-05` to `4.16e-06`, so different feature-channel gradients
were cancelling and the complete semantic geometry path was nearly closed. That
version remains recoverable at tag `v4-situation-a-implementation`.

The final A implementation uses a zero-initialized gate for each depth-feature
channel. It preserves exact identity but lets useful channels open independently.
On the same deterministic 50-step screen:

- mean absolute channel gate: `1.55494909e-05`;
- maximum absolute channel gate: `6.67973247e-04`;
- fraction of channels with `|gamma| > 1e-5`: `14.84375%`;
- relative applied feature residual: `6.57510384e-07`;
- relative candidate-depth change: `1.44502046e-08`;
- candidate-depth change was about 20.5x the scalar-gate version;
- all 812 audited pretrained tensors still had exactly zero change.

| metric | baseline | channel-gated A (50 steps) | difference |
|---|---:|---:|---:|
| PSNR | 34.6062737 | 34.6051636 | -0.0011101 dB |
| SSIM | 0.9654933 | 0.9654944 | +0.0000011 |
| LPIPS | 0.0631811 | 0.0632019 | +0.0000208 |
| initial PSNR | 32.9934235 | 32.9930458 | -0.0003777 dB |
| initial SSIM | 0.9557191 | 0.9557213 | +0.0000021 |

This short screen meets the engineering goal: the semantic path is active,
the baseline is not materially damaged, and the original model stays frozen.
It is not a paper-level multi-scene accuracy claim; that belongs to the later
formal training/ablation phase.

## Files

- Main module: `src/model/semantic_depth_initialization.py`
- Depth insertion: `src/model/encoder/unimatch/mv_unimatch.py`
- Encoder configuration/diagnostics: `src/model/encoder/encoder_resplat.py`
- Freeze routing: `src/main.py`, `src/model/semantic_gaussian.py`
- Experiment: `config/experiment/v4_situation_a_residual_injection_dl3dv.yaml`
- Train/eval: `scripts/v4_situation_a_train.sh`,
  `scripts/v4_situation_a_eval.sh`
- Checkpoint audit: `scripts/audit_v4_semantic_depth_checkpoint.py`
