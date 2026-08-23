# Version 3 Stage 5: Semantic-Gated Learned Ray-Depth Refinement

## Objective

Test a safer alternative to direct semantic geometry VJP. Semantic residuals only
identify uncertain Gaussians; the direction of the geometry correction is learned
from target-view RGB MSE and LPIPS losses.

## Method

- Keep the official ReSplat initializer and recurrent updater frozen.
- Keep the successful Stage 3B semantic feature correction (`gain=0.3`).
- Form Gaussian-aligned semantic VJP features (`D+2` channels).
- Concatenate them with the frozen recurrent Gaussian state.
- Predict one scalar per Gaussian with a zero-initialized two-layer MLP.
- Move each Gaussian only along its original source-camera ray.
- Bound displacement with `tanh`, local Gaussian scale, a semantic-confidence
  gate, and a global gain.
- Train only the 136 K-parameter ray-depth head. Direct semantic geometry VJP and
  semantic feature loss are disabled, so DINO does not directly choose the
  movement direction.

The zero initialization makes the first forward pass exactly identical to the
Stage 3B baseline.

## Verification

- Python syntax check: passed.
- Shell syntax and `git diff --check`: passed.
- Unit tests: 77/77 passed (3 new tests cover exact zero identity, ray-parallel
  bounded movement, and low-confidence suppression).
- 2-step GPU smoke test: passed on a single RTX 4090D.
- Trainable parameters: 136 K; frozen parameters: 224 M.
- Observed GPU memory: about 6.0 GiB during the smoke/short runs.
- After two steps the mean relative displacement became nonzero, confirming that
  target-view photometric gradients reach the new head through the renderer.

## Five-sample screening

All rows use the same fixed evaluation samples. The `gain=0` row loads the
50-step checkpoint but disables the new displacement; it exactly reproduces the
Stage 3B baseline and rules out frozen-buffer/checkpoint drift.

| Checkpoint / gain | Mean relative displacement | Semantic cosine | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| 50-step / 0.0 (strict control) | 0 | 0.932560015 | 27.844989395 | 0.852752054 | 0.131972204 |
| 20-step / 0.1 | 0.0002402 | 0.932560086 | 27.845033264 | 0.852751553 | 0.131965177 |
| 20-step / 0.5 | 0.0012012 | 0.932560194 | 27.845043564 | 0.852750289 | 0.131974025 |
| 20-step / 1.0 | 0.0024025 | 0.932559335 | 27.844931412 | 0.852746069 | 0.131980081 |
| 50-step / 0.1 | 0.0005462 | 0.932560301 | 27.845081329 | 0.852751124 | 0.131967917 |

For 50-step / 0.1 versus its strict gain-zero control:

- PSNR: `+0.000091934` dB, with 5/5 samples improving.
- SSIM: `-0.000000930`, with 3/5 samples improving.
- LPIPS: `-0.000004287` (lower is better), with 4/5 samples improving.
- Semantic cosine: `+0.000000286`, effectively unchanged.

## Conclusion

The implementation is functional and the learned direction has a weak positive
signal: PSNR improves on every screened sample and LPIPS improves on four of five
after 50 steps. However, the magnitude is far below a paper-level improvement,
and larger inference gains become unstable. Therefore this module remains
disabled by default and should not yet replace the Stage 3B main method.

The code is worth retaining as a controlled candidate for later multi-scene
training. The next meaningful test is not a larger inference gain, but training
the small head on multiple scenes for substantially more steps and comparing it
against the same-checkpoint `gain=0` control.

