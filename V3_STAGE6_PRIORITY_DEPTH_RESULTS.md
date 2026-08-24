# V3 Stage 6 — Priority-Gated Ray-Depth Position Refinement

## Decision

**Complete and promoted to the V3 mainline.** The revised position head is
identity-initialized, bounded, selective, and does not materially damage the
Stage-5 baseline. Its RGB effect is small, as expected from the conservative
gate, but mean PSNR and LPIPS improve in both the 20-step and 50-step screens.

This is an engineering validation on one overfit scene and five fixed evaluation
samples. It is not a final paper claim.

## What changed from the early prototype

The early ray-depth prototype used raster-VJP gradient magnitude as confidence.
That signal was close to all-on and did not distinguish a strong semantic
residual from a trustworthy geometric observation.

The promoted module uses

```text
priority_i = need_i * reliable_i
```

from Stage 4. `need` measures semantic mismatch and `reliable` measures whether
the Gaussian had enough raster/visibility support. The learned head receives the
recurrent state, Gaussian-aligned semantic residual, need, reliability,
calibrated uncertainty, and priority. RGB MSE and LPIPS learn the signed depth
direction; semantic evidence only selects where an update is permitted.

For Gaussian `i`, the update is

```text
delta_d_i = gain * mean(scale_i) * tanh(raw_i) * priority_i
mu_i' = mu_i + delta_d_i * normalized(source_ray_i)
```

Consequently:

- no tangential 3-D motion is possible;
- displacement is bounded by local Gaussian size;
- priority zero gives exact identity;
- the final layer is zero initialized;
- gain zero exactly recovers Stage 5.

## Engineering checks

- Python compile: passed.
- Shell syntax: passed.
- `git diff --check`: passed.
- Full unit-test suite: **87/87 passed**.
- Trainable parameters: about **137 K**; the 224 M base parameters remain frozen.
- Reported peak evaluation memory: **5,700,932,608 bytes** (about 5.31 GiB).
- First forward pass: exactly zero ray-depth displacement.

## Matched five-sample ablations

All comparisons load the same trained checkpoint. `gain=0` disables only the
position update; `gain=0.1` enables it.

### 20-step checkpoint

| Metric | Gain 0 | Gain 0.1 | Delta |
|---|---:|---:|---:|
| Semantic cosine | 0.932563388 | 0.932563567 | +0.000000179 |
| PSNR | 27.845875931 | 27.845917892 | +0.000041962 |
| SSIM | 0.852741206 | 0.852740204 | -0.000001001 |
| LPIPS (lower is better) | 0.131982413 | 0.131979957 | -0.000002456 |

Mean relative ray-depth displacement was `1.0576e-4` local scales; the maximum
was `0.00649`. The priority gate was active for about 21.91% of Gaussians.

### 50-step checkpoint

| Metric | Gain 0 | Gain 0.1 | Delta |
|---|---:|---:|---:|
| Semantic cosine | 0.932563388 | 0.932563627 | +0.000000238 |
| PSNR | 27.845875931 | 27.845947266 | +0.000071335 |
| SSIM | 0.852741206 | 0.852740657 | -0.000000548 |
| LPIPS (lower is better) | 0.131982403 | 0.131977560 | -0.000004843 |

Mean relative displacement was `1.0619e-4` local scales; the maximum was
`0.00791`. PSNR improved on 3/5 samples, LPIPS improved on 3/5, and SSIM improved
on 2/5 with one tie. The average changes are favorable for PSNR and LPIPS; the
SSIM change is far below the roadmap warning threshold.

## Interpretation

The improvement is deliberately tiny because the gate is selective and the
position bound is conservative. This stage should be retained for the integrated
model, not advertised as an isolated large-gain component. Its research value is
the coherent causal role of semantic residual and visibility uncertainty in
controlling safe 3-D Gaussian motion.

## Next step

Stage 7 will define a dedicated integrated configuration and jointly optimize the
new semantic heads `(z, u, opacity, scale, ray-depth)` while keeping the original
ReSplat backbone frozen for the first multi-scene engineering validation.
