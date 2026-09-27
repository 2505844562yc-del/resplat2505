# Semantic Multi-Hypothesis Training Contract

## Method boundary

The released ReSplat encoder is the frozen geometry provider. Recurrent refinement is disabled. The trainable path contains:

1. a semantic scene decoder that reuses frozen DINO tensors and an RGB boundary stem;
2. a boundary-aware one/two-mode depth decoder;
3. a cross-view semantic--geometry candidate verifier;
4. the mapping from verified candidates to renderer-ready Gaussian position, scale, and opacity.

The standard gsplat decoder remains unchanged. The current model has 210,342,267 encoder parameters; 986,954 (0.469%) are trainable.

## Chunk schema

Each `.torch` chunk is a list of scene dictionaries. A supervised scene must contain aligned per-frame arrays:

- `cameras`: `[F,18]`, using the existing ReSplat convention;
- `images`: `F` lossless RGB byte tensors;
- `depths`: `F` metric-depth PNG byte tensors (millimetres for ScanNet-style input);
- `semantics`: `F` lossless single-channel ID maps;
- `instances`: `F` lossless single-channel ID maps.

Semantic IDs must be remapped to `[0, num_classes-1]`; `255` is ignored. Instance ID `0` is ignored/background. Never store semantic or instance labels as JPEG.

Validate converted data before training:

```bash
python scripts/validate_semantic_chunks.py datasets/scannetpp_resplat \
  --split train --num-classes 100 --ignore-label 255
```

## Loss ownership

- `class`, `boundary`, `instance`, `confidence` train the semantic scene decoder.
- `second_gate` trains semantic-boundary activation of the second ray hypothesis.
- `hypothesis` trains mixture mass to cover metric ground-truth depth.
- `verification` trains cross-view candidate retention.
- RGB MSE continues through assembled Gaussians and ties all decisions to novel-view rendering quality.

If a modality is absent, only losses requiring that modality are skipped. This permits RGB-only DL3DV batches in mixed training without fabricated pseudo-labels.

## Recommended optimization sequence

1. Load `pretrained/resplat-base-dl3dv-256x448-view8-1934a04c.pth` with non-strict loading.
2. Keep the full ReSplat geometry provider frozen; train the 0.987M new parameters on labeled indoor data.
3. First stabilize class/boundary/instance heads, then enable all hypothesis and verifier weights. RGB MSE stays active throughout.
4. Select checkpoints on both rendering metrics and topology metrics; do not select on semantic accuracy alone.
5. Only after the frozen-backbone result is established, optionally unfreeze the final depth-decoder blocks at 10--20x lower learning rate as a separate ablation.

The RGB-only one-step smoke test is an implementation check, not evidence of method quality. Formal claims require supervised training plus baseline, multi-hypothesis-only, verifier-only, and joint ablations under the same split.
