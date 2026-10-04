# ScanNet++ data and training-path readiness (2026-10-04)

Branch: `semantic-multihyp-clean`.

## Converted data

- Scene `39f36da05b`, from official `nvs_sem_train`.
- 345 official DSLR training frames; the 24 official held-out images are excluded.
- Final resolution: 256 x 448; one scene chunk is about 62 MiB.
- Mean mesh-hit coverage: 90.2%; all 345 frames passed the chunk validator.
- Observed labels across this scene: 29 Top-100 semantic classes and 81 instances.

The converter uses official undistorted DSLR RGB and pinhole intrinsics, together
with COLMAP world-to-camera poses in the scan mesh coordinate system. Nerfstudio
world coordinates differ from the mesh coordinates and must not be used directly
for mesh projection. Intrinsics follow the same resize and crop as RGB.

Open3D raycasting determines the first visible triangle at every output pixel.
Ray directions have camera z=1, so the hit parameter is camera-z depth, stored
as lossless uint16 millimetre PNG. Semantic vertex IDs are remapped through the
official benchmark mapping to Top-100 IDs; unknown classes are 255. Instances
come from the official annotated vertex lists. Face labels use the first vertex,
matching the official toolkit convention. Pixels outside the mesh or the valid
image mask have invalid depth and ignored semantic/instance labels.

## Fixed integration problems

1. The old `bounded` sampler always returned two context views despite a YAML
   value of eight. The ScanNet++ config now selects existing `boundedv2`, with
   eight context views and two target views, matching the released view8 model.
2. Depth upsampler settings now match the DL3DV checkpoint: patch size 16,
   upsample factor 8 and lowest feature resolution 8. No frozen geometry weights
   are missing; shape-mismatched weights are not silently discarded.
3. Probability BCE for second-hypothesis and verifier supervision is evaluated
   in float32 outside autocast, avoiding the mixed-precision BCE runtime error.

## Verified complete path

`scripts/smoke_scannetpp_forward.py` loads the released checkpoint, samples a
real labeled batch, runs the new initialization pipeline, renders target RGB,
and backpropagates RGB MSE plus all seven semantic/topology losses.

- Context/target views: 8/2.
- All losses and all 66 observed trainable gradient tensors were finite.
- Frozen geometry parameters received no gradients.
- Peak allocated GPU memory: approximately 5.26 GiB on one RTX 4090 D.
- Machine-readable report: `outputs/scannetpp_smoke.json` (not committed).
- First-frame RGB, semantic overlay and depth QA: `outputs/scannetpp_projection_qa/`.

This is a readiness check using untrained new heads. The reported losses and
memory are not quality metrics or a formal training-memory guarantee. No claim
of improvement over ReSplat is established by this check. Formal evaluation must
use separate scenes and the official target-view protocol.

## Reproduction

```bash
python scripts/convert_scannetpp_scene.py --scene-id 39f36da05b \
  --image-split train --max-frames 0 \
  --output datasets/scannetpp_resplat/train/000000.torch
python scripts/validate_semantic_chunks.py datasets/scannetpp_resplat \
  --split train --sample-maps 345
python scripts/smoke_scannetpp_forward.py
```

The converter refuses to overwrite an existing chunk unless `--overwrite` is
explicitly supplied. Optional preprocessing dependency: Open3D 0.19.0, along
with its normal Python dependencies; the project also uses Pillow, plyfile,
NumPy, SciPy and PyTorch. Personalized dataset credentials remain outside Git.

Next work: define a separate development validation scene/protocol, then run a
short supervised optimization check before expanding the training scene set.
