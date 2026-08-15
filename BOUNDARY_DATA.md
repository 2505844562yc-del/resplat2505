# Offline boundary sidecars

Boundary supervision is stored separately from the original DL3DV torch
chunks. This keeps the source dataset immutable and allows different teacher
models to be compared without rebuilding DL3DV.

## Layout

For a boundary root, files must follow this structure:

    datasets/dl3dv_boundaries/
      train/<scene-id>/000000.npz
      train/<scene-id>/000001.npz
      test/<scene-id>/000000.npz

The six-digit file name is the zero-based frame index used by the DL3DV
chunk. Every NPZ file contains:

- boundary: uint8 [H,W] or float32 [H,W]
- confidence: uint8 [H,W] or float32 [H,W]

An optional leading singleton channel is accepted. Integer arrays are
normalized to [0,1]; floating-point arrays must already lie in [0,1].
Both maps must match the uncropped RGB resolution.

## Configuration

The native ReSplat behavior remains the default:

    dataset.load_boundaries=false

Enable sidecars explicitly:

    dataset.load_boundaries=true
    dataset.boundary_roots=[datasets/dl3dv_boundaries]
    dataset.boundary_missing_policy=error

Use error for experiments so incomplete supervision cannot pass silently.
The skip policy is intended only for constructing or auditing partially
generated sidecars. Boundary and confidence maps receive the same resize,
center crop, and horizontal reflection as RGB.
