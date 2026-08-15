#!/usr/bin/env python
"""Generate offline SAM 2.1 boundary supervision for DL3DV chunks."""

import argparse
import hashlib
import json
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-config",
        default="configs/sam2.1/sam2.1_hiera_s.yaml",
    )
    parser.add_argument("--stage", choices=("train", "test"), default="test")
    parser.add_argument("--scene")
    parser.add_argument("--max-scenes", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--points-per-side", type=int, default=24)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.88)
    parser.add_argument("--stability-score-thresh", type=float, default=0.92)
    parser.add_argument("--min-mask-area", type=int, default=64)
    parser.add_argument("--boundary-radius", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preview", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_image(encoded: torch.Tensor) -> np.ndarray:
    image = Image.open(BytesIO(encoded.numpy().tobytes())).convert("RGB")
    # PIL may expose a read-only view; SAM2 converts this array to a tensor.
    return np.asarray(image).copy()


def masks_to_boundary_maps(
    annotations: list[dict],
    image_shape: tuple[int, int],
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert SAM proposals into a union boundary and teacher confidence."""
    height, width = image_shape
    boundary = np.zeros((height, width), dtype=np.uint8)
    confidence = np.zeros((height, width), dtype=np.float32)
    kernel_size = 2 * radius + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    for annotation in annotations:
        mask = np.asarray(annotation["segmentation"], dtype=bool)
        if mask.shape != (height, width):
            raise ValueError(
                f"SAM mask shape {mask.shape} does not match {(height, width)}"
            )
        score = min(
            float(annotation["predicted_iou"]),
            float(annotation["stability_score"]),
        )
        confidence[mask] = np.maximum(confidence[mask], score)

        mask_u8 = mask.astype(np.uint8)
        dilated = cv2.dilate(mask_u8, kernel, iterations=1)
        eroded = cv2.erode(mask_u8, kernel, iterations=1)
        boundary[dilated != eroded] = 255

    confidence = np.rint(np.clip(confidence, 0, 1) * 255).astype(np.uint8)
    return boundary, confidence


def overlay_boundary(image: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    pixels = boundary > 0
    overlay[pixels] = (
        0.25 * overlay[pixels] + 0.75 * np.array([255, 32, 32])
    ).astype(np.uint8)
    return overlay


def load_index(dataset_root: Path, stage: str) -> dict[str, str]:
    path = dataset_root / stage / "index.json"
    with path.open("r") as file:
        return json.load(file)


def load_scene(dataset_root: Path, stage: str, chunk_name: str, scene: str):
    chunk = torch.load(dataset_root / stage / chunk_name, map_location="cpu")
    matches = [example for example in chunk if example["key"] == scene]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one scene {scene} in {chunk_name}, found {len(matches)}"
        )
    return matches[0]


def select_scenes(
    index: dict[str, str],
    requested_scene: str | None,
    max_scenes: int,
) -> list[str]:
    if requested_scene is not None:
        if requested_scene not in index:
            raise KeyError(f"Scene not found in index: {requested_scene}")
        return [requested_scene]
    return sorted(index)[:max_scenes]


def main() -> None:
    args = parse_args()
    if args.boundary_radius < 1:
        raise ValueError("--boundary-radius must be at least 1")
    if args.max_scenes < 1:
        raise ValueError("--max-scenes must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    checkpoint_hash = sha256(args.checkpoint)
    model = build_sam2(
        args.model_config,
        str(args.checkpoint),
        device=args.device,
        apply_postprocessing=False,
    )
    generator = SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        crop_n_layers=0,
        # Avoid depending on SAM2 optional CUDA connected-components extension.
        # Deterministic area filtering is applied to the proposals below.
        min_mask_region_area=0,
        output_mode="binary_mask",
    )

    index = load_index(args.dataset_root, args.stage)
    scenes = select_scenes(index, args.scene, args.max_scenes)
    manifest = {
        "teacher": "SAM 2.1 Hiera Small",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "model_config": args.model_config,
        "stage": args.stage,
        "parameters": {
            "points_per_side": args.points_per_side,
            "pred_iou_thresh": args.pred_iou_thresh,
            "stability_score_thresh": args.stability_score_thresh,
            "min_mask_area": args.min_mask_area,
            "boundary_radius": args.boundary_radius,
        },
        "scenes": {},
    }

    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if args.device.startswith("cuda")
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        for scene in scenes:
            example = load_scene(
                args.dataset_root, args.stage, index[scene], scene
            )
            output_dir = args.output_root / args.stage / scene
            output_dir.mkdir(parents=True, exist_ok=True)
            images = example["images"]
            frame_count = (
                len(images)
                if args.max_frames is None
                else min(len(images), args.max_frames)
            )
            scene_stats = []
            for frame_index in range(frame_count):
                output_path = output_dir / f"{frame_index:06d}.npz"
                if output_path.exists() and not args.overwrite:
                    scene_stats.append({"frame": frame_index, "skipped": True})
                    continue

                image = decode_image(images[frame_index])
                annotations = [
                    annotation
                    for annotation in generator.generate(image)
                    if int(annotation["area"]) >= args.min_mask_area
                ]
                boundary, confidence = masks_to_boundary_maps(
                    annotations, image.shape[:2], args.boundary_radius
                )
                np.savez_compressed(
                    output_path,
                    boundary=boundary,
                    confidence=confidence,
                )
                if args.preview:
                    preview = overlay_boundary(image, boundary)
                    Image.fromarray(preview).save(
                        output_dir / f"{frame_index:06d}_preview.jpg",
                        quality=92,
                    )
                scene_stats.append({
                    "frame": frame_index,
                    "masks": len(annotations),
                    "boundary_fraction": float((boundary > 0).mean()),
                    "mean_confidence": float(confidence.mean() / 255),
                })
                print(
                    f"{scene} frame {frame_index}/{frame_count - 1}: "
                    f"{len(annotations)} masks, "
                    f"{(boundary > 0).mean():.4f} boundary fraction"
                )
            manifest["scenes"][scene] = scene_stats

    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / f"manifest_{args.stage}.json").open("w") as file:
        json.dump(manifest, file, indent=2)


if __name__ == "__main__":
    main()
