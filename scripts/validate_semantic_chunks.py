"""Validate ReSplat-style chunks carrying semantic topology supervision."""

from argparse import ArgumentParser
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def decode_map(value: torch.Tensor, name: str) -> np.ndarray:
    if value.ndim >= 2:
        array = value.squeeze().cpu().numpy()
    else:
        array = np.asarray(Image.open(BytesIO(value.cpu().numpy().tobytes())))
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D lossless map, got {array.shape}")
    return array


def validate_example(
    example: dict,
    num_classes: int,
    ignore_label: int,
    sample_maps: int,
) -> tuple[int, int, int]:
    required = ("key", "cameras", "images", "depths", "semantics", "instances")
    missing = [key for key in required if key not in example]
    if missing:
        raise KeyError(f"scene {example.get('key', '<unknown>')} misses {missing}")

    counts = {key: len(example[key]) for key in required[1:]}
    if len(set(counts.values())) != 1:
        raise ValueError(f"scene {example['key']} has misaligned frame counts: {counts}")
    frame_count = counts["images"]
    if example["cameras"].shape[-1] != 18:
        raise ValueError(f"scene {example['key']} cameras must have shape [F,18]")

    checked = sorted(set(np.linspace(0, frame_count - 1, sample_maps, dtype=int)))
    observed_classes: set[int] = set()
    observed_instances: set[int] = set()
    for index in checked:
        semantic = decode_map(example["semantics"][index], "semantic")
        instance = decode_map(example["instances"][index], "instance")
        depth = decode_map(example["depths"][index], "depth")
        if semantic.shape != instance.shape or semantic.shape != depth.shape:
            raise ValueError(
                f"scene {example['key']} frame {index} label shapes disagree: "
                f"semantic={semantic.shape}, instance={instance.shape}, depth={depth.shape}"
            )
        invalid = (semantic != ignore_label) & (
            (semantic < 0) | (semantic >= num_classes)
        )
        if invalid.any():
            values = np.unique(semantic[invalid])[:20].tolist()
            raise ValueError(
                f"scene {example['key']} frame {index} has unmapped classes {values}"
            )
        if (instance < 0).any():
            raise ValueError(f"scene {example['key']} frame {index} has negative instance IDs")
        if not np.isfinite(depth).all():
            raise ValueError(f"scene {example['key']} frame {index} has non-finite depth")
        observed_classes.update(np.unique(semantic[semantic != ignore_label]).tolist())
        observed_instances.update(np.unique(instance[instance > 0]).tolist())
    return frame_count, len(observed_classes), len(observed_instances)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("root", type=Path, help="dataset root containing train/test")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--sample-maps", type=int, default=3)
    args = parser.parse_args()

    chunks = sorted((args.root / args.split).glob("*.torch"))
    if args.max_chunks > 0:
        chunks = chunks[: args.max_chunks]
    if not chunks:
        raise FileNotFoundError(f"no .torch chunks under {args.root / args.split}")

    scenes = frames = 0
    max_classes = max_instances = 0
    for chunk_path in chunks:
        chunk = torch.load(chunk_path, map_location="cpu", weights_only=False)
        if not isinstance(chunk, list):
            raise TypeError(f"{chunk_path} must contain a list of scene dictionaries")
        for example in chunk:
            count, classes, instances = validate_example(
                example,
                args.num_classes,
                args.ignore_label,
                args.sample_maps,
            )
            scenes += 1
            frames += count
            max_classes = max(max_classes, classes)
            max_instances = max(max_instances, instances)
        print(f"PASS {chunk_path.name}: {len(chunk)} scenes")
    print(
        f"VALID dataset: chunks={len(chunks)}, scenes={scenes}, frames={frames}, "
        f"max_sampled_classes={max_classes}, max_sampled_instances={max_instances}"
    )


if __name__ == "__main__":
    main()
