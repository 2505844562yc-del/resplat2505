#!/usr/bin/env python3
"""Summarize completed Stage-7C ablations without requiring training code."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("outputs/v3_stage7c")
VARIANTS = (
    "baseline",
    "semantic_z",
    "z_uncertainty",
    "z_uncertainty_support",
    "full_joint",
)
METRICS = (
    "psnr",
    "ssim",
    "lpips",
    "semantic_cosine",
    "semantic_uncertainty_mae",
    "semantic_uncertainty_correlation",
)


def metric_dir(variant: str, steps: int, test_len: int) -> Path:
    if variant == "baseline":
        return ROOT / variant / f"eval_{test_len}samples" / "metrics"
    return ROOT / variant / f"{steps}steps_eval_{test_len}samples" / "metrics"


def main(steps: int = 500, test_len: int = 20) -> None:
    rows: dict[str, dict[str, float | None]] = {}
    for variant in VARIANTS:
        directory = metric_dir(variant, steps, test_len)
        values: dict[str, float | None] = {}
        for metric in METRICS:
            path = directory / f"scores_{metric}_all.json"
            if not path.exists():
                values[metric] = None
                continue
            samples = json.loads(path.read_text())
            values[metric] = sum(samples) / len(samples)
        rows[variant] = values
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--test-len", type=int, default=20)
    args = parser.parse_args()
    main(args.steps, args.test_len)
