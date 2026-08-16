"""Visualize a Stage 8 multi-view boundary displacement diagnostic."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--radius", type=float, default=4.0)
    args = parser.parse_args()

    data = torch.load(args.input, map_location="cpu", weights_only=True)
    arrays = {key: value.squeeze().float().numpy() for key, value in data.items()}
    dx = arrays["dx"]
    dy = arrays["dy"]
    magnitude = np.sqrt(dx**2 + dy**2)
    supported = (arrays["rendered_boundary"] >= 0.1) & (arrays["confidence"] > 0)

    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    panels = (
        (arrays["rendered_boundary"], "Rendered soft boundary", "gray", 0, 1),
        (arrays["teacher_boundary"], "SAM2 teacher boundary", "gray", 0, 1),
        (magnitude, "Multi-view displacement magnitude (px)", "magma", 0, args.radius),
        (arrays["confidence"], "Depth-aware match confidence", "viridis", 0, 1),
        (arrays["visibility"], "Cross-view visibility", "viridis", 0, 1),
    )
    for axis, (image, title, cmap, low, high) in zip(axes.flat[:5], panels):
        handle = axis.imshow(image, cmap=cmap, vmin=low, vmax=high)
        axis.set_title(title)
        axis.axis("off")
        figure.colorbar(handle, ax=axis, fraction=0.046)

    axis = axes.flat[5]
    axis.imshow(arrays["teacher_boundary"], cmap="gray", vmin=0, vmax=1)
    step = 8
    yy, xx = np.mgrid[0 : dx.shape[0] : step, 0 : dx.shape[1] : step]
    mask = supported[::step, ::step]
    axis.quiver(
        xx[mask],
        yy[mask],
        dx[::step, ::step][mask],
        dy[::step, ::step][mask],
        magnitude[::step, ::step][mask],
        cmap="turbo",
        angles="xy",
        scale_units="xy",
        scale=0.35,
        width=0.003,
    )
    axis.set_title("Source-view correction vectors")
    axis.axis("off")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
