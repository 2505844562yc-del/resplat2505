"""Audit Situation A parameter updates against its V3 initialization."""

from argparse import ArgumentParser
from pathlib import Path

import torch


ADAPTER_TOKEN = "encoder.depth_predictor.semantic_depth_adapters"


def state_dict(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint.get("state_dict", checkpoint)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("trained", type=Path)
    args = parser.parse_args()

    base = state_dict(args.base)
    trained = state_dict(args.trained)
    adapter = {
        name: value.float()
        for name, value in trained.items()
        if ADAPTER_TOKEN in name and torch.is_tensor(value)
    }
    if not adapter:
        raise RuntimeError("trained checkpoint contains no Situation A parameters")

    residual_values = torch.cat(
        [
            value.reshape(-1)
            for name, value in adapter.items()
            if "residual_head" in name
        ]
    )
    gate_values = torch.cat(
        [
            value.reshape(-1)
            for name, value in adapter.items()
            if "gate_head.weight" in name
        ]
    )

    changed_base = []
    compared_base = 0
    maximum_base_delta = 0.0
    for name, base_value in base.items():
        if (
            ADAPTER_TOKEN in name
            or name not in trained
            or not torch.is_tensor(base_value)
            or not torch.is_floating_point(base_value)
            or "running_" in name
        ):
            continue
        trained_value = trained[name]
        if trained_value.shape != base_value.shape:
            continue
        compared_base += 1
        delta = (trained_value.float() - base_value.float()).abs().max().item()
        maximum_base_delta = max(maximum_base_delta, delta)
        if delta != 0:
            changed_base.append((name, delta))

    print(f"adapter_tensor_count={len(adapter)}")
    print(f"residual_head_abs_mean={residual_values.abs().mean().item():.9g}")
    print(f"residual_head_abs_max={residual_values.abs().max().item():.9g}")
    print(f"gate_head_weight_abs_mean={gate_values.abs().mean().item():.9g}")
    print(f"gate_head_weight_abs_max={gate_values.abs().max().item():.9g}")
    print(f"base_tensors_compared={compared_base}")
    print(f"base_tensors_changed={len(changed_base)}")
    print(f"base_parameter_max_delta={maximum_base_delta:.9g}")
    for name, delta in sorted(changed_base, key=lambda item: item[1], reverse=True)[:5]:
        print(f"changed_base={name}:{delta:.9g}")

    if residual_values.abs().max().item() == 0:
        raise RuntimeError("zero-initialized depth residual head did not learn")
    if changed_base:
        raise RuntimeError("one or more pretrained base tensors changed")


if __name__ == "__main__":
    main()
