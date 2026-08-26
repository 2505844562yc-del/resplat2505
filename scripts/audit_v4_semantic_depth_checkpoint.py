"""Audit semantic depth-adapter updates against their V3 initialization."""

from argparse import ArgumentParser
from pathlib import Path

import torch


ADAPTER_TOKENS = {
    "logit": "encoder.depth_predictor.semantic_depth_adapters",
    "feature": "encoder.depth_predictor.semantic_depth_feature_adapters",
}


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
    adapters = {
        kind: {
            name: value.float()
            for name, value in trained.items()
            if token in name and torch.is_tensor(value)
        }
        for kind, token in ADAPTER_TOKENS.items()
    }
    adapters = {kind: values for kind, values in adapters.items() if values}
    if not adapters:
        raise RuntimeError("trained checkpoint contains no semantic depth adapters")

    changed_base = []
    compared_base = 0
    maximum_base_delta = 0.0
    for name, base_value in base.items():
        if (
            any(token in name for token in ADAPTER_TOKENS.values())
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

    learned_residual = False
    print(f"adapter_group_count={len(adapters)}")
    for kind, adapter in adapters.items():
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
        residual_max = residual_values.abs().max().item()
        learned_residual = learned_residual or residual_max > 0
        print(f"{kind}_adapter_tensor_count={len(adapter)}")
        print(
            f"{kind}_residual_head_abs_mean="
            f"{residual_values.abs().mean().item():.9g}"
        )
        print(f"{kind}_residual_head_abs_max={residual_max:.9g}")
        print(
            f"{kind}_gate_head_weight_abs_mean="
            f"{gate_values.abs().mean().item():.9g}"
        )
        print(
            f"{kind}_gate_head_weight_abs_max="
            f"{gate_values.abs().max().item():.9g}"
        )
    print(f"base_tensors_compared={compared_base}")
    print(f"base_tensors_changed={len(changed_base)}")
    print(f"base_parameter_max_delta={maximum_base_delta:.9g}")
    for name, delta in sorted(changed_base, key=lambda item: item[1], reverse=True)[:5]:
        print(f"changed_base={name}:{delta:.9g}")

    if not learned_residual:
        raise RuntimeError("zero-initialized depth residual heads did not learn")
    if changed_base:
        raise RuntimeError("one or more pretrained base tensors changed")


if __name__ == "__main__":
    main()
