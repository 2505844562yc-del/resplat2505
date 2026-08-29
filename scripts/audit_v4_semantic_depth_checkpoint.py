"""Audit semantic depth-conditioning updates against pretrained ReSplat."""

from argparse import ArgumentParser
from pathlib import Path

import torch


ADAPTER_TOKENS = {
    "logit": "encoder.depth_predictor.semantic_depth_adapters",
    "feature": "encoder.depth_predictor.semantic_depth_feature_adapters",
    "situation_a": "encoder.depth_predictor.semantic_depth_residual_injections",
    "situation_b": "encoder.depth_predictor.semantic_depth_concat_projections",
    "semantic_projector": "encoder.semantic_feature_projector",
}


def is_depth_tail(name: str) -> bool:
    if ".depth_predictor.depth_head." in name:
        return True
    parts = name.split(".")
    if "regressor" not in parts:
        return False
    regressor_index = parts.index("regressor")
    tail = parts[regressor_index + 2 :]
    return bool(tail) and (tail[0] == "4" or tail[:2] == ["3", "out"])


def state_dict(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint.get("state_dict", checkpoint)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("trained", type=Path)
    parser.add_argument(
        "--allow-depth-tail",
        action="store_true",
        help="allow and require changes in the staged depth-tail layers",
    )
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
    changed_depth_tail = []
    compared_depth_tail = 0
    maximum_depth_tail_delta = 0.0
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
        if args.allow_depth_tail and is_depth_tail(name):
            compared_depth_tail += 1
            maximum_depth_tail_delta = max(maximum_depth_tail_delta, delta)
            if delta != 0:
                changed_depth_tail.append((name, delta))
            continue
        maximum_base_delta = max(maximum_base_delta, delta)
        if delta != 0:
            changed_base.append((name, delta))

    learned_residual = False
    print(f"adapter_group_count={len(adapters)}")
    for kind, adapter in adapters.items():
        print(f"{kind}_adapter_tensor_count={len(adapter)}")
        if kind == "situation_a":
            gamma_values = torch.cat(
                [value.reshape(-1) for name, value in adapter.items() if name.endswith(".gamma")]
            )
            gamma_max = gamma_values.abs().max().item()
            learned_residual = learned_residual or gamma_max > 0
            print(f"situation_a_gamma_abs_mean={gamma_values.abs().mean().item():.9g}")
            print(f"situation_a_gamma_abs_max={gamma_max:.9g}")
        elif kind == "situation_b":
            concat_values = torch.cat(
                [value.reshape(-1) for value in adapter.values()]
            )
            concat_max = concat_values.abs().max().item()
            learned_residual = learned_residual or concat_max > 0
            print(
                "situation_b_semantic_slice_abs_mean="
                f"{concat_values.abs().mean().item():.9g}"
            )
            print(f"situation_b_semantic_slice_abs_max={concat_max:.9g}")
        elif kind in {"logit", "feature"}:
            residual_values = torch.cat(
                [
                    value.reshape(-1)
                    for name, value in adapter.items()
                    if "residual_head" in name
                ]
            )
            residual_max = residual_values.abs().max().item()
            learned_residual = learned_residual or residual_max > 0
            print(
                f"{kind}_residual_head_abs_mean="
                f"{residual_values.abs().mean().item():.9g}"
            )
            print(f"{kind}_residual_head_abs_max={residual_max:.9g}")
    print(f"base_tensors_compared={compared_base}")
    print(f"base_tensors_changed={len(changed_base)}")
    print(f"base_parameter_max_delta={maximum_base_delta:.9g}")
    if args.allow_depth_tail:
        print(f"depth_tail_tensors_compared={compared_depth_tail}")
        print(f"depth_tail_tensors_changed={len(changed_depth_tail)}")
        print(f"depth_tail_parameter_max_delta={maximum_depth_tail_delta:.9g}")
    for name, delta in sorted(changed_base, key=lambda item: item[1], reverse=True)[:5]:
        print(f"changed_base={name}:{delta:.9g}")

    if not learned_residual:
        raise RuntimeError("semantic depth conditioning did not open from identity")
    if changed_base:
        raise RuntimeError("one or more pretrained base tensors changed")
    if args.allow_depth_tail and not changed_depth_tail:
        raise RuntimeError("staged depth-tail layers did not update")


if __name__ == "__main__":
    main()
