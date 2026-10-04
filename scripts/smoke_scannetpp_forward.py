"""One real ScanNet++ batch through the frozen ReSplat + semantic heads."""

from pathlib import Path
import sys
import json
from dataclasses import fields, replace

import torch
from hydra import compose, initialize_config_dir

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_typed_root_config
from src.dataset import get_dataset
from src.model.encoder import get_encoder
from src.model.decoder import get_decoder


def main():
    torch.manual_seed(1234)
    with initialize_config_dir(version_base=None, config_dir=str(Path("config").resolve())):
        cfg = compose(config_name="main", overrides=["+experiment=semantic_multihyp_scannetpp"])
    typed = load_typed_root_config(cfg)
    dataset = get_dataset(typed.dataset, "train", None)
    sample = next(iter(dataset))
    context = {
        key: value.unsqueeze(0).cuda() if torch.is_tensor(value) else value
        for key, value in sample["context"].items()
    }
    target = {key: value.unsqueeze(0).cuda() for key, value in sample["target"].items()}
    encoder, _ = get_encoder(typed.model.encoder)
    checkpoint = torch.load(
        "pretrained/resplat-base-dl3dv-256x448-view8-1934a04c.pth",
        map_location="cpu", weights_only=False,
    )["state_dict"]
    base_weights = {
        key.removeprefix("encoder."): value
        for key, value in checkpoint.items()
        if key.startswith("encoder.")
    }
    load_result = encoder.load_state_dict(base_weights, strict=False)
    missing_base = [key for key in load_result.missing_keys if not key.startswith("semantic_initializer.")]
    if missing_base:
        raise ValueError(f"frozen geometry weights missing: {missing_base}")
    print("loaded_base_keys", len(base_weights), "missing_new_keys", len(load_result.missing_keys), flush=True)
    encoder = encoder.cuda().train()
    print("context_views", context["image"].shape[1], flush=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = encoder(context, global_step=0)
        losses = encoder.compute_semantic_losses(output["semantic_multihyp"], context)
    decoder = get_decoder(typed.model.decoder, typed.dataset).cuda()
    gaussians = output["gaussians"]
    gaussians = replace(gaussians, **{
        field.name: getattr(gaussians, field.name).float()
        for field in fields(gaussians) if torch.is_tensor(getattr(gaussians, field.name))
    })
    rendered = decoder(gaussians, target["extrinsics"], target["intrinsics"],
                       target["near"], target["far"], tuple(target["image"].shape[-2:]))
    losses["rgb_mse"] = (rendered.color - target["image"]).square().mean()
    total = sum(losses.values())
    print("losses", {key: round(value.item(), 5) for key, value in losses.items()}, flush=True)
    if not torch.isfinite(total):
        raise ValueError("non-finite supervised loss")
    total.backward()
    grads = [
        parameter.grad for parameter in encoder.semantic_initializer.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not grads or not all(torch.isfinite(grad).all() for grad in grads):
        raise ValueError("missing or non-finite semantic gradient")
    if any(parameter.grad is not None for parameter in encoder.parameters() if not parameter.requires_grad):
        raise ValueError("frozen geometry unexpectedly received gradients")
    report = {"context_views": context["image"].shape[1],
              "target_views": target["image"].shape[1],
              "losses": {key: value.item() for key, value in losses.items()},
              "gradient_tensors": len(grads),
              "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
              "frozen_geometry_missing_keys": missing_base,
              "finite_forward_backward": True}
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/scannetpp_smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS finite forward/backward; gradient_tensors", len(grads),
          "peak_cuda_gib", round(torch.cuda.max_memory_allocated() / 2**30, 2), flush=True)


if __name__ == "__main__":
    main()
