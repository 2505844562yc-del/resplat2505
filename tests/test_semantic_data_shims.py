import torch

from src.dataset.shims.augmentation_shim import reflect_views
from src.dataset.shims.crop_shim import apply_crop_shim_to_views


def _views() -> dict:
    semantic = torch.arange(4 * 6).reshape(1, 4, 6)
    instance = semantic + 100
    depth = semantic.float() + 1.0
    return {
        "image": torch.rand(1, 3, 4, 6),
        "intrinsics": torch.eye(3).unsqueeze(0),
        "extrinsics": torch.eye(4).unsqueeze(0),
        "depth": depth,
        "semantic": semantic,
        "instance": instance,
    }


def test_reflection_keeps_all_dense_supervision_aligned() -> None:
    views = _views()
    reflected = reflect_views(views)
    for key in ("image", "depth", "semantic", "instance"):
        assert torch.equal(reflected[key], views[key].flip(-1))


def test_crop_uses_nearest_neighbor_for_discrete_labels() -> None:
    views = _views()
    cropped = apply_crop_shim_to_views(views, (2, 4))
    assert cropped["image"].shape[-2:] == (2, 4)
    assert cropped["depth"].shape[-2:] == (2, 4)
    assert cropped["semantic"].shape[-2:] == (2, 4)
    assert cropped["instance"].shape[-2:] == (2, 4)
    assert cropped["semantic"].dtype == torch.long
    assert cropped["instance"].dtype == torch.long
    assert set(cropped["semantic"].unique().tolist()).issubset(
        set(views["semantic"].unique().tolist())
    )
