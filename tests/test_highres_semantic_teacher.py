import torch

from src.dataset.shims.augmentation_shim import reflect_views
from src.dataset.shims.crop_shim import center_crop_to_aspect
from src.dataset.shims.patch_shim import apply_patch_shim_to_views


def test_center_crop_to_aspect_preserves_source_resolution() -> None:
    images = torch.rand(2, 3, 270, 480)

    cropped = center_crop_to_aspect(images, (256, 448))

    assert cropped.shape == (2, 3, 270, 472)


def test_patch_shim_keeps_highres_teacher_aligned() -> None:
    views = {
        "image": torch.rand(1, 2, 3, 270, 480),
        "semantic_teacher_image": torch.rand(1, 2, 3, 540, 960),
        "intrinsics": torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1),
    }

    cropped = apply_patch_shim_to_views(views, patch_size=64)

    assert cropped["image"].shape[-2:] == (256, 448)
    assert cropped["semantic_teacher_image"].shape[-2:] == (540, 945)


def test_horizontal_reflection_also_flips_semantic_teacher() -> None:
    image = torch.arange(12).reshape(1, 3, 2, 2).float()
    teacher = torch.arange(24).reshape(1, 3, 2, 4).float()
    views = {
        "image": image,
        "semantic_teacher_image": teacher,
        "extrinsics": torch.eye(4).unsqueeze(0),
    }

    reflected = reflect_views(views)

    assert torch.equal(reflected["image"], image.flip(-1))
    assert torch.equal(
        reflected["semantic_teacher_image"], teacher.flip(-1)
    )
