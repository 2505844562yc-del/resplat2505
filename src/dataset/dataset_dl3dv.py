import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional

import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset
import numpy as np
import random
import torch.nn.functional as F

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler


@dataclass
class DatasetDL3DVCfg(DatasetCfgCommon):
    name: Literal["dl3dv"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    test_chunk_interval: int
    train_times_per_scene: int
    test_times_per_scene: int
    ori_image_shape: list[int]
    # random crop training
    random_crop: bool
    max_size: list[int]
    min_size: list[int]

    skip_bad_shape: bool = True
    near: float = -1.0
    far: float = -1.0
    baseline_scale_bounds: bool = True
    shuffle_val: bool = True
    no_mix_test_set: bool = True
    load_depth: bool = False
    min_views: int = 0
    max_views: int = 0
    highres: bool = False
    sort_target_index: Optional[bool] = False
    overfit_max_views: Optional[int] = None
    sort_context_index: Optional[bool] = False
    use_index_to_load_chunk: Optional[bool] = False
    pose_align_first_view: bool = False  # align the camera pose to the first view
    pose_align_middle_view: bool = False  # align the camera pose to the middle view
    pose_align_last_view: bool = False  # align the camera pose to the last view
    pose_align_random_view: bool = False  # align the camera pose to a random view
    scale_extrinsics: float = 1.
    center_pose: bool = False  # center and normalize the pose by the distance to the center

    # mix re10k & dl3dv
    mix_re10k: bool = False
    re10k_min_view_dist: int = 40
    re10k_max_view_dist: int = 300

    # load remaining context views
    load_remain_context: bool = False
    num_remain_context: int = 8

    # pose format
    opencv_pose_format: bool = False  # use opencv pose format 

    # test on train set
    test_on_train: bool = False

    load_boundaries: bool = False
    boundary_roots: Optional[list[Path]] = None
    boundary_missing_policy: Literal["error", "skip"] = "error"

class DatasetDL3DV(IterableDataset):
    cfg: DatasetDL3DVCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetDL3DVCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        if cfg.near != -1:
            self.near = cfg.near
        if cfg.far != -1:
            self.far = cfg.far

        # Collect chunks.
        self.chunks = []
        for i, root in enumerate(cfg.roots):
            root = root / self.data_stage
            if self.cfg.use_index_to_load_chunk:
                with open(root / "index.json", "r") as f:
                    json_dict = json.load(f)
                root_chunks = sorted(list(set(json_dict.values())))
            else:
                root_chunks = sorted(
                    [path for path in root.iterdir() if path.suffix == ".torch"]
                )

            # mixed data training only evaluate on a single test set
            if cfg.no_mix_test_set and self.data_stage in ['val', 'test'] and i > 0:
                continue

            # balance the datasets for mixed dataset training
            # for gs: mix re10k, dl3dv
            if len(cfg.roots) > 1 and self.data_stage == 'train':
                if 'dl3dv' in str(root):
                    root_chunks = root_chunks * 8

            self.chunks.extend(root_chunks)
        if self.cfg.overfit_to_scene is not None:
            chunk_path = self.index[self.cfg.overfit_to_scene]
            self.chunks = [chunk_path] * len(self.chunks)
        if self.stage == "test":
            # fast testing
            self.chunks = self.chunks[:: cfg.test_chunk_interval]
        if self.stage == "val":
            self.chunks = self.chunks * int(1e6 // len(self.chunks))

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __iter__(self):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
            self.chunks = self.shuffle(self.chunks)

        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.chunks = [
                chunk
                for chunk_index, chunk in enumerate(self.chunks)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]

        for chunk_path in self.chunks:
            # Load the chunk.
            chunk = torch.load(chunk_path)

            if self.cfg.overfit_to_scene is not None:
                item = [x for x in chunk if x["key"]
                        == self.cfg.overfit_to_scene]
                assert len(item) == 1
                if self.stage == "test":
                    chunk = item
                else:
                    chunk = item * len(chunk)

            if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
                chunk = self.shuffle(chunk)

            times_per_scene = (
                self.cfg.test_times_per_scene
                if self.stage == "test"
                else self.cfg.train_times_per_scene
            )

            for run_idx in range(int(times_per_scene * len(chunk))):
                example = chunk[run_idx // times_per_scene]

                extrinsics, intrinsics = self.convert_poses(example["cameras"])

                scene = example["key"]

                try:
                    extra_kwargs = {}
                    if self.cfg.overfit_to_scene is not None and self.stage != "test":
                        extra_kwargs.update(
                            {
                                "max_num_views": (
                                    148
                                    if self.cfg.overfit_max_views is None
                                    else self.cfg.overfit_max_views
                                )
                            }
                        )

                    if self.cfg.mix_re10k and 'dl3dv' not in scene and self.stage == 'train':
                        is_re10k = True
                    else:
                        is_re10k = False

                    out_data = self.view_sampler.sample(
                        scene,
                        extrinsics,
                        intrinsics,
                        min_context_views=self.cfg.min_views,
                        max_context_views=self.cfg.max_views,
                        min_view_dist=self.cfg.re10k_min_view_dist if is_re10k else None,
                        max_view_dist=self.cfg.re10k_max_view_dist if is_re10k else None,
                        **extra_kwargs,
                    )
                    if isinstance(out_data, tuple):
                        context_indices, target_indices = out_data[:2]
                        c_list = [
                            (
                                context_indices.sort()[0]
                                if self.cfg.sort_context_index
                                else context_indices
                            )
                        ]
                        t_list = [
                            (
                                target_indices.sort()[0]
                                if self.cfg.sort_target_index
                                else target_indices
                            )
                        ]
                    if isinstance(out_data, list):
                        c_list = [
                            (
                                a.context.sort()[0]
                                if self.cfg.sort_context_index
                                else a.context
                            )
                            for a in out_data
                        ]
                        t_list = [
                            (
                                a.target.sort()[0]
                                if self.cfg.sort_target_index
                                else a.target
                            )
                            for a in out_data
                        ]

                except ValueError:
                    # Skip because the example doesn't have enough frames.
                    continue

                # Skip the example if the field of view is too wide.
                if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                    continue

                for context_indices, target_indices in zip(c_list, t_list):
                    # load remaining context views
                    if self.cfg.load_remain_context:
                        # randomly select fixed number of remaining views such that they can be batched
                        remaining_indices = get_remaining_indices(context_indices, target_indices,
                            self.cfg.num_remain_context)

                        # Load the images.
                        remain_context_images = [
                            example["images"][index.item()] for index in remaining_indices
                        ]

                        try:
                            remain_context_images = self.convert_images(remain_context_images)
                        except OSError:
                            # some data might be corrupted
                            continue

                    # Load the images.
                    context_images = [
                        example["images"][index.item()] for index in context_indices
                    ]

                    try:
                        context_images = self.convert_images(context_images)
                    except OSError:
                        # some data might be corrupted
                        continue

                    target_images = [
                        example["images"][index.item()] for index in target_indices
                    ]

                    try:
                        target_images = self.convert_images(target_images)
                    except OSError:
                        # some data might be corrupted
                        continue

                    if self.cfg.load_boundaries:
                        context_boundary_data = self.load_boundary_maps(
                            scene,
                            context_indices,
                            tuple(context_images.shape[-2:]),
                        )
                        target_boundary_data = self.load_boundary_maps(
                            scene,
                            target_indices,
                            tuple(target_images.shape[-2:]),
                        )
                        if (
                            context_boundary_data is None
                            or target_boundary_data is None
                        ):
                            continue
                        context_boundaries, context_boundary_confidence = (
                            context_boundary_data
                        )
                        target_boundaries, target_boundary_confidence = (
                            target_boundary_data
                        )
                        if self.cfg.load_remain_context:
                            remain_boundary_data = self.load_boundary_maps(
                                scene,
                                remaining_indices,
                                tuple(remain_context_images.shape[-2:]),
                            )
                            if remain_boundary_data is None:
                                continue
                            (
                                remain_boundaries,
                                remain_boundary_confidence,
                            ) = remain_boundary_data

                    # Skip the example if the images don't have the right shape.
                    if self.cfg.mix_re10k and 'dl3dv' not in scene:
                        if self.cfg.highres:
                            expected_shape = (3, 720, 1280)
                        else:
                            expected_shape = (3, 360, 640)
                    else:
                        expected_shape = tuple(
                            [3, *self.cfg.ori_image_shape]
                        )  # (3, 270, 480) or (3, 540, 960)

                    if self.stage in ['test', 'val'] or 'dl3dv' in scene:
                        expected_shape = tuple(
                            [3, *self.cfg.ori_image_shape]
                        )

                    context_image_invalid = context_images.shape[1:] != expected_shape
                    target_image_invalid = target_images.shape[1:] != expected_shape

                    if self.cfg.skip_bad_shape and (
                        context_image_invalid or target_image_invalid
                    ):
                        print(
                            f"Skipped bad example {example['key']}. Context shape was "
                            f"{context_images.shape}, target shape was "
                            f"{target_images.shape}, and expected shape was {expected_shape}"
                        )
                        continue

                    if self.cfg.load_remain_context:
                        remain_context_invalid = remain_context_images.shape[1:] != expected_shape

                        if self.cfg.skip_bad_shape and remain_context_invalid:
                            continue

                    if self.cfg.pose_align_random_view:
                        if self.stage == 'train':
                            rand_index = random.randint(0, context_indices.shape[0] - 1)
                            extrinsics = camera_normalization(extrinsics[context_indices][rand_index:rand_index+1], extrinsics)
                        else:
                            # middle view for val/test
                            rand_index = context_indices.shape[0] // 2
                            extrinsics = camera_normalization(extrinsics[context_indices][rand_index:rand_index+1], extrinsics)
                    else:
                        # align pose to the first view
                        if self.cfg.pose_align_first_view:
                            extrinsics = camera_normalization(extrinsics[context_indices][0:1], extrinsics)

                        # align pose to the middle view
                        if self.cfg.pose_align_middle_view:
                            mid_index = context_indices.shape[0] // 2
                            extrinsics = camera_normalization(extrinsics[context_indices][mid_index:mid_index+1], extrinsics)

                        # align pose to the last view
                        if self.cfg.pose_align_last_view:
                            extrinsics = camera_normalization(extrinsics[context_indices][-1:], extrinsics)

                    if self.cfg.center_pose:
                        extrinsics = center_norm_pose(extrinsics)

                    scale_factor = self.cfg.scale_extrinsics

                    # check the extrinsics
                    if any(torch.isnan(torch.det(extrinsics[context_indices][:, :3, :3]))):
                        continue

                    if any(torch.isnan(torch.det(extrinsics[target_indices][:, :3, :3]))):
                        continue

                    # check the extrinsics: translation could be very large
                    # https://github.com/DL3DV-10K/Dataset/issues/34
                    if (extrinsics[context_indices][:, :3, 3] > 1e3).any():
                        continue

                    if (extrinsics[target_indices][:, :3, 3] > 1e3).any():
                        continue

                    if not torch.allclose(torch.det(extrinsics[context_indices][:, :3, :3]), torch.det(extrinsics[context_indices][:, :3, :3]).new_tensor(1)):
                        continue
                    if not torch.allclose(torch.det(extrinsics[target_indices][:, :3, :3]), torch.det(extrinsics[target_indices][:, :3, :3]).new_tensor(1)):
                        continue

                    if self.cfg.load_remain_context:
                        if any(torch.isnan(torch.det(extrinsics[remaining_indices][:, :3, :3]))):
                            continue

                        if (extrinsics[remaining_indices][:, :3, 3] > 1e3).any():
                            continue

                        if not torch.allclose(torch.det(extrinsics[remaining_indices][:, :3, :3]), torch.det(extrinsics[remaining_indices][:, :3, :3]).new_tensor(1)):
                            continue

                    # scale the scene when necessary: only scale the extrinsics
                    extrinsics[:, :3, 3] *= scale_factor
                    example_out = {
                        "context": {
                            "extrinsics": extrinsics[context_indices],
                            "intrinsics": intrinsics[context_indices],
                            "image": context_images,
                            "near": self.get_bound("near", len(context_indices)),
                            "far": self.get_bound("far", len(context_indices)),
                            "index": context_indices,
                        },
                        "target": {
                            "extrinsics": extrinsics[target_indices],
                            "intrinsics": intrinsics[target_indices],
                            "image": target_images,
                            "near": self.get_bound("near", len(target_indices)),
                            "far": self.get_bound("far", len(target_indices)),
                            "index": target_indices,
                        },
                        "scene": scene,
                    }

                    if self.cfg.load_boundaries:
                        example_out["context"].update({
                            "boundary": context_boundaries,
                            "boundary_confidence": context_boundary_confidence,
                        })
                        example_out["target"].update({
                            "boundary": target_boundaries,
                            "boundary_confidence": target_boundary_confidence,
                        })

                    if self.cfg.load_remain_context:
                        example_out.update({
                            "context_remain": {
                                "extrinsics": extrinsics[remaining_indices],
                                "intrinsics": intrinsics[remaining_indices],
                                "image": remain_context_images,
                                "near": self.get_bound("near", len(remaining_indices)),
                                "far": self.get_bound("far", len(remaining_indices)),
                                "index": remaining_indices,
                            }
                            }
                        )
                        if self.cfg.load_boundaries:
                            example_out["context_remain"].update({
                                "boundary": remain_boundaries,
                                "boundary_confidence": (
                                    remain_boundary_confidence
                                ),
                            })

                    if self.stage == "train" and self.cfg.augment:
                        example_out = apply_augmentation_shim(example_out)
                    if self.cfg.image_shape == list(context_images.shape[2:]):
                        yield example_out
                    else:
                        if self.stage == "train" and self.cfg.random_crop:
                            crop_h = random.randint(self.cfg.min_size[0], self.cfg.max_size[0] + 1) // 64 * 64
                            crop_w = random.randint(self.cfg.min_size[1], self.cfg.max_size[1] + 1) // 64 * 64
                            crop_size = (crop_h, crop_w)
                            yield apply_crop_shim(example_out, crop_size)
                            
                        else:
                            yield apply_crop_shim(example_out, tuple(self.cfg.image_shape))

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style C2W matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32),
                     "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)

        if self.cfg.opencv_pose_format:
            return self.opengl_to_opencv(w2c.inverse()), intrinsics
        else:
            return w2c.inverse(), intrinsics
    
    def opengl_to_opencv(self, c2w):
        # https://github.com/DL3DV-10K/Dataset/issues/4#issuecomment-2019441741
        blender2opencv = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        blender2opencv = torch.tensor(blender2opencv, dtype=c2w.dtype, device=c2w.device).unsqueeze(0)
        c2w = torch.matmul(c2w, blender2opencv)
        c2w[:, 2, :] *= -1
        c2w = c2w[:, torch.tensor(np.array([1, 0, 2, 3])), :]
        c2w[:, 0:3, 1:3] *= -1

        return c2w

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def load_boundary_maps(
        self,
        scene: str,
        indices: Tensor,
        expected_shape: tuple[int, int],
    ) -> Optional[tuple[Tensor, Tensor]]:
        """Load offline boundary supervision without modifying DL3DV chunks.

        Expected layout is boundary_root/stage/scene/frame_index.npz.
        Each file must contain boundary and confidence arrays.
        """
        roots = self.cfg.boundary_roots
        if not roots:
            raise ValueError(
                "load_boundaries=true requires at least one boundary_roots entry"
            )

        boundaries = []
        confidences = []
        for index in indices.tolist():
            relative_path = (
                Path(self.data_stage) / scene / f"{int(index):06d}.npz"
            )
            path = next(
                (
                    Path(root) / relative_path
                    for root in roots
                    if (Path(root) / relative_path).is_file()
                ),
                None,
            )
            if path is None:
                if self.cfg.boundary_missing_policy == "skip":
                    return None
                searched = ", ".join(
                    str(Path(root) / relative_path) for root in roots
                )
                raise FileNotFoundError(
                    f"Missing boundary sidecar for {scene} frame {index}. "
                    f"Searched: {searched}"
                )

            with np.load(path, allow_pickle=False) as sidecar:
                missing = {"boundary", "confidence"} - set(sidecar.files)
                if missing:
                    raise KeyError(
                        f"{path} is missing arrays: {sorted(missing)}"
                    )
                boundary = self.convert_boundary_map(
                    sidecar["boundary"], path, "boundary"
                )
                confidence = self.convert_boundary_map(
                    sidecar["confidence"], path, "confidence"
                )
            if boundary.shape != confidence.shape:
                raise ValueError(
                    f"Boundary/confidence shape mismatch in {path}: "
                    f"{boundary.shape} vs {confidence.shape}"
                )
            if tuple(boundary.shape[-2:]) != expected_shape:
                raise ValueError(
                    f"Boundary shape mismatch in {path}: expected "
                    f"{expected_shape}, got {tuple(boundary.shape[-2:])}"
                )
            boundaries.append(boundary)
            confidences.append(confidence)

        return torch.stack(boundaries), torch.stack(confidences)

    @staticmethod
    def convert_boundary_map(
        array: np.ndarray,
        path: Path,
        name: str,
    ) -> Tensor:
        if array.ndim == 2:
            array = array[None]
        if array.ndim != 3 or array.shape[0] != 1:
            raise ValueError(
                f"{path}:{name} must have shape [H,W] or [1,H,W], "
                f"got {array.shape}"
            )
        is_integer = np.issubdtype(array.dtype, np.integer)
        tensor = torch.from_numpy(np.array(array, copy=True)).float()
        if is_integer:
            tensor /= float(np.iinfo(array.dtype).max)
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{path}:{name} contains non-finite values")
        if tensor.min() < 0 or tensor.max() > 1:
            raise ValueError(f"{path}:{name} must lie in [0, 1]")
        return tensor

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.test_on_train:
            return "train"
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            for i, root in enumerate(self.cfg.roots):
                if not (root / data_stage).is_dir():
                    continue

                # Load the root's index.
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v)
                         for k, v in index.items()}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # mixed data training only evaluate on a single test set
                if self.cfg.no_mix_test_set and data_stage == 'test' and i > 0:
                    continue

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index

    def __len__(self) -> int:
        if self.stage in ['train', 'test']:
            return (
                min(
                    len(self.index.keys()) * self.cfg.test_times_per_scene,
                    self.cfg.test_len,
                )
                if self.stage == "test" and self.cfg.test_len > 0
                else len(self.index.keys()) * self.cfg.train_times_per_scene
            )
        else:
            # set a very large value here to ensure the validation keep going
            # and do not exhaust; it will be wrap to length 1 anyway.
            return int(1e10)


def camera_normalization(pivotal_pose: torch.Tensor, poses: torch.Tensor):
    # [1, 4, 4], [N, 4, 4]

    camera_norm_matrix = torch.inverse(pivotal_pose)
    
    # normalize all views
    poses = torch.bmm(camera_norm_matrix.repeat(poses.shape[0], 1, 1), poses)

    return poses


def center_norm_pose(extrinsics):
    # extrinsics: [V, 4, 4]
    cam_centers = extrinsics[:, :3, 3]  # [V, 3]
    avg_center = cam_centers.mean(dim=0, keepdim=True)  # [1, 3]
    dist = (cam_centers - avg_center).norm(dim=1, keepdim=True)  # [V, 1]
    scale = dist.max()

    # translate
    extrinsics = extrinsics.clone()
    extrinsics[:, :3, 3] -= avg_center
    extrinsics[:, :3, 3] /= scale

    return extrinsics


def get_remaining_indices(context_indices: torch.Tensor, 
                          target_indices: torch.Tensor, 
                          num_remain_context: int) -> torch.Tensor:
    """
    Randomly selects a fixed number of remaining indices in the range [min(context), max(context)],
    excluding those in context or target. Pads by repeating if not enough remain.

    Args:
        context_indices (torch.Tensor): 1D tensor of context indices.
        target_indices (torch.Tensor): 1D tensor of target indices.
        num_remain_context (int): Number of remaining indices to return.

    Returns:
        torch.Tensor: 1D tensor of length `num_remain_context`.
    """
    if context_indices.numel() == 0:
        raise ValueError("context_indices must not be empty.")

    min_idx = torch.min(context_indices).item()
    max_idx = torch.max(context_indices).item()

    full_range = torch.arange(min_idx, max_idx + 1, dtype=torch.long)
    exclude_indices = torch.cat([context_indices, target_indices])
    mask = ~torch.isin(full_range, exclude_indices)

    remaining = full_range[mask]

    if remaining.numel() == 0:
        # Nothing to sample from; repeat the first context index (or any fallback)
        return context_indices[0].repeat(num_remain_context)

    return remaining.sort().values
