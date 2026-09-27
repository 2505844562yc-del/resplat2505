import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import torchvision.transforms as tf
import torch.nn.functional as F
from einops import rearrange, repeat
from jaxtyping import Float, Int64, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
from .dataset_dl3dv import get_remaining_indices


@dataclass
class DatasetRE10kCfg(DatasetCfgCommon):
    name: Literal["re10k"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    test_chunk_interval: int
    average_pose: bool 
    skip_bad_shape: bool = True
    near: float = -1.0
    far: float = -1.0
    baseline_scale_bounds: bool = True
    shuffle_val: bool = True
    train_times_per_scene: int = 1
    highres: bool = False
    scannet: bool = False
    tartanair: bool = False
    use_index_to_load_chunk: Optional[bool] = False
    load_depth: bool = False
    load_semantic: bool = False
    load_instance: bool = False
    pose_align_first_view: bool = False  # align the camera pose to the first view
    center_pose: bool = False  # center and normalize the pose by the distance to the center
    pose_align_middle_view: bool = False  # align the camera pose to the middle view

    scale_extrinsics: float = 1.

    # load remaining context views
    load_remain_context: bool = False


class DatasetRE10k(IterableDataset):
    cfg: DatasetRE10kCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetRE10kCfg,
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

            self.chunks.extend(root_chunks)
        if self.cfg.overfit_to_scene is not None:
            chunk_path = self.index[self.cfg.overfit_to_scene]
            self.chunks = [chunk_path] * len(self.chunks)
        if self.stage == "test":
            # testing on a subset for fast speed
            self.chunks = self.chunks[::cfg.test_chunk_interval]

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
                item = [x for x in chunk if x["key"] == self.cfg.overfit_to_scene]
                assert len(item) == 1
                chunk = item * len(chunk)

            if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
                chunk = self.shuffle(chunk)

            times_per_scene = (
                1
                if self.stage == "test"
                else self.cfg.train_times_per_scene
            )

            for run_idx in range(int(times_per_scene * len(chunk))):
                example = chunk[run_idx // times_per_scene]
                extrinsics, intrinsics = self.convert_poses(example["cameras"])
                scene = example["key"]

                try:
                    context_indices, target_indices = self.view_sampler.sample(
                        scene,
                        extrinsics,
                        intrinsics,
                    )
                except ValueError:
                    # Skip because the example doesn't have enough frames.
                    continue

                # Skip the example if the field of view is too wide.
                if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                    continue

                # load remaining context views
                if self.cfg.load_remain_context:
                    # randomly select fixed number of remaining views such that they can be batched
                    remaining_indices = get_remaining_indices(context_indices, target_indices,
                        0)

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
                context_images = self.convert_images(context_images)
                target_images = [
                    example["images"][index.item()] for index in target_indices
                ]
                target_images = self.convert_images(target_images)

                # Skip the example if the images don't have the right shape.
                if self.cfg.highres:
                    expected_shape = (3, 720, 1280)
                elif self.cfg.scannet or self.cfg.tartanair:
                    expected_shape = (3, 480, 640)
                else:
                    expected_shape = (3, 360, 640)
                context_image_invalid = context_images.shape[1:] != expected_shape
                target_image_invalid = target_images.shape[1:] != expected_shape
                if self.cfg.skip_bad_shape and (context_image_invalid or target_image_invalid):
                    print(
                        f"Skipped bad example {example['key']}. Context shape was "
                        f"{context_images.shape} and target shape was "
                        f"{target_images.shape}."
                    )
                    continue

                if self.cfg.load_remain_context:
                    remain_context_invalid = remain_context_images.shape[1:] != expected_shape

                    if self.cfg.skip_bad_shape and remain_context_invalid:
                        continue

                if self.cfg.average_pose:
                    extrinsics = self.preprocess_poses(extrinsics)

                # load depth
                if self.cfg.load_depth:
                    context_depths = [
                        example["depths"][index.item()] for index in context_indices
                    ]
                    if self.cfg.scannet:
                        context_depths = self.convert_scannet_depths(context_depths)
                    elif self.cfg.tartanair:
                        context_depths = self.convert_tartanair_depths(context_depths)
                    else:
                        raise NotImplementedError

                    target_depths = [
                        example["depths"][index.item()] for index in target_indices
                    ]
                    if self.cfg.scannet:
                        target_depths = self.convert_scannet_depths(target_depths)
                    elif self.cfg.tartanair:
                        target_depths = self.convert_tartanair_depths(target_depths)
                    else:
                        raise NotImplementedError

                if self.cfg.load_semantic:
                    context_semantics = self.convert_label_maps(
                        [example["semantics"][index.item()] for index in context_indices]
                    )
                    target_semantics = self.convert_label_maps(
                        [example["semantics"][index.item()] for index in target_indices]
                    )

                if self.cfg.load_instance:
                    context_instances = self.convert_label_maps(
                        [example["instances"][index.item()] for index in context_indices]
                    )
                    target_instances = self.convert_label_maps(
                        [example["instances"][index.item()] for index in target_indices]
                    )

                # align pose to the first view
                if self.cfg.pose_align_first_view:
                    extrinsics = camera_normalization(extrinsics[context_indices][0:1], extrinsics)

                # align pose to the middle view
                if self.cfg.pose_align_middle_view:
                    mid_index = context_indices.shape[0] // 2
                    extrinsics = camera_normalization(extrinsics[context_indices][mid_index:mid_index+1], extrinsics)

                if self.cfg.center_pose:
                    extrinsics = center_norm_pose(extrinsics)

                # check the extrinsics
                if any(torch.isnan(torch.det(extrinsics[context_indices][:, :3, :3]))):
                    continue

                if any(torch.isnan(torch.det(extrinsics[target_indices][:, :3, :3]))):
                    continue

                # scale the scene when necessary: only scale the extrinsics
                extrinsics[:, :3, 3] *= self.cfg.scale_extrinsics

                example = {
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

                if self.cfg.load_remain_context:
                    example.update({
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

                if self.cfg.load_depth:
                    example['context']['depth'] = context_depths
                    example['target']['depth'] = target_depths

                if self.cfg.load_semantic:
                    example["context"]["semantic"] = context_semantics
                    example["target"]["semantic"] = target_semantics

                if self.cfg.load_instance:
                    example["context"]["instance"] = context_instances
                    example["target"]["instance"] = target_instances

                if self.stage == "train" and self.cfg.augment:
                    example = apply_augmentation_shim(example)
                yield apply_crop_shim(example, tuple(self.cfg.image_shape))

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
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def convert_scannet_depths(
        self,
        depths: list[UInt8[Tensor, "..."]] | list[Tensor],
    ) -> Float[Tensor, "batch height width"]:
        torch_depths = []
        for depth in depths:
            depth = Image.open(BytesIO(depth.numpy().tobytes()))
            # mm to meter depth
            torch_depths.append(self.to_tensor(depth) / 1000.)
        return torch.stack(torch_depths).squeeze(1)

    @staticmethod
    def convert_label_maps(labels: list[Tensor]) -> Int64[Tensor, "batch height width"]:
        """Decode lossless label PNG bytes or accept already-decoded ID tensors."""
        decoded = []
        for label in labels:
            if label.ndim >= 2:
                tensor = label.long()
                if tensor.ndim == 3 and tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)
            else:
                image = Image.open(BytesIO(label.numpy().tobytes()))
                array = np.asarray(image, dtype=np.int64).copy()
                tensor = torch.from_numpy(array)
            if tensor.ndim != 2:
                raise ValueError(f"expected a 2D label map, got {tensor.shape}")
            decoded.append(tensor)
        return torch.stack(decoded)

    def convert_tartanair_depths(
        self,
        depths: list[UInt8[Tensor, "..."]] | list[Tensor],
    ) -> Float[Tensor, "batch height width"]:
        torch_depths = []
        for depth in depths:
            depth = np.load(BytesIO(depth.numpy().tobytes()))
            torch_depths.append(self.to_tensor(depth))
        return torch.stack(torch_depths).squeeze(1)
    
    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
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
                # Load the root's index.
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v) for k, v in index.items()}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index

    def __len__(self) -> int:
        return (
            min(len(self.index.keys()), self.cfg.test_len)
            if self.stage == "test" and self.cfg.test_len > 0
            else len(self.index.keys()) * self.cfg.train_times_per_scene
        )

    def preprocess_poses(
        self,
        in_c2ws: torch.Tensor,
        scene_scale_factor=1.35,
    ):
        """
        Ref: https://github.com/Haian-Jin/LVSM/blob/main/data/dataset_scene.py
        Preprocess the poses to:
        1. translate and rotate the scene to align the average camera direction and position
        2. rescale the whole scene to a fixed scale
        """

        # Translation and Rotation
        # align coordinate system (OpenCV coordinate) to the mean camera
        # center is the average of all camera centers
        # average direction vectors are computed from all camera direction vectors (average down and forward)
        center = in_c2ws[:, :3, 3].mean(0)
        avg_forward = F.normalize(in_c2ws[:, :3, 2].mean(0), dim=-1) # average forward direction (z of opencv camera)
        avg_down = in_c2ws[:, :3, 1].mean(0) # average down direction (y of opencv camera)
        avg_right = F.normalize(torch.cross(avg_down, avg_forward, dim=-1), dim=-1) # (x of opencv camera)
        avg_down = F.normalize(torch.cross(avg_forward, avg_right, dim=-1), dim=-1) # (y of opencv camera)

        avg_pose = torch.eye(4, device=in_c2ws.device) # average c2w matrix
        avg_pose[:3, :3] = torch.stack([avg_right, avg_down, avg_forward], dim=-1)
        avg_pose[:3, 3] = center 
        avg_pose = torch.linalg.inv(avg_pose) # average w2c matrix
        in_c2ws = avg_pose @ in_c2ws 


        # Rescale the whole scene to a fixed scale
        scene_scale = torch.max(torch.abs(in_c2ws[:, :3, 3]))
        scene_scale = scene_scale_factor * scene_scale

        in_c2ws[:, :3, 3] /= scene_scale

        return in_c2ws


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
