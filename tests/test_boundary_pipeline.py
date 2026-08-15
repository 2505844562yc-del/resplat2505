import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.dataset.dataset_dl3dv import DatasetDL3DV
from src.dataset.shims.augmentation_shim import reflect_views
from src.dataset.shims.crop_shim import apply_crop_shim_to_views


class BoundaryPipelineTest(unittest.TestCase):
    def make_dataset(self, root: Path, missing_policy: str = "error"):
        dataset = object.__new__(DatasetDL3DV)
        dataset.stage = "test"
        dataset.cfg = SimpleNamespace(
            boundary_roots=[root],
            boundary_missing_policy=missing_policy,
            test_on_train=False,
            overfit_to_scene=None,
        )
        return dataset

    def test_load_normalize_crop_and_reflect(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = "scene-a"
            scene_dir = root / "test" / scene
            scene_dir.mkdir(parents=True)

            boundary = np.zeros((6, 10), dtype=np.uint8)
            boundary[:, 2] = 255
            confidence = np.full((6, 10), 128, dtype=np.uint8)
            for index in (1, 3):
                np.savez_compressed(
                    scene_dir / f"{index:06d}.npz",
                    boundary=boundary,
                    confidence=confidence,
                )

            dataset = self.make_dataset(root)
            boundaries, confidences = dataset.load_boundary_maps(
                scene, torch.tensor([1, 3]), (6, 10)
            )
            self.assertEqual(boundaries.shape, (2, 1, 6, 10))
            self.assertEqual(confidences.shape, (2, 1, 6, 10))
            self.assertEqual(boundaries.min(), 0)
            self.assertEqual(boundaries.max(), 1)
            self.assertTrue(
                torch.allclose(
                    confidences,
                    torch.full_like(confidences, 128 / 255),
                )
            )

            views = {
                "image": torch.zeros(2, 3, 6, 10),
                "intrinsics": torch.eye(3).repeat(2, 1, 1),
                "extrinsics": torch.eye(4).repeat(2, 1, 1),
                "boundary": boundaries,
                "boundary_confidence": confidences,
            }
            cropped = apply_crop_shim_to_views(views, (4, 8))
            self.assertEqual(cropped["boundary"].shape, (2, 1, 4, 8))
            self.assertEqual(
                cropped["boundary_confidence"].shape, (2, 1, 4, 8)
            )
            reflected = reflect_views(cropped)
            self.assertTrue(
                torch.equal(
                    reflected["boundary"], cropped["boundary"].flip(-1)
                )
            )
            self.assertTrue(
                torch.equal(
                    reflected["boundary_confidence"],
                    cropped["boundary_confidence"].flip(-1),
                )
            )

    def test_missing_policy_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.make_dataset(Path(directory), "skip")
            result = dataset.load_boundary_maps(
                "missing-scene", torch.tensor([0]), (6, 10)
            )
            self.assertIsNone(result)

    def test_rejects_wrong_shape(self):
        with self.assertRaisesRegex(ValueError, "must have shape"):
            DatasetDL3DV.convert_boundary_map(
                np.zeros((3, 6, 10), dtype=np.float32),
                Path("bad.npz"),
                "boundary",
            )


if __name__ == "__main__":
    unittest.main()
