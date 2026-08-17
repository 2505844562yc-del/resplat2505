import unittest

import torch

from src.model.gaussian_utils import merge_gaussians
from src.model.types import Gaussians


def make_gaussians(count: int, semantic_dim: int | None = 16) -> Gaussians:
    return Gaussians(
        means=torch.zeros(1, count, 3),
        covariances=torch.eye(3).reshape(1, 1, 3, 3).repeat(1, count, 1, 1),
        harmonics=torch.zeros(1, count, 3, 4),
        opacities=torch.ones(1, count),
        scales=torch.ones(1, count, 3),
        rotations=torch.ones(1, count, 4),
        rotations_unnorm=torch.ones(1, count, 4),
        semantic_features=(
            torch.randn(1, count, semantic_dim)
            if semantic_dim is not None
            else None
        ),
    )


class SemanticGaussianMergeTest(unittest.TestCase):
    def test_merge_preserves_semantic_features(self):
        left = make_gaussians(2)
        right = make_gaussians(3)
        merged = merge_gaussians((left, right))
        self.assertEqual(merged.means.shape, (1, 5, 3))
        self.assertEqual(merged.semantic_features.shape, (1, 5, 16))
        self.assertTrue(
            torch.equal(merged.semantic_features[:, :2], left.semantic_features)
        )
        self.assertTrue(
            torch.equal(merged.semantic_features[:, 2:], right.semantic_features)
        )

    def test_merge_standard_gaussians_keeps_semantics_absent(self):
        merged = merge_gaussians((make_gaussians(1, None), make_gaussians(2, None)))
        self.assertIsNone(merged.semantic_features)

    def test_mixed_semantic_windows_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "semantic_features"):
            merge_gaussians((make_gaussians(1), make_gaussians(1, None)))

    def test_empty_merge_is_rejected(self):
        with self.assertRaises(ValueError):
            merge_gaussians(())

