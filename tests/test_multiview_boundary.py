import unittest

import torch

from src.model.multiview_boundary import (
    multiview_boundary_consensus_confidence,
)


class MultiViewBoundaryConsensusTest(unittest.TestCase):
    def setUp(self):
        self.boundary = torch.zeros(1, 2, 1, 7, 7)
        self.confidence = torch.ones_like(self.boundary)
        self.depth = torch.ones(1, 2, 7, 7)
        self.extrinsics = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1)
        self.intrinsics = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1)

    def test_identical_views_support_same_boundary(self):
        self.boundary[:, :, :, 3, 3] = 1
        effective, consensus = multiview_boundary_consensus_confidence(
            self.boundary,
            self.confidence,
            self.depth,
            self.extrinsics,
            self.intrinsics,
            radius=0,
            min_support_views=1,
        )
        self.assertGreater(float(consensus[0, 0, 0, 3, 3]), 0.99)
        self.assertGreater(float(effective[0, 0, 0, 3, 3]), 0.99)

    def test_unsupported_positive_boundary_is_suppressed(self):
        self.boundary[:, 0, :, 3, 3] = 1
        effective, consensus = multiview_boundary_consensus_confidence(
            self.boundary,
            self.confidence,
            self.depth,
            self.extrinsics,
            self.intrinsics,
            radius=0,
            min_support_views=1,
        )
        self.assertEqual(float(consensus[0, 0, 0, 3, 3]), 0.0)
        self.assertEqual(float(effective[0, 0, 0, 3, 3]), 0.0)
        self.assertEqual(float(effective[0, 0, 0, 0, 0]), 1.0)

    def test_depth_disagreement_rejects_boundary_support(self):
        self.boundary[:, :, :, 3, 3] = 1
        self.depth[:, 1] = 2
        _, consensus = multiview_boundary_consensus_confidence(
            self.boundary,
            self.confidence,
            self.depth,
            self.extrinsics,
            self.intrinsics,
            radius=0,
            depth_relative_tolerance=0.01,
            min_support_views=1,
        )
        self.assertLess(float(consensus[0, 0, 0, 3, 3]), 1e-6)

    def test_consensus_redistributes_positive_confidence(self):
        self.boundary[:, :, :, 3, 3] = 1
        self.boundary[:, 0, :, 1, 1] = 1
        effective, _ = multiview_boundary_consensus_confidence(
            self.boundary,
            self.confidence,
            self.depth,
            self.extrinsics,
            self.intrinsics,
            radius=0,
            min_support_views=1,
            blend=0.5,
        )
        self.assertGreater(
            float(effective[0, 0, 0, 3, 3]),
            float(effective[0, 0, 0, 1, 1]),
        )

    def test_single_view_is_a_noop(self):
        effective, consensus = multiview_boundary_consensus_confidence(
            self.boundary[:, :1],
            self.confidence[:, :1],
            self.depth[:, :1],
            self.extrinsics[:, :1],
            self.intrinsics[:, :1],
        )
        self.assertTrue(torch.equal(effective, self.confidence[:, :1]))
        self.assertTrue(torch.equal(consensus, torch.ones_like(consensus)))

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            multiview_boundary_consensus_confidence(
                self.boundary,
                self.confidence,
                self.depth,
                self.extrinsics,
                self.intrinsics,
                depth_relative_tolerance=0,
            )


if __name__ == "__main__":
    unittest.main()
