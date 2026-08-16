import unittest

import torch

from src.model.multiview_boundary import (
    boundary_consensus_reliability_feature,
    multiview_boundary_displacement_candidates,
    semantic_boundary_source_mask,
    multiview_boundary_consensus_confidence,
    signed_boundary_consensus_feature,
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

    def test_signed_consensus_feature_separates_support(self):
        boundary = torch.tensor([[[[[1.0, 1.0, 0.0]]]]])
        confidence = torch.tensor([[[[[0.8, 0.6, 1.0]]]]])
        consensus = torch.tensor([[[[[1.0, 0.0, 1.0]]]]])
        feature = signed_boundary_consensus_feature(
            boundary, confidence, consensus
        )
        expected = torch.tensor([[[[[0.8, -0.6, 0.0]]]]])
        self.assertTrue(torch.allclose(feature, expected))

    def test_signed_consensus_feature_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            signed_boundary_consensus_feature(
                self.boundary, self.confidence[..., :-1], self.boundary
            )

    def test_reliability_feature_is_non_negative_and_boundary_masked(self):
        boundary = torch.tensor([[[[[1.0, 1.0, 0.0]]]]])
        confidence = torch.tensor([[[[[0.8, 0.6, 1.0]]]]])
        consensus = torch.tensor([[[[[1.0, 0.0, 1.0]]]]])
        feature = boundary_consensus_reliability_feature(
            boundary, confidence, consensus
        )
        expected = torch.tensor([[[[[0.8, 0.0, 0.0]]]]])
        self.assertTrue(torch.allclose(feature, expected))

    def test_displacement_is_zero_for_aligned_boundaries(self):
        rendered = torch.zeros_like(self.boundary)
        rendered[:, :, :, 3, 3] = 1
        self.boundary[:, :, :, 3, 3] = 1
        dx, dy, confidence, visibility = (
            multiview_boundary_displacement_candidates(
                rendered,
                self.boundary,
                self.confidence,
                self.depth,
                self.extrinsics,
                self.intrinsics,
                radius=2,
            )
        )
        self.assertAlmostEqual(float(dx[0, 0, 0, 3, 3]), 0.0, places=5)
        self.assertAlmostEqual(float(dy[0, 0, 0, 3, 3]), 0.0, places=5)
        self.assertGreater(float(confidence[0, 0, 0, 3, 3]), 0.99)
        self.assertGreater(float(visibility[0, 0, 0, 3, 3]), 0.99)

    def test_displacement_points_to_shifted_target_boundary(self):
        rendered = torch.zeros_like(self.boundary)
        rendered[:, 0, :, 3, 3] = 1
        self.boundary[:, 1, :, 3, 4] = 1
        dx, dy, confidence, _ = multiview_boundary_displacement_candidates(
            rendered,
            self.boundary,
            self.confidence,
            self.depth,
            self.extrinsics,
            self.intrinsics,
            radius=2,
        )
        self.assertAlmostEqual(float(dx[0, 0, 0, 3, 3]), 1.0, places=4)
        self.assertAlmostEqual(float(dy[0, 0, 0, 3, 3]), 0.0, places=4)
        self.assertGreater(float(confidence[0, 0, 0, 3, 3]), 0.99)

    def test_displacement_rejects_depth_disagreement(self):
        rendered = torch.zeros_like(self.boundary)
        rendered[:, 0, :, 3, 3] = 1
        self.boundary[:, 1, :, 3, 4] = 1
        self.depth[:, 1] = 2
        dx, _, confidence, visibility = multiview_boundary_displacement_candidates(
            rendered,
            self.boundary,
            self.confidence,
            self.depth,
            self.extrinsics,
            self.intrinsics,
            radius=2,
            depth_relative_tolerance=0.01,
        )
        self.assertLess(float(confidence[0, 0, 0, 3, 3]), 1e-6)
        self.assertLess(float(dx[0, 0, 0, 3, 3]), 1e-6)
        self.assertGreater(float(visibility[0, 0, 0, 3, 3]), 0.99)

    def test_displacement_single_view_returns_zeros(self):
        outputs = multiview_boundary_displacement_candidates(
            self.boundary[:, :1],
            self.boundary[:, :1],
            self.confidence[:, :1],
            self.depth[:, :1],
            self.extrinsics[:, :1],
            self.intrinsics[:, :1],
        )
        self.assertTrue(all(torch.count_nonzero(item) == 0 for item in outputs))

    def test_opposite_displacements_reduce_directional_confidence(self):
        rendered = torch.zeros(1, 3, 1, 7, 7)
        rendered[:, 0, :, 3, 3] = 1
        teacher = torch.zeros_like(rendered)
        teacher[:, 1, :, 3, 4] = 1
        teacher[:, 2, :, 3, 2] = 1
        confidence = torch.ones_like(rendered)
        depth = torch.ones(1, 3, 7, 7)
        extrinsics = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 3, 1, 1)
        intrinsics = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 3, 1, 1)
        dx, dy, agreement, _ = multiview_boundary_displacement_candidates(
            rendered,
            teacher,
            confidence,
            depth,
            extrinsics,
            intrinsics,
            radius=2,
        )
        self.assertAlmostEqual(float(dx[0, 0, 0, 3, 3]), 0.0, places=4)
        self.assertAlmostEqual(float(dy[0, 0, 0, 3, 3]), 0.0, places=4)
        self.assertLess(float(agreement[0, 0, 0, 3, 3]), 0.01)

    def test_source_mask_rejects_rendered_edges_far_from_semantics(self):
        rendered = torch.zeros_like(self.boundary)
        rendered[:, :, :, 1, 1] = 0.8
        rendered[:, :, :, 3, 4] = 0.7
        teacher = torch.zeros_like(self.boundary)
        teacher[:, :, :, 3, 3] = 1
        masked = semantic_boundary_source_mask(
            rendered, teacher, self.confidence, radius=1
        )
        self.assertEqual(float(masked[0, 0, 0, 1, 1]), 0.0)
        self.assertAlmostEqual(float(masked[0, 0, 0, 3, 4]), 0.7, places=5)


if __name__ == "__main__":
    unittest.main()
