import unittest

import torch

from src.model.semantic_boundary import (
    balanced_boundary_l1,
    confidence_gated_boundary_features,
    confidence_gated_boundary_residual,
    local_boundary_alignment_vector,
    rgb_to_soft_boundary,
)


class SemanticBoundaryTest(unittest.TestCase):
    def test_soft_boundary_is_bounded_and_differentiable(self):
        image = torch.zeros(1, 1, 3, 9, 9, requires_grad=True)
        with torch.no_grad():
            image[:, :, :, :, 5:] = 1
        boundary = rgb_to_soft_boundary(image)
        self.assertEqual(boundary.shape, (1, 1, 1, 9, 9))
        self.assertGreaterEqual(float(boundary.min()), 0.0)
        self.assertLessEqual(float(boundary.max()), 1.0)
        self.assertGreater(float(boundary[..., 4:6].mean()), float(boundary[..., :2].mean()))
        boundary.mean().backward()
        self.assertTrue(torch.isfinite(image.grad).all())

    def test_confidence_gate_rejects_uncertain_pixels(self):
        image = torch.zeros(1, 1, 3, 5, 5)
        target = torch.ones(1, 1, 1, 5, 5)
        confidence = torch.full_like(target, 0.2)
        residual = confidence_gated_boundary_residual(
            image, target, confidence, confidence_floor=0.5
        )
        self.assertEqual(float(residual.abs().sum()), 0.0)

    def test_residual_gradient_features_add_direction_channels(self):
        image = torch.zeros(1, 1, 3, 7, 7, requires_grad=True)
        target = torch.zeros(1, 1, 1, 7, 7)
        target[..., :, 4:] = 1.0
        confidence = torch.ones_like(target)
        features = confidence_gated_boundary_features(
            image, target, confidence, mode="residual_gradient"
        )
        self.assertEqual(features.shape, (1, 1, 3, 7, 7))
        self.assertGreater(float(features[..., 1, :, 3:5].abs().sum()), 0.0)
        self.assertEqual(float(features[..., 2, 2:-2, 2:-2].abs().sum()), 0.0)
        features.mean().backward()
        self.assertTrue(torch.isfinite(image.grad).all())

    def test_local_alignment_points_to_nearby_target_boundary(self):
        predicted = torch.zeros(1, 1, 1, 9, 9, requires_grad=True)
        target = torch.zeros_like(predicted)
        with torch.no_grad():
            predicted[..., 4, 2] = 1.0
            target[..., 4, 5] = 1.0
        confidence = torch.ones_like(predicted)
        vector_x, vector_y = local_boundary_alignment_vector(
            predicted, target, confidence, radius=4, sigma=1.0
        )
        self.assertAlmostEqual(float(vector_x[..., 4, 2]), 0.75, places=5)
        self.assertAlmostEqual(float(vector_y[..., 4, 2]), 0.0, places=5)
        self.assertEqual(float(vector_x[..., :2].abs().sum()), 0.0)
        (vector_x.mean() + vector_y.mean()).backward()
        self.assertTrue(torch.isfinite(predicted.grad).all())

    def test_local_alignment_rejects_untrusted_target(self):
        predicted = torch.ones(1, 1, 1, 7, 7)
        target = torch.zeros_like(predicted)
        target[..., 3, 5] = 1.0
        vector_x, vector_y = local_boundary_alignment_vector(
            predicted, target, torch.zeros_like(target), radius=3
        )
        self.assertEqual(float(vector_x.abs().sum() + vector_y.abs().sum()), 0.0)

    def test_boundary_feature_mode_is_validated(self):
        image = torch.zeros(1, 1, 3, 5, 5)
        target = torch.zeros(1, 1, 1, 5, 5)
        with self.assertRaises(ValueError):
            confidence_gated_boundary_features(
                image, target, torch.ones_like(target), mode="unknown"
            )

    def test_balanced_loss_treats_sparse_positive_class_equally(self):
        target = torch.zeros(1, 1, 1, 4, 4)
        target[..., 0, 0] = 1
        weight = torch.ones_like(target)
        perfect = balanced_boundary_l1(target, target, weight)
        empty = balanced_boundary_l1(torch.zeros_like(target), target, weight)
        self.assertEqual(float(perfect), 0.0)
        self.assertAlmostEqual(float(empty), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
