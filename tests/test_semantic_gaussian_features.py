import unittest

import torch

from src.model.semantic_gaussian import (
    SemanticFeatureProjector,
    apply_semantic_state_residual,
)


class SemanticFeatureProjectorTest(unittest.TestCase):
    def test_output_is_normalized_and_resized(self):
        projector = SemanticFeatureProjector(32, 8)
        output = projector(torch.randn(2, 32, 3, 5), (6, 10))
        self.assertEqual(output.shape, (2, 8, 6, 10))
        norms = output.norm(dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_frozen_projection_is_deterministic(self):
        left = SemanticFeatureProjector(32, 8, seed=7)
        right = SemanticFeatureProjector(32, 8, seed=7)
        self.assertTrue(
            torch.equal(left.projection.weight, right.projection.weight)
        )
        self.assertFalse(left.projection.weight.requires_grad)

    def test_trainable_projection_can_be_enabled(self):
        projector = SemanticFeatureProjector(32, 8, trainable=True)
        self.assertTrue(projector.projection.weight.requires_grad)

    def test_invalid_feature_channels_are_rejected(self):
        projector = SemanticFeatureProjector(32, 8)
        with self.assertRaises(ValueError):
            projector(torch.randn(1, 16, 3, 3), (3, 3))


class SemanticStateResidualTest(unittest.TestCase):
    def test_residual_is_bounded_and_normalized(self):
        features = torch.nn.functional.normalize(torch.randn(1, 4, 8), dim=-1)
        residual = torch.full_like(features, 1000.0)
        output = apply_semantic_state_residual(features, residual, gain=0.1)
        self.assertTrue(
            torch.allclose(output.norm(dim=-1), torch.ones(1, 4), atol=1e-6)
        )
        self.assertTrue(torch.isfinite(output).all())

    def test_zero_gain_preserves_normalized_features(self):
        features = torch.nn.functional.normalize(torch.randn(1, 4, 8), dim=-1)
        output = apply_semantic_state_residual(
            features, torch.randn_like(features), gain=0.0
        )
        self.assertTrue(torch.allclose(output, features, atol=1e-6))

    def test_invalid_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            apply_semantic_state_residual(
                torch.randn(1, 4, 8), torch.randn(1, 4, 4)
            )
