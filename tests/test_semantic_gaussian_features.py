import unittest

import torch

from src.model.semantic_gaussian import (
    SemanticFeatureProjector,
    apply_semantic_state_residual,
    semantic_render_residual_features,
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


class SemanticRenderResidualFeaturesTest(unittest.TestCase):
    def test_identical_features_have_zero_residual(self):
        teacher = torch.nn.functional.normalize(
            torch.randn(1, 2, 8, 3, 4), dim=2
        )
        output = semantic_render_residual_features(
            teacher, teacher, torch.ones(1, 2, 3, 4)
        )
        self.assertEqual(output.shape, (1, 2, 10, 3, 4))
        self.assertTrue(torch.allclose(output[:, :, :9], torch.zeros_like(output[:, :, :9]), atol=1e-6))
        self.assertTrue(torch.allclose(output[:, :, -1], torch.ones(1, 2, 3, 4)))

    def test_low_alpha_suppresses_feedback(self):
        rendered = torch.randn(1, 1, 4, 2, 2)
        teacher = torch.randn_like(rendered)
        output = semantic_render_residual_features(
            rendered, teacher, torch.full((1, 1, 2, 2), 0.05), alpha_floor=0.1
        )
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            semantic_render_residual_features(
                torch.randn(1, 1, 4, 2, 2),
                torch.randn(1, 1, 4, 2, 3),
                torch.ones(1, 1, 2, 2),
            )
        with self.assertRaises(ValueError):
            semantic_render_residual_features(
                torch.randn(1, 1, 4, 2, 2),
                torch.randn(1, 1, 4, 2, 2),
                torch.ones(1, 1, 2, 2),
                alpha_floor=1.0,
            )
