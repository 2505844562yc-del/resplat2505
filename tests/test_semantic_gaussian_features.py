import unittest

import torch

from src.model.semantic_gaussian import (
    SemanticFeatureProjector,
    apply_semantic_geometry_vjp,
    apply_semantic_state_residual,
    confidence_gate_semantic_gradient_feedback,
    project_geometry_feedback_to_rays,
    semantic_gradient_feedback_features,
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


class SemanticGradientFeedbackFeaturesTest(unittest.TestCase):
    def test_gradient_becomes_bounded_descent_direction(self):
        gradient = torch.tensor([[[3.0, 4.0], [0.0, 0.0]]])
        output = semantic_gradient_feedback_features(gradient, relative_scale=1.0)
        self.assertEqual(output.shape, (1, 2, 4))
        self.assertTrue(torch.all(output[..., -2] >= 0))
        self.assertTrue(torch.all(output[..., -2] <= 1))
        self.assertTrue(torch.allclose(output[0, 0, :2], torch.tensor([-0.6, -0.8])))
        self.assertTrue(torch.equal(output[0, 1], torch.zeros(4)))

    def test_global_gradient_scale_does_not_change_feedback(self):
        gradient = torch.randn(2, 7, 5)
        first = semantic_gradient_feedback_features(gradient)
        second = semantic_gradient_feedback_features(gradient * 1000)
        self.assertTrue(torch.allclose(first, second, atol=1e-6))

    def test_invalid_gradient_is_rejected(self):
        with self.assertRaises(ValueError):
            semantic_gradient_feedback_features(torch.randn(2, 3, 4, 5))


class SemanticGeometryVJPTest(unittest.TestCase):
    def test_zero_gains_are_exact_identity(self):
        means = torch.randn(1, 5, 3)
        scales = torch.rand(1, 5, 3) + 0.1
        updated_means, updated_scales = apply_semantic_geometry_vjp(
            means,
            scales,
            torch.randn_like(means),
            torch.randn_like(scales),
        )
        self.assertTrue(torch.equal(updated_means, means))
        self.assertTrue(torch.equal(updated_scales, scales))

    def test_mean_step_is_bounded_by_local_scale(self):
        means = torch.zeros(1, 2, 3)
        scales = torch.tensor([[[2.0, 2.0, 2.0], [1.0, 2.0, 3.0]]])
        updated_means, _ = apply_semantic_geometry_vjp(
            means,
            scales,
            torch.full_like(means, 1000.0),
            None,
            mean_gain=0.1,
        )
        self.assertTrue(torch.all(updated_means.abs() <= 0.2 + 1e-6))

    def test_scale_step_is_multiplicatively_bounded(self):
        means = torch.zeros(1, 2, 3)
        scales = torch.ones_like(means)
        _, updated_scales = apply_semantic_geometry_vjp(
            means,
            scales,
            None,
            torch.full_like(scales, 1000.0),
            scale_gain=0.2,
        )
        self.assertTrue(
            torch.allclose(updated_scales, torch.full_like(scales, torch.exp(torch.tensor(0.2))))
        )

    def test_invalid_feedback_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_semantic_geometry_vjp(
                torch.randn(1, 4, 3),
                torch.rand(1, 4, 3),
                torch.randn(1, 4, 2),
                None,
                mean_gain=0.1,
            )


class SemanticGeometryConfidenceGateTest(unittest.TestCase):
    def test_zero_floor_preserves_directional_feedback(self):
        gradient = torch.randn(1, 8, 3)
        feedback = semantic_gradient_feedback_features(gradient)
        gated = confidence_gate_semantic_gradient_feedback(feedback, 0.0)
        self.assertTrue(torch.allclose(gated, feedback[..., :-2], atol=1e-6))

    def test_below_floor_is_suppressed(self):
        feedback = torch.tensor([[[0.1, 0.0, 0.0, 0.1, 1.0]]])
        gated = confidence_gate_semantic_gradient_feedback(feedback, 0.25)
        self.assertTrue(torch.equal(gated, torch.zeros_like(gated)))

    def test_invalid_floor_is_rejected(self):
        with self.assertRaises(ValueError):
            confidence_gate_semantic_gradient_feedback(
                torch.randn(1, 2, 5), 1.0
            )


class SemanticGeometryRayProjectionTest(unittest.TestCase):
    def test_removes_tangential_feedback(self):
        feedback = torch.tensor([[[2.0, 3.0, 4.0]]])
        rays = torch.tensor([[[1.0, 0.0, 0.0]]])
        projected = project_geometry_feedback_to_rays(feedback, rays)
        self.assertTrue(
            torch.equal(projected, torch.tensor([[[2.0, 0.0, 0.0]]]))
        )

    def test_projection_is_invariant_to_ray_length(self):
        feedback = torch.randn(2, 6, 3)
        rays = torch.randn_like(feedback)
        first = project_geometry_feedback_to_rays(feedback, rays)
        second = project_geometry_feedback_to_rays(feedback, rays * 100)
        self.assertTrue(torch.allclose(first, second, atol=1e-6))

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            project_geometry_feedback_to_rays(
                torch.randn(1, 4, 3), torch.randn(1, 4, 2)
            )
