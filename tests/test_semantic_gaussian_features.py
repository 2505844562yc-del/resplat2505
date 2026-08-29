import unittest

import torch

from src.model.semantic_gaussian import (
    SemanticFeatureProjector,
    apply_fixed_candidate_semantic_split,
    apply_learned_ray_depth_residual,
    apply_priority_gated_ray_depth_residual,
    apply_semantic_geometry_vjp,
    apply_semantic_state_residual,
    apply_semantic_support_residual,
    apply_semantic_updater_state_residual,
    apply_semantic_uncertainty_residual,
    confidence_gate_semantic_gradient_feedback,
    normalize_gaussian_visibility_support,
    project_geometry_feedback_to_rays,
    semantic_gradient_feedback_features,
    semantic_depth_parameter_trainable,
    semantic_depth_last_layers_trainable,
    semantic_joint_parameter_trainable,
    semantic_split_parameter_trainable,
    semantic_updater_adapter_parameter_trainable,
    semantic_updater_last_block_parameter_trainable,
    select_fixed_semantic_candidates,
    semantic_render_residual_features,
    semantic_uncertainty_components,
)


class SemanticJointParameterSelectionTest(unittest.TestCase):
    def test_all_promoted_heads_are_selected(self):
        for name in (
            "encoder.depth_predictor.semantic_depth_adapters.0.residual_head.weight",
            "encoder.depth_predictor.semantic_depth_feature_adapters.0.residual_head.weight",
            "encoder.depth_predictor.semantic_depth_residual_injections.0.gamma",
            "encoder.depth_predictor.semantic_depth_concat_projections.0.main_projection.weight",
            "encoder.semantic_state_head.0.weight",
            "encoder.semantic_uncertainty_head.2.bias",
            "encoder.semantic_support_head.0.weight",
            "encoder.semantic_ray_depth_head.2.weight",
            "encoder.semantic_updater_adapter.2.weight",
            "encoder.semantic_split_head.2.weight",
        ):
            self.assertTrue(semantic_joint_parameter_trainable(name))

    def test_semantic_depth_selection_is_narrow(self):
        self.assertTrue(
            semantic_depth_parameter_trainable(
                "encoder.depth_predictor.semantic_depth_adapters.0.gate_head.weight"
            )
        )
        self.assertTrue(
            semantic_depth_parameter_trainable(
                "encoder.depth_predictor.semantic_depth_feature_adapters.0.gate_head.weight"
            )
        )
        self.assertTrue(
            semantic_depth_parameter_trainable(
                "encoder.depth_predictor.semantic_depth_residual_injections.0.gamma"
            )
        )
        self.assertTrue(
            semantic_depth_parameter_trainable(
                "encoder.depth_predictor.semantic_depth_concat_projections.0.main_projection.weight"
            )
        )
        self.assertTrue(
            semantic_depth_parameter_trainable(
                "encoder.semantic_feature_projector.projection.weight"
            )
        )
        self.assertFalse(
            semantic_depth_parameter_trainable(
                "encoder.depth_predictor.depth_head.0.2.weight"
            )
        )

    def test_situation_a_staged_unfreeze_selects_only_depth_tail(self):
        for name in (
            "encoder.depth_predictor.regressor.0.3.out.2.weight",
            "encoder.depth_predictor.regressor.0.4.weight",
            "encoder.depth_predictor.depth_head.0.2.weight",
        ):
            self.assertTrue(semantic_depth_last_layers_trainable(name))
        for name in (
            "encoder.depth_predictor.regressor.0.3.input_blocks.0.0.weight",
            "encoder.depth_predictor.pretrained.blocks.11.weight",
            "encoder.depth_predictor.transformer.layers.0.weight",
            "encoder.update_head.0.weight",
        ):
            self.assertFalse(semantic_depth_last_layers_trainable(name))

    def test_backbone_and_projector_remain_frozen(self):
        for name in (
            "encoder.update_head.0.weight",
            "encoder.depth_predictor.transformer.weight",
            "encoder.semantic_feature_projector.projection.weight",
            "decoder.some_parameter",
        ):
            self.assertFalse(semantic_joint_parameter_trainable(name))

    def test_adapter_only_selection_is_narrow(self):
        self.assertTrue(
            semantic_updater_adapter_parameter_trainable(
                "encoder.semantic_updater_adapter.2.weight"
            )
        )
        self.assertFalse(
            semantic_updater_adapter_parameter_trainable(
                "encoder.semantic_state_head.0.weight"
            )
        )

    def test_last_updater_block_selection_is_exact(self):
        self.assertTrue(
            semantic_updater_last_block_parameter_trainable(
                "encoder.update_module.1.blocks.3.mlp.fc2.weight", 4
            )
        )
        self.assertFalse(
            semantic_updater_last_block_parameter_trainable(
                "encoder.update_module.1.blocks.2.mlp.fc2.weight", 4
            )
        )
        self.assertFalse(
            semantic_updater_last_block_parameter_trainable(
                "encoder.update_head.6.weight", 4
            )
        )
        with self.assertRaises(ValueError):
            semantic_updater_last_block_parameter_trainable("anything", 0)

    def test_split_only_selection_is_narrow(self):
        self.assertTrue(
            semantic_split_parameter_trainable(
                "encoder.semantic_split_head.2.weight"
            )
        )
        self.assertFalse(
            semantic_split_parameter_trainable(
                "encoder.semantic_updater_adapter.2.weight"
            )
        )


class FixedCandidateSemanticSplitTest(unittest.TestCase):
    def _inputs(self):
        means = torch.tensor(
            [[[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0],
              [0.0, 0.0, 4.0]]]
        )
        scales = torch.full_like(means, 0.2)
        opacities = torch.tensor([[0.2, 0.4, 0.6, 0.8]])
        raw = torch.zeros(1, 4, 4)
        priority = torch.tensor([[[0.1], [0.9], [0.7], [0.0]]])
        return means, scales, opacities, means.clone(), raw, priority

    def test_fixed_budget_selects_highest_priority(self):
        *_, priority = self._inputs()
        indices = select_fixed_semantic_candidates(priority, 0.5)
        self.assertEqual(indices.shape, (1, 2))
        self.assertEqual(set(indices[0].tolist()), {1, 2})

    def test_zero_residual_preserves_geometry_and_combined_alpha(self):
        means, scales, opacities, rays, raw, priority = self._inputs()
        indices = select_fixed_semantic_candidates(priority, 0.5)
        (
            parent_opacity,
            child_means,
            child_scales,
            child_opacity,
            depth_delta,
            log_scale_delta,
            _,
        ) = apply_fixed_candidate_semantic_split(
            means, scales, opacities, rays, raw, priority, indices
        )
        selected_means = means.gather(
            1, indices[..., None].expand(-1, -1, 3)
        )
        selected_scales = scales.gather(
            1, indices[..., None].expand(-1, -1, 3)
        )
        selected_original_opacity = opacities.gather(1, indices)
        selected_parent_opacity = parent_opacity.gather(1, indices)
        combined_alpha = 1 - (
            (1 - selected_parent_opacity) * (1 - child_opacity)
        )
        self.assertTrue(torch.equal(child_means, selected_means))
        self.assertTrue(torch.equal(child_scales, selected_scales))
        self.assertTrue(torch.equal(depth_delta, torch.zeros_like(depth_delta)))
        self.assertTrue(
            torch.equal(log_scale_delta, torch.zeros_like(log_scale_delta))
        )
        self.assertTrue(
            torch.allclose(combined_alpha, selected_original_opacity, atol=1e-6)
        )

    def test_updates_are_priority_gated_and_bounded(self):
        means, scales, opacities, rays, raw, priority = self._inputs()
        raw[..., 0] = 1000
        raw[..., 1:] = -1000
        indices = select_fixed_semantic_candidates(priority, 0.5)
        result = apply_fixed_candidate_semantic_split(
            means,
            scales,
            opacities,
            rays,
            raw,
            priority,
            indices,
            depth_gain=0.1,
            scale_gain=0.05,
        )
        depth_delta, log_scale_delta = result[4], result[5]
        selected_priority = priority.squeeze(-1).gather(1, indices)
        local_scale = scales.mean(dim=-1).gather(1, indices)
        self.assertTrue(
            torch.all(
                depth_delta.squeeze(-1).abs()
                <= 0.1 * selected_priority * local_scale + 1e-6
            )
        )
        self.assertTrue(
            torch.all(
                log_scale_delta.abs()
                <= 0.05 * selected_priority[..., None] + 1e-6
            )
        )

    def test_invalid_configuration_is_rejected(self):
        *_, priority = self._inputs()
        with self.assertRaises(ValueError):
            select_fixed_semantic_candidates(priority.squeeze(-1), 0.25)
        with self.assertRaises(ValueError):
            select_fixed_semantic_candidates(priority, 0.0)


class SemanticUpdaterStateResidualTest(unittest.TestCase):
    def test_zero_residual_is_exact_identity(self):
        state = torch.randn(5, 8)
        output, applied = apply_semantic_updater_state_residual(
            state, torch.zeros_like(state), torch.ones(5, 1), gain=0.1
        )
        self.assertTrue(torch.equal(output, state))
        self.assertTrue(torch.equal(applied, torch.zeros_like(applied)))

    def test_priority_gates_and_bounds_update(self):
        state = torch.zeros(3, 4)
        raw = torch.full_like(state, 1000.0)
        priority = torch.tensor([[0.0], [0.5], [1.0]])
        output, applied = apply_semantic_updater_state_residual(
            state, raw, priority, gain=0.2
        )
        self.assertTrue(torch.equal(output[0], state[0]))
        self.assertTrue(torch.allclose(applied[1], torch.full((4,), 0.1)))
        self.assertTrue(torch.allclose(applied[2], torch.full((4,), 0.2)))
        self.assertLessEqual(applied.abs().max().item(), 0.2 + 1e-6)

    def test_invalid_shapes_and_gain_are_rejected(self):
        state = torch.zeros(3, 4)
        with self.assertRaises(ValueError):
            apply_semantic_updater_state_residual(
                state, torch.zeros(3, 3), torch.ones(3, 1)
            )
        with self.assertRaises(ValueError):
            apply_semantic_updater_state_residual(
                state, torch.zeros_like(state), torch.ones(3), gain=0.1
            )
        with self.assertRaises(ValueError):
            apply_semantic_updater_state_residual(
                state, torch.zeros_like(state), torch.ones(3, 1), gain=-0.1
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


class LearnedRayDepthResidualTest(unittest.TestCase):
    def test_zero_raw_residual_is_exact_identity(self):
        means = torch.randn(1, 5, 3)
        scales = torch.rand_like(means) + 0.1
        updated, delta = apply_learned_ray_depth_residual(
            means,
            scales,
            torch.randn_like(means),
            torch.zeros(1, 5, 1),
            torch.ones(1, 5, 1),
            gain=0.2,
        )
        self.assertTrue(torch.equal(updated, means))
        self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))

    def test_displacement_is_bounded_and_parallel_to_ray(self):
        means = torch.zeros(1, 1, 3)
        scales = torch.full_like(means, 2.0)
        updated, delta = apply_learned_ray_depth_residual(
            means,
            scales,
            torch.tensor([[[0.0, 0.0, 10.0]]]),
            torch.tensor([[[1000.0]]]),
            torch.ones(1, 1, 1),
            gain=0.1,
        )
        self.assertTrue(torch.allclose(delta, torch.tensor([[[0.2]]])))
        self.assertTrue(
            torch.allclose(updated, torch.tensor([[[0.0, 0.0, 0.2]]]))
        )

    def test_low_confidence_is_suppressed(self):
        means = torch.randn(1, 2, 3)
        updated, _ = apply_learned_ray_depth_residual(
            means,
            torch.ones_like(means),
            torch.randn_like(means),
            torch.ones(1, 2, 1),
            torch.full((1, 2, 1), 0.1),
            gain=0.2,
            confidence_floor=0.25,
        )
        self.assertTrue(torch.equal(updated, means))


class PriorityGatedRayDepthResidualTest(unittest.TestCase):
    def test_zero_raw_residual_is_exact_identity(self):
        means = torch.randn(1, 5, 3)
        updated, delta = apply_priority_gated_ray_depth_residual(
            means,
            torch.rand_like(means) + 0.1,
            torch.randn_like(means),
            torch.zeros(1, 5, 1),
            torch.ones(1, 5, 1),
            gain=0.2,
        )
        self.assertTrue(torch.equal(updated, means))
        self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))

    def test_displacement_is_bounded_parallel_and_priority_gated(self):
        means = torch.zeros(1, 2, 3)
        scales = torch.full_like(means, 2.0)
        updated, delta = apply_priority_gated_ray_depth_residual(
            means,
            scales,
            torch.tensor([[[0.0, 0.0, 10.0], [10.0, 0.0, 0.0]]]),
            torch.full((1, 2, 1), 1000.0),
            torch.tensor([[[0.5], [0.0]]]),
            gain=0.1,
        )
        self.assertTrue(torch.allclose(delta[0, 0], torch.tensor([0.1])))
        self.assertTrue(torch.equal(delta[0, 1], torch.zeros(1)))
        self.assertTrue(
            torch.allclose(updated[0, 0], torch.tensor([0.0, 0.0, 0.1]))
        )
        self.assertTrue(torch.equal(updated[0, 1], means[0, 1]))

    def test_invalid_priority_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_priority_gated_ray_depth_residual(
                torch.zeros(1, 3, 3),
                torch.ones(1, 3, 3),
                torch.ones(1, 3, 3),
                torch.zeros(1, 3, 1),
                torch.ones(1, 3),
            )


class SemanticUncertaintyTest(unittest.TestCase):
    def test_support_normalization_is_scale_invariant(self):
        support = torch.tensor([[1.0, 2.0, 4.0]])
        first = normalize_gaussian_visibility_support(support)
        second = normalize_gaussian_visibility_support(support * 1000)
        self.assertEqual(first.shape, (1, 3, 1))
        self.assertTrue(torch.allclose(first, second, atol=1e-6))

    def test_need_and_reliability_are_separate(self):
        # Directional channels, relative magnitude, active.
        feedback = torch.tensor(
            [[[0.0, 0.0, 0.10, 1.0], [0.0, 0.0, 0.70, 1.0]]]
        )
        support = torch.tensor([[[1.0], [0.0]]])
        need, reliable, uncertainty, priority = semantic_uncertainty_components(
            feedback,
            support,
            need_floor=0.25,
            reliability_floor=0.05,
        )
        self.assertEqual(need[0, 0, 0].item(), 0.0)
        self.assertGreater(need[0, 1, 0].item(), 0.0)
        self.assertGreater(reliable[0, 0, 0].item(), 0.0)
        self.assertEqual(reliable[0, 1, 0].item(), 0.0)
        self.assertTrue(torch.equal(priority, torch.zeros_like(priority)))
        self.assertGreater(uncertainty[0, 1, 0].item(), 0.99)

    def test_zero_learned_residual_preserves_deterministic_uncertainty(self):
        base = torch.rand(2, 7, 1)
        output = apply_semantic_uncertainty_residual(
            base,
            torch.zeros_like(base),
            torch.rand_like(base),
            gain=0.25,
        )
        self.assertTrue(torch.equal(output, base))

    def test_uncertainty_calibration_is_bounded_and_reliability_gated(self):
        base = torch.tensor([[[0.5], [0.5]]])
        output = apply_semantic_uncertainty_residual(
            base,
            torch.tensor([[[1000.0], [-1000.0]]]),
            torch.tensor([[[1.0], [0.0]]]),
            gain=0.25,
        )
        self.assertTrue(torch.all((0 <= output) & (output <= 1)))
        self.assertTrue(torch.allclose(output[0, 0], torch.tensor([0.75])))
        self.assertTrue(torch.equal(output[0, 1], base[0, 1]))


class SemanticSupportResidualTest(unittest.TestCase):
    def test_zero_raw_residual_is_exact_identity(self):
        scales = torch.rand(1, 5, 3) + 0.1
        logits = torch.randn(1, 5, 1)
        updated_scales, updated_logits, scale_delta, opacity_delta = (
            apply_semantic_support_residual(
                scales,
                logits,
                torch.zeros(1, 5, 4),
                torch.ones(1, 5, 1),
            )
        )
        self.assertTrue(torch.equal(updated_scales, scales))
        self.assertTrue(torch.equal(updated_logits, logits))
        self.assertTrue(torch.equal(scale_delta, torch.zeros_like(scale_delta)))
        self.assertTrue(torch.equal(opacity_delta, torch.zeros_like(opacity_delta)))

    def test_updates_are_bounded_and_priority_gated(self):
        scales = torch.ones(1, 2, 3)
        logits = torch.zeros(1, 2, 1)
        raw = torch.tensor(
            [[[1000.0, 1000.0, 1000.0, 1000.0],
              [-1000.0, -1000.0, -1000.0, -1000.0]]]
        )
        updated_scales, updated_logits, scale_delta, opacity_delta = (
            apply_semantic_support_residual(
                scales,
                logits,
                raw,
                torch.tensor([[[1.0], [0.0]]]),
                opacity_gain=0.1,
                scale_gain=0.05,
            )
        )
        self.assertTrue(torch.allclose(opacity_delta[0, 0], torch.tensor([0.1])))
        self.assertTrue(torch.equal(opacity_delta[0, 1], torch.zeros(1)))
        self.assertTrue(torch.allclose(scale_delta[0, 0], torch.full((3,), 0.05)))
        self.assertTrue(torch.equal(scale_delta[0, 1], torch.zeros(3)))
        self.assertTrue(
            torch.allclose(updated_scales[0, 0], torch.full((3,), torch.exp(torch.tensor(0.05))))
        )
        self.assertTrue(torch.equal(updated_scales[0, 1], scales[0, 1]))
        self.assertTrue(torch.allclose(updated_logits[0, 0], torch.tensor([0.1])))

    def test_invalid_priority_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_semantic_support_residual(
                torch.ones(1, 3, 3),
                torch.zeros(1, 3, 1),
                torch.zeros(1, 3, 4),
                torch.ones(1, 3),
            )
