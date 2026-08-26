import unittest

import torch

from src.model.semantic_depth_initialization import (
    SemanticDepthFeatureAdapter,
    SemanticDepthLogitAdapter,
    boundary_weighted_semantic_feature_loss,
    normalized_depth_entropy,
)


class NormalizedDepthEntropyTest(unittest.TestCase):
    def test_uniform_distribution_has_unit_entropy(self):
        entropy = normalized_depth_entropy(torch.zeros(2, 8, 3, 4))
        self.assertTrue(torch.allclose(entropy, torch.ones_like(entropy)))

    def test_peaked_distribution_has_low_entropy(self):
        logits = torch.full((1, 4, 2, 2), -20.0)
        logits[:, 0] = 20.0
        self.assertLess(normalized_depth_entropy(logits).max().item(), 1e-5)

    def test_invalid_candidate_count_is_rejected(self):
        with self.assertRaises(ValueError):
            normalized_depth_entropy(torch.zeros(1, 1, 2, 2))


class BoundaryWeightedSemanticFeatureLossTest(unittest.TestCase):
    def _inputs(self):
        cosine = torch.tensor([[[[0.9, 0.0], [0.8, 0.7]]]])
        alpha = torch.ones_like(cosine)
        boundary = torch.zeros(1, 1, 1, 2, 2)
        boundary[..., 0, 1] = 1
        confidence = torch.ones_like(boundary)
        return cosine, alpha, boundary, confidence

    def test_zero_weight_matches_valid_mean(self):
        cosine, alpha, boundary, confidence = self._inputs()
        loss, weights, trusted, boundary_error = (
            boundary_weighted_semantic_feature_loss(
                cosine, alpha, boundary, confidence, boundary_weight=0
            )
        )
        self.assertTrue(torch.allclose(loss, (1 - cosine).mean()))
        self.assertTrue(torch.equal(weights, torch.ones_like(cosine)))
        self.assertEqual(trusted.sum().item(), 1)
        self.assertAlmostEqual(boundary_error.item(), 1.0, places=6)

    def test_boundary_weight_emphasizes_boundary_error(self):
        cosine, alpha, boundary, confidence = self._inputs()
        baseline = (1 - cosine).mean()
        loss, weights, _, _ = boundary_weighted_semantic_feature_loss(
            cosine, alpha, boundary, confidence, boundary_weight=4
        )
        self.assertGreater(loss.item(), baseline.item())
        self.assertEqual(weights[0, 0, 0, 1].item(), 5.0)

    def test_low_confidence_boundary_is_not_emphasized(self):
        cosine, alpha, boundary, confidence = self._inputs()
        confidence.zero_()
        loss, weights, trusted, _ = boundary_weighted_semantic_feature_loss(
            cosine,
            alpha,
            boundary,
            confidence,
            boundary_weight=4,
            confidence_floor=0.5,
        )
        self.assertTrue(torch.allclose(loss, (1 - cosine).mean()))
        self.assertTrue(torch.equal(weights, torch.ones_like(cosine)))
        self.assertEqual(trusted.count_nonzero().item(), 0)

    def test_invalid_shapes_and_ranges_are_rejected(self):
        cosine, alpha, boundary, confidence = self._inputs()
        with self.assertRaises(ValueError):
            boundary_weighted_semantic_feature_loss(
                cosine, alpha[..., :1], boundary, confidence, 1
            )
        with self.assertRaises(ValueError):
            boundary_weighted_semantic_feature_loss(
                cosine, alpha, boundary, confidence, -1
            )


class SemanticDepthFeatureAdapterTest(unittest.TestCase):
    def _adapter(self, max_relative_residual=0.1):
        return SemanticDepthFeatureAdapter(
            semantic_channels=4,
            depth_feature_channels=8,
            hidden_channels=8,
            max_relative_residual=max_relative_residual,
        )

    def _inputs(self):
        depth = torch.randn(2, 8, 5, 7)
        semantic = torch.randn(2, 4, 3, 4)
        logits = torch.randn(2, 6, 5, 7)
        return depth, semantic, logits

    def test_zero_initialized_adapter_is_exact_identity(self):
        adapter = self._adapter()
        depth, semantic, logits = self._inputs()
        conditioned, gate, residual = adapter(depth, semantic, logits)
        self.assertTrue(torch.equal(conditioned, depth))
        self.assertEqual(residual.count_nonzero().item(), 0)
        self.assertTrue(torch.all((gate >= 0) & (gate <= 1)))

    def test_feature_head_receives_gradient(self):
        adapter = self._adapter()
        depth, semantic, logits = self._inputs()
        conditioned, _, _ = adapter(depth, semantic, logits)
        conditioned.square().mean().backward()
        gradient = adapter.residual_head.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(gradient.abs().sum().item(), 0)

    def test_residual_is_bounded_relative_to_local_feature_scale(self):
        adapter = self._adapter(max_relative_residual=0.2)
        with torch.no_grad():
            adapter.residual_head.bias.fill_(100)
        depth, semantic, logits = self._inputs()
        _, _, residual = adapter(depth, semantic, logits)
        local_scale = depth.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-3)
        self.assertTrue(torch.all(residual.abs() <= 0.2 * local_scale + 1e-6))

    def test_invalid_inputs_are_rejected(self):
        adapter = self._adapter()
        depth, semantic, logits = self._inputs()
        with self.assertRaises(ValueError):
            adapter(depth[:, :4], semantic, logits)
        with self.assertRaises(ValueError):
            adapter(depth, semantic[:1], logits)


class SemanticDepthLogitAdapterTest(unittest.TestCase):
    def _adapter(self, max_residual=0.25):
        return SemanticDepthLogitAdapter(
            semantic_channels=16,
            depth_feature_channels=8,
            num_depth_candidates=12,
            hidden_channels=10,
            max_logit_residual=max_residual,
        )

    def test_zero_initialized_adapter_is_exact_identity(self):
        adapter = self._adapter()
        base = torch.randn(2, 12, 4, 6)
        fused, gate, residual, encoded = adapter(
            base,
            torch.randn(2, 8, 4, 6),
            torch.randn(2, 16, 8, 12),
        )
        self.assertTrue(torch.equal(fused, base))
        self.assertEqual(residual.count_nonzero().item(), 0)
        self.assertEqual(gate.shape, (2, 1, 4, 6))
        self.assertEqual(encoded.shape, (2, 10, 4, 6))

    def test_residual_is_bounded(self):
        adapter = self._adapter(max_residual=0.2)
        torch.nn.init.constant_(adapter.residual_head.weight, 100.0)
        base = torch.randn(1, 12, 3, 5)
        fused, gate, residual, _ = adapter(
            base,
            torch.randn(1, 8, 3, 5),
            torch.randn(1, 16, 6, 10),
        )
        self.assertLessEqual(residual.abs().max().item(), 0.2 + 1e-6)
        self.assertGreaterEqual(gate.min().item(), 0.0)
        self.assertLessEqual(gate.max().item(), 1.0)
        self.assertLessEqual((fused - base).abs().max().item(), 0.2 + 1e-6)

    def test_new_head_receives_gradient(self):
        adapter = self._adapter()
        fused, _, _, _ = adapter(
            torch.randn(1, 12, 3, 5),
            torch.randn(1, 8, 3, 5),
            torch.randn(1, 16, 6, 10),
        )
        fused.square().mean().backward()
        self.assertGreater(adapter.residual_head.weight.grad.abs().sum().item(), 0)

    def test_mismatched_candidate_count_is_rejected(self):
        adapter = self._adapter()
        with self.assertRaises(ValueError):
            adapter(
                torch.zeros(1, 8, 2, 2),
                torch.zeros(1, 8, 2, 2),
                torch.zeros(1, 16, 2, 2),
            )


if __name__ == "__main__":
    unittest.main()
