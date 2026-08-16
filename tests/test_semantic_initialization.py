import unittest

import torch

from src.model.semantic_initialization import (
    SemanticGeometryAdapter,
    boundary_proximity_features,
)


class SemanticInitializationTest(unittest.TestCase):
    def test_boundary_proximity_has_expected_falloff(self):
        boundary = torch.zeros(1, 1, 5, 5)
        confidence = torch.ones_like(boundary)
        boundary[..., 2, 2] = 1
        features = boundary_proximity_features(
            boundary, confidence, confidence_floor=0.5, max_radius=2
        )
        self.assertTrue(torch.equal(features[:, 0:1], boundary))
        self.assertEqual(features[0, 1, 2, 2].item(), 1.0)
        self.assertEqual(features[0, 2, 2, 2].item(), 1.0)
        self.assertAlmostEqual(features[0, 2, 2, 1].item(), 2 / 3, places=6)
        self.assertAlmostEqual(features[0, 2, 0, 0].item(), 1 / 3, places=6)

    def test_low_confidence_boundary_has_no_trusted_support(self):
        boundary = torch.ones(1, 1, 3, 3)
        confidence = torch.full_like(boundary, 0.2)
        features = boundary_proximity_features(
            boundary, confidence, confidence_floor=0.5, max_radius=1
        )
        self.assertTrue(torch.equal(features[:, 0:1], boundary))
        self.assertEqual(features[:, 1:].count_nonzero().item(), 0)

    def test_batch_and_view_prefix_is_preserved(self):
        boundary = torch.zeros(2, 3, 1, 4, 6)
        confidence = torch.ones_like(boundary)
        features = boundary_proximity_features(boundary, confidence)
        self.assertEqual(features.shape, (2, 3, 3, 4, 6))

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            boundary_proximity_features(
                torch.zeros(1, 1, 3, 3), torch.zeros(1, 2, 3, 3)
            )
        with self.assertRaises(ValueError):
            boundary_proximity_features(
                torch.zeros(1, 1, 3, 3), torch.zeros(1, 1, 3, 3), max_radius=0
            )


class SemanticGeometryAdapterTest(unittest.TestCase):
    def test_zero_initialized_adapter_is_exact_identity(self):
        adapter = SemanticGeometryAdapter(8, hidden_channels=4)
        base = torch.randn(2, 8, 4, 6)
        semantic = torch.randn(2, 3, 8, 12)
        fused, gate, residual = adapter(base, semantic)
        self.assertTrue(torch.equal(fused, base))
        self.assertEqual(gate.shape, (2, 1, 4, 6))
        self.assertEqual(residual.count_nonzero().item(), 0)

    def test_projection_receives_gradient(self):
        adapter = SemanticGeometryAdapter(4, hidden_channels=3)
        fused, _, _ = adapter(
            torch.randn(1, 4, 3, 5), torch.randn(1, 3, 6, 10)
        )
        fused.square().mean().backward()
        self.assertGreater(adapter.projection.weight.grad.abs().sum().item(), 0)

    def test_invalid_semantic_channels_are_rejected(self):
        adapter = SemanticGeometryAdapter(4)
        with self.assertRaises(ValueError):
            adapter(torch.zeros(1, 4, 2, 2), torch.zeros(1, 2, 2, 2))


if __name__ == "__main__":
    unittest.main()
