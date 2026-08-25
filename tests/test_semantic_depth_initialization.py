import unittest

import torch

from src.model.semantic_depth_initialization import (
    SemanticDepthLogitAdapter,
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
