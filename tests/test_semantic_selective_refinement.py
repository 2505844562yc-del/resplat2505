import unittest

import torch

from src.model.semantic_selective_refinement import (
    apply_selective_geometry_residual,
)


class SemanticSelectiveRefinementTest(unittest.TestCase):
    def setUp(self):
        self.means = torch.ones(1, 2, 3)
        self.scales = 2 * torch.ones(1, 2, 3)

    def test_zero_correction_is_exact_identity(self):
        outputs = apply_selective_geometry_residual(
            self.means,
            self.scales,
            torch.zeros(1, 2, 6),
            torch.ones(1, 2, 1),
            torch.ones(1, 2, 1, dtype=torch.bool),
        )
        self.assertTrue(torch.equal(outputs[0], self.means))
        self.assertTrue(torch.equal(outputs[1], self.scales))

    def test_inactive_tokens_are_unchanged(self):
        outputs = apply_selective_geometry_residual(
            self.means,
            self.scales,
            10 * torch.ones(1, 2, 6),
            torch.ones(1, 2, 1),
            torch.tensor([[[True], [False]]]),
            gain=0.5,
            reference_floor=0.0,
        )
        self.assertFalse(torch.equal(outputs[0][:, :1], self.means[:, :1]))
        self.assertTrue(torch.equal(outputs[0][:, 1:], self.means[:, 1:]))
        self.assertTrue(torch.equal(outputs[1][:, 1:], self.scales[:, 1:]))

    def test_axes_and_parameter_groups_are_independent(self):
        corrections = torch.tensor([[[10.0, 0.0, -10.0, 0.0, 10.0, 0.0]]])
        means, scales = apply_selective_geometry_residual(
            self.means[:, :1],
            self.scales[:, :1],
            corrections,
            torch.ones(1, 1, 1),
            torch.ones(1, 1, 1, dtype=torch.bool),
            gain=0.5,
            reference_floor=0.0,
        )
        self.assertTrue(torch.allclose(means, torch.tensor([[[1.5, 1.0, 0.5]]])))
        self.assertTrue(torch.allclose(scales, torch.tensor([[[2.0, 3.0, 2.0]]])))

    def test_invalid_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            apply_selective_geometry_residual(
                self.means,
                self.scales,
                torch.zeros(1, 2, 5),
                torch.ones(1, 2, 1),
                torch.ones(1, 2, 1, dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()
