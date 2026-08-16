import unittest

import torch

from src.model.semantic_routing import apply_semantic_parameter_routes


class SemanticParameterRoutingTest(unittest.TestCase):
    def setUp(self):
        self.deltas = (
            torch.ones(1, 2, 3),
            torch.ones(1, 2, 3),
            torch.ones(1, 2, 4),
            torch.ones(1, 2, 1),
            torch.ones(1, 2, 27),
        )

    def test_zero_routes_are_exact_identity(self):
        outputs = apply_semantic_parameter_routes(
            *self.deltas, torch.zeros(1, 2, 2), gain=0.5
        )
        self.assertTrue(all(torch.equal(old, new) for old, new in zip(self.deltas, outputs)))

    def test_geometry_and_appearance_routes_are_separate(self):
        routes = torch.tensor([[[10.0, -10.0], [10.0, -10.0]]])
        outputs = apply_semantic_parameter_routes(*self.deltas, routes, gain=0.5)
        for geometry in outputs[:4]:
            self.assertTrue(torch.allclose(geometry, torch.full_like(geometry, 1.5)))
        self.assertTrue(torch.allclose(outputs[4], torch.full_like(outputs[4], 0.5)))

    def test_geometry_only_route_keeps_appearance_unchanged(self):
        routes = torch.full((1, 2, 1), 10.0)
        outputs = apply_semantic_parameter_routes(*self.deltas, routes, gain=0.5)
        for geometry in outputs[:4]:
            self.assertTrue(torch.allclose(geometry, torch.full_like(geometry, 1.5)))
        self.assertTrue(torch.equal(outputs[4], self.deltas[4]))

    def test_invalid_route_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_semantic_parameter_routes(
                *self.deltas, torch.zeros(1, 2, 3), gain=0.5
            )


if __name__ == "__main__":
    unittest.main()
