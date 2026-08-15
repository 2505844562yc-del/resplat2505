import unittest

import numpy as np

from scripts.generate_boundary_sidecars import masks_to_boundary_maps


class BoundaryGenerationTest(unittest.TestCase):
    def test_boundary_and_confidence(self):
        mask = np.zeros((7, 9), dtype=bool)
        mask[2:5, 3:7] = True
        boundary, confidence = masks_to_boundary_maps(
            [{
                "segmentation": mask,
                "predicted_iou": 0.9,
                "stability_score": 0.8,
            }],
            mask.shape,
            radius=1,
        )
        self.assertEqual(boundary.dtype, np.uint8)
        self.assertEqual(confidence.dtype, np.uint8)
        self.assertEqual(boundary.shape, mask.shape)
        self.assertEqual(confidence.shape, mask.shape)
        self.assertEqual(confidence[3, 4], round(0.8 * 255))
        self.assertEqual(confidence[0, 0], 0)
        self.assertGreater(np.count_nonzero(boundary), 0)

    def test_overlapping_masks_keep_best_confidence(self):
        first = np.zeros((5, 5), dtype=bool)
        second = np.zeros((5, 5), dtype=bool)
        first[1:4, 1:4] = True
        second[2:5, 2:5] = True
        _, confidence = masks_to_boundary_maps(
            [
                {
                    "segmentation": first,
                    "predicted_iou": 0.7,
                    "stability_score": 0.9,
                },
                {
                    "segmentation": second,
                    "predicted_iou": 0.95,
                    "stability_score": 0.85,
                },
            ],
            first.shape,
            radius=1,
        )
        self.assertEqual(confidence[2, 2], round(0.85 * 255))

    def test_rejects_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            masks_to_boundary_maps(
                [{
                    "segmentation": np.zeros((3, 3), dtype=bool),
                    "predicted_iou": 1.0,
                    "stability_score": 1.0,
                }],
                (4, 4),
                radius=1,
            )


if __name__ == "__main__":
    unittest.main()
