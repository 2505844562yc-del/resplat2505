import numpy as np
import open3d as o3d

from scripts.convert_scannetpp_scene import render_labels, resize_geometry


def test_resize_geometry_matches_scale_then_center_crop():
    k = np.array([[622.0, 0.0, 876.0], [0.0, 623.0, 584.0], [0.0, 0.0, 1.0]])
    scaled, size, crop = resize_geometry(1752, 1168, k, 448, 256)
    assert size == (448, 299)
    assert crop == (0, 21, 448, 277)
    assert np.isclose(scaled[0, 2], 224.0)
    assert np.isclose(scaled[1, 2], 128.5)


def test_raycast_depth_and_face_labels_use_camera_z():
    scene = o3d.t.geometry.RaycastingScene()
    vertices = np.array([[-2.0, -2.0, 2.0], [2.0, -2.0, 2.0], [0.0, 2.0, 2.0]], dtype=np.float32)
    triangles = np.array([[0, 1, 2]], dtype=np.uint32)
    scene.add_triangles(o3d.core.Tensor(vertices), o3d.core.Tensor(triangles))
    k = np.array([[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]])
    depth, semantic, instance, valid = render_labels(
        scene, np.array([7], dtype=np.uint8), np.array([3], dtype=np.uint16),
        np.eye(4), k, 4, 4,
    )
    assert valid[2, 2]
    assert depth[2, 2] == 2000
    assert semantic[2, 2] == 7
    assert instance[2, 2] == 3
