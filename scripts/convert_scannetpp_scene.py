"""Convert one ScanNet++ DSLR scene into a supervised ReSplat chunk.

RGB comes from official resized *undistorted* images. COLMAP poses remain in the
scan mesh frame; Nerfstudio poses do not. Open3D raycasting gives visible mesh
triangle IDs and metric camera-z depth at the final training resolution.
"""

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from plyfile import PlyData
from scipy.spatial.transform import Rotation
import torch


def read_colmap_poses(path: Path) -> dict[str, np.ndarray]:
    poses = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 10 or not fields[0].isdigit() or not fields[9].lower().endswith((".jpg", ".png")):
            continue
        qw, qx, qy, qz = map(float, fields[1:5])
        translation = np.asarray(fields[5:8], dtype=np.float64)
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        w2c[:3, 3] = translation
        poses[Path(fields[9]).name] = w2c
    return poses


def resize_geometry(width: int, height: int, intrinsic: np.ndarray, out_width: int, out_height: int):
    scale = max(out_width / width, out_height / height)
    scaled_width = round(width * scale)
    scaled_height = round(height * scale)
    left = (scaled_width - out_width) // 2
    top = (scaled_height - out_height) // 2
    k = intrinsic.copy()
    k[0, 0] *= scaled_width / width
    k[1, 1] *= scaled_height / height
    k[0, 2] = k[0, 2] * scaled_width / width - left
    k[1, 2] = k[1, 2] * scaled_height / height - top
    return k, (scaled_width, scaled_height), (left, top, left + out_width, top + out_height)


def remap_semantic_ids(raw_root: Path) -> np.ndarray:
    classes = (raw_root / "metadata/semantic_classes.txt").read_text().splitlines()
    top100 = (raw_root / "metadata/semantic_benchmark/top100.txt").read_text().splitlines()
    top_index = {name.strip(): index for index, name in enumerate(top100)}
    with (raw_root / "metadata/semantic_benchmark/map_benchmark.csv").open(newline="") as handle:
        rows = {row["class"].strip(): row["semantic_map_to"].strip() for row in csv.DictReader(handle)}
    remap = np.full(len(classes), 255, dtype=np.uint8)
    for index, name in enumerate(classes):
        name = name.strip()
        target = rows.get(name, "") or name
        if target in top_index:
            remap[index] = top_index[target]
    return remap


def vertex_instances(scans: Path, vertex_count: int) -> np.ndarray:
    annotation = json.loads((scans / "segments_anno.json").read_text())
    ids = np.zeros(vertex_count, dtype=np.uint16)
    for group in annotation["segGroups"]:
        vertices = np.asarray(group["segments"], dtype=np.int64)
        if vertices.size and (vertices.min() < 0 or vertices.max() >= vertex_count):
            raise ValueError("instance annotation contains out-of-range vertex IDs")
        ids[vertices] = int(group["objectId"])
    return ids


def encode_png(array: np.ndarray) -> torch.Tensor:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return torch.from_numpy(np.frombuffer(buffer.getvalue(), dtype=np.uint8).copy())


def render_labels(ray_scene, face_semantic: np.ndarray, face_instance: np.ndarray,
                  w2c: np.ndarray, k: np.ndarray, width: int, height: int):
    u, v = np.meshgrid(np.arange(width, dtype=np.float32) + 0.5,
                       np.arange(height, dtype=np.float32) + 0.5)
    directions_cam = np.stack(((u - k[0, 2]) / k[0, 0],
                               (v - k[1, 2]) / k[1, 1], np.ones_like(u)), axis=-1)
    c2w = np.linalg.inv(w2c)
    directions = directions_cam @ c2w[:3, :3].T
    origins = np.broadcast_to(c2w[:3, 3].astype(np.float32), directions.shape)
    rays = np.concatenate((origins, directions.astype(np.float32)), axis=-1)
    hits = ray_scene.cast_rays(o3d.core.Tensor(rays))
    depth = hits["t_hit"].numpy()
    faces = hits["primitive_ids"].numpy()
    valid = np.isfinite(depth) & (depth >= 0.1) & (depth <= 20.0) & (faces < len(face_semantic))
    depth_mm = np.zeros((height, width), dtype=np.uint16)
    semantics = np.full((height, width), 255, dtype=np.uint8)
    instances = np.zeros((height, width), dtype=np.uint16)
    depth_mm[valid] = np.rint(depth[valid] * 1000).astype(np.uint16)
    semantics[valid] = face_semantic[faces[valid]]
    instances[valid] = face_instance[faces[valid]]
    return depth_mm, semantics, instances, valid


def convert(args):
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}; pass --overwrite explicitly")
    index_path = args.output.parent / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    if args.scene_id in index and index[args.scene_id] != args.output.name and not args.overwrite:
        raise ValueError(f"scene {args.scene_id} is already indexed in {index_path}")
    raw_root = args.raw_root
    scene_root = raw_root / "data" / args.scene_id
    dslr = scene_root / "dslr"
    scans = scene_root / "scans"
    split = json.loads((dslr / "train_test_lists.json").read_text())
    names = split[args.image_split]
    if args.max_frames:
        names = names[:args.max_frames]
    poses = read_colmap_poses(dslr / "colmap/images.txt")
    transforms = json.loads((dslr / "nerfstudio/transforms_undistorted.json").read_text())
    if transforms.get("camera_model") != "PINHOLE":
        raise ValueError("expected the official undistorted PINHOLE camera")
    width, height = int(transforms["w"]), int(transforms["h"])
    k_native = np.asarray([[transforms["fl_x"], 0, transforms["cx"]],
                           [0, transforms["fl_y"], transforms["cy"]],
                           [0, 0, 1]], dtype=np.float64)
    k, scaled_size, crop_box = resize_geometry(width, height, k_native, args.width, args.height)

    mesh = o3d.io.read_triangle_mesh(str(scans / "mesh_aligned_0.05.ply"))
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.triangles, dtype=np.uint32)
    semantic_vertices = np.asarray(PlyData.read(scans / "mesh_aligned_0.05_semantic.ply")["vertex"]["label"])
    if len(vertices) != len(semantic_vertices):
        raise ValueError("RGB and semantic mesh vertex counts differ")
    semantic_remap = remap_semantic_ids(raw_root)
    raw_labels = semantic_vertices[triangles[:, 0]]
    face_semantic = np.full(len(triangles), 255, dtype=np.uint8)
    label_valid = (raw_labels >= 0) & (raw_labels < len(semantic_remap))
    face_semantic[label_valid] = semantic_remap[raw_labels[label_valid]]
    instances = vertex_instances(scans, len(vertices))
    face_instance = instances[triangles[:, 0]]
    ray_scene = o3d.t.geometry.RaycastingScene()
    ray_scene.add_triangles(o3d.core.Tensor(vertices), o3d.core.Tensor(triangles))

    example = {"key": args.scene_id, "cameras": [], "images": [], "depths": [],
               "semantics": [], "instances": []}
    valid_rates = []
    for number, name in enumerate(names, 1):
        if name not in poses:
            raise KeyError(f"COLMAP pose missing for {name}")
        rgb_path = dslr / "resized_undistorted_images" / name
        mask_path = dslr / "resized_undistorted_masks" / f"{Path(name).stem}.png"
        with Image.open(rgb_path) as image:
            if image.size != (width, height):
                raise ValueError(f"unexpected image shape for {name}: {image.size}")
            rgb = np.asarray(image.convert("RGB").resize(scaled_size, Image.Resampling.LANCZOS).crop(crop_box))
        with Image.open(mask_path) as mask:
            valid_mask = np.asarray(mask.resize(scaled_size, Image.Resampling.NEAREST).crop(crop_box)) > 0
        depth, semantic, instance, valid = render_labels(
            ray_scene, face_semantic, face_instance, poses[name], k, args.width, args.height)
        depth[~valid_mask] = 0
        semantic[~valid_mask] = 255
        instance[~valid_mask] = 0
        valid_rates.append(float((valid & valid_mask).mean()))
        camera = np.concatenate(([k[0, 0] / args.width, k[1, 1] / args.height,
                                  k[0, 2] / args.width, k[1, 2] / args.height, 0, 0],
                                 poses[name][:3].reshape(-1))).astype(np.float32)
        example["cameras"].append(camera)
        example["images"].append(encode_png(rgb))
        example["depths"].append(encode_png(depth))
        example["semantics"].append(encode_png(semantic))
        example["instances"].append(encode_png(instance))
        if number == 1 and args.qa_dir is not None:
            args.qa_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(args.qa_dir / "rgb.png")
            palette = np.random.default_rng(0).integers(48, 255, size=(256, 3), dtype=np.uint8)
            palette[255] = 0
            colors = palette[semantic]
            overlay = np.where((semantic != 255)[..., None],
                               0.55 * rgb + 0.45 * colors, rgb).astype(np.uint8)
            Image.fromarray(overlay).save(args.qa_dir / "semantic_overlay.png")
            depth_vis = np.clip(depth.astype(np.float32) / 10000 * 255, 0, 255).astype(np.uint8)
            Image.fromarray(depth_vis).save(args.qa_dir / "depth.png")
        if number == 1 or number % 20 == 0 or number == len(names):
            print(f"rendered {number}/{len(names)}; visible pixels {valid_rates[-1]:.1%}", flush=True)

    example["cameras"] = torch.from_numpy(np.stack(example["cameras"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save([example], args.output)
    index[args.scene_id] = args.output.name
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    print(f"saved {args.output}; frames={len(names)}; mean visible={np.mean(valid_rates):.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("/root/autodl-tmp/scannetpp_raw"))
    parser.add_argument("--scene-id", default="39f36da05b")
    parser.add_argument("--image-split", choices=("train", "test"), default="train")
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--qa-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    convert(parser.parse_args())
