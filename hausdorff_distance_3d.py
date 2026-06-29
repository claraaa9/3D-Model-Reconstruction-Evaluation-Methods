"""Evaluate 3D reconstruction fidelity with Hausdorff distance."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def load_mesh_as_points(file_path: str | Path, num_points: int = 10_000) -> np.ndarray:
    """
    Load a 3D mesh file and sample points from its surface.

    Parameters
    ----------
    file_path:
        Path to the 3D model file, e.g. .obj, .stl, .ply, .glb.
    num_points:
        Number of points sampled from the mesh surface.

    Returns
    -------
    points:
        A numpy array with shape (num_points, 3).
    """
    mesh = trimesh.load(file_path, force="mesh")

    if mesh.is_empty:
        raise ValueError(f"Mesh is empty: {file_path}")

    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points.astype(np.float64)


def normalize_points(points: np.ndarray) -> np.ndarray:
    """
    Normalize point cloud.

    1. Move its center to the origin.
    2. Scale it so that the largest dimension of its bounding box is 1.
    """
    points = np.asarray(points, dtype=np.float64)

    center = points.mean(axis=0)
    points = points - center

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    scale = np.max(bbox_max - bbox_min)

    if scale == 0:
        raise ValueError("Point cloud has zero scale.")

    return points / scale


def directed_hausdorff_distance(
    source: np.ndarray,
    target: np.ndarray,
    percentile: float = 100.0,
) -> float:
    """
    Compute directed Hausdorff distance from source to target.

    For each source point, find its nearest target point. Standard directed
    Hausdorff distance is the maximum of those nearest-neighbor distances.
    A percentile lower than 100 gives a robust variant that reduces the impact
    of isolated noise points.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source must have shape (n, 3).")

    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("target must have shape (m, 3).")

    nearest_distances, _ = cKDTree(target).query(source)

    if percentile == 100:
        return float(nearest_distances.max())

    return float(np.percentile(nearest_distances, percentile))


def hausdorff_distance(
    reference: np.ndarray,
    generated: np.ndarray,
    percentile: float = 100.0,
) -> tuple[float, float, float]:
    """
    Compute bidirectional Hausdorff distance between two point clouds.

    Parameters
    ----------
    reference:
        Point cloud sampled from the baseline/reference model A.
    generated:
        Point cloud sampled from the generated model B.
    percentile:
        100 means standard Hausdorff distance. Values such as 95 or 99 compute
        a robust percentile Hausdorff distance.

    Returns
    -------
    total_hd:
        max(reference_to_generated, generated_to_reference).
    reference_to_generated:
        Directed distance from baseline model A to generated model B.
    generated_to_reference:
        Directed distance from generated model B to baseline model A.
    """
    reference_to_generated = directed_hausdorff_distance(
        reference,
        generated,
        percentile=percentile,
    )
    generated_to_reference = directed_hausdorff_distance(
        generated,
        reference,
        percentile=percentile,
    )
    total_hd = max(reference_to_generated, generated_to_reference)

    return total_hd, reference_to_generated, generated_to_reference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a generated 3D model against a reference model with Hausdorff distance.",
    )
    parser.add_argument("model_a", help="Path to baseline/reference model A.")
    parser.add_argument("model_b", help="Path to generated model B to evaluate.")
    parser.add_argument(
        "--num-points",
        type=int,
        default=10_000,
        help="Number of surface points sampled from each mesh. Default: 10000.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip centering and scale normalization.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=100.0,
        help="Use robust percentile Hausdorff distance, e.g. 95 or 99. Default: 100.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for repeatable surface sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_points <= 0:
        raise ValueError("--num-points must be greater than 0.")

    if not 0 < args.percentile <= 100:
        raise ValueError("--percentile must be in the range (0, 100].")

    if args.seed is not None:
        np.random.seed(args.seed)

    reference_points = load_mesh_as_points(args.model_a, args.num_points)
    generated_points = load_mesh_as_points(args.model_b, args.num_points)

    if not args.no_normalize:
        reference_points = normalize_points(reference_points)
        generated_points = normalize_points(generated_points)

    total_hd, reference_to_generated, generated_to_reference = hausdorff_distance(
        reference_points,
        generated_points,
        percentile=args.percentile,
    )

    metric_name = "Hausdorff Distance"
    if args.percentile < 100:
        metric_name = f"{args.percentile:g}th Percentile Hausdorff Distance"

    print(metric_name)
    print("-" * len(metric_name))
    print(f"A reference -> B generated: {reference_to_generated:.8f}")
    print(f"B generated -> A reference: {generated_to_reference:.8f}")
    print(f"Bidirectional max:          {total_hd:.8f}")


if __name__ == "__main__":
    main()
