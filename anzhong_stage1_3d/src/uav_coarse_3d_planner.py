"""Preflight-only robust multi-cover planner for a reconstructed building mesh."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw


SCALAR_TYPES = {
    "char": "i1", "uchar": "u1", "short": "<i2", "ushort": "<u2",
    "int": "<i4", "uint": "<u4", "float": "<f4", "double": "<f8",
}


@dataclass(frozen=True)
class SurfacePatch:
    center: np.ndarray
    normal: np.ndarray
    area: float
    region: str
    cluster_id: int
    importance: float


def read_triangle_mesh(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    vertex_properties = []
    vertex_count = face_count = 0
    face_count_type, face_index_type = "uchar", "int"
    section = None
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError("Expected a PLY mesh")
        while True:
            line = stream.readline().decode("ascii").strip()
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1]); section = "vertex"
            elif line.startswith("element face "):
                face_count = int(line.split()[-1]); section = "face"
            elif line.startswith("property ") and section == "vertex":
                parts = line.split()
                if len(parts) == 3 and parts[1] in SCALAR_TYPES:
                    vertex_properties.append((parts[2], SCALAR_TYPES[parts[1]]))
            elif line.startswith("property list ") and section == "face":
                parts = line.split()
                face_count_type, face_index_type = parts[2], parts[3]
            elif line == "end_header":
                offset = stream.tell(); break
    dtype = np.dtype(vertex_properties)
    vertices_raw = np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=(vertex_count,))
    vertices = np.column_stack([vertices_raw[a] for a in ("x", "y", "z")]).astype(np.float64)
    face_offset = offset + dtype.itemsize * vertex_count
    face_dtype = np.dtype([
        ("count", SCALAR_TYPES[face_count_type]),
        ("indices", SCALAR_TYPES[face_index_type], (3,)),
    ])
    faces_raw = np.memmap(path, mode="r", dtype=face_dtype, offset=face_offset, shape=(face_count,))
    if not np.all(faces_raw["count"] == 3):
        raise ValueError("Only triangular COLMAP PLY meshes are supported")
    return vertices, np.asarray(faces_raw["indices"], dtype=np.int64)


def sample_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    count: int,
    frame_center: np.ndarray,
    basis: np.ndarray,
    bounds_low: np.ndarray,
    bounds_high: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    valid = area2 > 1e-10
    tri, cross, area2 = tri[valid], cross[valid], area2[valid]
    centers = tri.mean(axis=1)
    normals = cross / area2[:, None]

    def inside_mesh(point: np.ndarray) -> bool:
        direction = np.array([1.0, 0.371, 0.193])
        direction /= np.linalg.norm(direction)
        edge1 = tri[:, 1] - tri[:, 0]
        edge2 = tri[:, 2] - tri[:, 0]
        h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
        determinant = np.sum(edge1 * h, axis=1)
        usable = np.abs(determinant) > 1e-10
        inverse = np.zeros_like(determinant)
        inverse[usable] = 1.0 / determinant[usable]
        s = point[None, :] - tri[:, 0]
        u = inverse * np.sum(s * h, axis=1)
        q = np.cross(s, edge1)
        v = inverse * np.sum(np.broadcast_to(direction, q.shape) * q, axis=1)
        distance = inverse * np.sum(edge2 * q, axis=1)
        hits = usable & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1.0 + 1e-9) & (distance > 1e-7)
        return bool(np.count_nonzero(hits) % 2)

    epsilon = max(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)) * 1e-5, 1e-4)
    inside_plus = np.array([inside_mesh(center + normal * epsilon) for center, normal in zip(centers, normals)])
    inside_minus = np.array([inside_mesh(center - normal * epsilon) for center, normal in zip(centers, normals)])
    internal = inside_plus & inside_minus
    flip = inside_plus & ~inside_minus
    normals[flip] *= -1
    tri, centers, normals, area2 = tri[~internal], centers[~internal], normals[~internal], area2[~internal]
    local_centers = to_local(centers, frame_center, basis)
    local_normals = rotate_to_local(normals, basis)
    span = bounds_high - bounds_low
    inside = np.all(
        (local_centers >= bounds_low - 0.03 * span)
        & (local_centers <= bounds_high + 0.03 * span),
        axis=1,
    )
    ground = (
        (local_centers[:, 2] < bounds_low[2] + 0.10 * span[2])
        & (np.abs(local_normals[:, 2]) > 0.65)
    )
    downward_noninspection = local_normals[:, 2] < -0.75
    keep = inside & ~ground & ~downward_noninspection
    tri, centers, normals, area2 = tri[keep], centers[keep], normals[keep], area2[keep]
    if not len(centers):
        raise ValueError("No inspectable surface remained after mesh filtering")
    if len(centers) > count:
        rng = np.random.default_rng(7)
        probability = area2 / area2.sum()
        selected = rng.choice(len(centers), size=count, replace=False, p=probability)
        centers, normals, area2 = centers[selected], normals[selected], area2[selected]
    elif len(centers) < count:
        # Lightweight LoD1 meshes have very few large triangles. Sample their
        # interiors by area so coverage certification does not collapse to one
        # target at each triangle center.
        rng = np.random.default_rng(7)
        probability = area2 / area2.sum()
        selected = rng.choice(len(centers), size=count, replace=True, p=probability)
        sampled_triangles = tri[selected]
        u = np.sqrt(rng.random(count))
        v = rng.random(count)
        centers = (
            (1.0 - u)[:, None] * sampled_triangles[:, 0]
            + (u * (1.0 - v))[:, None] * sampled_triangles[:, 1]
            + (u * v)[:, None] * sampled_triangles[:, 2]
        )
        normals = normals[selected]
        area2 = area2[selected]
    return centers, normals, area2 / 2.0


def geometry_frame(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = np.median(vertices, axis=0)
    centered = vertices - center
    # Coarse-model inputs are canonicalized to Z-up. A full 3D PCA can swap
    # the vertical axis with a long facade axis and create underground views.
    values, vectors = np.linalg.eigh(np.cov(centered[:, :2], rowvar=False))
    horizontal_x = vectors[:, np.argmax(values)]
    horizontal_y = np.array([-horizontal_x[1], horizontal_x[0]])
    basis = np.column_stack(
        [
            np.array([horizontal_x[0], horizontal_x[1], 0.0]),
            np.array([horizontal_y[0], horizontal_y[1], 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
    )
    aligned = np.dot(centered, basis)
    low, high = np.percentile(aligned, [1, 99], axis=0)
    return center, basis, low, high


def generate_candidates(low: np.ndarray, high: np.ndarray, azimuths: int, levels: int, stand_off: float) -> Tuple[np.ndarray, np.ndarray]:
    middle = (low + high) / 2
    half = (high - low) / 2
    z_levels = np.linspace(low[2] + 0.15 * (high[2] - low[2]), high[2] - 0.05 * (high[2] - low[2]), levels)
    positions, aims = [], []
    for z in z_levels:
        for angle in np.linspace(0, 2 * math.pi, azimuths, endpoint=False):
            position = np.array([
                middle[0] + (half[0] + stand_off) * math.cos(angle),
                middle[1] + (half[1] + stand_off) * math.sin(angle),
                z,
            ])
            positions.append(position)
            aims.append(np.array([middle[0], middle[1], z]))
    roof_aim_z = high[2] - 0.15 * (high[2] - low[2])
    for angle in np.linspace(0, 2 * math.pi, max(12, azimuths // 2), endpoint=False):
        position = np.array([
            middle[0] + 0.65 * half[0] * math.cos(angle),
            middle[1] + 0.65 * half[1] * math.sin(angle),
            high[2] + stand_off,
        ])
        positions.append(position)
        aims.append(np.array([middle[0], middle[1], roof_aim_z]))
    return np.asarray(positions), np.asarray(aims)


def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True), 1e-12)


def to_local(points: np.ndarray, center: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.einsum("ij,jk->ik", points - center, basis)


def to_world(points_local: np.ndarray, center: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.einsum("ij,kj->ik", points_local, basis) + center


def rotate_to_local(vectors: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.einsum("ij,jk->ik", vectors, basis)


def rotate_to_world(vectors_local: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.einsum("ij,kj->ik", vectors_local, basis)


def point_in_polygon_xy(point_xy: np.ndarray, polygon_xy: np.ndarray) -> bool:
    x, y = float(point_xy[0]), float(point_xy[1])
    inside = False
    n = len(polygon_xy)
    if n < 3:
        return False
    for i in range(n):
        x1, y1 = polygon_xy[i]
        x2, y2 = polygon_xy[(i + 1) % n]
        intersects = ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / max(y2 - y1, 1e-12) + x1
        )
        if intersects:
            inside = not inside
    return inside


def assign_targets_to_buildings(
    targets_world: np.ndarray,
    scene_json_path: Path | None,
) -> List[str]:
    if scene_json_path is None or not scene_json_path.exists():
        return ["scene"] * len(targets_world)
    data = json.loads(scene_json_path.read_text(encoding="utf-8"))
    buildings = data.get("buildings", [])
    assignments: List[str] = []
    for target in targets_world:
        assigned = None
        for building in buildings:
            footprint = np.asarray(building.get("footprint", []), dtype=np.float64)
            if len(footprint) >= 3 and point_in_polygon_xy(target[:2], footprint):
                assigned = str(building.get("name", building.get("id", "building")))
                break
        assignments.append(assigned or "unassigned")
    return assignments


def orthonormal_tangent(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    trial = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    tangent = np.cross(normal, trial)
    tangent /= max(np.linalg.norm(tangent), 1e-12)
    bitangent = np.cross(normal, tangent)
    bitangent /= max(np.linalg.norm(bitangent), 1e-12)
    return tangent, bitangent


def region_sampling_params(region: str, stand_off: float) -> Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]:
    if region == "roof":
        return (
            (0.85 * stand_off, 1.10 * stand_off),
            (-0.35, 0.0, 0.35),
            (-0.35, 0.0, 0.35),
        )
    if region == "corner_transition":
        return (
            (0.85 * stand_off, 1.10 * stand_off, 1.35 * stand_off),
            (-0.55, 0.0, 0.55),
            (-0.35, 0.0, 0.35),
        )
    if region == "occlusion_sensitive":
        return (
            (0.75 * stand_off, stand_off, 1.20 * stand_off),
            (-0.40, 0.0, 0.40),
            (-0.25, 0.0, 0.25),
        )
    return (
        (0.85 * stand_off, 1.15 * stand_off),
        (-0.30, 0.0, 0.30),
        (-0.25, 0.0, 0.25),
    )


def classify_surface_regions(targets: np.ndarray, normals: np.ndarray, areas: np.ndarray) -> List[str]:
    if len(targets) == 0:
        return []
    distances = np.linalg.norm(targets[:, None, :] - targets[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    k = min(6, len(targets) - 1) if len(targets) > 1 else 0
    regions: List[str] = []
    mean_area = float(np.mean(areas)) if len(areas) else 0.0
    for index, normal in enumerate(normals):
        region = "wall"
        if normal[2] > 0.72:
            region = "roof"
        elif abs(normal[2]) > 0.35:
            region = "sloped_transition"
        if k > 0:
            neighbor_ids = np.argpartition(distances[index], k)[:k]
            neighbor_normals = normals[neighbor_ids]
            angular_complexity = float(np.mean(1.0 - np.clip(neighbor_normals @ normal, -1.0, 1.0)))
            local_spacing = float(np.mean(distances[index, neighbor_ids]))
            if angular_complexity > 0.22:
                region = "corner_transition"
            elif local_spacing < 0.06 * np.linalg.norm(targets.max(axis=0) - targets.min(axis=0)):
                region = "occlusion_sensitive" if region != "roof" else region
        if areas[index] > 2.0 * max(mean_area, 1e-9) and region == "wall":
            region = "wall"
        regions.append(region)
    return regions


def cluster_surface_patches(
    targets: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    regions: List[str],
    scene_span: float,
) -> List[SurfacePatch]:
    if len(targets) == 0:
        return []
    cluster_radius = {
        "roof": 0.11 * scene_span,
        "wall": 0.08 * scene_span,
        "occlusion_sensitive": 0.06 * scene_span,
        "corner_transition": 0.05 * scene_span,
        "sloped_transition": 0.07 * scene_span,
    }
    patches: List[SurfacePatch] = []
    cluster_id = 0
    for region in sorted(set(regions)):
        region_ids = [index for index, label in enumerate(regions) if label == region]
        pending = set(region_ids)
        radius = cluster_radius.get(region, 0.07 * scene_span)
        while pending:
            seed = max(pending, key=lambda idx: areas[idx])
            pending.remove(seed)
            cluster_members = [seed]
            seed_center = targets[seed]
            seed_normal = normals[seed]
            absorbed = []
            for other in pending:
                if np.linalg.norm(targets[other] - seed_center) <= radius and np.dot(normals[other], seed_normal) >= 0.88:
                    absorbed.append(other)
            for other in absorbed:
                pending.remove(other)
                cluster_members.append(other)
            member_centers = targets[cluster_members]
            member_normals = normals[cluster_members]
            member_areas = areas[cluster_members]
            importance = float(member_areas.sum()) * (
                1.30 if region == "corner_transition" else 1.20 if region == "occlusion_sensitive" else 1.0
            )
            center = np.average(member_centers, axis=0, weights=member_areas)
            normal = np.average(member_normals, axis=0, weights=member_areas)
            normal = normal / max(np.linalg.norm(normal), 1e-12)
            patches.append(
                SurfacePatch(
                    center=center,
                    normal=normal,
                    area=float(member_areas.sum()),
                    region=region,
                    cluster_id=cluster_id,
                    importance=importance,
                )
            )
            cluster_id += 1
    return patches


def generate_partitioned_candidates(
    patches: List[SurfacePatch],
    stand_off: float,
    max_candidates: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
    positions: List[np.ndarray] = []
    aims: List[np.ndarray] = []
    metadata: List[Dict[str, float]] = []
    ranked = sorted(patches, key=lambda patch: (-patch.importance, patch.cluster_id))
    for patch in ranked:
        tangent, bitangent = orthonormal_tangent(patch.normal)
        distances, lateral_offsets, vertical_offsets = region_sampling_params(patch.region, stand_off)
        for distance_value in distances:
            for lateral in lateral_offsets:
                for vertical in vertical_offsets:
                    if patch.region == "roof":
                        radial = tangent * distance_value * lateral + bitangent * distance_value * vertical
                        position = patch.center + patch.normal * distance_value + radial
                    else:
                        position = (
                            patch.center
                            + patch.normal * distance_value
                            + tangent * distance_value * lateral
                            + bitangent * distance_value * vertical * 0.65
                        )
                    positions.append(position)
                    aims.append(patch.center)
                    metadata.append(
                        {
                            "cluster_id": float(patch.cluster_id),
                            "region_code": float(
                                {"roof": 0, "wall": 1, "occlusion_sensitive": 2, "corner_transition": 3, "sloped_transition": 4}.get(patch.region, 5)
                            ),
                            "importance": patch.importance,
                        }
                    )
        if patch.region in {"corner_transition", "occlusion_sensitive"}:
            tangent, bitangent = orthonormal_tangent(patch.normal)
            for angle in np.linspace(0, 2 * math.pi, 6, endpoint=False):
                radial = math.cos(angle) * tangent + math.sin(angle) * bitangent
                position = patch.center + patch.normal * 1.15 * stand_off + radial * 0.35 * stand_off
                positions.append(position)
                aims.append(patch.center)
                metadata.append(
                    {
                        "cluster_id": float(patch.cluster_id),
                        "region_code": 3.0 if patch.region == "corner_transition" else 2.0,
                        "importance": 1.1 * patch.importance,
                    }
                )
    if not positions:
        return np.zeros((0, 3)), np.zeros((0, 3)), []
    positions_array = np.asarray(positions)
    aims_array = np.asarray(aims)
    if len(positions_array) > max_candidates:
        scores = np.array([item["importance"] for item in metadata], dtype=float)
        keep = np.argsort(-scores)[:max_candidates]
        positions_array = positions_array[keep]
        aims_array = aims_array[keep]
        metadata = [metadata[index] for index in keep]
    return positions_array, aims_array, metadata


def coverage_matrix(
    candidates: np.ndarray,
    aims: np.ndarray,
    targets: np.ndarray,
    normals: np.ndarray,
    stand_off: float,
    fov_deg: float,
    max_incidence_deg: float,
    quality_threshold: float,
    min_standoff: float,
    max_standoff: float,
    target_weights: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    coverage, quality, _, _, _ = coverage_quality_matrices(
        candidates,
        aims,
        targets,
        normals,
        stand_off,
        fov_deg,
        max_incidence_deg,
        quality_threshold,
        min_standoff,
        max_standoff,
        target_weights,
    )
    return coverage, quality


def coverage_quality_matrices(
    candidates: np.ndarray,
    aims: np.ndarray,
    targets: np.ndarray,
    normals: np.ndarray,
    stand_off: float,
    fov_deg: float,
    max_incidence_deg: float,
    quality_threshold: float,
    min_standoff: float,
    max_standoff: float,
    target_weights: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coverage = np.zeros((len(candidates), len(targets)), dtype=bool)
    quality = np.zeros((len(candidates), len(targets)), dtype=np.float32)
    incidence_quality = np.zeros((len(candidates), len(targets)), dtype=np.float32)
    distance_quality_matrix = np.zeros((len(candidates), len(targets)), dtype=np.float32)
    visibility_quality = np.zeros((len(candidates), len(targets)), dtype=np.float32)
    fov_cos = math.cos(math.radians(fov_deg / 2))
    incidence_cos = math.cos(math.radians(max_incidence_deg))
    scene_span = np.linalg.norm(targets.max(axis=0) - targets.min(axis=0))
    for index, (camera, aim) in enumerate(zip(candidates, aims)):
        target_to_camera = camera[None, :] - targets
        distance = np.linalg.norm(target_to_camera, axis=1)
        incidence = np.sum(normals * normalize(target_to_camera), axis=1)
        incidence_component = np.clip(incidence, 0.0, 1.0).astype(np.float32)
        camera_to_target = -target_to_camera
        forward = normalize((aim - camera)[None, :])[0]
        in_frustum = np.sum(normalize(camera_to_target) * forward, axis=1) >= fov_cos
        distance_quality = np.exp(
            -0.5 * ((distance - stand_off) / max(1.5 * stand_off, 1e-6)) ** 2
        ).astype(np.float32)
        q = incidence_component * distance_quality
        if target_weights is not None:
            q = q * target_weights

        # Coarse angular z-buffer: reject points hidden behind a nearer surface.
        up_hint = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up_hint)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        direction = normalize(camera_to_target)
        horizontal = np.sum(direction * right, axis=1)
        vertical = np.sum(direction * up, axis=1)
        bx = np.clip(((horizontal + 0.8) / 1.6 * 159).astype(int), 0, 159)
        by = np.clip(((vertical + 0.8) / 1.6 * 119).astype(int), 0, 119)
        bins = by * 160 + bx
        nearest = np.full(160 * 120, np.inf)
        np.minimum.at(nearest, bins, distance)
        not_occluded = distance <= nearest[bins] + 0.025 * scene_span
        distance_valid = (distance >= min_standoff) & (distance <= max_standoff)
        visibility_component = (in_frustum & not_occluded & distance_valid).astype(np.float32)
        valid = in_frustum & (incidence >= incidence_cos) & not_occluded & distance_valid & (q >= quality_threshold)
        coverage[index] = valid
        quality[index] = np.where(valid, q, 0)
        incidence_quality[index] = np.where(valid, incidence_component, 0.0)
        distance_quality_matrix[index] = np.where(valid, distance_quality, 0.0)
        visibility_quality[index] = visibility_component
    return coverage, quality, incidence_quality, distance_quality_matrix, visibility_quality


def select_multicover(
    coverage: np.ndarray,
    quality: np.ndarray,
    positions: np.ndarray,
    required: int,
    candidate_region_codes: np.ndarray | None = None,
) -> Tuple[list, np.ndarray]:
    remaining = np.full(coverage.shape[1], required, dtype=np.int16)
    selected = []
    available = np.ones(coverage.shape[0], dtype=bool)
    current = None
    scale = max(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)), 1e-6)
    while remaining.max() > 0:
        useful = remaining > 0
        gains = ((coverage[:, useful]) * np.minimum(quality[:, useful] + 0.25, 1.0)).sum(axis=1)
        if current is not None:
            travel = np.linalg.norm(positions - positions[current], axis=1) / scale
            gains /= 1.0 + 0.2 * travel
        if candidate_region_codes is not None:
            region_bonus = np.where(candidate_region_codes >= 2.0, 1.12, 1.0)
            gains *= region_bonus
        gains[~available] = -1
        choice = int(np.argmax(gains))
        if gains[choice] <= 0:
            break
        selected.append(choice)
        available[choice] = False
        remaining = np.maximum(0, remaining - coverage[choice].astype(np.int16))
        current = choice
    return selected, remaining


def route_nearest_neighbor(selected: list, positions: np.ndarray) -> list:
    if not selected:
        return []
    pending = set(selected[1:])
    route = [selected[0]]
    while pending:
        last = route[-1]
        nxt = min(pending, key=lambda item: np.linalg.norm(positions[item] - positions[last]))
        route.append(nxt); pending.remove(nxt)
    return route


def route_metrics(route: list, positions: np.ndarray, turn_weight: float = 0.0):
    if len(route) < 2:
        return 0.0, 0.0, 0.0, 0.0
    points = positions[route]
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    climb = np.abs(np.diff(points[:, 2]))
    angles = []
    for first, second in zip(segments, segments[1:]):
        denominator = max(np.linalg.norm(first) * np.linalg.norm(second), 1e-12)
        angles.append(math.acos(float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))))
    length = float(lengths.sum())
    mean_turn = float(np.mean(angles)) if angles else 0.0
    max_turn = float(np.max(angles)) if angles else 0.0
    climb_cost = 0.45 * float(climb.sum())
    objective = length + climb_cost + turn_weight * float(np.sum(angles))
    return objective, length, mean_turn, max_turn


def route_multistart_optimize(
    selected: list,
    positions: np.ndarray,
    restarts: int,
    proposals: int,
    turn_weight: float,
) -> Tuple[list, list[dict], dict]:
    if not selected:
        return [], [], {}
    rng = np.random.default_rng(29)
    baseline = route_nearest_neighbor(selected, positions)
    baseline_metrics = route_metrics(baseline, positions, turn_weight)
    best_route = baseline[:]
    best_metrics = baseline_metrics
    history = []
    for restart in range(max(1, restarts)):
        if restart == 0:
            current = baseline[:]
        else:
            pending = set(selected)
            current = [int(rng.choice(list(pending)))]
            pending.remove(current[0])
            while pending:
                nearest = sorted(
                    pending,
                    key=lambda item: np.linalg.norm(positions[item] - positions[current[-1]]),
                )[: min(4, len(pending))]
                choice = int(rng.choice(nearest))
                current.append(choice)
                pending.remove(choice)
        current_metrics = route_metrics(current, positions, turn_weight)
        for _ in range(max(0, proposals)):
            i, j = sorted(rng.choice(len(current), size=2, replace=False))
            if j - i < 2:
                continue
            proposal = current[:i] + list(reversed(current[i : j + 1])) + current[j + 1 :]
            proposal_metrics = route_metrics(proposal, positions, turn_weight)
            if proposal_metrics[0] + 1e-9 < current_metrics[0]:
                current, current_metrics = proposal, proposal_metrics
        if current_metrics[0] < best_metrics[0]:
            best_route, best_metrics = current[:], current_metrics
        history.append(
            {
                "restart": restart + 1,
                "best_objective": best_metrics[0],
                "best_length_m": best_metrics[1],
                "best_mean_turn_deg": math.degrees(best_metrics[2]),
                "best_max_turn_deg": math.degrees(best_metrics[3]),
            }
        )
    baseline_summary = {
        "objective": baseline_metrics[0],
        "length_m": baseline_metrics[1],
        "mean_turn_deg": math.degrees(baseline_metrics[2]),
        "max_turn_deg": math.degrees(baseline_metrics[3]),
    }
    return best_route, history, baseline_summary


def filter_candidate_constraints(
    candidates: np.ndarray,
    aims: np.ndarray,
    targets: np.ndarray,
    center: np.ndarray,
    basis: np.ndarray,
    min_standoff: float,
    max_standoff: float,
    min_altitude: float,
    max_altitude: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if not len(candidates):
        return candidates, aims
    world_candidates = to_world(candidates, center, basis)
    world_candidates[:, 2] = np.clip(world_candidates[:, 2], min_altitude, max_altitude)
    candidates = to_local(world_candidates, center, basis)
    nearest_surface = np.min(
        np.linalg.norm(candidates[:, None, :] - targets[None, :, :], axis=2), axis=1
    )
    world_altitude = world_candidates[:, 2]
    keep = (
        (nearest_surface >= min_standoff)
        & (nearest_surface <= max_standoff)
        & (world_altitude >= min_altitude)
        & (world_altitude <= max_altitude)
    )
    return candidates[keep], aims[keep]


def targeted_candidates(
    targets: np.ndarray,
    normals: np.ndarray,
    missing: np.ndarray,
    stand_off: float,
    max_standoff: float,
) -> Tuple[np.ndarray, np.ndarray]:
    positions, aims = [], []
    for index in missing[:300]:
        target = targets[index]
        normal = normals[index]
        if normal[2] > 0.7:
            for distance in sorted({stand_off, max_standoff}):
                positions.append(target + normal * distance)
                aims.append(target)
                for angle in np.linspace(0, 2 * math.pi, 8, endpoint=False):
                    radial = np.array([math.cos(angle), math.sin(angle), 0.0])
                    positions.append(target + normal * distance + radial * distance * 0.35)
                    aims.append(target)
            continue
        tangent = np.cross(normal, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(tangent) < 1e-6:
            tangent = np.cross(normal, np.array([1.0, 0.0, 0.0]))
        tangent /= np.linalg.norm(tangent)
        for distance in sorted({stand_off, max_standoff}):
            for offset in (-0.4, 0.0, 0.4):
                for vertical_offset in (-0.2, 0.0, 0.2):
                    positions.append(
                        target
                        + normal * distance
                        + tangent * distance * offset
                        + np.array([0.0, 0.0, distance * vertical_offset])
                    )
                    aims.append(target)
    return np.asarray(positions), np.asarray(aims)


def render_plan(targets: np.ndarray, positions: np.ndarray, route: list, output: Path) -> None:
    width, height = 1200, 900
    all_xy = np.vstack([targets[:, :2], positions[:, :2]])
    low, high = all_xy.min(axis=0), all_xy.max(axis=0)
    span = np.maximum(high - low, 1e-6)
    scale = min((width - 80) / span[0], (height - 80) / span[1])
    def pixel(point):
        return (int(40 + (point[0] - low[0]) * scale), int(height - 40 - (point[1] - low[1]) * scale))
    image = Image.new("RGB", (width, height), (244, 247, 248))
    draw = ImageDraw.Draw(image)
    for point in targets[::max(1, len(targets) // 2500)]:
        x, y = pixel(point); draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(90, 111, 124))
    route_points = [pixel(positions[index]) for index in route]
    if len(route_points) > 1:
        draw.line(route_points, fill=(38, 120, 181), width=3)
    for order, point in enumerate(route_points, start=1):
        x, y = point; draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(214, 90, 74), outline="white", width=2)
        draw.text((x + 8, y - 8), str(order), fill=(23, 50, 77))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def plan(args: argparse.Namespace) -> Dict[str, object]:
    vertices, faces = read_triangle_mesh(args.mesh)
    center, basis, low, high = geometry_frame(vertices)
    center = np.ascontiguousarray(center, dtype=np.float64)
    basis = np.ascontiguousarray(basis, dtype=np.float64)
    low = np.ascontiguousarray(low, dtype=np.float64)
    high = np.ascontiguousarray(high, dtype=np.float64)
    targets, normals, areas = sample_surface(
        vertices, faces, args.max_targets, center, basis, low, high
    )
    targets = np.ascontiguousarray(targets, dtype=np.float64)
    normals = np.ascontiguousarray(normals, dtype=np.float64)
    areas = np.ascontiguousarray(areas, dtype=np.float64)
    targets_local = to_local(targets, center, basis)
    normals_local = rotate_to_local(normals, basis)
    scene_span = np.linalg.norm(high - low)
    regions = classify_surface_regions(targets_local, normals_local, areas)
    patches = cluster_surface_patches(targets_local, normals_local, areas, regions, scene_span)
    diagonal = np.linalg.norm(high - low)
    stand_off = args.stand_off if args.stand_off else 0.18 * diagonal
    min_standoff = float(args.min_standoff) if args.min_standoff is not None else 0.65 * stand_off
    max_standoff = float(args.max_standoff) if args.max_standoff is not None else 1.8 * stand_off
    min_altitude = float(args.min_altitude) if args.min_altitude is not None else 2.0
    max_altitude = float(args.max_altitude) if args.max_altitude is not None else 120.0
    candidates, aims, candidate_metadata = generate_partitioned_candidates(
        patches,
        stand_off,
        max_candidates=max(2000, args.azimuths * args.levels * 12),
    )
    if not len(candidates):
        candidates, aims = generate_candidates(low, high, args.azimuths, args.levels, stand_off)
        candidate_metadata = [
            {"cluster_id": -1.0, "region_code": 1.0, "importance": 1.0}
            for _ in range(len(candidates))
        ]
    candidates, aims = filter_candidate_constraints(
        candidates, aims, targets_local, center, basis,
        min_standoff, max_standoff, min_altitude, max_altitude,
    )
    candidate_metadata = candidate_metadata[: len(candidates)]
    region_weights = np.array(
        [
            1.20 if region == "corner_transition" else 1.15 if region == "occlusion_sensitive" else 1.05 if region == "roof" else 1.0
            for region in regions
        ],
        dtype=np.float32,
    )
    coverage, quality = coverage_matrix(
        candidates, aims, targets_local, normals_local, stand_off,
        args.fov, args.max_incidence, args.quality_threshold,
        min_standoff, max_standoff,
        region_weights,
    )
    candidate_region_codes = np.array([item["region_code"] for item in candidate_metadata], dtype=np.float32)
    selected, remaining = select_multicover(
        coverage,
        quality,
        candidates,
        args.required_views,
        candidate_region_codes,
    )
    if np.any(remaining > 0):
        focused_positions, focused_aims = targeted_candidates(
            targets_local, normals_local, np.flatnonzero(remaining > 0), stand_off, max_standoff
        )
        if len(focused_positions):
            focused_positions, focused_aims = filter_candidate_constraints(
                focused_positions, focused_aims, targets_local, center, basis,
                min_standoff, max_standoff, min_altitude, max_altitude,
            )
            focused_coverage, focused_quality = coverage_matrix(
                focused_positions, focused_aims, targets_local, normals_local,
                stand_off, args.fov, args.max_incidence, args.quality_threshold,
                min_standoff, max_standoff,
                region_weights,
            )
            candidates = np.vstack([candidates, focused_positions])
            aims = np.vstack([aims, focused_aims])
            coverage = np.vstack([coverage, focused_coverage])
            quality = np.vstack([quality, focused_quality])
            candidate_metadata.extend(
                [
                    {"cluster_id": -2.0, "region_code": 2.0, "importance": 0.8}
                    for _ in range(len(focused_positions))
                ]
            )
            candidate_region_codes = np.concatenate(
                [candidate_region_codes, np.full(len(focused_positions), 2.0, dtype=np.float32)]
            )
            selected, remaining = select_multicover(
                coverage, quality, candidates, args.required_views, candidate_region_codes
            )
    route, route_history, baseline_route = route_multistart_optimize(
        selected,
        candidates,
        int(getattr(args, "route_restarts", 32)),
        int(getattr(args, "route_proposals", 1200)),
        float(getattr(args, "turn_weight", 2.0)),
    )
    positions_world = to_world(candidates, center, basis)
    aims_world = to_world(aims, center, basis)
    targets_world = to_world(targets_local, center, basis)
    normals_world = rotate_to_world(normals_local, basis)
    target_buildings = assign_targets_to_buildings(targets_world, getattr(args, "scene_json", None))
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    uncovered_path = output / "uncovered_targets.csv"
    with uncovered_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["target", "remaining_views", "region", "x", "y", "z", "nx", "ny", "nz"])
        for target_index in np.flatnonzero(remaining > 0):
            writer.writerow(
                [target_index, int(remaining[target_index]), regions[target_index], *targets_world[target_index], *normals_world[target_index]]
            )
    history_path = output / "route_optimization.csv"
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        history_fields = [
            "restart",
            "best_objective",
            "best_length_m",
            "best_mean_turn_deg",
            "best_max_turn_deg",
        ]
        writer = csv.DictWriter(stream, fieldnames=history_fields)
        writer.writeheader()
        if route_history:
            writer.writerows(route_history)
    csv_path = output / "one_shot_waypoints.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sequence", "candidate", "region_code", "cluster_id", "x", "y", "z", "aim_x", "aim_y", "aim_z", "yaw_deg", "pitch_deg", "take_photo"])
        for sequence, index in enumerate(route, start=1):
            direction = aims_world[index] - positions_world[index]
            horizontal = math.hypot(direction[0], direction[1])
            yaw = math.degrees(math.atan2(direction[1], direction[0]))
            pitch = math.degrees(math.atan2(direction[2], horizontal))
            cluster_id = candidate_metadata[index]["cluster_id"] if index < len(candidate_metadata) else -1.0
            region_code = candidate_metadata[index]["region_code"] if index < len(candidate_metadata) else -1.0
            writer.writerow([sequence, index, region_code, cluster_id, *positions_world[index], *aims_world[index], yaw, pitch, 1])
    candidate_summary_path = output / "candidate_region_summary.csv"
    region_counts: Dict[str, int] = {}
    for patch in patches:
        region_counts[patch.region] = region_counts.get(patch.region, 0) + 1
    with candidate_summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["region", "cluster_count"])
        for region_name, count in sorted(region_counts.items()):
            writer.writerow([region_name, count])
    building_summary_path = output / "building_coverage_summary.csv"
    building_total: Dict[str, int] = {}
    building_certified: Dict[str, int] = {}
    for building_name, remaining_views in zip(target_buildings, remaining):
        building_total[building_name] = building_total.get(building_name, 0) + 1
        if int(remaining_views) == 0:
            building_certified[building_name] = building_certified.get(building_name, 0) + 1
    weakest_building_coverage = 1.0
    with building_summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["building", "targets", "certified_targets", "coverage_ratio"])
        for building_name in sorted(building_total):
            total = building_total[building_name]
            certified = building_certified.get(building_name, 0)
            ratio = certified / max(total, 1)
            weakest_building_coverage = min(weakest_building_coverage, ratio)
            writer.writerow([building_name, total, certified, ratio])
    preview = output / "one_shot_plan_top.png"
    render_plan(targets_local, candidates, route, preview)
    covered = int(np.count_nonzero(remaining == 0))
    route_length = sum(
        np.linalg.norm(positions_world[route[i]] - positions_world[route[i - 1]])
        for i in range(1, len(route))
    )
    _, _, mean_turn, max_turn = route_metrics(route, candidates)
    coverage_certified = covered == len(targets)
    missing_mask = remaining > 0
    missing_by_surface = {
        "roof": int(np.count_nonzero(missing_mask & (normals_world[:, 2] > 0.7))),
        "wall": int(np.count_nonzero(missing_mask & (np.abs(normals_world[:, 2]) <= 0.7))),
        "downward": int(np.count_nonzero(missing_mask & (normals_world[:, 2] < -0.7))),
    }
    flight_release = (
        "ready_for_controller_validation"
        if coverage_certified and args.metric_scale_known
        else "blocked_metric_scale_or_flight_constraints"
    )
    result = {
        "status": "coverage_certified" if coverage_certified else "infeasible",
        "flight_release": flight_release,
        "mesh": str(args.mesh),
        "target_faces": len(targets),
        "candidate_views": len(candidates),
        "surface_clusters": len(patches),
        "selected_views": len(route),
        "required_views_per_target": args.required_views,
        "certified_targets": covered,
        "certified_coverage": covered / len(targets),
        "stand_off_model_units": stand_off,
        "minimum_standoff_m": min_standoff,
        "maximum_standoff_m": max_standoff,
        "minimum_altitude_m": min_altitude,
        "maximum_altitude_m": max_altitude,
        "route_length_model_units": float(route_length),
        "baseline_route": baseline_route,
        "optimized_mean_turn_deg": math.degrees(mean_turn),
        "optimized_max_turn_deg": math.degrees(max_turn),
        "route_search_restarts": int(getattr(args, "route_restarts", 32)),
        "route_2opt_proposals_per_restart": int(getattr(args, "route_proposals", 1200)),
        "route_optimization_history": str(history_path),
        "uncovered_targets": str(uncovered_path),
        "candidate_region_summary": str(candidate_summary_path),
        "building_coverage_summary": str(building_summary_path),
        "weakest_building_coverage": weakest_building_coverage,
        "missing_by_surface": missing_by_surface,
        "waypoints": str(csv_path),
        "preview": str(preview),
    }
    (output / "one_shot_plan.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Certify a no-recapture inspection plan before takeoff.")
    parser.add_argument("mesh", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--scene-json", type=Path)
    parser.add_argument("--max-targets", type=int, default=3000)
    parser.add_argument("--azimuths", type=int, default=48)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--required-views", type=int, default=2)
    parser.add_argument("--stand-off", type=float)
    parser.add_argument("--min-standoff", type=float)
    parser.add_argument("--max-standoff", type=float)
    parser.add_argument("--min-altitude", type=float, default=2.0)
    parser.add_argument("--max-altitude", type=float, default=120.0)
    parser.add_argument("--fov", type=float, default=75.0)
    parser.add_argument("--max-incidence", type=float, default=72.0)
    parser.add_argument("--quality-threshold", type=float, default=0.18)
    parser.add_argument("--route-restarts", type=int, default=32)
    parser.add_argument("--route-proposals", type=int, default=1200)
    parser.add_argument("--turn-weight", type=float, default=2.0)
    parser.add_argument(
        "--metric-scale-known",
        action="store_true",
        help="Confirm that mesh units are meters from RTK/GCP/known dimensions.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    plan(parse_args())
