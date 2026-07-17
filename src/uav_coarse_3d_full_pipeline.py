#!/usr/bin/env python3
"""Integrated coarse-3D pipeline: candidate generation -> view selection -> path planning.

This connects the current research mainline into one executable prototype:

1. coarse 3D proxy mesh -> surface-normal + partition/cluster candidate generation;
2. viewpoint-selection agent -> greedy / SCP-ILP / Maskable PPO best-of-N;
3. path-planning agent -> cluster/building-level ordering + 2D A* obstacle avoidance + B-spline smoothing.

The path stage is intentionally compact and runnable on a laptop. It reuses
the candidate-generation clustering cues: first sort selected viewpoints at
the building and cluster levels, then connect them with obstacle-aware graph
search, and finally smooth the route under geometric validity checks.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
MPL_CACHE = ROOT / ".matplotlib_cache"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
MPL_CACHE.mkdir(exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import CubicSpline, splprep, splev

from uav_coarse_3d_planner import (
    point_in_polygon_xy,
    route_multistart_optimize,
)
from uav_coarse_3d_viewpoint_rl import (
    Coarse3DScenario,
    SelectionPolicyOutput,
    SelectionResult,
    SeedRunResult,
    augment_scenario_with_post_rescue_candidates,
    build_scenario,
    selected_variant_points_world,
    compress_candidates,
    evaluate_selection,
    greedy_selection,
    run_maskable_ppo,
    scp_ilp_selection,
    train_and_evaluate_multi_seed,
)


Point2 = Tuple[float, float]
Point3 = Tuple[float, float, float]


@dataclass(frozen=True)
class PathScenario3D:
    method: str
    selected_indices: List[int]
    selected_points: np.ndarray
    aims_world: np.ndarray
    region_codes: np.ndarray
    cluster_ids: np.ndarray
    sector_ids: np.ndarray
    shot_counts: np.ndarray
    standoff_distances: np.ndarray
    owners: List[str]
    start: np.ndarray
    pair_costs: np.ndarray
    paths_3d: Dict[Tuple[int, int], List[Point3]]
    polygons: List[np.ndarray]
    polygon_names: List[str]
    heights: Dict[str, float]
    bounds: Tuple[float, float, float, float]
    z_limits: Tuple[float, float]
    return_to_start: bool


@dataclass(frozen=True)
class PathPlanResult:
    method: str
    route_local_indices: List[int]
    route_world_indices: List[int]
    polyline_3d: List[Point3]
    path_length_m: float
    smooth_path_length_m: float


def english_building_alias(name: str) -> str:
    alias_map = {
        "安中A座东西向高段": "Anzhong Building",
        "安中A座南北向低段": "Anzhong Building",
        "安中B座东西向高段": "Anzhong Building",
        "安中B座南北向低段": "Anzhong Building",
        "安中大楼连接体": "Anzhong Building",
        "建工实验大厅": "Construction Lab",
        "钟楼": "Clock Tower",
        "计算中心": "Computing Center",
    }
    return alias_map.get(name, name if name.isascii() else "")


def read_scene_obstacles(scene_json: Path) -> Tuple[List[np.ndarray], List[str], Dict[str, float]]:
    data = json.loads(scene_json.read_text(encoding="utf-8"))
    polygons: List[np.ndarray] = []
    polygon_names: List[str] = []
    heights: Dict[str, float] = {}
    for building in data.get("buildings", []):
        footprint = np.asarray(building.get("footprint", []), dtype=np.float64)
        if len(footprint) >= 3:
            polygons.append(footprint)
            polygon_names.append(str(building.get("name", building.get("id", "building"))))
        heights[str(building.get("name", building.get("id", "building")))] = float(building.get("height", 24.0))
    return polygons, polygon_names, heights


def scene_bounds(polygons: Sequence[np.ndarray], points: np.ndarray) -> Tuple[float, float, float, float]:
    xs = [float(value) for polygon in polygons for value in polygon[:, 0]] + list(points[:, 0])
    ys = [float(value) for polygon in polygons for value in polygon[:, 1]] + list(points[:, 1])
    return min(xs) - 10.0, max(xs) + 10.0, min(ys) - 10.0, max(ys) + 10.0


def inflate_polygon_bbox(polygon: np.ndarray, margin: float) -> Tuple[float, float, float, float]:
    return (
        float(np.min(polygon[:, 0]) - margin),
        float(np.max(polygon[:, 0]) + margin),
        float(np.min(polygon[:, 1]) - margin),
        float(np.max(polygon[:, 1]) + margin),
    )


def cell_blocked(point: Point2, polygons: Sequence[np.ndarray], margin: float) -> bool:
    p = np.asarray(point, dtype=np.float64)
    for polygon in polygons:
        bbox = inflate_polygon_bbox(polygon, margin)
        if bbox[0] <= p[0] <= bbox[1] and bbox[2] <= p[1] <= bbox[3]:
            if point_in_polygon_xy(p, polygon):
                return True
    return False


def ccw(a: Point2, b: Point2, c: Point2) -> bool:
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(a: Point2, b: Point2, c: Point2, d: Point2) -> bool:
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def direct_segment_clear(start: Point2, goal: Point2, polygons: Sequence[np.ndarray], margin: float) -> bool:
    mid = ((start[0] + goal[0]) * 0.5, (start[1] + goal[1]) * 0.5)
    if cell_blocked(mid, polygons, margin):
        return False
    for polygon in polygons:
        bbox = inflate_polygon_bbox(polygon, margin)
        if (
            max(start[0], goal[0]) < bbox[0]
            or min(start[0], goal[0]) > bbox[1]
            or max(start[1], goal[1]) < bbox[2]
            or min(start[1], goal[1]) > bbox[3]
        ):
            continue
        for i in range(len(polygon)):
            c = (float(polygon[i, 0]), float(polygon[i, 1]))
            d = (float(polygon[(i + 1) % len(polygon), 0]), float(polygon[(i + 1) % len(polygon), 1]))
            if segments_intersect(start, goal, c, d):
                return False
    return True


def validate_polyline_xy(points: Sequence[Point2], polygons: Sequence[np.ndarray], margin: float = 1.5) -> bool:
    if len(points) < 2:
        return True
    return all(
        direct_segment_clear(points[i - 1], points[i], polygons, margin)
        for i in range(1, len(points))
    )


def astar_xy(
    start: Point2,
    goal: Point2,
    polygons: Sequence[np.ndarray],
    bounds: Tuple[float, float, float, float],
    step: float = 2.5,
    margin: float = 1.5,
) -> List[Point2]:
    min_x, max_x, min_y, max_y = bounds

    def to_grid(point: Point2) -> Tuple[int, int]:
        return (
            int(round((point[0] - min_x) / step)),
            int(round((point[1] - min_y) / step)),
        )

    def to_world(node: Tuple[int, int]) -> Point2:
        return (min_x + node[0] * step, min_y + node[1] * step)

    start_node = to_grid(start)
    goal_node = to_grid(goal)
    if direct_segment_clear(start, goal, polygons, margin):
        return [start, goal]
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    frontier: List[Tuple[float, float, Tuple[int, int]]] = []
    heapq.heappush(frontier, (0.0, 0.0, start_node))
    came_from: Dict[Tuple[int, int], Tuple[int, int] | None] = {start_node: None}
    cost_so_far: Dict[Tuple[int, int], float] = {start_node: 0.0}

    while frontier:
        _, current_cost, current = heapq.heappop(frontier)
        if current == goal_node:
            break
        for dx, dy in offsets:
            nxt = (current[0] + dx, current[1] + dy)
            world = to_world(nxt)
            if not (min_x <= world[0] <= max_x and min_y <= world[1] <= max_y):
                continue
            if cell_blocked(world, polygons, margin):
                continue
            step_cost = math.hypot(dx, dy) * step
            new_cost = current_cost + step_cost
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + math.dist(world, goal)
                heapq.heappush(frontier, (priority, new_cost, nxt))
                came_from[nxt] = current

    if goal_node not in came_from:
        return [start, goal]
    path_nodes = []
    cursor = goal_node
    while cursor is not None:
        path_nodes.append(cursor)
        cursor = came_from[cursor]
    path_nodes.reverse()
    path = [start]
    for node in path_nodes[1:-1]:
        path.append(to_world(node))
    path.append(goal)
    return path


def path_length_2d(points: Sequence[Point2]) -> float:
    if len(points) < 2:
        return 0.0
    return float(sum(math.dist(points[i - 1], points[i]) for i in range(1, len(points))))


def path_length_3d(points: Sequence[Point3]) -> float:
    if len(points) < 2:
        return 0.0
    return float(sum(math.dist(points[i - 1], points[i]) for i in range(1, len(points))))


def wrap_angle_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def heading_deg_from_delta(dx: float, dy: float) -> float | None:
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None
    return math.degrees(math.atan2(dy, dx))


def climb_angle_deg_from_delta(dx: float, dy: float, dz: float) -> float:
    horizontal = math.hypot(dx, dy)
    if horizontal < 1e-9:
        return 90.0 if dz >= 0.0 else -90.0
    return math.degrees(math.atan2(dz, horizontal))


def prism_blocked_3d(
    point: Point3,
    polygons: Sequence[np.ndarray],
    polygon_names: Sequence[str],
    heights: Dict[str, float],
    margin_xy: float = 0.8,
    margin_z: float = 0.3,
) -> bool:
    px, py, pz = float(point[0]), float(point[1]), float(point[2])
    for polygon, polygon_name in zip(polygons, polygon_names):
        roof = float(heights.get(polygon_name, 24.0))
        if pz > roof + margin_z:
            continue
        bbox = inflate_polygon_bbox(polygon, margin_xy)
        if bbox[0] <= px <= bbox[1] and bbox[2] <= py <= bbox[3]:
            if point_in_polygon_xy(np.array([px, py], dtype=np.float64), polygon):
                return True
    return False


def segment_clear_3d(
    start: Point3,
    goal: Point3,
    polygons: Sequence[np.ndarray],
    polygon_names: Sequence[str],
    heights: Dict[str, float],
    margin_xy: float = 0.8,
    margin_z: float = 0.3,
) -> bool:
    length = math.dist(start, goal)
    steps = max(int(math.ceil(length / 1.0)), 2)
    for i in range(steps + 1):
        t = i / steps
        point = (
            (1.0 - t) * start[0] + t * goal[0],
            (1.0 - t) * start[1] + t * goal[1],
            (1.0 - t) * start[2] + t * goal[2],
        )
        if prism_blocked_3d(point, polygons, polygon_names, heights, margin_xy=margin_xy, margin_z=margin_z):
            return False
    return True


def segment_safe_3d(
    start: Point3,
    goal: Point3,
    polygons: Sequence[np.ndarray],
    polygon_names: Sequence[str],
    heights: Dict[str, float],
    margin_xy: float = 2.0,
    margin_z: float = 0.8,
) -> bool:
    return segment_clear_3d(
        start,
        goal,
        polygons,
        polygon_names,
        heights,
        margin_xy=margin_xy,
        margin_z=margin_z,
    )


def compress_polyline_3d(
    points: Sequence[Point3],
    distance_eps: float = 0.8,
    angle_eps_deg: float = 6.0,
) -> List[Point3]:
    if len(points) <= 2:
        return list(points)
    compact: List[Point3] = [points[0]]
    for point in points[1:]:
        if math.dist(point, compact[-1]) > distance_eps:
            compact.append(point)
    if len(compact) <= 2:
        return compact
    simplified: List[Point3] = [compact[0]]
    cos_threshold = math.cos(math.radians(angle_eps_deg))
    for i in range(1, len(compact) - 1):
        a = np.asarray(simplified[-1], dtype=float)
        b = np.asarray(compact[i], dtype=float)
        c = np.asarray(compact[i + 1], dtype=float)
        ab = b - a
        bc = c - b
        norm_ab = np.linalg.norm(ab)
        norm_bc = np.linalg.norm(bc)
        if norm_ab < 1e-6 or norm_bc < 1e-6:
            continue
        cosine = float(np.dot(ab, bc) / (norm_ab * norm_bc))
        if cosine > cos_threshold:
            continue
        simplified.append(compact[i])
    simplified.append(compact[-1])
    return simplified


def chaikin_smooth_3d(points: Sequence[Point3], iterations: int = 2) -> List[Point3]:
    smoothed = [tuple(point) for point in points]
    for _ in range(iterations):
        if len(smoothed) < 3:
            return smoothed
        refined: List[Point3] = [smoothed[0]]
        for p0, p1 in zip(smoothed[:-1], smoothed[1:]):
            q = (
                0.75 * p0[0] + 0.25 * p1[0],
                0.75 * p0[1] + 0.25 * p1[1],
                0.75 * p0[2] + 0.25 * p1[2],
            )
            r = (
                0.25 * p0[0] + 0.75 * p1[0],
                0.25 * p0[1] + 0.75 * p1[1],
                0.25 * p0[2] + 0.75 * p1[2],
            )
            refined.extend([q, r])
        refined.append(smoothed[-1])
        smoothed = refined
    return smoothed


def line_intersection_2d(
    p1: np.ndarray,
    d1: np.ndarray,
    p2: np.ndarray,
    d2: np.ndarray,
) -> np.ndarray | None:
    matrix = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]], dtype=float)
    rhs = p2 - p1
    det = np.linalg.det(matrix)
    if abs(det) < 1e-8:
        return None
    t, _ = np.linalg.solve(matrix, rhs)
    return p1 + t * d1


def fillet_corner_3d(
    a: Point3,
    b: Point3,
    c: Point3,
    min_turn_radius: float,
    samples_per_90deg: int = 4,
) -> List[Point3] | None:
    pa = np.asarray(a, dtype=float)
    pb = np.asarray(b, dtype=float)
    pc = np.asarray(c, dtype=float)
    v1 = pb[:2] - pa[:2]
    v2 = pc[:2] - pb[:2]
    len1 = float(np.linalg.norm(v1))
    len2 = float(np.linalg.norm(v2))
    if len1 < 1e-6 or len2 < 1e-6:
        return None
    u1 = v1 / len1
    u2 = v2 / len2
    dot = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
    turn_angle = float(math.acos(dot))
    if turn_angle < math.radians(12.0) or abs(math.pi - turn_angle) < math.radians(8.0):
        return None
    tangent = min(
        min_turn_radius * math.tan(turn_angle / 2.0),
        0.35 * len1,
        0.35 * len2,
    )
    if tangent < 1.0:
        return None
    radius = tangent / max(math.tan(turn_angle / 2.0), 1e-6)
    entry_xy = pb[:2] - u1 * tangent
    exit_xy = pb[:2] + u2 * tangent
    turn_sign = 1.0 if (u1[0] * u2[1] - u1[1] * u2[0]) > 0.0 else -1.0
    normal1 = np.array([-u1[1], u1[0]]) * turn_sign
    normal2 = np.array([-u2[1], u2[0]]) * turn_sign
    center = line_intersection_2d(entry_xy, normal1, exit_xy, normal2)
    if center is None:
        return None
    start_angle = math.atan2(entry_xy[1] - center[1], entry_xy[0] - center[0])
    end_angle = math.atan2(exit_xy[1] - center[1], exit_xy[0] - center[0])
    if turn_sign > 0.0 and end_angle <= start_angle:
        end_angle += 2.0 * math.pi
    if turn_sign < 0.0 and end_angle >= start_angle:
        end_angle -= 2.0 * math.pi
    angle_span = abs(end_angle - start_angle)
    samples = max(3, int(math.ceil(angle_span / (math.pi / 2.0) * samples_per_90deg)))
    entry_z = float(pb[2] - tangent / len1 * (pb[2] - pa[2]))
    exit_z = float(pb[2] + tangent / len2 * (pc[2] - pb[2]))
    arc_points: List[Point3] = []
    for idx in range(samples + 1):
        t = idx / samples
        angle = start_angle + (end_angle - start_angle) * t
        z = entry_z + (exit_z - entry_z) * t
        arc_points.append(
            (
                float(center[0] + radius * math.cos(angle)),
                float(center[1] + radius * math.sin(angle)),
                float(z),
            )
        )
    return arc_points


def motion_primitive_smooth_3d(
    points: Sequence[Point3],
    polygons: Sequence[np.ndarray],
    polygon_names: Sequence[str],
    heights: Dict[str, float],
    min_turn_radius: float = 10.0,
    max_climb_deg: float = 35.0,
) -> List[Point3]:
    if len(points) < 3:
        return list(points)
    deduped = compress_polyline_3d(points)
    if len(deduped) < 3:
        return deduped
    smoothed: List[Point3] = [deduped[0]]
    for i in range(1, len(deduped) - 1):
        a, b, c = deduped[i - 1], deduped[i], deduped[i + 1]
        arc = fillet_corner_3d(a, b, c, min_turn_radius=min_turn_radius)
        if arc is None:
            if math.dist(smoothed[-1], b) > 1e-6:
                smoothed.append(b)
            continue
        if math.dist(smoothed[-1], arc[0]) > 1e-6:
            smoothed.append(arc[0])
        smoothed.extend(arc[1:-1])
        smoothed.append(arc[-1])
    if math.dist(smoothed[-1], deduped[-1]) > 1e-6:
        smoothed.append(deduped[-1])
    min_z = max(2.0, min(point[2] for point in deduped) - 0.5)
    max_z = max(max(heights.values(), default=24.0) + 12.0, max(point[2] for point in deduped) + 0.5)
    clipped: List[Point3] = [
        (float(point[0]), float(point[1]), float(np.clip(point[2], min_z, max_z)))
        for point in smoothed
    ]
    for p0, p1 in zip(clipped[:-1], clipped[1:]):
        climb = abs(climb_angle_deg_from_delta(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]))
        if climb > max_climb_deg + 1e-6:
            return deduped
    if not all(
        segment_clear_3d(clipped[i - 1], clipped[i], polygons, polygon_names, heights)
        for i in range(1, len(clipped))
    ):
        return deduped
    raw_length = path_length_3d(deduped)
    smooth_length = path_length_3d(clipped)
    if smooth_length < 0.92 * raw_length or smooth_length > 1.10 * raw_length:
        return deduped
    return clipped


def quadratic_bezier_3d(a: Point3, b: Point3, c: Point3, samples: int = 10) -> List[Point3]:
    pa = np.asarray(a, dtype=float)
    pb = np.asarray(b, dtype=float)
    pc = np.asarray(c, dtype=float)
    curve: List[Point3] = []
    for idx in range(samples + 1):
        t = idx / samples
        point = ((1 - t) ** 2) * pa + 2 * (1 - t) * t * pb + (t ** 2) * pc
        curve.append((float(point[0]), float(point[1]), float(point[2])))
    return curve


def smooth_vertical_transition_triplets(
    points: Sequence[Point3],
    polygons: Sequence[np.ndarray],
    polygon_names: Sequence[str],
    heights: Dict[str, float],
    max_climb_deg: float = 35.0,
) -> List[Point3]:
    if len(points) < 3:
        return list(points)
    refined: List[Point3] = [points[0]]
    i = 1
    while i < len(points) - 1:
        a = refined[-1]
        b = points[i]
        c = points[i + 1]
        ab_xy = math.hypot(b[0] - a[0], b[1] - a[1])
        bc_xy = math.hypot(c[0] - b[0], c[1] - b[1])
        vertical_pattern = (ab_xy < 1e-6 and bc_xy > 1.0) or (ab_xy > 1.0 and bc_xy < 1e-6)
        if vertical_pattern:
            curve = quadratic_bezier_3d(a, b, c, samples=8)
            valid = True
            for p0, p1 in zip(curve[:-1], curve[1:]):
                if not segment_clear_3d(p0, p1, polygons, polygon_names, heights):
                    valid = False
                    break
                climb = abs(climb_angle_deg_from_delta(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]))
                if climb > max_climb_deg + 1e-6:
                    valid = False
                    break
            if valid:
                refined.extend(curve[1:])
                i += 2
                continue
        if math.dist(refined[-1], b) > 1e-6:
            refined.append(b)
        i += 1
    if math.dist(refined[-1], points[-1]) > 1e-6:
        refined.append(points[-1])
    return refined


def cubic_hermite_refine_3d(
    points: Sequence[Point3],
    polygons: Sequence[np.ndarray],
    polygon_names: Sequence[str],
    heights: Dict[str, float],
    max_climb_deg: float = 35.0,
    samples_per_segment: int = 4,
) -> List[Point3]:
    if len(points) < 3:
        return list(points)
    p = np.asarray(points, dtype=float)
    tangents = np.zeros_like(p)
    tangents[0] = 0.5 * (p[1] - p[0])
    tangents[-1] = 0.5 * (p[-1] - p[-2])
    for i in range(1, len(p) - 1):
        tangents[i] = 0.35 * (p[i + 1] - p[i - 1])
    refined: List[Point3] = [tuple(p[0].tolist())]
    for i in range(len(p) - 1):
        p0 = p[i]
        p1 = p[i + 1]
        m0 = tangents[i]
        m1 = tangents[i + 1]
        for s in range(1, samples_per_segment + 1):
            t = s / samples_per_segment
            h00 = 2 * t ** 3 - 3 * t ** 2 + 1
            h10 = t ** 3 - 2 * t ** 2 + t
            h01 = -2 * t ** 3 + 3 * t ** 2
            h11 = t ** 3 - t ** 2
            point = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
            refined.append((float(point[0]), float(point[1]), float(point[2])))
    min_z = max(2.0, min(point[2] for point in points) - 0.25)
    max_z = max(max(heights.values(), default=24.0) + 12.0, max(point[2] for point in points) + 0.25)
    clipped = [
        (float(point[0]), float(point[1]), float(np.clip(point[2], min_z, max_z)))
        for point in refined
    ]
    raw_length = path_length_3d(points)
    smooth_length = path_length_3d(clipped)
    if smooth_length < 0.92 * raw_length or smooth_length > 1.18 * raw_length:
        return list(points)
    for p0, p1 in zip(clipped[:-1], clipped[1:]):
        if not segment_clear_3d(p0, p1, polygons, polygon_names, heights):
            return list(points)
        climb = abs(climb_angle_deg_from_delta(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]))
        if climb > max_climb_deg + 1e-6:
            return list(points)
    return clipped


def spiral_transition_3d(
    start: Point3,
    end: Point3,
    turns: float = 0.75,
    radius: float = 7.5,
    samples: int = 24,
) -> List[Point3]:
    start_xy = np.asarray(start[:2], dtype=float)
    end_xy = np.asarray(end[:2], dtype=float)
    direction = end_xy - start_xy
    dist = float(np.linalg.norm(direction))
    if dist < 1e-6:
        direction = np.array([1.0, 0.0], dtype=float)
        dist = 1.0
    direction = direction / dist
    normal = np.array([-direction[1], direction[0]], dtype=float)
    center = start_xy + normal * radius
    start_angle = math.atan2(start_xy[1] - center[1], start_xy[0] - center[0])
    end_angle = start_angle + 2.0 * math.pi * turns
    path: List[Point3] = []
    for idx in range(samples + 1):
        t = idx / samples
        angle = start_angle + (end_angle - start_angle) * t
        lateral_scale = min(1.0, max(0.0, t))
        point = (
            float(center[0] + radius * lateral_scale * math.cos(angle)),
            float(center[1] + radius * lateral_scale * math.sin(angle)),
            float(start[2] + (end[2] - start[2]) * t),
        )
        path.append(point)
    path[0] = tuple(start)
    path[-1] = tuple(end)
    return path


def polynomial_refine_3d(
    points: Sequence[Point3],
    polygons: Sequence[np.ndarray],
    polygon_names: Sequence[str],
    heights: Dict[str, float],
    sample_factor: int = 6,
) -> List[Point3]:
    if len(points) < 4:
        return list(points)
    deduped = compress_polyline_3d(points, distance_eps=0.6, angle_eps_deg=4.0)
    if len(deduped) < 4:
        return deduped
    t = [0.0]
    for i in range(1, len(deduped)):
        t.append(t[-1] + math.dist(deduped[i - 1], deduped[i]))
    t_arr = np.asarray(t, dtype=float)
    xyz = np.asarray(deduped, dtype=float)
    if t_arr[-1] < 1e-6:
        return deduped
    v0 = xyz[1] - xyz[0]
    v1 = xyz[-1] - xyz[-2]
    try:
        spline_x = CubicSpline(t_arr, xyz[:, 0], bc_type=((1, float(v0[0])), (1, float(v1[0]))))
        spline_y = CubicSpline(t_arr, xyz[:, 1], bc_type=((1, float(v0[1])), (1, float(v1[1]))))
        spline_z = CubicSpline(t_arr, xyz[:, 2], bc_type=((1, float(v0[2])), (1, float(v1[2]))))
        samples = max(len(deduped) * sample_factor, len(deduped) + 12)
        sample_t = np.linspace(0.0, float(t_arr[-1]), samples)
        refined = [
            (float(spline_x(tt)), float(spline_y(tt)), float(spline_z(tt)))
            for tt in sample_t
        ]
        refined[0] = tuple(deduped[0])
        refined[-1] = tuple(deduped[-1])
        min_z = max(2.0, min(pt[2] for pt in deduped) - 0.5)
        max_z = max(max(heights.values(), default=24.0) + 12.0, max(pt[2] for pt in deduped) + 0.5)
        refined = [(x, y, float(np.clip(z, min_z, max_z))) for x, y, z in refined]
        if not all(
            segment_clear_3d(refined[i - 1], refined[i], polygons, polygon_names, heights)
            for i in range(1, len(refined))
        ):
            return deduped
        raw_length = path_length_3d(deduped)
        refined_length = path_length_3d(refined)
        if refined_length < 0.88 * raw_length or refined_length > 1.18 * raw_length:
            return deduped
        return refined
    except Exception:
        return deduped


def build_roadmap_nodes(
    polygons: Sequence[np.ndarray],
    margin: float,
) -> List[Point2]:
    nodes: List[Point2] = []
    for polygon in polygons:
        min_x, max_x, min_y, max_y = inflate_polygon_bbox(polygon, margin)
        nodes.extend(
            [
                (min_x, min_y),
                (min_x, max_y),
                (max_x, min_y),
                (max_x, max_y),
                ((min_x + max_x) * 0.5, min_y),
                ((min_x + max_x) * 0.5, max_y),
                (min_x, (min_y + max_y) * 0.5),
                (max_x, (min_y + max_y) * 0.5),
            ]
        )
    deduped: List[Point2] = []
    for node in nodes:
        if all(math.dist(node, existing) > 1e-6 for existing in deduped):
            deduped.append(node)
    return deduped


def build_visibility_adjacency(
    nodes: Sequence[Point2],
    polygons: Sequence[np.ndarray],
    margin: float,
) -> Dict[int, List[Tuple[int, float]]]:
    adjacency: Dict[int, List[Tuple[int, float]]] = {idx: [] for idx in range(len(nodes))}
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if direct_segment_clear(nodes[i], nodes[j], polygons, margin):
                cost = math.dist(nodes[i], nodes[j])
                adjacency[i].append((j, cost))
                adjacency[j].append((i, cost))
    return adjacency


def dijkstra_route(
    start_idx: int,
    goal_idx: int,
    nodes: Sequence[Point2],
    adjacency: Dict[int, List[Tuple[int, float]]],
) -> List[Point2]:
    frontier: List[Tuple[float, int]] = [(0.0, start_idx)]
    came_from: Dict[int, int | None] = {start_idx: None}
    costs: Dict[int, float] = {start_idx: 0.0}
    while frontier:
        current_cost, current = heapq.heappop(frontier)
        if current == goal_idx:
            break
        if current_cost > costs[current] + 1e-9:
            continue
        for nxt, edge_cost in adjacency[current]:
            new_cost = current_cost + edge_cost
            if nxt not in costs or new_cost + 1e-9 < costs[nxt]:
                costs[nxt] = new_cost
                came_from[nxt] = current
                heapq.heappush(frontier, (new_cost, nxt))
    if goal_idx not in came_from:
        return []
    index_path = []
    cursor: int | None = goal_idx
    while cursor is not None:
        index_path.append(cursor)
        cursor = came_from[cursor]
    index_path.reverse()
    return [nodes[idx] for idx in index_path]


def bspline_smooth_xy(points: Sequence[Point2], sample_factor: int = 10) -> List[Point2]:
    if len(points) < 4:
        return list(points)
    deduped = [points[0]]
    for point in points[1:]:
        if math.dist(point, deduped[-1]) > 1e-6:
            deduped.append(point)
    if len(deduped) < 4:
        return deduped
    try:
        x = np.array([point[0] for point in deduped], dtype=float)
        y = np.array([point[1] for point in deduped], dtype=float)
        tck, _ = splprep([x, y], s=0.0, k=min(3, len(deduped) - 1))
        samples = max(len(deduped) * sample_factor, len(deduped) + 8)
        u_new = np.linspace(0.0, 1.0, samples)
        x_new, y_new = splev(u_new, tck)
        return [(float(px), float(py)) for px, py in zip(x_new, y_new)]
    except Exception:
        return deduped


def compress_xy_polyline(
    points: Sequence[Point2],
    distance_eps: float = 1.0,
    turn_sin_eps: float = 0.03,
) -> List[Point2]:
    if len(points) <= 2:
        return list(points)
    compact: List[Point2] = [points[0]]
    for point in points[1:]:
        if math.dist(point, compact[-1]) > distance_eps:
            compact.append(point)
    if len(compact) <= 2:
        return compact
    simplified: List[Point2] = [compact[0]]
    for i in range(1, len(compact) - 1):
        a = np.asarray(simplified[-1], dtype=float)
        b = np.asarray(compact[i], dtype=float)
        c = np.asarray(compact[i + 1], dtype=float)
        ab = b - a
        bc = c - b
        norm_ab = np.linalg.norm(ab)
        norm_bc = np.linalg.norm(bc)
        if norm_ab < 1e-6 or norm_bc < 1e-6:
            continue
        cross = abs(ab[0] * bc[1] - ab[1] * bc[0]) / (norm_ab * norm_bc)
        if cross < turn_sin_eps:
            continue
        simplified.append(compact[i])
    simplified.append(compact[-1])
    return simplified


def smooth_xy_segment(
    points: Sequence[Point2],
    polygons: Sequence[np.ndarray],
    margin: float = 1.25,
) -> List[Point2]:
    if len(points) < 3:
        return list(points)
    raw = compress_xy_polyline(points)
    if len(raw) < 3:
        return raw
    smoothed = bspline_smooth_xy(raw, sample_factor=12)
    raw_length = path_length_2d(raw)
    smooth_length = path_length_2d(smoothed)
    if not validate_polyline_xy(smoothed, polygons, margin):
        return raw
    if smooth_length < 0.90 * raw_length or smooth_length > 1.18 * raw_length:
        return raw
    if math.dist(smoothed[0], raw[0]) > 1.0 or math.dist(smoothed[-1], raw[-1]) > 1.0:
        return raw
    smoothed[0] = raw[0]
    smoothed[-1] = raw[-1]
    return smoothed


def lift_xy_path_to_3d(
    xy_path: Sequence[Point2],
    z_start: float,
    z_goal: float,
    cruise_z: float,
) -> List[Point3]:
    path_3d: List[Point3] = []
    if not xy_path:
        return path_3d
    path_3d.append((xy_path[0][0], xy_path[0][1], z_start))
    if cruise_z > z_start + 1e-6:
        path_3d.append((xy_path[0][0], xy_path[0][1], cruise_z))
    for point in xy_path[1:-1]:
        path_3d.append((point[0], point[1], cruise_z))
    if cruise_z > z_goal + 1e-6:
        path_3d.append((xy_path[-1][0], xy_path[-1][1], cruise_z))
    path_3d.append((xy_path[-1][0], xy_path[-1][1], z_goal))
    compact = [path_3d[0]]
    for point in path_3d[1:]:
        if math.dist(point, compact[-1]) > 1e-6:
            compact.append(point)
    return compact


def point_to_segment_distance_xy(point: Point2, start: Point2, goal: Point2) -> float:
    px, py = float(point[0]), float(point[1])
    sx, sy = float(start[0]), float(start[1])
    gx, gy = float(goal[0]), float(goal[1])
    dx = gx - sx
    dy = gy - sy
    denom = dx * dx + dy * dy
    if denom < 1e-9:
        return math.hypot(px - sx, py - sy)
    t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / denom))
    proj_x = sx + t * dx
    proj_y = sy + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def local_path_roof_height(
    xy_path: Sequence[Point2],
    polygons: Sequence[np.ndarray],
    polygon_names: Sequence[str],
    heights: Dict[str, float],
    corridor_margin: float = 6.0,
) -> float:
    if len(xy_path) < 2:
        return 0.0
    local_max = 0.0
    for polygon, polygon_name in zip(polygons, polygon_names):
        roof = float(heights.get(polygon_name, 24.0))
        centroid = np.mean(polygon, axis=0)
        centroid_xy = (float(centroid[0]), float(centroid[1]))
        hit = False
        for i in range(1, len(xy_path)):
            a = xy_path[i - 1]
            b = xy_path[i]
            bbox = inflate_polygon_bbox(polygon, corridor_margin)
            if (
                max(a[0], b[0]) < bbox[0]
                or min(a[0], b[0]) > bbox[1]
                or max(a[1], b[1]) < bbox[2]
                or min(a[1], b[1]) > bbox[3]
            ):
                continue
            if point_to_segment_distance_xy(centroid_xy, a, b) <= corridor_margin:
                hit = True
                break
        if hit:
            local_max = max(local_max, roof)
    return local_max


def side_detour_xy_path(
    start: Point2,
    goal: Point2,
    polygons: Sequence[np.ndarray],
    bounds: Tuple[float, float, float, float],
    margin: float = 2.0,
) -> List[Point2]:
    nodes: List[Point2] = [start, goal]
    for node in build_roadmap_nodes(polygons, margin + 1.6):
        if not cell_blocked(node, polygons, margin):
            nodes.append(node)
    adjacency = build_visibility_adjacency(nodes, polygons, margin)
    routed = dijkstra_route(0, 1, nodes, adjacency)
    if routed:
        routed = compress_xy_polyline(routed, distance_eps=0.8, turn_sin_eps=0.025)
        return smooth_xy_segment(routed, polygons, margin=max(1.1, margin - 0.4))
    return astar_xy(start, goal, polygons, bounds, step=2.5, margin=margin)


def astar_3d(
    start: Point3,
    goal: Point3,
    polygons: Sequence[np.ndarray],
    polygon_names: Sequence[str],
    heights: Dict[str, float],
    bounds: Tuple[float, float, float, float],
    z_limits: Tuple[float, float],
    step_xy: float = 4.0,
    step_z: float = 2.5,
    max_climb_deg: float = 35.0,
    min_turn_radius: float = 10.0,
    safety_margin_xy: float = 2.0,
    safety_margin_z: float = 0.8,
) -> List[Point3]:
    min_x, max_x, min_y, max_y = bounds
    min_z, max_z = z_limits

    def to_grid(point: Point3) -> Tuple[int, int, int]:
        return (
            int(round((point[0] - min_x) / step_xy)),
            int(round((point[1] - min_y) / step_xy)),
            int(round((point[2] - min_z) / step_z)),
        )

    def to_world(node: Tuple[int, int, int]) -> Point3:
        return (
            min_x + node[0] * step_xy,
            min_y + node[1] * step_xy,
            min_z + node[2] * step_z,
        )

    start_node = to_grid(start)
    goal_node = to_grid(goal)
    if segment_safe_3d(start, goal, polygons, polygon_names, heights, margin_xy=safety_margin_xy, margin_z=safety_margin_z):
        return [start, goal]

    offsets: List[Tuple[int, int, int]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                if dx == 0 and dy == 0:
                    continue
                offsets.append((dx, dy, dz))
    max_turn_deg = math.degrees(2.0 * math.asin(min(1.0, step_xy / max(2.0 * min_turn_radius, step_xy + 1e-6))))
    frontier: List[Tuple[float, float, Tuple[int, int, int], float | None]] = []
    heapq.heappush(frontier, (0.0, 0.0, start_node, None))
    came_from: Dict[Tuple[Tuple[int, int, int], float | None], Tuple[Tuple[int, int, int], float | None] | None] = {
        (start_node, None): None
    }
    cost_so_far: Dict[Tuple[Tuple[int, int, int], float | None], float] = {(start_node, None): 0.0}
    best_goal_state: Tuple[Tuple[int, int, int], float | None] | None = None

    while frontier:
        _, current_cost, current, prev_heading = heapq.heappop(frontier)
        current_state = (current, prev_heading)
        if current_cost > cost_so_far.get(current_state, float("inf")) + 1e-9:
            continue
        current_world = to_world(current)
        if math.dist(current_world, goal) <= max(step_xy * 1.3, step_z * 1.6):
            best_goal_state = current_state
            break
        for dx, dy, dz in offsets:
            nxt = (current[0] + dx, current[1] + dy, current[2] + dz)
            nxt_world = to_world(nxt)
            if not (min_x <= nxt_world[0] <= max_x and min_y <= nxt_world[1] <= max_y and min_z <= nxt_world[2] <= max_z):
                continue
            if prism_blocked_3d(nxt_world, polygons, polygon_names, heights, margin_xy=safety_margin_xy, margin_z=safety_margin_z):
                continue
            if not segment_safe_3d(current_world, nxt_world, polygons, polygon_names, heights, margin_xy=safety_margin_xy, margin_z=safety_margin_z):
                continue
            heading = heading_deg_from_delta(dx, dy)
            climb_deg = abs(climb_angle_deg_from_delta(dx * step_xy, dy * step_xy, dz * step_z))
            if climb_deg > max_climb_deg:
                continue
            turn_penalty = 0.0
            if prev_heading is not None and heading is not None:
                heading_change = abs(wrap_angle_deg(heading - prev_heading))
                if heading_change > max_turn_deg + 1e-6:
                    continue
                turn_penalty = 0.18 * (heading_change / max(max_turn_deg, 1e-6))
            climb_penalty = 0.10 * (climb_deg / max(max_climb_deg, 1e-6))
            step_cost = math.dist(current_world, nxt_world) * (1.0 + turn_penalty + climb_penalty)
            nxt_state = (nxt, heading)
            new_cost = current_cost + step_cost
            if new_cost + 1e-9 < cost_so_far.get(nxt_state, float("inf")):
                cost_so_far[nxt_state] = new_cost
                heuristic = math.dist(nxt_world, goal)
                heapq.heappush(frontier, (new_cost + heuristic, new_cost, nxt, heading))
                came_from[nxt_state] = current_state

    if best_goal_state is None:
        # Side-detour first, then use only a local obstacle-aware cruise height.
        xy_path = side_detour_xy_path(
            (float(start[0]), float(start[1])),
            (float(goal[0]), float(goal[1])),
            polygons,
            bounds,
            margin=safety_margin_xy,
        )
        local_roof = local_path_roof_height(
            xy_path,
            polygons,
            polygon_names,
            heights,
            corridor_margin=max(5.0, safety_margin_xy + 2.0),
        )
        local_cruise_z = max(
            float(start[2]),
            float(goal[2]),
            local_roof + max(2.0, safety_margin_z + 1.2),
        )
        local_cruise_z = min(local_cruise_z, max_z)
        fallback = lift_xy_path_to_3d(xy_path, float(start[2]), float(goal[2]), float(local_cruise_z))
        fallback = compress_polyline_3d(fallback)
        if len(fallback) >= 2 and all(
            segment_safe_3d(
                fallback[i - 1],
                fallback[i],
                polygons,
                polygon_names,
                heights,
                margin_xy=safety_margin_xy,
                margin_z=safety_margin_z,
            )
            for i in range(1, len(fallback))
        ):
            return fallback
        # Final conservative fallback.
        cruise_z = max(float(start[2]), float(goal[2]), min(max_z, local_roof + 4.0))
        fallback = [
            start,
            (start[0], start[1], cruise_z),
            (goal[0], goal[1], cruise_z),
            goal,
        ]
        return compress_polyline_3d(fallback)

    nodes: List[Point3] = [goal]
    cursor: Tuple[Tuple[int, int, int], float | None] | None = best_goal_state
    while cursor is not None:
        nodes.append(to_world(cursor[0]))
        cursor = came_from[cursor]
    nodes.append(start)
    nodes.reverse()
    return compress_polyline_3d(nodes)


def try_dubins_connection_3d(
    start: Point3,
    goal: Point3,
    polygons: Sequence[np.ndarray],
    polygon_names: Sequence[str],
    heights: Dict[str, float],
    min_turn_radius: float = 10.0,
    safety_margin_xy: float = 2.0,
    safety_margin_z: float = 0.8,
) -> List[Point3] | None:
    if not segment_safe_3d(start, goal, polygons, polygon_names, heights, margin_xy=safety_margin_xy, margin_z=safety_margin_z):
        return None
    direct_distance = math.dist(start, goal)
    if direct_distance < 1.5:
        return [start, goal]
    start_arr = np.asarray(start, dtype=float)
    goal_arr = np.asarray(goal, dtype=float)
    direction = goal_arr - start_arr
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return [start, goal]
    unit = direction / norm
    lead = min(max(min_turn_radius * 0.6, 3.0), 0.25 * direct_distance)
    mid1 = tuple((start_arr + unit * lead).tolist())
    mid2 = tuple((goal_arr - unit * lead).tolist())
    raw = compress_polyline_3d([start, mid1, mid2, goal], distance_eps=0.1, angle_eps_deg=2.0)
    curved = motion_primitive_smooth_3d(
        raw,
        polygons,
        polygon_names,
        heights,
        min_turn_radius=min_turn_radius,
        max_climb_deg=35.0,
    )
    if len(curved) < 2:
        return None
    if not all(
        segment_safe_3d(
            curved[i - 1],
            curved[i],
            polygons,
            polygon_names,
            heights,
            margin_xy=safety_margin_xy,
            margin_z=safety_margin_z,
        )
        for i in range(1, len(curved))
    ):
        return None
    curved_length = path_length_3d(curved)
    if curved_length > 1.25 * direct_distance:
        return None
    return curved


def pairwise_astar_data(
    points: np.ndarray,
    polygons: Sequence[np.ndarray],
    polygon_names: Sequence[str],
    heights: Dict[str, float],
    owners: Sequence[str],
    bounds: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, Dict[Tuple[int, int], List[Point3]]]:
    count = len(points)
    costs = np.zeros((count + 1, count + 1), dtype=np.float32)
    paths_3d: Dict[Tuple[int, int], List[Point3]] = {}
    start = np.array([bounds[0] + 4.0, bounds[2] + 4.0, max(6.0, float(np.min(points[:, 2])) if len(points) else 6.0)], dtype=np.float64)
    all_points = np.vstack([start[None, :], points])
    max_roof = max(heights.values(), default=24.0)
    min_z = max(2.0, float(np.min(all_points[:, 2])) - 2.0)
    max_z = max(max_roof + 12.0, float(np.max(all_points[:, 2])) + 8.0)
    z_limits = (min_z, max_z)
    for i in range(count + 1):
        for j in range(i + 1, count + 1):
            start_point = tuple(float(x) for x in all_points[i])
            goal_point = tuple(float(x) for x in all_points[j])
            dubins_path = try_dubins_connection_3d(
                start_point,
                goal_point,
                polygons,
                polygon_names,
                heights,
                min_turn_radius=10.0,
            )
            if dubins_path is not None:
                path_3d = dubins_path
                cost = path_length_3d(path_3d)
                paths_3d[(i, j)] = path_3d
                paths_3d[(j, i)] = list(reversed(path_3d))
            else:
                xy_dist = math.dist(start_point[:2], goal_point[:2])
                z_penalty = abs(start_point[2] - goal_point[2])
                cost = 1.35 * xy_dist + 1.10 * z_penalty + 18.0
                paths_3d[(i, j)] = []
                paths_3d[(j, i)] = []
            costs[i, j] = cost
            costs[j, i] = cost
    return costs, paths_3d


def build_path_scenario(
    scenario: Coarse3DScenario,
    selection: SelectionResult,
    scene_json: Path,
    return_to_start: bool = True,
) -> PathScenario3D:
    polygons, polygon_names, heights = read_scene_obstacles(scene_json)
    selected_indices = list(selection.selected_indices)
    selected_points = selected_variant_points_world(
        scenario,
        selected_indices,
        selected_layer_ids=selection.selected_layer_ids if selection.selected_layer_ids else None,
        selected_standoff_factors=selection.selected_standoff_factors if selection.selected_standoff_factors else None,
    )
    owners = [scenario.candidate_owners[index] for index in selected_indices]
    bounds = scene_bounds(polygons, selected_points)
    pair_costs, paths_3d = pairwise_astar_data(selected_points, polygons, polygon_names, heights, owners, bounds)
    start = np.array([bounds[0] + 4.0, bounds[2] + 4.0, max(6.0, float(np.min(selected_points[:, 2])) if len(selected_points) else 6.0)], dtype=np.float64)
    max_roof = max(heights.values(), default=24.0)
    min_z = max(2.0, float(np.min(np.vstack([start[None, :], selected_points])[:, 2])) - 2.0)
    max_z = max(max_roof + 12.0, float(np.max(np.vstack([start[None, :], selected_points])[:, 2])) + 8.0)
    standoff_distances = np.linalg.norm(selected_points - scenario.aims_world[selected_indices], axis=1).astype(np.float32) if len(selected_points) else np.zeros((0,), dtype=np.float32)
    return PathScenario3D(
        method=selection.method,
        selected_indices=selected_indices,
        selected_points=selected_points,
        aims_world=scenario.aims_world[selected_indices],
        region_codes=scenario.candidate_region_codes[selected_indices],
        cluster_ids=scenario.candidate_cluster_ids[selected_indices],
        sector_ids=scenario.candidate_sector_ids[selected_indices],
        shot_counts=np.asarray(selection.shot_counts, dtype=np.int32),
        standoff_distances=standoff_distances,
        owners=owners,
        start=start,
        pair_costs=pair_costs,
        paths_3d=paths_3d,
        polygons=polygons,
        polygon_names=polygon_names,
        heights=heights,
        bounds=bounds,
        z_limits=(min_z, max_z),
        return_to_start=bool(return_to_start),
    )


def nearest_centroid_order(
    keys: Sequence[str],
    centroids: Dict[str, np.ndarray],
    start_xy: np.ndarray,
) -> List[str]:
    pending = list(keys)
    current = np.asarray(start_xy[:2], dtype=np.float64)
    order: List[str] = []
    while pending:
        next_key = min(pending, key=lambda item: float(np.linalg.norm(centroids[item] - current)))
        order.append(next_key)
        current = centroids[next_key]
        pending.remove(next_key)
    return order


def order_points_from_anchor(
    indices: Sequence[int],
    points: np.ndarray,
    current_point: np.ndarray,
) -> List[int]:
    if not indices:
        return []
    if len(indices) == 1:
        return [int(indices[0])]
    remaining = list(int(idx) for idx in indices)
    route: List[int] = []
    current = np.asarray(current_point, dtype=np.float64)
    while remaining:
        next_idx = min(remaining, key=lambda idx: float(np.linalg.norm(points[idx] - current)))
        route.append(next_idx)
        current = points[next_idx]
        remaining.remove(next_idx)
    subset_points = points[route]
    subset_route, _, _ = route_multistart_optimize(
        list(range(len(route))),
        subset_points,
        restarts=8,
        proposals=120,
        turn_weight=1.5,
    )
    return [route[idx] for idx in subset_route]


def local_route_to_world_indices(path_scenario: PathScenario3D, local_indices: Sequence[int]) -> List[int]:
    return [path_scenario.selected_indices[index] for index in local_indices]


def route_cost_from_pairs(path_scenario: PathScenario3D, local_order: Sequence[int]) -> float:
    if not local_order:
        return 0.0
    nodes = [0] + [int(index) + 1 for index in local_order] + [0]
    return float(
        sum(float(path_scenario.pair_costs[left, right]) for left, right in zip(nodes[:-1], nodes[1:]))
    )


def two_opt_refine(path_scenario: PathScenario3D, local_order: Sequence[int], max_passes: int = 3) -> List[int]:
    best = list(int(index) for index in local_order)
    if len(best) < 4:
        return best
    best_cost = route_cost_from_pairs(path_scenario, best)
    for _ in range(max(int(max_passes), 1)):
        improved = False
        for i in range(len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                if j - i < 1:
                    continue
                candidate = best[:i] + list(reversed(best[i : j + 1])) + best[j + 1 :]
                candidate_cost = route_cost_from_pairs(path_scenario, candidate)
                if candidate_cost + 1e-6 < best_cost:
                    best = candidate
                    best_cost = candidate_cost
                    improved = True
        if not improved:
            break
    return best


def order_mainline(path_scenario: PathScenario3D) -> List[int]:
    local_points = path_scenario.selected_points
    route, _, _ = route_multistart_optimize(
        list(range(len(local_points))),
        local_points,
        restarts=24,
        proposals=800,
        turn_weight=2.0,
    )
    return two_opt_refine(path_scenario, route, max_passes=2)


def assemble_path_3d(path_scenario: PathScenario3D, order: Sequence[int]) -> List[Point3]:
    node_order = [0] + [index + 1 for index in order]
    if path_scenario.return_to_start:
        node_order.append(0)
    path_3d: List[Point3] = [tuple(path_scenario.start.tolist())]
    for prev_node, next_node in zip(node_order, node_order[1:]):
        segment_key = (prev_node, next_node)
        cached_segment = path_scenario.paths_3d.get(segment_key, [])
        if not cached_segment:
            if prev_node == 0:
                start_point = tuple(float(x) for x in path_scenario.start)
            else:
                start_point = tuple(float(x) for x in path_scenario.selected_points[prev_node - 1])
            if next_node == 0:
                goal_point = tuple(float(x) for x in path_scenario.start)
            else:
                goal_point = tuple(float(x) for x in path_scenario.selected_points[next_node - 1])
            cached_segment = try_dubins_connection_3d(
                start_point,
                goal_point,
                path_scenario.polygons,
                path_scenario.polygon_names,
                path_scenario.heights,
                min_turn_radius=10.0,
            )
            if cached_segment is None:
                cached_segment = astar_3d(
                    start_point,
                    goal_point,
                    path_scenario.polygons,
                    path_scenario.polygon_names,
                    path_scenario.heights,
                    path_scenario.bounds,
                    path_scenario.z_limits,
                )
            path_scenario.paths_3d[segment_key] = cached_segment
            path_scenario.paths_3d[(next_node, prev_node)] = list(reversed(cached_segment))
        segment_3d = motion_primitive_smooth_3d(
            cached_segment,
            path_scenario.polygons,
            path_scenario.polygon_names,
            path_scenario.heights,
        )
        segment_3d = smooth_leg_3d(
            segment_3d,
            path_scenario.polygons,
            path_scenario.polygon_names,
            path_scenario.heights,
        )
        path_3d.extend(segment_3d[1:])
    return path_3d


def smooth_path_3d(
    path_3d: Sequence[Point3],
    polygons: Sequence[np.ndarray],
    polygon_names: Sequence[str],
    heights: Dict[str, float],
) -> List[Point3]:
    stage1 = smooth_vertical_transition_triplets(
        path_3d,
        polygons,
        polygon_names,
        heights,
    )
    stage2 = polynomial_refine_3d(
        stage1,
        polygons,
        polygon_names,
        heights,
    )
    stage3 = cubic_hermite_refine_3d(
        stage2,
        polygons,
        polygon_names,
        heights,
    )
    return motion_primitive_smooth_3d(
        stage3,
        polygons,
        polygon_names,
        heights,
        min_turn_radius=10.0,
        max_climb_deg=35.0,
    )


def dense_curve_refine_3d(
    path_3d: Sequence[Point3],
    polygons: Sequence[np.ndarray],
    polygon_names: Sequence[str],
    heights: Dict[str, float],
    sample_factor: int = 10,
) -> List[Point3]:
    """Densify a safe 3D path into a smoother executable-looking curve.

    This does not change the route structure. It only reparameterizes the
    already-safe polyline with a chord-length cubic spline, then validates the
    denser curve against obstacle and climb constraints.
    """

    if len(path_3d) < 4:
        return list(path_3d)
    deduped = compress_polyline_3d(path_3d, distance_eps=0.4, angle_eps_deg=8.0)
    if len(deduped) < 4:
        return deduped

    arc = [0.0]
    for i in range(1, len(deduped)):
        arc.append(arc[-1] + math.dist(deduped[i - 1], deduped[i]))
    arc_arr = np.asarray(arc, dtype=float)
    if arc_arr[-1] < 1e-6:
        return deduped

    xyz = np.asarray(deduped, dtype=float)
    v0 = xyz[1] - xyz[0]
    v1 = xyz[-1] - xyz[-2]
    try:
        spline_x = CubicSpline(arc_arr, xyz[:, 0], bc_type=((1, float(v0[0])), (1, float(v1[0]))))
        spline_y = CubicSpline(arc_arr, xyz[:, 1], bc_type=((1, float(v0[1])), (1, float(v1[1]))))
        spline_z = CubicSpline(arc_arr, xyz[:, 2], bc_type=((1, float(v0[2])), (1, float(v1[2]))))
        samples = max(len(deduped) * sample_factor, len(deduped) + 24)
        sample_t = np.linspace(0.0, float(arc_arr[-1]), samples)
        refined: List[Point3] = [
            (float(spline_x(tt)), float(spline_y(tt)), float(spline_z(tt)))
            for tt in sample_t
        ]
        refined[0] = tuple(deduped[0])
        refined[-1] = tuple(deduped[-1])
        min_z = max(2.0, min(point[2] for point in deduped) - 0.5)
        max_z = max(max(heights.values(), default=24.0) + 12.0, max(point[2] for point in deduped) + 0.5)
        refined = [
            (float(x), float(y), float(np.clip(z, min_z, max_z)))
            for x, y, z in refined
        ]
        if not all(
            segment_clear_3d(refined[i - 1], refined[i], polygons, polygon_names, heights)
            for i in range(1, len(refined))
        ):
            refined = []
        else:
            raw_length = path_length_3d(deduped)
            refined_length = path_length_3d(refined)
            if 0.84 * raw_length <= refined_length <= 1.30 * raw_length:
                for p0, p1 in zip(refined[:-1], refined[1:]):
                    if abs(climb_angle_deg_from_delta(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])) > 35.0 + 1e-6:
                        refined = []
                        break
            else:
                refined = []
        if refined:
            return dense_curve_refine_3d(refined, polygons, polygon_names, heights, sample_factor=max(6, sample_factor // 2))
    except Exception:
        refined = []

    try:
        chaikin = chaikin_smooth_3d(deduped, iterations=2)
        if len(chaikin) < 3:
            return deduped
        if not all(
            segment_clear_3d(chaikin[i - 1], chaikin[i], polygons, polygon_names, heights)
            for i in range(1, len(chaikin))
        ):
            return deduped
        raw_length = path_length_3d(deduped)
        chaikin_length = path_length_3d(chaikin)
        if 0.82 * raw_length <= chaikin_length <= 1.35 * raw_length:
            for p0, p1 in zip(chaikin[:-1], chaikin[1:]):
                if abs(climb_angle_deg_from_delta(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])) > 35.0 + 1e-6:
                    return deduped
            return dense_curve_refine_3d(chaikin, polygons, polygon_names, heights, sample_factor=max(6, sample_factor // 2))
    except Exception:
        pass
    return deduped


def display_smooth_leg_3d(
    leg_3d: Sequence[Point3],
    sample_factor: int = 36,
) -> List[Point3]:
    """Smooth one leg for presentation, prioritizing curvature over safety checks."""

    if len(leg_3d) < 4:
        return list(leg_3d)
    deduped = compress_polyline_3d(leg_3d, distance_eps=0.25, angle_eps_deg=2.0)
    if len(deduped) < 4:
        return list(leg_3d)
    xyz = np.asarray(deduped, dtype=float)
    arc = [0.0]
    for i in range(1, len(deduped)):
        arc.append(arc[-1] + math.dist(deduped[i - 1], deduped[i]))
    arc_arr = np.asarray(arc, dtype=float)
    if arc_arr[-1] < 1e-6:
        return list(leg_3d)
    try:
        k = min(3, len(deduped) - 1)
        spline, _ = splprep(
            [xyz[:, 0], xyz[:, 1], xyz[:, 2]],
            u=arc_arr,
            s=0.45 * len(deduped),
            k=k,
        )
        samples = max(len(deduped) * sample_factor, len(deduped) + 24)
        sample_t = np.unique(
            np.concatenate(
                [
                    np.linspace(0.0, float(arc_arr[-1]), samples),
                    arc_arr,
                ]
            )
        )
        xs, ys, zs = splev(sample_t, spline)
        refined = [(float(x), float(y), float(z)) for x, y, z in zip(xs, ys, zs)]
        refined[0] = tuple(deduped[0])
        refined[-1] = tuple(deduped[-1])
        for knot_t, knot_point in zip(arc_arr, deduped):
            insert_idx = int(np.argmin(np.abs(sample_t - knot_t)))
            refined[insert_idx] = tuple(knot_point)
        return refined
    except Exception:
        pass
    try:
        spline_x = CubicSpline(arc_arr, xyz[:, 0], bc_type="natural")
        spline_y = CubicSpline(arc_arr, xyz[:, 1], bc_type="natural")
        spline_z = CubicSpline(arc_arr, xyz[:, 2], bc_type="natural")
        samples = max(len(deduped) * sample_factor, len(deduped) + 24)
        sample_t = np.unique(
            np.concatenate(
                [
                    np.linspace(0.0, float(arc_arr[-1]), samples),
                    arc_arr,
                ]
            )
        )
        refined = [
            (float(spline_x(tt)), float(spline_y(tt)), float(spline_z(tt)))
            for tt in sample_t
        ]
        refined[0] = tuple(deduped[0])
        refined[-1] = tuple(deduped[-1])
        for knot_t, knot_point in zip(arc_arr, deduped):
            insert_idx = int(np.argmin(np.abs(sample_t - knot_t)))
            refined[insert_idx] = tuple(knot_point)
        return refined
    except Exception:
        return list(leg_3d)


def smooth_leg_3d(
    leg_3d: Sequence[Point3],
    polygons: Sequence[np.ndarray],
    polygon_names: Sequence[str],
    heights: Dict[str, float],
) -> List[Point3]:
    if len(leg_3d) < 2:
        return list(leg_3d)
    # Keep the turn-radius primitives, but do not apply whole-leg display smoothing.
    return compress_polyline_3d(list(leg_3d), distance_eps=0.2, angle_eps_deg=1.0)


def evaluate_path_method(
    path_scenario: PathScenario3D,
    local_order: Sequence[int],
    method: str,
) -> PathPlanResult:
    polyline_3d = assemble_path_3d(path_scenario, local_order)
    return PathPlanResult(
        method=method,
        route_local_indices=list(local_order),
        route_world_indices=local_route_to_world_indices(path_scenario, local_order),
        polyline_3d=polyline_3d,
        path_length_m=path_length_3d(polyline_3d),
        smooth_path_length_m=path_length_3d(polyline_3d),
    )


def plot_path_3d(
    scenario: Coarse3DScenario,
    path_scenario: PathScenario3D,
    path_result: PathPlanResult,
    output: Path,
) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    placed_labels: set[str] = set()
    roof_palette = ["#fdba74", "#93c5fd", "#86efac", "#f9a8d4", "#c4b5fd", "#67e8f9"]
    for idx, polygon in enumerate(path_scenario.polygons):
        polygon_name = path_scenario.polygon_names[idx] if idx < len(path_scenario.polygon_names) else f"building_{idx + 1}"
        height = path_scenario.heights.get(polygon_name, 18.0)
        centroid = np.mean(polygon, axis=0)
        bottom = [(float(x), float(y), 0.0) for x, y in polygon]
        top = [(float(x), float(y), float(height)) for x, y in polygon]
        faces = [top]
        for i in range(len(polygon)):
            j = (i + 1) % len(polygon)
            faces.append([bottom[i], bottom[j], top[j], top[i]])
        color = roof_palette[idx % len(roof_palette)]
        collection = Poly3DCollection(
            faces,
            facecolors=color,
            edgecolors="#334155",
            linewidths=0.8,
            alpha=0.16,
        )
        ax.add_collection3d(collection)
        label = english_building_alias(polygon_name)
        if label and label not in placed_labels:
            ax.text(
                float(centroid[0]),
                float(centroid[1]),
                float(height + 2.0),
                label,
                fontsize=8,
                color="#1e293b",
                ha="center",
            )
            placed_labels.add(label)
    targets = scenario.targets_world
    ax.scatter(targets[:, 0], targets[:, 1], targets[:, 2], s=4, c="#94a3b8", alpha=0.18, label="surface targets")
    selected = path_scenario.selected_points
    ax.scatter(
        selected[:, 0],
        selected[:, 1],
        selected[:, 2],
        s=70,
        c="#ef4444",
        edgecolors="white",
        linewidths=0.8,
        alpha=0.95,
        depthshade=False,
        label="selected viewpoints",
    )
    path_xyz = np.asarray(path_result.polyline_3d, dtype=float)
    ax.plot(path_xyz[:, 0], path_xyz[:, 1], path_xyz[:, 2], color="#0f766e", linewidth=3.0, label=path_result.method)
    ax.scatter(
        [path_scenario.start[0]],
        [path_scenario.start[1]],
        [path_scenario.start[2]],
        s=140,
        c="#111827",
        marker="*",
        edgecolors="white",
        linewidths=0.8,
        depthshade=False,
        label="base point",
    )
    ax.set_xlim(path_scenario.bounds[0], path_scenario.bounds[1])
    ax.set_ylim(path_scenario.bounds[2], path_scenario.bounds[3])
    max_height = max(path_scenario.heights.values(), default=25.0)
    max_path_z = float(np.max(path_xyz[:, 2])) if len(path_xyz) else 25.0
    ax.set_zlim(0.0, max(max_height + 12.0, max_path_z + 6.0))
    try:
        ax.set_box_aspect(
            (
                path_scenario.bounds[1] - path_scenario.bounds[0],
                path_scenario.bounds[3] - path_scenario.bounds[2],
                max(max_height + 12.0, max_path_z + 6.0),
            )
        )
    except Exception:
        pass
    ax.view_init(elev=28, azim=-58)
    ax.set_title(path_result.method)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend(loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_waypoints_csv(path_result: PathPlanResult, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sequence", "x", "y", "z"])
        for index, point in enumerate(path_result.polyline_3d, start=1):
            writer.writerow([index, point[0], point[1], point[2]])


def write_pipeline_outputs(
    selection_results: Sequence[SelectionResult],
    seed_runs: Sequence[SeedRunResult],
    path_results: Sequence[PathPlanResult],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_by_method = {result.method: result for result in selection_results}
    rows = []
    for path_result in path_results:
        selection_key = "Maskable PPO best-of-N"
        if path_result.method.startswith("Maskable PPO best-of-N"):
            selection = selection_by_method[selection_key]
        else:
            selection = selection_by_method.get(selection_key, next(iter(selection_by_method.values())))
        rows.append(
            {
                "method": path_result.method,
                "selected_views": selection.selected_views,
                "certified_coverage": selection.certified_coverage,
                "forward_overlap_ratio": selection.mean_photo_overlap,
                "lateral_overlap_ratio": selection.route_photo_overlap,
                "weakest_structure_part_coverage": selection.weakest_building_coverage,
                "average_quality": selection.average_quality,
                "average_photo_score": selection.average_photo_score,
                "incidence_quality": selection.mean_incidence_quality,
                "distance_quality": selection.mean_distance_quality,
                "visibility_quality": selection.mean_visibility_quality,
                "total_photos": selection.photo_count,
                "path_length_m": path_result.path_length_m,
                "smooth_path_length_m": path_result.smooth_path_length_m,
            }
        )
    with (output_dir / "coarse_3d_full_pipeline_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "coarse_3d_full_pipeline_results.md").open("w", encoding="utf-8") as stream:
        stream.write("# 安中大楼粗三维全链路结果\n\n")
        stream.write("主线：候选视点生成（法向+分区/聚类） -> 视点选取（Maskable PPO best-of-N，含 shots/standoff） -> 路径规划（mainline 顺序优化 -> 安全直连优先，否则 3D A* -> 转弯半径平滑）。\n\n")
        stream.write("| 方法 | 视点数 | 认证覆盖率 | 前向重叠 | 旁向重叠 | 最弱结构段覆盖率 | 平均质量 | 拍照质量 | 入射质量 | 距离质量 | 可见性质量 | 总张数 | 路径长度 | 平滑后路径 |\n")
        stream.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            stream.write(
                f"| {row['method']} | {row['selected_views']} | {row['certified_coverage']:.1%} | "
                f"{row['forward_overlap_ratio']:.1%} | {row['lateral_overlap_ratio']:.1%} | "
                f"{row['weakest_structure_part_coverage']:.1%} | {row['average_quality']:.3f} | "
                f"{row['average_photo_score']:.3f} | {row['incidence_quality']:.3f} | "
                f"{row['distance_quality']:.3f} | {row['visibility_quality']:.3f} | {row['total_photos']} | "
                f"{row['path_length_m']:.1f} | {row['smooth_path_length_m']:.1f} |\n"
            )
        if seed_runs:
            stream.write("\n## 视点选取 PPO Multi-Seed\n\n")
            stream.write("| Seed | 认证覆盖率 | 前向重叠 | 旁向重叠 | 最弱结构段覆盖率 | 平均质量 | 拍照质量 | 入射质量 | 距离质量 | 可见性质量 | 总张数 | 视点数 |\n")
            stream.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for item in seed_runs:
                stream.write(
                    f"| {item.seed} | {item.result.certified_coverage:.1%} | "
                    f"{item.result.mean_photo_overlap:.1%} | "
                    f"{item.result.route_photo_overlap:.1%} | "
                    f"{item.result.weakest_building_coverage:.1%} | {item.result.average_quality:.3f} | "
                    f"{item.result.average_photo_score:.3f} | {item.result.mean_incidence_quality:.3f} | "
                    f"{item.result.mean_distance_quality:.3f} | {item.result.mean_visibility_quality:.3f} | {item.result.photo_count} | "
                    f"{item.result.selected_views} |\n"
                )


def write_selected_capture_csv(path_scenario: PathScenario3D, local_order: Sequence[int], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sequence", "world_index", "owner", "sector", "cluster", "x", "y", "z", "aim_x", "aim_y", "aim_z", "shots", "standoff_m"])
        for sequence, local_index in enumerate(local_order, start=1):
            point = path_scenario.selected_points[int(local_index)]
            aim = path_scenario.aims_world[int(local_index)]
            writer.writerow(
                [
                    sequence,
                    int(path_scenario.selected_indices[int(local_index)]),
                    path_scenario.owners[int(local_index)],
                    int(path_scenario.sector_ids[int(local_index)]),
                    int(path_scenario.cluster_ids[int(local_index)]),
                    float(point[0]),
                    float(point[1]),
                    float(point[2]),
                    float(aim[0]),
                    float(aim[1]),
                    float(aim[2]),
                    int(path_scenario.shot_counts[int(local_index)]),
                    float(path_scenario.standoff_distances[int(local_index)]),
                ]
            )


def write_station_shot_summary_csv(path_scenario: PathScenario3D, local_order: Sequence[int], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sequence", "world_index", "owner", "shots", "standoff_m", "x", "y", "z"])
        for sequence, local_index in enumerate(local_order, start=1):
            point = path_scenario.selected_points[int(local_index)]
            writer.writerow(
                [
                    sequence,
                    int(path_scenario.selected_indices[int(local_index)]),
                    path_scenario.owners[int(local_index)],
                    int(path_scenario.shot_counts[int(local_index)]),
                    float(path_scenario.standoff_distances[int(local_index)]),
                    float(point[0]),
                    float(point[1]),
                    float(point[2]),
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrated coarse-3D selection + path-planning pipeline.")
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--scene-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "coarse_3d_full_pipeline")
    parser.add_argument("--max-targets", type=int, default=900)
    parser.add_argument("--required-views", type=int, default=1)
    parser.add_argument("--max-generated-candidates", type=int, default=3600)
    parser.add_argument("--ppo-candidates", type=int, default=384)
    parser.add_argument("--selection-timesteps", type=int, default=8000)
    parser.add_argument("--selection-seed-sweep", type=str, default="7,11,17,23,29")
    parser.add_argument("--return-to-base", dest="return_to_base", action="store_true", default=True)
    parser.add_argument("--no-return-to-base", dest="return_to_base", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_scenario = build_scenario(
        args.mesh,
        args.scene_json,
        max_targets=args.max_targets,
        required_views=args.required_views,
        max_candidates=args.max_generated_candidates,
        fov=75.0,
        max_incidence=72.0,
        quality_threshold=0.18,
        coverage_goal=1.0,
        forward_overlap_goal=0.80,
        lateral_overlap_goal=0.60,
        enable_transition_rescue=False,
    )
    scenario, keep_indices = compress_candidates(base_scenario, args.ppo_candidates, return_indices=True)
    greedy = evaluate_selection(scenario, greedy_selection(scenario), "Region-aware greedy")
    ilp = evaluate_selection(scenario, scp_ilp_selection(scenario), "SCP/ILP")
    seed_values = [int(part) for part in args.selection_seed_sweep.split(",") if part.strip()]
    selection_model, ppo_best, seed_runs, _training_history = train_and_evaluate_multi_seed(
        scenario,
        total_timesteps=args.selection_timesteps,
        seeds=seed_values,
    )
    raw_policy = run_maskable_ppo(scenario, selection_model)
    mapped_policy = SelectionPolicyOutput(
        selected_indices=[int(keep_indices[int(idx)]) for idx in raw_policy.selected_indices],
        shot_counts=list(raw_policy.shot_counts),
        selected_layer_ids=list(raw_policy.selected_layer_ids),
        selected_standoff_factors=list(raw_policy.selected_standoff_factors),
    )
    augmented_scenario = augment_scenario_with_post_rescue_candidates(
        base_scenario,
        mapped_policy,
    )
    ppo_selection = evaluate_selection(
        augmented_scenario,
        mapped_policy,
        ppo_best.method,
    )
    path_scenario = build_path_scenario(
        augmented_scenario,
        ppo_selection,
        args.scene_json,
        return_to_start=args.return_to_base,
    )
    mainline_order = order_mainline(path_scenario)
    mainline_path = evaluate_path_method(path_scenario, mainline_order, "Maskable PPO best-of-N path mainline")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_model.save(output_dir / "selection_maskable_ppo")
    write_waypoints_csv(mainline_path, output_dir / "mainline_path_waypoints.csv")
    write_selected_capture_csv(path_scenario, mainline_order, output_dir / "mainline_selected_captures.csv")
    write_station_shot_summary_csv(path_scenario, mainline_order, output_dir / "mainline_station_shots.csv")
    plot_path_3d(scenario, path_scenario, mainline_path, output_dir / "mainline_path_3d.png")
    write_pipeline_outputs(
        [greedy, ilp, ppo_selection],
        seed_runs,
        [mainline_path],
        output_dir,
    )

    print(
        f"Selection best-of-N: coverage={ppo_selection.certified_coverage:.1%}, "
        f"forward={ppo_selection.mean_photo_overlap:.1%}, "
        f"lateral={ppo_selection.route_photo_overlap:.1%}, "
        f"photo={ppo_selection.average_photo_score:.3f}, "
        f"shots={ppo_selection.photo_count}, selected={ppo_selection.selected_views}"
    )
    print(
        f"Path mainline: raw={mainline_path.path_length_m:.1f}, smooth={mainline_path.smooth_path_length_m:.1f}"
    )


if __name__ == "__main__":
    main()
