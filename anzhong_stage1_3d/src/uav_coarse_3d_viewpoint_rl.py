#!/usr/bin/env python3
"""PPO viewpoint-selection prototype on the coarse 3D planning scene.

This stage keeps the current research decomposition explicit:

1. the coarse-model stage provides a preflight 3D proxy mesh;
2. our candidate generator uses surface normals plus partition/cluster logic;
3. viewpoint selection compares region-aware greedy, SCP/ILP, and PPO
   on the same compressed candidate pool;
4. path organization still reuses the current geometric route optimizer.

The PPO action space uses a compact continuous interface: the policy chooses a
candidate rank, a shot-count preference, and a continuous standoff distance.
This keeps the learned component limited to viewpoint selection while the
coarse geometry and route evaluation remain explicit and interpretable.
"""

from __future__ import annotations

import argparse
import copy
import csv
import collections
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

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from scipy.optimize import Bounds, LinearConstraint, milp
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from uav_coarse_3d_planner import (
    assign_targets_to_buildings,
    classify_surface_regions,
    cluster_surface_patches,
    coverage_matrix,
    coverage_quality_matrices,
    filter_candidate_constraints,
    generate_candidates,
    generate_partitioned_candidates,
    geometry_frame,
    normalize,
    point_in_polygon_xy,
    target_context_features,
    read_triangle_mesh,
    render_plan,
    rotate_to_local,
    route_metrics,
    route_multistart_optimize,
    prune_candidate_block,
    sample_surface,
    select_multicover,
    targeted_candidates,
    to_local,
)


@dataclass(frozen=True)
class Coarse3DScenario:
    targets_local: np.ndarray
    normals_local: np.ndarray
    targets_world: np.ndarray
    target_buildings: List[str]
    candidates_local: np.ndarray
    candidates_world: np.ndarray
    aims_world: np.ndarray
    coverage: np.ndarray
    quality: np.ndarray
    candidate_owners: List[str]
    candidate_region_codes: np.ndarray
    candidate_cluster_ids: np.ndarray
    candidate_region_ids: np.ndarray
    candidate_sector_ids: np.ndarray
    target_region_ids: np.ndarray
    target_sector_ids: np.ndarray
    candidate_importance: np.ndarray
    stand_off: float
    required_views: int
    min_standoff: float
    max_standoff: float
    standoff_factor_min: float
    standoff_factor_max: float
    scene_center_world: np.ndarray
    scene_basis: np.ndarray
    scene_low_world: np.ndarray
    scene_high_world: np.ndarray


@dataclass(frozen=True)
class PPOTrainingConfig:
    total_timesteps: int = 8_000
    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 8
    gamma: float = 0.98
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    clip_range: float = 0.2
    eval_every_steps: int = 1_000
    seed: int = 17
    quality_good_fraction_target: float = 0.70
    quality_mean_target: float = 0.55
    weakest_quality_target: float = 0.12


@dataclass(frozen=True)
class SelectionResult:
    method: str
    selected_indices: List[int]
    route_indices: List[int]
    route_length: float
    certified_coverage: float
    weakest_building_coverage: float
    weakest_region_coverage: float
    mean_region_gap: float
    weakest_sector_coverage: float
    selected_views: int
    average_quality: float
    objective: float
    mean_quality_normalized: float = 0.0
    quality_good_fraction: float = 0.0
    weakest_quality_normalized: float = 0.0
    mean_incidence_quality: float = 0.0
    mean_distance_quality: float = 0.0
    mean_visibility_quality: float = 0.0
    mean_photo_overlap: float = 0.0
    route_photo_overlap: float = 0.0
    shot_counts: Tuple[int, ...] = ()
    route_shot_counts: Tuple[int, ...] = ()
    selected_layer_ids: Tuple[int, ...] = ()
    route_layer_ids: Tuple[int, ...] = ()
    selected_standoff_factors: Tuple[float, ...] = ()
    route_standoff_factors: Tuple[float, ...] = ()
    photo_count: int = 0

    @property
    def forward_overlap_ratio(self) -> float:
        return float(self.mean_photo_overlap)

    @property
    def lateral_overlap_ratio(self) -> float:
        return float(self.route_photo_overlap)

    @property
    def total_photos(self) -> int:
        return int(self.photo_count)

    @property
    def average_photo_score(self) -> float:
        overlap_term = 0.58 * float(self.mean_photo_overlap) + 0.26 * float(self.route_photo_overlap)
        quality_term = 0.62 * float(self.mean_quality_normalized) + 0.38 * float(self.quality_good_fraction)
        shot_term = min(float(self.photo_count) / max(float(self.selected_views * MAX_SHOTS_PER_STATION), 1.0), 1.0)
        return float(np.clip(0.52 * quality_term + 0.34 * overlap_term + 0.14 * shot_term, 0.0, 1.0))


@dataclass(frozen=True)
class SelectionPolicyOutput:
    selected_indices: List[int]
    shot_counts: List[int]
    selected_layer_ids: List[int]
    selected_standoff_factors: List[float]


@dataclass(frozen=True)
class SeedRunResult:
    seed: int
    result: SelectionResult


@dataclass(frozen=True)
class TrainingHistoryRow:
    step: int
    mean_coverage: float
    mean_weakest_building: float
    mean_weakest_region: float
    mean_region_gap: float
    mean_weakest_sector: float
    mean_quality: float
    mean_quality_normalized: float
    quality_good_fraction: float
    weakest_quality_normalized: float
    mean_route_length: float
    mean_forward_overlap: float
    mean_lateral_overlap: float
    score: float


WEAKEST_BUILDING_FLOOR = 0.80
WEAKEST_BUILDING_POST_FLOOR_SLOPE = 0.25
FORWARD_OVERLAP_TARGET = 0.80
LATERAL_OVERLAP_TARGET = 0.60
COVERAGE_TARGET = 1.00
MAX_SHOTS_PER_STATION = 9
LEGACY_STANDOFF_LAYER_FACTORS = tuple(np.linspace(0.75, 1.20, 61, dtype=np.float64).tolist())
QUALITY_GOOD_RATIO = 0.25


def floor_sensitive_coverage(value: float) -> float:
    """Piecewise score that keeps improving strongly below the floor and flattens above it."""

    if value <= WEAKEST_BUILDING_FLOOR:
        return float(value)
    return float(
        WEAKEST_BUILDING_FLOOR
        + WEAKEST_BUILDING_POST_FLOOR_SLOPE * (value - WEAKEST_BUILDING_FLOOR)
    )


def floor_sensitive_gain(after: float, before: float) -> float:
    return floor_sensitive_coverage(after) - floor_sensitive_coverage(before)


def selection_score(result: SelectionResult) -> float:
    forward_term = min(result.mean_photo_overlap / max(FORWARD_OVERLAP_TARGET, 1e-6), 1.0)
    lateral_term = min(result.route_photo_overlap / max(LATERAL_OVERLAP_TARGET, 1e-6), 1.0)
    overlap_balance = 1.0 - abs(result.mean_photo_overlap - FORWARD_OVERLAP_TARGET) - abs(result.route_photo_overlap - LATERAL_OVERLAP_TARGET)
    return (
        3.2 * result.certified_coverage
        + 2.1 * forward_term
        + 1.5 * lateral_term
        + 0.45 * result.weakest_building_coverage
        + 0.25 * result.weakest_region_coverage
        + 0.20 * result.weakest_sector_coverage
        + 0.30 * result.mean_quality_normalized
        + 0.15 * result.quality_good_fraction
        + 0.18 * result.weakest_quality_normalized
        + 0.10 * overlap_balance
        - 0.0005 * result.route_length
    )


def candidate_score_vector(scenario: Coarse3DScenario) -> np.ndarray:
    num_targets = max(len(scenario.targets_world), 1)
    raw_gain = scenario.coverage.sum(axis=1).astype(np.float32) / num_targets
    quality_gain = scenario.quality.mean(axis=1)
    region_bonus = np.where(scenario.candidate_region_codes >= 2.0, 1.12, 1.0)
    center_distance = np.linalg.norm(
        scenario.candidates_world[:, :2] - scenario.scene_center_world[None, :2],
        axis=1,
    )
    center_distance = center_distance / max(float(center_distance.max()), 1e-6)
    owner_bonus = np.where(np.asarray(scenario.candidate_owners, dtype=object) == "unassigned", -7.5, 0.35)
    return (
        (0.58 * quality_gain + 0.30 * raw_gain) * region_bonus
        + 0.14 * center_distance
        + 0.03 * scenario.candidate_importance
        + owner_bonus
    )


def _jaccard_overlap(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = np.count_nonzero(a | b)
    if union == 0:
        return 0.0
    return float(np.count_nonzero(a & b) / union)


def _camera_yaw_pitch(camera: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    direction = np.asarray(target, dtype=np.float64) - np.asarray(camera, dtype=np.float64)
    horizontal = math.hypot(float(direction[0]), float(direction[1]))
    yaw = math.degrees(math.atan2(float(direction[1]), float(direction[0])))
    pitch = math.degrees(math.atan2(float(direction[2]), horizontal))
    return yaw, pitch


def _rotate_target_by_offsets(
    camera: np.ndarray,
    target: np.ndarray,
    yaw_offset_deg: float,
    pitch_offset_deg: float,
) -> np.ndarray:
    camera = np.asarray(camera, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    direction = target - camera
    distance = max(float(np.linalg.norm(direction)), 1e-6)
    yaw, pitch = _camera_yaw_pitch(camera, target)
    yaw = math.radians(yaw + yaw_offset_deg)
    pitch = math.radians(np.clip(pitch + pitch_offset_deg, -89.0, 89.0))
    horizontal = distance * math.cos(pitch)
    return np.array(
        [
            camera[0] + horizontal * math.cos(yaw),
            camera[1] + horizontal * math.sin(yaw),
            camera[2] + distance * math.sin(pitch),
        ],
        dtype=np.float64,
    )


def station_shot_counts(scenario: Coarse3DScenario, route_indices: Sequence[int]) -> List[int]:
    if not route_indices:
        return []
    scores = np.asarray(scenario.candidate_importance, dtype=np.float64)
    route_scores = np.array([float(scores[int(index)]) for index in route_indices], dtype=np.float64)
    if np.allclose(route_scores.max(), route_scores.min()):
        normalized = np.ones_like(route_scores) * 0.5
    else:
        normalized = (route_scores - route_scores.min()) / max(float(route_scores.max() - route_scores.min()), 1e-9)
    counts = [int(np.clip(round(1.0 + 6.0 * float(score)), 1, 7)) for score in normalized]
    counts[0] = max(counts[0], 3)
    counts[-1] = max(counts[-1], 3)
    return counts


def build_photo_schedule(
    scenario: Coarse3DScenario,
    route_indices: Sequence[int],
    station_shot_counts_override: Sequence[int] | None = None,
    station_standoff_factors_override: Sequence[float] | None = None,
    front_overlap: float = FORWARD_OVERLAP_TARGET,
    side_overlap: float = LATERAL_OVERLAP_TARGET,
) -> tuple[np.ndarray, np.ndarray]:
    route_indices = [int(index) for index in route_indices]
    if not route_indices:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    counts = list(station_shot_counts_override) if station_shot_counts_override is not None else station_shot_counts(scenario, route_indices)
    counts = [max(1, int(count)) for count in counts]
    if station_standoff_factors_override is None:
        factors = [1.0 for _ in route_indices]
    else:
        factors = [float(value) for value in station_standoff_factors_override]
    yaw_span = max(8.0, (1.0 - front_overlap) * 80.0)
    pitch_span = max(2.0, (1.0 - side_overlap) * 10.0)
    cameras_world: List[np.ndarray] = []
    aims_world: List[np.ndarray] = []
    for station_rank, index in enumerate(route_indices):
        factor = factors[min(station_rank, len(factors) - 1)] if factors else 1.0
        camera = standoff_position_world(scenario, int(index), factor)
        target = np.asarray(scenario.aims_world[int(index)], dtype=np.float64)
        count = counts[min(station_rank, len(counts) - 1)]
        if count > 1:
            yaw_offsets = np.linspace(-yaw_span, yaw_span, num=count, dtype=np.float64)
            pitch_offsets = np.sin(np.linspace(-0.5 * np.pi, 0.5 * np.pi, num=count, dtype=np.float64)) * pitch_span
        else:
            yaw_offsets = np.array([0.0], dtype=np.float64)
            pitch_offsets = np.array([0.0], dtype=np.float64)
        for yaw_offset, pitch_offset in zip(yaw_offsets.tolist(), pitch_offsets.tolist()):
            cameras_world.append(camera)
            aims_world.append(_rotate_target_by_offsets(camera, target, yaw_offset, pitch_offset))
    return np.asarray(cameras_world, dtype=np.float64), np.asarray(aims_world, dtype=np.float64)


def photo_schedule_overlap_metrics(
    scenario: Coarse3DScenario,
    route_indices: Sequence[int],
    station_shot_counts_override: Sequence[int] | None = None,
    station_standoff_factors_override: Sequence[float] | None = None,
    front_overlap: float = FORWARD_OVERLAP_TARGET,
    side_overlap: float = LATERAL_OVERLAP_TARGET,
) -> tuple[float, float]:
    cameras_world, aims_world = build_photo_schedule(
        scenario,
        route_indices,
        station_shot_counts_override=station_shot_counts_override,
        station_standoff_factors_override=station_standoff_factors_override,
        front_overlap=front_overlap,
        side_overlap=side_overlap,
    )
    if len(cameras_world) == 0:
        return 0.0, 0.0
    view_dirs = aims_world - cameras_world
    norms = np.linalg.norm(view_dirs, axis=1, keepdims=True)
    view_dirs = view_dirs / np.maximum(norms, 1e-9)
    counts = list(station_shot_counts_override) if station_shot_counts_override is not None else station_shot_counts(scenario, route_indices)
    counts = [max(1, int(count)) for count in counts]
    station_groups: Dict[int, List[int]] = {}
    cursor = 0
    for station_idx, count in enumerate(counts):
        station_groups[station_idx] = list(range(cursor, min(cursor + count, len(view_dirs))))
        cursor += count
    forward_scores: List[float] = []
    for station_idx, indices in station_groups.items():
        if len(indices) < 2:
            continue
        for first, second in zip(indices[:-1], indices[1:]):
            cosine = float(np.clip(np.dot(view_dirs[first], view_dirs[second]), -1.0, 1.0))
            angle_deg = math.degrees(math.acos(cosine))
            forward_scores.append(max(0.0, 1.0 - angle_deg / 45.0))
    lateral_scores: List[float] = []
    station_ids = sorted(station_groups)
    for first_station, second_station in zip(station_ids[:-1], station_ids[1:]):
        first_indices = station_groups[first_station]
        second_indices = station_groups[second_station]
        if not first_indices or not second_indices:
            continue
        first_mean = np.mean(view_dirs[first_indices], axis=0)
        second_mean = np.mean(view_dirs[second_indices], axis=0)
        first_mean = first_mean / max(float(np.linalg.norm(first_mean)), 1e-9)
        second_mean = second_mean / max(float(np.linalg.norm(second_mean)), 1e-9)
        cosine = float(np.clip(np.dot(first_mean, second_mean), -1.0, 1.0))
        angle_deg = math.degrees(math.acos(cosine))
        lateral_scores.append(max(0.0, 1.0 - angle_deg / 145.0))
    return (
        float(np.mean(forward_scores)) if forward_scores else 0.0,
        float(np.mean(lateral_scores)) if lateral_scores else 0.0,
    )


def local_station_forward_overlap(shot_count: int, front_overlap: float = FORWARD_OVERLAP_TARGET) -> float:
    count = max(1, int(shot_count))
    if count < 2:
        return 0.0
    yaw_span = max(8.0, (1.0 - front_overlap) * 80.0)
    yaw_step = (2.0 * yaw_span) / max(count - 1, 1)
    return float(np.clip(1.0 - yaw_step / 45.0, 0.0, 1.0))


def station_view_direction(
    scenario: Coarse3DScenario,
    candidate_idx: int,
    standoff_factor: float,
) -> np.ndarray:
    camera = standoff_position_world(scenario, int(candidate_idx), float(standoff_factor))
    aim = np.asarray(scenario.aims_world[int(candidate_idx)], dtype=np.float64)
    direction = aim - camera
    norm = max(float(np.linalg.norm(direction)), 1e-9)
    return direction / norm


def local_pair_lateral_overlap(
    scenario: Coarse3DScenario,
    first_idx: int,
    first_standoff_factor: float,
    second_idx: int,
    second_standoff_factor: float,
) -> float:
    first_dir = station_view_direction(scenario, first_idx, first_standoff_factor)
    second_dir = station_view_direction(scenario, second_idx, second_standoff_factor)
    cosine = float(np.clip(np.dot(first_dir, second_dir), -1.0, 1.0))
    angle_deg = math.degrees(math.acos(cosine))
    return float(np.clip(1.0 - angle_deg / 145.0, 0.0, 1.0))


def photo_overlap_metrics(
    coverage: np.ndarray,
    selected_indices: Sequence[int],
    route_indices: Sequence[int] | None = None,
) -> tuple[float, float]:
    selected_indices = [int(index) for index in selected_indices]
    if len(selected_indices) < 2:
        return 0.0, 0.0
    selected_cov = np.asarray(coverage[selected_indices], dtype=bool)
    pairwise: List[float] = []
    for i in range(len(selected_indices)):
        for j in range(i + 1, len(selected_indices)):
            pairwise.append(_jaccard_overlap(selected_cov[i], selected_cov[j]))
    mean_pairwise = float(np.mean(pairwise)) if pairwise else 0.0
    if route_indices is None:
        return mean_pairwise, 0.0
    route_indices = [int(index) for index in route_indices if 0 <= int(index) < len(selected_indices)]
    if len(route_indices) < 2:
        return mean_pairwise, 0.0
    route_cov = selected_cov[route_indices]
    adjacent = [_jaccard_overlap(route_cov[i - 1], route_cov[i]) for i in range(1, len(route_cov))]
    return mean_pairwise, float(np.mean(adjacent)) if adjacent else 0.0


def standoff_position_world(scenario: Coarse3DScenario, candidate_idx: int, standoff_factor: float) -> np.ndarray:
    factor = float(np.clip(standoff_factor, float(scenario.standoff_factor_min), float(scenario.standoff_factor_max)))
    base = np.asarray(scenario.candidates_world[int(candidate_idx)], dtype=np.float64)
    aim = np.asarray(scenario.aims_world[int(candidate_idx)], dtype=np.float64)
    delta = base - aim
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-9:
        return base.copy()
    return aim + delta * factor


def legacy_standoff_layer_index(scenario: Coarse3DScenario, standoff_factor: float) -> int:
    factor = float(np.clip(standoff_factor, float(scenario.standoff_factor_min), float(scenario.standoff_factor_max)))
    legacy_factors = np.asarray(LEGACY_STANDOFF_LAYER_FACTORS, dtype=np.float64)
    return int(np.argmin(np.abs(legacy_factors - factor)))


def variant_position_world(scenario: Coarse3DScenario, candidate_idx: int, layer_idx: int) -> np.ndarray:
    layer_idx = int(np.clip(layer_idx, 0, len(LEGACY_STANDOFF_LAYER_FACTORS) - 1))
    return standoff_position_world(scenario, candidate_idx, float(LEGACY_STANDOFF_LAYER_FACTORS[layer_idx]))


def selected_variant_points_world(
    scenario: Coarse3DScenario,
    selected_indices: Sequence[int],
    selected_layer_ids: Sequence[int] | None = None,
    selected_standoff_factors: Sequence[float] | None = None,
) -> np.ndarray:
    selected_indices = [int(index) for index in selected_indices]
    if not selected_indices:
        return np.zeros((0, 3), dtype=np.float64)
    if selected_standoff_factors is not None:
        factors = [float(value) for value in selected_standoff_factors]
    elif selected_layer_ids is not None:
        factors = [float(LEGACY_STANDOFF_LAYER_FACTORS[int(np.clip(layer, 0, len(LEGACY_STANDOFF_LAYER_FACTORS) - 1))]) for layer in selected_layer_ids]
    else:
        factors = [1.0 for _ in selected_indices]
    return np.asarray(
        [standoff_position_world(scenario, idx, factor) for idx, factor in zip(selected_indices, factors)],
        dtype=np.float64,
    )


def candidate_variant_metrics(
    scenario: Coarse3DScenario,
    candidate_idx: int,
    standoff_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    coverage, quality, _, _, _ = candidate_variant_metric_components(scenario, candidate_idx, standoff_factor)
    return coverage, quality


def candidate_variant_metric_components(
    scenario: Coarse3DScenario,
    candidate_idx: int,
    standoff_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    position_world = standoff_position_world(scenario, candidate_idx, standoff_factor)
    aim_world = np.asarray(scenario.aims_world[int(candidate_idx)], dtype=np.float64)
    position_local = to_local(position_world[None, :], scenario.scene_center_world, scenario.scene_basis)[0]
    aim_local = to_local(aim_world[None, :], scenario.scene_center_world, scenario.scene_basis)[0]
    coverage, quality, incidence_quality, distance_quality, visibility_quality = coverage_quality_matrices(
        position_local[None, :],
        aim_local[None, :],
        scenario.targets_local,
        scenario.normals_local,
        float(scenario.stand_off),
        75.0,
        72.0,
        0.18,
        float(scenario.min_standoff),
        float(scenario.max_standoff),
        None,
    )
    return coverage[0], quality[0], incidence_quality[0], distance_quality[0], visibility_quality[0]


ANZHONG_BUILDING_ALIASES = {
    "安中A座东西向高段": "Anzhong Building",
    "安中A座南北向低段": "Anzhong Building",
    "安中B座东西向高段": "Anzhong Building",
    "安中B座南北向低段": "Anzhong Building",
    "安中大楼连接体": "Anzhong Building",
}


def canonical_building_name(name: str) -> str:
    if name in ANZHONG_BUILDING_ALIASES:
        return ANZHONG_BUILDING_ALIASES[name]
    return name if name.isascii() else name


def candidate_sector_ids_from_world(points_world: np.ndarray, center_world: np.ndarray, bins: int = 8) -> np.ndarray:
    if len(points_world) == 0:
        return np.zeros((0,), dtype=np.int32)
    deltas = points_world[:, :2] - center_world[None, :2]
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])
    normalized = (angles + np.pi) / (2.0 * np.pi)
    return np.clip((normalized * bins).astype(np.int32), 0, bins - 1)


def candidate_side_ids_from_local(points_local: np.ndarray) -> np.ndarray:
    if len(points_local) == 0:
        return np.zeros((0,), dtype=np.int32)
    x = np.asarray(points_local[:, 0], dtype=np.float64)
    center = float(np.median(x))
    return (x > center).astype(np.int32)


def principal_axis_2d(points_local: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(points_local) == 0:
        return np.array([1.0, 0.0], dtype=np.float64), np.array([0.0, 1.0], dtype=np.float64)
    pts = np.asarray(points_local[:, :2], dtype=np.float64)
    centered = pts - pts.mean(axis=0, keepdims=True)
    # SVD is numerically more stable than covariance-eigendecomposition when
    # the point cloud is nearly collinear or when the building is sampled with
    # very uneven triangle density.
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    normal = np.array([-axis[1], axis[0]], dtype=np.float64)
    return axis, normal


def symmetry_diagnostics(points_local: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    if len(points_local) == 0:
        return {}
    axis, normal = principal_axis_2d(points_local)
    centered = np.asarray(points_local[:, :2], dtype=np.float64) - np.mean(points_local[:, :2], axis=0, keepdims=True)
    signed = centered @ normal
    left = signed <= 0
    right = ~left
    diagnostics: Dict[str, float] = {
        "left_count": float(np.count_nonzero(left)),
        "right_count": float(np.count_nonzero(right)),
        "left_mean_score": float(np.mean(scores[left])) if np.any(left) else 0.0,
        "right_mean_score": float(np.mean(scores[right])) if np.any(right) else 0.0,
        "left_median_score": float(np.median(scores[left])) if np.any(left) else 0.0,
        "right_median_score": float(np.median(scores[right])) if np.any(right) else 0.0,
        "score_gap": float(abs(np.mean(scores[left]) - np.mean(scores[right]))) if np.any(left) and np.any(right) else 0.0,
        "axis_x": float(axis[0]),
        "axis_y": float(axis[1]),
    }
    return diagnostics


def building_symmetry_diagnostics(
    scenario: Coarse3DScenario,
    building_name: str = "Anzhong Building",
) -> Dict[str, float | str]:
    if len(scenario.targets_local) == 0:
        return {}
    target_mask = np.array(
        [canonical_building_name(name) == building_name for name in scenario.target_buildings],
        dtype=bool,
    )
    candidate_mask = np.array(
        [canonical_building_name(name) == building_name for name in scenario.candidate_owners],
        dtype=bool,
    )
    if np.any(candidate_mask):
        points = scenario.candidates_local[candidate_mask]
        scores = candidate_score_vector(scenario)[candidate_mask]
    elif np.any(target_mask):
        points = scenario.targets_local[target_mask]
        scores = np.ones(len(points), dtype=np.float64)
    else:
        points = scenario.candidates_local
        scores = candidate_score_vector(scenario)
    diagnostics = symmetry_diagnostics(points, scores)
    diagnostics["building_name"] = building_name
    diagnostics["target_count"] = float(np.count_nonzero(target_mask))
    diagnostics["candidate_count"] = float(np.count_nonzero(candidate_mask))
    return diagnostics


def _weighted_kmeans(
    points: np.ndarray,
    weights: np.ndarray,
    k: int,
    iterations: int = 12,
) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0, points.shape[1] if points.ndim == 2 else 0), dtype=np.float64)
    k = max(1, min(int(k), len(points)))
    order = np.argsort(-weights)
    centers = [points[int(order[0])].astype(np.float64)]
    for idx in order[1:]:
        if len(centers) >= k:
            break
        candidate = points[int(idx)]
        min_dist = min(float(np.linalg.norm(candidate - center)) for center in centers)
        if min_dist >= 0.35 * float(np.linalg.norm(np.ptp(points, axis=0))):
            centers.append(candidate.astype(np.float64))
    while len(centers) < k:
        centers.append(points[int(order[len(centers) % len(order)])].astype(np.float64))
    centers = np.asarray(centers[:k], dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    for _ in range(max(int(iterations), 1)):
        distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for region_id in range(k):
            members = np.flatnonzero(labels == region_id)
            if len(members) == 0:
                continue
            region_weights = np.maximum(weights[members], 1e-6)
            new_centers[region_id] = np.average(points[members], axis=0, weights=region_weights)
        centers = new_centers
    return centers


def assign_points_to_region_ids(points: np.ndarray, region_centers: np.ndarray) -> np.ndarray:
    if len(points) == 0 or len(region_centers) == 0:
        return np.zeros((len(points),), dtype=np.int32)
    distances = np.linalg.norm(points[:, None, :] - region_centers[None, :, :], axis=2)
    return np.argmin(distances, axis=1).astype(np.int32)


def candidate_is_too_close(
    scenario: Coarse3DScenario,
    candidate_idx: int,
    chosen_indices: Sequence[int],
    min_xy_spacing: float,
    min_z_spacing: float,
) -> bool:
    if not chosen_indices:
        return False
    point = scenario.candidates_world[candidate_idx]
    cluster_id = int(scenario.candidate_cluster_ids[candidate_idx]) if len(scenario.candidate_cluster_ids) else -1
    owner = scenario.candidate_owners[candidate_idx] if len(scenario.candidate_owners) else "scene"
    for chosen_idx in chosen_indices:
        other = scenario.candidates_world[chosen_idx]
        xy_dist = float(np.linalg.norm(point[:2] - other[:2]))
        z_dist = abs(float(point[2] - other[2]))
        same_cluster = cluster_id == int(scenario.candidate_cluster_ids[chosen_idx]) if len(scenario.candidate_cluster_ids) else False
        same_owner = owner == scenario.candidate_owners[chosen_idx] if len(scenario.candidate_owners) else False
        if xy_dist < min_xy_spacing and (same_cluster or same_owner or z_dist < min_z_spacing):
            return True
    return False


def diversify_selected_indices(
    scenario: Coarse3DScenario,
    selected_indices: Sequence[int],
) -> List[int]:
    if len(selected_indices) <= 1:
        return list(selected_indices)
    target_count = len(selected_indices)
    min_xy_spacing = max(4.0, 0.45 * float(scenario.stand_off))
    min_z_spacing = max(3.0, 0.20 * float(scenario.stand_off))
    scores = candidate_score_vector(scenario)
    ranked_selected = sorted(
        [int(idx) for idx in selected_indices],
        key=lambda idx: (
            float(scores[idx]),
            float(np.sum(scenario.coverage[idx])),
            float(np.sum(scenario.quality[idx])),
        ),
        reverse=True,
    )
    diversified: List[int] = []
    for idx in ranked_selected:
        if not candidate_is_too_close(scenario, idx, diversified, min_xy_spacing, min_z_spacing):
            diversified.append(idx)
    remaining_need = np.full(len(scenario.targets_world), scenario.required_views, dtype=np.int16)
    for idx in diversified:
        remaining_need = np.maximum(0, remaining_need - scenario.coverage[idx].astype(np.int16))
    available = [idx for idx in range(len(scenario.candidates_world)) if idx not in diversified]
    while len(diversified) < target_count and available:
        best_idx = None
        best_score = -1e18
        need = remaining_need > 0
        used_regions = set(int(scenario.candidate_region_ids[idx]) for idx in diversified)
        for idx in available:
            raw_gain = float(np.count_nonzero(scenario.coverage[idx] & need))
            quality_gain = float(scenario.quality[idx, need].sum()) if np.any(need) else 0.0
            if diversified:
                xy_distances = [
                    float(np.linalg.norm(scenario.candidates_world[idx][:2] - scenario.candidates_world[other][:2]))
                    for other in diversified
                ]
                min_xy = min(xy_distances)
            else:
                min_xy = min_xy_spacing
            diversity_bonus = min(min_xy / max(min_xy_spacing, 1e-6), 2.0)
            region_bonus = 1.5 if int(scenario.candidate_region_ids[idx]) not in used_regions else 0.0
            close_penalty = 3.5 if candidate_is_too_close(scenario, idx, diversified, min_xy_spacing, min_z_spacing) else 0.0
            value = 5.0 * raw_gain + 1.2 * quality_gain + 1.0 * diversity_bonus + region_bonus + 0.2 * float(scores[idx]) - close_penalty
            if value > best_score:
                best_score = value
                best_idx = idx
        if best_idx is None:
            break
        diversified.append(int(best_idx))
        remaining_need = np.maximum(0, remaining_need - scenario.coverage[int(best_idx)].astype(np.int16))
        available.remove(int(best_idx))
    present_regions = {int(scenario.candidate_region_ids[idx]) for idx in diversified}
    missing_regions = [int(region_id) for region_id in sorted(set(map(int, scenario.candidate_region_ids.tolist()))) if int(region_id) not in present_regions]
    if missing_regions and diversified:
        region_scores = candidate_score_vector(scenario)
        region_counts = collections.Counter(int(scenario.candidate_region_ids[idx]) for idx in diversified)
        for region_id in missing_regions:
            candidates = [
                idx
                for idx in range(len(scenario.candidates_world))
                if int(scenario.candidate_region_ids[idx]) == region_id and idx not in diversified
            ]
            if not candidates:
                continue
            best_new = max(candidates, key=lambda idx: float(region_scores[idx]))
            if len(diversified) < target_count:
                diversified.append(int(best_new))
                continue
            removable = min(
                diversified,
                key=lambda idx: (
                    region_counts[int(scenario.candidate_region_ids[idx])],
                    float(region_scores[idx]),
                ),
            )
            diversified.remove(removable)
            region_counts[int(scenario.candidate_region_ids[removable])] -= 1
            diversified.append(int(best_new))
            region_counts[region_id] += 1
    return diversified


def sector_coverage_from_remaining(
    remaining: np.ndarray,
    sector_ids: np.ndarray,
    target_sector_ids: np.ndarray,
) -> Dict[int, float]:
    sector_coverages: Dict[int, float] = {}
    if len(target_sector_ids) == 0:
        return sector_coverages
    for sector_id in sorted(set(int(x) for x in np.asarray(target_sector_ids, dtype=np.int32).tolist())):
        indices = np.flatnonzero(np.asarray(target_sector_ids, dtype=np.int32) == sector_id)
        if len(indices) == 0:
            sector_coverages[sector_id] = 0.0
        else:
            sector_coverages[sector_id] = float(np.mean((remaining[indices] == 0).astype(np.float32)))
    return sector_coverages


def region_remaining_fractions(
    remaining: np.ndarray,
    region_ids: np.ndarray,
) -> Dict[int, float]:
    fractions: Dict[int, float] = {}
    if len(region_ids) == 0:
        return fractions
    region_ids = np.asarray(region_ids, dtype=np.int32)
    for region_id in sorted(set(int(x) for x in region_ids.tolist())):
        indices = np.flatnonzero(region_ids == region_id)
        if len(indices) == 0:
            continue
        fractions[region_id] = float(np.mean(remaining[indices].astype(np.float32)))
    return fractions


def infer_candidate_owners(
    coverage: np.ndarray,
    target_buildings: Sequence[str],
) -> List[str]:
    owners: List[str] = []
    for candidate_idx in range(coverage.shape[0]):
        covered_buildings = [target_buildings[target_idx] for target_idx in np.flatnonzero(coverage[candidate_idx])]
        if not covered_buildings:
            owners.append("unassigned")
        else:
            values, counts = np.unique(np.asarray(covered_buildings, dtype=object), return_counts=True)
            owners.append(str(values[int(np.argmax(counts))]))
    return owners


def build_scenario(
    mesh: Path,
    scene_json: Path | None,
    max_targets: int,
    required_views: int,
    max_candidates: int,
    fov: float,
    max_incidence: float,
    quality_threshold: float,
    coverage_goal: float | None = None,
    forward_overlap_goal: float | None = None,
    lateral_overlap_goal: float | None = None,
) -> Coarse3DScenario:
    vertices, faces = read_triangle_mesh(mesh)
    center, basis, low, high = geometry_frame(vertices)
    targets_world, normals_world, areas = sample_surface(
        vertices, faces, max_targets, center, basis, low, high
    )
    targets_local = to_local(targets_world, center, basis)
    normals_local = rotate_to_local(normals_world, basis)
    scene_span = np.linalg.norm(high - low)
    regions = classify_surface_regions(targets_local, normals_local, areas)
    patches = cluster_surface_patches(targets_local, normals_local, areas, regions, scene_span)
    raw_target_buildings = assign_targets_to_buildings(targets_world, scene_json)
    target_contexts, target_outward_world = target_context_features(
        targets_world,
        normals_world,
        raw_target_buildings,
        scene_json,
    )
    target_outward_local = rotate_to_local(target_outward_world, basis)
    patch_centers = np.asarray([patch.center for patch in patches], dtype=np.float64)
    patch_weights = np.asarray([patch.importance for patch in patches], dtype=np.float64)
    if len(patch_centers):
        region_count = max(3, min(8, max(1, len(patch_centers) // 4)))
        region_centers = _weighted_kmeans(patch_centers, patch_weights, region_count)
    else:
        region_centers = np.zeros((0, 3), dtype=np.float64)
    stand_off = 0.18 * float(np.linalg.norm(high - low))
    min_standoff = max(3.0, 0.06 * scene_span)
    max_standoff = min(15.0, 0.18 * scene_span)
    if max_standoff <= min_standoff + 1e-6:
        max_standoff = min_standoff + max(1.5, 0.05 * scene_span)
    standoff_factor_min = min_standoff / max(stand_off, 1e-6)
    standoff_factor_max = max_standoff / max(stand_off, 1e-6)
    region_weights = np.array(
        [
            1.20 if region == "corner_transition" else 1.15 if region == "occlusion_sensitive" else 1.05 if region == "roof" else 1.0
            for region in regions
        ],
        dtype=np.float32,
    )

    candidates_local, aims_local, metadata = generate_partitioned_candidates(
        patches,
        stand_off,
        max_candidates=max_candidates,
    )
    extra_candidates: List[np.ndarray] = []
    extra_aims: List[np.ndarray] = []
    extra_meta: List[Dict[str, float | str]] = []
    for factor, azimuths, levels in ((0.92, 28, 4), (1.08, 32, 4)):
        ring_local, ring_aims = generate_candidates(low, high, azimuths, levels, stand_off * factor)
        ring_local, ring_aims = filter_candidate_constraints(
            ring_local,
            ring_aims,
            targets_local,
            center,
            basis,
            min_standoff * 0.92,
            max_standoff * 1.05,
            2.0,
            120.0,
        )
        if len(ring_local):
            extra_candidates.append(ring_local)
            extra_aims.append(ring_aims)
            extra_meta.extend(
                [
                    {
                        "cluster_id": -3.0,
                        "region_code": 1.0,
                        "importance": 0.20 + 0.02 * factor,
                        "region_name": "wall",
                        "source": "legacy_ring",
                    }
                    for _ in range(len(ring_local))
                ]
            )
    if extra_candidates:
        candidates_local = np.concatenate([candidates_local] + extra_candidates, axis=0)
        aims_local = np.concatenate([aims_local] + extra_aims, axis=0)
        metadata.extend(extra_meta)
    if not len(candidates_local):
        candidates_local, aims_local = generate_candidates(low, high, 24, 3, stand_off)
        metadata = [{"cluster_id": -1.0, "region_code": 1.0, "importance": 1.0, "region_name": "wall", "source": "fallback_ring"} for _ in range(len(candidates_local))]
    candidates_local, aims_local, keep_mask = filter_candidate_constraints(
        candidates_local,
        aims_local,
        targets_local,
        center,
        basis,
        min_standoff,
        max_standoff,
        2.0,
        120.0,
        return_mask=True,
    )
    metadata = [item for item, keep in zip(metadata, keep_mask) if keep]
    coverage, quality = coverage_matrix(
        candidates_local,
        aims_local,
        targets_local,
        normals_local,
        stand_off,
        fov,
        max_incidence,
        quality_threshold,
        min_standoff,
        max_standoff,
        region_weights,
    )
    candidate_region_codes = np.array([item["region_code"] for item in metadata], dtype=np.float32)
    _selected_seed_cover, remaining_cover = select_multicover(
        coverage,
        quality,
        candidates_local,
        required_views,
        candidate_region_codes,
    )
    if np.any(remaining_cover > 0):
        missing_indices = np.flatnonzero(remaining_cover > 0)
        supplemental_local, supplemental_aims, supplemental_meta = targeted_candidates(
            targets_local,
            normals_local,
            missing_indices,
            stand_off,
            max_standoff,
            regions,
            target_contexts,
            target_outward_local,
        )
        if len(supplemental_local):
            supplemental_local, supplemental_aims, supplemental_keep_mask = filter_candidate_constraints(
                supplemental_local,
                supplemental_aims,
                targets_local,
                center,
                basis,
                min_standoff,
                max_standoff,
                2.0,
                120.0,
                return_mask=True,
            )
            supplemental_meta = [item for item, keep in zip(supplemental_meta, supplemental_keep_mask) if keep]
            supplemental_coverage, supplemental_quality = coverage_matrix(
                supplemental_local,
                supplemental_aims,
                targets_local,
                normals_local,
                stand_off,
                fov,
                max_incidence,
                quality_threshold,
                min_standoff,
                max_standoff,
                region_weights,
            )
            rescue_budget = min(max(max_candidates // 6, 240), max(720, len(missing_indices) * 3))
            supplemental_local, supplemental_aims, supplemental_meta, supplemental_coverage, supplemental_quality = prune_candidate_block(
                supplemental_local,
                supplemental_aims,
                supplemental_meta,
                supplemental_coverage,
                supplemental_quality,
                rescue_budget,
                missing_indices,
            )
            candidates_local = np.concatenate([candidates_local, supplemental_local], axis=0)
            aims_local = np.concatenate([aims_local, supplemental_aims], axis=0)
            coverage = np.concatenate([coverage, supplemental_coverage], axis=0)
            quality = np.concatenate([quality, supplemental_quality], axis=0)
            metadata.extend(supplemental_meta)
            candidate_region_codes = np.concatenate(
                [candidate_region_codes, np.array([item["region_code"] for item in supplemental_meta], dtype=np.float32)]
            )
    unique_sectors = candidate_sector_ids_from_world(
        np.einsum("ij,kj->ik", candidates_local, basis) + center,
        np.asarray(center, dtype=np.float64),
        bins=8,
    )
    if len(np.unique(unique_sectors)) < 4:
        for factor, azimuths, levels in ((0.78, max(48, max_candidates // 18), 4), (1.15, max(64, max_candidates // 14), 5)):
            supplemental_local, supplemental_aims = generate_candidates(low, high, azimuths, levels, stand_off * factor)
            supplemental_local, supplemental_aims = filter_candidate_constraints(
                supplemental_local,
                supplemental_aims,
                targets_local,
                center,
                basis,
                min_standoff * 0.92,
                max_standoff * 1.05,
                2.0,
                120.0,
            )
            if len(supplemental_local):
                supplemental_coverage, supplemental_quality = coverage_matrix(
                    supplemental_local,
                    supplemental_aims,
                    targets_local,
                    normals_local,
                    stand_off,
                    fov,
                    max_incidence,
                    quality_threshold,
                    min_standoff,
                    max_standoff,
                    region_weights,
                )
                supplemental_meta = [
                    {"cluster_id": -2.0, "region_code": 1.0, "importance": 0.20 + 0.02 * factor, "region_name": "wall", "source": "supplemental_ring"}
                    for _ in range(len(supplemental_local))
                ]
                candidates_local = np.concatenate([candidates_local, supplemental_local], axis=0)
                aims_local = np.concatenate([aims_local, supplemental_aims], axis=0)
                coverage = np.concatenate([coverage, supplemental_coverage], axis=0)
                quality = np.concatenate([quality, supplemental_quality], axis=0)
                metadata.extend(supplemental_meta)
    candidates_world = np.einsum("ij,kj->ik", candidates_local, basis) + center
    aims_world = np.einsum("ij,kj->ik", aims_local, basis) + center
    if scene_json is not None and scene_json.exists() and len(candidates_world):
        data = json.loads(scene_json.read_text(encoding="utf-8"))
        valid_mask = np.ones(len(candidates_world), dtype=bool)
        for idx, point in enumerate(candidates_world):
            for building in data.get("buildings", []):
                footprint = np.asarray(building.get("footprint", []), dtype=np.float64)
                if len(footprint) < 3:
                    continue
                roof = float(building.get("height", 24.0))
                # Reject prism-interior viewpoints: these are the main source of
                # “穿模” in the LoD1 building rendering.
                if point_in_polygon_xy(point[:2], footprint) and float(point[2]) <= roof + 1.0:
                    valid_mask[idx] = False
                    break
        if np.any(~valid_mask):
            candidates_local = candidates_local[valid_mask]
            aims_local = aims_local[valid_mask]
            coverage = coverage[valid_mask]
            quality = quality[valid_mask]
            candidates_world = candidates_world[valid_mask]
            aims_world = aims_world[valid_mask]
            metadata = [item for keep, item in zip(valid_mask.tolist(), metadata) if keep]
    target_buildings = [canonical_building_name(name) for name in raw_target_buildings]
    candidate_owners = [canonical_building_name(name) for name in infer_candidate_owners(coverage, target_buildings)]
    candidate_region_ids = assign_points_to_region_ids(candidates_local, region_centers)
    target_region_ids = assign_points_to_region_ids(targets_local, region_centers)
    candidate_sector_ids = candidate_sector_ids_from_world(candidates_world, np.asarray(center, dtype=np.float64), bins=8)
    target_sector_ids = candidate_sector_ids_from_world(targets_world, np.asarray(center, dtype=np.float64), bins=8)
    return Coarse3DScenario(
        targets_local=targets_local,
        normals_local=normals_local,
        targets_world=targets_world,
        target_buildings=target_buildings,
        candidates_local=candidates_local,
        candidates_world=candidates_world,
        aims_world=aims_world,
        coverage=coverage,
        quality=quality,
        candidate_owners=candidate_owners,
        candidate_region_codes=np.array([item["region_code"] for item in metadata], dtype=np.float32),
        candidate_cluster_ids=np.array([item["cluster_id"] for item in metadata], dtype=np.float32),
        candidate_region_ids=candidate_region_ids,
        candidate_sector_ids=candidate_sector_ids,
        target_region_ids=target_region_ids,
        target_sector_ids=target_sector_ids,
        candidate_importance=np.array([item["importance"] for item in metadata], dtype=np.float32),
        stand_off=stand_off,
        required_views=required_views,
        min_standoff=min_standoff,
        max_standoff=max_standoff,
        standoff_factor_min=standoff_factor_min,
        standoff_factor_max=standoff_factor_max,
        scene_center_world=np.asarray(center, dtype=np.float64),
        scene_basis=np.asarray(basis, dtype=np.float64),
        scene_low_world=np.asarray(low, dtype=np.float64),
        scene_high_world=np.asarray(high, dtype=np.float64),
    )


def compress_candidates(
    scenario: Coarse3DScenario,
    keep_candidates: int,
    seed: int | None = None,
    return_indices: bool = False,
) -> Coarse3DScenario | tuple[Coarse3DScenario, np.ndarray]:
    if keep_candidates <= 0 or len(scenario.candidates_local) <= keep_candidates:
        return (scenario, np.arange(len(scenario.candidates_local), dtype=int)) if return_indices else scenario
    scores = candidate_score_vector(scenario)
    rng = np.random.default_rng(seed)
    anchor_indices = greedy_selection(scenario)
    chosen: List[int] = list(dict.fromkeys(int(idx) for idx in anchor_indices[:keep_candidates]))
    ranked_global = list(np.argsort(-scores))
    if seed is None:
        for idx in ranked_global:
            if int(idx) in chosen:
                continue
            chosen.append(int(idx))
            if len(chosen) >= keep_candidates:
                break
    else:
        remaining = [int(idx) for idx in ranked_global[: min(len(ranked_global), keep_candidates * 4)] if int(idx) not in chosen]
        if remaining:
            weights = np.asarray([max(float(scores[idx]), 1e-6) for idx in remaining], dtype=np.float64)
            weights = weights / weights.sum()
            extra = rng.choice(
                np.asarray(remaining, dtype=int),
                size=min(len(remaining), keep_candidates - len(chosen)),
                replace=False,
                p=weights,
            )
            chosen.extend(int(idx) for idx in extra)
        if len(chosen) < keep_candidates:
            for idx in ranked_global:
                if int(idx) in chosen:
                    continue
                chosen.append(int(idx))
                if len(chosen) >= keep_candidates:
                    break
    keep = np.asarray(
        sorted(
            set(int(idx) for idx in chosen),
            key=lambda idx: (
                -float(scores[idx]),
                -float(scenario.candidate_importance[int(idx)]),
                int(idx),
            ),
        )[:keep_candidates],
        dtype=int,
    )
    compressed = Coarse3DScenario(
        targets_local=scenario.targets_local,
        normals_local=scenario.normals_local,
        targets_world=scenario.targets_world,
        target_buildings=scenario.target_buildings,
        candidates_local=scenario.candidates_local[keep],
        candidates_world=scenario.candidates_world[keep],
        aims_world=scenario.aims_world[keep],
        coverage=scenario.coverage[keep],
        quality=scenario.quality[keep],
        candidate_owners=[scenario.candidate_owners[int(index)] for index in keep],
        candidate_region_codes=scenario.candidate_region_codes[keep],
        candidate_cluster_ids=scenario.candidate_cluster_ids[keep],
        candidate_region_ids=scenario.candidate_region_ids[keep],
        candidate_sector_ids=scenario.candidate_sector_ids[keep],
        target_region_ids=scenario.target_region_ids,
        target_sector_ids=scenario.target_sector_ids,
        candidate_importance=scenario.candidate_importance[keep],
        stand_off=scenario.stand_off,
        min_standoff=scenario.min_standoff,
        max_standoff=scenario.max_standoff,
        standoff_factor_min=scenario.standoff_factor_min,
        standoff_factor_max=scenario.standoff_factor_max,
        required_views=scenario.required_views,
        scene_center_world=scenario.scene_center_world,
        scene_basis=scenario.scene_basis,
        scene_low_world=scenario.scene_low_world,
        scene_high_world=scenario.scene_high_world,
    )
    return (compressed, keep) if return_indices else compressed


class Coarse3DScenarioRepository:
    def __init__(self, base_scenario: Coarse3DScenario, keep_candidates: int, variant_seeds: Sequence[int]) -> None:
        self.base_scenario = base_scenario
        self.keep_candidates = keep_candidates
        self.variant_seeds = tuple(variant_seeds)
        if not self.variant_seeds:
            raise ValueError("variant_seeds must not be empty")
        self._cache: Dict[int, Coarse3DScenario] = {}

    def get(self, seed: int) -> Coarse3DScenario:
        if seed not in self._cache:
            self._cache[seed] = compress_candidates(self.base_scenario, self.keep_candidates, seed=seed)
        return self._cache[seed]


def evaluate_selection(
    scenario: Coarse3DScenario,
    selected_data: Sequence[int] | SelectionPolicyOutput | tuple[Sequence[int], Sequence[int]],
    method: str,
    apply_diversification: bool = True,
) -> SelectionResult:
    selected_shot_counts: List[int] | None = None
    selected_layer_ids: List[int] | None = None
    selected_standoff_factors: List[float] | None = None
    if isinstance(selected_data, SelectionPolicyOutput):
        selected_indices = list(selected_data.selected_indices)
        selected_shot_counts = [int(count) for count in selected_data.shot_counts]
        selected_layer_ids = [int(layer) for layer in selected_data.selected_layer_ids]
        selected_standoff_factors = [float(value) for value in selected_data.selected_standoff_factors]
    elif isinstance(selected_data, tuple) and len(selected_data) == 2:
        selected_indices = [int(index) for index in selected_data[0]]
        selected_shot_counts = [int(count) for count in selected_data[1]]
    else:
        selected_indices = [int(index) for index in selected_data]
    if apply_diversification and selected_shot_counts is None:
        selected_indices = diversify_selected_indices(scenario, selected_indices)
    if selected_standoff_factors is None and selected_layer_ids is not None:
        selected_standoff_factors = [float(LEGACY_STANDOFF_LAYER_FACTORS[int(np.clip(layer, 0, len(LEGACY_STANDOFF_LAYER_FACTORS) - 1))]) for layer in selected_layer_ids]
    if selected_standoff_factors is None:
        selected_standoff_factors = [1.0 for _ in selected_indices]
    selected_points_world = selected_variant_points_world(
        scenario,
        selected_indices,
        selected_layer_ids=selected_layer_ids,
        selected_standoff_factors=selected_standoff_factors,
    )
    route_positions_local = np.array(scenario.candidates_local, copy=True)
    for index, world_point in zip(selected_indices, selected_points_world):
        route_positions_local[int(index)] = to_local(world_point[None, :], scenario.scene_center_world, scenario.scene_basis)[0]
    if selected_indices:
        route_indices, _, _ = route_multistart_optimize(
            list(selected_indices),
            route_positions_local,
            restarts=24,
            proposals=800,
            turn_weight=2.0,
        )
        objective, route_length, _, _ = route_metrics(route_indices, route_positions_local, turn_weight=2.0)
    else:
        route_indices = []
        objective = route_length = 0.0
    if selected_shot_counts is None:
        selected_shot_counts = station_shot_counts(scenario, selected_indices)
    selected_shot_lookup = {int(idx): int(count) for idx, count in zip(selected_indices, selected_shot_counts)}
    route_shot_counts = [selected_shot_lookup.get(int(idx), 1) for idx in route_indices]
    selected_factor_lookup = {int(idx): float(factor) for idx, factor in zip(selected_indices, selected_standoff_factors)}
    route_standoff_factors = [selected_factor_lookup.get(int(idx), 1.0) for idx in route_indices]
    selected_layer_lookup = {int(idx): int(layer) for idx, layer in zip(selected_indices, selected_layer_ids)} if selected_layer_ids is not None else {}
    route_layer_ids = [selected_layer_lookup.get(int(idx), 0) for idx in route_indices]
    remaining = np.full(len(scenario.targets_world), scenario.required_views, dtype=np.int16)
    best_quality = np.zeros(len(scenario.targets_world), dtype=np.float32)
    best_incidence_quality = np.zeros(len(scenario.targets_world), dtype=np.float32)
    best_distance_quality = np.zeros(len(scenario.targets_world), dtype=np.float32)
    best_visibility_quality = np.zeros(len(scenario.targets_world), dtype=np.float32)
    for index, factor in zip(selected_indices, selected_standoff_factors):
        coverage, quality, incidence_quality, distance_quality, visibility_quality = candidate_variant_metric_components(
            scenario,
            int(index),
            float(factor),
        )
        remaining = np.maximum(0, remaining - coverage.astype(np.int16))
        improved = quality > best_quality
        best_quality = np.maximum(best_quality, quality)
        best_incidence_quality = np.where(improved, incidence_quality, best_incidence_quality)
        best_distance_quality = np.where(improved, distance_quality, best_distance_quality)
        best_visibility_quality = np.maximum(best_visibility_quality, visibility_quality)
    certified_mask = remaining == 0
    building_total: Dict[str, int] = {}
    building_certified: Dict[str, int] = {}
    for building_name, certified in zip(scenario.target_buildings, certified_mask):
        building_total[building_name] = building_total.get(building_name, 0) + 1
        if certified:
            building_certified[building_name] = building_certified.get(building_name, 0) + 1
    weakest = min(
        building_certified.get(name, 0) / max(total, 1)
        for name, total in building_total.items()
    )
    target_region_ids = np.asarray(scenario.target_region_ids, dtype=np.int32)
    if len(target_region_ids):
        weakest_region = min(
            float(np.mean(certified_mask[target_region_ids == region_id].astype(np.float32)))
            for region_id in sorted(set(int(x) for x in target_region_ids.tolist()))
            if np.any(target_region_ids == region_id)
        )
        mean_region_gap = float(
            np.mean(
                [
                    float(np.mean((remaining[target_region_ids == region_id] / max(scenario.required_views, 1)).astype(np.float32)))
                    for region_id in sorted(set(int(x) for x in target_region_ids.tolist()))
                    if np.any(target_region_ids == region_id)
                ]
            )
        )
    else:
        weakest_region = 0.0
        mean_region_gap = 0.0
    target_sector_ids = np.asarray(scenario.target_sector_ids, dtype=np.int32)
    if len(target_sector_ids):
        weakest_sector = min(
            float(np.mean(certified_mask[target_sector_ids == sector_id].astype(np.float32)))
            for sector_id in sorted(set(int(x) for x in target_sector_ids.tolist()))
            if np.any(target_sector_ids == sector_id)
        )
    else:
        weakest_sector = 0.0
    mean_photo_overlap, route_photo_overlap = photo_schedule_overlap_metrics(
        scenario,
        route_indices,
        station_shot_counts_override=route_shot_counts,
        station_standoff_factors_override=route_standoff_factors,
    )
    quality_scale = max(float(np.max(scenario.quality)), 1e-6)
    mean_quality_normalized = float(np.mean(best_quality) / quality_scale)
    weakest_quality_normalized = float(np.min(best_quality) / quality_scale) if len(best_quality) else 0.0
    quality_good_fraction = float(np.mean((best_quality >= QUALITY_GOOD_RATIO * quality_scale).astype(np.float32)))
    mean_incidence_quality = float(np.mean(best_incidence_quality))
    mean_distance_quality = float(np.mean(best_distance_quality))
    mean_visibility_quality = float(np.mean(best_visibility_quality))
    return SelectionResult(
        method=method,
        selected_indices=list(selected_indices),
        route_indices=list(route_indices),
        route_length=float(route_length),
        certified_coverage=float(np.mean(certified_mask.astype(np.float32))),
        weakest_building_coverage=float(weakest),
        weakest_region_coverage=float(weakest_region),
        mean_region_gap=float(mean_region_gap),
        weakest_sector_coverage=float(weakest_sector),
        selected_views=len(selected_indices),
        average_quality=float(np.mean(best_quality)),
        objective=float(objective),
        mean_quality_normalized=mean_quality_normalized,
        quality_good_fraction=quality_good_fraction,
        weakest_quality_normalized=weakest_quality_normalized,
        mean_incidence_quality=mean_incidence_quality,
        mean_distance_quality=mean_distance_quality,
        mean_visibility_quality=mean_visibility_quality,
        mean_photo_overlap=float(mean_photo_overlap),
        route_photo_overlap=float(route_photo_overlap),
        shot_counts=tuple(int(count) for count in selected_shot_counts),
        route_shot_counts=tuple(int(count) for count in route_shot_counts),
        selected_layer_ids=tuple(int(layer) for layer in selected_layer_ids) if selected_layer_ids is not None else tuple(),
        route_layer_ids=tuple(int(layer) for layer in route_layer_ids),
        selected_standoff_factors=tuple(float(value) for value in selected_standoff_factors),
        route_standoff_factors=tuple(float(value) for value in route_standoff_factors),
        photo_count=int(sum(selected_shot_counts)),
    )


def greedy_selection(scenario: Coarse3DScenario) -> List[int]:
    selected, _ = select_multicover(
        scenario.coverage,
        scenario.quality,
        scenario.candidates_local,
        scenario.required_views,
        scenario.candidate_region_codes,
    )
    return selected


def scp_ilp_selection(scenario: Coarse3DScenario) -> List[int]:
    x_count = len(scenario.candidates_local)
    y_count = len(scenario.targets_world)
    total_vars = x_count + y_count
    objective = np.zeros(total_vars, dtype=float)
    objective[:x_count] = 1.0 + 0.02 * np.linalg.norm(
        scenario.candidates_world - scenario.scene_center_world[None, :], axis=1
    ) / max(np.linalg.norm(np.ptp(scenario.candidates_world, axis=0)), 1e-6)
    objective[:x_count] -= 0.03 * np.where(scenario.candidate_region_codes >= 2.0, 1.0, 0.0)
    objective[x_count:] = -0.001

    integrality = np.ones(total_vars, dtype=int)

    def build_problem(strict_building_floor: bool):
        lower = np.zeros(total_vars, dtype=float)
        upper = np.ones(total_vars, dtype=float)
        rows: List[np.ndarray] = []
        lb: List[float] = []
        ub: List[float] = []

        for target_idx in range(y_count):
            row = np.zeros(total_vars, dtype=float)
            row[x_count + target_idx] = 1.0
            covering = 0
            for candidate_idx in range(x_count):
                if scenario.coverage[candidate_idx, target_idx]:
                    row[candidate_idx] -= 1.0 / max(scenario.required_views, 1)
                    covering += 1
            rows.append(row)
            lb.append(-np.inf)
            ub.append(0.0)
            if covering < scenario.required_views:
                upper[x_count + target_idx] = 0.0

        if strict_building_floor:
            building_to_indices: Dict[str, List[int]] = {}
            for idx, building_name in enumerate(scenario.target_buildings):
                building_to_indices.setdefault(building_name, []).append(idx)
            for indices in building_to_indices.values():
                row = np.zeros(total_vars, dtype=float)
                for target_idx in indices:
                    row[x_count + target_idx] = 1.0
                rows.append(row)
                lb.append(math.ceil(0.45 * len(indices)))
                ub.append(np.inf)
            global_row = np.zeros(total_vars, dtype=float)
            global_row[x_count:] = 1.0
            rows.append(global_row)
            lb.append(math.ceil(0.50 * y_count))
            ub.append(np.inf)
        return lower, upper, rows, lb, ub

    for strict in (True, False):
        lower, upper, rows, lb, ub = build_problem(strict)
        result = milp(
            c=objective,
            integrality=integrality,
            bounds=Bounds(lower, upper),
            constraints=LinearConstraint(np.vstack(rows), np.asarray(lb), np.asarray(ub)),
        )
        if result.success and result.x is not None:
            return [int(idx) for idx, keep in enumerate(result.x[:x_count] > 0.5) if keep]
    raise RuntimeError("SCP/ILP failed in both strict and relaxed modes")


class Coarse3DSelectionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: Coarse3DScenario,
        max_steps: int,
        scenario_repository: Coarse3DScenarioRepository | None = None,
        episode_seeds: Sequence[int] | None = None,
        action_mode: str = "continuous",
    ) -> None:
        super().__init__()
        self.scenario = scenario
        self.scenario_repository = scenario_repository
        self.episode_seeds = tuple(episode_seeds or ())
        self.seed_cursor = 0
        self.max_steps = max_steps
        self.action_mode = action_mode
        self.max_shots_per_station = MAX_SHOTS_PER_STATION
        self.num_distance_layers = len(LEGACY_STANDOFF_LAYER_FACTORS)
        if self.action_mode == "discrete":
            self.action_space = spaces.MultiDiscrete(
                [
                    max(len(scenario.candidates_local), 1),
                    self.max_shots_per_station,
                    self.num_distance_layers,
                ]
            )
        else:
            self.action_space = spaces.Box(
                low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        self.observation_space = spaces.Dict(
            {
                "global": spaces.Box(low=-10.0, high=10.0, shape=(19,), dtype=np.float32),
                "candidates": spaces.Box(
                    low=-10.0,
                    high=10.0,
                    shape=(len(scenario.candidates_local), 13),
                    dtype=np.float32,
                ),
            }
        )
        self._set_scenario(scenario)

    def _set_scenario(self, scenario: Coarse3DScenario) -> None:
        self.scenario = scenario
        self.quality_scale = max(float(np.max(scenario.quality)), 1e-6)
        self.quality_good_threshold = QUALITY_GOOD_RATIO * self.quality_scale
        self.quality_mean_target = PPOTrainingConfig().quality_mean_target
        self.quality_good_fraction_target = PPOTrainingConfig().quality_good_fraction_target
        self.weakest_quality_target = PPOTrainingConfig().weakest_quality_target
        self.selected_mask = np.zeros(len(scenario.candidates_local), dtype=bool)
        self.selected_order: List[int] = []
        self.selected_shot_counts: List[int] = []
        self.selected_layer_ids: List[int] = []
        self.selected_standoff_factors: List[float] = []
        self.remaining = np.full(len(scenario.targets_world), scenario.required_views, dtype=np.int16)
        self.best_quality = np.zeros(len(scenario.targets_world), dtype=np.float32)
        self.steps = 0
        self.current_position = scenario.scene_center_world.copy()
        self.building_names = sorted(set(scenario.target_buildings))
        self.building_target_indices = {
            name: np.array([idx for idx, b in enumerate(scenario.target_buildings) if b == name], dtype=int)
            for name in self.building_names
        }
        self.owner_buildings = list(scenario.candidate_owners)
        self.cluster_ids = np.asarray(scenario.candidate_cluster_ids, dtype=np.int32)
        self.region_ids = np.asarray(scenario.candidate_region_ids, dtype=np.int32)
        self.sector_ids = np.asarray(scenario.candidate_sector_ids, dtype=np.int32)
        self.target_region_ids = np.asarray(scenario.target_region_ids, dtype=np.int32)
        self.target_sector_ids = np.asarray(scenario.target_sector_ids, dtype=np.int32)
        self.unique_clusters = sorted(set(int(x) for x in self.cluster_ids.tolist()))
        self.unique_regions = sorted(set(int(x) for x in self.region_ids.tolist()))
        self.unique_sectors = sorted(set(int(x) for x in self.sector_ids.tolist()))
        self.unique_target_regions = sorted(set(int(x) for x in self.target_region_ids.tolist()))
        self.unique_target_sectors = sorted(set(int(x) for x in self.target_sector_ids.tolist()))
        self.cluster_target_indices = self._build_cluster_target_indices()
        self.region_target_indices = self._build_region_target_indices()
        self.sector_target_indices = self._build_sector_target_indices()
        self.target_support = np.maximum(1, np.count_nonzero(scenario.coverage, axis=0)).astype(np.float32)
        z_values = scenario.candidates_world[:, 2]
        if len(z_values) >= 4 and float(np.max(z_values) - np.min(z_values)) > 1e-6:
            z_edges = np.quantile(z_values, [0.0, 0.25, 0.5, 0.75, 1.0])
            z_edges = np.maximum.accumulate(z_edges)
            z_edges[-1] += 1e-6
        else:
            z_edges = np.array([float(np.min(z_values, initial=0.0)), float(np.max(z_values, initial=1.0)) + 1e-6])
        self.candidate_z_bins = np.clip(np.digitize(z_values, z_edges[1:-1], right=False), 0, max(len(z_edges) - 2, 0))
        self.z_bin_counts = np.zeros(max(int(np.max(self.candidate_z_bins)) + 1 if len(self.candidate_z_bins) else 1, 1), dtype=np.int32)
        self.sector_counts = np.zeros(max(int(np.max(self.sector_ids)) + 1 if len(self.sector_ids) else 1, 1), dtype=np.int32)
        self.region_counts = np.zeros(max(int(np.max(self.region_ids)) + 1 if len(self.region_ids) else 1, 1), dtype=np.int32)
        self.static_candidate_features = self._build_static_candidate_features()

    def _mean_quality_normalized(self) -> float:
        return float(np.mean(self.best_quality) / max(self.quality_scale, 1e-6))

    def _weakest_quality_normalized(self) -> float:
        return float(np.min(self.best_quality) / max(self.quality_scale, 1e-6)) if len(self.best_quality) else 0.0

    def _good_quality_fraction(self) -> float:
        if len(self.best_quality) == 0:
            return 0.0
        return float(np.mean((self.best_quality >= self.quality_good_threshold).astype(np.float32)))

    def _build_cluster_target_indices(self) -> Dict[int, np.ndarray]:
        mapping: Dict[int, List[int]] = {}
        for cluster_id in self.unique_clusters:
            member_ids = np.flatnonzero(self.cluster_ids == cluster_id)
            if len(member_ids) == 0:
                continue
            covered = np.any(self.scenario.coverage[member_ids], axis=0)
            mapping[cluster_id] = list(np.flatnonzero(covered))
        return {key: np.asarray(value, dtype=int) for key, value in mapping.items()}

    def _build_region_target_indices(self) -> Dict[int, np.ndarray]:
        mapping: Dict[int, List[int]] = {}
        for region_id in self.unique_target_regions:
            indices = np.flatnonzero(self.target_region_ids == region_id)
            if len(indices) == 0:
                continue
            mapping[region_id] = list(indices)
        return {key: np.asarray(value, dtype=int) for key, value in mapping.items()}

    def _build_sector_target_indices(self) -> Dict[int, np.ndarray]:
        mapping: Dict[int, List[int]] = {}
        for sector_id in self.unique_target_sectors:
            indices = np.flatnonzero(self.target_sector_ids == sector_id)
            if len(indices) == 0:
                continue
            mapping[sector_id] = list(indices)
        return {key: np.asarray(value, dtype=int) for key, value in mapping.items()}

    def _building_coverage(self, building_name: str) -> float:
        indices = self.building_target_indices[building_name]
        return float(np.mean((self.remaining[indices] == 0).astype(np.float32)))

    def _weakest_building_coverage(self) -> float:
        return min(self._building_coverage(name) for name in self.building_names)

    def _effective_coverage(self) -> float:
        return float(np.mean((self.remaining == 0).astype(np.float32)))

    def _current_overlap_metrics(self) -> tuple[float, float]:
        if len(self.selected_order) == 0:
            return 0.0, 0.0
        return photo_schedule_overlap_metrics(
            self.scenario,
            self.selected_order,
            station_shot_counts_override=self.selected_shot_counts,
            station_standoff_factors_override=self.selected_standoff_factors,
        )

    def _cluster_coverage(self, cluster_id: int) -> float:
        indices = self.cluster_target_indices.get(cluster_id)
        if indices is None or len(indices) == 0:
            return 0.0
        return float(np.mean((self.remaining[indices] == 0).astype(np.float32)))

    def _region_coverage(self, region_id: int) -> float:
        indices = self.region_target_indices.get(region_id)
        if indices is None or len(indices) == 0:
            return 0.0
        return float(np.mean((self.remaining[indices] == 0).astype(np.float32)))

    def _weakest_region_coverage(self) -> float:
        if not self.unique_target_regions:
            return 0.0
        return min(self._region_coverage(region_id) for region_id in self.unique_target_regions)

    def _region_remaining_fraction(self, region_id: int) -> float:
        indices = self.region_target_indices.get(region_id)
        if indices is None or len(indices) == 0:
            return 0.0
        return float(np.mean(self.remaining[indices] / max(self.scenario.required_views, 1)))

    def _mean_region_remaining_fraction(self) -> float:
        if not self.unique_target_regions:
            return 0.0
        return float(np.mean([self._region_remaining_fraction(region_id) for region_id in self.unique_target_regions]))

    def _region_remaining_std(self) -> float:
        if not self.unique_target_regions:
            return 0.0
        values = [self._region_remaining_fraction(region_id) for region_id in self.unique_target_regions]
        return float(np.std(values))

    def _most_deficient_region_id(self) -> int:
        if not self.unique_target_regions:
            return -1
        return int(max(self.unique_target_regions, key=lambda region_id: self._region_remaining_fraction(region_id)))

    def _most_deficient_region_fraction(self) -> float:
        if not self.unique_target_regions:
            return 0.0
        return float(max(self._region_remaining_fraction(region_id) for region_id in self.unique_target_regions))

    def _sector_coverage(self, sector_id: int) -> float:
        indices = self.sector_target_indices.get(sector_id)
        if indices is None or len(indices) == 0:
            return 0.0
        return float(np.mean((self.remaining[indices] == 0).astype(np.float32)))

    def _weakest_sector_coverage(self) -> float:
        if not self.unique_target_sectors:
            return 0.0
        return min(self._sector_coverage(sector_id) for sector_id in self.unique_target_sectors)

    def _sector_remaining_fraction(self, sector_id: int) -> float:
        indices = self.sector_target_indices.get(sector_id)
        if indices is None or len(indices) == 0:
            return 0.0
        return float(np.mean(self.remaining[indices] / max(self.scenario.required_views, 1)))

    def _build_static_candidate_features(self) -> np.ndarray:
        positions = self.scenario.candidates_world
        center_distance = np.linalg.norm(positions - self.scenario.scene_center_world[None, :], axis=1)
        max_center_distance = max(float(center_distance.max()), 1e-6)
        mean_quality = self.scenario.quality.mean(axis=1)
        raw_gain = self.scenario.coverage.sum(axis=1).astype(np.float32) / max(len(self.scenario.targets_world), 1)
        importance = self.scenario.candidate_importance / max(float(self.scenario.candidate_importance.max()), 1e-6)
        z_values = positions[:, 2]
        z_norm = (z_values - z_values.min()) / max(float(z_values.max() - z_values.min()), 1e-6)
        unassigned_flag = (np.asarray(self.scenario.candidate_owners, dtype=object) == "unassigned").astype(np.float32)
        support_gain = (
            (self.scenario.coverage.astype(np.float32) / self.target_support[None, :]).sum(axis=1)
            / max(len(self.scenario.targets_world), 1)
        )
        return np.column_stack(
            [
                mean_quality,
                raw_gain,
                support_gain,
                self.scenario.candidate_region_codes / 4.0,
                importance,
                center_distance / max_center_distance,
                z_norm,
                unassigned_flag,
            ]
        ).astype(np.float32)

    def _candidate_action_mask(self) -> np.ndarray:
        useful = (self.scenario.coverage[:, self.remaining > 0]).any(axis=1)
        candidate_valid = (~self.selected_mask) & useful
        if not np.any(candidate_valid) and len(candidate_valid):
            candidate_valid = candidate_valid.copy()
            candidate_valid[0] = True
        return candidate_valid

    def action_masks(self) -> np.ndarray:
        candidate_valid = self._candidate_action_mask()
        if self.action_mode == "discrete":
            shot_valid = np.ones(self.max_shots_per_station, dtype=bool)
            layer_valid = np.ones(self.num_distance_layers, dtype=bool)
            return np.concatenate([candidate_valid.astype(bool), shot_valid, layer_valid])
        return candidate_valid

    def _candidate_priority_scores(self) -> np.ndarray:
        need = self.remaining > 0
        if np.any(need):
            gains = self.scenario.coverage[:, need].sum(axis=1).astype(np.float32) / max(np.count_nonzero(need), 1)
            quality_gain = self.scenario.quality[:, need].sum(axis=1).astype(np.float32) / max(np.count_nonzero(need), 1)
            scarcity_gain = (
                (
                    self.scenario.coverage[:, need].astype(np.float32)
                    / self.target_support[need][None, :]
                ).sum(axis=1)
                / max(np.count_nonzero(need), 1)
            )
        else:
            gains = np.zeros(len(self.scenario.candidates_local), dtype=np.float32)
            quality_gain = np.zeros(len(self.scenario.candidates_local), dtype=np.float32)
            scarcity_gain = np.zeros(len(self.scenario.candidates_local), dtype=np.float32)
        travel = np.linalg.norm(
            self.scenario.candidates_world - self.current_position[None, :],
            axis=1,
        )
        travel = travel / max(float(travel.max()), 1e-6)
        center_distance = np.linalg.norm(
            self.scenario.candidates_world[:, :2] - self.scenario.scene_center_world[None, :2],
            axis=1,
        )
        center_distance = center_distance / max(float(center_distance.max()), 1e-6)
        unassigned_flag = (np.asarray(self.scenario.candidate_owners, dtype=object) == "unassigned").astype(np.float32)
        owner_deficit = np.array(
            [1.0 - self._building_coverage(name) if name in self.building_target_indices else 0.0 for name in self.owner_buildings],
            dtype=np.float32,
        )
        cluster_deficit = np.array([1.0 - self._cluster_coverage(int(cluster_id)) for cluster_id in self.cluster_ids], dtype=np.float32)
        region_deficit = np.array([1.0 - self._region_coverage(int(region_id)) for region_id in self.region_ids], dtype=np.float32)
        base = candidate_score_vector(self.scenario).astype(np.float32)
        return (
            0.35 * base
            + 1.25 * gains
            + 1.05 * quality_gain
            + 0.80 * scarcity_gain
            + 0.50 * owner_deficit
            + 0.32 * region_deficit
            + 0.20 * cluster_deficit
            + 0.24 * center_distance
            - 0.90 * unassigned_flag
            - 0.30 * travel
        )

    def _ordered_valid_candidates(self) -> np.ndarray:
        candidate_valid = self._candidate_action_mask()
        valid_indices = np.flatnonzero(candidate_valid)
        if len(valid_indices) == 0:
            return valid_indices
        scores = self._candidate_priority_scores()[valid_indices]
        return valid_indices[np.argsort(-scores, kind="mergesort")]

    def _decode_action(self, action: int | Sequence[int] | np.ndarray) -> tuple[int, int, float, float, int]:
        raw = np.asarray(action).reshape(-1)
        if self.action_mode == "discrete":
            if raw.size < 3:
                raw = np.pad(raw, (0, max(0, 3 - raw.size)), constant_values=0)
            candidate_raw = int(raw[0])
            shot_raw = int(raw[1])
            layer_idx = int(raw[2])
            valid_order = self._ordered_valid_candidates()
            if len(valid_order) == 0:
                raise RuntimeError("no valid candidate available")
            candidate_rank = int(np.clip(candidate_raw, 0, len(valid_order) - 1))
            candidate_idx = int(valid_order[candidate_rank])
            shot_count = int(np.clip(1 + shot_raw, 1, self.max_shots_per_station))
            layer_idx = int(np.clip(layer_idx, 0, self.num_distance_layers - 1))
            standoff_factor = float(LEGACY_STANDOFF_LAYER_FACTORS[layer_idx])
            return candidate_idx, shot_count, standoff_factor, 0.0, layer_idx
        raw = raw.astype(np.float32)
        if raw.size < 3:
            raw = np.pad(raw, (0, max(0, 3 - raw.size)), constant_values=0.0)
        clipped = np.clip(raw[:3], 0.0, 1.0)
        projection_penalty = float(np.linalg.norm(raw[:3] - clipped))
        valid_order = self._ordered_valid_candidates()
        if len(valid_order) == 0:
            raise RuntimeError("no valid candidate available")
        candidate_rank = int(np.clip(round(float(clipped[0]) * max(len(valid_order) - 1, 0)), 0, len(valid_order) - 1))
        candidate_idx = int(valid_order[candidate_rank])
        shot_count = int(np.clip(round(1.0 + float(clipped[1]) * (self.max_shots_per_station - 1)), 1, self.max_shots_per_station))
        standoff_factor = float(
            self.scenario.standoff_factor_min
            + float(clipped[2]) * (self.scenario.standoff_factor_max - self.scenario.standoff_factor_min)
        )
        layer_idx = legacy_standoff_layer_index(self.scenario, standoff_factor)
        return candidate_idx, shot_count, standoff_factor, projection_penalty, layer_idx

    def _observation(self) -> Dict[str, np.ndarray]:
        gains = np.zeros(len(self.scenario.candidates_local), dtype=np.float32)
        quality_gain = np.zeros(len(self.scenario.candidates_local), dtype=np.float32)
        if np.any(self.remaining > 0):
            need = self.remaining > 0
            gains = self.scenario.coverage[:, need].sum(axis=1).astype(np.float32) / max(np.count_nonzero(need), 1)
            quality_gain = self.scenario.quality[:, need].sum(axis=1).astype(np.float32) / max(np.count_nonzero(need), 1)
            scarcity_gain = (
                (
                    self.scenario.coverage[:, need].astype(np.float32)
                    / self.target_support[need][None, :]
                ).sum(axis=1)
                / max(np.count_nonzero(need), 1)
            )
        else:
            scarcity_gain = np.zeros(len(self.scenario.candidates_local), dtype=np.float32)
        travel = np.linalg.norm(
            self.scenario.candidates_world - self.current_position[None, :],
            axis=1,
        )
        travel = travel / max(float(travel.max()), 1e-6)
        center_distance = np.linalg.norm(
            self.scenario.candidates_world[:, :2] - self.scenario.scene_center_world[None, :2],
            axis=1,
        )
        center_distance = center_distance / max(float(center_distance.max()), 1e-6)
        unassigned_flag = (np.asarray(self.scenario.candidate_owners, dtype=object) == "unassigned").astype(np.float32)
        owner_deficit = np.array(
            [1.0 - self._building_coverage(name) if name in self.building_target_indices else 0.0 for name in self.owner_buildings],
            dtype=np.float32,
        )
        cluster_deficit = np.array(
            [1.0 - self._cluster_coverage(int(cluster_id)) for cluster_id in self.cluster_ids],
            dtype=np.float32,
        )
        region_deficit = np.array(
            [
                1.0 - self._region_coverage(int(region_id))
                for region_id in self.region_ids
            ],
            dtype=np.float32,
        )
        candidate_matrix = np.zeros((len(self.scenario.candidates_local), 13), dtype=np.float32)
        candidate_matrix[:, :8] = self.static_candidate_features
        candidate_matrix[:, 8] = gains + 0.5 * quality_gain
        candidate_matrix[:, 9] = scarcity_gain
        candidate_matrix[:, 10] = 1.0 - travel
        candidate_matrix[:, 11] = owner_deficit
        candidate_matrix[:, 12] = 0.5 * cluster_deficit + 0.5 * region_deficit + 0.2 * center_distance - 0.6 * unassigned_flag
        building_coverages = [self._building_coverage(name) for name in self.building_names]
        current_forward_overlap, current_lateral_overlap = self._current_overlap_metrics()
        global_features = np.array(
            [
                self._effective_coverage(),
                self._mean_quality_normalized(),
                self._weakest_building_coverage(),
                self._weakest_region_coverage(),
                self._good_quality_fraction(),
                self._weakest_quality_normalized(),
                float(np.std(building_coverages)),
                float(np.mean(building_coverages)),
                float(np.min([self._cluster_coverage(cluster_id) for cluster_id in self.unique_clusters]) if self.unique_clusters else 0.0),
                float(np.mean([self._region_coverage(region_id) for region_id in self.unique_target_regions]) if self.unique_target_regions else 0.0),
                self._mean_region_remaining_fraction(),
                self._region_remaining_std(),
                self._most_deficient_region_fraction(),
                self.steps / max(self.max_steps, 1),
                float(np.mean(self.remaining / max(self.scenario.required_views, 1))),
                float(np.mean(self.selected_mask.astype(np.float32))),
                float(np.mean(self.action_masks().astype(np.float32))),
                current_forward_overlap,
                current_lateral_overlap,
            ],
            dtype=np.float32,
        )
        return {"global": global_features, "candidates": candidate_matrix}

    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        super().reset(seed=seed)
        if self.scenario_repository is not None and self.episode_seeds:
            variant_seed = self.episode_seeds[self.seed_cursor % len(self.episode_seeds)]
            self.seed_cursor += 1
            self._set_scenario(self.scenario_repository.get(int(variant_seed)))
        else:
            self._set_scenario(self.scenario)
        return self._observation(), {}

    def step(self, action: int | Sequence[int] | np.ndarray):
        valid_order = self._ordered_valid_candidates()
        if len(valid_order) == 0:
            return self._observation(), 0.0, True, False, {"no_valid_action": True}
        candidate_idx, shot_count, standoff_factor, projection_penalty, layer_idx = self._decode_action(action)
        before_effective = self._effective_coverage()
        before_forward_overlap, before_lateral_overlap = self._current_overlap_metrics()
        before_weakest = self._weakest_building_coverage()
        before_region_weakest = self._weakest_region_coverage()
        before_mean_quality = self._mean_quality_normalized()
        before_good_quality_fraction = self._good_quality_fraction()
        before_weakest_quality = self._weakest_quality_normalized()
        owner_name = self.owner_buildings[candidate_idx]
        before_owner = self._building_coverage(owner_name) if owner_name in self.building_target_indices else 0.0
        cluster_id = int(self.cluster_ids[candidate_idx])
        before_cluster = self._cluster_coverage(cluster_id)
        region_id = int(self.region_ids[candidate_idx]) if len(self.region_ids) else 0
        before_region = self._region_coverage(region_id)
        before_region_remaining = self._region_remaining_fraction(region_id)
        before_mean_region_remaining = self._mean_region_remaining_fraction()
        before_region_remaining_std = self._region_remaining_std()
        before_max_region_remaining = self._most_deficient_region_fraction()
        most_deficient_region = self._most_deficient_region_id()
        need = self.remaining > 0
        variant_coverage, variant_quality = candidate_variant_metrics(self.scenario, candidate_idx, standoff_factor)
        candidate_position_world = standoff_position_world(self.scenario, candidate_idx, standoff_factor)
        center_shell = float(
            np.linalg.norm(candidate_position_world[:2] - self.scenario.scene_center_world[:2])
        ) / max(float(np.linalg.norm(self.scenario.scene_high_world[:2] - self.scenario.scene_low_world[:2])), 1e-6)
        owner_is_unassigned = 1.0 if owner_name == "unassigned" else 0.0
        raw_gain = float(np.count_nonzero(variant_coverage & need)) / max(np.count_nonzero(need), 1)
        weighted_gain = float(variant_quality[need].sum()) / max(np.count_nonzero(need), 1)
        region_indices = self.region_target_indices.get(region_id)
        if region_indices is not None and len(region_indices) > 0:
            region_need = self.remaining[region_indices] > 0
            region_raw_gain = float(np.count_nonzero(variant_coverage[region_indices] & region_need)) / max(len(region_indices), 1)
            region_weighted_gain = float(variant_quality[region_indices][region_need].sum()) / max(len(region_indices), 1)
        else:
            region_raw_gain = 0.0
            region_weighted_gain = 0.0
        scarcity_gain = float(
            (
                variant_coverage[need].astype(np.float32)
                / self.target_support[need]
            ).sum()
            / max(np.count_nonzero(need), 1)
        ) if np.any(need) else 0.0
        travel_penalty = float(np.linalg.norm(candidate_position_world - self.current_position)) / max(
            self.scenario.stand_off * 6.0,
            1e-6,
        )
        standoff_penalty = abs(standoff_factor - 1.0)
        spacing_penalty = 0.0
        if np.any(self.selected_mask):
            chosen = np.flatnonzero(self.selected_mask)
            chosen_points = selected_variant_points_world(
                self.scenario,
                self.selected_order,
                selected_layer_ids=self.selected_layer_ids,
                selected_standoff_factors=self.selected_standoff_factors,
            )
            xy_dist = np.linalg.norm(chosen_points[:, :2] - candidate_position_world[None, :2], axis=1)
            z_dist = np.abs(chosen_points[:, 2] - candidate_position_world[2])
            same_cluster = self.cluster_ids[chosen] == cluster_id
            same_owner = np.array([self.owner_buildings[idx] == owner_name for idx in chosen], dtype=bool)
            close_mask = (xy_dist < max(4.0, 0.45 * self.scenario.stand_off)) & (same_cluster | same_owner | (z_dist < max(3.0, 0.20 * self.scenario.stand_off)))
            if np.any(close_mask):
                spacing_penalty = float(np.max((max(4.0, 0.45 * self.scenario.stand_off) - xy_dist[close_mask]) / max(max(4.0, 0.45 * self.scenario.stand_off), 1e-6)))
        previous_station_idx = int(self.selected_order[-1]) if self.selected_order else None
        previous_station_factor = float(self.selected_standoff_factors[-1]) if self.selected_standoff_factors else 1.0
        self.selected_mask[candidate_idx] = True
        self.selected_order.append(int(candidate_idx))
        self.selected_shot_counts.append(int(shot_count))
        self.selected_layer_ids.append(int(layer_idx))
        self.selected_standoff_factors.append(float(standoff_factor))
        self.remaining = np.maximum(0, self.remaining - variant_coverage.astype(np.int16))
        self.best_quality = np.maximum(self.best_quality, variant_quality)
        self.current_position = candidate_position_world
        self.steps += 1
        z_bin = int(self.candidate_z_bins[candidate_idx]) if len(self.candidate_z_bins) else 0
        before_z_bin_used = float(np.count_nonzero(self.z_bin_counts > 0)) / max(len(self.z_bin_counts), 1)
        self.z_bin_counts[z_bin] += 1
        after_z_bin_used = float(np.count_nonzero(self.z_bin_counts > 0)) / max(len(self.z_bin_counts), 1)
        before_region_used = float(np.count_nonzero(self.region_counts > 0)) / max(len(self.region_counts), 1)
        self.region_counts[region_id] += 1
        after_region_used = float(np.count_nonzero(self.region_counts > 0)) / max(len(self.region_counts), 1)
        after_owner = self._building_coverage(owner_name) if owner_name in self.building_target_indices else before_owner
        after_cluster = self._cluster_coverage(cluster_id)
        after_region = self._region_coverage(region_id)
        after_region_remaining = self._region_remaining_fraction(region_id)
        after_mean_region_remaining = self._mean_region_remaining_fraction()
        after_region_remaining_std = self._region_remaining_std()
        after_max_region_remaining = self._most_deficient_region_fraction()
        after_mean_quality = self._mean_quality_normalized()
        after_good_quality_fraction = self._good_quality_fraction()
        after_weakest_quality = self._weakest_quality_normalized()
        after_forward_overlap, after_lateral_overlap = self._current_overlap_metrics()
        quality_norm_gain = float(np.mean(variant_quality) / max(self.quality_scale, 1e-6))
        quality_new_fraction = float(np.count_nonzero((variant_quality >= self.quality_good_threshold) & need) / max(np.count_nonzero(need), 1)) if np.any(need) else 0.0
        shot_readiness = max(float(shot_count) - 1.0, 0.0) / max(float(self.max_shots_per_station - 1), 1.0)
        forward_gap_before = max(FORWARD_OVERLAP_TARGET - before_forward_overlap, 0.0)
        forward_need_ratio = min(forward_gap_before / max(FORWARD_OVERLAP_TARGET, 1e-6), 1.0)
        lateral_need_ratio = min(
            max(LATERAL_OVERLAP_TARGET - before_lateral_overlap, 0.0) / max(LATERAL_OVERLAP_TARGET, 1e-6),
            1.0,
        )
        coverage_need_ratio = min(
            max(COVERAGE_TARGET - before_effective, 0.0) / max(COVERAGE_TARGET, 1e-6),
            1.0,
        )
        desired_shot_readiness = min(
            1.0,
            0.62
            + 0.20 * forward_need_ratio
            + 0.10 * lateral_need_ratio
            + 0.08 * coverage_need_ratio,
        )
        shot_alignment = max(0.0, 1.0 - abs(shot_readiness - desired_shot_readiness))
        local_forward_overlap = local_station_forward_overlap(shot_count)
        local_lateral_overlap = (
            local_pair_lateral_overlap(
                self.scenario,
                previous_station_idx,
                previous_station_factor,
                candidate_idx,
                standoff_factor,
            )
            if previous_station_idx is not None
            else 0.0
        )
        region_priority_bonus = 0.0
        if region_id == most_deficient_region:
            region_priority_bonus = 10.0 * (before_max_region_remaining - after_max_region_remaining)
        quality_reward = (
            34.0 * (after_mean_quality - before_mean_quality)
            + 24.0 * (after_good_quality_fraction - before_good_quality_fraction)
            + 24.0 * (after_weakest_quality - before_weakest_quality)
            + 14.0 * quality_norm_gain
            + 10.0 * quality_new_fraction
        )
        overlap_reward = (
            30.0 * (after_forward_overlap - before_forward_overlap)
            + 24.0 * (after_lateral_overlap - before_lateral_overlap)
            + 10.0 * (min(after_forward_overlap, FORWARD_OVERLAP_TARGET) - min(before_forward_overlap, FORWARD_OVERLAP_TARGET))
            + 8.0 * (min(after_lateral_overlap, LATERAL_OVERLAP_TARGET) - min(before_lateral_overlap, LATERAL_OVERLAP_TARGET))
        )
        shot_structure_reward = (
            24.0 * forward_gap_before * shot_alignment
            + 16.0 * forward_gap_before * shot_readiness
            - 8.0 * forward_gap_before * (1.0 - shot_readiness)
        )
        local_overlap_reward = (
            22.0 * local_forward_overlap
            + 14.0 * min(local_forward_overlap / max(FORWARD_OVERLAP_TARGET, 1e-6), 1.0)
            + 14.0 * local_lateral_overlap
            + 8.0 * min(local_lateral_overlap / max(LATERAL_OVERLAP_TARGET, 1e-6), 1.0)
        )
        shot_cost_penalty = (
            0.01 if before_forward_overlap < FORWARD_OVERLAP_TARGET else 0.10
        ) * max(int(shot_count) - 1, 0)
        reward = (
            22.0 * (self._effective_coverage() - before_effective)
            + 24.0 * floor_sensitive_gain(self._weakest_building_coverage(), before_weakest)
            + 34.0 * (self._weakest_region_coverage() - before_region_weakest)
            + 16.0 * (before_mean_region_remaining - after_mean_region_remaining)
            + 8.0 * (before_region_remaining_std - after_region_remaining_std)
            + region_priority_bonus
            + 14.0 * (after_owner - before_owner)
            + 6.0 * (after_cluster - before_cluster)
            + 28.0 * (after_region - before_region)
            + 12.0 * (before_region_remaining - after_region_remaining)
            + 14.0 * region_raw_gain
            + 10.0 * region_weighted_gain
            + 3.0 * (after_z_bin_used - before_z_bin_used)
            + 5.0 * (after_region_used - before_region_used)
            + 4.0 * raw_gain
            + 2.8 * weighted_gain
            + 5.0 * scarcity_gain
            + 2.2 * center_shell
            - 3.0 * owner_is_unassigned
            + quality_reward
            + overlap_reward
            + local_overlap_reward
            + shot_structure_reward
            - 0.42 * travel_penalty
            - 0.75 * standoff_penalty
            - 2.4 * spacing_penalty
            - 0.35 * projection_penalty
            - shot_cost_penalty
            - 0.05
        )
        coverage_done = self._effective_coverage() >= COVERAGE_TARGET - 1e-6
        overlap_done = after_forward_overlap >= FORWARD_OVERLAP_TARGET and after_lateral_overlap >= LATERAL_OVERLAP_TARGET
        quality_done = (
            after_mean_quality >= self.quality_mean_target
            and after_good_quality_fraction >= self.quality_good_fraction_target
            and after_weakest_quality >= self.weakest_quality_target
        )
        coverage_progress = min(self._effective_coverage() / max(COVERAGE_TARGET, 1e-6), 1.0)
        overlap_progress = min(
            min(
                after_forward_overlap / max(FORWARD_OVERLAP_TARGET, 1e-6),
                after_lateral_overlap / max(LATERAL_OVERLAP_TARGET, 1e-6),
            ),
            1.0,
        )
        quality_progress = min(
            min(
                after_mean_quality / max(self.quality_mean_target, 1e-6),
                after_good_quality_fraction / max(self.quality_good_fraction_target, 1e-6),
                after_weakest_quality / max(self.weakest_quality_target, 1e-6),
            ),
            1.0,
        )
        completion_bonus = 90.0 if coverage_done and overlap_done and quality_done else 0.0
        completion_bonus += 16.0 * coverage_progress
        completion_bonus += 16.0 * overlap_progress
        completion_bonus += 16.0 * quality_progress
        done = (
            self.steps >= self.max_steps
            or (coverage_done and overlap_done and quality_done)
            or len(valid_order) <= 1
        )
        return self._observation(), float(reward + completion_bonus), done, False, {
            "projection_penalty": projection_penalty,
            "quality_done": quality_done,
        }


def train_continuous_distance_ppo(
    scenario: Coarse3DScenario,
    training: PPOTrainingConfig,
) -> Tuple[PPO, List[TrainingHistoryRow]]:
    repository = Coarse3DScenarioRepository(
        scenario,
        keep_candidates=len(scenario.candidates_local),
        variant_seeds=tuple(range(training.seed, training.seed + 8)),
    )

    def make_env():
        env = Coarse3DSelectionEnv(
            scenario,
            max_steps=selection_episode_cap(scenario),
            scenario_repository=repository,
            episode_seeds=repository.variant_seeds,
        )
        return Monitor(env)

    vec_env = DummyVecEnv([make_env])
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        learning_rate=training.learning_rate,
        n_steps=training.n_steps,
        batch_size=training.batch_size,
        n_epochs=training.n_epochs,
        gamma=training.gamma,
        gae_lambda=training.gae_lambda,
        ent_coef=training.ent_coef,
        vf_coef=training.vf_coef,
        clip_range=training.clip_range,
        verbose=0,
        policy_kwargs={"net_arch": {"pi": [256, 128], "vf": [256, 128]}},
        seed=training.seed,
    )
    callback = SelectionEvalCallback(
        scenario,
        eval_every_steps=training.eval_every_steps,
        policy_runner=run_continuous_distance_ppo,
    )
    model.learn(total_timesteps=training.total_timesteps, progress_bar=False, callback=callback)
    if callback.best_state_dict is not None:
        model.policy.load_state_dict(callback.best_state_dict)
    return model, callback.history


def selection_episode_cap(scenario: Coarse3DScenario) -> int:
    if len(scenario.candidates_local) == 0:
        return 1
    return max(1, min(len(scenario.candidates_local), max(192, len(scenario.candidates_local) // 3)))


def run_continuous_distance_ppo(scenario: Coarse3DScenario, model: PPO) -> SelectionPolicyOutput:
    env = Coarse3DSelectionEnv(scenario, max_steps=selection_episode_cap(scenario))
    observation, _ = env.reset()
    while True:
        valid_order = env._ordered_valid_candidates()
        if len(valid_order) == 0:
            break
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return SelectionPolicyOutput(
        selected_indices=[int(index) for index in env.selected_order],
        shot_counts=[int(count) for count in env.selected_shot_counts],
        selected_layer_ids=[int(layer) for layer in env.selected_layer_ids],
        selected_standoff_factors=[float(value) for value in env.selected_standoff_factors],
    )


def train_maskable_ppo(
    scenario: Coarse3DScenario,
    training: PPOTrainingConfig,
) -> Tuple[MaskablePPO, List[TrainingHistoryRow]]:
    repository = Coarse3DScenarioRepository(
        scenario,
        keep_candidates=len(scenario.candidates_local),
        variant_seeds=tuple(range(training.seed, training.seed + 8)),
    )

    def make_env():
        env = Coarse3DSelectionEnv(
            scenario,
            max_steps=selection_episode_cap(scenario),
            scenario_repository=repository,
            episode_seeds=repository.variant_seeds,
            action_mode="discrete",
        )
        return Monitor(ActionMasker(env, lambda wrapped_env: wrapped_env.unwrapped.action_masks()))

    vec_env = DummyVecEnv([make_env])
    model = MaskablePPO(
        "MultiInputPolicy",
        vec_env,
        learning_rate=training.learning_rate,
        n_steps=training.n_steps,
        batch_size=training.batch_size,
        n_epochs=training.n_epochs,
        gamma=training.gamma,
        gae_lambda=training.gae_lambda,
        ent_coef=training.ent_coef,
        vf_coef=training.vf_coef,
        clip_range=training.clip_range,
        verbose=0,
        policy_kwargs={"net_arch": {"pi": [256, 128], "vf": [256, 128]}},
        seed=training.seed,
    )
    callback = SelectionEvalCallback(scenario, eval_every_steps=training.eval_every_steps, policy_runner=run_maskable_ppo)
    model.learn(total_timesteps=training.total_timesteps, progress_bar=False, callback=callback)
    if callback.best_state_dict is not None:
        model.policy.load_state_dict(callback.best_state_dict)
    return model, callback.history


def run_maskable_ppo(scenario: Coarse3DScenario, model: MaskablePPO) -> SelectionPolicyOutput:
    env = Coarse3DSelectionEnv(scenario, max_steps=selection_episode_cap(scenario), action_mode="discrete")
    observation, _ = env.reset()
    while True:
        masks = env.action_masks()
        if not np.any(masks):
            break
        action, _ = model.predict(observation, deterministic=True, action_masks=masks)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return SelectionPolicyOutput(
        selected_indices=[int(index) for index in env.selected_order],
        shot_counts=[int(count) for count in env.selected_shot_counts],
        selected_layer_ids=[int(layer) for layer in env.selected_layer_ids],
        selected_standoff_factors=[float(value) for value in env.selected_standoff_factors],
    )


def plot_selection_3d(
    scenario: Coarse3DScenario,
    result: SelectionResult,
    output: Path,
) -> None:
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    targets = scenario.targets_world
    selected = selected_variant_points_world(
        scenario,
        result.selected_indices,
        selected_layer_ids=result.selected_layer_ids if result.selected_layer_ids else None,
        selected_standoff_factors=result.selected_standoff_factors if result.selected_standoff_factors else None,
    )
    ax.scatter(targets[:, 0], targets[:, 1], targets[:, 2], s=4, c="#94a3b8", alpha=0.22, label="surface targets")
    if len(selected):
        ax.scatter(selected[:, 0], selected[:, 1], selected[:, 2], s=26, c="#dc2626", alpha=0.95, label="selected viewpoints")
        route = selected_variant_points_world(
            scenario,
            result.route_indices,
            selected_layer_ids=result.route_layer_ids if result.route_layer_ids else None,
            selected_standoff_factors=result.route_standoff_factors if result.route_standoff_factors else None,
        ) if result.route_indices else selected
        if len(route) >= 2:
            ax.plot(route[:, 0], route[:, 1], route[:, 2], color="#0f766e", linewidth=2.0, label="view order")
    center = scenario.scene_center_world
    ax.scatter([center[0]], [center[1]], [center[2]], c="#111827", s=60, marker="*", label="base point")
    ax.set_title(result.method)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend(loc="best")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


class SelectionEvalCallback(BaseCallback):
    def __init__(
        self,
        scenario: Coarse3DScenario,
        eval_every_steps: int,
        policy_runner,
    ) -> None:
        super().__init__()
        self.scenario = scenario
        self.eval_every_steps = eval_every_steps
        self.policy_runner = policy_runner
        self.best_score = -math.inf
        self.history: List[TrainingHistoryRow] = []
        self.best_state_dict = None

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_every_steps != 0:
            return True
        result = evaluate_selection(
            self.scenario,
            self.policy_runner(self.scenario, self.model),
            "eval",
        )
        score = selection_score(result)
        self.history.append(
            TrainingHistoryRow(
                step=int(self.num_timesteps),
                mean_coverage=result.certified_coverage,
                mean_weakest_building=result.weakest_building_coverage,
                mean_weakest_region=result.weakest_region_coverage,
                mean_region_gap=result.mean_region_gap,
                mean_weakest_sector=result.weakest_sector_coverage,
                mean_quality=result.average_quality,
                mean_quality_normalized=result.mean_quality_normalized,
                quality_good_fraction=result.quality_good_fraction,
                weakest_quality_normalized=result.weakest_quality_normalized,
                mean_route_length=result.route_length,
                mean_forward_overlap=result.mean_photo_overlap,
                mean_lateral_overlap=result.route_photo_overlap,
                score=score,
            )
        )
        if score > self.best_score:
            self.best_score = score
            self.best_state_dict = copy.deepcopy(self.model.policy.state_dict())
        return True


def train_and_evaluate_multi_seed(
    scenario: Coarse3DScenario,
    total_timesteps: int,
    seeds: Sequence[int],
) -> Tuple[MaskablePPO, SelectionResult, List[SeedRunResult], List[Dict[str, float | int]]]:
    if not seeds:
        raise ValueError("seeds must not be empty")
    ranked: List[SeedRunResult] = []
    best_model: MaskablePPO | None = None
    best_result: SelectionResult | None = None
    best_score = -math.inf
    history_rows: List[Dict[str, float | int]] = []
    for seed in seeds:
        training = PPOTrainingConfig(total_timesteps=total_timesteps, seed=int(seed))
        model, history = train_maskable_ppo(scenario, training)
        for row in history:
            history_rows.append(
                {
                    "seed": int(seed),
                    "step": row.step,
                    "coverage": row.mean_coverage,
                    "weakest_building": row.mean_weakest_building,
                    "weakest_region": row.mean_weakest_region,
                    "region_gap": row.mean_region_gap,
                    "weakest_sector": row.mean_weakest_sector,
                    "quality": row.mean_quality,
                    "quality_norm": row.mean_quality_normalized,
                    "quality_good_fraction": row.quality_good_fraction,
                    "weakest_quality_norm": row.weakest_quality_normalized,
                    "route_length": row.mean_route_length,
                    "forward_overlap": row.mean_forward_overlap,
                    "lateral_overlap": row.mean_lateral_overlap,
                    "score": row.score,
                }
            )
        result = evaluate_selection(
            scenario,
            run_maskable_ppo(scenario, model),
            f"Maskable PPO (seed={seed})",
        )
        score = selection_score(result)
        ranked.append(SeedRunResult(seed=int(seed), result=result))
        if score > best_score:
            best_score = score
            best_model = model
            best_result = SelectionResult(
                method="Maskable PPO best-of-N",
                selected_indices=result.selected_indices,
                route_indices=result.route_indices,
                route_length=result.route_length,
                certified_coverage=result.certified_coverage,
                weakest_building_coverage=result.weakest_building_coverage,
                weakest_region_coverage=result.weakest_region_coverage,
                mean_region_gap=result.mean_region_gap,
                weakest_sector_coverage=result.weakest_sector_coverage,
                selected_views=result.selected_views,
                average_quality=result.average_quality,
                objective=result.objective,
                mean_quality_normalized=result.mean_quality_normalized,
                quality_good_fraction=result.quality_good_fraction,
                weakest_quality_normalized=result.weakest_quality_normalized,
                mean_incidence_quality=result.mean_incidence_quality,
                mean_distance_quality=result.mean_distance_quality,
                mean_visibility_quality=result.mean_visibility_quality,
                mean_photo_overlap=result.mean_photo_overlap,
                route_photo_overlap=result.route_photo_overlap,
                shot_counts=result.shot_counts,
                route_shot_counts=result.route_shot_counts,
                selected_layer_ids=result.selected_layer_ids,
                route_layer_ids=result.route_layer_ids,
                selected_standoff_factors=result.selected_standoff_factors,
                route_standoff_factors=result.route_standoff_factors,
                photo_count=result.photo_count,
            )
    assert best_model is not None and best_result is not None
    return best_model, best_result, ranked, history_rows


def write_outputs(
    scenario: Coarse3DScenario,
    results: Sequence[SelectionResult],
    output_dir: Path,
    seed_runs: Sequence[SeedRunResult] | None = None,
    training_history: Sequence[Dict[str, float | int]] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        rows.append(
            {
                "method": result.method,
                "candidate_views": len(scenario.candidates_world),
                "selected_views": result.selected_views,
                "certified_coverage": result.certified_coverage,
                "weakest_structure_part_coverage": result.weakest_building_coverage,
                "weakest_region_coverage": result.weakest_region_coverage,
                "mean_region_gap": result.mean_region_gap,
                "weakest_sector_coverage": result.weakest_sector_coverage,
                "average_quality": result.average_quality,
                "mean_quality_normalized": result.mean_quality_normalized,
                "quality_good_fraction": result.quality_good_fraction,
                "weakest_quality_normalized": result.weakest_quality_normalized,
                "mean_incidence_quality": result.mean_incidence_quality,
                "mean_distance_quality": result.mean_distance_quality,
                "mean_visibility_quality": result.mean_visibility_quality,
                "mean_photo_overlap": result.mean_photo_overlap,
                "route_photo_overlap": result.route_photo_overlap,
                "photo_count": result.photo_count,
                "shot_counts": ",".join(str(count) for count in result.shot_counts),
                "selected_standoff_factors": ",".join(f"{value:.3f}" for value in result.selected_standoff_factors),
                "route_length_m": result.route_length,
                "route_objective": result.objective,
            }
        )
    with (output_dir / "coarse_3d_viewpoint_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "coarse_3d_viewpoint_results.md").open("w", encoding="utf-8") as stream:
        stream.write("# 粗三维场景：候选视点选择对比\n\n")
        stream.write("| 方法 | 候选点 | 选中视点 | 总照片数 | 认证覆盖率 | 最弱结构段覆盖率 | 最弱区域覆盖率 | 区域缺口 | 最弱扇区覆盖率 | 平均质量 | 平均质量(归一化) | 质量达标占比 | 最弱质量(归一化) | 入射质量 | 距离质量 | 可见性质量 | 前向重叠 | 旁向重叠 | 路径长度 | 距离因子 |\n")
        stream.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            stream.write(
                f"| {row['method']} | {row['candidate_views']} | {row['selected_views']} | {row['photo_count']} | "
                f"{row['certified_coverage']:.1%} | {row['weakest_structure_part_coverage']:.1%} | "
                f"{row['weakest_region_coverage']:.1%} | {row['mean_region_gap']:.1%} | {row['weakest_sector_coverage']:.1%} | "
                f"{row['average_quality']:.3f} | {row['mean_quality_normalized']:.3f} | {row['quality_good_fraction']:.1%} | {row['weakest_quality_normalized']:.3f} | "
                f"{row['mean_incidence_quality']:.3f} | {row['mean_distance_quality']:.3f} | {row['mean_visibility_quality']:.3f} | "
                f"{row['mean_photo_overlap']:.1%} | {row['route_photo_overlap']:.1%} | {row['route_length_m']:.1f} | {row['selected_standoff_factors']} |\n"
            )
        if seed_runs:
            stream.write("\n## PPO Multi-Seed\n\n")
            stream.write("| Seed | 选中视点 | 总照片数 | 认证覆盖率 | 最弱结构段覆盖率 | 最弱区域覆盖率 | 区域缺口 | 最弱扇区覆盖率 | 平均质量 | 平均质量(归一化) | 质量达标占比 | 最弱质量(归一化) | 入射质量 | 距离质量 | 可见性质量 | 前向重叠 | 旁向重叠 | 路径长度 | 距离因子 |\n")
            stream.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
            for item in seed_runs:
                stream.write(
                    f"| {item.seed} | {item.result.selected_views} | {item.result.photo_count} | "
                    f"{item.result.certified_coverage:.1%} | "
                    f"{item.result.weakest_building_coverage:.1%} | "
                    f"{item.result.weakest_region_coverage:.1%} | "
                    f"{item.result.mean_region_gap:.1%} | "
                    f"{item.result.weakest_sector_coverage:.1%} | "
                    f"{item.result.average_quality:.3f} | "
                    f"{item.result.mean_quality_normalized:.3f} | "
                    f"{item.result.quality_good_fraction:.1%} | "
                    f"{item.result.weakest_quality_normalized:.3f} | "
                    f"{item.result.mean_incidence_quality:.3f} | "
                    f"{item.result.mean_distance_quality:.3f} | "
                    f"{item.result.mean_visibility_quality:.3f} | "
                    f"{item.result.mean_photo_overlap:.1%} | "
                    f"{item.result.route_photo_overlap:.1%} | "
                    f"{item.result.route_length:.1f} | {','.join(f'{value:.3f}' for value in item.result.selected_standoff_factors)} |\n"
                )
    for result in results:
        preview_name = result.method.lower().replace("/", "_").replace(" ", "_")
        plot_selection_3d(scenario, result, output_dir / f"{preview_name}_selection_3d.png")
    with (output_dir / "coarse_3d_candidate_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "candidate_views": int(len(scenario.candidates_world)),
                "targets": int(len(scenario.targets_world)),
                "required_views_per_target": int(scenario.required_views),
                "region_counts": {
                    str(code): int(np.count_nonzero(scenario.candidate_region_codes == code))
                    for code in sorted(set(float(x) for x in scenario.candidate_region_codes))
                },
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )
    if seed_runs:
        with (output_dir / "coarse_3d_ppo_seed_sweep.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "seed",
                    "selected_views",
                    "photo_count",
                    "certified_coverage",
                    "weakest_structure_part_coverage",
                    "weakest_region_coverage",
                    "mean_region_gap",
                    "weakest_sector_coverage",
                    "average_quality",
                    "mean_quality_normalized",
                    "quality_good_fraction",
                    "weakest_quality_normalized",
                    "mean_incidence_quality",
                    "mean_distance_quality",
                    "mean_visibility_quality",
                    "route_length_m",
                    "route_objective",
                ],
            )
            writer.writeheader()
            for item in seed_runs:
                writer.writerow(
                    {
                        "seed": item.seed,
                        "selected_views": item.result.selected_views,
                        "photo_count": item.result.photo_count,
                        "certified_coverage": item.result.certified_coverage,
                        "weakest_structure_part_coverage": item.result.weakest_building_coverage,
                        "weakest_region_coverage": item.result.weakest_region_coverage,
                        "mean_region_gap": item.result.mean_region_gap,
                        "weakest_sector_coverage": item.result.weakest_sector_coverage,
                        "average_quality": item.result.average_quality,
                        "mean_quality_normalized": item.result.mean_quality_normalized,
                        "quality_good_fraction": item.result.quality_good_fraction,
                        "weakest_quality_normalized": item.result.weakest_quality_normalized,
                        "mean_incidence_quality": item.result.mean_incidence_quality,
                        "mean_distance_quality": item.result.mean_distance_quality,
                        "mean_visibility_quality": item.result.mean_visibility_quality,
                        "route_length_m": item.result.route_length,
                        "route_objective": item.result.objective,
                    }
                )
    if training_history:
        with (output_dir / "coarse_3d_ppo_training_history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(training_history[0]))
            writer.writeheader()
            writer.writerows(training_history)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D coarse-scene PPO viewpoint-selection prototype.")
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--scene-json", type=Path)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "coarse_3d_viewpoint_rl")
    parser.add_argument("--max-targets", type=int, default=900)
    parser.add_argument(
        "--required-views",
        type=int,
        default=1,
        help="Kept for compatibility; PPO training now uses single-view coverage targets.",
    )
    parser.add_argument("--max-generated-candidates", type=int, default=3600)
    parser.add_argument(
        "--ppo-candidates",
        type=int,
        default=0,
        help="How many candidate viewpoints PPO sees; 0 keeps the full candidate pool.",
    )
    parser.add_argument("--fov", type=float, default=75.0)
    parser.add_argument("--max-incidence", type=float, default=72.0)
    parser.add_argument("--quality-threshold", type=float, default=0.18)
    parser.add_argument("--timesteps", type=int, default=8_000)
    parser.add_argument("--ppo-seed", type=int, default=17)
    parser.add_argument("--ppo-seed-sweep", type=str, default="")
    parser.add_argument("--policy-mode", choices=("continuous", "maskable"), default="continuous")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = build_scenario(
        args.mesh,
        args.scene_json,
        max_targets=args.max_targets,
        required_views=1,
        max_candidates=args.max_generated_candidates,
        fov=args.fov,
        max_incidence=args.max_incidence,
        quality_threshold=args.quality_threshold,
    )
    scenario = compress_candidates(scenario, args.ppo_candidates)
    greedy = evaluate_selection(scenario, greedy_selection(scenario), "Region-aware greedy")
    ilp = evaluate_selection(scenario, scp_ilp_selection(scenario), "SCP/ILP")
    seed_runs: List[SeedRunResult] | None = None
    training_history: List[Dict[str, float | int]] | None = None
    if args.ppo_seed_sweep.strip():
        seeds = [int(part) for part in args.ppo_seed_sweep.split(",") if part.strip()]
        model, ppo, seed_runs, training_history = train_and_evaluate_multi_seed(
            scenario,
            total_timesteps=args.timesteps,
            seeds=seeds,
        )
    else:
        training = PPOTrainingConfig(total_timesteps=args.timesteps, seed=args.ppo_seed)
        if args.policy_mode == "maskable":
            model, history = train_maskable_ppo(scenario, training)
        else:
            model, history = train_continuous_distance_ppo(scenario, training)
        training_history = [
            {
                "seed": training.seed,
                "step": row.step,
                "coverage": row.mean_coverage,
                "weakest_building": row.mean_weakest_building,
                "weakest_region": row.mean_weakest_region,
                "region_gap": row.mean_region_gap,
                "weakest_sector": row.mean_weakest_sector,
                "quality": row.mean_quality,
                "quality_norm": row.mean_quality_normalized,
                "quality_good_fraction": row.quality_good_fraction,
                "weakest_quality_norm": row.weakest_quality_normalized,
                "route_length": row.mean_route_length,
                "forward_overlap": row.mean_forward_overlap,
                "lateral_overlap": row.mean_lateral_overlap,
                "score": row.score,
            }
            for row in history
        ]
        if args.policy_mode == "maskable":
            ppo = evaluate_selection(scenario, run_maskable_ppo(scenario, model), "Maskable PPO + discrete distance")
        else:
            ppo = evaluate_selection(scenario, run_continuous_distance_ppo(scenario, model), "PPO continuous-distance")
    write_outputs(scenario, [greedy, ilp, ppo], args.output_dir, seed_runs=seed_runs, training_history=training_history)
    if args.policy_mode == "maskable":
        model.save(args.output_dir / "coarse_3d_maskable_ppo_discrete_distance")
    else:
        model.save(args.output_dir / "coarse_3d_ppo_continuous_distance")
    for result in (greedy, ilp, ppo):
        print(
            f"{result.method}: selected={result.selected_views}, "
            f"coverage={result.certified_coverage:.1%}, "
            f"weakest={result.weakest_building_coverage:.1%}, "
            f"region={result.weakest_region_coverage:.1%}, "
            f"gap={result.mean_region_gap:.1%}, "
            f"sector={result.weakest_sector_coverage:.1%}, "
            f"path={result.route_length:.1f}"
        )


if __name__ == "__main__":
    main()
