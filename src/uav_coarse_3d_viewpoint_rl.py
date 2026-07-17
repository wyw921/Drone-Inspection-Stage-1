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
    read_triangle_mesh,
    render_plan,
    rotate_to_local,
    route_metrics,
    route_multistart_optimize,
    sample_surface,
    select_multicover,
    targeted_candidates,
    to_local,
    to_world,
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
    eval_every_steps: int = 2_000
    seed: int = 17
    variant_seed_base: int = 101
    variant_seed_count: int = 10
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
    quality_mean_target = PPOTrainingConfig().quality_mean_target
    quality_good_target = PPOTrainingConfig().quality_good_fraction_target
    weakest_quality_target = PPOTrainingConfig().weakest_quality_target
    overlap_balance = 1.0 - abs(result.mean_photo_overlap - FORWARD_OVERLAP_TARGET) - abs(result.route_photo_overlap - LATERAL_OVERLAP_TARGET)
    quality_balance = (
        1.0
        - abs(result.mean_quality_normalized - quality_mean_target)
        - 0.6 * abs(result.quality_good_fraction - quality_good_target)
    )
    coverage_done = result.certified_coverage >= COVERAGE_TARGET - 1e-6
    overlap_done = result.mean_photo_overlap >= FORWARD_OVERLAP_TARGET and result.route_photo_overlap >= LATERAL_OVERLAP_TARGET
    quality_done = (
        result.mean_quality_normalized >= quality_mean_target
        and result.quality_good_fraction >= quality_good_target
        and result.weakest_quality_normalized >= weakest_quality_target
    )
    threshold_bonus = 1.4 if coverage_done and overlap_done and quality_done else 0.0
    threshold_penalty = (
        1.6 * max(FORWARD_OVERLAP_TARGET - result.mean_photo_overlap, 0.0)
        + 1.1 * max(LATERAL_OVERLAP_TARGET - result.route_photo_overlap, 0.0)
        + 0.8 * max(quality_mean_target - result.mean_quality_normalized, 0.0)
        + 0.5 * max(quality_good_target - result.quality_good_fraction, 0.0)
        + 0.5 * max(weakest_quality_target - result.weakest_quality_normalized, 0.0)
    )
    return (
        3.2 * result.certified_coverage
        + 2.8 * forward_term
        + 2.0 * lateral_term
        + 0.45 * result.weakest_building_coverage
        + 0.25 * result.weakest_region_coverage
        + 0.20 * result.weakest_sector_coverage
        + 0.42 * result.mean_quality_normalized
        + 0.22 * result.quality_good_fraction
        + 0.24 * result.weakest_quality_normalized
        + 0.10 * overlap_balance
        + 0.10 * quality_balance
        + threshold_bonus
        - threshold_penalty
        - 0.0005 * result.route_length
    )


def selection_rank_key(result: SelectionResult) -> tuple[float, float, float, float, float, float]:
    quality_mean_target = PPOTrainingConfig().quality_mean_target
    quality_good_target = PPOTrainingConfig().quality_good_fraction_target
    weakest_quality_target = PPOTrainingConfig().weakest_quality_target
    completion = float(
        result.certified_coverage >= COVERAGE_TARGET - 1e-6
        and result.mean_photo_overlap >= FORWARD_OVERLAP_TARGET
        and result.route_photo_overlap >= LATERAL_OVERLAP_TARGET
        and result.mean_quality_normalized >= quality_mean_target
        and result.quality_good_fraction >= quality_good_target
        and result.weakest_quality_normalized >= weakest_quality_target
    )
    overlap_progress = min(
        min(
            result.mean_photo_overlap / max(FORWARD_OVERLAP_TARGET, 1e-6),
            result.route_photo_overlap / max(LATERAL_OVERLAP_TARGET, 1e-6),
        ),
        1.0,
    )
    quality_progress = min(
        min(
            result.mean_quality_normalized / max(quality_mean_target, 1e-6),
            result.quality_good_fraction / max(quality_good_target, 1e-6),
            result.weakest_quality_normalized / max(weakest_quality_target, 1e-6),
        ),
        1.0,
    )
    return (
        completion,
        result.certified_coverage,
        overlap_progress,
        quality_progress,
        selection_score(result),
        -result.route_length,
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
    owner_bonus = np.where(np.asarray(scenario.candidate_owners, dtype=object) == "unassigned", -8.5, 0.28)
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


def rebalance_route_shot_counts(
    scenario: Coarse3DScenario,
    route_indices: Sequence[int],
    route_standoff_factors: Sequence[float] | None = None,
) -> List[int]:
    route_indices = [int(index) for index in route_indices]
    if not route_indices:
        return []
    scores = np.asarray(scenario.candidate_importance, dtype=np.float64)
    route_scores = np.array([float(scores[int(index)]) for index in route_indices], dtype=np.float64)
    if np.allclose(route_scores.max(), route_scores.min()):
        normalized = np.ones_like(route_scores) * 0.5
    else:
        normalized = (route_scores - route_scores.min()) / max(float(route_scores.max() - route_scores.min()), 1e-9)
    if route_standoff_factors is None:
        factors = [1.0 for _ in route_indices]
    else:
        factors = [float(value) for value in route_standoff_factors]
    soft_forward_goal = max(FORWARD_OVERLAP_TARGET, 0.85)
    soft_lateral_goal = max(LATERAL_OVERLAP_TARGET, 0.70)
    counts: List[int] = []
    hard_caps: List[int] = []
    station_priority: List[float] = []
    station_hardness: List[float] = []
    for rank, index in enumerate(route_indices):
        factor = factors[min(rank, len(factors) - 1)] if factors else 1.0
        owner = scenario.candidate_owners[int(index)] if int(index) < len(scenario.candidate_owners) else "unassigned"
        neighbor_need = 0.0
        if rank > 0:
            prev_factor = factors[min(rank - 1, len(factors) - 1)] if factors else 1.0
            prev_overlap = local_pair_lateral_overlap(scenario, route_indices[rank - 1], prev_factor, index, factor)
            neighbor_need = max(neighbor_need, max(LATERAL_OVERLAP_TARGET - prev_overlap, 0.0))
        if rank + 1 < len(route_indices):
            next_factor = factors[min(rank + 1, len(factors) - 1)] if factors else 1.0
            next_overlap = local_pair_lateral_overlap(scenario, index, factor, route_indices[rank + 1], next_factor)
            neighbor_need = max(neighbor_need, max(LATERAL_OVERLAP_TARGET - next_overlap, 0.0))
        importance_level = int(round(2.0 * float(normalized[rank])))
        base = 2 + importance_level
        if owner == "unassigned":
            base = max(base, 3)
        if neighbor_need > 0.08:
            base = max(base, 4)
        if neighbor_need > 0.18:
            base = max(base, 5)
        hard_cap = 9 if owner == "unassigned" or neighbor_need > 0.18 else 7
        counts.append(int(np.clip(base, 1, hard_cap)))
        hard_caps.append(int(hard_cap))
        hardness = float(normalized[rank]) + 1.8 * float(neighbor_need) + (0.35 if owner == "unassigned" else 0.0)
        station_priority.append(hardness)
        station_hardness.append(hardness)

    current_forward, current_lateral = photo_schedule_overlap_metrics(
        scenario,
        route_indices,
        station_shot_counts_override=counts,
        station_standoff_factors_override=factors,
    )
    photo_budget = min(max(len(route_indices) * 5, 240), len(route_indices) * 7)
    max_iterations = 256
    iterations = 0
    while (
        (current_forward < FORWARD_OVERLAP_TARGET - 1e-4 or current_lateral < LATERAL_OVERLAP_TARGET - 1e-4)
        and sum(counts) < photo_budget
        and iterations < max_iterations
    ):
        best_station = None
        best_score = -1e18
        best_metrics = None
        for station_idx in range(len(counts)):
            if counts[station_idx] >= hard_caps[station_idx]:
                continue
            trial_counts = list(counts)
            trial_counts[station_idx] += 1
            trial_forward, trial_lateral = photo_schedule_overlap_metrics(
                scenario,
                route_indices,
                station_shot_counts_override=trial_counts,
                station_standoff_factors_override=factors,
            )
            gain_forward = trial_forward - current_forward
            gain_lateral = trial_lateral - current_lateral
            if gain_forward <= 1e-6 and gain_lateral <= 1e-6:
                continue
            forward_need = max(FORWARD_OVERLAP_TARGET - current_forward, 0.0)
            lateral_need = max(LATERAL_OVERLAP_TARGET - current_lateral, 0.0)
            effective_forward_gain = min(gain_forward, forward_need + gain_forward)
            effective_lateral_gain = min(gain_lateral, lateral_need + gain_lateral)
            owner = scenario.candidate_owners[int(route_indices[station_idx])] if int(route_indices[station_idx]) < len(scenario.candidate_owners) else "unassigned"
            owner_bonus = 0.003 if owner == "unassigned" else 0.0
            marginal_benefit = (
                1.0 * effective_forward_gain
                + 0.35 * effective_lateral_gain
                + owner_bonus
            )
            time_cost = 0.010 + 0.0025 * float(trial_counts[station_idx] - 1)
            efficiency = marginal_benefit / max(time_cost, 1e-6)
            score = (
                20.0 * effective_forward_gain
                + 7.0 * effective_lateral_gain
                + 0.8 * efficiency
                - 0.018 * float(trial_counts[station_idx])
            )
            # Only accept extra photos whose marginal gain is clearly worth the extra capture time.
            if current_forward < FORWARD_OVERLAP_TARGET - 1e-4:
                if effective_forward_gain < 0.0010 and efficiency < 0.24:
                    continue
            else:
                if effective_lateral_gain < 0.0009 and efficiency < 0.20:
                    continue
            if score > best_score:
                best_score = score
                best_station = station_idx
                best_metrics = (trial_forward, trial_lateral)
        if best_station is None or best_metrics is None or best_score <= 0.0:
            break
        counts[best_station] += 1
        current_forward, current_lateral = best_metrics
        iterations += 1

    # If we are just under the forward target, allow only the highest-value few stations
    # to stretch into 7/8 shots before ever considering 9.
    if current_forward < FORWARD_OVERLAP_TARGET - 1e-4:
        ranked = np.argsort(-np.asarray(station_priority, dtype=np.float64))
        elite_count = min(12, max(5, len(route_indices) // 8))
        for rank_idx in ranked[:elite_count].tolist():
            hard_caps[rank_idx] = max(hard_caps[rank_idx], 9)
        extra_iterations = 0
        while (
            current_forward < FORWARD_OVERLAP_TARGET - 1e-4
            and sum(counts) < min(photo_budget + 54, len(route_indices) * 8)
            and extra_iterations < 180
        ):
            best_station = None
            best_score = -1e18
            best_metrics = None
            for station_idx in ranked[:elite_count].tolist():
                if counts[station_idx] >= hard_caps[station_idx]:
                    continue
                trial_counts = list(counts)
                trial_counts[station_idx] += 1
                trial_forward, trial_lateral = photo_schedule_overlap_metrics(
                    scenario,
                    route_indices,
                    station_shot_counts_override=trial_counts,
                    station_standoff_factors_override=factors,
                )
                gain_forward = trial_forward - current_forward
                gain_lateral = trial_lateral - current_lateral
                if gain_forward <= 1e-6 and gain_lateral <= 1e-6:
                    continue
                efficiency = gain_forward / max(0.010 + 0.003 * float(trial_counts[station_idx] - 1), 1e-6)
                if gain_forward < 0.00035 and efficiency < 0.09:
                    continue
                score = 44.0 * gain_forward + 5.5 * gain_lateral + 1.1 * efficiency - 0.008 * float(trial_counts[station_idx])
                if score > best_score:
                    best_score = score
                    best_station = station_idx
                    best_metrics = (trial_forward, trial_lateral)
            if best_station is None or best_metrics is None or best_score <= 0.0:
                break
            counts[best_station] += 1
            current_forward, current_lateral = best_metrics
            extra_iterations += 1

    # After clearing the hard minimum, spend remaining photo budget on a softer
    # overlap target so reconstruction does not stop at the bare threshold.
    ranked = np.argsort(-np.asarray(station_priority, dtype=np.float64))
    comfort_count = min(max(8, len(route_indices) // 3), max(10, len(route_indices) // 2))
    comfort_station_ids = ranked[:comfort_count].tolist()
    comfort_budget = min(photo_budget + 48, len(route_indices) * 8)
    comfort_iterations = 0
    while (
        (current_forward < soft_forward_goal - 1e-4 or current_lateral < soft_lateral_goal - 1e-4)
        and sum(counts) < comfort_budget
        and comfort_iterations < 140
    ):
        best_station = None
        best_score = -1e18
        best_metrics = None
        for station_idx in comfort_station_ids:
            if counts[station_idx] >= 9:
                continue
            if counts[station_idx] >= 6 and station_hardness[station_idx] < 0.78:
                continue
            if counts[station_idx] >= 7 and station_hardness[station_idx] < 1.02:
                continue
            trial_counts = list(counts)
            trial_counts[station_idx] += 1
            trial_forward, trial_lateral = photo_schedule_overlap_metrics(
                scenario,
                route_indices,
                station_shot_counts_override=trial_counts,
                station_standoff_factors_override=factors,
            )
            gain_forward = trial_forward - current_forward
            gain_lateral = trial_lateral - current_lateral
            if gain_forward <= 1e-6 and gain_lateral <= 1e-6:
                continue
            forward_need = max(soft_forward_goal - current_forward, 0.0)
            lateral_need = max(soft_lateral_goal - current_lateral, 0.0)
            effective_forward_gain = min(gain_forward, forward_need + gain_forward)
            effective_lateral_gain = min(gain_lateral, lateral_need + gain_lateral)
            efficiency = (
                1.15 * effective_forward_gain + 0.70 * effective_lateral_gain
            ) / max(0.011 + 0.0032 * float(trial_counts[station_idx] - 1), 1e-6)
            if effective_forward_gain < 0.00045 and effective_lateral_gain < 0.00040 and efficiency < 0.075:
                continue
            score = (
                24.0 * effective_forward_gain
                + 16.0 * effective_lateral_gain
                + 1.25 * efficiency
                + 0.18 * station_hardness[station_idx]
                - 0.010 * float(trial_counts[station_idx])
            )
            if score > best_score:
                best_score = score
                best_station = station_idx
                best_metrics = (trial_forward, trial_lateral)
        if best_station is None or best_metrics is None or best_score <= 0.0:
            break
        counts[best_station] += 1
        current_forward, current_lateral = best_metrics
        comfort_iterations += 1
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


def primary_building_group_name(name: str) -> str:
    canonical = canonical_building_name(name)
    if canonical == "Anzhong Building":
        return canonical
    if canonical.startswith("安中A座") or canonical.startswith("安中B座") or canonical.startswith("安中A-B连接体") or canonical.startswith("安中连接体"):
        return "Anzhong Building"
    if canonical.startswith("北教学楼群西楼"):
        return "North Teaching West"
    if canonical.startswith("北教学楼群中楼") or canonical.startswith("北教学楼群二层架空连廊") or canonical.startswith("北教学楼群入口挑空门厅"):
        return "North Teaching Middle"
    if canonical.startswith("北教学楼群东楼"):
        return "North Teaching East"
    return canonical


def active_primary_building_groups(
    target_buildings: Sequence[str],
    min_targets: int = 24,
    min_fraction: float = 0.06,
) -> List[str]:
    if not target_buildings:
        return []
    grouped = [primary_building_group_name(name) for name in target_buildings]
    counts = collections.Counter(grouped)
    total = max(len(grouped), 1)
    active = [
        name
        for name, count in counts.items()
        if count >= min_targets or (count / total) >= min_fraction
    ]
    if active:
        return sorted(active)
    return sorted(counts)


def inferred_scene_mode(target_buildings: Sequence[str]) -> str:
    active_groups = active_primary_building_groups(target_buildings)
    return "campus" if len(active_groups) >= 3 else "single"


def building_aware_clustered_missing_indices(
    targets_local: np.ndarray,
    target_buildings: Sequence[str],
    missing_indices: np.ndarray,
    scene_span: float,
    per_group_clusters: int,
    max_total_clusters: int,
) -> np.ndarray:
    if len(missing_indices) == 0:
        return np.zeros((0,), dtype=int)
    grouped_missing: Dict[str, List[int]] = collections.defaultdict(list)
    for index in missing_indices.tolist():
        if 0 <= int(index) < len(target_buildings):
            grouped_missing[primary_building_group_name(target_buildings[int(index)])].append(int(index))
    prioritized_groups = sorted(grouped_missing.items(), key=lambda item: (-len(item[1]), item[0]))
    selected: List[int] = []
    for _group_name, indices in prioritized_groups:
        clustered = clustered_difficult_missing_indices(
            targets_local,
            np.asarray(indices, dtype=int),
            scene_span,
            max_clusters=per_group_clusters,
        )
        for index in clustered.tolist():
            if int(index) not in selected:
                selected.append(int(index))
            if len(selected) >= max_total_clusters:
                return np.asarray(selected, dtype=int)
    return np.asarray(selected, dtype=int)


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


def augment_selection_with_cluster_rescue(
    scenario: Coarse3DScenario,
    selected_indices: Sequence[int],
    selected_shot_counts: Sequence[int],
    selected_layer_ids: Sequence[int] | None,
    selected_standoff_factors: Sequence[float],
    max_added: int = 2,
) -> tuple[List[int], List[int], List[int], List[float]]:
    scene_mode = inferred_scene_mode(scenario.target_buildings)
    selected_indices = [int(idx) for idx in selected_indices]
    shot_counts = [int(count) for count in selected_shot_counts]
    layer_ids = [int(layer) for layer in (selected_layer_ids or [legacy_standoff_layer_index(scenario, factor) for factor in selected_standoff_factors])]
    standoff_factors = [float(value) for value in selected_standoff_factors]
    if not selected_indices or max_added <= 0:
        return selected_indices, shot_counts, layer_ids, standoff_factors

    remaining = np.full(len(scenario.targets_world), scenario.required_views, dtype=np.int16)
    for index, factor in zip(selected_indices, standoff_factors):
        coverage, _quality = candidate_variant_metrics(scenario, int(index), float(factor))
        remaining = np.maximum(0, remaining - coverage.astype(np.int16))
    if not np.any(remaining > 0):
        return selected_indices, shot_counts, layer_ids, standoff_factors

    scene_span = float(np.linalg.norm(scenario.scene_high_world - scenario.scene_low_world))
    hard_missing = difficult_transition_missing_indices(
        scenario.targets_local,
        scenario.normals_local,
        scenario.target_buildings,
        classify_surface_regions(scenario.targets_local, scenario.normals_local, np.ones(len(scenario.targets_local), dtype=np.float32)),
        np.flatnonzero(remaining > 0),
        limit=72 if scene_mode == "campus" else 40,
    )
    if scene_mode == "campus":
        effective_max_added = max(max_added, 7)
        cluster_indices = building_aware_clustered_missing_indices(
            scenario.targets_local,
            scenario.target_buildings,
            hard_missing,
            scene_span,
            per_group_clusters=2,
            max_total_clusters=min(effective_max_added * 2, 14),
        )
    else:
        effective_max_added = max_added
        cluster_indices = clustered_difficult_missing_indices(
            scenario.targets_local,
            hard_missing,
            scene_span,
            max_clusters=min(max_added, 8),
        )
    if len(cluster_indices) == 0:
        return selected_indices, shot_counts, layer_ids, standoff_factors

    chosen = list(selected_indices)
    chosen_positions = selected_variant_points_world(
        scenario,
        chosen,
        selected_layer_ids=layer_ids,
        selected_standoff_factors=standoff_factors,
    )
    min_xy_spacing = max(6.0, 0.55 * float(scenario.stand_off))
    default_factor = 1.0
    default_layer = legacy_standoff_layer_index(scenario, default_factor)
    local_radius_xy = max(0.10 * scene_span, 7.0)
    local_radius_z = max(0.14 * scene_span, 5.0)
    route_anchor = chosen_positions if len(chosen_positions) else np.zeros((0, 3), dtype=np.float64)

    added_any = False
    for cluster_target_idx in cluster_indices.tolist():
        if len(chosen) >= len(selected_indices) + effective_max_added:
            break
        target_point = scenario.targets_world[int(cluster_target_idx)]
        local_missing = np.flatnonzero(
            (remaining > 0)
            & (np.linalg.norm(scenario.targets_world[:, :2] - target_point[None, :2], axis=1) <= local_radius_xy)
            & (np.abs(scenario.targets_world[:, 2] - target_point[2]) <= local_radius_z)
        )
        if len(local_missing) == 0:
            local_missing = np.asarray([int(cluster_target_idx)], dtype=int)

        best_idx = None
        best_score = -1e18
        best_total_cover_gain = 0.0
        best_path_cost = 0.0
        for idx in range(len(scenario.candidates_world)):
            if idx in chosen:
                continue
            owner_name = scenario.candidate_owners[idx] if idx < len(scenario.candidate_owners) else "unassigned"
            factor = default_factor
            coverage, quality = candidate_variant_metrics(scenario, idx, factor)
            local_cover = coverage[local_missing].astype(bool)
            if not np.any(local_cover):
                continue
            total_cover = coverage & (remaining > 0)
            total_cover_gain = float(np.count_nonzero(total_cover))
            if total_cover_gain < 2.0:
                continue
            point_world = standoff_position_world(scenario, idx, factor)
            if len(chosen_positions):
                xy_dist = np.linalg.norm(chosen_positions[:, :2] - point_world[None, :2], axis=1)
                z_dist = np.abs(chosen_positions[:, 2] - point_world[2])
                if np.any((xy_dist < 0.82 * min_xy_spacing) & (z_dist < max(4.0, 0.25 * float(scenario.stand_off)))):
                    continue
            local_cover_gain = float(np.count_nonzero(local_cover))
            local_quality_gain = float(np.sum(quality[local_missing]))
            distance_to_cluster = float(np.linalg.norm(point_world[:2] - target_point[:2]))
            if len(route_anchor):
                anchor_dist = np.linalg.norm(route_anchor[:, :2] - point_world[None, :2], axis=1)
                extra_path_cost = float(np.min(anchor_dist))
            else:
                extra_path_cost = distance_to_cluster
            owner_bonus = 0.35 if owner_name == "unassigned" else 0.15
            gain_score = (
                8.0 * local_cover_gain
                + 3.0 * total_cover_gain
                + 1.6 * local_quality_gain
                + owner_bonus
            )
            detour_penalty = 0.24 * extra_path_cost + 0.06 * distance_to_cluster
            path_cost = max(extra_path_cost + 0.35 * distance_to_cluster, 1e-6)
            efficiency = gain_score / path_cost
            # Prefer fixing overlap with more shots on existing stations first.
            # A rescue point must therefore bring clearly meaningful coverage
            # before we accept a long spatial detour.
            if extra_path_cost > max(34.0, 0.95 * float(scenario.stand_off)) and total_cover_gain < (5.0 if scene_mode == "campus" else 6.0):
                continue
            if extra_path_cost > max(52.0 if scene_mode == "campus" else 46.0, (1.35 if scene_mode == "campus" else 1.20) * float(scenario.stand_off)) and total_cover_gain < (8.0 if scene_mode == "campus" else 9.0):
                continue
            if distance_to_cluster > max(30.0 if scene_mode == "campus" else 26.0, (0.82 if scene_mode == "campus" else 0.75) * float(scenario.stand_off)) and local_cover_gain < 3.0:
                continue
            if total_cover_gain < 3.0 and efficiency < (0.26 if scene_mode == "campus" else 0.30):
                continue
            if path_cost > max(48.0 if scene_mode == "campus" else 42.0, (1.25 if scene_mode == "campus" else 1.15) * float(scenario.stand_off)) and efficiency < (0.32 if scene_mode == "campus" else 0.38):
                continue
            score = gain_score - detour_penalty + 3.0 * min(efficiency, 2.5)
            if score > best_score:
                best_score = score
                best_idx = idx
                best_total_cover_gain = total_cover_gain
                best_path_cost = path_cost
        if best_idx is None:
            continue
        # Final accept: the coverage gain must still be worth the path extension.
        if best_total_cover_gain / max(best_path_cost, 1e-6) < (0.10 if scene_mode == "campus" else 0.12) and best_total_cover_gain < (4.0 if scene_mode == "campus" else 5.0):
            continue
        if best_path_cost > max(58.0 if scene_mode == "campus" else 52.0, (1.48 if scene_mode == "campus" else 1.35) * float(scenario.stand_off)) and best_total_cover_gain < (7.0 if scene_mode == "campus" else 8.0):
            continue
        if best_score < (9.5 if scene_mode == "campus" else 11.0):
            continue
        chosen.append(int(best_idx))
        shot_counts.append(3)
        layer_ids.append(int(default_layer))
        standoff_factors.append(float(default_factor))
        added_any = True
        added_point = standoff_position_world(scenario, int(best_idx), float(default_factor))
        chosen_positions = np.vstack([chosen_positions, added_point[None, :]])
        route_anchor = chosen_positions
        added_coverage, _added_quality = candidate_variant_metrics(scenario, int(best_idx), float(default_factor))
        remaining = np.maximum(0, remaining - added_coverage.astype(np.int16))
        if not np.any(remaining > 0):
            break
    if not added_any:
        return selected_indices, shot_counts[: len(selected_indices)], layer_ids[: len(selected_indices)], standoff_factors[: len(selected_indices)]
    return chosen, shot_counts, layer_ids, standoff_factors


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


def owner_aware_rescue_scores(
    supplemental_coverage: np.ndarray,
    supplemental_quality: np.ndarray,
    missing_indices: np.ndarray,
    target_buildings: Sequence[str],
    supplemental_owners: Sequence[str],
) -> np.ndarray:
    if len(supplemental_coverage) == 0 or len(missing_indices) == 0:
        return np.zeros(len(supplemental_coverage), dtype=np.float32)
    missing_labels = np.asarray([target_buildings[int(idx)] for idx in missing_indices], dtype=object)
    missing_owner_counts = collections.Counter(str(name) for name in missing_labels.tolist())
    max_owner_need = max((count for name, count in missing_owner_counts.items() if name != "unassigned"), default=0)
    scores = np.zeros(len(supplemental_coverage), dtype=np.float32)
    for row_idx in range(len(supplemental_coverage)):
        covered_missing = supplemental_coverage[row_idx, missing_indices].astype(bool)
        quality_missing = supplemental_quality[row_idx, missing_indices].astype(np.float32)
        cover_gain = float(np.count_nonzero(covered_missing))
        quality_gain = float(np.sum(quality_missing))
        if cover_gain <= 0.0:
            scores[row_idx] = -5.0
            continue
        hit_labels = missing_labels[covered_missing]
        hit_counts = collections.Counter(str(name) for name in hit_labels.tolist())
        assigned_hit_count = sum(count for name, count in hit_counts.items() if name != "unassigned")
        unassigned_hit_count = int(hit_counts.get("unassigned", 0))
        owner_name = str(supplemental_owners[row_idx]) if row_idx < len(supplemental_owners) else "unassigned"
        owner_need_bonus = 0.0
        if owner_name != "unassigned" and max_owner_need > 0:
            owner_need_bonus = 1.4 * float(missing_owner_counts.get(owner_name, 0)) / float(max_owner_need)
        owner_match_hits = float(hit_counts.get(owner_name, 0)) if owner_name != "unassigned" else 0.0
        unique_assigned_hits = len([name for name in hit_counts if name != "unassigned"])
        unassigned_owner_penalty = 1.6 if owner_name == "unassigned" else 0.0
        scores[row_idx] = float(
            1.8 * cover_gain
            + 2.0 * quality_gain
            + 0.8 * assigned_hit_count
            + 0.6 * owner_match_hits
            + 0.25 * unique_assigned_hits
            + owner_need_bonus
            - 0.35 * unassigned_hit_count
            - unassigned_owner_penalty
        )
    return scores


def difficult_transition_missing_indices(
    targets_local: np.ndarray,
    normals_local: np.ndarray,
    target_buildings: Sequence[str],
    regions: Sequence[str],
    missing_indices: np.ndarray,
    limit: int = 220,
) -> np.ndarray:
    if len(missing_indices) == 0:
        return np.zeros(0, dtype=int)
    missing_indices = np.asarray(missing_indices, dtype=int)
    points = np.asarray(targets_local, dtype=np.float64)
    labels = np.asarray([canonical_building_name(name) for name in target_buildings], dtype=object)
    scores: List[tuple[float, int]] = []
    if len(points) > 1:
        distances = np.linalg.norm(points[missing_indices, None, :] - points[None, :, :], axis=2)
    else:
        distances = np.zeros((len(missing_indices), len(points)), dtype=np.float64)
    for row_idx, target_idx in enumerate(missing_indices.tolist()):
        label = str(labels[target_idx])
        region = str(regions[target_idx]) if target_idx < len(regions) else "wall"
        score = 0.0
        if label == "unassigned":
            score += 3.0
        if region in {"corner_transition", "occlusion_sensitive"}:
            score += 2.4
        elif region == "sloped_transition":
            score += 1.2
        normal = normals_local[target_idx]
        score += 0.45 * abs(float(normal[2]))
        if len(points) > 1:
            row = distances[row_idx]
            row[target_idx] = np.inf
            k = min(12, len(points) - 1)
            neighbor_ids = np.argpartition(row, k)[:k] if k > 0 else np.zeros(0, dtype=int)
            neighbor_labels = {str(labels[idx]) for idx in neighbor_ids.tolist() if str(labels[idx]) != label}
            assigned_neighbors = {name for name in neighbor_labels if name != "unassigned"}
            if assigned_neighbors:
                score += 0.9 + 0.55 * len(assigned_neighbors)
            if label == "unassigned":
                near_assigned = sum(1 for idx in neighbor_ids.tolist() if str(labels[idx]) != "unassigned")
                score += 0.08 * near_assigned
        scores.append((score, int(target_idx)))
    scores.sort(key=lambda item: (-item[0], item[1]))
    return np.asarray([idx for _score, idx in scores[:limit]], dtype=int)


def clustered_difficult_missing_indices(
    targets_local: np.ndarray,
    prioritized_missing_indices: np.ndarray,
    scene_span: float,
    max_clusters: int = 12,
) -> np.ndarray:
    prioritized_missing_indices = np.asarray(prioritized_missing_indices, dtype=int)
    if len(prioritized_missing_indices) == 0:
        return np.zeros(0, dtype=int)
    xy_radius = max(0.08 * scene_span, 3.0)
    z_radius = max(0.12 * scene_span, 4.0)
    chosen: List[int] = []
    for idx in prioritized_missing_indices.tolist():
        point = targets_local[int(idx)]
        too_close = False
        for chosen_idx in chosen:
            chosen_point = targets_local[int(chosen_idx)]
            if (
                float(np.linalg.norm(point[:2] - chosen_point[:2])) <= xy_radius
                and abs(float(point[2] - chosen_point[2])) <= z_radius
            ):
                too_close = True
                break
        if too_close:
            continue
        chosen.append(int(idx))
        if len(chosen) >= max_clusters:
            break
    return np.asarray(chosen, dtype=int)


def focused_transition_cluster_candidates(
    targets: np.ndarray,
    normals: np.ndarray,
    cluster_indices: np.ndarray,
    target_buildings: Sequence[str],
    stand_off: float,
    max_standoff: float,
) -> Tuple[np.ndarray, np.ndarray]:
    positions: List[np.ndarray] = []
    aims: List[np.ndarray] = []
    if len(cluster_indices) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    focus_distances = sorted(
        {
            max_standoff,
            min(max_standoff * 1.14, max_standoff + 0.24 * stand_off),
        }
    )
    for index in cluster_indices.tolist():
        target = np.asarray(targets[int(index)], dtype=np.float64)
        normal = np.asarray(normals[int(index)], dtype=np.float64)
        label = canonical_building_name(str(target_buildings[int(index)])) if int(index) < len(target_buildings) else "unassigned"
        tangent = np.cross(normal, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        if np.linalg.norm(tangent) < 1e-6:
            tangent = np.cross(normal, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        tangent = tangent / max(np.linalg.norm(tangent), 1e-12)
        bitangent = np.cross(normal, tangent)
        bitangent = bitangent / max(np.linalg.norm(bitangent), 1e-12)
        if normal[2] > 0.7:
            for distance in focus_distances:
                for angle in np.linspace(0, 2 * math.pi, 8, endpoint=False):
                    radial = math.cos(angle) * tangent + math.sin(angle) * bitangent
                    positions.append(target + normal * distance + radial * distance * 0.32)
                    aims.append(target)
            continue
        lateral_offsets = (-0.45, 0.0, 0.45)
        vertical_offsets = (-0.18, 0.0, 0.18)
        for distance in focus_distances:
            for lateral in lateral_offsets:
                for vertical in vertical_offsets:
                    positions.append(
                        target
                        + normal * distance
                        + tangent * distance * lateral
                        + bitangent * distance * vertical * 0.34
                    )
                    aims.append(target)
        if label == "unassigned":
            for angle in np.linspace(0, 2 * math.pi, 6, endpoint=False):
                radial = math.cos(angle) * tangent + math.sin(angle) * bitangent
                positions.append(target + normal * focus_distances[-1] + radial * 0.36 * focus_distances[-1])
                aims.append(target)
    return np.asarray(positions, dtype=np.float64), np.asarray(aims, dtype=np.float64)


def post_rescue_outer_shell_candidates(
    targets: np.ndarray,
    normals: np.ndarray,
    focus_indices: np.ndarray,
    target_buildings: Sequence[str],
    stand_off: float,
    max_standoff: float,
) -> Tuple[np.ndarray, np.ndarray]:
    positions: List[np.ndarray] = []
    aims: List[np.ndarray] = []
    if len(focus_indices) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    outer_distances = sorted(
        {
            min(max_standoff * 1.10, max_standoff + 0.28 * stand_off),
            min(max_standoff * 1.22, max_standoff + 0.45 * stand_off),
            min(max_standoff * 1.34, max_standoff + 0.62 * stand_off),
        }
    )
    for index in focus_indices.tolist():
        target = np.asarray(targets[int(index)], dtype=np.float64)
        normal = np.asarray(normals[int(index)], dtype=np.float64)
        label = canonical_building_name(str(target_buildings[int(index)])) if int(index) < len(target_buildings) else "unassigned"
        tangent = np.cross(normal, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        if np.linalg.norm(tangent) < 1e-6:
            tangent = np.cross(normal, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        tangent = tangent / max(np.linalg.norm(tangent), 1e-12)
        bitangent = np.cross(normal, tangent)
        bitangent = bitangent / max(np.linalg.norm(bitangent), 1e-12)
        if normal[2] > 0.7:
            for distance in outer_distances:
                for angle in np.linspace(0, 2 * math.pi, 10, endpoint=False):
                    radial = math.cos(angle) * tangent + math.sin(angle) * bitangent
                    positions.append(target + normal * distance + radial * distance * 0.40)
                    aims.append(target)
            continue
        lateral_offsets = (-0.75, -0.38, 0.0, 0.38, 0.75)
        vertical_offsets = (-0.30, -0.12, 0.12, 0.30)
        for distance in outer_distances:
            for lateral in lateral_offsets:
                for vertical in vertical_offsets:
                    positions.append(
                        target
                        + normal * distance
                        + tangent * distance * lateral
                        + bitangent * distance * vertical * 0.48
                    )
                    aims.append(target)
        if label == "unassigned":
            for distance in outer_distances[-2:]:
                for angle in np.linspace(0, 2 * math.pi, 8, endpoint=False):
                    radial = math.cos(angle) * tangent + math.sin(angle) * bitangent
                    positions.append(target + normal * distance + radial * 0.48 * distance)
                    aims.append(target)
    return np.asarray(positions, dtype=np.float64), np.asarray(aims, dtype=np.float64)


def append_candidates_to_scenario(
    scenario: Coarse3DScenario,
    candidates_local: np.ndarray,
    aims_local: np.ndarray,
    coverage: np.ndarray,
    quality: np.ndarray,
    owners: Sequence[str],
    cluster_id: float,
    region_code: float,
    importance: float,
) -> Coarse3DScenario:
    if len(candidates_local) == 0:
        return scenario
    candidates_world = to_world(candidates_local, scenario.scene_center_world, scenario.scene_basis)
    aims_world = to_world(aims_local, scenario.scene_center_world, scenario.scene_basis)
    extra_count = len(candidates_local)
    return Coarse3DScenario(
        targets_local=scenario.targets_local,
        normals_local=scenario.normals_local,
        targets_world=scenario.targets_world,
        target_buildings=scenario.target_buildings,
        candidates_local=np.concatenate([scenario.candidates_local, candidates_local], axis=0),
        candidates_world=np.concatenate([scenario.candidates_world, candidates_world], axis=0),
        aims_world=np.concatenate([scenario.aims_world, aims_world], axis=0),
        coverage=np.concatenate([scenario.coverage, coverage], axis=0),
        quality=np.concatenate([scenario.quality, quality], axis=0),
        candidate_owners=list(scenario.candidate_owners) + [str(owner) for owner in owners],
        candidate_region_codes=np.concatenate([scenario.candidate_region_codes, np.full(extra_count, region_code, dtype=np.float32)], axis=0),
        candidate_cluster_ids=np.concatenate([scenario.candidate_cluster_ids, np.full(extra_count, cluster_id, dtype=np.float32)], axis=0),
        candidate_region_ids=np.concatenate([scenario.candidate_region_ids, np.full(extra_count, int(max(scenario.candidate_region_ids.max(initial=0), 0) + 1), dtype=np.int32)], axis=0),
        candidate_sector_ids=np.concatenate([scenario.candidate_sector_ids, candidate_sector_ids_from_world(candidates_world, scenario.scene_center_world, bins=8)], axis=0),
        target_region_ids=scenario.target_region_ids,
        target_sector_ids=scenario.target_sector_ids,
        candidate_importance=np.concatenate([scenario.candidate_importance, np.full(extra_count, importance, dtype=np.float32)], axis=0),
        stand_off=scenario.stand_off,
        required_views=scenario.required_views,
        min_standoff=scenario.min_standoff,
        max_standoff=scenario.max_standoff,
        standoff_factor_min=scenario.standoff_factor_min,
        standoff_factor_max=scenario.standoff_factor_max,
        scene_center_world=scenario.scene_center_world,
        scene_basis=scenario.scene_basis,
        scene_low_world=scenario.scene_low_world,
        scene_high_world=scenario.scene_high_world,
    )


def augment_scenario_with_post_rescue_candidates(
    scenario: Coarse3DScenario,
    selected_data: SelectionPolicyOutput,
    max_clusters: int = 2,
    keep_candidates: int = 18,
) -> Coarse3DScenario:
    scene_mode = inferred_scene_mode(scenario.target_buildings)
    selected_indices = [int(idx) for idx in selected_data.selected_indices]
    selected_shot_counts = [int(count) for count in selected_data.shot_counts]
    selected_layer_ids = [int(layer) for layer in selected_data.selected_layer_ids]
    standoff_factors = [float(value) for value in selected_data.selected_standoff_factors]
    if not selected_indices:
        return scenario
    # Rescue is only for real coverage holes. If the plan is already coverage-safe,
    # any remaining overlap issue must be fixed by adding shots, not by injecting
    # more stations into the candidate pool.
    pre_result = _selection_result_from_explicit_plan(
        scenario,
        selected_indices,
        selected_shot_counts,
        selected_layer_ids,
        standoff_factors,
        method="post_rescue_probe",
        route_restarts=12,
        route_proposals=320,
    )
    if pre_result.route_indices:
        route_shots = rebalance_route_shot_counts(
            scenario,
            pre_result.route_indices,
            pre_result.route_standoff_factors,
        )
        route_lookup = {
            int(idx): int(count)
            for idx, count in zip(pre_result.route_indices, route_shots)
        }
        selected_shot_counts = [
            route_lookup.get(int(idx), int(count))
            for idx, count in zip(selected_indices, selected_shot_counts)
        ]
        pre_result = _selection_result_from_explicit_plan(
            scenario,
            selected_indices,
            selected_shot_counts,
            selected_layer_ids,
            standoff_factors,
            method="post_rescue_probe",
            route_restarts=12,
            route_proposals=320,
        )
    if (
        pre_result.certified_coverage >= 0.995
        and pre_result.weakest_region_coverage >= 0.90
        and pre_result.weakest_sector_coverage >= 0.90
    ):
        return scenario
    remaining = np.full(len(scenario.targets_world), scenario.required_views, dtype=np.int16)
    for index, factor in zip(selected_indices, standoff_factors):
        coverage, _quality = candidate_variant_metrics(scenario, int(index), float(factor))
        remaining = np.maximum(0, remaining - coverage.astype(np.int16))
    if not np.any(remaining > 0):
        return scenario

    missing_indices = np.flatnonzero(remaining > 0)
    scene_span = float(np.linalg.norm(scenario.scene_high_world - scenario.scene_low_world))
    regions = classify_surface_regions(
        scenario.targets_local,
        scenario.normals_local,
        np.ones(len(scenario.targets_local), dtype=np.float32),
    )
    hard_missing = difficult_transition_missing_indices(
        scenario.targets_local,
        scenario.normals_local,
        scenario.target_buildings,
        regions,
        missing_indices,
        limit=84 if scene_mode == "campus" else 48,
    )
    if scene_mode == "campus":
        max_clusters = max(max_clusters, 4)
        keep_candidates = max(keep_candidates, 32)
        cluster_indices = building_aware_clustered_missing_indices(
            scenario.targets_local,
            scenario.target_buildings,
            hard_missing,
            scene_span,
            per_group_clusters=2,
            max_total_clusters=12,
        )
    else:
        cluster_indices = clustered_difficult_missing_indices(
            scenario.targets_local,
            hard_missing,
            scene_span,
            max_clusters=max_clusters,
        )
    if len(cluster_indices) == 0:
        return scenario

    transition_local, transition_aims = focused_transition_cluster_candidates(
        scenario.targets_local,
        scenario.normals_local,
        cluster_indices,
        scenario.target_buildings,
        scenario.stand_off,
        scenario.max_standoff,
    )
    outer_local, outer_aims = post_rescue_outer_shell_candidates(
        scenario.targets_local,
        scenario.normals_local,
        hard_missing[: min(len(hard_missing), 18 if scene_mode == "campus" else 10)],
        scenario.target_buildings,
        scenario.stand_off,
        scenario.max_standoff,
    )
    targeted_local, targeted_aims = targeted_candidates(
        scenario.targets_local,
        scenario.normals_local,
        hard_missing[: min(len(hard_missing), 30 if scene_mode == "campus" else 18)],
        scenario.stand_off,
        scenario.max_standoff,
    )
    local_parts = [part for part in [transition_local, outer_local, targeted_local] if len(part)]
    aim_parts = [part for part in [transition_aims, outer_aims, targeted_aims] if len(part)]
    combined_local = np.concatenate(local_parts, axis=0) if local_parts else np.zeros((0, 3), dtype=np.float64)
    combined_aims = np.concatenate(aim_parts, axis=0) if aim_parts else np.zeros((0, 3), dtype=np.float64)
    if len(combined_local) == 0:
        return scenario
    combined_local, combined_aims = filter_candidate_constraints(
        combined_local,
        combined_aims,
        scenario.targets_local,
        scenario.scene_center_world,
        scenario.scene_basis,
        max(0.78 * scenario.min_standoff, 0.5 * scenario.stand_off),
        min(1.22 * scenario.max_standoff, scenario.max_standoff + 0.45 * scenario.stand_off),
        1.0,
        150.0,
    )
    if len(combined_local) == 0:
        return scenario
    region_weights = np.array(
        [
            1.20 if region == "corner_transition" else 1.15 if region == "occlusion_sensitive" else 1.05 if region == "roof" else 1.0
            for region in regions
        ],
        dtype=np.float32,
    )
    coverage, quality = coverage_matrix(
        combined_local,
        combined_aims,
        scenario.targets_local,
        scenario.normals_local,
        scenario.stand_off,
        75.0,
        72.0,
        0.18,
        scenario.min_standoff,
        scenario.max_standoff,
        region_weights,
    )
    selected_points = selected_variant_points_world(
        scenario,
        selected_indices,
        selected_layer_ids=list(selected_data.selected_layer_ids),
        selected_standoff_factors=standoff_factors,
    )
    candidate_world = to_world(combined_local, scenario.scene_center_world, scenario.scene_basis)
    cluster_world = scenario.targets_world[cluster_indices]
    scores: List[tuple[float, int]] = []
    for idx in range(len(combined_local)):
        total_cover_gain = float(np.count_nonzero(coverage[idx] & (remaining > 0)))
        if total_cover_gain < 2.0:
            continue
        point_world = candidate_world[idx]
        extra_path_cost = float(np.min(np.linalg.norm(selected_points[:, :2] - point_world[None, :2], axis=1))) if len(selected_points) else 0.0
        cluster_cost = float(np.min(np.linalg.norm(cluster_world[:, :2] - point_world[None, :2], axis=1))) if len(cluster_world) else 0.0
        path_cost = max(extra_path_cost + 0.35 * cluster_cost, 1e-6)
        efficiency = total_cover_gain / path_cost
        if efficiency < (0.06 if scene_mode == "campus" else 0.075) and total_cover_gain < (3.0 if scene_mode == "campus" else 4.0):
            continue
        local_quality_gain = float(np.sum(quality[idx, remaining > 0]))
        score = 4.5 * total_cover_gain + 0.6 * local_quality_gain + 22.0 * efficiency - (0.07 if scene_mode == "campus" else 0.08) * path_cost
        scores.append((score, idx))
    if not scores:
        return scenario
    scores.sort(reverse=True)
    keep = np.asarray([idx for _score, idx in scores[:keep_candidates]], dtype=int)
    kept_local = combined_local[keep]
    kept_aims = combined_aims[keep]
    kept_coverage = coverage[keep]
    kept_quality = quality[keep]
    kept_owners = infer_candidate_owners(kept_coverage, scenario.target_buildings)
    return append_candidates_to_scenario(
        scenario,
        kept_local,
        kept_aims,
        kept_coverage,
        kept_quality,
        kept_owners,
        cluster_id=-8.0,
        region_code=4.6,
        importance=1.35,
    )


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
    enable_transition_rescue: bool = True,
) -> Coarse3DScenario:
    vertices, faces = read_triangle_mesh(mesh)
    center, basis, low, high = geometry_frame(vertices)
    targets_world, normals_world, areas = sample_surface(
        vertices, faces, max_targets, center, basis, low, high
    )
    target_buildings = [canonical_building_name(name) for name in assign_targets_to_buildings(targets_world, scene_json)]
    targets_local = to_local(targets_world, center, basis)
    normals_local = rotate_to_local(normals_world, basis)
    scene_span = np.linalg.norm(high - low)
    regions = classify_surface_regions(targets_local, normals_local, areas)
    patches = cluster_surface_patches(targets_local, normals_local, areas, regions, scene_span)
    patch_centers = np.asarray([patch.center for patch in patches], dtype=np.float64)
    patch_weights = np.asarray([patch.importance for patch in patches], dtype=np.float64)
    if len(patch_centers):
        region_count = max(3, min(8, max(1, len(patch_centers) // 4)))
        region_centers = _weighted_kmeans(patch_centers, patch_weights, region_count)
    else:
        region_centers = np.zeros((0, 3), dtype=np.float64)
    stand_off = 0.18 * float(np.linalg.norm(high - low))
    min_standoff = 0.65 * stand_off
    max_standoff = 1.80 * stand_off
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
    if not len(candidates_local):
        candidates_local, aims_local = generate_candidates(low, high, 24, 3, stand_off)
        metadata = [{"cluster_id": -1.0, "region_code": 1.0, "importance": 1.0} for _ in range(len(candidates_local))]
    candidates_local, aims_local = filter_candidate_constraints(
        candidates_local,
        aims_local,
        targets_local,
        center,
        basis,
        min_standoff,
        max_standoff,
        2.0,
        120.0,
    )
    metadata = metadata[: len(candidates_local)]
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
    _, remaining_cover = select_multicover(
        coverage,
        quality,
        candidates_local,
        required_views,
        candidate_region_codes,
    )
    if np.any(remaining_cover > 0):
        missing_indices = np.flatnonzero(remaining_cover > 0)
        supplemental_local, supplemental_aims = targeted_candidates(
            targets_local,
            normals_local,
            missing_indices,
            stand_off,
            max_standoff,
        )
        if len(supplemental_local):
            supplemental_local, supplemental_aims = filter_candidate_constraints(
                supplemental_local,
                supplemental_aims,
                targets_local,
                center,
                basis,
                min_standoff,
                max_standoff,
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
                rescue_budget = min(max(180, len(missing_indices) * 2), 420)
                supplemental_owners = infer_candidate_owners(supplemental_coverage, target_buildings)
                rescue_scores = owner_aware_rescue_scores(
                    supplemental_coverage,
                    supplemental_quality,
                    missing_indices,
                    target_buildings,
                    supplemental_owners,
                )
                keep = np.argsort(-rescue_scores)[: min(len(supplemental_local), rescue_budget)]
                supplemental_local = supplemental_local[keep]
                supplemental_aims = supplemental_aims[keep]
                supplemental_coverage = supplemental_coverage[keep]
                supplemental_quality = supplemental_quality[keep]
                supplemental_owners = [supplemental_owners[int(idx)] for idx in keep]
                supplemental_meta = [
                    {
                        "cluster_id": -2.0,
                        "region_code": 2.0 if owner != "unassigned" else 1.5,
                        "importance": 0.88 if owner != "unassigned" else 0.64,
                    }
                    for owner in supplemental_owners
                ]
                candidates_local = np.concatenate([candidates_local, supplemental_local], axis=0)
                aims_local = np.concatenate([aims_local, supplemental_aims], axis=0)
                coverage = np.concatenate([coverage, supplemental_coverage], axis=0)
                quality = np.concatenate([quality, supplemental_quality], axis=0)
                metadata.extend(supplemental_meta)
                current_region_codes = np.array([item["region_code"] for item in metadata], dtype=np.float32)
                _, remaining_cover = select_multicover(
                    coverage,
                    quality,
                    candidates_local,
                    required_views,
                    current_region_codes,
                )
    if enable_transition_rescue and np.any(remaining_cover > 0):
        hard_missing_indices = difficult_transition_missing_indices(
            targets_local,
            normals_local,
            target_buildings,
            regions,
            np.flatnonzero(remaining_cover > 0),
        )
        focused_cluster_indices = clustered_difficult_missing_indices(
            targets_local,
            hard_missing_indices,
            scene_span,
            max_clusters=6,
        )
        transition_local, transition_aims = focused_transition_cluster_candidates(
            targets_local,
            normals_local,
            focused_cluster_indices,
            target_buildings,
            stand_off,
            max_standoff,
        )
        if len(transition_local):
            transition_local, transition_aims = filter_candidate_constraints(
                transition_local,
                transition_aims,
                targets_local,
                center,
                basis,
                min_standoff,
                max_standoff,
                2.0,
                120.0,
            )
            if len(transition_local):
                transition_coverage, transition_quality = coverage_matrix(
                    transition_local,
                    transition_aims,
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
                transition_owners = infer_candidate_owners(transition_coverage, target_buildings)
                transition_budget = min(max(12, len(focused_cluster_indices) * 3), 24)
                transition_scores = owner_aware_rescue_scores(
                    transition_coverage,
                    transition_quality,
                    focused_cluster_indices,
                    target_buildings,
                    transition_owners,
                )
                transition_scores += np.where(
                    np.asarray(transition_owners, dtype=object) == "unassigned",
                    -0.45,
                    0.35,
                ).astype(np.float32)
                keep = np.argsort(-transition_scores)[: min(len(transition_local), transition_budget)]
                transition_local = transition_local[keep]
                transition_aims = transition_aims[keep]
                transition_coverage = transition_coverage[keep]
                transition_quality = transition_quality[keep]
                transition_owners = [transition_owners[int(idx)] for idx in keep]
                transition_meta = [
                    {
                        "cluster_id": -3.0,
                        "region_code": 3.4 if owner != "unassigned" else 2.4,
                        "importance": 0.90 if owner != "unassigned" else 0.70,
                    }
                    for owner in transition_owners
                ]
                candidates_local = np.concatenate([candidates_local, transition_local], axis=0)
                aims_local = np.concatenate([aims_local, transition_aims], axis=0)
                coverage = np.concatenate([coverage, transition_coverage], axis=0)
                quality = np.concatenate([quality, transition_quality], axis=0)
                metadata.extend(transition_meta)
    unique_sectors = candidate_sector_ids_from_world(
        np.einsum("ij,kj->ik", candidates_local, basis) + center,
        np.asarray(center, dtype=np.float64),
        bins=8,
    )
    if len(np.unique(unique_sectors)) < 4:
        supplemental_local, supplemental_aims = generate_candidates(low, high, max(24, max_candidates), 4, stand_off)
        supplemental_local, supplemental_aims = filter_candidate_constraints(
            supplemental_local,
            supplemental_aims,
            targets_local,
            center,
            basis,
            min_standoff,
            max_standoff,
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
                {"cluster_id": -2.0, "region_code": 5.0, "importance": 0.85}
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


def _selection_result_from_explicit_plan(
    scenario: Coarse3DScenario,
    selected_indices: Sequence[int],
    selected_shot_counts: Sequence[int],
    selected_layer_ids: Sequence[int] | None,
    selected_standoff_factors: Sequence[float],
    method: str,
    route_restarts: int = 24,
    route_proposals: int = 800,
) -> SelectionResult:
    selected_indices = [int(index) for index in selected_indices]
    selected_shot_counts = [int(count) for count in selected_shot_counts]
    selected_standoff_factors = [float(value) for value in selected_standoff_factors]
    selected_layer_ids = [int(layer) for layer in selected_layer_ids] if selected_layer_ids is not None else None
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
            restarts=route_restarts,
            proposals=route_proposals,
            turn_weight=2.0,
        )
        objective, route_length, _, _ = route_metrics(route_indices, route_positions_local, turn_weight=2.0)
    else:
        route_indices = []
        objective = route_length = 0.0
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
    ) if building_total else 0.0
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


def prune_selection_for_route(
    scenario: Coarse3DScenario,
    selected_indices: Sequence[int],
    selected_shot_counts: Sequence[int],
    selected_layer_ids: Sequence[int] | None,
    selected_standoff_factors: Sequence[float],
    method: str,
    max_iterations: int = 10,
) -> tuple[List[int], List[int], List[int] | None, List[float], SelectionResult]:
    current_indices = [int(idx) for idx in selected_indices]
    current_shots = [int(count) for count in selected_shot_counts]
    current_layers = [int(layer) for layer in selected_layer_ids] if selected_layer_ids is not None else None
    current_factors = [float(value) for value in selected_standoff_factors]
    current_result = _selection_result_from_explicit_plan(
        scenario,
        current_indices,
        current_shots,
        current_layers,
        current_factors,
        method,
        route_restarts=12,
        route_proposals=240,
    )
    if len(current_indices) <= 3:
        return current_indices, current_shots, current_layers, current_factors, current_result
    cfg = PPOTrainingConfig()
    coverage_floor = max(0.0, current_result.certified_coverage - 0.0025)
    forward_floor = max(FORWARD_OVERLAP_TARGET, current_result.mean_photo_overlap - 0.015)
    lateral_floor = max(LATERAL_OVERLAP_TARGET, current_result.route_photo_overlap - 0.015)
    quality_mean_floor = max(cfg.quality_mean_target, current_result.mean_quality_normalized - 0.015)
    quality_good_floor = max(cfg.quality_good_fraction_target, current_result.quality_good_fraction - 0.02)
    weakest_quality_floor = max(cfg.weakest_quality_target, current_result.weakest_quality_normalized - 0.02)
    iterations = 0
    while iterations < max_iterations and len(current_indices) > 3:
        best_candidate: tuple[float, int, SelectionResult] | None = None
        for remove_idx in list(current_result.route_indices):
            if int(remove_idx) not in current_indices:
                continue
            pos = current_indices.index(int(remove_idx))
            trial_indices = current_indices[:pos] + current_indices[pos + 1 :]
            trial_shots = current_shots[:pos] + current_shots[pos + 1 :]
            trial_layers = current_layers[:pos] + current_layers[pos + 1 :] if current_layers is not None else None
            trial_factors = current_factors[:pos] + current_factors[pos + 1 :]
            trial_result = _selection_result_from_explicit_plan(
                scenario,
                trial_indices,
                trial_shots,
                trial_layers,
                trial_factors,
                method,
                route_restarts=6,
                route_proposals=120,
            )
            if trial_result.certified_coverage < coverage_floor:
                continue
            if trial_result.mean_photo_overlap < forward_floor or trial_result.route_photo_overlap < lateral_floor:
                continue
            if trial_result.mean_quality_normalized < quality_mean_floor:
                continue
            if trial_result.quality_good_fraction < quality_good_floor:
                continue
            if trial_result.weakest_quality_normalized < weakest_quality_floor:
                continue
            path_gain = current_result.route_length - trial_result.route_length
            if path_gain < 8.0:
                continue
            penalty = (
                180.0 * max(current_result.certified_coverage - trial_result.certified_coverage, 0.0)
                + 70.0 * max(current_result.mean_photo_overlap - trial_result.mean_photo_overlap, 0.0)
                + 55.0 * max(current_result.route_photo_overlap - trial_result.route_photo_overlap, 0.0)
                + 40.0 * max(current_result.mean_quality_normalized - trial_result.mean_quality_normalized, 0.0)
                + 25.0 * max(current_result.quality_good_fraction - trial_result.quality_good_fraction, 0.0)
                + 25.0 * max(current_result.weakest_quality_normalized - trial_result.weakest_quality_normalized, 0.0)
            )
            score = path_gain - penalty
            if best_candidate is None or score > best_candidate[0]:
                best_candidate = (score, pos, trial_result)
        if best_candidate is None or best_candidate[0] <= 0.0:
            break
        _, pos, accepted_result = best_candidate
        current_indices.pop(pos)
        current_shots.pop(pos)
        current_factors.pop(pos)
        if current_layers is not None:
            current_layers.pop(pos)
        current_result = accepted_result
        iterations += 1
    return current_indices, current_shots, current_layers, current_factors, current_result


def evaluate_selection(
    scenario: Coarse3DScenario,
    selected_data: Sequence[int] | SelectionPolicyOutput | tuple[Sequence[int], Sequence[int]],
    method: str,
    apply_diversification: bool = True,
    apply_path_pruning: bool = True,
    apply_shot_rebalance: bool = True,
) -> SelectionResult:
    is_structured_ppo = isinstance(selected_data, SelectionPolicyOutput)
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
    if selected_shot_counts is None:
        selected_shot_counts = station_shot_counts(scenario, selected_indices)
    if apply_shot_rebalance and is_structured_ppo and selected_indices and selected_shot_counts is not None:
        pre_result = _selection_result_from_explicit_plan(
            scenario,
            selected_indices,
            selected_shot_counts,
            selected_layer_ids,
            selected_standoff_factors,
            method,
            route_restarts=12,
            route_proposals=320,
        )
        if pre_result.route_indices:
            pre_rebalanced_route_shots = rebalance_route_shot_counts(
                scenario,
                pre_result.route_indices,
                pre_result.route_standoff_factors,
            )
            pre_route_lookup = {
                int(idx): int(count)
                for idx, count in zip(pre_result.route_indices, pre_rebalanced_route_shots)
            }
            selected_shot_counts = [pre_route_lookup.get(int(idx), int(count)) for idx, count in zip(selected_indices, selected_shot_counts)]
    if selected_shot_counts is not None:
        pre_rescue_result = _selection_result_from_explicit_plan(
            scenario,
            selected_indices,
            selected_shot_counts,
            selected_layer_ids,
            selected_standoff_factors,
            method,
            route_restarts=12,
            route_proposals=320,
        )
        # Rescue is gated only by coverage weakness, never by overlap weakness.
        need_rescue = (
            pre_rescue_result.certified_coverage < 0.995
            or pre_rescue_result.weakest_region_coverage < 0.90
            or pre_rescue_result.weakest_sector_coverage < 0.90
        )
        if need_rescue:
            selected_indices, selected_shot_counts, selected_layer_ids, selected_standoff_factors = augment_selection_with_cluster_rescue(
                scenario,
                selected_indices,
                selected_shot_counts,
                selected_layer_ids,
                selected_standoff_factors,
                max_added=4,
            )
    if apply_path_pruning and is_structured_ppo and selected_shot_counts is not None:
        selected_indices, selected_shot_counts, selected_layer_ids, selected_standoff_factors, _pruned_result = prune_selection_for_route(
            scenario,
            selected_indices,
            selected_shot_counts,
            selected_layer_ids,
            selected_standoff_factors,
            method,
        )
    result = _selection_result_from_explicit_plan(
        scenario,
        selected_indices,
        selected_shot_counts,
        selected_layer_ids,
        selected_standoff_factors,
        method,
    )
    if apply_shot_rebalance and is_structured_ppo and result.route_indices:
        rebalanced_route_shots = rebalance_route_shot_counts(
            scenario,
            result.route_indices,
            result.route_standoff_factors,
        )
        route_lookup = {int(idx): int(count) for idx, count in zip(result.route_indices, rebalanced_route_shots)}
        rebalanced_selected_shots = [route_lookup.get(int(idx), 1) for idx in selected_indices]
        result = _selection_result_from_explicit_plan(
            scenario,
            selected_indices,
            rebalanced_selected_shots,
            selected_layer_ids,
            selected_standoff_factors,
            method,
        )
    return result


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
            - 1.10 * unassigned_flag
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
        candidate_matrix[:, 12] = 0.5 * cluster_deficit + 0.5 * region_deficit + 0.2 * center_distance - 0.75 * unassigned_flag
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
            34.0 * forward_gap_before * shot_alignment
            + 24.0 * forward_gap_before * shot_readiness
            - 4.0 * forward_gap_before * (1.0 - shot_readiness)
        )
        local_overlap_reward = (
            28.0 * local_forward_overlap
            + 18.0 * min(local_forward_overlap / max(FORWARD_OVERLAP_TARGET, 1e-6), 1.0)
            + 14.0 * local_lateral_overlap
            + 8.0 * min(local_lateral_overlap / max(LATERAL_OVERLAP_TARGET, 1e-6), 1.0)
        )
        shot_cost_penalty = (
            0.004 if before_forward_overlap < FORWARD_OVERLAP_TARGET else 0.10
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
            - 4.2 * owner_is_unassigned
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
    variant_seeds = tuple(range(training.variant_seed_base, training.variant_seed_base + training.variant_seed_count))
    repository = Coarse3DScenarioRepository(
        scenario,
        keep_candidates=len(scenario.candidates_local),
        variant_seeds=variant_seeds,
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
    variant_seeds = tuple(range(training.variant_seed_base, training.variant_seed_base + training.variant_seed_count))
    repository = Coarse3DScenarioRepository(
        scenario,
        keep_candidates=len(scenario.candidates_local),
        variant_seeds=variant_seeds,
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
        self.best_rank_key: tuple[float, float, float, float, float, float] | None = None
        self.history: List[TrainingHistoryRow] = []
        self.best_state_dict = None

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_every_steps != 0:
            return True
        result = evaluate_selection(
            self.scenario,
            self.policy_runner(self.scenario, self.model),
            "eval",
            apply_path_pruning=False,
            apply_shot_rebalance=False,
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
        rank_key = selection_rank_key(result)
        if self.best_rank_key is None or rank_key > self.best_rank_key:
            self.best_rank_key = rank_key
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
        if best_result is None or selection_rank_key(result) > selection_rank_key(best_result):
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
    parser.add_argument("--max-generated-candidates", type=int, default=2400)
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
