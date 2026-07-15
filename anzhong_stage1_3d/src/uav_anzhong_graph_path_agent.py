#!/usr/bin/env python3
"""Enhanced Anzhong pipeline with graph-attention path planning.

This script keeps the current two upstream agents unchanged:

1. candidate generation: facade normal + partition / clustering;
2. viewpoint selection: Maskable PPO over the generated candidate pool.

It replaces the downstream path organization stage with:

3. edge-aware graph attention ordering + A* routing + B-spline smoothing.

The implementation is intentionally lightweight and self-contained so it can
run with the project's current local dependencies (PyTorch, SB3, SciPy) and
does not require torch_geometric.
"""

from __future__ import annotations

import csv
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import Bounds, LinearConstraint, milp
from sb3_contrib import MaskablePPO

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
MPL_CACHE = ROOT / ".matplotlib_cache"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
MPL_CACHE.mkdir(exist_ok=True)
matplotlib.use("Agg")

from uav_2d_planner import EPS, Params, Point, distance
from uav_anzhong_viewpoint_rl import (
    AnzhongSceneRepository,
    FastAnzhongInspectionEnv,
    build_policy_observation,
)
from uav_multi_building_agents import (
    Building,
    CandidateView,
    ClearanceGridRouter,
    FacadeTarget,
    PlanResult,
    ReconstructionSuitabilityEvaluationAgent,
    best_quality,
    obstacle_aware_segment,
    path_length,
    plot_plan,
    segment_clear_with_clearance,
    segment_cost,
)
from uav_rl_inspection import EnvironmentConfig


@dataclass(frozen=True)
class GraphPathTrainingConfig:
    """Compact training plan for the path-ordering graph attention agent."""

    train_scene_seeds: Tuple[int, ...] = tuple(range(1, 9))
    eval_scene_seeds: Tuple[int, ...] = (9, 10)
    test_scene_seeds: Tuple[int, ...] = (0,)
    epochs: int = 70
    hidden_dim: int = 128
    heads: int = 4
    layers: int = 3
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    seed: int = 13
    device: str = "cpu"


@dataclass
class PathScenario:
    seed: int
    buildings: List[Building]
    targets: List[FacadeTarget]
    selected: List[CandidateView]
    start: Point
    best_quality: Dict[int, float]
    router: ClearanceGridRouter
    pair_costs: np.ndarray
    node_features: np.ndarray
    edge_features: np.ndarray
    baseline_order: List[int]
    bounds: Tuple[float, float, float, float]


class CoverageSetCoverSelector:
    """Fast coverage-focused selector for path-agent training scenes."""

    def __init__(
        self,
        params: Params,
        config: EnvironmentConfig,
        global_coverage_margin: float = 0.05,
    ) -> None:
        self.params = params
        self.config = config
        self.global_coverage_margin = global_coverage_margin

    def _required_count(self, total: int, ratio: float) -> int:
        return max(1, int(math.ceil(total * ratio - EPS)))

    def select(
        self,
        buildings: Sequence[Building],
        targets: Sequence[FacadeTarget],
        candidates: Sequence[CandidateView],
    ) -> List[CandidateView]:
        target_index = {target.id: index for index, target in enumerate(targets)}
        x_count = len(candidates)
        y_count = len(targets)
        total_vars = x_count + y_count

        objective = np.zeros(total_vars, dtype=float)
        objective[:x_count] = np.array([1.0 + 0.05 * candidate.mode_cost for candidate in candidates])
        objective[x_count:] = np.array([-0.001 * target.weight for target in targets])

        integrality = np.ones(total_vars, dtype=int)
        lower_bounds = np.zeros(total_vars, dtype=float)
        upper_bounds = np.ones(total_vars, dtype=float)
        rows: List[np.ndarray] = []
        lb: List[float] = []
        ub: List[float] = []

        for target in targets:
            row = np.zeros(total_vars, dtype=float)
            row[x_count + target_index[target.id]] = 1.0
            covering_candidates = 0
            for candidate_index, candidate in enumerate(candidates):
                if candidate.coverage.get(target.id, 0.0) >= self.params.effective_quality_threshold:
                    row[candidate_index] -= 1.0
                    covering_candidates += 1
            rows.append(row)
            lb.append(-np.inf)
            ub.append(0.0)
            if covering_candidates == 0:
                upper_bounds[x_count + target_index[target.id]] = 0.0

        required_ratio = self.config.minimum_per_building_effective_coverage
        for building in buildings:
            building_targets = [target for target in targets if target.building_id == building.id]
            row = np.zeros(total_vars, dtype=float)
            for target in building_targets:
                row[x_count + target_index[target.id]] = 1.0
            rows.append(row)
            lb.append(self._required_count(len(building_targets), required_ratio))
            ub.append(np.inf)

        global_ratio = min(0.95, required_ratio + self.global_coverage_margin)
        global_row = np.zeros(total_vars, dtype=float)
        for target in targets:
            global_row[x_count + target_index[target.id]] = 1.0
        rows.append(global_row)
        lb.append(self._required_count(len(targets), global_ratio))
        ub.append(np.inf)

        result = milp(
            c=objective,
            integrality=integrality,
            bounds=Bounds(lower_bounds, upper_bounds),
            constraints=LinearConstraint(np.vstack(rows), np.array(lb), np.array(ub)),
        )
        if not result.success or result.x is None:
            raise RuntimeError(f"Coverage set-cover selection failed: {result.message}")
        selection = result.x[:x_count] > 0.5
        return [candidate for candidate, keep in zip(candidates, selection) if keep]


def candidate_region_difficulty(region: str) -> float:
    return {
        "flat": 0.0,
        "convex": 0.5,
        "concave": 0.8,
        "occlusion_bottleneck": 1.0,
    }.get(region, 0.3)


def route_path_with_router(
    route: Sequence[CandidateView],
    start: Point,
    router: ClearanceGridRouter,
) -> List[Point]:
    ordered_points = [start] + [candidate.point for candidate in route] + [start]
    path = [ordered_points[0]]
    for index in range(len(ordered_points) - 1):
        try:
            leg = router.route(ordered_points[index], ordered_points[index + 1])
        except RuntimeError:
            leg = obstacle_aware_segment(ordered_points[index], ordered_points[index + 1], router.buildings)
        path.extend(leg[1:])
    return path


def validate_polyline(
    points: Sequence[Point],
    buildings: Sequence[Building],
    clearance: float = 0.7,
) -> bool:
    return all(
        segment_clear_with_clearance(points[index], points[index + 1], buildings, clearance)
        for index in range(len(points) - 1)
    )


def bspline_smooth_path(
    points: Sequence[Point],
    buildings: Sequence[Building],
    sample_factor: int = 10,
) -> List[Point]:
    if len(points) < 4:
        return list(points)

    from scipy.interpolate import splev, splprep

    deduped = [points[0]]
    for point in points[1:]:
        if distance(point, deduped[-1]) > 1e-6:
            deduped.append(point)
    if len(deduped) < 4:
        return deduped

    raw_length = path_length(deduped)
    x = np.array([point[0] for point in deduped], dtype=float)
    y = np.array([point[1] for point in deduped], dtype=float)
    try:
        tck, _ = splprep([x, y], s=0.0, k=min(3, len(deduped) - 1))
        samples = max(len(deduped) * sample_factor, len(deduped) + 8)
        u_new = np.linspace(0.0, 1.0, samples)
        x_new, y_new = splev(u_new, tck)
        smoothed = [(float(px), float(py)) for px, py in zip(x_new, y_new)]
        if validate_polyline(smoothed, buildings) and path_length(smoothed) <= raw_length * 1.18:
            return smoothed
    except Exception:
        pass
    return deduped


def closed_route_cost(order: Sequence[int], pair_costs: np.ndarray) -> float:
    if not order:
        return 0.0
    total = pair_costs[0, order[0]]
    for left, right in zip(order, order[1:]):
        total += pair_costs[left, right]
    total += pair_costs[order[-1], 0]
    return float(total)


def baseline_order_from_pair_costs(pair_costs: np.ndarray) -> List[int]:
    node_count = pair_costs.shape[0]
    remaining = list(range(1, node_count))
    order: List[int] = []
    current = 0
    while remaining:
        next_node = min(remaining, key=lambda node: pair_costs[current, node])
        order.append(next_node)
        remaining.remove(next_node)
        current = next_node

    best = list(order)
    improved = True
    iterations = 0
    while improved and iterations < 8:
        improved = False
        iterations += 1
        for left in range(len(best) - 1):
            for right in range(left + 2, len(best) + (0 if left else -1)):
                proposal = best[:left] + list(reversed(best[left:right])) + best[right:]
                if closed_route_cost(proposal, pair_costs) + EPS < closed_route_cost(best, pair_costs):
                    best = proposal
                    improved = True
    return best


def polish_order_with_pair_costs(order: Sequence[int], pair_costs: np.ndarray, max_rounds: int = 6) -> List[int]:
    best = list(order)
    if len(best) < 4:
        return best
    improved = True
    rounds = 0
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1
        current_cost = closed_route_cost(best, pair_costs)
        for left in range(len(best) - 1):
            for right in range(left + 2, len(best) + (0 if left else -1)):
                proposal = best[:left] + list(reversed(best[left:right])) + best[right:]
                proposal_cost = closed_route_cost(proposal, pair_costs)
                if proposal_cost + EPS < current_cost:
                    best = proposal
                    current_cost = proposal_cost
                    improved = True
    return best


def select_views_with_maskable_ppo(
    repository: AnzhongSceneRepository,
    config: EnvironmentConfig,
    model: MaskablePPO,
    seed: int,
) -> Tuple[List[Building], List[FacadeTarget], List[CandidateView], Point]:
    env = FastAnzhongInspectionEnv(repository, config)
    env.reset(seed)
    while True:
        mask = env.action_mask()
        if not np.any(mask):
            break
        observation = build_policy_observation(env, model.action_space.n)
        padded_mask = np.pad(mask, (0, model.action_space.n - len(mask)), constant_values=False)
        action, _ = model.predict(observation, deterministic=True, action_masks=padded_mask)
        _, done, _ = env.step(int(action))
        if done:
            break
    return list(env.buildings), list(env.targets), list(env.selected), env.start


class PathScenarioRepository:
    """Connect candidate generation, PPO selection, and graph path planning."""

    def __init__(
        self,
        params: Params,
        config: EnvironmentConfig,
        selection_model_path: Path,
    ) -> None:
        self.params = params
        self.config = config
        self.viewpoint_repository = AnzhongSceneRepository(params, config)
        self.selection_model = MaskablePPO.load(selection_model_path)
        self.coverage_selector = CoverageSetCoverSelector(params, config)
        self._cache: Dict[Tuple[int, str], PathScenario] = {}

    def get(self, seed: int, selection_mode: str = "set_cover") -> PathScenario:
        cache_key = (seed, selection_mode)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if selection_mode == "maskable_ppo":
            buildings, targets, selected, start = select_views_with_maskable_ppo(
                self.viewpoint_repository,
                self.config,
                self.selection_model,
                seed,
            )
        else:
            buildings, targets, candidates, _ = self.viewpoint_repository.get(seed)
            selected = self.coverage_selector.select(buildings, targets, candidates)
            start = (12.0, 8.0)
        router = ClearanceGridRouter(buildings)
        all_points = [start] + [view.point for view in selected]
        pair_costs = np.zeros((len(all_points), len(all_points)), dtype=np.float32)
        for left in range(len(all_points)):
            for right in range(left + 1, len(all_points)):
                pair_cost = float(segment_cost(all_points[left], all_points[right], buildings))
                pair_costs[left, right] = pair_cost
                pair_costs[right, left] = pair_cost

        xs = [point[0] for building in buildings for point in building.polygon]
        ys = [point[1] for building in buildings for point in building.polygon]
        bounds = (min(xs) - 8.0, max(xs) + 8.0, min(ys) - 8.0, max(ys) + 8.0)
        node_features = self._node_features(selected, start, targets, buildings, pair_costs, bounds)
        edge_features = self._edge_features(selected, start, buildings, pair_costs)
        baseline_order = baseline_order_from_pair_costs(pair_costs)
        scenario = PathScenario(
            seed=seed,
            buildings=buildings,
            targets=targets,
            selected=selected,
            start=start,
            best_quality=best_quality(selected, targets),
            router=router,
            pair_costs=pair_costs,
            node_features=node_features,
            edge_features=edge_features,
            baseline_order=baseline_order,
            bounds=bounds,
        )
        self._cache[cache_key] = scenario
        return scenario

    def _node_features(
        self,
        selected: Sequence[CandidateView],
        start: Point,
        targets: Sequence[FacadeTarget],
        buildings: Sequence[Building],
        pair_costs: np.ndarray,
        bounds: Tuple[float, float, float, float],
    ) -> np.ndarray:
        min_x, max_x, min_y, max_y = bounds
        span_x = max(max_x - min_x, 1.0)
        span_y = max(max_y - min_y, 1.0)
        buildings_by_id = {building.id: index for index, building in enumerate(buildings)}
        features = []
        cost_scale = max(float(np.max(pair_costs)), 1.0)
        features.append(
            np.array(
                [
                    (start[0] - min_x) / span_x,
                    (start[1] - min_y) / span_y,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                dtype=np.float32,
            )
        )
        target_count = max(len(targets), 1)
        for local_index, candidate in enumerate(selected, start=1):
            effective_support = sum(
                1
                for quality in candidate.coverage.values()
                if quality >= self.params.effective_quality_threshold
            ) / target_count
            coverage_mass = sum(candidate.coverage.values()) / target_count
            features.append(
                np.array(
                    [
                        (candidate.point[0] - min_x) / span_x,
                        (candidate.point[1] - min_y) / span_y,
                        candidate.direction[0],
                        candidate.direction[1],
                        candidate_region_difficulty(candidate.region),
                        buildings_by_id[candidate.source_building] / max(1, len(buildings) - 1),
                        candidate.mode_cost,
                        effective_support,
                        pair_costs[0, local_index] / cost_scale,
                        0.0,
                    ],
                    dtype=np.float32,
                )
            )
        return np.vstack(features)

    def _edge_features(
        self,
        selected: Sequence[CandidateView],
        start: Point,
        buildings: Sequence[Building],
        pair_costs: np.ndarray,
    ) -> np.ndarray:
        points = [start] + [candidate.point for candidate in selected]
        cost_scale = max(float(np.max(pair_costs)), 1.0)
        distance_scale = max(max(distance(points[left], points[right]) for left in range(len(points)) for right in range(len(points))), 1.0)
        features = np.zeros((len(points), len(points), 5), dtype=np.float32)
        for left in range(len(points)):
            for right in range(len(points)):
                if left == right:
                    continue
                euclidean = distance(points[left], points[right]) / distance_scale
                obstacle_cost = pair_costs[left, right] / cost_scale
                cost_gap = max(0.0, obstacle_cost - euclidean)
                if left == 0 or right == 0:
                    building_switch = 0.0
                    view_turn = 0.0
                else:
                    building_switch = float(selected[left - 1].source_building != selected[right - 1].source_building)
                    dot_value = (
                        selected[left - 1].direction[0] * selected[right - 1].direction[0]
                        + selected[left - 1].direction[1] * selected[right - 1].direction[1]
                    )
                    view_turn = 0.5 * (1.0 - max(-1.0, min(1.0, dot_value)))
                bottleneck_bias = 0.5 * (
                    (candidate_region_difficulty(selected[left - 1].region) if left > 0 else 0.0)
                    + (candidate_region_difficulty(selected[right - 1].region) if right > 0 else 0.0)
                )
                features[left, right] = np.array(
                    [euclidean, obstacle_cost, cost_gap, building_switch, 0.5 * view_turn + 0.5 * bottleneck_bias],
                    dtype=np.float32,
                )
        return features


class EdgeAwareGraphAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, heads: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_bias = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, heads),
        )
        self.edge_value = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.ff_norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_state: torch.Tensor, edge_state: torch.Tensor) -> torch.Tensor:
        node_count = node_state.shape[0]
        query = self.q_proj(node_state).view(node_count, self.heads, self.head_dim)
        key = self.k_proj(node_state).view(node_count, self.heads, self.head_dim)
        value = self.v_proj(node_state).view(node_count, self.heads, self.head_dim)
        logits = torch.einsum("ihd,jhd->ijh", query, key) / math.sqrt(self.head_dim)
        logits = logits + self.edge_bias(edge_state)
        attention = torch.softmax(logits, dim=1)
        edge_value = self.edge_value(edge_state).view(node_count, node_count, self.heads, self.head_dim)
        messages = torch.einsum("ijh,jhd->ihd", attention, value) + torch.einsum("ijh,ijhd->ihd", attention, edge_value)
        updated = self.out_proj(messages.reshape(node_count, self.hidden_dim))
        node_state = self.norm(node_state + updated)
        node_state = self.ff_norm(node_state + self.ff(node_state))
        return node_state


class EdgeAwareRouteDecoder(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(3 * hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.edge_bias = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def step_logits(
        self,
        node_state: torch.Tensor,
        edge_state: torch.Tensor,
        current_index: int,
        visited: torch.Tensor,
    ) -> torch.Tensor:
        visited = visited.clone()
        remaining = ~visited
        remaining[0] = False
        if torch.any(remaining):
            global_context = node_state[remaining].mean(dim=0)
        else:
            global_context = node_state.mean(dim=0)
        query = self.query(torch.cat([node_state[current_index], global_context, node_state[0]], dim=0))
        key = self.key(node_state)
        logits = torch.matmul(key, query) / math.sqrt(node_state.shape[-1])
        logits = logits + self.edge_bias(edge_state[current_index]).squeeze(-1)
        logits[visited] = -1e9
        logits[0] = -1e9
        return logits


class EdgeAwareGraphAttentionPlanner(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, heads: int, layers: int) -> None:
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, edge_dim)
        self.layers = nn.ModuleList(
            [EdgeAwareGraphAttentionLayer(hidden_dim, edge_dim, heads) for _ in range(layers)]
        )
        self.decoder = EdgeAwareRouteDecoder(hidden_dim, edge_dim)

    def encode(self, node_features: torch.Tensor, edge_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        node_state = self.node_proj(node_features)
        edge_state = self.edge_proj(edge_features)
        for layer in self.layers:
            node_state = layer(node_state, edge_state)
        return node_state, edge_state

    def loss(self, node_features: torch.Tensor, edge_features: torch.Tensor, target_order: Sequence[int]) -> torch.Tensor:
        node_state, edge_state = self.encode(node_features, edge_features)
        visited = torch.zeros(node_state.shape[0], dtype=torch.bool, device=node_state.device)
        visited[0] = True
        current_index = 0
        losses = []
        for target in target_order:
            logits = self.decoder.step_logits(node_state, edge_state, current_index, visited.clone())
            losses.append(F.cross_entropy(logits.unsqueeze(0), torch.tensor([target], device=node_state.device)))
            visited[target] = True
            current_index = target
        return torch.stack(losses).mean()

    @torch.no_grad()
    def decode(self, node_features: torch.Tensor, edge_features: torch.Tensor) -> List[int]:
        node_state, edge_state = self.encode(node_features, edge_features)
        visited = torch.zeros(node_state.shape[0], dtype=torch.bool, device=node_state.device)
        visited[0] = True
        current_index = 0
        order: List[int] = []
        while len(order) + 1 < node_state.shape[0]:
            logits = self.decoder.step_logits(node_state, edge_state, current_index, visited)
            next_index = int(torch.argmax(logits).item())
            if visited[next_index]:
                break
            visited[next_index] = True
            order.append(next_index)
            current_index = next_index
        missing = [index for index in range(1, node_state.shape[0]) if index not in order]
        order.extend(missing)
        return order


class GraphAttentionPathPlanningAgent:
    """Graph-attention ordering + A* routing + B-spline smoothing."""

    def __init__(self, model: EdgeAwareGraphAttentionPlanner, device: torch.device) -> None:
        self.model = model
        self.device = device

    def order_views(self, scenario: PathScenario) -> List[int]:
        node_features = torch.tensor(scenario.node_features, dtype=torch.float32, device=self.device)
        edge_features = torch.tensor(scenario.edge_features, dtype=torch.float32, device=self.device)
        return [index - 1 for index in self.model.decode(node_features, edge_features)]

    def plan(self, scenario: PathScenario) -> PlanResult:
        order = polish_order_with_pair_costs(
            [index + 1 for index in self.order_views(scenario)],
            scenario.pair_costs,
        )
        route = [scenario.selected[index - 1] for index in order]
        geometric = route_path_with_router(route, scenario.start, scenario.router)
        smoothed = bspline_smooth_path(geometric, scenario.buildings)
        return PlanResult(
            name="edge-aware graph attention + A* + B-spline",
            selected=list(scenario.selected),
            route=route,
            path_points=smoothed,
            best_quality=dict(scenario.best_quality),
            stop_reason="graph-attention ordering with deterministic geometric routing",
        )


def build_baseline_plan(scenario: PathScenario) -> PlanResult:
    route = [scenario.selected[index - 1] for index in scenario.baseline_order]
    path = route_path_with_router(route, scenario.start, scenario.router)
    return PlanResult(
        name="current path baseline",
        selected=list(scenario.selected),
        route=route,
        path_points=path,
        best_quality=dict(scenario.best_quality),
        stop_reason="nearest-neighbour + 2-opt surrogate order + A* routing",
    )


def route_cost_from_result(result: PlanResult, scenario: PathScenario) -> float:
    if not result.route:
        return 0.0
    point_to_index = {candidate.id: local_index + 1 for local_index, candidate in enumerate(scenario.selected)}
    order = [point_to_index[candidate.id] for candidate in result.route]
    return closed_route_cost(order, scenario.pair_costs)


def train_graph_attention_path_agent(
    repository: PathScenarioRepository,
    config: GraphPathTrainingConfig,
) -> Tuple[EdgeAwareGraphAttentionPlanner, List[Dict[str, float]]]:
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device)
    first_scenario = repository.get(config.train_scene_seeds[0], selection_mode="set_cover")
    model = EdgeAwareGraphAttentionPlanner(
        node_dim=first_scenario.node_features.shape[1],
        edge_dim=first_scenario.edge_features.shape[2],
        hidden_dim=config.hidden_dim,
        heads=config.heads,
        layers=config.layers,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_eval_cost = math.inf
    best_state: Dict[str, torch.Tensor] | None = None
    history: List[Dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses = []
        scene_seeds = list(config.train_scene_seeds)
        random.shuffle(scene_seeds)
        for seed in scene_seeds:
            scenario = repository.get(seed, selection_mode="set_cover")
            node_features = torch.tensor(scenario.node_features, dtype=torch.float32, device=device)
            edge_features = torch.tensor(scenario.edge_features, dtype=torch.float32, device=device)
            target_order = list(scenario.baseline_order)
            optimizer.zero_grad()
            loss = model.loss(node_features, edge_features, target_order)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        model.eval()
        eval_costs = []
        for seed in config.eval_scene_seeds:
            scenario = repository.get(seed, selection_mode="set_cover")
            node_features = torch.tensor(scenario.node_features, dtype=torch.float32, device=device)
            edge_features = torch.tensor(scenario.edge_features, dtype=torch.float32, device=device)
            order = model.decode(node_features, edge_features)
            eval_costs.append(closed_route_cost(order, scenario.pair_costs))
        mean_eval_cost = float(np.mean(eval_costs))
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean(epoch_losses)),
                "eval_route_cost_m": mean_eval_cost,
            }
        )
        if mean_eval_cost + EPS < best_eval_cost:
            best_eval_cost = mean_eval_cost
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def evaluate_methods(
    repository: PathScenarioRepository,
    training: GraphPathTrainingConfig,
    agent: GraphAttentionPathPlanningAgent,
) -> List[Dict[str, float | str]]:
    evaluator = ReconstructionSuitabilityEvaluationAgent()
    rows: List[Dict[str, float | str]] = []
    for seed in training.test_scene_seeds:
        scenario = repository.get(seed, selection_mode="maskable_ppo")
        baseline = build_baseline_plan(scenario)
        learned = agent.plan(scenario)
        for result in (baseline, learned):
            metrics = evaluator.evaluate(
                result,
                scenario.targets,
                scenario.buildings,
                Params(
                    target_spacing=4.0,
                    min_distance=3.0,
                    max_distance=20.0,
                    effective_quality_threshold=0.50,
                ),
            )
            rows.append(
                {
                    "method": result.name,
                    "seed": seed,
                    "selected_views": metrics["selected_views"],
                    "effective_coverage": metrics["effective_coverage"],
                    "minimum_building_effective_coverage": min(
                        values["effective_coverage"] for values in metrics["per_building"].values()
                    ),
                    "average_best_quality": metrics["average_best_quality"],
                    "path_length_m": metrics["path_length_m"],
                    "route_cost_proxy_m": route_cost_from_result(result, scenario),
                    "observation_redundancy": metrics["observation_redundancy"],
                }
            )
    return rows


def write_results(
    rows: Sequence[Dict[str, float | str]],
    history: Sequence[Dict[str, float]],
) -> None:
    with (OUTPUT_DIR / "anzhong_graph_path_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUTPUT_DIR / "anzhong_graph_path_training_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with (OUTPUT_DIR / "anzhong_graph_path_results.md").open("w", encoding="utf-8") as file:
        file.write("# 安中场景：Graph Attention 路径规划智能体\n\n")
        file.write("完整主线：法向+分区/聚类候选生成 -> Maskable PPO 视点选取 -> edge-aware graph attention + A* + B-spline 路径规划。\n\n")
        file.write("| 方法 | 视点数 | 有效覆盖 | 最弱建筑覆盖 | 平均质量 | 路径长度 | 路径代理代价 |\n")
        file.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            file.write(
                f"| {row['method']} | {row['selected_views']} | "
                f"{float(row['effective_coverage']):.1%} | "
                f"{float(row['minimum_building_effective_coverage']):.1%} | "
                f"{float(row['average_best_quality']):.3f} | "
                f"{float(row['path_length_m']):.1f} m | "
                f"{float(row['route_cost_proxy_m']):.1f} m |\n"
            )


def run() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    params = Params(
        target_spacing=4.0,
        min_distance=3.0,
        max_distance=20.0,
        effective_quality_threshold=0.50,
    )
    selection_config = EnvironmentConfig(
        max_steps=48,
        minimum_per_building_effective_coverage=0.55,
        q_eff=params.effective_quality_threshold,
        reward_effective_coverage=70.0,
        reward_quality=16.0,
        reward_scarcity=10.0,
        reward_fairness=12.0,
        penalty_path=0.34,
    )
    training = GraphPathTrainingConfig()
    selection_model_path = OUTPUT_DIR / "anzhong_maskable_ppo_best.zip"
    repository = PathScenarioRepository(params, selection_config, selection_model_path)

    print("Training edge-aware graph attention path agent on jittered Anzhong scenes...")
    model, history = train_graph_attention_path_agent(repository, training)
    torch.save(model.state_dict(), OUTPUT_DIR / "anzhong_graph_attention_path_best.pt")

    agent = GraphAttentionPathPlanningAgent(model, torch.device(training.device))
    rows = evaluate_methods(repository, training, agent)
    write_results(rows, history)

    scenario = repository.get(training.test_scene_seeds[0], selection_mode="maskable_ppo")
    baseline = build_baseline_plan(scenario)
    learned = agent.plan(scenario)
    plot_plan(
        baseline,
        scenario.buildings,
        scenario.targets,
        scenario.selected,
        scenario.start,
        OUTPUT_DIR / "anzhong_graph_path_baseline_plan.png",
        params,
    )
    plot_plan(
        learned,
        scenario.buildings,
        scenario.targets,
        scenario.selected,
        scenario.start,
        OUTPUT_DIR / "anzhong_graph_path_attention_plan.png",
        params,
    )

    for row in rows:
        print(
            f"{row['method']}: effective={float(row['effective_coverage']):.1%}, "
            f"min-building={float(row['minimum_building_effective_coverage']):.1%}, "
            f"path={float(row['path_length_m']):.1f} m, "
            f"proxy={float(row['route_cost_proxy_m']):.1f} m"
        )


if __name__ == "__main__":
    run()
