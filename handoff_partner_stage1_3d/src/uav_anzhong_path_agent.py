#!/usr/bin/env python3
"""Compact three-layer path planning agent for the Anzhong scene.

This prototype keeps the current viewpoint generation and viewpoint-selection
modules unchanged.  It only replaces the *path organization* stage after a
viewpoint set has already been selected.

Three layers:
1. PPO learns the visiting order over the selected viewpoints.
2. A* generates clearance-safe geometric paths between consecutive views.
3. B-spline smooths the A* polyline into a more executable flight trajectory.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.interpolate import splprep, splev
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
MPL_CACHE = ROOT / ".matplotlib_cache"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
MPL_CACHE.mkdir(exist_ok=True)
import matplotlib

matplotlib.use("Agg")

from uav_2d_planner import EPS, Params, Point, distance
from uav_anzhong_viewpoint_rl import AnzhongSceneRepository
from uav_multi_building_agents import (
    Building,
    CandidateView,
    ClearanceGridRouter,
    FacadeTarget,
    PlanResult,
    ReconstructionSuitabilityEvaluationAgent,
    best_quality,
    nearest_neighbor_route,
    path_length,
    plot_plan,
    route_path,
    segment_clear_with_clearance,
    segment_cost,
    two_opt,
)
from uav_rl_inspection import EnvironmentConfig


@dataclass(frozen=True)
class PathPlanningTrainingConfig:
    """Small PPO path-ordering setup that stays runnable on a laptop."""

    train_seeds: Tuple[int, ...] = (0,)
    eval_seeds: Tuple[int, ...] = (0,)
    test_seeds: Tuple[int, ...] = (0,)
    total_timesteps: int = 4_000
    learning_rate: float = 3e-4
    n_steps: int = 128
    batch_size: int = 64
    n_epochs: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    clip_range: float = 0.2
    eval_every_steps: int = 1_000
    seed: int = 23


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
    bounds: Tuple[float, float, float, float]
    building_slots: Dict[str, int]


class CoverageSetCoverSelector:
    """Reuse the current candidate pool, but keep only the coverage-side ILP.

    This avoids accidentally invoking the old path-organization routine while
    we prototype a standalone path-planning agent.
    """

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
        target_index = {target.id: idx for idx, target in enumerate(targets)}
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
            for candidate_idx, candidate in enumerate(candidates):
                if candidate.coverage.get(target.id, 0.0) >= self.params.effective_quality_threshold:
                    row[candidate_idx] -= 1.0
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


class PathScenarioRepository:
    """Build selected-viewpoint routing problems from the existing codebase."""

    def __init__(self, params: Params, config: EnvironmentConfig) -> None:
        self.params = params
        self.config = config
        self.viewpoint_repository = AnzhongSceneRepository(params, config)
        self.selector = CoverageSetCoverSelector(params, config)
        self._cache: Dict[int, PathScenario] = {}

    def get(self, seed: int) -> PathScenario:
        if seed in self._cache:
            return self._cache[seed]

        buildings, targets, candidates, _ = self.viewpoint_repository.get(seed)
        start = (12.0, 8.0)
        selected = self.selector.select(buildings, targets, candidates)
        router = ClearanceGridRouter(buildings)
        all_points = [start] + [view.point for view in selected]
        pair_costs = np.zeros((len(all_points), len(all_points)), dtype=np.float32)
        for i in range(len(all_points)):
            for j in range(i + 1, len(all_points)):
                cost = float(segment_cost(all_points[i], all_points[j], buildings))
                pair_costs[i, j] = cost
                pair_costs[j, i] = cost

        xs = [point[0] for building in buildings for point in building.polygon]
        ys = [point[1] for building in buildings for point in building.polygon]
        bounds = (min(xs) - 8.0, max(xs) + 8.0, min(ys) - 8.0, max(ys) + 8.0)
        building_slots = {building.id: idx for idx, building in enumerate(buildings)}
        scenario = PathScenario(
            seed=seed,
            buildings=buildings,
            targets=targets,
            selected=selected,
            start=start,
            best_quality=best_quality(selected, targets),
            router=router,
            pair_costs=pair_costs,
            bounds=bounds,
            building_slots=building_slots,
        )
        self._cache[seed] = scenario
        return scenario


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
        path.extend(router.route(ordered_points[index], ordered_points[index + 1])[1:])
    return path


def validate_polyline(points: Sequence[Point], buildings: Sequence[Building], clearance: float = 0.7) -> bool:
    return all(
        segment_clear_with_clearance(points[index], points[index + 1], buildings, clearance)
        for index in range(len(points) - 1)
    )


def bspline_smooth_path(
    points: Sequence[Point],
    buildings: Sequence[Building],
    sample_factor: int = 10,
) -> List[Point]:
    """Smooth a path by cubic B-spline interpolation, with safe fallback."""

    if len(points) < 4:
        return list(points)

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
        smooth_length = path_length(smoothed)
        if (
            validate_polyline(smoothed, buildings)
            and smooth_length <= raw_length * 1.18
        ):
            return smoothed
    except Exception:
        pass
    return deduped


def build_baseline_plan(scenario: PathScenario) -> PlanResult:
    route = two_opt(
        nearest_neighbor_route(scenario.selected, scenario.start, scenario.buildings),
        scenario.start,
        scenario.buildings,
    )
    path = route_path(route, scenario.start, scenario.buildings)
    return PlanResult(
        name="current path baseline",
        selected=list(scenario.selected),
        route=route,
        path_points=path,
        best_quality=dict(scenario.best_quality),
        stop_reason="baseline nearest-neighbour + 2-opt + A* routing",
    )


class ThreeLayerPathPlanningAgent:
    """PPO ordering + A* path + B-spline smoothing."""

    def __init__(self, model: MaskablePPO, max_views: int) -> None:
        self.model = model
        self.max_views = max_views

    def order_views(self, scenario: PathScenario) -> List[int]:
        env = PathOrderingEnv(PathScenarioRepository(Params(), EnvironmentConfig()), (), self.max_views)
        env.load_scenario(scenario)
        observation = env.observation()
        while True:
            masks = env.action_masks()
            if not np.any(masks):
                break
            action, _ = self.model.predict(observation, action_masks=masks, deterministic=True)
            observation, _, terminated, _, _ = env.step(int(action))
            if terminated:
                break
        return list(env.visit_order)

    def plan(self, scenario: PathScenario) -> PlanResult:
        order = self.order_views(scenario)
        route = [scenario.selected[action_id] for action_id in order]
        geometric = route_path_with_router(route, scenario.start, scenario.router)
        smooth = bspline_smooth_path(geometric, scenario.buildings)
        return PlanResult(
            name="PPO + A* + B-spline path agent",
            selected=list(scenario.selected),
            route=route,
            path_points=smooth,
            best_quality=dict(scenario.best_quality),
            stop_reason="three-layer path optimization",
        )


class PathOrderingEnv(gym.Env):
    """Compact path-ordering environment over a fixed selected viewpoint set."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        repository: PathScenarioRepository,
        scenario_seeds: Sequence[int],
        max_views: int,
    ) -> None:
        super().__init__()
        self.repository = repository
        self.scenario_seeds = list(scenario_seeds)
        self.max_views = max_views
        self.cursor = 0
        self.scenario: PathScenario | None = None
        self.current_index = 0
        self.visit_order: List[int] = []
        self.visited: List[bool] = []
        self.cumulative_cost = 0.0
        self.action_space = spaces.Discrete(max_views)
        self.observation_space = spaces.Dict(
            {
                "global": spaces.Box(low=-2.0, high=2.0, shape=(6,), dtype=np.float32),
                "candidates": spaces.Box(low=-2.0, high=2.0, shape=(max_views, 9), dtype=np.float32),
            }
        )

    def load_scenario(self, scenario: PathScenario) -> None:
        self.scenario = scenario
        self.current_index = 0
        self.visit_order = []
        self.visited = [False] * len(scenario.selected)
        self.cumulative_cost = 0.0

    def _next_seed(self) -> int:
        seed = self.scenario_seeds[self.cursor % len(self.scenario_seeds)]
        self.cursor += 1
        return seed

    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        super().reset(seed=seed)
        scenario_seed = self._next_seed()
        if options and "scenario_seed" in options:
            scenario_seed = int(options["scenario_seed"])
        self.load_scenario(self.repository.get(scenario_seed))
        return self.observation(), {}

    def action_masks(self) -> np.ndarray:
        assert self.scenario is not None
        mask = np.zeros(self.max_views, dtype=bool)
        for action_id in range(len(self.scenario.selected)):
            mask[action_id] = not self.visited[action_id]
        return mask

    def observation(self) -> Dict[str, np.ndarray]:
        assert self.scenario is not None
        min_x, max_x, min_y, max_y = self.scenario.bounds
        span_x = max(max_x - min_x, 1.0)
        span_y = max(max_y - min_y, 1.0)
        current_point = self.scenario.start if self.current_index == 0 else self.scenario.selected[self.current_index - 1].point
        global_features = np.array(
            [
                (current_point[0] - min_x) / span_x,
                (current_point[1] - min_y) / span_y,
                len(self.visit_order) / max(1, len(self.scenario.selected)),
                (len(self.scenario.selected) - len(self.visit_order)) / max(1, len(self.scenario.selected)),
                min(2.0, self.cumulative_cost / 1000.0),
                self.scenario.pair_costs[self.current_index, 0] / max(np.max(self.scenario.pair_costs), 1.0),
            ],
            dtype=np.float32,
        )
        matrix = np.zeros((self.max_views, 9), dtype=np.float32)
        cost_scale = max(float(np.max(self.scenario.pair_costs)), 1.0)
        for action_id, candidate in enumerate(self.scenario.selected):
            dx = (candidate.point[0] - current_point[0]) / span_x
            dy = (candidate.point[1] - current_point[1]) / span_y
            leg_cost = self.scenario.pair_costs[self.current_index, action_id + 1] / cost_scale
            start_cost = self.scenario.pair_costs[0, action_id + 1] / cost_scale
            matrix[action_id] = np.array(
                [
                    dx,
                    dy,
                    leg_cost,
                    start_cost,
                    (candidate.point[0] - min_x) / span_x,
                    (candidate.point[1] - min_y) / span_y,
                    candidate_region_difficulty(candidate.region),
                    self.scenario.building_slots[candidate.source_building] / 4.0,
                    1.0 if not self.visited[action_id] else -1.0,
                ],
                dtype=np.float32,
            )
        return {"global": global_features, "candidates": matrix}

    def step(self, action: int):
        assert self.scenario is not None
        if action >= len(self.scenario.selected) or self.visited[action]:
            return self.observation(), -1.0, True, False, {"invalid_action": True}

        next_index = action + 1
        leg_cost = float(self.scenario.pair_costs[self.current_index, next_index])
        self.cumulative_cost += leg_cost
        self.current_index = next_index
        self.visited[action] = True
        self.visit_order.append(action)
        done = len(self.visit_order) == len(self.scenario.selected)
        reward = -leg_cost / 100.0
        if done:
            return_cost = float(self.scenario.pair_costs[self.current_index, 0])
            self.cumulative_cost += return_cost
            reward += 1.0 - return_cost / 100.0
        return self.observation(), reward, done, False, {"leg_cost": leg_cost}


class PathEvalCallback(BaseCallback):
    """Compact evaluation callback that keeps the shortest mean route model."""

    def __init__(
        self,
        repository: PathScenarioRepository,
        eval_seeds: Sequence[int],
        max_views: int,
        best_model_path: Path,
        eval_every_steps: int,
    ) -> None:
        super().__init__()
        self.repository = repository
        self.eval_seeds = tuple(eval_seeds)
        self.max_views = max_views
        self.best_model_path = best_model_path
        self.eval_every_steps = eval_every_steps
        self.best_mean_path = math.inf
        self.history: List[Dict[str, float]] = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_every_steps != 0:
            return True
        mean_path = float(np.mean([evaluate_order_model(self.repository.get(seed), self.model, self.max_views)[1] for seed in self.eval_seeds]))
        self.history.append({"step": float(self.num_timesteps), "mean_path_length_m": mean_path})
        if mean_path + EPS < self.best_mean_path:
            self.best_mean_path = mean_path
            self.model.save(self.best_model_path)
        return True


def evaluate_order_model(
    scenario: PathScenario,
    model: MaskablePPO,
    max_views: int,
) -> Tuple[List[int], float]:
    env = PathOrderingEnv(PathScenarioRepository(Params(), EnvironmentConfig()), (), max_views)
    env.load_scenario(scenario)
    observation = env.observation()
    while True:
        masks = env.action_masks()
        if not np.any(masks):
            break
        action, _ = model.predict(observation, action_masks=masks, deterministic=True)
        observation, _, terminated, _, _ = env.step(int(action))
        if terminated:
            break
    return list(env.visit_order), env.cumulative_cost


def build_path_order_model(
    repository: PathScenarioRepository,
    training: PathPlanningTrainingConfig,
) -> Tuple[MaskablePPO, int]:
    max_views = max(len(repository.get(seed).selected) for seed in (*training.train_seeds, *training.eval_seeds, *training.test_seeds))

    def make_env():
        env = PathOrderingEnv(repository, training.train_seeds, max_views)
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
        policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
        seed=training.seed,
        verbose=0,
    )
    return model, max_views


def evaluate_methods(
    repository: PathScenarioRepository,
    training: PathPlanningTrainingConfig,
    model: MaskablePPO,
    max_views: int,
) -> List[Dict[str, float | str]]:
    evaluator = ReconstructionSuitabilityEvaluationAgent()
    path_agent = ThreeLayerPathPlanningAgent(model, max_views)
    rows: List[Dict[str, float | str]] = []
    for seed in training.test_seeds:
        scenario = repository.get(seed)
        baseline = build_baseline_plan(scenario)
        rl_path = path_agent.plan(scenario)
        for result in (baseline, rl_path):
            metrics = evaluator.evaluate(result, scenario.targets, scenario.buildings, Params(target_spacing=4.0, min_distance=3.0, max_distance=20.0, effective_quality_threshold=0.50))
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
                    "observation_redundancy": metrics["observation_redundancy"],
                }
            )
    return rows


def write_results(
    rows: Sequence[Dict[str, float | str]],
    history: Sequence[Dict[str, float]],
) -> None:
    with (OUTPUT_DIR / "anzhong_path_agent_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUTPUT_DIR / "anzhong_path_agent_training_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with (OUTPUT_DIR / "anzhong_path_agent_results.md").open("w", encoding="utf-8") as file:
        file.write("# 安中大楼：三层路径规划智能体原型\n\n")
        file.write("固定使用当前代码选中的视点集合，不修改候选视点生成与视点选取模块。\n")
        file.write("对比当前路径组织 baseline 与新的 `PPO 顺序 + A* + B-spline` 三层路径规划智能体。\n\n")
        file.write("| 方法 | 视点数 | 有效覆盖 | 最弱建筑覆盖 | 平均质量 | 路径长度 |\n")
        file.write("|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            file.write(
                f"| {row['method']} | {row['selected_views']} | "
                f"{float(row['effective_coverage']):.1%} | "
                f"{float(row['minimum_building_effective_coverage']):.1%} | "
                f"{float(row['average_best_quality']):.3f} | "
                f"{float(row['path_length_m']):.1f} m |\n"
            )


def run() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    params = Params(
        target_spacing=4.0,
        min_distance=3.0,
        max_distance=20.0,
        effective_quality_threshold=0.50,
    )
    config = EnvironmentConfig(
        max_steps=48,
        minimum_per_building_effective_coverage=0.55,
        q_eff=params.effective_quality_threshold,
        reward_effective_coverage=70.0,
        reward_quality=16.0,
        reward_scarcity=10.0,
        reward_fairness=12.0,
        penalty_path=0.34,
    )
    training = PathPlanningTrainingConfig()
    repository = PathScenarioRepository(params, config)
    model, max_views = build_path_order_model(repository, training)
    best_model_path = OUTPUT_DIR / "anzhong_path_order_ppo_best"
    callback = PathEvalCallback(
        repository,
        training.eval_seeds,
        max_views,
        best_model_path,
        training.eval_every_steps,
    )
    print("Training compact PPO path-ordering agent...")
    model.learn(total_timesteps=training.total_timesteps, callback=callback, progress_bar=False)
    if best_model_path.with_suffix(".zip").exists():
        model = MaskablePPO.load(best_model_path, env=model.get_env())

    rows = evaluate_methods(repository, training, model, max_views)
    write_results(rows, callback.history or [{"step": 0.0, "mean_path_length_m": 0.0}])

    scenario = repository.get(training.test_seeds[0])
    baseline = build_baseline_plan(scenario)
    path_agent = ThreeLayerPathPlanningAgent(model, max_views)
    learned = path_agent.plan(scenario)
    plot_plan(
        baseline,
        scenario.buildings,
        scenario.targets,
        scenario.selected,
        scenario.start,
        OUTPUT_DIR / "anzhong_path_agent_baseline_plan.png",
        params,
    )
    plot_plan(
        learned,
        scenario.buildings,
        scenario.targets,
        scenario.selected,
        scenario.start,
        OUTPUT_DIR / "anzhong_path_agent_three_layer_plan.png",
        params,
    )
    for row in rows:
        print(
            f"{row['method']}: effective={float(row['effective_coverage']):.1%}, "
            f"min-building={float(row['minimum_building_effective_coverage']):.1%}, "
            f"path={float(row['path_length_m']):.1f} m"
        )


if __name__ == "__main__":
    run()
