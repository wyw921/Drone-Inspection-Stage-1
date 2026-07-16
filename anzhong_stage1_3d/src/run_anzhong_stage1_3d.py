"""Convenience runner for the Anzhong stage-1 coarse-model outputs.

This keeps the current stage focused on:
1. coarse-model generation provides the 3D proxy mesh;
2. our planner upgrades candidate generation to surface-normal + partition/cluster;
3. planner exports one-shot coverage and route metrics for comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from uav_coarse_3d_planner import plan


DEFAULT_COARSE_MODEL_DIR = Path(
    "/Users/wyw/Desktop/无人机建筑巡检系统_完整程序包_20260702_232635/"
    "agentic_inspection_pipeline/example_result/coarse_model"
)


def build_args(coarse_model_dir: Path, output_dir: Path, max_targets: int) -> argparse.Namespace:
    mesh = coarse_model_dir / "coarse_scene.ply"
    scene_json = coarse_model_dir / "coarse_scene.json"
    return argparse.Namespace(
        mesh=mesh,
        output_dir=output_dir,
        scene_json=scene_json if scene_json.exists() else None,
        max_targets=max_targets,
        azimuths=24,
        levels=3,
        required_views=2,
        stand_off=None,
        min_standoff=None,
        max_standoff=None,
        min_altitude=2.0,
        max_altitude=120.0,
        fov=75.0,
        max_incidence=72.0,
        quality_threshold=0.18,
        route_restarts=32,
        route_proposals=1200,
        turn_weight=2.0,
        metric_scale_known=True,
    )


def main() -> None:
    cli = argparse.ArgumentParser(description="Run the Anzhong stage-1 3D planning prototype.")
    cli.add_argument(
        "--coarse-model-dir",
        "--partner-dir",
        dest="coarse_model_dir",
        type=Path,
        default=DEFAULT_COARSE_MODEL_DIR,
    )
    cli.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/anzhong_stage1_3d"),
    )
    cli.add_argument("--max-targets", type=int, default=1200)
    args = cli.parse_args()
    planner_args = build_args(args.coarse_model_dir, args.output_dir, args.max_targets)
    plan(planner_args)


if __name__ == "__main__":
    main()
