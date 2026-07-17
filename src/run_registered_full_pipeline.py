#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from coarse_model_registry import REGISTERED_COARSE_MODELS, get_registered_model
from uav_coarse_3d_full_pipeline import main as full_pipeline_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full pipeline from a registered coarse model.")
    parser.add_argument("--model-key", type=str, default="anzhong_tower_single")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--max-generated-candidates", type=int, default=3600)
    parser.add_argument("--ppo-candidates", type=int, default=384)
    parser.add_argument("--selection-timesteps", type=int, default=2000)
    parser.add_argument("--selection-seed-sweep", type=str, default="17")
    parser.add_argument("--return-to-base", dest="return_to_base", action="store_true", default=True)
    parser.add_argument("--no-return-to-base", dest="return_to_base", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_models:
        for key in sorted(REGISTERED_COARSE_MODELS):
            model = REGISTERED_COARSE_MODELS[key]
            print(f"{key}: {model.description} [{model.coarse_model_dir}]")
        return
    model = get_registered_model(args.model_key)
    output_dir = args.output_dir or (
        Path("outputs") / f"{args.model_key}_full_pipeline"
    )
    import sys

    sys.argv = [
        "uav_coarse_3d_full_pipeline.py",
        str(model.mesh),
        "--scene-json",
        str(model.scene_json),
        "--output-dir",
        str(output_dir),
        "--max-targets",
        str(args.max_targets or model.default_max_targets),
        "--max-generated-candidates",
        str(args.max_generated_candidates),
        "--ppo-candidates",
        str(args.ppo_candidates),
        "--selection-timesteps",
        str(args.selection_timesteps),
        "--selection-seed-sweep",
        str(args.selection_seed_sweep),
        "--return-to-base" if args.return_to_base else "--no-return-to-base",
    ]
    full_pipeline_main()


if __name__ == "__main__":
    main()
