# Drone Inspection Stage 1

Drone Inspection Stage 1 is a research repository for UAV inspection planning and coarse-scene 3D reconstruction workflow design.
The current public codebase focuses on viewpoint planning, capture scheduling, and safe path generation for the Anzhong building and nearby campus-style coarse models.

## Overview

The main workflow is:

`candidate viewpoint generation -> Maskable PPO viewpoint selection -> route generation -> export and visualization`

The current optimization objectives emphasize:

- surface coverage
- forward overlap within a capture station
- lateral overlap across neighboring stations
- quality-oriented viewpoint selection
- path length and safety-aware routing

## Recommended Branches

This repository currently exposes two main public branches:

- [`stable/anzhong-stage1-3d-current`](https://github.com/wyw921/Drone-Inspection-Stage-1/tree/stable/anzhong-stage1-3d-current)
  Current stable snapshot for code reading, reproduction, and extension.
- [`preserve-anzhong-stage1-3d-shot8-v1`](https://github.com/wyw921/Drone-Inspection-Stage-1/tree/preserve-anzhong-stage1-3d-shot8-v1)
  Preserved earlier baseline branch retained for comparison.

The `main` branch serves as a public landing page and repository guide.

## Stable Snapshot Contents

The stable snapshot branch includes:

- the current stable `src/` implementation
- three registered coarse 3D scene models
- a unified registry and full-pipeline runner
- representative plots, CSV summaries, and markdown result reports

Stable snapshot link:
[`stable/anzhong-stage1-3d-current`](https://github.com/wyw921/Drone-Inspection-Stage-1/tree/stable/anzhong-stage1-3d-current)

## Registered Coarse Models

The stable snapshot currently provides three registered scenes:

- `anzhong_tower_single`
- `anzhong_tower_plus_north_teaching`
- `anzhong_surrounding_buildings`

Model details are documented in:
[`coarse_models/README.md`](https://github.com/wyw921/Drone-Inspection-Stage-1/blob/stable/anzhong-stage1-3d-current/coarse_models/README.md)

## Primary Code Entry Points

The most useful files for orientation are:

- [`src/run_registered_full_pipeline.py`](https://github.com/wyw921/Drone-Inspection-Stage-1/blob/stable/anzhong-stage1-3d-current/src/run_registered_full_pipeline.py)
- [`src/coarse_model_registry.py`](https://github.com/wyw921/Drone-Inspection-Stage-1/blob/stable/anzhong-stage1-3d-current/src/coarse_model_registry.py)
- [`src/uav_coarse_3d_full_pipeline.py`](https://github.com/wyw921/Drone-Inspection-Stage-1/blob/stable/anzhong-stage1-3d-current/src/uav_coarse_3d_full_pipeline.py)
- [`src/uav_coarse_3d_viewpoint_rl.py`](https://github.com/wyw921/Drone-Inspection-Stage-1/blob/stable/anzhong-stage1-3d-current/src/uav_coarse_3d_viewpoint_rl.py)
- [`src/uav_coarse_3d_planner.py`](https://github.com/wyw921/Drone-Inspection-Stage-1/blob/stable/anzhong-stage1-3d-current/src/uav_coarse_3d_planner.py)

## Method Summary

The current Stage 1 pipeline uses:

1. a relatively large candidate viewpoint pool generated from coarse 3D geometry
2. a structured Maskable PPO action design:
   `viewpoint index + shot count + standoff layer`
3. joint optimization over coverage, local overlap, quality metrics, and route cost
4. quality tracking through:
   `mean_quality_normalized`, `quality_good_fraction`, and `weakest_quality_normalized`
5. safety-aware route connection that prefers direct or Dubins-like connections and falls back to 3D A* when collision or clearance constraints are violated
6. local turn-radius smoothing rather than whole-path global smoothing

## Getting Started

List registered models:

```bash
python3 src/run_registered_full_pipeline.py --list-models
```

Run the full pipeline on one registered scene:

```bash
python3 src/run_registered_full_pipeline.py --model-key anzhong_tower_single
```

## Baseline and Release

Preserved baseline branch:
[`preserve-anzhong-stage1-3d-shot8-v1`](https://github.com/wyw921/Drone-Inspection-Stage-1/tree/preserve-anzhong-stage1-3d-shot8-v1)

Reference release:
[Anzhong Stage1 3D Preserve Fullscale Shot8 V1](https://github.com/wyw921/Drone-Inspection-Stage-1/releases/tag/anzhong-stage1-3d-preserve-fullscale-shot8-v1)

## Notes

- The GitHub version is curated for readability, reproducibility, and rollback-friendly structure.
- Larger historical training artifacts are intentionally not mirrored in full on the public stable branch.
