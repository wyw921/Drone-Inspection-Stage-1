# Anzhong Stage 1 3D

This branch contains the current stable snapshot of the Stage 1 drone inspection pipeline for coarse 3D building scenes.
It is intended as the primary public entry point for reading the code, reproducing experiments, and extending the workflow to additional coarse models.

## Scope

The pipeline focuses on viewpoint planning and safe route generation for coarse building models, with an emphasis on photogrammetry-friendly capture plans.

Core optimization targets include:

- surface coverage
- forward overlap within a station
- lateral overlap across neighboring stations
- capture quality indicators
- path length and clearance-aware safety

## Repository Contents

This branch includes:

- the current stable `src/` implementation
- three registered coarse 3D scene models
- a unified model registry and execution entry point
- representative outputs, plots, CSV summaries, and markdown reports

## Registered Coarse Models

The following scene models are available in this snapshot:

| model_key | Scene | Intended use |
|---|---|---|
| `anzhong_tower_single` | Anzhong tower single building | single-building baseline |
| `anzhong_tower_plus_north_teaching` | Anzhong tower plus north teaching buildings | multi-building generalization test |
| `anzhong_surrounding_buildings` | Anzhong surrounding buildings | compact campus-style comparison scene |

Model files are stored under:

```text
coarse_models/<model_key>/coarse_model/
```

See [`coarse_models/README.md`](./coarse_models/README.md) for model-level notes.

## Quick Start

List all registered models:

```bash
python3 src/run_registered_full_pipeline.py --list-models
```

Run the full pipeline on a selected scene:

```bash
python3 src/run_registered_full_pipeline.py --model-key anzhong_tower_single
```

## Recommended Entry Points

- [`src/run_registered_full_pipeline.py`](./src/run_registered_full_pipeline.py)
  Unified entry point for selecting a registered scene and launching the full pipeline.
- [`src/coarse_model_registry.py`](./src/coarse_model_registry.py)
  Registry for model keys, paths, and scene descriptions.
- [`src/uav_coarse_3d_full_pipeline.py`](./src/uav_coarse_3d_full_pipeline.py)
  End-to-end orchestration from candidate generation to export.
- [`src/uav_coarse_3d_viewpoint_rl.py`](./src/uav_coarse_3d_viewpoint_rl.py)
  Maskable PPO viewpoint-selection logic.
- [`src/uav_coarse_3d_planner.py`](./src/uav_coarse_3d_planner.py)
  Geometric preprocessing, candidate generation, and coverage-quality computation.

## Method Summary

The current Stage 1 workflow follows this structure:

1. Generate a relatively large candidate viewpoint pool from the coarse 3D scene.
2. Use a structured Maskable PPO action space:
   `viewpoint index + shot count + standoff layer`.
3. Optimize viewpoint selection jointly for:
   coverage, forward overlap, lateral overlap, quality metrics, and path cost.
4. Track quality through:
   `mean_quality_normalized`, `quality_good_fraction`, and `weakest_quality_normalized`.
5. Attempt safe direct or Dubins-like segment connections first, and fall back to 3D A* when clearance or collision constraints are violated.
6. Keep long-range motion piecewise linear and apply smoothing only where turn-radius feasibility is required.

## Branch Guide

- `main`
  Public landing page for the repository.
- `stable/anzhong-stage1-3d-current`
  Current stable snapshot for code reading and reproduction.
- `preserve-anzhong-stage1-3d-shot8-v1`
  Preserved earlier baseline branch for comparison.
- `codex/stable-snapshot-20260717`
  Raw backup snapshot retained during repository cleanup.

## Baseline and Release

Preserved baseline branch:
[`preserve-anzhong-stage1-3d-shot8-v1`](https://github.com/wyw921/Drone-Inspection-Stage-1/tree/preserve-anzhong-stage1-3d-shot8-v1)

Reference release:
[Anzhong Stage1 3D Preserve Fullscale Shot8 V1](https://github.com/wyw921/Drone-Inspection-Stage-1/releases/tag/anzhong-stage1-3d-preserve-fullscale-shot8-v1)

## Included Outputs

Representative outputs are provided in:

- `outputs/anzhong_capture_full_pipeline_mainline_seed17_nospiral/`
- `outputs/anzhong_selection_stabilized_formal/`

These directories include:

- path visualizations
- selected-capture CSV exports
- waypoint CSV exports
- PPO training history
- markdown and CSV result summaries

## Notes

- This GitHub snapshot prioritizes readability, reproducibility, and rollback-friendly structure.
- Larger historical training artifacts and full local workspaces are intentionally not mirrored in full on this branch.
