# Coarse Models

This directory contains the coarse 3D scene models currently registered for the Stage 1 pipeline.

## Layout

Each scene is stored under:

```text
coarse_models/<model_key>/coarse_model/
```

The directory may include:

- `coarse_scene.ply`
  primary coarse geometry used by the pipeline
- `coarse_scene.json`
  structured scene description when available
- `coarse_scene.obj`
  exchange geometry when available

## Available Models

| model_key | Scene | Notes |
|---|---|---|
| `anzhong_tower_single` | Anzhong tower single building | baseline single-building coarse model |
| `anzhong_tower_plus_north_teaching` | Anzhong tower plus north teaching buildings | campus-style scene built from 21 original blocks, merged into roughly 4 primary buildings |
| `anzhong_surrounding_buildings` | Anzhong surrounding buildings | compact multi-building comparison scene including Construction Lab, Anzhong Gate, Computing Center, and Clock Tower |

## Pipeline Entry Points

- List registered scenes:
  `python3 src/run_registered_full_pipeline.py --list-models`
- Run the full pipeline on one scene:
  `python3 src/run_registered_full_pipeline.py --model-key <model_key>`

## Usage Notes

- `anzhong_tower_single` is the main single-building baseline scene.
- `anzhong_tower_plus_north_teaching` is the preferred multi-building generalization scene.
- `anzhong_surrounding_buildings` is useful for shorter multi-building comparison runs.
