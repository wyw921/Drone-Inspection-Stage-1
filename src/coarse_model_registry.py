from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parents[1]
COARSE_MODELS_DIR = ROOT / "coarse_models"


@dataclass(frozen=True)
class RegisteredCoarseModel:
    key: str
    coarse_model_dir: Path
    scene_type: str
    description: str
    default_max_targets: int = 900

    @property
    def mesh(self) -> Path:
        return self.coarse_model_dir / "coarse_scene.ply"

    @property
    def scene_json(self) -> Path:
        return self.coarse_model_dir / "coarse_scene.json"


REGISTERED_COARSE_MODELS: Dict[str, RegisteredCoarseModel] = {
    "anzhong_tower_single": RegisteredCoarseModel(
        key="anzhong_tower_single",
        coarse_model_dir=COARSE_MODELS_DIR / "anzhong_tower_single" / "coarse_model",
        scene_type="single",
        description="Anzhong tower single-building coarse model.",
    ),
    "anzhong_tower_plus_north_teaching": RegisteredCoarseModel(
        key="anzhong_tower_plus_north_teaching",
        coarse_model_dir=COARSE_MODELS_DIR / "anzhong_tower_plus_north_teaching" / "coarse_model",
        scene_type="campus",
        description="Anzhong tower plus north teaching buildings campus scene.",
    ),
    "anzhong_surrounding_buildings": RegisteredCoarseModel(
        key="anzhong_surrounding_buildings",
        coarse_model_dir=COARSE_MODELS_DIR / "anzhong_surrounding_buildings" / "coarse_model",
        scene_type="multi",
        description="Anzhong surrounding buildings comparison scene.",
    ),
}


def get_registered_model(model_key: str) -> RegisteredCoarseModel:
    if model_key not in REGISTERED_COARSE_MODELS:
        available = ", ".join(sorted(REGISTERED_COARSE_MODELS))
        raise KeyError(f"Unknown model_key={model_key!r}. Available: {available}")
    model = REGISTERED_COARSE_MODELS[model_key]
    if not model.mesh.exists():
        raise FileNotFoundError(f"Missing mesh: {model.mesh}")
    if not model.scene_json.exists():
        raise FileNotFoundError(f"Missing scene json: {model.scene_json}")
    return model
