"""Settings loaded from environment variables / .env file, plus scenario config models."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM
    openrouter_api_key: SecretStr | None = Field(
        None,
        description="OpenRouter API key (required for LLM calls)",
    )
    llm_model: str = Field(
        "openai/gpt-4o-mini",
        description="LLM model id on OpenRouter",
    )
    llm_timeout_s: float = Field(30.0, description="LLM request timeout in seconds")

    # Simulator
    sim_gui: bool = Field(False, description="Show PyBullet GUI window")

    # Storage
    data_dir: str = Field(
        "./data",
        description="Root directory for SQLite, ChromaDB, recordings, results",
    )

    # Mission budgets
    max_llm_calls: int = Field(40, description="Max LLM calls per mission")
    max_cost_usd: float = Field(0.50, description="Max LLM cost per mission in USD")
    max_steps: int = Field(50, description="Max decision steps per mission")
    max_mission_s: float = Field(300.0, description="Max mission wall-clock duration in seconds")

    # Geofence defaults (metres, world frame)
    geofence_x_min: float = Field(-2.7, description="Geofence min X")
    geofence_x_max: float = Field(2.7, description="Geofence max X")
    geofence_y_min: float = Field(-2.7, description="Geofence min Y")
    geofence_y_max: float = Field(2.7, description="Geofence max Y")
    geofence_z_min: float = Field(0.1, description="Geofence min altitude (m)")
    geofence_z_max: float = Field(2.2, description="Geofence max altitude (m)")

    # Recording
    record_missions: bool = Field(True, description="Save JSONL mission log and GIF")
    log_level: str = Field("INFO", description="Python logging level")

    # Scenarios
    scenarios_dir: str = Field("configs/scenarios", description="Directory of scenario YAML files")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


# ---------------------------------------------------------------------------
# Scenario config models (loaded from YAML, not env vars)
# ---------------------------------------------------------------------------


class ObstacleSpec(BaseModel):
    """A static obstacle in the scene."""

    shape: Literal["box", "sphere", "cylinder"] = "box"
    size: list[float]  # box: [lx, ly, lz]; sphere: [r]; cylinder: [r, h]
    pose: list[float]  # [x, y, z] — world-frame centre


class ScenarioTargetSpec(BaseModel):
    """A target object to be placed (possibly randomised) in the scene."""

    label: str
    shape: Literal["cube", "sphere", "cylinder"] = "cube"
    color: str = "red"
    size: float = Field(0.2, gt=0.0)
    # zone: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    zone: list[list[float]]


class RoomBounds(BaseModel):
    """Axis-aligned bounding box of the room in metres."""

    x: list[float]  # [min, max]
    y: list[float]
    z: list[float]


class GeofenceOverride(BaseModel):
    """Per-scenario geofence bounds (metres, world frame).

    All four fields are required when the block is present so a partial
    override cannot silently leave one axis at the Settings default.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float


class ScenarioConfig(BaseModel):
    """Full scenario loaded from a YAML file."""

    scene_id: str
    room: RoomBounds
    geofence: GeofenceOverride | None = None  # per-scenario geofence override (M5)
    obstacles: list[ObstacleSpec] = []
    targets: list[ScenarioTargetSpec] = []
    start_pose: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])

    @classmethod
    def from_yaml(cls, path: str | Path) -> ScenarioConfig:
        """Load and validate a scenario YAML file."""
        with open(path) as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)
