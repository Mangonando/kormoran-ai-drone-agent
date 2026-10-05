"""Settings loaded from environment variables / .env file."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr
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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
