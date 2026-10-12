"""Pydantic models for long-term memory (M5).

EpisodicRecord — one row per mission (SQLite).
MemoryNote     — one semantic note (ChromaDB document).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EpisodicRecord(BaseModel):
    """Summary of a completed mission stored in the episodic SQLite store.

    Intentionally excludes raw frames/images — only structured data.
    """

    mission_id: str
    ts: str  # ISO-8601 UTC timestamp
    scene_id: str
    target_label: str
    outcome: Literal["found", "not_found", "timeout", "error"]
    steps: int
    duration_s: float
    path: list[dict]  # list of {x, y, z, step} — no images
    found_pose: dict | None = None  # {x, y, z, yaw_deg} if outcome=="found"
    found_from_pose: dict | None = None  # pose from which target was first detected
    model: str = ""
    tokens: int = 0
    cost_usd: float = 0.0


class MemoryNote(BaseModel):
    """A single semantic memory note stored in ChromaDB.

    kind values:
    - "location"  — where a target was found / not found
    - "strategy"  — what navigation tactic worked / failed
    - "scene"     — general scene structure (walls, gaps, layout)
    """

    note_id: str  # UUID
    scene_id: str
    target_label: str
    kind: Literal["location", "strategy", "scene"]
    text: str = Field(..., description="Natural-language note text (used as embedding input)")
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    created_at: str  # ISO-8601 UTC
    mission_id: str
