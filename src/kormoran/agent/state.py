"""MissionState TypedDict and related helpers for the LangGraph mission loop."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Minimum area fraction for a detection to count as "found".
#: Camera is forward-facing at 1 m altitude; cube (z=0.1 m) first enters the
#: vertical FOV at ~2.2 m horizontal distance where area_frac ≈ 0.8-1 %.
#: Using 0.003 (0.3 %) lets the drone confirm the target from ~3 m, well within
#: the viewable range.  See docs/decisions.md for the camera-geometry analysis.
FOUND_AREA_FRAC: float = 0.003

#: Maximum |center_norm[0]| for a detection to count as centred.
FOUND_CENTER_TOL: float = 0.6

#: Number of consecutive frames with a qualifying detection to declare found.
FOUND_CONSECUTIVE: int = 1

#: Grid cell size in metres for visited-cell tracking.
GRID_CELL_SIZE_M: float = 0.5


# ---------------------------------------------------------------------------
# Typed sub-records
# ---------------------------------------------------------------------------


class StepRecord(TypedDict):
    step: int
    action_kind: str
    reasoning: str
    result_status: str
    detections_count: int
    pose_x: float
    pose_y: float
    pose_z: float


class GridCell(TypedDict):
    x_idx: int
    y_idx: int


class TokenUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    llm_calls: int
    est_cost_usd: float


# ---------------------------------------------------------------------------
# Mission state
# ---------------------------------------------------------------------------


class MissionState(TypedDict):
    mission_id: str
    scene_id: str
    target: dict  # TargetSpec.model_dump()
    step: int
    max_steps: int
    started_at: float
    drone: dict  # DroneState.model_dump()
    last_detections: list[dict]  # [Detection.model_dump()]
    observation: str
    recent_steps: Annotated[list[StepRecord], operator.add]
    history_summary: str
    visited: Annotated[list[GridCell], operator.add]
    recalled: list[dict]  # stub for M5
    status: str  # "planning"|"flying"|"found"|"not_found"|"aborted"
    safety_events: Annotated[list[dict], operator.add]
    usage: TokenUsage
    last_action: dict | None  # Action.model_dump()
    last_result: dict | None  # MoveResult.model_dump()
    consecutive_found: int
    last_verdict_kind: str  # "allow" | "clamp" | "reject" — set by validate_node each step
