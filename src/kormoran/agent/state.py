"""MissionState TypedDict and related helpers for the LangGraph mission loop."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Minimum area fraction for a detection to count as "found".
#: Camera FOV=60°, drone at z=1.0 m, cube center at z=0.1 m (size=0.2 m).
#: The cube first enters the vertical FOV at ~1.4 m horizontal distance and
#: peaks at ~0.8 % area_frac around 2 m — it cannot reach 1 % from a
#: forward-facing camera while targets are at floor level (z=0.1 m).
#: 0.005 (0.5 %) is the highest reliably achievable value; tighter thresholds
#: require elevating targets to z=0.5 m.
FOUND_AREA_FRAC: float = 0.005

#: Maximum |center_norm[0]| for a detection to count as centred.
#: Tightened from 0.6 → 0.3: cube must be near the horizontal frame centre.
FOUND_CENTER_TOL: float = 0.3

#: Number of consecutive qualifying frames needed before the land reflex may fire.
#: One confirmed frame is enough — the land reflex (REFLEX_LAND_FRAC) then closes the gap.
FOUND_CONSECUTIVE: int = 1

#: Grid cell size in metres for visited-cell tracking.
GRID_CELL_SIZE_M: float = 0.5

# ---------------------------------------------------------------------------
# Approach reflex constants
# ---------------------------------------------------------------------------

#: Minimum area_frac for a detection to trigger the approach reflex.
#: Below this threshold the LLM is consulted instead — it can decide whether
#: to approach, scan, or navigate around obstacles.  0.002 (0.2%) ensures
#: the descent-then-approach reflex fires as soon as a target is detected,
#: preventing the drone from approaching at cruising altitude and losing
#: the target out the bottom of the camera FOV.
REFLEX_AREA_FRAC: float = 0.002

#: Drone altitude to descend to before approaching a detected target.
#: Slightly above the standard target height (0.5 m) so the target is near
#: eye level and fills a meaningful portion of the frame at close range.
REFLEX_ALTITUDE_M: float = 0.6

#: area_frac threshold that triggers an automatic land reflex.
#: At z=0.6 m flight height and target z=0.5 m, a 0.2 m cube fills ~3 % of
#: the frame at ~0.5 m horizontal distance — drone is essentially on top of it.
REFLEX_LAND_FRAC: float = 0.03

#: Minimum safe flight altitude.  If the drone's z drops below this value the
#: plan_node fires a takeoff recovery reflex before any other decision.
#: Chosen below the approach altitude (0.6 m) so normal descend → approach
#: sequences never trigger it, but a physics-induced ground-crash does.
MIN_FLIGHT_Z_M: float = 0.3


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
