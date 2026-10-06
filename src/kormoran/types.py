"""Core domain types shared across all Kormoran modules."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field


class Pose(BaseModel):
    """World-frame position + yaw."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw_deg: float = 0.0


class DroneState(BaseModel):
    """Snapshot of drone state returned by every DroneInterface backend."""

    pose: Pose
    battery_pct: float = Field(100.0, ge=0.0, le=100.0)
    is_flying: bool = False
    t: float = 0.0  # seconds since connect()


class Frame(BaseModel):
    """A single RGB camera frame."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: NDArray[np.uint8]  # H x W x 3, uint8 RGB
    t: float  # seconds since connect()
    fov_deg: float = 60.0


class Detection(BaseModel):
    """One detected object in a frame."""

    label: str
    bbox_px: tuple[int, int, int, int]  # x, y, w, h (top-left origin)
    center_norm: tuple[float, float]  # horizontal/vertical in [-1, 1]; (0,0) = frame centre
    area_frac: float = Field(..., ge=0.0, le=1.0)  # fraction of frame area
    confidence: float = Field(..., ge=0.0, le=1.0)


class TargetSpec(BaseModel):
    """What the agent is looking for."""

    label: str
    color: str | None = None
    shape: str | None = None


class Action(BaseModel):
    """The validated action the LLM (or any Planner) may return."""

    kind: Literal[
        "scan",
        "advance",
        "strafe",
        "climb",
        "descend",
        "rotate",
        "approach",
        "land",
        "report_not_found",
    ]
    distance_m: float | None = Field(None, ge=0.2, le=2.0)
    angle_deg: float | None = Field(None, ge=-180.0, le=180.0)
    reasoning: str = Field(..., max_length=300)
    confidence: float = Field(..., ge=0.0, le=1.0)


class MoveResult(BaseModel):
    """Outcome of a single skill execution."""

    status: Literal["reached", "clamped", "aborted"]
    pose: Pose
    elapsed_s: float
    reason: str | None = None


class SafetyEvent(BaseModel):
    """A veto or constraint applied by the SafetySupervisor."""

    t: float
    kind: str  # e.g. "geofence", "altitude", "battery", "watchdog"
    action_kind: str | None = None
    reason: str


class ScanResult(BaseModel):
    """Result of a multi-step scan skill (one frame per angular step)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    frames: list[Frame]
    yaws_deg: list[float]  # drone yaw at the moment each frame was captured
    elapsed_s: float
    final_pose: Pose


class Verdict(BaseModel):
    """Safety supervisor decision on a proposed action."""

    kind: Literal["allow", "clamp", "reject"]
    action: Action  # original if allowed, clamped if clamped, land if rejected
    reason: str
    events: list[SafetyEvent] = Field(default_factory=list)
