"""Protocol definitions for Kormoran's dependency-injection boundaries.

These protocols are the *only* thing that agent, memory, perception, and UI code
may depend on.  Concrete backends (SimDrone, MockDrone, TelloDrone) live in
``kormoran.drones`` and are instantiated only in ``cli.py`` / config.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from kormoran.types import Action, Detection, DroneState, Frame, MoveResult, TargetSpec


@runtime_checkable
class DroneInterface(Protocol):
    """Minimal interface every drone backend must implement.

    All commands are **relative, body-frame, metres/degrees** so that the same
    agent code drives both the sim and a real Tello.
    """

    def connect(self) -> None:
        """Open the connection to the drone / simulator."""
        ...

    def disconnect(self) -> None:
        """Close the connection cleanly."""
        ...

    def takeoff(self, altitude_m: float) -> MoveResult:
        """Arm, spin up, and climb to *altitude_m* above the takeoff point."""
        ...

    def land(self) -> MoveResult:
        """Descend and disarm."""
        ...

    def emergency_stop(self) -> None:
        """Cut motors immediately (no return value — must never fail)."""
        ...

    def move_relative(
        self,
        dx_m: float,
        dy_m: float,
        dz_m: float,
        timeout_s: float,
    ) -> MoveResult:
        """Move *dx_m* forward, *dy_m* left, *dz_m* up in the drone body frame."""
        ...

    def rotate(self, yaw_deg: float, timeout_s: float) -> MoveResult:
        """Rotate *yaw_deg* degrees (positive = counter-clockwise from above)."""
        ...

    def get_state(self) -> DroneState:
        """Return the latest pose estimate, battery, and flight status."""
        ...

    def get_frame(self) -> Frame:
        """Capture and return the latest camera frame as RGB uint8 HxWx3."""
        ...


@runtime_checkable
class Perceiver(Protocol):
    """Detects targets in a camera frame."""

    def detect(self, frame: Frame, target: TargetSpec) -> list[Detection]:
        """Return all detections of *target* found in *frame*."""
        ...


@runtime_checkable
class Planner(Protocol):
    """Decides the next action given the current mission context."""

    def plan(self, ctx: object) -> Action:
        """Return the next validated ``Action``.  *ctx* is a ``PlanContext`` (M4)."""
        ...
