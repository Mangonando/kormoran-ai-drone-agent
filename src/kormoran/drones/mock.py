"""MockDrone — a scripted, in-process drone backend for unit tests.

It uses simple dead-reckoning kinematics (no real physics), returns synthetic
frames, and supports programmable failure modes to exercise error-handling code.
"""

from __future__ import annotations

import logging
import math
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray

from kormoran.types import DroneState, Frame, MoveResult, Pose

logger = logging.getLogger(__name__)

# Synthetic frame dimensions used for all fixture frames
_IMG_H: int = 48
_IMG_W: int = 64


class FailMode(Enum):
    """Programmable failure modes for testing error paths."""

    NONE = auto()
    NEVER_ARRIVES = auto()  # move_relative always times out
    BATTERY_DROP = auto()  # battery drains immediately to 0


def _make_synthetic_frame(t: float) -> NDArray[np.uint8]:
    """Generate a deterministic gradient frame that changes with *t*."""
    img = np.zeros((_IMG_H, _IMG_W, 3), dtype=np.uint8)
    # Horizontal gradient varying with t so sequential frames differ
    phase = int(t * 30) % 256
    img[:, :, 0] = np.linspace(phase, (phase + 128) % 256, _IMG_W, dtype=np.uint8)
    img[:, :, 1] = np.linspace(60, 200, _IMG_H, dtype=np.uint8).reshape(-1, 1)
    img[:, :, 2] = 100
    return img


class MockDrone:
    """Test-double drone that satisfies DroneInterface without any external deps.

    Parameters
    ----------
    fail_mode:
        Controls which operations fail (see ``FailMode``).
    battery_pct:
        Starting battery level (0–100).
    move_speed_m_s:
        Simulated movement speed for elapsed-time calculation.
    """

    def __init__(
        self,
        fail_mode: FailMode = FailMode.NONE,
        battery_pct: float = 100.0,
        move_speed_m_s: float = 1.0,
    ) -> None:
        self._fail_mode = fail_mode
        self._battery_pct = battery_pct
        self._move_speed = move_speed_m_s

        self._pose = Pose()
        self._is_flying = False
        self._connected = False
        self._t: float = 0.0

    # ------------------------------------------------------------------
    # DroneInterface implementation
    # ------------------------------------------------------------------

    def connect(self) -> None:
        self._connected = True
        logger.debug("MockDrone connected")

    def disconnect(self) -> None:
        self._connected = False
        logger.debug("MockDrone disconnected")

    def takeoff(self, altitude_m: float) -> MoveResult:
        self._require_connected()
        if self._fail_mode is FailMode.BATTERY_DROP:
            self._battery_pct = 0.0
        dist = abs(altitude_m - self._pose.z)
        elapsed = dist / self._move_speed
        self._t += elapsed
        self._pose = Pose(x=self._pose.x, y=self._pose.y, z=altitude_m, yaw_deg=self._pose.yaw_deg)
        self._is_flying = True
        logger.debug("MockDrone takeoff to %.2f m (%.3f s)", altitude_m, elapsed)
        return MoveResult(status="reached", pose=self._pose, elapsed_s=elapsed)

    def land(self) -> MoveResult:
        self._require_connected()
        elapsed = self._pose.z / self._move_speed
        self._t += elapsed
        self._pose = Pose(x=self._pose.x, y=self._pose.y, z=0.0, yaw_deg=self._pose.yaw_deg)
        self._is_flying = False
        return MoveResult(status="reached", pose=self._pose, elapsed_s=elapsed)

    def emergency_stop(self) -> None:
        self._is_flying = False
        logger.warning("MockDrone emergency stop")

    def move_relative(
        self,
        dx_m: float,
        dy_m: float,
        dz_m: float,
        timeout_s: float,
    ) -> MoveResult:
        self._require_connected()
        dist = math.sqrt(dx_m**2 + dy_m**2 + dz_m**2)
        elapsed = dist / self._move_speed
        if self._fail_mode is FailMode.NEVER_ARRIVES or elapsed > timeout_s:
            return MoveResult(
                status="clamped",
                pose=self._pose,
                elapsed_s=timeout_s,
                reason="timeout",
            )
        # Apply body-frame displacement
        yaw_rad = math.radians(self._pose.yaw_deg)
        dx_w = dx_m * math.cos(yaw_rad) - dy_m * math.sin(yaw_rad)
        dy_w = dx_m * math.sin(yaw_rad) + dy_m * math.cos(yaw_rad)
        self._pose = Pose(
            x=self._pose.x + dx_w,
            y=self._pose.y + dy_w,
            z=self._pose.z + dz_m,
            yaw_deg=self._pose.yaw_deg,
        )
        self._t += elapsed
        return MoveResult(status="reached", pose=self._pose, elapsed_s=elapsed)

    def rotate(self, yaw_deg: float, timeout_s: float) -> MoveResult:  # noqa: ARG002
        self._require_connected()
        elapsed = abs(yaw_deg) / 90.0  # 90 deg/s notional rate
        self._pose = Pose(
            x=self._pose.x,
            y=self._pose.y,
            z=self._pose.z,
            yaw_deg=(self._pose.yaw_deg + yaw_deg) % 360.0,
        )
        self._t += elapsed
        return MoveResult(status="reached", pose=self._pose, elapsed_s=elapsed)

    def get_state(self) -> DroneState:
        self._require_connected()
        if self._fail_mode is FailMode.BATTERY_DROP:
            self._battery_pct = 0.0
        return DroneState(
            pose=self._pose,
            battery_pct=max(0.0, self._battery_pct),
            is_flying=self._is_flying,
            t=self._t,
        )

    def get_frame(self) -> Frame:
        self._require_connected()
        img: NDArray[np.uint8] = _make_synthetic_frame(self._t)
        return Frame(image=img, t=self._t, fov_deg=60.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_connected(self) -> None:
        if not self._connected:
            msg = "MockDrone is not connected — call connect() first"
            raise RuntimeError(msg)

    # Allow attribute access for tests that inspect internal state
    @property
    def pose(self) -> Pose:
        return self._pose

    @property
    def is_flying(self) -> bool:
        return self._is_flying

    @property
    def t(self) -> float:
        return self._t

    def __repr__(self) -> str:
        return f"MockDrone(fail_mode={self._fail_mode}, flying={self._is_flying})"
