"""Interface-conformance test suite for DroneInterface backends.

These tests run only against ``MockDrone`` (marker: unit, no sim, no network).
The same parameterised fixture will later include ``SimDrone`` under the ``sim``
marker so every backend is held to identical expectations.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from kormoran.drones.mock import FailMode, MockDrone
from kormoran.interfaces import DroneInterface
from kormoran.types import Frame, MoveResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_mock() -> MockDrone:
    """Return a freshly connected MockDrone."""
    drone = MockDrone()
    drone.connect()
    return drone


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mock_drone_satisfies_protocol() -> None:
    """MockDrone must be structurally compatible with DroneInterface."""
    drone = MockDrone()
    assert isinstance(drone, DroneInterface)


# ---------------------------------------------------------------------------
# connect / disconnect
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_connect_disconnect() -> None:
    drone = MockDrone()
    drone.connect()
    drone.disconnect()
    # Reconnecting after disconnect should not raise
    drone.connect()
    drone.disconnect()


@pytest.mark.unit
def test_operations_require_connection() -> None:
    drone = MockDrone()
    with pytest.raises(RuntimeError, match="not connected"):
        drone.get_state()


# ---------------------------------------------------------------------------
# takeoff / land
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_takeoff_reaches_altitude() -> None:
    drone = make_mock()
    result = drone.takeoff(altitude_m=1.0)
    assert result.status == "reached"
    assert abs(result.pose.z - 1.0) < 0.01
    assert drone.is_flying


@pytest.mark.unit
def test_land_returns_to_ground() -> None:
    drone = make_mock()
    drone.takeoff(altitude_m=1.0)
    result = drone.land()
    assert result.status == "reached"
    assert result.pose.z == pytest.approx(0.0, abs=0.01)
    assert not drone.is_flying


# ---------------------------------------------------------------------------
# move_relative
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_move_forward() -> None:
    drone = make_mock()
    drone.takeoff(altitude_m=1.0)
    result = drone.move_relative(dx_m=0.5, dy_m=0.0, dz_m=0.0, timeout_s=10.0)
    assert result.status == "reached"
    assert abs(result.pose.x - 0.5) < 0.01


@pytest.mark.unit
def test_move_uses_body_frame() -> None:
    """After a 90° CCW rotation, 'forward' (dx=1) should map to world +y."""
    drone = make_mock()
    drone.takeoff(altitude_m=1.0)
    drone.rotate(yaw_deg=90.0, timeout_s=5.0)
    result = drone.move_relative(dx_m=1.0, dy_m=0.0, dz_m=0.0, timeout_s=10.0)
    assert result.status == "reached"
    # World x should be ~0, world y should be ~1
    assert abs(result.pose.x) < 0.05
    assert abs(result.pose.y - 1.0) < 0.05


@pytest.mark.unit
def test_move_timeout_returns_clamped() -> None:
    drone = MockDrone(fail_mode=FailMode.NEVER_ARRIVES)
    drone.connect()
    drone.takeoff(altitude_m=1.0)
    result = drone.move_relative(dx_m=0.5, dy_m=0.0, dz_m=0.0, timeout_s=0.01)
    assert result.status == "clamped"
    assert result.reason == "timeout"


# ---------------------------------------------------------------------------
# rotate
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rotate_changes_yaw() -> None:
    drone = make_mock()
    drone.takeoff(altitude_m=1.0)
    result = drone.rotate(yaw_deg=90.0, timeout_s=5.0)
    assert result.status == "reached"
    assert abs(result.pose.yaw_deg - 90.0) < 1.0


@pytest.mark.unit
def test_rotate_accumulates() -> None:
    drone = make_mock()
    drone.takeoff(altitude_m=1.0)
    for _ in range(4):
        drone.rotate(yaw_deg=90.0, timeout_s=5.0)
    state = drone.get_state()
    assert abs(state.pose.yaw_deg % 360.0) < 1.0


# ---------------------------------------------------------------------------
# get_state
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_state_returns_drone_state() -> None:
    drone = make_mock()
    drone.takeoff(altitude_m=1.5)
    state = drone.get_state()
    assert state.is_flying
    assert abs(state.pose.z - 1.5) < 0.01
    assert 0.0 <= state.battery_pct <= 100.0
    assert state.t >= 0.0


@pytest.mark.unit
def test_battery_drop_mode() -> None:
    drone = MockDrone(fail_mode=FailMode.BATTERY_DROP)
    drone.connect()
    state = drone.get_state()
    assert state.battery_pct == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# get_frame
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_frame_returns_valid_frame() -> None:
    drone = make_mock()
    frame = drone.get_frame()
    assert isinstance(frame, Frame)
    assert frame.image.dtype == np.uint8
    assert frame.image.ndim == 3
    assert frame.image.shape[2] == 3  # RGB
    assert frame.image.var() > 0  # non-uniform


@pytest.mark.unit
def test_frames_differ_over_time() -> None:
    """Sequential frames should not be identical (t changes)."""
    drone = make_mock()
    drone.takeoff(altitude_m=1.0)
    f1 = drone.get_frame()
    drone.move_relative(dx_m=0.5, dy_m=0.0, dz_m=0.0, timeout_s=5.0)
    f2 = drone.get_frame()
    assert not np.array_equal(f1.image, f2.image)


# ---------------------------------------------------------------------------
# emergency_stop
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_emergency_stop_sets_not_flying() -> None:
    drone = make_mock()
    drone.takeoff(altitude_m=1.0)
    assert drone.is_flying
    drone.emergency_stop()
    assert not drone.is_flying


# ---------------------------------------------------------------------------
# scan-like pattern (8 rotations capturing a frame each)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_scan_pattern_captures_8_distinct_yaws() -> None:
    drone = make_mock()
    drone.takeoff(altitude_m=1.0)

    yaws: list[float] = []
    for _ in range(8):
        drone.rotate(yaw_deg=45.0, timeout_s=5.0)
        state = drone.get_state()
        yaws.append(state.pose.yaw_deg)

    assert len(yaws) == 8
    # All yaws should be distinct
    assert len({round(y, 1) for y in yaws}) == 8

    # After 8 × 45° the drone should be back to ~360° = ~0°
    final_yaw_wrapped = drone.get_state().pose.yaw_deg % 360.0
    assert abs(final_yaw_wrapped) < 1.0 or abs(final_yaw_wrapped - 360.0) < 1.0


# ---------------------------------------------------------------------------
# MoveResult schema
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_move_result_has_valid_pose() -> None:
    drone = make_mock()
    result: MoveResult = drone.takeoff(altitude_m=0.8)
    assert result.pose.z == pytest.approx(0.8)
    assert math.isfinite(result.elapsed_s)
    assert result.elapsed_s >= 0.0
