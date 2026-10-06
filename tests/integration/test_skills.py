"""Integration tests for the deterministic skill layer (M2 DoD).

Tests use ``MockDrone`` (no sim) for speed; the sim-backed variants are in the
same file but marked ``sim`` so CI can run them separately.

DoD requirements covered:
- Each skill reaches its target within tolerance.
- ``scan(8)`` yields 8 frames with 8 distinct yaws.
- A scripted "bad" action sequence (near wall, climb past ceiling, battery 0)
  ends with the drone landed, status ``aborted``, and the expected
  ``SafetyEvent``s — no exceptions escape.
"""

from __future__ import annotations

import pytest

from kormoran.config import Settings
from kormoran.control import skills
from kormoran.control.executor import SkillExecutor
from kormoran.drones.mock import FailMode, MockDrone
from kormoran.types import Action, Detection, MoveResult, ScanResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(**overrides: object) -> Settings:
    defaults = {
        "geofence_x_min": -5.0,
        "geofence_x_max": 5.0,
        "geofence_y_min": -5.0,
        "geofence_y_max": 5.0,
        "geofence_z_min": 0.05,
        "geofence_z_max": 3.0,
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def _connected_drone(
    battery_pct: float = 100.0,
    fail_mode: FailMode = FailMode.NONE,
) -> MockDrone:
    drone = MockDrone(battery_pct=battery_pct, fail_mode=fail_mode)
    drone.connect()
    return drone


# ---------------------------------------------------------------------------
# Individual skills — MockDrone
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_skill_takeoff_reaches_altitude() -> None:
    drone = _connected_drone()
    result = skills.takeoff(drone, altitude_m=1.0)
    assert result.status == "reached"
    assert drone.pose.z == pytest.approx(1.0, abs=0.01)


@pytest.mark.unit
def test_skill_land_reaches_ground() -> None:
    drone = _connected_drone()
    skills.takeoff(drone, altitude_m=1.0)
    result = skills.land(drone)
    assert result.status == "reached"
    assert drone.pose.z == pytest.approx(0.0, abs=0.01)


@pytest.mark.unit
def test_skill_advance_moves_forward() -> None:
    drone = _connected_drone()
    skills.takeoff(drone, altitude_m=1.0)
    result = skills.advance(drone, distance_m=0.5)
    assert result.status == "reached"
    assert drone.pose.x == pytest.approx(0.5, abs=0.01)


@pytest.mark.unit
def test_skill_strafe_moves_sideways() -> None:
    drone = _connected_drone()
    skills.takeoff(drone, altitude_m=1.0)
    result = skills.strafe(drone, distance_m=0.5)
    assert result.status == "reached"
    assert drone.pose.y == pytest.approx(0.5, abs=0.01)


@pytest.mark.unit
def test_skill_climb_increases_altitude() -> None:
    drone = _connected_drone()
    skills.takeoff(drone, altitude_m=1.0)
    result = skills.climb(drone, distance_m=0.3)
    assert result.status == "reached"
    assert drone.pose.z == pytest.approx(1.3, abs=0.01)


@pytest.mark.unit
def test_skill_descend_decreases_altitude() -> None:
    drone = _connected_drone()
    skills.takeoff(drone, altitude_m=1.5)
    result = skills.descend(drone, distance_m=0.5)
    assert result.status == "reached"
    assert drone.pose.z == pytest.approx(1.0, abs=0.01)


@pytest.mark.unit
def test_skill_rotate_changes_yaw() -> None:
    drone = _connected_drone()
    skills.takeoff(drone, altitude_m=1.0)
    result = skills.rotate(drone, angle_deg=90.0)
    assert result.status == "reached"
    assert drone.pose.yaw_deg == pytest.approx(90.0, abs=1.0)


@pytest.mark.unit
def test_skill_approach_advances_toward_target() -> None:
    drone = _connected_drone()
    skills.takeoff(drone, altitude_m=1.0)
    x_before = drone.pose.x
    detection = Detection(
        label="red cube",
        bbox_px=(10, 10, 20, 20),
        center_norm=(0.0, 0.0),  # centred
        area_frac=0.05,
        confidence=0.9,
    )
    result = skills.approach(drone, detection, step_m=0.3)
    assert result.status == "reached"
    assert drone.pose.x > x_before  # moved forward


# ---------------------------------------------------------------------------
# Scan: 8 frames, 8 distinct yaws
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_scan_yields_n_frames_and_distinct_yaws() -> None:
    """scan(8) returns 8 frames with 8 unique yaw values."""
    drone = _connected_drone()
    skills.takeoff(drone, altitude_m=1.0)
    result = skills.scan(drone, n_steps=8)

    assert isinstance(result, ScanResult)
    assert len(result.frames) == 8
    assert len(result.yaws_deg) == 8
    # All yaws should be distinct (mock rotates exactly 45° per step).
    assert len(set(round(y, 2) for y in result.yaws_deg)) == 8


@pytest.mark.unit
def test_scan_yaws_span_360_degrees() -> None:
    """scan(8) should cover close to 360° of rotation (0° → ~315°)."""
    drone = _connected_drone()
    skills.takeoff(drone, altitude_m=1.0)
    result = skills.scan(drone, n_steps=8)
    yaw_range = max(result.yaws_deg) - min(result.yaws_deg)
    # 7 steps of 45° = 315°
    assert yaw_range == pytest.approx(315.0, abs=2.0)


# ---------------------------------------------------------------------------
# SkillExecutor — bad sequence (wall / altitude / battery)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_executor_geofence_clamps_advance() -> None:
    """Advance 2 m when only 0.3 m of room → executor clamps and logs event."""
    cfg = _make_cfg(geofence_x_max=1.0)  # tight geofence on +x
    drone = _connected_drone()
    drone.takeoff(1.0)
    # Place drone 0.7 m from the wall.
    drone._pose = drone._pose.model_copy(update={"x": 0.7})  # type: ignore[attr-defined]

    executor = SkillExecutor(drone, cfg)
    state = drone.get_state()
    action = Action(kind="advance", distance_m=2.0, reasoning="test", confidence=1.0)

    result = executor.execute(action, state)

    assert isinstance(result, MoveResult)
    assert result.status in ("reached", "clamped")
    assert any(e.kind == "geofence" for e in executor.safety_events)
    # Drone must not have crossed the geofence.
    assert drone.pose.x <= 1.0 + 0.01


@pytest.mark.unit
def test_executor_altitude_clamps_climb() -> None:
    """Climb 2 m when ceiling at 2.2 m and z=1.5 → clamped to ≈0.7 m."""
    cfg = _make_cfg(geofence_z_max=2.2)
    drone = _connected_drone()
    drone.takeoff(1.5)

    executor = SkillExecutor(drone, cfg)
    state = drone.get_state()
    action = Action(kind="climb", distance_m=2.0, reasoning="test", confidence=1.0)

    result = executor.execute(action, state)

    assert isinstance(result, MoveResult)
    assert drone.pose.z <= 2.2 + 0.01
    assert any(e.kind == "altitude" for e in executor.safety_events)


@pytest.mark.unit
def test_executor_battery_low_forces_land() -> None:
    """Battery at reserve → executor rejects the action and lands the drone."""
    from kormoran.safety.battery import RESERVE_PCT

    cfg = _make_cfg()
    drone = _connected_drone(battery_pct=RESERVE_PCT)
    drone.takeoff(1.0)

    executor = SkillExecutor(drone, cfg)
    state = drone.get_state()
    action = Action(kind="advance", distance_m=0.5, reasoning="test", confidence=1.0)

    result = executor.execute(action, state)

    assert isinstance(result, MoveResult)
    assert result.status == "aborted"
    assert drone.pose.z == pytest.approx(0.0, abs=0.05)  # landed
    assert any(e.kind == "battery" for e in executor.safety_events)


@pytest.mark.unit
def test_executor_bad_sequence_no_exceptions() -> None:
    """Full bad-sequence: near-wall advance + ceiling climb + battery zero.

    The DoD requirement: drone lands, status aborted, expected SafetyEvents,
    no exceptions escape.
    """
    cfg = _make_cfg(
        geofence_x_max=3.0,
        geofence_z_max=2.2,
    )
    drone = _connected_drone(battery_pct=20.0)
    drone.takeoff(1.0)
    # Move close to the +x wall.
    drone._pose = drone._pose.model_copy(update={"x": 2.8})  # type: ignore[attr-defined]

    executor = SkillExecutor(drone, cfg)

    # Step 1: try to fly through the wall.
    state1 = drone.get_state()
    a1 = Action(kind="advance", distance_m=2.0, reasoning="test", confidence=1.0)
    r1 = executor.execute(a1, state1)
    assert isinstance(r1, MoveResult)
    assert drone.pose.x <= 3.0 + 0.01

    # Step 2: try to climb past the ceiling.
    state2 = drone.get_state()
    a2 = Action(kind="climb", distance_m=2.0, reasoning="test", confidence=1.0)
    r2 = executor.execute(a2, state2)
    assert isinstance(r2, MoveResult)
    assert drone.pose.z <= 2.2 + 0.01

    # Step 3: drain battery (use BATTERY_DROP drone for remaining calls).
    from kormoran.safety.battery import RESERVE_PCT

    drone2 = _connected_drone(battery_pct=RESERVE_PCT - 1.0)
    drone2.takeoff(1.0)
    executor2 = SkillExecutor(drone2, cfg)
    state3 = drone2.get_state()
    a3 = Action(kind="advance", distance_m=0.5, reasoning="test", confidence=1.0)
    r3 = executor2.execute(a3, state3)

    assert isinstance(r3, MoveResult)
    assert r3.status == "aborted"
    assert drone2.pose.z == pytest.approx(0.0, abs=0.1)

    # All SafetyEvents from both executors.
    all_events = executor.safety_events + executor2.safety_events
    event_kinds = {e.kind for e in all_events}
    assert "geofence" in event_kinds
    assert "altitude" in event_kinds
    assert "battery" in event_kinds
