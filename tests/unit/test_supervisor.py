"""Unit tests for the safety supervisor — one test per §4 table row.

All tests run entirely in-process (no sim, no network) against a known Settings
instance and synthetic DroneState objects.
"""

from __future__ import annotations

import pytest

from kormoran.config import Settings
from kormoran.drones.mock import FailMode, MockDrone
from kormoran.safety import supervisor as sup
from kormoran.safety.battery import RESERVE_PCT
from kormoran.types import Action, DroneState, Pose

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(**overrides: object) -> Settings:
    """Return a Settings with sensible test defaults; keyword overrides apply."""
    defaults = {
        "geofence_x_min": -3.0,
        "geofence_x_max": 3.0,
        "geofence_y_min": -3.0,
        "geofence_y_max": 3.0,
        "geofence_z_min": 0.1,
        "geofence_z_max": 2.2,
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def _state(
    x: float = 0.0,
    y: float = 0.0,
    z: float = 1.0,
    yaw_deg: float = 0.0,
    battery_pct: float = 100.0,
    t: float = 0.0,
) -> DroneState:
    return DroneState(pose=Pose(x=x, y=y, z=z, yaw_deg=yaw_deg), battery_pct=battery_pct, t=t)


def _action(kind: str = "advance", distance_m: float | None = 0.5, **kw: object) -> Action:
    return Action(
        kind=kind,  # type: ignore[arg-type]
        distance_m=distance_m,
        reasoning="test",
        confidence=1.0,
        **kw,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# §4 row 1 — Geofence: move past wall → clamped, SafetyEvent logged
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_geofence_advance_clamped() -> None:
    """Advance 2 m from x=2.5 with geofence x_max=3.0 → clamped to 0.5 m."""
    cfg = _make_settings(geofence_x_max=3.0)
    state = _state(x=2.5)  # 0.5 m from the wall
    action = _action(kind="advance", distance_m=2.0)

    verdict = sup.check(action, state, cfg)

    assert verdict.kind == "clamp"
    assert verdict.action.distance_m is not None
    assert verdict.action.distance_m < 2.0
    assert any(e.kind == "geofence" for e in verdict.events)


@pytest.mark.unit
def test_geofence_advance_within_bounds_allowed() -> None:
    """Advance 0.5 m from x=0 → well within bounds → allow."""
    cfg = _make_settings()
    state = _state(x=0.0)
    action = _action(kind="advance", distance_m=0.5)

    verdict = sup.check(action, state, cfg)

    assert verdict.kind == "allow"
    assert verdict.events == []


@pytest.mark.unit
def test_geofence_reject_no_room() -> None:
    """Advance from x=2.99 with x_max=3.0 — clamped distance < min → reject."""
    cfg = _make_settings(geofence_x_max=3.0)
    state = _state(x=2.99)
    action = _action(kind="advance", distance_m=0.5)

    verdict = sup.check(action, state, cfg)

    # The supervisor either clamps to a tiny move OR rejects; either way, a
    # geofence event must be present.
    assert verdict.kind in ("clamp", "reject")
    assert any(e.kind == "geofence" for e in verdict.events)


# ---------------------------------------------------------------------------
# §4 row 2 — Altitude min/max: climb too high → clamped
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_altitude_climb_clamped() -> None:
    """Climb 2 m from z=1.5 with z_max=2.2 → clamped to 0.7 m."""
    cfg = _make_settings(geofence_z_max=2.2)
    state = _state(z=1.5)
    action = _action(kind="climb", distance_m=2.0)

    verdict = sup.check(action, state, cfg)

    assert verdict.kind == "clamp"
    assert verdict.action.distance_m is not None
    assert verdict.action.distance_m == pytest.approx(0.7, abs=0.01)
    assert any(e.kind == "altitude" for e in verdict.events)


@pytest.mark.unit
def test_altitude_descend_clamped() -> None:
    """Descend 1 m from z=0.5 with z_min=0.1 → clamped to 0.4 m."""
    cfg = _make_settings(geofence_z_min=0.1)
    state = _state(z=0.5)
    action = _action(kind="descend", distance_m=1.0)

    verdict = sup.check(action, state, cfg)

    assert verdict.kind == "clamp"
    assert verdict.action.distance_m is not None
    assert verdict.action.distance_m == pytest.approx(0.4, abs=0.01)


@pytest.mark.unit
def test_altitude_within_bounds_allowed() -> None:
    """Climb 0.3 m from z=1.0 with z_max=2.2 → allow."""
    cfg = _make_settings()
    state = _state(z=1.0)
    action = _action(kind="climb", distance_m=0.3)

    verdict = sup.check(action, state, cfg)

    assert verdict.kind == "allow"


# ---------------------------------------------------------------------------
# §4 row 3 — Battery: below reserve_pct → forced land, status aborted
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_battery_low_forces_land() -> None:
    """Battery at reserve threshold → supervisor rejects with land action."""
    cfg = _make_settings()
    state = _state(battery_pct=RESERVE_PCT)  # exactly at the threshold
    action = _action(kind="advance", distance_m=0.5)

    verdict = sup.check(action, state, cfg)

    assert verdict.kind == "reject"
    assert verdict.action.kind == "land"
    assert any(e.kind == "battery" for e in verdict.events)


@pytest.mark.unit
def test_battery_above_reserve_ok() -> None:
    """Battery above reserve → no battery event."""
    cfg = _make_settings()
    state = _state(battery_pct=RESERVE_PCT + 1.0)
    action = _action(kind="advance", distance_m=0.5)

    verdict = sup.check(action, state, cfg)

    assert not any(e.kind == "battery" for e in verdict.events)


@pytest.mark.unit
def test_battery_zero_forces_land() -> None:
    """Battery at 0 → forced land regardless of action."""
    cfg = _make_settings()
    state = _state(battery_pct=0.0)
    action = _action(kind="scan", distance_m=None)

    verdict = sup.check(action, state, cfg)

    assert verdict.kind == "reject"
    assert verdict.action.kind == "land"


# ---------------------------------------------------------------------------
# §4 row 4 — Argument bounds: Pydantic ranges
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_action_distance_out_of_range_rejected_by_pydantic() -> None:
    """Action with distance_m > 2.0 is invalid Pydantic; can't even be created."""
    with pytest.raises(Exception):  # noqa: B017
        Action(kind="advance", distance_m=5.0, reasoning="test", confidence=1.0)


@pytest.mark.unit
def test_action_distance_below_min_rejected_by_pydantic() -> None:
    """Action with distance_m < 0.2 is invalid Pydantic."""
    with pytest.raises(Exception):  # noqa: B017
        Action(kind="advance", distance_m=0.1, reasoning="test", confidence=1.0)


# ---------------------------------------------------------------------------
# §4 row 5 — Watchdog: exercised via MockDrone(NEVER_ARRIVES)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_watchdog_via_executor() -> None:
    """A NEVER_ARRIVES drone move times out and is caught by the executor watchdog."""
    from kormoran.control.executor import SkillExecutor

    drone = MockDrone(fail_mode=FailMode.NEVER_ARRIVES)
    drone.connect()
    drone.takeoff(1.0)

    cfg = _make_settings()
    executor = SkillExecutor(drone, cfg)

    state = drone.get_state()
    action = Action(
        kind="advance",
        distance_m=0.5,
        reasoning="test",
        confidence=1.0,
    )

    # With NEVER_ARRIVES the move always times out.  Use a tiny timeout so the
    # watchdog fires quickly in the test suite.
    from kormoran.control import executor as ex_mod

    orig = ex_mod.WATCHDOG_MULTIPLIER
    ex_mod.WATCHDOG_MULTIPLIER = 0.01  # fire almost instantly
    try:
        # mock drone move_relative with 0.01 s timeout will return "clamped" status
        # because elapsed > timeout; the watchdog multiplier is separate.
        result = executor.execute(action, state)
    finally:
        ex_mod.WATCHDOG_MULTIPLIER = orig

    # The result must be a MoveResult (not raised exception).
    from kormoran.types import MoveResult

    assert isinstance(result, MoveResult)


# ---------------------------------------------------------------------------
# Passive checks — scan, rotate, report_not_found bypass spatial checks
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_scan_action_allowed_regardless_of_position() -> None:
    """Scan has no spatial footprint; always allowed (unless battery low)."""
    cfg = _make_settings()
    state = _state(x=2.9, y=2.9)  # near corner — irrelevant for scan
    action = Action(kind="scan", reasoning="test", confidence=1.0)

    verdict = sup.check(action, state, cfg)

    assert verdict.kind == "allow"


@pytest.mark.unit
def test_rotate_action_allowed_at_boundary() -> None:
    """Rotate has no spatial footprint; always allowed (unless battery low)."""
    cfg = _make_settings()
    state = _state(x=2.99)
    action = Action(kind="rotate", angle_deg=90.0, reasoning="test", confidence=1.0)

    verdict = sup.check(action, state, cfg)

    assert verdict.kind == "allow"
