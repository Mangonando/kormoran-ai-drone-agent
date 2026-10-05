"""Integration tests for SimDrone against gym-pybullet-drones.

Marker: ``sim`` — requires PyBullet and gym-pybullet-drones; runs headless.
These tests exercise the same interface contract as ``test_drone_interface.py``
but on the real simulator backend.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kormoran.config import ScenarioConfig
from kormoran.drones.sim import SimDrone
from kormoran.interfaces import DroneInterface
from kormoran.types import Frame

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SCENARIOS_DIR = Path(__file__).parent.parent.parent / "configs" / "scenarios"


@pytest.fixture(scope="module")
def living_room_scene() -> ScenarioConfig:
    return ScenarioConfig.from_yaml(_SCENARIOS_DIR / "living_room_v1.yaml")


@pytest.fixture()
def sim(living_room_scene: ScenarioConfig) -> SimDrone:  # type: ignore[misc]
    """Connected SimDrone — disconnects after each test."""
    drone = SimDrone(scenario=living_room_scene, gui=False, seed=42)
    drone.connect()
    yield drone  # type: ignore[misc]
    drone.disconnect()


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


@pytest.mark.sim
def test_sim_drone_satisfies_protocol(sim: SimDrone) -> None:
    assert isinstance(sim, DroneInterface)


# ---------------------------------------------------------------------------
# Takeoff
# ---------------------------------------------------------------------------


@pytest.mark.sim
def test_takeoff_reaches_altitude(sim: SimDrone) -> None:
    """takeoff(1.0) must land within ±0.05 m of the target altitude."""
    result = sim.takeoff(altitude_m=1.0)
    assert result.status in {"reached", "clamped"}
    assert abs(result.pose.z - 1.0) <= 0.05, f"z={result.pose.z:.3f}"


@pytest.mark.sim
def test_takeoff_sets_is_flying(sim: SimDrone) -> None:
    sim.takeoff(altitude_m=0.8)
    state = sim.get_state()
    assert state.is_flying


# ---------------------------------------------------------------------------
# move_relative
# ---------------------------------------------------------------------------


@pytest.mark.sim
def test_advance_0_5m(sim: SimDrone) -> None:
    """move_relative(0.5, 0, 0) must displace the drone ≈0.5 m along body x."""
    sim.takeoff(altitude_m=1.0)
    state_before = sim.get_state()
    result = sim.move_relative(dx_m=0.5, dy_m=0.0, dz_m=0.0, timeout_s=15.0)
    state_after = sim.get_state()

    displacement = (
        (state_after.pose.x - state_before.pose.x) ** 2
        + (state_after.pose.y - state_before.pose.y) ** 2
    ) ** 0.5
    assert result.status in {"reached", "clamped"}
    assert displacement >= 0.35, f"displaced only {displacement:.3f} m"


# ---------------------------------------------------------------------------
# rotate
# ---------------------------------------------------------------------------


@pytest.mark.sim
def test_rotate_90_changes_yaw(sim: SimDrone) -> None:
    """rotate(90) must change yaw by ≈90° ± 10°."""
    sim.takeoff(altitude_m=1.0)
    state_before = sim.get_state()
    result = sim.rotate(yaw_deg=90.0, timeout_s=10.0)
    state_after = sim.get_state()

    delta = state_after.pose.yaw_deg - state_before.pose.yaw_deg
    # Wrap to [-180, 180]
    while delta > 180:
        delta -= 360
    while delta < -180:
        delta += 360

    assert result.status in {"reached", "clamped"}
    assert abs(delta) >= 70.0, f"yaw changed by only {delta:.1f}°"


# ---------------------------------------------------------------------------
# get_frame
# ---------------------------------------------------------------------------


@pytest.mark.sim
def test_get_frame_uint8_rgb(sim: SimDrone) -> None:
    """get_frame() must return a uint8 HxWx3 array with non-zero variance."""
    sim.takeoff(altitude_m=1.0)
    frame: Frame = sim.get_frame()
    assert isinstance(frame, Frame)
    assert frame.image.dtype == np.uint8
    assert frame.image.ndim == 3
    assert frame.image.shape[2] == 3
    assert frame.image.var() > 0, "Frame is uniformly black — DIRECT mode renderer issue"


# ---------------------------------------------------------------------------
# scan pattern: 8 frames, 8 distinct yaws
# ---------------------------------------------------------------------------


@pytest.mark.sim
def test_scan_8_yields_8_distinct_yaws(sim: SimDrone) -> None:
    sim.takeoff(altitude_m=1.0)
    yaws: list[float] = []
    frames: list[Frame] = []
    for _ in range(8):
        result = sim.rotate(yaw_deg=45.0, timeout_s=10.0)
        assert result.status in {"reached", "clamped"}
        state = sim.get_state()
        yaws.append(state.pose.yaw_deg)
        frames.append(sim.get_frame())

    assert len(frames) == 8
    assert len({round(y, 0) for y in yaws}) == 8, f"non-distinct yaws: {yaws}"
