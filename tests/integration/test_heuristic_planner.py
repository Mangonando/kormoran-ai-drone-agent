"""Integration test: HeuristicPlanner finds red cube in living_room_v1.

Requires a live PyBullet environment (marker ``sim``).
DoD: ≥ 2/3 of tested seeds reach "found" (lighter than the full 8/10 DoD run).
"""

from __future__ import annotations

import pytest

from kormoran.agent.graph import build_mission_graph, make_initial_state, new_mission_id
from kormoran.agent.planners import HeuristicPlanner
from kormoran.config import ScenarioConfig, get_settings
from kormoran.drones.sim import SimDrone
from kormoran.perception.color_detector import ColorDetector
from kormoran.types import TargetSpec

_SCENARIO = "living_room_v1"
_TARGET_LABEL = "red cube"
_MAX_STEPS = 50
_SEEDS = [1, 2, 3]
_MIN_FOUND = 2  # DoD: ≥ 2/3 seeds


@pytest.mark.sim
def test_heuristic_finds_red_cube_in_majority_of_seeds() -> None:
    """HeuristicPlanner finds 'red cube' in ≥ 2/3 seeds of living_room_v1."""
    settings = get_settings()
    scene = ScenarioConfig.from_yaml(f"configs/scenarios/{_SCENARIO}.yaml")

    target_spec: TargetSpec | None = None
    for t in scene.targets:
        if t.label == _TARGET_LABEL:
            target_spec = TargetSpec(label=t.label, color=t.color, shape=t.shape)
            break
    assert target_spec is not None, f"{_TARGET_LABEL!r} not in scenario {_SCENARIO!r}"

    found_count = 0
    for seed in _SEEDS:
        drone = SimDrone(scenario=scene, gui=False, seed=seed)
        planner = HeuristicPlanner()
        perceiver = ColorDetector()

        drone.connect()
        try:
            drone.takeoff(altitude_m=1.0)
            drone_state = drone.get_state()

            mission_id = new_mission_id()
            graph = build_mission_graph(drone, planner, perceiver, settings)
            initial = make_initial_state(
                mission_id, scene.scene_id, target_spec, drone_state, _MAX_STEPS
            )
            thread_config = {"configurable": {"thread_id": mission_id}}
            final = graph.invoke(initial, config=thread_config)
            status = final.get("status", "unknown")
            steps = final.get("step", 0)
            print(f"  seed={seed} status={status} steps={steps}")
            if status == "found":
                found_count += 1
        finally:
            try:
                drone.land()
            except Exception:  # noqa: BLE001, S110
                pass
            drone.disconnect()

    print(f"\nHeuristic planner: {found_count}/{len(_SEEDS)} seeds found")
    assert found_count >= _MIN_FOUND, (
        f"HeuristicPlanner found target in only {found_count}/{len(_SEEDS)} seeds "
        f"(need ≥ {_MIN_FOUND})"
    )
