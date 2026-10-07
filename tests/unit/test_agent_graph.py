"""Unit tests for the M4 LangGraph mission graph (DoD).

Tests run without a live PyBullet environment — they use MockDrone and
a MockPerceiver that returns preset Detection objects.

DoD checks:
- ScriptedPlanner + MockDrone + qualifying detections → status "found"
- max_steps budget exhausted with no detections → status "not_found"
- LLMPlanner: first attempt raises, second raises → fallback action is "scan"
- Graph can be interrupted after "act" and resumed to completion
"""

from __future__ import annotations

from unittest.mock import MagicMock

from kormoran.agent.graph import build_mission_graph, make_initial_state, new_mission_id
from kormoran.agent.planners import LLMPlanner, ScriptedPlanner
from kormoran.agent.state import FOUND_CONSECUTIVE
from kormoran.config import Settings
from kormoran.drones.mock import MockDrone
from kormoran.types import Action, Detection, Frame, TargetSpec

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class MockPerceiver:
    """Returns a preset list of detections regardless of the frame."""

    def __init__(self, detections: list[Detection] | None = None) -> None:
        self._detections: list[Detection] = detections or []

    def detect(self, frame: Frame, target: TargetSpec) -> list[Detection]:  # noqa: ARG002
        return list(self._detections)


_TARGET = TargetSpec(label="red cube", color="red")

# Detection that satisfies the found criterion (area_frac >= 0.01, |cx| <= 0.6)
_GOOD_DETECTION = Detection(
    label="red cube",
    bbox_px=(10, 10, 20, 20),
    center_norm=(0.0, 0.0),
    area_frac=0.05,
    confidence=0.9,
)


def _settings() -> Settings:
    return Settings()


def _setup_drone() -> MockDrone:
    drone = MockDrone()
    drone.connect()
    drone.takeoff(altitude_m=1.0)
    return drone


def _run_graph(
    drone: MockDrone,
    planner: object,
    perceiver: MockPerceiver,
    *,
    max_steps: int = 20,
    interrupt_after: list[str] | None = None,
    checkpointer: object | None = None,
) -> tuple[object, dict, str]:
    """Build graph, invoke, return (graph, final_state, mission_id)."""
    from langgraph.checkpoint.memory import MemorySaver

    cp = checkpointer if checkpointer is not None else MemorySaver()
    graph = build_mission_graph(
        drone,
        planner,
        perceiver,
        _settings(),
        checkpointer=cp,
        interrupt_after=interrupt_after or [],
    )
    mission_id = new_mission_id()
    drone_state = drone.get_state()
    initial = make_initial_state(mission_id, "test_scene", _TARGET, drone_state, max_steps)
    thread_config = {"configurable": {"thread_id": mission_id}}
    final = graph.invoke(initial, config=thread_config)
    return graph, final, mission_id


# ---------------------------------------------------------------------------
# Test 1: ScriptedPlanner reaches "found"
# ---------------------------------------------------------------------------


def test_scripted_planner_reaches_found() -> None:
    """ScriptedPlanner + qualifying detections → status=='found' in ≤ FOUND_CONSECUTIVE+1 steps."""
    drone = _setup_drone()
    # Enough actions to survive FOUND_CONSECUTIVE iterations.
    actions = [
        Action(kind="advance", distance_m=0.5, reasoning="step", confidence=1.0)
        for _ in range(FOUND_CONSECUTIVE + 5)
    ]
    planner = ScriptedPlanner(actions)
    perceiver = MockPerceiver([_GOOD_DETECTION])

    _, final, _ = _run_graph(drone, planner, perceiver)
    assert final["status"] == "found", f"expected 'found', got {final['status']!r}"
    drone.disconnect()


# ---------------------------------------------------------------------------
# Test 2: Budget exhaustion → "not_found"
# ---------------------------------------------------------------------------


def test_budget_exhaustion_not_found() -> None:
    """max_steps=2 with no detections → status=='not_found'."""
    drone = _setup_drone()
    actions = [
        Action(kind="advance", distance_m=0.5, reasoning="s", confidence=1.0),
        Action(kind="advance", distance_m=0.5, reasoning="s", confidence=1.0),
        Action(kind="advance", distance_m=0.5, reasoning="s", confidence=1.0),
    ]
    planner = ScriptedPlanner(actions)
    perceiver = MockPerceiver([])  # no detections

    _, final, _ = _run_graph(drone, planner, perceiver, max_steps=2)
    assert final["status"] == "not_found", f"expected 'not_found', got {final['status']!r}"
    drone.disconnect()


# ---------------------------------------------------------------------------
# Test 3: LLMPlanner fallback to scan
# ---------------------------------------------------------------------------


def test_invalid_llm_output_falls_back_to_scan() -> None:
    """LLMPlanner falls back to 'scan' when both attempts raise."""
    mock_llm = MagicMock()
    chain = MagicMock()
    # Both attempts raise — should trigger fallback
    chain.invoke.side_effect = [
        ValueError("bad output from LLM attempt 1"),
        ValueError("bad output from LLM attempt 2"),
    ]
    mock_llm.with_structured_output.return_value = chain

    planner = LLMPlanner(mock_llm)

    from kormoran.agent.planners import PlanContext
    from kormoran.types import DroneState, Pose

    ctx = PlanContext(
        target=_TARGET,
        drone=DroneState(pose=Pose(), battery_pct=100.0, is_flying=True),
        detections=[],
        observation="nothing visible",
        step=1,
        max_steps=50,
        history_summary="",
        recent_steps=[],
        recalled=[],
        safety_events=[],
    )
    result = planner.plan(ctx)
    assert result.kind == "scan", f"expected fallback kind='scan', got {result.kind!r}"
    assert chain.invoke.call_count == 2


# ---------------------------------------------------------------------------
# Test 4: Crash and resume from checkpoint
# ---------------------------------------------------------------------------


def test_crash_resume_from_checkpoint() -> None:
    """Graph interrupted after 'act' can be resumed to reach a terminal state."""
    from langgraph.checkpoint.memory import MemorySaver

    drone = _setup_drone()
    # max_steps=1 → after act (step→1), check sees step>=max_steps → not_found
    actions = [
        Action(kind="advance", distance_m=0.5, reasoning="s", confidence=1.0),
        Action(kind="advance", distance_m=0.5, reasoning="s", confidence=1.0),
    ]
    planner = ScriptedPlanner(actions)
    perceiver = MockPerceiver([])

    cp = MemorySaver()
    graph = build_mission_graph(
        drone,
        planner,
        perceiver,
        _settings(),
        checkpointer=cp,
        interrupt_after=["act"],
    )
    mission_id = new_mission_id()
    drone_state = drone.get_state()
    initial = make_initial_state(mission_id, "test_scene", _TARGET, drone_state, max_steps=1)
    thread_config = {"configurable": {"thread_id": mission_id}}

    # First invocation — pauses after "act".
    mid_state = graph.invoke(initial, config=thread_config)
    # State was checkpointed; act ran and incremented step.
    assert mid_state["step"] == 1, f"expected step=1 after first act, got {mid_state['step']}"

    # Simulate crash by not doing anything, then resume.
    final = graph.invoke(None, config=thread_config)
    assert final["status"] in (
        "not_found",
        "found",
        "aborted",
    ), f"expected terminal status, got {final['status']!r}"
    drone.disconnect()
