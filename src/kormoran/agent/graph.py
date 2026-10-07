"""LangGraph StateGraph builder for the Kormoran mission loop.

Usage
-----
::

    from kormoran.agent.graph import build_mission_graph, make_initial_state

    graph = build_mission_graph(drone, planner, perceiver, config, target)
    initial = make_initial_state("m001", "living_room_v1", target, drone, max_steps=50)
    config_dict = {"configurable": {"thread_id": initial["mission_id"]}}
    final = graph.invoke(initial, config=config_dict)
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from kormoran.agent.nodes import (
    make_act_node,
    make_check_node,
    make_perceive_node,
    make_plan_node,
    make_recall_memory_node,
    make_validate_node,
    reflect_node,
    route_check,
)
from kormoran.agent.state import MissionState, TokenUsage

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from kormoran.config import Settings
    from kormoran.interfaces import DroneInterface, Perceiver
    from kormoran.recording import MissionLogger
    from kormoran.types import DroneState, TargetSpec


def build_mission_graph(
    drone: DroneInterface,
    planner: object,
    perceiver: Perceiver,
    config: Settings,
    *,
    mission_logger: MissionLogger | None = None,
    checkpointer: object | None = None,
    interrupt_after: list[str] | None = None,
) -> CompiledStateGraph:
    """Compile and return the LangGraph mission graph.

    Parameters
    ----------
    drone:
        Any DroneInterface-conformant backend (already connected).
    planner:
        A ScriptedPlanner, HeuristicPlanner, or LLMPlanner instance.
    perceiver:
        A Perceiver-conformant object (e.g. ColorDetector).
    config:
        Settings for geofence and safety checks.
    checkpointer:
        LangGraph checkpointer; defaults to MemorySaver (in-process).
    interrupt_after:
        List of node names to interrupt after (for testing crash/resume).
    """
    from kormoran.control.executor import SkillExecutor

    executor = SkillExecutor(drone, config)

    builder: StateGraph = StateGraph(MissionState)

    builder.add_node("recall_memory", make_recall_memory_node())
    builder.add_node("perceive", make_perceive_node(drone, perceiver))
    builder.add_node("plan", make_plan_node(planner))
    builder.add_node("validate", make_validate_node(config))
    builder.add_node("act", make_act_node(executor, perceiver, mission_logger))
    builder.add_node("check", make_check_node())
    builder.add_node("reflect_and_store", reflect_node)

    builder.set_entry_point("recall_memory")
    builder.add_edge("recall_memory", "perceive")
    builder.add_edge("perceive", "plan")
    builder.add_edge("plan", "validate")
    builder.add_edge("validate", "act")
    builder.add_edge("act", "check")
    builder.add_conditional_edges(
        "check",
        route_check,
        {"continue": "perceive", "done": "reflect_and_store"},
    )
    builder.add_edge("reflect_and_store", END)

    cp = checkpointer if checkpointer is not None else MemorySaver()
    return builder.compile(
        checkpointer=cp,
        interrupt_after=interrupt_after or [],
    )


def make_initial_state(
    mission_id: str,
    scene_id: str,
    target: TargetSpec,
    drone_state: DroneState,
    max_steps: int,
) -> MissionState:
    """Build the initial MissionState for a new mission."""
    return MissionState(
        mission_id=mission_id,
        scene_id=scene_id,
        target=target.model_dump(),
        step=0,
        max_steps=max_steps,
        started_at=time.time(),
        drone=drone_state.model_dump(),
        last_detections=[],
        observation="",
        recent_steps=[],
        history_summary="",
        visited=[],
        recalled=[],
        status="planning",
        safety_events=[],
        usage=TokenUsage(
            prompt_tokens=0,
            completion_tokens=0,
            llm_calls=0,
            est_cost_usd=0.0,
        ),
        last_action=None,
        last_result=None,
        consecutive_found=0,
        last_verdict_kind="allow",
    )


def new_mission_id() -> str:
    """Generate a short unique mission ID."""
    return uuid.uuid4().hex[:8]
