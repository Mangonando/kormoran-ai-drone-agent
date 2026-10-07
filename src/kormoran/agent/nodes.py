"""Node factory functions for the LangGraph mission graph.

Each public function returns a callable suitable for ``StateGraph.add_node``.
Dependencies are injected via closure so the graph builder (graph.py) stays
the only place that knows about backends.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import TYPE_CHECKING

from kormoran.agent.state import (
    FOUND_AREA_FRAC,
    FOUND_CENTER_TOL,
    FOUND_CONSECUTIVE,
    GRID_CELL_SIZE_M,
    GridCell,
    MissionState,
    StepRecord,
    TokenUsage,
)
from kormoran.types import Action, Detection, DroneState, MoveResult, ScanResult

if TYPE_CHECKING:
    from kormoran.config import Settings
    from kormoran.control.executor import SkillExecutor
    from kormoran.interfaces import DroneInterface, Perceiver
    from kormoran.recording import MissionLogger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _best_detection(detections: list[Detection]) -> Detection | None:
    """Return the detection with highest confidence × area_frac."""
    if not detections:
        return None
    return max(detections, key=lambda d: d.confidence * d.area_frac)


def _cell_from_pose(x: float, y: float) -> GridCell:
    return GridCell(
        x_idx=int(math.floor(x / GRID_CELL_SIZE_M)),
        y_idx=int(math.floor(y / GRID_CELL_SIZE_M)),
    )


# ---------------------------------------------------------------------------
# recall_memory node (stub — M5 will fill this in)
# ---------------------------------------------------------------------------


def make_recall_memory_node() -> Callable[[MissionState], dict]:
    def recall_memory_node(state: MissionState) -> dict:  # noqa: ARG001
        return {"recalled": []}

    return recall_memory_node


# ---------------------------------------------------------------------------
# perceive node
# ---------------------------------------------------------------------------


def make_perceive_node(
    drone: DroneInterface,
    perceiver: Perceiver,
) -> Callable[[MissionState], dict]:
    def perceive_node(state: MissionState) -> dict:
        from kormoran.perception.summarizer import summarize
        from kormoran.types import TargetSpec

        target = TargetSpec.model_validate(state["target"])
        frame = drone.get_frame()
        detections = perceiver.detect(frame, target)
        observation = summarize(detections, target)
        drone_state: DroneState = drone.get_state()

        return {
            "drone": drone_state.model_dump(),
            "last_detections": [d.model_dump() for d in detections],
            "observation": observation,
        }

    return perceive_node


# ---------------------------------------------------------------------------
# plan node
# ---------------------------------------------------------------------------


def make_plan_node(planner: object) -> Callable[[MissionState], dict]:
    def plan_node(state: MissionState) -> dict:
        from kormoran.agent.planners import PlanContext
        from kormoran.types import SafetyEvent, TargetSpec

        target = TargetSpec.model_validate(state["target"])
        drone_state = DroneState.model_validate(state["drone"])
        detections = [Detection.model_validate(d) for d in state["last_detections"]]
        safety_events = [SafetyEvent.model_validate(e) for e in state["safety_events"]]

        ctx = PlanContext(
            target=target,
            drone=drone_state,
            detections=detections,
            observation=state["observation"],
            step=state["step"],
            max_steps=state["max_steps"],
            history_summary=state["history_summary"],
            recent_steps=list(state["recent_steps"]),
            recalled=list(state["recalled"]),
            safety_events=safety_events,
        )

        try:
            action: Action = planner.plan(ctx)  # type: ignore[union-attr]
        except StopIteration:
            action = Action(
                kind="report_not_found",
                reasoning="planner exhausted its action list",
                confidence=1.0,
            )

        update: dict = {"last_action": action.model_dump()}

        # Accumulate LLM token usage if the planner tracks it.
        tu = getattr(planner, "last_token_usage", None)
        if tu:
            old = state.get("usage") or {}
            update["usage"] = TokenUsage(
                prompt_tokens=old.get("prompt_tokens", 0) + tu.get("prompt_tokens", 0),
                completion_tokens=old.get("completion_tokens", 0) + tu.get("completion_tokens", 0),
                llm_calls=old.get("llm_calls", 0) + tu.get("llm_calls", 0),
                est_cost_usd=old.get("est_cost_usd", 0.0) + tu.get("est_cost_usd", 0.0),
            )

        return update

    return plan_node


# ---------------------------------------------------------------------------
# validate node
# ---------------------------------------------------------------------------


def make_validate_node(config: Settings) -> Callable[[MissionState], dict]:
    def validate_node(state: MissionState) -> dict:
        from kormoran.safety import supervisor as sup

        if state["last_action"] is None:
            return {}

        action = Action.model_validate(state["last_action"])
        drone_state = DroneState.model_validate(state["drone"])
        verdict = sup.check(action, drone_state, config)

        new_events = [e.model_dump() for e in verdict.events]
        return {
            "last_action": verdict.action.model_dump(),
            "safety_events": new_events,
            "last_verdict_kind": verdict.kind,
        }

    return validate_node


# ---------------------------------------------------------------------------
# act node
# ---------------------------------------------------------------------------


def make_act_node(
    executor: SkillExecutor,
    perceiver: Perceiver,
    mission_logger: MissionLogger | None = None,
) -> Callable[[MissionState], dict]:
    def act_node(state: MissionState) -> dict:
        from kormoran.types import TargetSpec

        if state["last_action"] is None:
            return {}

        # Use model_construct to skip re-validation: the supervisor may have
        # clamped distance_m below the planner-facing ge=0.2 bound, and that
        # is intentional — the supervisor's verdict is already trusted.
        action = Action.model_construct(**state["last_action"])
        drone_state = DroneState.model_validate(state["drone"])
        target = TargetSpec.model_validate(state["target"])

        # Pass best current detection for approach skill.
        current_dets = [Detection.model_validate(d) for d in state["last_detections"]]
        detection = _best_detection(current_dets)

        result = executor.execute(action, drone_state, detection)

        new_safety_events = [e.model_dump() for e in executor.safety_events]
        executor.safety_events.clear()

        last_detections: list[dict] = list(state["last_detections"])

        # Special handling for scan: run detector on every captured frame,
        # rotate drone to face best detection found.
        if isinstance(result, ScanResult):
            best_det: Detection | None = None
            best_score = -1.0
            best_yaw = 0.0

            for frame, yaw in zip(result.frames, result.yaws_deg, strict=True):
                frame_dets = perceiver.detect(frame, target)
                for d in frame_dets:
                    score = d.confidence * d.area_frac
                    if score > best_score:
                        best_score = score
                        best_det = d
                        best_yaw = yaw

            if best_det is not None:
                last_detections = [best_det.model_dump()]
                # Rotate so the drone faces the best detection yaw.
                current_state = executor._drone.get_state()  # noqa: SLF001
                delta_yaw = best_yaw - current_state.pose.yaw_deg
                # Normalise to (-180, 180].
                while delta_yaw > 180.0:
                    delta_yaw -= 360.0
                while delta_yaw <= -180.0:
                    delta_yaw += 360.0
                if abs(delta_yaw) > 5.0:
                    rotate_action = Action(
                        kind="rotate",
                        angle_deg=delta_yaw,
                        reasoning="facing best detection after scan",
                        confidence=1.0,
                    )
                    executor.execute(rotate_action, current_state)
                    executor.safety_events.clear()
            else:
                last_detections = []

            # Represent scan as a MoveResult for the log.
            result = MoveResult(
                status="reached",
                pose=result.final_pose,
                elapsed_s=result.elapsed_s,
            )

        result_dict = result.model_dump() if isinstance(result, MoveResult) else {}

        # Build step record.
        action_obj = action
        step_record = StepRecord(
            step=state["step"],
            action_kind=action_obj.kind,
            reasoning=action_obj.reasoning,
            result_status=result_dict.get("status", "?"),
            detections_count=len(last_detections),
            pose_x=result_dict.get("pose", {}).get("x", 0.0),
            pose_y=result_dict.get("pose", {}).get("y", 0.0),
            pose_z=result_dict.get("pose", {}).get("z", 0.0),
        )

        # Track visited cells.
        pose_dict = result_dict.get("pose", {})
        cell = _cell_from_pose(pose_dict.get("x", 0.0), pose_dict.get("y", 0.0))

        if mission_logger is not None:
            mission_logger.log_step(
                step=state["step"],
                action=state["last_action"] or {},
                result=result_dict,
                detections=last_detections,
                pose=result_dict.get("pose", {}),
                usage=dict(state.get("usage") or {}),
                events=new_safety_events,
            )

        return {
            "last_result": result_dict,
            "last_detections": last_detections,
            "recent_steps": [step_record],  # operator.add appends this
            "visited": [cell],  # operator.add appends this
            "safety_events": new_safety_events,
            "step": state["step"] + 1,
        }

    return act_node


# ---------------------------------------------------------------------------
# check node — routing decision
# ---------------------------------------------------------------------------


def make_check_node() -> Callable[[MissionState], dict]:
    def check_node(state: MissionState) -> dict:
        step = state["step"]
        max_steps = state["max_steps"]
        last_action = state.get("last_action") or {}
        action_kind = last_action.get("kind", "")
        last_result = state.get("last_result") or {}
        result_status = last_result.get("status", "")

        # Abort: drone landed via safety rejection.
        if result_status == "aborted" and action_kind not in ("scan",):
            logger.warning("check_node: mission aborted (result status=aborted)")
            return {"status": "aborted"}

        # Terminal actions.
        if action_kind == "report_not_found":
            return {"status": "not_found"}
        if action_kind == "land":
            # Safety-forced land (supervisor reject) → aborted.
            # Voluntary land chosen by the LLM → found.
            if state.get("last_verdict_kind") == "reject":
                logger.warning("check_node: supervisor-forced landing → aborted")
                return {"status": "aborted"}
            return {"status": "found"}

        # Budget exhausted.
        if step >= max_steps:
            return {"status": "not_found"}

        # Found criterion: detection large + centred for FOUND_CONSECUTIVE steps.
        detections = state["last_detections"]
        consecutive = state["consecutive_found"]
        if detections:
            det = detections[0]
            area_ok = det.get("area_frac", 0.0) >= FOUND_AREA_FRAC
            cx = det.get("center_norm", [0.0, 0.0])
            if isinstance(cx, list | tuple):
                cx_val = cx[0]
            else:
                cx_val = 0.0
            center_ok = abs(cx_val) <= FOUND_CENTER_TOL
            if area_ok and center_ok:
                consecutive += 1
            else:
                consecutive = 0
        else:
            consecutive = 0

        if consecutive >= FOUND_CONSECUTIVE:
            return {"status": "found", "consecutive_found": consecutive}

        return {"status": "flying", "consecutive_found": consecutive}

    return check_node


# ---------------------------------------------------------------------------
# reflect_and_store node (stub — M5 will persist to ChromaDB/SQLite)
# ---------------------------------------------------------------------------


def reflect_node(state: MissionState) -> dict:  # noqa: ARG001
    return {}


# ---------------------------------------------------------------------------
# Routing function for conditional edges
# ---------------------------------------------------------------------------


def route_check(state: MissionState) -> str:
    terminal = {"found", "not_found", "aborted"}
    return "done" if state["status"] in terminal else "continue"
