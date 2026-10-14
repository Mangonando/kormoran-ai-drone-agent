"""Node factory functions for the LangGraph mission graph.

Each public function returns a callable suitable for ``StateGraph.add_node``.
Dependencies are injected via closure so the graph builder (graph.py) stays
the only place that knows about backends.
"""

from __future__ import annotations

import logging
import time
import uuid as _uuid
from collections.abc import Callable
from typing import TYPE_CHECKING

from kormoran.agent.state import (
    ESCAPE_MARGIN_M,
    ESCAPE_QUADRANT_ANGLES,
    ESCAPE_YAW_TOLERANCE_DEG,
    FOUND_AREA_FRAC,
    FOUND_CENTER_TOL,
    GRID_CELL_SIZE_M,
    MAX_CONSECUTIVE_ROTATES,
    MAX_CONSECUTIVE_SCANS,
    MAX_EXPLORE_Z_M,
    MIN_FLIGHT_Z_M,
    REFLEX_ALTITUDE_M,
    REFLEX_AREA_FRAC,
    REFLEX_LAND_FRAC,
    STUCK_WINDOW,
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
    from kormoran.memory.episodic import EpisodicStore
    from kormoran.memory.semantic import SemanticStore
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


def _detect_walls(
    recent_steps: list[StepRecord],
    cluster_tol: float = 0.3,
    min_hits: int = 2,
) -> list[tuple[str, float]]:
    """Return a list of detected walls from clamped advance/approach positions.

    Each wall is ``("x", value)`` or ``("y", value)`` indicating a vertical or
    horizontal wall at the given world coordinate.
    """
    clamped = [
        r
        for r in recent_steps
        if r["result_status"] == "clamped" and r["action_kind"] in ("advance", "approach")
    ]
    walls: list[tuple[str, float]] = []
    # Vertical walls (constant x)
    xs = sorted(r["pose_x"] for r in clamped)
    for x0 in xs:
        hits = [x for x in xs if abs(x - x0) < cluster_tol]
        if len(hits) >= min_hits:
            wall_x = sum(hits) / len(hits)
            if not any(axis == "x" and abs(val - wall_x) < cluster_tol for axis, val in walls):
                walls.append(("x", wall_x))
    # Horizontal walls (constant y)
    ys = sorted(r["pose_y"] for r in clamped)
    for y0 in ys:
        hits = [y for y in ys if abs(y - y0) < cluster_tol]
        if len(hits) >= min_hits:
            wall_y = sum(hits) / len(hits)
            if not any(axis == "y" and abs(val - wall_y) < cluster_tol for axis, val in walls):
                walls.append(("y", wall_y))
    return walls


def _is_stuck(recent_steps: list[StepRecord], window: int = 12, radius: float = 0.5) -> bool:
    """Return True if the drone hasn't moved more than *radius* m in *window* steps."""
    if len(recent_steps) < window:
        return False
    tail = recent_steps[-window:]
    xs = [r["pose_x"] for r in tail]
    ys = [r["pose_y"] for r in tail]
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    return dx < radius and dy < radius


def _detect_gaps(
    steps: list[StepRecord],
    cluster_tol: float = 0.3,
) -> list[tuple[str, float, float]]:
    """Detect gaps in walls the drone successfully crossed.

    Uses a stricter wall detection (min_hits=3) and filters out geofence
    boundaries to avoid false positives.  Only interior walls matter.

    Returns list of (axis, wall_value, gap_coordinate) tuples.
    E.g. ("x", 0.0, 0.3) means the wall is at x≈0 and the drone
    crossed it at y≈0.3.
    """
    # Stricter: require 3+ hits to be considered a real wall.
    walls = _detect_walls(steps, cluster_tol=cluster_tol, min_hits=3)
    if not walls:
        return []

    # Filter out geofence boundaries (not real indoor walls).
    interior_walls = [
        (axis, val)
        for axis, val in walls
        if (axis == "x" and -2.0 < val < 2.0) or (axis == "y" and -1.5 < val < 1.5)
    ]
    if not interior_walls:
        return []

    # Find successful advances (not clamped, not takeoff/scan/rotate).
    advances = [
        r for r in steps if r["action_kind"] == "advance" and r["result_status"] == "reached"
    ]

    gaps: list[tuple[str, float, float]] = []
    seen: set[tuple[str, float]] = set()

    for axis, val in interior_walls:
        for i, adv in enumerate(advances):
            if i == 0:
                continue
            prev = advances[i - 1]
            if axis == "x":
                # Crossed the wall if prev_x and curr_x are on opposite sides.
                if (prev["pose_x"] - val) * (adv["pose_x"] - val) < 0:
                    gap_y = round(adv["pose_y"] * 2) / 2
                    key = (axis, gap_y)
                    if key not in seen:
                        seen.add(key)
                        gaps.append(("x", val, gap_y))
            elif axis == "y":
                if (prev["pose_y"] - val) * (adv["pose_y"] - val) < 0:
                    gap_x = round(adv["pose_x"] * 2) / 2
                    key = (axis, gap_x)
                    if key not in seen:
                        seen.add(key)
                        gaps.append(("y", val, gap_x))

    return gaps


def _cell_from_pose(x: float, y: float) -> GridCell:
    # Use round() instead of floor() so sub-millimetre PID drift at cell boundaries
    # (e.g. x = -0.0001 after a scan at the origin) does not falsely record the
    # adjacent cell and corrupt the coverage quadrant counts.
    return GridCell(
        x_idx=int(round(x / GRID_CELL_SIZE_M)),
        y_idx=int(round(y / GRID_CELL_SIZE_M)),
    )


# ---------------------------------------------------------------------------
# recall_memory node
# ---------------------------------------------------------------------------


def make_recall_memory_node(
    semantic: SemanticStore | None = None,
) -> Callable[[MissionState], dict]:
    """Return a node that retrieves relevant notes from the SemanticStore.

    If *semantic* is None (--no-memory mode) the node is a no-op.
    """

    def recall_memory_node(state: MissionState) -> dict:
        if semantic is None:
            return {"recalled": []}

        from kormoran.memory.retrieval import mark_stale_if_contradicted, retrieve_for_context
        from kormoran.types import TargetSpec

        target = TargetSpec.model_validate(state["target"])
        drone = DroneState.model_validate(state["drone"])
        recalled = retrieve_for_context(
            semantic,
            scene_id=state["scene_id"],
            target_label=target.label,
            drone_x=drone.pose.x,
            drone_y=drone.pose.y,
            step=state["step"],
        )
        # Halve confidence on any location notes that are contradicted by
        # the current observation (no detection near a previously noted location).
        mark_stale_if_contradicted(
            semantic,
            recalled=recalled,
            drone_x=drone.pose.x,
            drone_y=drone.pose.y,
            has_detection=bool(state.get("last_detections")),
        )
        return {"recalled": recalled}

    return recall_memory_node


# ---------------------------------------------------------------------------
# perceive node
# ---------------------------------------------------------------------------


def make_perceive_node(
    drone: DroneInterface,
    perceiver: Perceiver,
    mission_logger: MissionLogger | None = None,
) -> Callable[[MissionState], dict]:
    def perceive_node(state: MissionState) -> dict:
        from kormoran.perception.summarizer import summarize
        from kormoran.types import TargetSpec

        target = TargetSpec.model_validate(state["target"])
        frame = drone.get_frame()
        if mission_logger is not None:
            mission_logger.add_frame(frame)
            if hasattr(drone, "get_top_frame"):
                mission_logger.add_top_frame(drone.get_top_frame())  # type: ignore[union-attr]
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


def make_plan_node(planner: object, config: Settings) -> Callable[[MissionState], dict]:
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
            visited=list(state["visited"]),
        )

        # ------------------------------------------------------------------
        # Crash recovery reflex: if the drone is near ground level it has
        # collided with an obstacle and been thrown down by the physics
        # engine.  Re-takeoff before anything else so the mission can
        # continue instead of burning all remaining steps on the ground.
        # ------------------------------------------------------------------
        if drone_state.pose.z < MIN_FLIGHT_Z_M:
            logger.info(
                "plan_node reflex: z=%.3f < %.1f → takeoff (crash recovery)",
                drone_state.pose.z,
                MIN_FLIGHT_Z_M,
            )
            return {
                "last_action": Action(
                    kind="takeoff",
                    reasoning="reflex: drone near ground — re-taking off to recover altitude",
                    confidence=1.0,
                ).model_dump()
            }

        # ------------------------------------------------------------------
        # Approach reflex: bypass the planner when the target is visible.
        # Sequence: descend to approach altitude first, then approach.
        # No LLM call is made — this saves tokens and reacts immediately.
        #
        # Suppressed if the previous approach was clamped (timed-out): that
        # signals the path is blocked (e.g. a wall between the drone and the
        # apparent target position).  Fall through to the planner so it can
        # rotate and find a clear route.
        # ------------------------------------------------------------------
        # Suppress approach if any of the last 3 steps was a clamped approach.
        # Checking a window (rather than just last_action) handles the case where
        # a crash-recovery climb fires between the blocked approach and the next
        # plan decision — the climb resets last_action but the wall is still there.
        approach_blocked = any(
            r["action_kind"] == "approach" and r["result_status"] == "clamped"
            for r in list(state.get("recent_steps", []))[-3:]
        )

        reflex_action: Action | None = None
        if detections and detections[0].area_frac >= REFLEX_AREA_FRAC and not approach_blocked:
            area = detections[0].area_frac
            cx = detections[0].center_norm[0] if detections[0].center_norm else 0.0
            descent_needed = drone_state.pose.z - REFLEX_ALTITUDE_M
            if area >= REFLEX_LAND_FRAC and (
                state.get("consecutive_found", 0) >= 1 or abs(cx) <= FOUND_CENTER_TOL
            ):
                # Drone is right on top of the target — land immediately if
                # either we already confirmed a qualifying frame OR the target
                # is large and centred right now (no point risking another step).
                reflex_action = Action(
                    kind="land",
                    reasoning="reflex: target fills frame — landing to confirm",
                    confidence=1.0,
                )
            elif descent_needed >= 0.2:
                reflex_action = Action(
                    kind="descend",
                    distance_m=round(min(descent_needed, 2.0), 2),
                    reasoning="reflex: target spotted — descending to approach altitude",
                    confidence=1.0,
                )
            else:
                reflex_action = Action(
                    kind="approach",
                    reasoning="reflex: target spotted — approaching",
                    confidence=1.0,
                )
            logger.info(
                "plan_node reflex: area_frac=%.3f → %s",
                detections[0].area_frac,
                reflex_action.kind,
            )

        if approach_blocked and detections:
            logger.info(
                "plan_node: approach suppressed (previous approach clamped) — falling to planner"
            )

        if reflex_action is not None:
            update = {"last_action": reflex_action.model_dump()}
            # If landing immediately on a centred target, ensure
            # consecutive_found is set so check_node recognises the find.
            if reflex_action.kind == "land":
                update["consecutive_found"] = max(state.get("consecutive_found", 0), 1)
            return update

        # ------------------------------------------------------------------
        # Altitude cap: if the drone drifted above 1.1 m during exploration
        # (no detection visible), force a descend before anything else.
        # At z > 1.1 the downward camera angle makes floor-level targets
        # nearly invisible.
        # ------------------------------------------------------------------
        if drone_state.pose.z > MAX_EXPLORE_Z_M and not detections:
            excess = round(drone_state.pose.z - (REFLEX_ALTITUDE_M + 0.3), 2)
            if excess >= 0.2:
                logger.info(
                    "plan_node: z=%.2f > %.1f with no detection → descending %.2fm",
                    drone_state.pose.z,
                    MAX_EXPLORE_Z_M,
                    excess,
                )
                return {
                    "last_action": Action(
                        kind="descend",
                        distance_m=min(excess, 2.0),
                        reasoning="altitude cap: too high to detect floor targets — descending",
                        confidence=1.0,
                    ).model_dump()
                }

        try:
            action: Action = planner.plan(ctx)  # type: ignore[union-attr]
        except StopIteration:
            action = Action(
                kind="report_not_found",
                reasoning="planner exhausted its action list",
                confidence=1.0,
            )

        recent = list(state.get("recent_steps", []))

        # Scan spam guardrail: if 3+ consecutive scans from roughly the
        # same position, force an advance to move to new ground.
        if action.kind == "scan":
            consecutive_scans = 0
            for r in reversed(recent):
                if r["action_kind"] == "scan":
                    consecutive_scans += 1
                else:
                    break
            if consecutive_scans >= MAX_CONSECUTIVE_SCANS:
                logger.info(
                    "plan_node guardrail: %d consecutive scans → forcing advance 0.8m",
                    consecutive_scans,
                )
                action = Action(
                    kind="advance",
                    distance_m=0.8,
                    reasoning="guardrail: too many scans from same spot — advancing",
                    confidence=1.0,
                )

        # Stuck detector: if position hasn't changed in 8 steps, force an
        # escape toward the least-visited quadrant.  Alternates between
        # rotate (to face the escape direction) and advance (to move there).
        # Excludes quadrants that would send the drone into a nearby geofence
        # boundary (within 0.5 m).
        # If the same escape direction was tried recently (last 4 stuck
        # escapes), skip it and try the next-least-visited quadrant.
        if _is_stuck(recent):
            visited = list(state.get("visited", []))
            ne = sum(1 for c in visited if c["x_idx"] >= 0 and c["y_idx"] >= 0)
            se = sum(1 for c in visited if c["x_idx"] >= 0 and c["y_idx"] < 0)
            nw = sum(1 for c in visited if c["x_idx"] < 0 and c["y_idx"] >= 0)
            sw = sum(1 for c in visited if c["x_idx"] < 0 and c["y_idx"] < 0)

            # Filter out quadrants near geofence boundaries.
            px, py = drone_state.pose.x, drone_state.pose.y
            margin = ESCAPE_MARGIN_M
            candidates = [("NE", ne), ("SE", se), ("NW", nw), ("SW", sw)]
            feasible = []
            for name, count in candidates:
                bad = False
                if "E" in name and px > config.geofence_x_max - margin:
                    bad = True
                if "W" in name and px < config.geofence_x_min + margin:
                    bad = True
                if "N" in name and py > config.geofence_y_max - margin:
                    bad = True
                if "S" in name and py < config.geofence_y_min + margin:
                    bad = True
                if not bad:
                    feasible.append((name, count))

            if not feasible:
                feasible = candidates

            # Check which escape directions were tried recently and failed
            # (drone is still stuck after trying them).
            recent_escapes = [
                r["reasoning"].split("toward ")[-1].split(" quadrant")[0]
                for r in recent[-STUCK_WINDOW:]
                if "stuck escape" in r.get("reasoning", "")
            ]
            # Sort by least-visited, then skip recently-tried directions.
            feasible.sort(key=lambda t: t[1])
            chosen = feasible[0]  # default: least visited
            for name, count in feasible:
                if name not in recent_escapes:
                    chosen = (name, count)
                    break

            least = chosen
            escape_yaw = ESCAPE_QUADRANT_ANGLES[least[0]]
            delta = escape_yaw - drone_state.pose.yaw_deg
            while delta > 180:
                delta -= 360
            while delta <= -180:
                delta += 360

            # If already facing the right way, advance.
            if abs(delta) < ESCAPE_YAW_TOLERANCE_DEG:
                logger.info("plan_node: STUCK → advancing 1.0m toward %s", least[0])
                action = Action(
                    kind="advance",
                    distance_m=1.0,
                    reasoning=f"stuck escape: advancing toward {least[0]} quadrant",
                    confidence=1.0,
                )
            else:
                logger.info(
                    "plan_node: STUCK for 8+ steps → escaping toward %s (rotate %.0f°)",
                    least[0],
                    delta,
                )
                action = Action(
                    kind="rotate",
                    angle_deg=delta,
                    reasoning=f"stuck escape: rotating toward {least[0]} quadrant",
                    confidence=1.0,
                )
            return {"last_action": action.model_dump()}

        # Guardrail: if the LLM returns 3+ consecutive rotates, force
        # movement.  Allows rotate→rotate (genuine navigation) but prevents
        # infinite spin.  Only fires after the 3rd consecutive rotate.
        if action.kind == "rotate":
            consecutive_rotates = 0
            for r in reversed(recent):
                if r["action_kind"] == "rotate":
                    consecutive_rotates += 1
                else:
                    break
            if consecutive_rotates >= MAX_CONSECUTIVE_ROTATES:
                logger.info(
                    "plan_node guardrail: %d consecutive rotates → forcing advance 0.8m",
                    consecutive_rotates + 1,
                )
                action = Action(
                    kind="advance",
                    distance_m=0.8,
                    reasoning="guardrail: forced advance after 3+ consecutive rotates",
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

        try:
            action = Action.model_validate(state["last_action"])
        except Exception:  # noqa: BLE001
            # Malformed action from LLM — replace with a safe scan.
            logger.warning("validate_node: invalid Action dict — falling back to scan")
            action = Action(
                kind="scan",
                reasoning="fallback: previous action was malformed",
                confidence=0.3,
            )
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

        result = executor.execute(action, drone_state, detection, target=target)

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
        pose_dict = result_dict.get("pose", {})
        px = pose_dict.get("x", 0.0)
        py = pose_dict.get("y", 0.0)
        pz = pose_dict.get("z", 0.0)

        # Build step record.
        step_record = StepRecord(
            step=state["step"],
            action_kind=action.kind,
            reasoning=action.reasoning,
            result_status=result_dict.get("status", "?"),
            detections_count=len(last_detections),
            pose_x=px,
            pose_y=py,
            pose_z=pz,
        )

        # Track visited cells.
        cell = _cell_from_pose(px, py)

        if mission_logger is not None:
            mission_logger.log_step(
                step=state["step"],
                action=state["last_action"] or {},
                result=result_dict,
                detections=last_detections,
                pose=pose_dict,
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


def make_check_node(
    stop_flag_path: object | None = None,
) -> Callable[[MissionState], dict]:
    """Return the check node.

    Parameters
    ----------
    stop_flag_path:
        Optional ``pathlib.Path`` to a sentinel file written by the
        Streamlit dashboard's Emergency Stop button.  When the file exists
        the mission is aborted gracefully (→ ``reflect_and_store``).
    """

    def check_node(state: MissionState) -> dict:
        # Emergency stop: UI wrote the sentinel file → abort cleanly.
        if stop_flag_path is not None:
            from pathlib import Path

            if Path(stop_flag_path).exists():
                logger.warning("check_node: emergency stop flag detected — aborting mission")
                return {"status": "aborted"}

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
            if state.get("last_verdict_kind") == "reject":
                logger.warning("check_node: supervisor-forced landing → aborted")
                return {"status": "aborted"}
            # Voluntary land: only counts as found if the drone had a qualifying
            # detection in the previous step (consecutive_found > 0).  Landing
            # without a confirmed sighting is treated as not_found.
            if state.get("consecutive_found", 0) > 0:
                return {"status": "found"}
            logger.info("check_node: LLM landed without prior detection → not_found")
            return {"status": "not_found"}

        # Budget exhausted.
        if step >= max_steps:
            return {"status": "not_found"}

        # Update consecutive_found counter so the land reflex knows when to fire.
        # "found" is only declared via land (voluntary or reflex) + consecutive_found > 0,
        # forcing the drone to physically reach the target before the mission ends.
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
            consecutive = consecutive + 1 if (area_ok and center_ok) else 0
        else:
            consecutive = 0

        # Build a deterministic history summary from ALL steps so far.
        # This gives the LLM a "map" of the mission even after old steps
        # scroll out of the recent_steps window.
        summary = _build_history_summary(list(state["recent_steps"]))

        return {
            "status": "flying",
            "consecutive_found": consecutive,
            "history_summary": summary,
        }

    return check_node


def _build_history_summary(steps: list[StepRecord]) -> str:
    """Build a deterministic text summary of the full mission history."""
    if not steps:
        return ""

    # Collect all positions visited.
    xs = [r["pose_x"] for r in steps]
    ys = [r["pose_y"] for r in steps]

    # Identify wall positions from clamped advances.
    clamped = [
        r
        for r in steps
        if r["result_status"] == "clamped" and r["action_kind"] in ("advance", "approach")
    ]
    crashes = sum(
        1 for r in steps if r["action_kind"] == "takeoff" and "reflex" in r.get("reasoning", "")
    )

    # Cluster clamped positions to identify walls.
    wall_hints: list[str] = []
    if clamped:
        cx = [r["pose_x"] for r in clamped]
        cy = [r["pose_y"] for r in clamped]
        # Check for horizontal walls (constant y).
        for threshold_y in sorted(set(round(y, 0) for y in cy)):
            hits = sum(1 for y in cy if abs(y - threshold_y) < 0.3)
            if hits >= 2:
                wall_hints.append(f"wall at y≈{threshold_y:.0f} ({hits} hits)")
        # Check for vertical walls (constant x).
        for threshold_x in sorted(set(round(x, 0) for x in cx)):
            hits = sum(1 for x in cx if abs(x - threshold_x) < 0.3)
            if hits >= 2:
                wall_hints.append(f"wall at x≈{threshold_x:.0f} ({hits} hits)")

    parts: list[str] = []
    parts.append(
        f"Steps so far: {len(steps)}. "
        f"Area covered: x=[{min(xs):.1f},{max(xs):.1f}] y=[{min(ys):.1f},{max(ys):.1f}]."
    )
    if wall_hints:
        parts.append(f"Walls detected: {'; '.join(wall_hints)}.")
    if crashes:
        parts.append(f"Crashes: {crashes}.")

    # Flag areas NOT yet visited.
    positions = list(zip(xs, ys, strict=True))
    visited_ne = any(x >= 0 and y >= 0 for x, y in positions)
    visited_se = any(x >= 0 and y < 0 for x, y in positions)
    not_visited = []
    if not visited_ne:
        not_visited.append("NE (x>0, y>0)")
    if not visited_se:
        not_visited.append("SE (x>0, y<0)")
    if not_visited:
        parts.append(f"NOT YET VISITED: {', '.join(not_visited)}.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# reflect_and_store node
# ---------------------------------------------------------------------------


def make_reflect_node(
    episodic: EpisodicStore | None = None,
    semantic: SemanticStore | None = None,
    llm: object | None = None,
) -> Callable[[MissionState], dict]:
    """Return a node that persists mission results and generates memory notes.

    If both stores are None (--no-memory mode) the node is a no-op.
    LLM reflection is only performed when *llm* is provided.
    """

    def reflect_node(state: MissionState) -> dict:
        from datetime import UTC, datetime

        from kormoran.memory.schemas import EpisodicRecord
        from kormoran.types import TargetSpec

        target = TargetSpec.model_validate(state["target"])
        usage = state.get("usage") or {}

        # ------------------------------------------------------------------
        # Build the path from recent_steps (no frames — privacy rule).
        # ------------------------------------------------------------------
        path: list[dict] = [
            {
                "x": r["pose_x"],
                "y": r["pose_y"],
                "z": r["pose_z"],
                "step": r["step"],
            }
            for r in state["recent_steps"]
        ]

        # Determine outcome
        status = state.get("status", "error")
        outcome_map = {"found": "found", "not_found": "not_found", "aborted": "error"}
        outcome = outcome_map.get(status, "error")

        found_pose: dict | None = None
        found_from_pose: dict | None = None
        if outcome == "found" and path:
            last = path[-1]
            found_pose = {"x": last["x"], "y": last["y"], "z": last["z"], "yaw_deg": 0.0}
            if len(path) >= 2:
                prev = path[-2]
                found_from_pose = {"x": prev["x"], "y": prev["y"], "z": prev["z"], "yaw_deg": 0.0}

        now_iso = datetime.now(UTC).isoformat()
        duration_s = time.time() - float(state.get("started_at", time.time()))

        record = EpisodicRecord(
            mission_id=state["mission_id"],
            ts=now_iso,
            scene_id=state["scene_id"],
            target_label=target.label,
            outcome=outcome,
            steps=state["step"],
            duration_s=round(duration_s, 2),
            path=path,
            found_pose=found_pose,
            found_from_pose=found_from_pose,
            model=usage.get("model", ""),
            tokens=usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
            cost_usd=usage.get("est_cost_usd", 0.0),
        )

        if episodic is not None:
            try:
                episodic.store(record)
                logger.info("reflect_node: episodic record stored (mission=%s)", record.mission_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("reflect_node: episodic store failed: %s", exc)

        if semantic is not None and llm is not None:
            from kormoran.memory.reflection import reflect

            # Extract wall hit positions from clamped advances.
            wall_hits = [
                {"x": r["pose_x"], "y": r["pose_y"]}
                for r in state["recent_steps"]
                if r["result_status"] == "clamped" and r["action_kind"] in ("advance", "approach")
            ]

            # Deterministic gap detection: if the drone successfully crossed
            # a coordinate where it was previously clamped, record the
            # crossing position as a "scene" note so future missions know
            # where the gap is.  No LLM call needed.
            gaps = _detect_gaps(list(state["recent_steps"]))
            if gaps and semantic is not None:
                from kormoran.memory.schemas import MemoryNote

                now_gap = datetime.now(UTC).isoformat()
                for axis, cross_val, gap_coord in gaps:
                    other = "y" if axis == "x" else "x"
                    text = (
                        f"Gap in {axis}≈{cross_val:.1f} wall: "
                        f"drone crossed at {other}≈{gap_coord:.1f}."
                    )
                    gap_note = MemoryNote(
                        note_id=str(_uuid.uuid4()),
                        scene_id=state["scene_id"],
                        target_label=target.label,
                        kind="scene",
                        text=text,
                        confidence=0.95,
                        created_at=now_gap,
                        mission_id=state["mission_id"],
                    )
                    try:
                        semantic.store(gap_note)
                        logger.info("reflect_node: gap note stored: %s", text)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("reflect_node: gap note store failed: %s", exc)

            notes = reflect(record, llm, wall_hits=wall_hits)

            # After a successful find, wipe ALL prior notes for this
            # scene+target.  The current mission's notes are fresher and
            # more accurate — old notes cause memory pollution that
            # confuses the LLM in later runs.
            if outcome == "found":
                old_notes = semantic.list_notes(
                    scene_id=record.scene_id,
                    target_label=record.target_label,
                )
                # Only wipe target-specific notes (location, strategy).
                # Keep wall and scene (gap) notes — they describe the room
                # layout and are shared across all targets in the same scene.
                old_ids = [
                    n["note_id"]
                    for n in old_notes
                    if n.get("mission_id") != record.mission_id
                    and n.get("kind") not in ("wall", "scene")
                ]
                if old_ids:
                    try:
                        semantic._col.delete(ids=old_ids)  # noqa: SLF001
                        logger.info(
                            "reflect_node: wiped %d old target-specific notes after find",
                            len(old_ids),
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("reflect_node: old note wipe failed: %s", exc)

            # Deduplicate: if a note with the same kind already exists
            # for this scene+target, replace it instead of adding another.
            existing = semantic.list_notes(
                scene_id=record.scene_id,
                target_label=record.target_label,
            )
            existing_by_kind: dict[str, str] = {}
            for e in existing:
                kind = e.get("kind", "")
                if kind not in existing_by_kind:
                    existing_by_kind[kind] = e["note_id"]

            for note in notes:
                # If a note of this kind already exists, reuse its ID
                # so the upsert replaces it instead of adding a duplicate.
                if note.kind in existing_by_kind:
                    note = note.model_copy(update={"note_id": existing_by_kind[note.kind]})
                    existing_by_kind.pop(note.kind)  # only replace first match
                try:
                    semantic.store(note)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("reflect_node: semantic store failed for note: %s", exc)
            if notes:
                logger.info("reflect_node: %d memory notes stored (deduped)", len(notes))

            # Prune to keep only the top notes per scene+target.
            n_pruned = semantic.prune(
                scene_id=record.scene_id,
                target_label=record.target_label,
                keep=5,
            )
            if n_pruned:
                logger.info("reflect_node: pruned %d low-confidence notes", n_pruned)

        return {}

    return reflect_node


# ---------------------------------------------------------------------------
# Routing function for conditional edges
# ---------------------------------------------------------------------------


def route_check(state: MissionState) -> str:
    terminal = {"found", "not_found", "aborted"}
    return "done" if state["status"] in terminal else "continue"
