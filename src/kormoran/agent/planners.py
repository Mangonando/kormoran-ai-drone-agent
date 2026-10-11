"""Planner implementations for the Kormoran mission loop.

Three planners are provided:
- ScriptedPlanner  — replays a fixed list of Actions (testing / demos)
- HeuristicPlanner — deterministic rule-based planner (no LLM)
- LLMPlanner       — delegates to an LLM via OpenRouter (uses langchain_openai)
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

from kormoran.agent.prompts import SYSTEM_PROMPT
from kormoran.types import Action, Detection, DroneState, SafetyEvent, TargetSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PlanContext — input to every Planner.plan()
# ---------------------------------------------------------------------------


class PlanContext(BaseModel):
    """All information a planner needs to choose the next action."""

    target: TargetSpec
    drone: DroneState
    detections: list[Detection]
    observation: str
    step: int
    max_steps: int
    history_summary: str
    recent_steps: list[dict]
    recalled: list[dict]
    safety_events: list[SafetyEvent]
    visited: list[dict]  # list of GridCell dicts {x_idx, y_idx}
    # Geofence bounds — used by burst direction selection to avoid heading into walls.
    geofence_x_min: float = -10.0
    geofence_x_max: float = 10.0
    geofence_y_min: float = -10.0
    geofence_y_max: float = 10.0


# ---------------------------------------------------------------------------
# ScriptedPlanner
# ---------------------------------------------------------------------------


class ScriptedPlanner:
    """Replays a fixed sequence of Actions; raises StopIteration when exhausted."""

    def __init__(self, actions: list[Action]) -> None:
        self._actions = list(actions)
        self._idx = 0

    def plan(self, ctx: PlanContext) -> Action:  # noqa: ARG002
        if self._idx >= len(self._actions):
            raise StopIteration("ScriptedPlanner: action list exhausted")
        action = self._actions[self._idx]
        self._idx += 1
        return action


# ---------------------------------------------------------------------------
# HeuristicPlanner
# ---------------------------------------------------------------------------

_SCAN_INTERVAL: int = 4  # scan every N steps if no detection (HeuristicPlanner only)


class HeuristicPlanner:
    """Deterministic rule-based planner; no LLM calls.

    Strategy:
    - step 0 → scan
    - detection present and path clear → approach (distance 0.3 m)
    - every SCAN_INTERVAL steps → scan
    - advance blocked → gap search: rotate 45° and probe until advance succeeds
      or a full 360° is exhausted (then scan)
    - otherwise → advance 0.5 m; after 3 in a row rotate 90°
    """

    def __init__(self) -> None:
        self._consecutive_advances: int = 0
        self._gap_search_active: bool = False
        self._gap_rotations: int = 0

    def plan(self, ctx: PlanContext) -> Action:
        step = ctx.step
        recent = list(ctx.recent_steps)

        # Out of budget?
        if step >= ctx.max_steps - 1:
            return Action(
                kind="report_not_found",
                reasoning="budget nearly exhausted; declaring not found",
                confidence=0.9,
            )

        # First step: scan.
        if step == 0:
            return Action(kind="scan", reasoning="initial survey scan", confidence=1.0)

        # Was any recent move blocked? Look back 3 steps so a crash-recovery
        # takeoff between the blocked advance and the next plan() call doesn't
        # reset the gap-search trigger.
        last_blocked = any(
            r["action_kind"] in ("advance", "approach") and r["result_status"] == "clamped"
            for r in recent[-3:]
        )

        # Exit gap search if a recent advance succeeded (gap found).
        if self._gap_search_active and any(
            r["action_kind"] == "advance" and r["result_status"] == "reached" for r in recent[-3:]
        ):
            self._gap_search_active = False
            self._gap_rotations = 0

        # Enter gap search on first block.
        if last_blocked and not self._gap_search_active:
            self._gap_search_active = True
            self._gap_rotations = 0

        # Target detected — only approach when the path appears clear.
        if ctx.detections and not self._gap_search_active and not last_blocked:
            self._consecutive_advances = 0
            return Action(
                kind="approach",
                distance_m=0.3,
                reasoning=f"detection {ctx.detections[0].label!r} in view; approaching",
                confidence=0.85,
            )

        # Periodic scan to look around.
        if step % _SCAN_INTERVAL == 0:
            return Action(
                kind="scan",
                reasoning=f"periodic scan at step {step}",
                confidence=0.9,
            )

        # Gap search: rotate 45° increments until an advance goes through.
        if self._gap_search_active:
            # Look back up to 3 steps for the most recent non-takeoff action.
            last_kind = ""
            for r in reversed(recent[-3:]):
                if r["action_kind"] != "takeoff":
                    last_kind = r["action_kind"]
                    break
            if last_kind == "rotate":
                # Just rotated — probe whether the path is now clear.
                self._consecutive_advances = 0
                return Action(
                    kind="advance",
                    distance_m=0.5,
                    reasoning=f"gap search — probing after {self._gap_rotations} rotation(s)",
                    confidence=0.8,
                )
            # Last was a blocked move — rotate 45° to find the gap.
            self._gap_rotations += 1
            if self._gap_rotations > 8:
                # Full 360° with no success — give up and scan for new context.
                self._gap_search_active = False
                self._gap_rotations = 0
                return Action(
                    kind="scan",
                    reasoning="gap search exhausted — scanning for new direction",
                    confidence=0.7,
                )
            return Action(
                kind="rotate",
                angle_deg=45.0,
                reasoning=f"gap search — rotating 45° (attempt {self._gap_rotations}/8)",
                confidence=0.9,
            )

        # After too many advances in a row, rotate to explore a new direction.
        if self._consecutive_advances >= 3:
            self._consecutive_advances = 0
            return Action(
                kind="rotate",
                angle_deg=90.0,
                reasoning="rotating to explore new direction after repeated advances",
                confidence=0.8,
            )

        # Default: advance.
        self._consecutive_advances += 1
        return Action(
            kind="advance",
            distance_m=0.5,
            reasoning="exploring — advancing to cover new ground",
            confidence=0.7,
        )


# ---------------------------------------------------------------------------
# LLMPlanner
# ---------------------------------------------------------------------------

_FALLBACK_ACTION = Action(
    kind="scan",
    reasoning="LLM output invalid after retry — falling back to scan",
    confidence=0.3,
)


def _extract_usage(raw_msg: object | None) -> dict:
    """Extract token counts from a LangChain AIMessage."""
    if raw_msg is None:
        return {}
    meta = getattr(raw_msg, "response_metadata", {}) or {}
    tu: dict = meta.get("token_usage") or meta.get("usage") or {}
    if not tu:
        # Newer LangChain stores usage in usage_metadata.
        um = getattr(raw_msg, "usage_metadata", {}) or {}
        tu = {
            "prompt_tokens": um.get("input_tokens", 0),
            "completion_tokens": um.get("output_tokens", 0),
        }
    prompt = int(tu.get("prompt_tokens", 0))
    completion = int(tu.get("completion_tokens", 0))
    # Approximate cost: gpt-4o-mini rates ($0.15/M input, $0.60/M output).
    est_cost = prompt * 0.15e-6 + completion * 0.60e-6
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "llm_calls": 1,
        "est_cost_usd": est_cost,
    }


class LLMPlanner:
    """Calls an LLM to plan the next action.

    Uses ``with_structured_output(Action, include_raw=True)`` to capture token
    usage alongside the parsed action.  On the first failure it retries once;
    on a second failure it falls back to ``scan``.

    All strategic decisions (where to go, when to scan, how to navigate around
    walls) are delegated to the LLM.  Low-level reflexes (crash recovery,
    approach when target fills frame) live in plan_node, not here.
    """

    def __init__(self, llm: object) -> None:
        self._llm = llm  # ChatOpenAI instance
        self.last_token_usage: dict = {}

    def plan(self, ctx: PlanContext) -> Action:
        chain = self._llm.with_structured_output(Action, include_raw=True)  # type: ignore[union-attr]

        messages = [
            ("system", SYSTEM_PROMPT),
            ("human", self._build_user_message(ctx)),
        ]

        for attempt in range(2):
            try:
                raw_result = chain.invoke(messages)
                parsed = raw_result.get("parsed") if isinstance(raw_result, dict) else raw_result
                parsing_error = (
                    raw_result.get("parsing_error") if isinstance(raw_result, dict) else None
                )
                if parsing_error is not None:
                    raise parsing_error  # type: ignore[misc]
                if not isinstance(parsed, Action):
                    raise ValueError(f"expected Action, got {type(parsed)}")
                raw_msg = raw_result.get("raw") if isinstance(raw_result, dict) else None
                self.last_token_usage = _extract_usage(raw_msg)
                return parsed
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLMPlanner attempt %d failed: %s", attempt + 1, exc)
                if attempt == 0:
                    continue
                break

        logger.warning("LLMPlanner: falling back to scan after 2 failures")
        self.last_token_usage = {}
        return _FALLBACK_ACTION

    @staticmethod
    def _coverage_summary(visited: list[dict]) -> str:
        """Summarise visited grid cells by quadrant and flag unexplored areas."""
        unique = {(c["x_idx"], c["y_idx"]) for c in visited}
        if not unique:
            return "none yet"
        nw = sum(1 for x, y in unique if x < 0 and y >= 0)
        ne = sum(1 for x, y in unique if x >= 0 and y >= 0)
        sw = sum(1 for x, y in unique if x < 0 and y < 0)
        se = sum(1 for x, y in unique if x >= 0 and y < 0)
        total = len(unique)
        quad_str = f"NW={nw} NE={ne} SW={sw} SE={se}"
        unexplored = [q for q, n in [("NW", nw), ("NE", ne), ("SW", sw), ("SE", se)] if n == 0]
        hint = f" — UNEXPLORED: {', '.join(unexplored)}" if unexplored else ""
        return f"{total} cells ({quad_str}){hint}"

    @staticmethod
    def _build_user_message(ctx: PlanContext) -> str:
        dets = ctx.detections
        det_str = (
            f"{len(dets)} detection(s): "
            + "; ".join(f"{d.label} area={d.area_frac:.3f} cx={d.center_norm[0]:.2f}" for d in dets)
            if dets
            else "no detections"
        )
        pose = ctx.drone.pose
        recent = (
            "; ".join(
                f"step{r['step']}:{r['action_kind']}→{r['result_status']}"
                f"@({r['pose_x']:.1f},{r['pose_y']:.1f})"
                for r in ctx.recent_steps[-5:]
            )
            or "none"
        )
        if ctx.safety_events:
            last = ctx.safety_events[-1]
            n = len(ctx.safety_events)
            safety_str = f"[{last.kind}] {last.reason} ({n} total this mission)"
        else:
            safety_str = "none"
        coverage = LLMPlanner._coverage_summary(ctx.visited)
        clamped = [
            f"({r['pose_x']:.1f},{r['pose_y']:.1f})"
            for r in ctx.recent_steps
            if r["result_status"] == "clamped" and r["action_kind"] in ("advance", "approach")
        ]
        wall_hints = "near " + "; ".join(clamped[-5:]) if clamped else "none yet"

        return (
            f"Target: {ctx.target.label}\n"
            f"Step: {ctx.step}/{ctx.max_steps}\n"
            f"Pose: x={pose.x:.2f} y={pose.y:.2f} z={pose.z:.2f} yaw={pose.yaw_deg:.1f}°\n"
            f"Perception: {det_str}\n"
            f"Observation: {ctx.observation}\n"
            f"Recent history: {recent}\n"
            f"Known obstacles (from blocked advances): {wall_hints}\n"
            f"Safety alerts (recent): {safety_str}\n"
            f"Coverage: {coverage}\n"
            f"Summary: {ctx.history_summary or 'none'}\n"
        )
