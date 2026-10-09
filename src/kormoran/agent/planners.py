"""Planner implementations for the Kormoran mission loop.

Three planners are provided:
- ScriptedPlanner  — replays a fixed list of Actions (testing / demos)
- HeuristicPlanner — deterministic rule-based planner (no LLM)
- LLMPlanner       — delegates to an LLM via OpenRouter (uses langchain_openai)
"""

from __future__ import annotations

import logging
import math

from pydantic import BaseModel

from kormoran.agent.prompts import SYSTEM_PROMPT
from kormoran.types import Action, Detection, DroneState, Pose, SafetyEvent, TargetSpec

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

_SCAN_INTERVAL: int = 4  # scan every N steps if no detection (Heuristic + LLM)


class HeuristicPlanner:
    """Deterministic rule-based planner; no LLM calls.

    Strategy:
    - step 0 → scan
    - detection present → approach (distance 0.3 m)
    - every SCAN_INTERVAL steps → scan
    - otherwise → advance 0.5 m; if advance blocked, rotate 45°
    """

    def __init__(self) -> None:
        self._consecutive_advances: int = 0

    def plan(self, ctx: PlanContext) -> Action:
        step = ctx.step

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

        # Target detected — approach it.
        if ctx.detections:
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

    A mandatory scan reflex fires every ``_SCAN_INTERVAL`` steps (when no
    target is currently detected) so the drone surveys the scene regularly
    regardless of what the LLM would otherwise choose.
    """

    def __init__(self, llm: object) -> None:
        self._llm = llm  # ChatOpenAI instance
        self.last_token_usage: dict = {}
        self._last_geo_count: int = 0  # total geofence events seen at end of last plan()
        self._consecutive_geo: int = 0  # new geofence events since last clear step
        self._committed_advances: int = 0  # forced advance burst remaining steps
        self._burst_target_yaw: float | None = None  # heading the burst must face

    def plan(self, ctx: PlanContext) -> Action:
        # Track NEW geofence events since the last plan() call (not accumulated total).
        current_geo_count = sum(1 for e in ctx.safety_events if e.kind == "geofence")
        new_geo = current_geo_count - self._last_geo_count
        if new_geo > 0:
            self._consecutive_geo += new_geo
        else:
            self._consecutive_geo = 0
        self._last_geo_count = current_geo_count

        # Hard override: detection visible — always approach, never let the LLM rotate away.
        if ctx.detections:
            best = max(ctx.detections, key=lambda d: d.confidence * d.area_frac)
            self._committed_advances = 0
            self._burst_target_yaw = None
            self.last_token_usage = {}
            return Action(
                kind="approach",
                distance_m=0.3,
                reasoning=f"detection {best.label!r} visible — approaching immediately",
                confidence=0.95,
            )

        # Mandatory periodic scan: don't rely on the LLM following prompt rules.
        if ctx.step > 0 and ctx.step % _SCAN_INTERVAL == 0 and not ctx.detections:
            self.last_token_usage = {}
            return Action(
                kind="scan",
                reasoning=f"periodic survey scan at step {ctx.step}",
                confidence=1.0,
            )

        # Consecutive rejection guard: if 2+ new geofence rejections in a row,
        # force a large rotation, abort any active burst, and reset the counter.
        if self._consecutive_geo >= 2:
            self._consecutive_geo = 0
            self._committed_advances = 0
            self._burst_target_yaw = None
            self.last_token_usage = {}
            logger.info(
                "LLMPlanner: %d consecutive geofence rejections — forcing 135° rotation",
                current_geo_count,
            )
            return Action(
                kind="rotate",
                angle_deg=135.0,
                reasoning="consecutive geofence rejections — rotating to find a clear path",
                confidence=1.0,
            )

        # Committed advance burst: drain remaining forced advances before consulting the LLM.
        # If the drone's heading is >45° off the target yaw, rotate toward it first
        # (without consuming a burst step) so advances go in the right direction.
        if self._committed_advances > 0:
            target_yaw = self._burst_target_yaw
            if target_yaw is not None:
                delta = target_yaw - ctx.drone.pose.yaw_deg
                while delta > 180.0:
                    delta -= 360.0
                while delta <= -180.0:
                    delta += 360.0
                if abs(delta) > 45.0:
                    self.last_token_usage = {}
                    logger.info(
                        "LLMPlanner: burst realign — rotating %.0f° toward target yaw %.0f°",
                        delta,
                        target_yaw,
                    )
                    return Action(
                        kind="rotate",
                        angle_deg=delta,
                        reasoning=f"burst realign — rotating to face target ({target_yaw:.0f}°)",
                        confidence=1.0,
                    )
            self._committed_advances -= 1
            self.last_token_usage = {}
            logger.info(
                "LLMPlanner: committed advance burst (%d remaining)", self._committed_advances
            )
            return Action(
                kind="advance",
                distance_m=1.0,
                reasoning="committed advance burst — travelling to unexplored area",
                confidence=1.0,
            )

        # No-advance guard: if the last 5 steps contain no advance the drone is
        # spinning/scanning in place — start a 3-step committed advance burst (3 m total).
        # Compute the target heading now (from coverage) so the burst goes toward the least
        # explored area, not toward whatever wall the scan happened to leave the drone facing.
        recent_kinds = [r["action_kind"] for r in ctx.recent_steps[-5:]]
        if len(recent_kinds) >= 5 and not any(k == "advance" for k in recent_kinds):
            target_yaw = self._least_visited_quadrant_yaw(
                ctx.visited,
                ctx.drone.pose,
                ctx.geofence_x_min,
                ctx.geofence_x_max,
                ctx.geofence_y_min,
                ctx.geofence_y_max,
            )
            self._burst_target_yaw = target_yaw
            self._committed_advances = 3
            self.last_token_usage = {}
            logger.info(
                "LLMPlanner: no advance in last 5 steps — starting burst toward %.0f°",
                target_yaw,
            )
            # Rotate toward the target direction first; advances will follow via burst-drain.
            delta = target_yaw - ctx.drone.pose.yaw_deg
            while delta > 180.0:
                delta -= 360.0
            while delta <= -180.0:
                delta += 360.0
            return Action(
                kind="rotate",
                angle_deg=delta,
                reasoning=f"no advance in last 5 steps — aligning to {target_yaw:.0f}° for burst",
                confidence=1.0,
            )

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
        # Quadrant counts (grid origin = world origin; NE = positive x AND y).
        nw = sum(1 for x, y in unique if x < 0 and y >= 0)
        ne = sum(1 for x, y in unique if x >= 0 and y >= 0)
        sw = sum(1 for x, y in unique if x < 0 and y < 0)
        se = sum(1 for x, y in unique if x >= 0 and y < 0)
        total = len(unique)
        quad_str = f"NW={nw} NE={ne} SW={sw} SE={se}"
        # Flag quadrants with zero or very few visits.
        unexplored = [q for q, n in [("NW", nw), ("NE", ne), ("SW", sw), ("SE", se)] if n == 0]
        hint = f" — UNEXPLORED: {', '.join(unexplored)}" if unexplored else ""
        return f"{total} cells ({quad_str}){hint}"

    @staticmethod
    def _least_visited_quadrant_yaw(
        visited: list[dict],
        pose: Pose,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        advance_m: float = 1.0,
    ) -> float:
        """Return the yaw (degrees) toward the least-visited quadrant that is reachable.

        Filters out headings where a single advance step would immediately hit the
        geofence so the burst doesn't repeatedly slam into a corner wall.

        Quadrant → world heading mapping (advance dx=cos(yaw), dy=sin(yaw)):
          NW (x<0, y≥0) → 135°   SW (x<0, y<0) → -135°
          NE (x≥0, y≥0) →  45°   SE (x≥0, y<0) →  -45°

        Fallback: if all quadrant diagonals are blocked, aim toward room centre.
        """
        unique = {(c["x_idx"], c["y_idx"]) for c in visited}
        counts = {
            "NW": sum(1 for x, y in unique if x < 0 and y >= 0),
            "NE": sum(1 for x, y in unique if x >= 0 and y >= 0),
            "SW": sum(1 for x, y in unique if x < 0 and y < 0),
            "SE": sum(1 for x, y in unique if x >= 0 and y < 0),
        }
        _yaw = {"NW": 135.0, "NE": 45.0, "SW": -135.0, "SE": -45.0}

        margin = 0.4  # keep a little extra clear of the exact geofence edge
        safe_x_min = x_min + margin
        safe_x_max = x_max - margin
        safe_y_min = y_min + margin
        safe_y_max = y_max - margin

        for quad in sorted(counts, key=lambda q: counts[q]):
            yaw_rad = math.radians(_yaw[quad])
            nx = pose.x + math.cos(yaw_rad) * advance_m
            ny = pose.y + math.sin(yaw_rad) * advance_m
            if safe_x_min <= nx <= safe_x_max and safe_y_min <= ny <= safe_y_max:
                return _yaw[quad]

        # All quadrant diagonals blocked — head toward room centre.
        return math.degrees(math.atan2(-pose.y, -pose.x))

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
                for r in ctx.recent_steps[-5:]
            )
            or "none"
        )
        # Show only the most recent safety event with total count.
        # Showing 3 accumulated events confused the LLM into over-rotating on stale alerts.
        if ctx.safety_events:
            last = ctx.safety_events[-1]
            n = len(ctx.safety_events)
            safety_str = f"[{last.kind}] {last.reason} ({n} total this mission)"
        else:
            safety_str = "none"
        coverage = LLMPlanner._coverage_summary(ctx.visited)
        return (
            f"Target: {ctx.target.label}\n"
            f"Step: {ctx.step}/{ctx.max_steps}\n"
            f"Pose: x={pose.x:.2f} y={pose.y:.2f} z={pose.z:.2f} yaw={pose.yaw_deg:.1f}°\n"
            f"Perception: {det_str}\n"
            f"Observation: {ctx.observation}\n"
            f"Recent history: {recent}\n"
            f"Safety alerts (recent): {safety_str}\n"
            f"Coverage: {coverage}\n"
            f"Summary: {ctx.history_summary or 'none'}\n"
        )
