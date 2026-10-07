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

    def plan(self, ctx: PlanContext) -> Action:
        # Mandatory periodic scan: don't rely on the LLM following prompt rules.
        if ctx.step > 0 and ctx.step % _SCAN_INTERVAL == 0 and not ctx.detections:
            self.last_token_usage = {}
            return Action(
                kind="scan",
                reasoning=f"periodic survey scan at step {ctx.step}",
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
        return (
            f"Target: {ctx.target.label}\n"
            f"Step: {ctx.step}/{ctx.max_steps}\n"
            f"Pose: x={pose.x:.2f} y={pose.y:.2f} z={pose.z:.2f} yaw={pose.yaw_deg:.1f}°\n"
            f"Perception: {det_str}\n"
            f"Observation: {ctx.observation}\n"
            f"Recent history: {recent}\n"
            f"Summary: {ctx.history_summary or 'none'}\n"
        )
