"""Safety supervisor — pure function that vets every proposed Action.

``check`` is intentionally stateless: it takes the current ``DroneState`` and
``Settings`` and returns a ``Verdict``.  No side effects, no I/O.  This makes
it trivially testable and independent of any specific drone backend or LLM.

Checks performed (in order):
1. Battery reserve — if battery is at or below ``RESERVE_PCT``, force a land.
2. Geofence — predict the end pose; clamp the move distance so the drone stays
   inside the configured bounding box.  Reject if the clamped distance would be
   smaller than the action's lower bound (0.2 m).
3. Altitude bounds — same clamping logic applied to z alone.

Any clamping or rejection produces a ``SafetyEvent`` in the returned ``Verdict``.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from kormoran.safety.battery import RESERVE_PCT, is_low
from kormoran.types import Action, DroneState, Pose, SafetyEvent, Verdict

if TYPE_CHECKING:
    from kormoran.config import Settings

logger = logging.getLogger(__name__)

# Minimum distance that a clamped move must still achieve (metres).
_MIN_CLAMPED_M: float = 0.05

# Reasoning / confidence placeholders injected into synthesised land actions.
_LAND_REASONING: str = "safety supervisor forced landing"
_LAND_CONFIDENCE: float = 1.0


def check(action: Action, state: DroneState, config: Settings) -> Verdict:
    """Vet *action* against the current *state* and geofence *config*.

    Returns a ``Verdict`` with kind ``"allow"``, ``"clamp"``, or ``"reject"``.
    A rejection always replaces *action* with a ``land`` command so the executor
    can act on the verdict without branching.
    """
    events: list[SafetyEvent] = []

    # ------------------------------------------------------------------
    # 1. Battery check — highest priority
    # ------------------------------------------------------------------
    if is_low(state.battery_pct):
        ev = SafetyEvent(
            t=state.t,
            kind="battery",
            action_kind=action.kind,
            reason=f"battery {state.battery_pct:.1f}% ≤ reserve {RESERVE_PCT}%",
        )
        events.append(ev)
        logger.warning("supervisor reject: %s", ev.reason)
        land_action = _make_land_action()
        return Verdict(kind="reject", action=land_action, reason=ev.reason, events=events)

    # ------------------------------------------------------------------
    # 2. Geofence + altitude checks
    # ------------------------------------------------------------------
    action, events = _check_spatial(action, state.pose, config, events, state.t)

    # If _check_spatial replaced the action with a safety fallback, return as reject.
    # land = hard stop (battery / altitude); rotate = soft reorient (geofence horizontal).
    if events and action.kind in ("land", "rotate"):
        return Verdict(
            kind="reject",
            action=action,
            reason=events[-1].reason,
            events=events,
        )

    verdict_kind = "clamp" if events else "allow"
    return Verdict(kind=verdict_kind, action=action, reason=_summary(events), events=events)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_land_action() -> Action:
    return Action(
        kind="land",
        reasoning=_LAND_REASONING,
        confidence=_LAND_CONFIDENCE,
    )


def _make_rotate_action(reason: str) -> Action:
    """Rotate 90° away from the wall — faster than a full scan and changes heading."""
    return Action(
        kind="rotate",
        angle_deg=90.0,
        reasoning=f"safety supervisor: {reason} — rotating to find a clear direction",
        confidence=1.0,
    )


def _summary(events: list[SafetyEvent]) -> str:
    if not events:
        return "ok"
    return "; ".join(e.reason for e in events)


def _check_spatial(
    action: Action,
    pose: Pose,
    config: Settings,
    events: list[SafetyEvent],
    t: float,
) -> tuple[Action, list[SafetyEvent]]:
    """Apply geofence and altitude clamping.  Returns the (possibly modified) action."""
    kind = action.kind

    if kind in ("land", "report_not_found", "scan"):
        # These actions don't translate to a spatial displacement.
        return action, events

    if kind in ("advance", "strafe"):
        return _check_geofence_horizontal(action, pose, config, events, t)

    if kind == "climb":
        dist = action.distance_m or 0.5
        return _check_altitude(action, pose, config, events, t, dz=dist)

    if kind == "descend":
        dist = action.distance_m or 0.5
        return _check_altitude(action, pose, config, events, t, dz=-dist)

    if kind == "rotate":
        # Rotation doesn't change position; no spatial check needed.
        return action, events

    if kind == "approach":
        # Approach advances a small bounded step; treat it like advance.
        return _check_geofence_horizontal(action, pose, config, events, t)

    return action, events


def _check_altitude(
    action: Action,
    pose: Pose,
    config: Settings,
    events: list[SafetyEvent],
    t: float,
    dz: float,
) -> tuple[Action, list[SafetyEvent]]:
    """Clamp *dz* so z stays within [geofence_z_min, geofence_z_max]."""
    z_min = config.geofence_z_min
    z_max = config.geofence_z_max
    z_end = pose.z + dz

    if z_min <= z_end <= z_max:
        return action, events

    clamped_dz = _clamp_delta(pose.z, dz, z_min, z_max)

    if abs(clamped_dz) < _MIN_CLAMPED_M:
        ev = SafetyEvent(
            t=t,
            kind="altitude",
            action_kind=action.kind,
            reason=(
                f"altitude {pose.z:.2f} m already at limit "
                f"({'min' if dz < 0 else 'max'} {z_min if dz < 0 else z_max} m)"
            ),
        )
        events.append(ev)
        logger.warning("supervisor reject: %s", ev.reason)
        return _make_land_action(), events

    # Clamp the distance_m field.
    ev = SafetyEvent(
        t=t,
        kind="altitude",
        action_kind=action.kind,
        reason=(
            f"altitude clamped from {pose.z + dz:.2f} m to "
            f"{pose.z + clamped_dz:.2f} m (limit {z_min}–{z_max} m)"
        ),
    )
    events.append(ev)
    logger.debug("supervisor clamp: %s", ev.reason)
    new_distance = abs(clamped_dz)
    return action.model_copy(update={"distance_m": new_distance}), events


def _check_geofence_horizontal(
    action: Action,
    pose: Pose,
    config: Settings,
    events: list[SafetyEvent],
    t: float,
) -> tuple[Action, list[SafetyEvent]]:
    """Clamp *distance_m* so the drone stays inside the horizontal geofence."""
    dist = action.distance_m or 0.5
    yaw_rad = math.radians(pose.yaw_deg)

    if action.kind == "advance":
        dx_w = dist * math.cos(yaw_rad)
        dy_w = dist * math.sin(yaw_rad)
    elif action.kind in ("strafe", "approach"):
        # strafe: body +y; approach approximates a forward move
        dx_w = dist * math.cos(yaw_rad) if action.kind == "approach" else -dist * math.sin(yaw_rad)
        dy_w = dist * math.sin(yaw_rad) if action.kind == "approach" else dist * math.cos(yaw_rad)
    else:
        return action, events

    x_min, x_max = config.geofence_x_min, config.geofence_x_max
    y_min, y_max = config.geofence_y_min, config.geofence_y_max

    x_end = pose.x + dx_w
    y_end = pose.y + dy_w

    if x_min <= x_end <= x_max and y_min <= y_end <= y_max:
        return action, events

    # Compute maximum fraction of the move that keeps the drone inside.
    scale = _max_scale_2d(pose.x, pose.y, dx_w, dy_w, x_min, x_max, y_min, y_max)
    clamped_dist = scale * dist

    if clamped_dist < _MIN_CLAMPED_M:
        reason = (
            f"geofence: move would reach ({x_end:.2f}, {y_end:.2f}) m, "
            f"outside [{x_min}, {x_max}] × [{y_min}, {y_max}]; "
            "no safe distance available — rotating to find a clear direction"
        )
        ev = SafetyEvent(
            t=t,
            kind="geofence",
            action_kind=action.kind,
            reason=reason,
        )
        events.append(ev)
        logger.warning("supervisor reject: %s", ev.reason)
        return _make_rotate_action(reason), events

    ev = SafetyEvent(
        t=t,
        kind="geofence",
        action_kind=action.kind,
        reason=(
            f"geofence: distance clamped {dist:.2f}→{clamped_dist:.2f} m "
            f"(end ({x_end:.2f}, {y_end:.2f}) outside bounds)"
        ),
    )
    events.append(ev)
    logger.debug("supervisor clamp: %s", ev.reason)
    return action.model_copy(update={"distance_m": clamped_dist}), events


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _clamp_delta(start: float, delta: float, lo: float, hi: float) -> float:
    """Return the maximum *delta* such that ``start + delta`` stays in [lo, hi]."""
    if delta > 0:
        return min(delta, hi - start)
    if delta < 0:
        return max(delta, lo - start)
    return 0.0


def _max_scale_2d(
    x: float,
    y: float,
    dx: float,
    dy: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> float:
    """Return the maximum scale s ∈ [0, 1] such that (x+s·dx, y+s·dy) is in bounds."""
    s = 1.0

    if dx > 0:
        s = min(s, (x_max - x) / dx)
    elif dx < 0:
        s = min(s, (x_min - x) / dx)

    if dy > 0:
        s = min(s, (y_max - y) / dy)
    elif dy < 0:
        s = min(s, (y_min - y) / dy)

    return max(0.0, s)
