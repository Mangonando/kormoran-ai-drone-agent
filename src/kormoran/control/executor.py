"""SkillExecutor — maps a validated Action to a skill call with watchdog protection.

The executor:
1. Runs the safety supervisor on every action.
2. Maps the (possibly clamped) action to the right skill function.
3. Wraps the skill call in a threading watchdog: if the skill exceeds
   ``WATCHDOG_MULTIPLIER × timeout_s`` the drone is emergency-stopped and
   a ``MoveResult(status="aborted", reason="watchdog")`` is returned.

The executor keeps a running ``safety_events`` log so callers (the agent's
``act`` node in M4) can append events to the mission state.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

from kormoran.control import skills
from kormoran.interfaces import DroneInterface
from kormoran.safety import supervisor as sup
from kormoran.types import (
    Action,
    Detection,
    DroneState,
    MoveResult,
    Pose,
    SafetyEvent,
    ScanResult,
    Verdict,
)

if TYPE_CHECKING:
    from kormoran.config import Settings

logger = logging.getLogger(__name__)

#: Watchdog fires after this multiple of the nominal skill timeout.
WATCHDOG_MULTIPLIER: float = 2.0

#: Default per-skill timeout in seconds used when the Action doesn't specify one.
DEFAULT_TIMEOUT_S: float = 10.0


class SkillExecutor:
    """Translates validated ``Action`` objects into drone skill calls.

    Parameters
    ----------
    drone:
        Any ``DroneInterface``-conformant backend.
    config:
        Settings instance used to pull geofence bounds for the supervisor.
    """

    def __init__(self, drone: DroneInterface, config: Settings) -> None:
        self._drone = drone
        self._config = config
        self.safety_events: list[SafetyEvent] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        action: Action,
        state: DroneState,
        detection: Detection | None = None,
    ) -> MoveResult | ScanResult:
        """Execute *action* after vetting it through the safety supervisor.

        Returns
        -------
        MoveResult or ScanResult
            ``ScanResult`` only for ``action.kind == "scan"``.  All other kinds
            return ``MoveResult``.
        """
        verdict: Verdict = sup.check(action, state, self._config)
        self.safety_events.extend(verdict.events)

        if verdict.kind == "reject":
            logger.warning(
                "executor: action %r rejected — %s; landing",
                action.kind,
                verdict.reason,
            )
            land_result = self._drone.land()
            final_state = self._drone.get_state()
            return MoveResult(
                status="aborted",
                pose=final_state.pose,
                elapsed_s=land_result.elapsed_s,
                reason=verdict.reason,
            )

        effective = verdict.action  # may be clamped
        if verdict.kind == "clamp":
            logger.debug(
                "executor: action %r clamped — %s",
                action.kind,
                verdict.reason,
            )

        result = self._dispatch(effective, detection)

        # Drift recovery: scan and rotate can cause small physical displacements
        # that push the drone through the geofence margin into the physical wall.
        # If that happened, silently fly back inside before returning so the next
        # perceive_node sees a valid, flyable position.
        current_pose = self._drone.get_state().pose
        self._recover_from_drift(current_pose)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    #: Inward buffer from the geofence edge used as the recovery target.
    _RECOVERY_BUFFER_M: float = 0.2

    def _recover_from_drift(self, pose: Pose) -> None:  # noqa: ANN001
        """If *pose* is outside the geofence, fly back to the nearest in-bounds point."""
        cfg = self._config
        x, y = pose.x, pose.y

        in_x = cfg.geofence_x_min <= x <= cfg.geofence_x_max
        in_y = cfg.geofence_y_min <= y <= cfg.geofence_y_max
        if in_x and in_y:
            return

        buf = self._RECOVERY_BUFFER_M
        tx = max(cfg.geofence_x_min + buf, min(cfg.geofence_x_max - buf, x))
        ty = max(cfg.geofence_y_min + buf, min(cfg.geofence_y_max - buf, y))
        dx_w, dy_w = tx - x, ty - y

        if abs(dx_w) < 0.05 and abs(dy_w) < 0.05:
            return  # close enough — skip to avoid oscillation

        yaw_rad = math.radians(pose.yaw_deg)
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
        dx_b = dx_w * cos_y + dy_w * sin_y
        dy_b = -dx_w * sin_y + dy_w * cos_y

        logger.info(
            "executor: drift recovery (%.2f, %.2f) → (%.2f, %.2f)",
            x,
            y,
            tx,
            ty,
        )
        self._drone.move_relative(dx_b, dy_b, 0.0, timeout_s=5.0)

    def _dispatch(
        self,
        action: Action,
        detection: Detection | None,
    ) -> MoveResult | ScanResult:
        """Route *action* to the matching skill, wrapped in a watchdog."""
        kind = action.kind
        dist = action.distance_m or 0.5
        angle = action.angle_deg or 45.0
        timeout = DEFAULT_TIMEOUT_S

        if kind == "land":
            return self._run_with_watchdog(
                lambda: skills.land(self._drone),
                timeout_s=timeout,
                action_kind=kind,
            )
        if kind == "scan":
            scan_result = self._run_with_watchdog(
                lambda: skills.scan(self._drone),
                timeout_s=timeout * 8,  # scan does 8 rotations
                action_kind=kind,
            )
            return scan_result  # type: ignore[return-value]
        if kind == "advance":
            return self._run_with_watchdog(
                lambda: skills.advance(self._drone, dist, timeout),
                timeout_s=timeout,
                action_kind=kind,
            )
        if kind == "strafe":
            return self._run_with_watchdog(
                lambda: skills.strafe(self._drone, dist, timeout),
                timeout_s=timeout,
                action_kind=kind,
            )
        if kind == "climb":
            return self._run_with_watchdog(
                lambda: skills.climb(self._drone, dist, timeout),
                timeout_s=timeout,
                action_kind=kind,
            )
        if kind == "descend":
            return self._run_with_watchdog(
                lambda: skills.descend(self._drone, dist, timeout),
                timeout_s=timeout,
                action_kind=kind,
            )
        if kind == "rotate":
            return self._run_with_watchdog(
                lambda: skills.rotate(self._drone, angle, timeout),
                timeout_s=timeout,
                action_kind=kind,
            )
        if kind == "approach":
            if detection is None:
                state = self._drone.get_state()
                return MoveResult(
                    status="aborted",
                    pose=state.pose,
                    elapsed_s=0.0,
                    reason="approach called without a detection",
                )
            det = detection  # capture for lambda
            return self._run_with_watchdog(
                lambda: skills.approach(self._drone, det, dist, timeout),
                timeout_s=timeout,
                action_kind=kind,
            )
        if kind == "report_not_found":
            state = self._drone.get_state()
            return MoveResult(status="reached", pose=state.pose, elapsed_s=0.0)

        msg = f"unknown action kind: {kind!r}"
        raise ValueError(msg)

    def _run_with_watchdog(
        self,
        fn: Callable[[], MoveResult | ScanResult],
        timeout_s: float,
        action_kind: str,
    ) -> MoveResult | ScanResult:
        """Run *fn* on a thread, killing the drone if it doesn't finish in time."""
        result_box: list[MoveResult | ScanResult | None] = [None]
        exc_box: list[BaseException | None] = [None]

        def _target() -> None:
            try:
                result_box[0] = fn()
            except Exception as exc:  # noqa: BLE001
                exc_box[0] = exc

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(timeout=timeout_s * WATCHDOG_MULTIPLIER)

        if thread.is_alive():
            logger.error(
                "watchdog: %r skill exceeded %.1f s — emergency stop",
                action_kind,
                timeout_s * WATCHDOG_MULTIPLIER,
            )
            self._drone.emergency_stop()
            ev = SafetyEvent(
                t=0.0,  # no reliable time source here without state
                kind="watchdog",
                action_kind=action_kind,
                reason=(
                    f"skill {action_kind!r} timed out after {timeout_s * WATCHDOG_MULTIPLIER:.1f} s"
                ),
            )
            self.safety_events.append(ev)
            state = self._drone.get_state()
            return MoveResult(
                status="aborted",
                pose=state.pose,
                elapsed_s=timeout_s * WATCHDOG_MULTIPLIER,
                reason="watchdog timeout",
            )

        if exc_box[0] is not None:
            raise exc_box[0]  # type: ignore[misc]

        assert result_box[0] is not None  # noqa: S101
        return result_box[0]
