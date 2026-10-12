"""Deterministic macro-skills that execute one Action on a DroneInterface.

Each skill is a module-level function that takes a drone backend and returns a
``MoveResult`` (or ``ScanResult`` for the multi-step scan).  Skills are blocking,
bounded by a ``timeout_s`` parameter, and safe to call from the executor's
watchdog thread.

Body-frame convention (matches DroneInterface.move_relative):
  +x = forward, +y = left, +z = up.
"""

from __future__ import annotations

import logging
import math

from kormoran.interfaces import DroneInterface
from kormoran.types import Detection, MoveResult, ScanResult

logger = logging.getLogger(__name__)

# Default timeouts used when the caller doesn't override.
_DEFAULT_MOVE_TIMEOUT_S: float = 10.0
_DEFAULT_SCAN_STEP_TIMEOUT_S: float = 5.0


def takeoff(drone: DroneInterface, altitude_m: float = 1.0) -> MoveResult:
    """Arm and climb to *altitude_m* above the takeoff point."""
    logger.debug("skill takeoff altitude_m=%.2f", altitude_m)
    return drone.takeoff(altitude_m)


def land(drone: DroneInterface) -> MoveResult:
    """Descend and disarm."""
    logger.debug("skill land")
    return drone.land()


def scan(
    drone: DroneInterface,
    n_steps: int = 8,
    step_timeout_s: float = _DEFAULT_SCAN_STEP_TIMEOUT_S,
) -> ScanResult:
    """Rotate 360° in *n_steps* equal increments, capturing one frame per step.

    Returns a ``ScanResult`` with one frame and one yaw per step.  If any
    rotation step is aborted (e.g. watchdog), the scan terminates early.
    """
    logger.debug("skill scan n_steps=%d", n_steps)
    step_deg = 360.0 / n_steps
    frames = []
    yaws_deg = []
    total_elapsed = 0.0

    for i in range(n_steps):
        state = drone.get_state()
        frame = drone.get_frame()
        frames.append(frame)
        yaws_deg.append(state.pose.yaw_deg)
        logger.debug("scan step %d/%d yaw=%.1f°", i + 1, n_steps, state.pose.yaw_deg)

        result = drone.rotate(step_deg, timeout_s=step_timeout_s)
        total_elapsed += result.elapsed_s
        if result.status == "aborted":
            logger.warning("scan aborted at step %d: %s", i + 1, result.reason)
            break

    final_state = drone.get_state()
    return ScanResult(
        frames=frames,
        yaws_deg=yaws_deg,
        elapsed_s=total_elapsed,
        final_pose=final_state.pose,
    )


def advance(
    drone: DroneInterface,
    distance_m: float,
    timeout_s: float = _DEFAULT_MOVE_TIMEOUT_S,
) -> MoveResult:
    """Move *distance_m* forward in the drone body frame (+x).

    Uses a short per-step timeout and checks forward progress so that wall
    collisions are detected quickly instead of burning the full timeout.
    Returns ``clamped / path blocked`` when the drone barely moved forward or
    z dropped significantly (physics-ejection from a wall collision).
    """
    logger.debug("skill advance distance_m=%.2f", distance_m)
    before = drone.get_state().pose
    step_timeout = min(timeout_s, 3.0)
    result = drone.move_relative(distance_m, 0.0, 0.0, step_timeout)

    dz = result.pose.z - before.z
    yaw_rad = math.radians(before.yaw_deg)
    actual_dx = result.pose.x - before.x
    actual_dy = result.pose.y - before.y
    forward_progress = actual_dx * math.cos(yaw_rad) + actual_dy * math.sin(yaw_rad)

    if dz < -0.2 or forward_progress < distance_m * 0.3:
        logger.info(
            "skill advance: path blocked (dz=%.2f fwd=%.2f vs %.2f)",
            dz,
            forward_progress,
            distance_m,
        )
        # Stabilize: if the drone dropped, hover back up to pre-advance
        # altitude to prevent a ground crash on the next step.
        if dz < -0.2:
            recover_alt = before.z - result.pose.z + 0.1
            logger.info(
                "skill advance: stabilizing +%.1fm after wall hit (z=%.2f)",
                recover_alt,
                result.pose.z,
            )
            recover_result = drone.move_relative(0.0, 0.0, recover_alt, 5.0)
            return MoveResult(
                status="clamped",
                pose=recover_result.pose,
                elapsed_s=result.elapsed_s + recover_result.elapsed_s,
                reason="path blocked",
            )
        return MoveResult(
            status="clamped",
            pose=result.pose,
            elapsed_s=result.elapsed_s,
            reason="path blocked",
        )

    return result


def strafe(
    drone: DroneInterface,
    distance_m: float,
    timeout_s: float = _DEFAULT_MOVE_TIMEOUT_S,
) -> MoveResult:
    """Move *distance_m* sideways in the drone body frame (+y = left)."""
    logger.debug("skill strafe distance_m=%.2f", distance_m)
    return drone.move_relative(0.0, distance_m, 0.0, timeout_s)


def climb(
    drone: DroneInterface,
    distance_m: float,
    timeout_s: float = _DEFAULT_MOVE_TIMEOUT_S,
) -> MoveResult:
    """Rise *distance_m* (+z)."""
    logger.debug("skill climb distance_m=%.2f", distance_m)
    return drone.move_relative(0.0, 0.0, distance_m, timeout_s)


def descend(
    drone: DroneInterface,
    distance_m: float,
    timeout_s: float = _DEFAULT_MOVE_TIMEOUT_S,
) -> MoveResult:
    """Drop *distance_m* (-z)."""
    logger.debug("skill descend distance_m=%.2f", distance_m)
    return drone.move_relative(0.0, 0.0, -distance_m, timeout_s)


def rotate(
    drone: DroneInterface,
    angle_deg: float,
    timeout_s: float = _DEFAULT_MOVE_TIMEOUT_S,
) -> MoveResult:
    """Rotate *angle_deg* degrees (positive = counter-clockwise from above)."""
    logger.debug("skill rotate angle_deg=%.1f", angle_deg)
    return drone.rotate(angle_deg, timeout_s)


def approach(
    drone: DroneInterface,
    detection: Detection,
    step_m: float = 0.3,
    timeout_s: float = _DEFAULT_MOVE_TIMEOUT_S,
    perceiver: object | None = None,
    target: object | None = None,
) -> MoveResult:
    """Closed-loop visual servo: rotate to centre, descend if needed, advance.

    When *perceiver* and *target* are provided the skill re-acquires the target
    after each micro-step, giving reliable convergence even when the initial
    detection is far off-centre or near the bottom of the frame.

    Falls back to the original open-loop behaviour when perceiver is ``None``.

    Parameters
    ----------
    detection:
        Initial ``Detection`` from the Perceiver.
    step_m:
        Forward advance distance (bounded to ``[0.1, 0.5]`` m).
    perceiver:
        A ``Perceiver``-conformant object used to re-detect after corrections.
    target:
        A ``TargetSpec`` passed to ``perceiver.detect()``.
    """
    step_m = max(0.1, min(step_m, 0.5))

    # --- Phase 1: rotate until target is horizontally centred ---------------
    cx = detection.center_norm[0]
    for _ in range(4):  # max 4 correction attempts
        if abs(cx) <= 0.15:
            break
        yaw_correction = -cx * 30.0
        logger.debug("skill approach: yaw correction %.1f° (cx=%.2f)", yaw_correction, cx)
        drone.rotate(yaw_correction, timeout_s=3.0)

        # Re-acquire after rotation.
        det = _reacquire(drone, perceiver, target)
        if det is None:
            break
        detection = det
        cx = detection.center_norm[0]

    # --- Phase 2: descend if target is in lower part of frame ---------------
    cy = detection.center_norm[1]
    if cy > 0.5:
        descend_m = min((cy - 0.3) * 0.6, 0.4)
        logger.debug("skill approach: descending %.2fm (cy=%.2f)", descend_m, cy)
        drone.move_relative(0.0, 0.0, -descend_m, timeout_s=3.0)

        det = _reacquire(drone, perceiver, target)
        if det is not None:
            detection = det

    # --- Phase 3: advance forward -------------------------------------------
    dy = -detection.center_norm[0] * step_m * 0.3
    logger.debug("skill approach step_m=%.2f cx=%.2f dy=%.2f", step_m, detection.center_norm[0], dy)

    before = drone.get_state().pose
    step_timeout = min(timeout_s, 3.0)
    result = drone.move_relative(step_m, dy, 0.0, step_timeout)

    # --- Progress check ---
    dz = result.pose.z - before.z
    yaw_rad = math.radians(before.yaw_deg)
    actual_dx = result.pose.x - before.x
    actual_dy = result.pose.y - before.y
    forward_progress = actual_dx * math.cos(yaw_rad) + actual_dy * math.sin(yaw_rad)

    if dz < -0.2 or forward_progress < step_m * 0.3:
        logger.info(
            "skill approach: path blocked (dz=%.2f fwd=%.2f vs %.2f)",
            dz,
            forward_progress,
            step_m,
        )
        if dz < -0.2:
            recover_alt = before.z - result.pose.z + 0.1
            logger.info(
                "skill approach: stabilizing +%.1fm after wall hit (z=%.2f)",
                recover_alt,
                result.pose.z,
            )
            recover_result = drone.move_relative(0.0, 0.0, recover_alt, 5.0)
            return MoveResult(
                status="clamped",
                pose=recover_result.pose,
                elapsed_s=result.elapsed_s + recover_result.elapsed_s,
                reason="path blocked",
            )
        return MoveResult(
            status="clamped",
            pose=result.pose,
            elapsed_s=result.elapsed_s,
            reason="path blocked",
        )

    return result


def _reacquire(
    drone: DroneInterface,
    perceiver: object | None,
    target: object | None,
) -> Detection | None:
    """Take a fresh frame, run the detector, return the best detection or None."""
    if perceiver is None or target is None:
        return None
    frame = drone.get_frame()
    dets: list[Detection] = perceiver.detect(frame, target)  # type: ignore[union-attr]
    if not dets:
        return None
    return max(dets, key=lambda d: d.confidence * d.area_frac)
