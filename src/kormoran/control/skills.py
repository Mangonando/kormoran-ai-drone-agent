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
    """Move *distance_m* forward in the drone body frame (+x)."""
    logger.debug("skill advance distance_m=%.2f", distance_m)
    return drone.move_relative(distance_m, 0.0, 0.0, timeout_s)


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
) -> MoveResult:
    """Advance toward a detected target, correcting lateral offset.

    Computes a small forward step combined with a lateral correction derived
    from the detection's horizontal centre offset, so that successive approach
    calls converge the target toward the frame centre.

    Parameters
    ----------
    detection:
        A ``Detection`` returned by the Perceiver; ``center_norm[0]`` is the
        horizontal offset in [-1, 1] (positive = right of centre).
    step_m:
        Forward advance distance (bounded to ``[0.1, 0.5]`` m regardless of
        what the caller passes).
    """
    step_m = max(0.1, min(step_m, 0.5))
    cx = detection.center_norm[0]  # horizontal offset: positive = right
    # Small lateral correction: strafe left when target is right of centre.
    dy = -cx * step_m * 0.5
    logger.debug("skill approach step_m=%.2f cx=%.2f dy=%.2f", step_m, cx, dy)
    return drone.move_relative(step_m, dy, 0.0, timeout_s)
