"""SimDrone — DroneInterface backed by gym-pybullet-drones.

Wraps ``CtrlAviary`` (one drone) + ``DSLPIDControl`` to provide the same
body-frame, metres/degrees API that all other backends expose.

VERIFY items confirmed in M1:
- gym-pybullet-drones installs under uv + Python 3.12 after patching
  pybullet 3.2.7's vendored zutil.h (see docs/decisions.md).
- ``CtrlAviary`` does not expose ``vision_attributes``; we post-hoc patch
  ``IMG_RES`` to enable ``_getDroneImages``.
- Frames are non-black in DIRECT mode (headless) with the patch applied.
"""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from kormoran.types import DroneState, Frame, MoveResult, Pose

if TYPE_CHECKING:
    from kormoran.config import ScenarioConfig

logger = logging.getLogger(__name__)

# Camera resolution for _getDroneImages
_IMG_W: int = 128
_IMG_H: int = 96
_IMG_FOV_DEG: float = 60.0

# PID convergence tolerances
_TOL_POS_M: float = 0.05  # position within 5 cm
_TOL_YAW_RAD: float = math.radians(5.0)  # yaw within 5 degrees

# Battery drain model (sim only)
_BATTERY_DRAIN_PCT_PER_S: float = 0.05  # 0.05 % per second → ~33 min


def _state_to_pose(state: NDArray[np.float64]) -> Pose:
    """Convert a gym-pybullet-drones state vector to a Pose."""
    return Pose(
        x=float(state[0]),
        y=float(state[1]),
        z=float(state[2]),
        yaw_deg=math.degrees(float(state[9])),
    )


def _body_to_world(dx_b: float, dy_b: float, yaw_rad: float) -> tuple[float, float]:
    """Rotate a body-frame (forward, left) displacement to world frame."""
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    return dx_b * cos_y - dy_b * sin_y, dx_b * sin_y + dy_b * cos_y


class SimDrone:
    """Drone backend wrapping CtrlAviary + DSLPIDControl.

    Parameters
    ----------
    scenario:
        The scenario config that defines the room, geofence, and objects.
    gui:
        Whether to show the PyBullet GUI (set ``False`` for headless tests).
    seed:
        Random seed forwarded to the scene builder.
    """

    def __init__(
        self,
        scenario: ScenarioConfig,
        gui: bool = False,
        seed: int = 0,
    ) -> None:
        self._scenario = scenario
        self._gui = gui
        self._seed = seed

        self._env: object | None = None  # CtrlAviary, typed as object to avoid top-level import
        self._ctrl: object | None = None  # DSLPIDControl
        self._is_flying: bool = False
        self._connect_time: float = 0.0

    # ------------------------------------------------------------------
    # DroneInterface
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Create the PyBullet environment and DSLPIDControl, then build the scene."""
        # Deferred import keeps the top-level import clean and allows the rest of
        # the codebase to run without pybullet installed.
        from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
        from gym_pybullet_drones.envs import CtrlAviary
        from gym_pybullet_drones.utils.enums import DroneModel, Physics

        from kormoran.scenes.builder import SceneBuilder

        sp = self._scenario.start_pose
        initial_xyz = np.array([[sp[0], sp[1], 0.05]])  # 5 cm above ground

        self._env = CtrlAviary(
            num_drones=1,
            physics=Physics.PYB,
            gui=self._gui,
            initial_xyzs=initial_xyz,
            user_debug_gui=False,
        )
        self._env.reset()  # type: ignore[union-attr]

        # Post-hoc vision patch (CtrlAviary doesn't forward vision_attributes)
        self._env.IMG_RES = np.array([_IMG_W, _IMG_H])  # type: ignore[union-attr]
        self._env.IMG_FRAME_PER_SEC = 24  # type: ignore[union-attr]
        self._env.IMG_CAPTURE_FREQ = int(self._env.PYB_FREQ / 24)  # type: ignore[union-attr]
        self._env.rgb = np.zeros((1, _IMG_H, _IMG_W, 4))  # type: ignore[union-attr]
        self._env.dep = np.ones((1, _IMG_H, _IMG_W))  # type: ignore[union-attr]
        self._env.seg = np.zeros((1, _IMG_H, _IMG_W))  # type: ignore[union-attr]

        self._ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)

        # Build room + objects inside the existing PyBullet world
        builder = SceneBuilder(self._scenario, self._seed)
        builder.build(self._env.CLIENT)  # type: ignore[union-attr]

        self._connect_time = time.monotonic()
        logger.info("SimDrone connected (gui=%s, scene=%s)", self._gui, self._scenario.scene_id)

    def disconnect(self) -> None:
        if self._env is not None:
            self._env.close()  # type: ignore[union-attr]
            self._env = None
            self._ctrl = None
        logger.info("SimDrone disconnected")

    def takeoff(self, altitude_m: float) -> MoveResult:
        self._require_connected()
        state = self._get_state_vec()
        target = np.array([state[0], state[1], altitude_m])
        result = self._fly_to(target, tol=_TOL_POS_M, timeout_s=15.0)
        if result.status != "aborted":
            self._is_flying = True
        return result

    def land(self) -> MoveResult:
        self._require_connected()
        state = self._get_state_vec()
        target = np.array([state[0], state[1], 0.1])
        result = self._fly_to(target, tol=0.08, timeout_s=20.0)
        self._is_flying = False
        return result

    def emergency_stop(self) -> None:
        if self._env is not None:
            zero = np.zeros((1, 4))
            self._env.step(zero)  # type: ignore[union-attr]
        self._is_flying = False
        logger.warning("SimDrone emergency stop")

    def move_relative(
        self,
        dx_m: float,
        dy_m: float,
        dz_m: float,
        timeout_s: float,
    ) -> MoveResult:
        self._require_connected()
        state = self._get_state_vec()
        yaw_rad = float(state[9])
        dx_w, dy_w = _body_to_world(dx_m, dy_m, yaw_rad)
        target = np.array([state[0] + dx_w, state[1] + dy_w, state[2] + dz_m])
        return self._fly_to(target, tol=_TOL_POS_M, timeout_s=timeout_s, hold_yaw=yaw_rad)

    def rotate(self, yaw_deg: float, timeout_s: float) -> MoveResult:
        self._require_connected()
        state = self._get_state_vec()
        current_pos = state[0:3].copy()
        target_yaw = float(state[9]) + math.radians(yaw_deg)
        return self._rotate_to(current_pos, target_yaw, timeout_s)

    def get_state(self) -> DroneState:
        self._require_connected()
        state = self._get_state_vec()
        elapsed = time.monotonic() - self._connect_time
        battery = max(0.0, 100.0 - elapsed * _BATTERY_DRAIN_PCT_PER_S)
        return DroneState(
            pose=_state_to_pose(state),
            battery_pct=battery,
            is_flying=self._is_flying,
            t=elapsed,
        )

    def get_frame(self) -> Frame:
        self._require_connected()
        env = self._env
        rgba_raw, _, _ = env._getDroneImages(0, segmentation=False)  # type: ignore[union-attr]
        # RGBA int64 → RGB uint8
        rgba: NDArray[np.uint8] = np.array(rgba_raw, dtype=np.uint8)
        rgb: NDArray[np.uint8] = rgba[:, :, :3]
        elapsed = time.monotonic() - self._connect_time
        return Frame(image=rgb, t=elapsed, fov_deg=_IMG_FOV_DEG)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_connected(self) -> None:
        if self._env is None or self._ctrl is None:
            msg = "SimDrone is not connected — call connect() first"
            raise RuntimeError(msg)

    def _get_state_vec(self) -> NDArray[np.float64]:
        return self._env._getDroneStateVector(0)  # type: ignore[union-attr]

    def _step(self, target_pos: NDArray[np.float64], target_rpy: NDArray[np.float64]) -> None:
        """Compute RPMs via DSLPIDControl and advance the sim by one step."""
        state = self._get_state_vec()
        rpm, _, _ = self._ctrl.computeControlFromState(  # type: ignore[union-attr]
            control_timestep=1.0 / self._env.CTRL_FREQ,  # type: ignore[union-attr]
            state=state,
            target_pos=target_pos,
            target_rpy=target_rpy,
        )
        self._env.step(np.array([rpm]))  # type: ignore[union-attr]

    def _fly_to(
        self,
        target_pos: NDArray[np.float64],
        tol: float,
        timeout_s: float,
        hold_yaw: float | None = None,
    ) -> MoveResult:
        """PID loop: run until position within *tol* metres or *timeout_s* exceeded."""
        t_start = time.monotonic()
        if hold_yaw is None:
            hold_yaw = float(self._get_state_vec()[9])
        target_rpy = np.array([0.0, 0.0, hold_yaw])

        while True:
            state = self._get_state_vec()
            dist = float(np.linalg.norm(target_pos - state[0:3]))
            elapsed = time.monotonic() - t_start
            if dist < tol:
                return MoveResult(status="reached", pose=_state_to_pose(state), elapsed_s=elapsed)
            if elapsed > timeout_s:
                return MoveResult(
                    status="clamped",
                    pose=_state_to_pose(state),
                    elapsed_s=elapsed,
                    reason="timeout",
                )
            self._step(target_pos, target_rpy)

    def _rotate_to(
        self,
        hold_pos: NDArray[np.float64],
        target_yaw_rad: float,
        timeout_s: float,
    ) -> MoveResult:
        """PID loop: rotate to *target_yaw_rad* while holding *hold_pos*."""
        t_start = time.monotonic()
        target_rpy = np.array([0.0, 0.0, target_yaw_rad])

        while True:
            state = self._get_state_vec()
            yaw_err = abs(float(state[9]) - target_yaw_rad)
            # Wrap to [-π, π]
            while yaw_err > math.pi:
                yaw_err -= 2 * math.pi
            yaw_err = abs(yaw_err)
            elapsed = time.monotonic() - t_start
            if yaw_err < _TOL_YAW_RAD:
                return MoveResult(status="reached", pose=_state_to_pose(state), elapsed_s=elapsed)
            if elapsed > timeout_s:
                return MoveResult(
                    status="clamped",
                    pose=_state_to_pose(state),
                    elapsed_s=elapsed,
                    reason="timeout",
                )
            self._step(hold_pos, target_rpy)

    def __repr__(self) -> str:
        return f"SimDrone(scene={self._scenario.scene_id!r}, gui={self._gui})"
