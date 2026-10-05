"""SceneBuilder — loads a ScenarioConfig into an active PyBullet world.

Must be called from within an active PyBullet physics client (i.e. after
``CtrlAviary.reset()``).  Returns a mapping of ``{label: body_id}`` so the
perception module can look up ground-truth object ids (eval only).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from kormoran.config import ScenarioConfig

logger = logging.getLogger(__name__)

# RGBA colour map for PyBullet visual shapes
_COLORS: dict[str, list[float]] = {
    "red": [1.0, 0.0, 0.0, 1.0],
    "blue": [0.0, 0.0, 1.0, 1.0],
    "green": [0.0, 0.8, 0.0, 1.0],
    "yellow": [1.0, 1.0, 0.0, 1.0],
    "white": [1.0, 1.0, 1.0, 1.0],
    "grey": [0.5, 0.5, 0.5, 1.0],
    "orange": [1.0, 0.5, 0.0, 1.0],
}

_WALL_RGBA: list[float] = [0.8, 0.8, 0.8, 0.6]


class SceneBuilder:
    """Builds a 3-D scene from a ``ScenarioConfig`` inside a PyBullet world.

    Parameters
    ----------
    scenario:
        The loaded scenario configuration.
    seed:
        Random seed for target-position randomisation.
    """

    def __init__(self, scenario: ScenarioConfig, seed: int = 0) -> None:
        self._scenario = scenario
        self._rng = np.random.default_rng(seed)
        self._target_ids: dict[str, int] = {}

    def build(self, physics_client: int) -> dict[str, int]:
        """Populate the PyBullet world and return ``{target_label: body_id}``."""
        import pybullet as p

        room = self._scenario.room
        # Room walls (thin static boxes around the perimeter + ceiling)
        self._build_walls(p, physics_client, room)

        # Obstacles
        for obs in self._scenario.obstacles:
            self._load_obstacle(p, physics_client, obs)

        # Targets
        for tgt in self._scenario.targets:
            pose = self._sample_pose(tgt.zone)
            body_id = self._load_primitive(p, physics_client, tgt.shape, tgt.color, tgt.size, pose)
            self._target_ids[tgt.label] = body_id
            logger.debug("Placed %r at (%.2f, %.2f, %.2f)", tgt.label, *pose)

        return self._target_ids

    @property
    def target_ids(self) -> dict[str, int]:
        """Body IDs keyed by target label (valid after ``build()``)."""
        return dict(self._target_ids)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample_pose(self, zone: list[list[float]]) -> tuple[float, float, float]:
        """Sample a random (x, y, z) within the zone [[xmin,xmax],[ymin,ymax],[zmin,zmax]]."""
        x = float(self._rng.uniform(zone[0][0], zone[0][1]))
        y = float(self._rng.uniform(zone[1][0], zone[1][1]))
        # z might be a fixed value (both ends equal) or a range
        z_lo, z_hi = zone[2][0], zone[2][1]
        z = float(self._rng.uniform(z_lo, z_hi)) if z_hi > z_lo else z_lo
        return (x, y, z)

    def _load_primitive(
        self,
        p: object,
        client: int,
        shape: str,
        color: str,
        size: float,
        pose: tuple[float, float, float],
    ) -> int:
        """Create a coloured primitive (cube/sphere/cylinder) in PyBullet."""
        rgba = _COLORS.get(color.lower(), [0.5, 0.5, 0.5, 1.0])
        half = size / 2.0

        if shape == "cube":
            vis_id = p.createVisualShape(  # type: ignore[attr-defined]
                shapeType=p.GEOM_BOX,  # type: ignore[attr-defined]
                halfExtents=[half, half, half],
                rgbaColor=rgba,
                physicsClientId=client,
            )
            col_id = p.createCollisionShape(  # type: ignore[attr-defined]
                shapeType=p.GEOM_BOX,  # type: ignore[attr-defined]
                halfExtents=[half, half, half],
                physicsClientId=client,
            )
        elif shape == "sphere":
            vis_id = p.createVisualShape(  # type: ignore[attr-defined]
                shapeType=p.GEOM_SPHERE,  # type: ignore[attr-defined]
                radius=half,
                rgbaColor=rgba,
                physicsClientId=client,
            )
            col_id = p.createCollisionShape(  # type: ignore[attr-defined]
                shapeType=p.GEOM_SPHERE,  # type: ignore[attr-defined]
                radius=half,
                physicsClientId=client,
            )
        elif shape == "cylinder":
            vis_id = p.createVisualShape(  # type: ignore[attr-defined]
                shapeType=p.GEOM_CYLINDER,  # type: ignore[attr-defined]
                radius=half / 2.0,
                length=size,
                rgbaColor=rgba,
                physicsClientId=client,
            )
            col_id = p.createCollisionShape(  # type: ignore[attr-defined]
                shapeType=p.GEOM_CYLINDER,  # type: ignore[attr-defined]
                radius=half / 2.0,
                height=size,
                physicsClientId=client,
            )
        else:
            logger.warning("Unknown shape %r — defaulting to box", shape)
            vis_id = p.createVisualShape(  # type: ignore[attr-defined]
                shapeType=p.GEOM_BOX,  # type: ignore[attr-defined]
                halfExtents=[half, half, half],
                rgbaColor=rgba,
                physicsClientId=client,
            )
            col_id = p.createCollisionShape(  # type: ignore[attr-defined]
                shapeType=p.GEOM_BOX,  # type: ignore[attr-defined]
                halfExtents=[half, half, half],
                physicsClientId=client,
            )

        body_id: int = p.createMultiBody(  # type: ignore[attr-defined]
            baseMass=0,  # static
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=list(pose),
            physicsClientId=client,
        )
        return body_id

    def _load_obstacle(self, p: object, client: int, obs: object) -> None:
        """Load one obstacle spec into the world."""
        shape = getattr(obs, "shape", "box")
        size = getattr(obs, "size", [1.0, 1.0, 1.0])
        pose = getattr(obs, "pose", [0.0, 0.0, 0.5])

        if shape == "box":
            half = [s / 2.0 for s in size]
            col_id = p.createCollisionShape(  # type: ignore[attr-defined]
                shapeType=p.GEOM_BOX,
                halfExtents=half,
                physicsClientId=client,  # type: ignore[attr-defined]
            )
            vis_id = p.createVisualShape(  # type: ignore[attr-defined]
                shapeType=p.GEOM_BOX,  # type: ignore[attr-defined]
                halfExtents=half,
                rgbaColor=_COLORS["grey"],
                physicsClientId=client,
            )
        else:
            return  # Only box obstacles in M1

        p.createMultiBody(  # type: ignore[attr-defined]
            baseMass=0,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=pose[:3],
            physicsClientId=client,
        )

    def _build_walls(self, p: object, client: int, room: object) -> None:
        """Create thin static wall boxes around the room perimeter."""
        x_min, x_max = room.x[0], room.x[1]
        y_min, y_max = room.y[0], room.y[1]
        z_max = room.z[1]
        thickness = 0.05
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        lx = (x_max - x_min) / 2
        ly = (y_max - y_min) / 2
        lz = z_max / 2

        wall_specs = [
            # (half-extents, position)
            ([thickness, ly, lz], [x_min - thickness, cy, lz]),  # west wall
            ([thickness, ly, lz], [x_max + thickness, cy, lz]),  # east wall
            ([lx, thickness, lz], [cx, y_min - thickness, lz]),  # south wall
            ([lx, thickness, lz], [cx, y_max + thickness, lz]),  # north wall
        ]
        for half, pos in wall_specs:
            col_id = p.createCollisionShape(  # type: ignore[attr-defined]
                shapeType=p.GEOM_BOX,
                halfExtents=half,
                physicsClientId=client,  # type: ignore[attr-defined]
            )
            vis_id = p.createVisualShape(  # type: ignore[attr-defined]
                shapeType=p.GEOM_BOX,  # type: ignore[attr-defined]
                halfExtents=half,
                rgbaColor=_WALL_RGBA,
                physicsClientId=client,
            )
            p.createMultiBody(  # type: ignore[attr-defined]
                baseMass=0,
                baseCollisionShapeIndex=col_id,
                baseVisualShapeIndex=vis_id,
                basePosition=pos,
                physicsClientId=client,
            )
