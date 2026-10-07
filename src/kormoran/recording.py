"""Mission recording utilities.

v0 (M1): save a list of ``Frame`` objects to an animated GIF.
v1 (M4): JSONL mission log — one JSON line per step.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from PIL import Image

from kormoran.types import Frame

logger = logging.getLogger(__name__)


def save_gif(
    frames: list[Frame],
    path: str | Path,
    fps: int = 12,
) -> Path:
    """Save *frames* to an animated GIF at *path*.

    Parameters
    ----------
    frames:
        Ordered list of ``Frame`` objects (RGB uint8 HxWx3).
    path:
        Output file path.  Parent directory is created if it does not exist.
    fps:
        Frames per second in the output GIF.

    Returns
    -------
    Path
        Resolved path of the written file.
    """
    if not frames:
        msg = "save_gif: frames list is empty"
        raise ValueError(msg)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    pil_frames = [Image.fromarray(f.image, mode="RGB") for f in frames]
    duration_ms = int(1000 / fps)

    pil_frames[0].save(
        out,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    logger.info("Saved %d frames → %s", len(frames), out)
    return out


class MissionLogger:
    """Append-only JSONL log for a single mission (one record per step).

    Parameters
    ----------
    log_path:
        Path to the ``.jsonl`` file.  Parent directory is created on first write.
    """

    def __init__(self, log_path: Path) -> None:
        self._path = log_path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log_step(
        self,
        step: int,
        action: dict,
        result: dict,
        detections: list[dict],
        pose: dict,
        usage: dict,
        events: list[dict],
    ) -> None:
        """Append one step record to the JSONL file."""
        record = {
            "step": step,
            "ts": time.time(),
            "action": action,
            "result": result,
            "detections": detections,
            "pose": pose,
            "usage": usage,
            "safety_events": events,
        }
        with self._path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
