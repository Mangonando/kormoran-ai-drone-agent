"""Mission recording utilities.

v0 (M1): save a list of ``Frame`` objects to an animated GIF.
v1 (M4): JSONL mission log added when the full agent loop is implemented.
"""

from __future__ import annotations

import logging
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
