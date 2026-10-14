"""Emergency-stop sentinel file helpers (M7).

The UI writes a zero-byte sentinel file; the agent's ``check_node`` polls
for it each step and, when found, forces a ``land`` action.

File path: ``<data_dir>/stop/<mission_id>.stop``
"""

from __future__ import annotations

from pathlib import Path


def stop_flag_path(data_dir: str | Path, mission_id: str) -> Path:
    """Return the canonical path for a mission's stop flag."""
    return Path(data_dir) / "stop" / f"{mission_id}.stop"


def write_stop_flag(data_dir: str | Path, mission_id: str) -> Path:
    """Create the stop-flag file; return its path."""
    path = stop_flag_path(data_dir, mission_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def is_stop_requested(data_dir: str | Path, mission_id: str) -> bool:
    """Return True if the stop flag exists for this mission."""
    return stop_flag_path(data_dir, mission_id).exists()


def clear_stop_flag(data_dir: str | Path, mission_id: str) -> None:
    """Remove the stop flag (cleanup after mission ends)."""
    path = stop_flag_path(data_dir, mission_id)
    path.unlink(missing_ok=True)
