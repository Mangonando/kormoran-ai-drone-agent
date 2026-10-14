"""JSONL mission log reader for the Streamlit dashboard (M7).

Reads a ``data/missions/<mission_id>.jsonl`` file produced by
``MissionLogger.log_step()`` and returns structured Python objects.

Each log line has the shape::

    {
        "step": int,
        "ts":   float,   # Unix timestamp
        "action": {"kind": str, "distance_m": float|null, "angle_deg": float|null,
                   "reasoning": str, "confidence": float},
        "result": {"status": str, "pose": {...}, "elapsed_s": float},
        "detections": [...],
        "pose": {"x": float, "y": float, "z": float, "yaw_deg": float},
        "usage": {"prompt_tokens": int, "completion_tokens": int,
                  "llm_calls": int, "est_cost_usd": float},
        "safety_events": [...],
    }
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def read_log(log_path: str | Path) -> list[dict]:
    """Parse a JSONL mission log and return a list of step records.

    Silently skips malformed lines so a partial write at mission end does
    not break the UI.  Returns an empty list if the file does not exist yet.
    """
    from kormoran.jsonl import read_jsonl

    return read_jsonl(log_path)


def latest_status_from_log(steps: list[dict]) -> str:
    """Infer the mission status from the last log entry.

    Returns one of: "found", "not_found", "aborted", "error", "running".
    """
    if not steps:
        return "running"
    last = steps[-1]
    action_kind = (last.get("action") or {}).get("kind", "")
    if action_kind == "land":
        return "found"
    if action_kind == "report_not_found":
        return "not_found"
    result_status = (last.get("result") or {}).get("status", "")
    if result_status == "aborted":
        return "aborted"
    # No terminal action recorded — either still running or timed out.
    # Use a 2-minute threshold: if the log hasn't been written to in
    # over 2 minutes, the subprocess is dead.
    import time

    last_ts = last.get("ts", 0)
    if last_ts and (time.time() - last_ts) > 120:
        return "error"
    return "running"


def cumulative_usage(steps: list[dict]) -> dict:
    """Return the final cumulative token/cost usage from the last step."""
    if not steps:
        return {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "est_cost_usd": 0.0}
    return steps[-1].get("usage") or {}


def trajectory(steps: list[dict]) -> list[tuple[float, float]]:
    """Extract (x, y) positions from each step's pose."""
    return [(s["pose"]["x"], s["pose"]["y"]) for s in steps if "pose" in s and "x" in s["pose"]]
