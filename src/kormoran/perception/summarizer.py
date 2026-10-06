"""Deterministic text summariser — renders detections as a prompt-ready string.

The output format is stable (no randomness, no LLM calls) so it can be snapshot-tested.

Example output:
    "red cube: centre-left, small (3% of frame), confidence 0.80"
    "No red cube detected."
"""

from __future__ import annotations

from kormoran.types import Detection, TargetSpec

# Thresholds for positional and size descriptions
_H_THRESH = 0.25  # |cx_norm| > this → left/right
_V_THRESH = 0.25  # |cy_norm| > this → above/below
_SMALL_THRESH = 0.03  # area_frac < 3% → "small"
_LARGE_THRESH = 0.12  # area_frac > 12% → "large"


def _horizontal_pos(cx: float) -> str:
    if cx > _H_THRESH:
        return "right"
    if cx < -_H_THRESH:
        return "left"
    return "centre"


def _vertical_pos(cy: float) -> str:
    # cy_norm positive = lower in frame (OpenCV row order)
    if cy < -_V_THRESH:
        return "high"
    if cy > _V_THRESH:
        return "low"
    return ""


def _size_label(area_frac: float) -> str:
    if area_frac <= _SMALL_THRESH:
        return "small"
    if area_frac >= _LARGE_THRESH:
        return "large"
    return "medium"


def _describe_detection(det: Detection) -> str:
    cx, cy = det.center_norm
    h_pos = _horizontal_pos(cx)
    v_pos = _vertical_pos(cy)
    pos = f"{h_pos}-{v_pos}" if v_pos else h_pos
    size = _size_label(det.area_frac)
    area_pct = det.area_frac * 100.0
    return f"{det.label}: {pos}, {size} ({area_pct:.0f}% of frame), confidence {det.confidence:.2f}"


def summarize(detections: list[Detection], target: TargetSpec) -> str:
    """Render *detections* as a single-line string for the LLM prompt.

    Parameters
    ----------
    detections:
        Detections for the current target, as returned by ``Perceiver.detect``.
        If empty, a "not detected" line is returned.
    target:
        The target spec (used in the "not detected" message).

    Returns
    -------
    str
        A deterministic, human-readable description.  Multiple detections are
        joined with `` | ``.
    """
    if not detections:
        return f"No {target.label} detected."
    return " | ".join(_describe_detection(d) for d in detections)
