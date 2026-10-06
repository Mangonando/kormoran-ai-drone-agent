"""Snapshot tests for the deterministic summarizer.

The output format must never change silently — any change to summarizer.py that
affects these assertions requires updating them deliberately.
"""

from __future__ import annotations

import pytest

from kormoran.perception.summarizer import summarize
from kormoran.types import Detection, TargetSpec


def _det(
    label: str = "red cube",
    cx: float = 0.0,
    cy: float = 0.0,
    area_frac: float = 0.03,
    confidence: float = 0.85,
) -> Detection:
    """Build a Detection from just the fields the summarizer cares about."""
    return Detection(
        label=label,
        bbox_px=(0, 0, 10, 10),
        center_norm=(cx, cy),
        area_frac=area_frac,
        confidence=confidence,
    )


def _target(label: str = "red cube", color: str = "red") -> TargetSpec:
    return TargetSpec(label=label, color=color)


@pytest.mark.unit
def test_no_detection_message() -> None:
    """Empty detection list produces 'not detected' message."""
    result = summarize([], _target())
    assert result == "No red cube detected."


@pytest.mark.unit
def test_centre_small_snapshot() -> None:
    """Centre, small (3% of frame) → canonical snapshot."""
    det = _det(cx=0.0, cy=0.0, area_frac=0.03, confidence=0.85)
    result = summarize([det], _target())
    assert result == "red cube: centre, small (3% of frame), confidence 0.85"


@pytest.mark.unit
def test_centre_left_small_snapshot() -> None:
    """Left of centre → 'left'."""
    det = _det(cx=-0.5, cy=0.0, area_frac=0.02, confidence=0.70)
    result = summarize([det], _target())
    assert result == "red cube: left, small (2% of frame), confidence 0.70"


@pytest.mark.unit
def test_right_large_snapshot() -> None:
    """Right, large blob."""
    det = _det(cx=0.6, cy=0.0, area_frac=0.15, confidence=0.95)
    result = summarize([det], _target())
    assert result == "red cube: right, large (15% of frame), confidence 0.95"


@pytest.mark.unit
def test_centre_high_medium_snapshot() -> None:
    """Top of frame (negative cy_norm = high)."""
    det = _det(cx=0.0, cy=-0.4, area_frac=0.06, confidence=0.80)
    result = summarize([det], _target())
    assert result == "red cube: centre-high, medium (6% of frame), confidence 0.80"


@pytest.mark.unit
def test_multiple_detections_joined() -> None:
    """Two detections → joined with ' | '."""
    d1 = _det(cx=-0.5, area_frac=0.10, confidence=0.90)
    d2 = _det(cx=0.5, area_frac=0.04, confidence=0.60)
    result = summarize([d1, d2], _target())
    assert " | " in result
    assert result.count("red cube") == 2


@pytest.mark.unit
def test_area_rounding() -> None:
    """area_frac is displayed as integer % (no decimals)."""
    det = _det(area_frac=0.037)
    result = summarize([det], _target())
    assert "4%" in result  # 0.037 * 100 = 3.7 → rounds to 4


@pytest.mark.unit
def test_absent_target_label_used() -> None:
    """Label from target spec is used in 'not detected' message."""
    t = TargetSpec(label="blue sphere", color="blue")
    result = summarize([], t)
    assert "blue sphere" in result
