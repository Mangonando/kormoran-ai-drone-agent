"""Unit tests for ColorDetector using synthetic fixture frames.

No simulator, no network.  All frames are generated in-process with numpy so
the tests run in milliseconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from kormoran.perception.color_detector import ColorDetector
from kormoran.types import Frame, TargetSpec

# ---------------------------------------------------------------------------
# Fixture-frame helpers
# ---------------------------------------------------------------------------


def _make_frame(img: np.ndarray) -> Frame:
    """Wrap a uint8 HxWx3 array in a Frame."""
    return Frame(image=img.astype(np.uint8), t=0.0, fov_deg=60.0)


def _blank(h: int = 96, w: int = 128) -> np.ndarray:
    """Return a completely black frame (no colour blobs)."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _with_blob(
    color_rgb: tuple[int, int, int],
    cx: int = 64,
    cy: int = 48,
    radius: int = 18,
    h: int = 96,
    w: int = 128,
) -> np.ndarray:
    """Return a frame containing one circular blob of *color_rgb* at (cx, cy)."""
    img = _blank(h, w)
    # Vectorised circle fill
    ys, xs = np.mgrid[0:h, 0:w]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius**2
    img[mask] = color_rgb
    return img


def _target(label: str = "test", color: str = "red") -> TargetSpec:
    return TargetSpec(label=label, color=color)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_true_positive_red_blob() -> None:
    """A red blob in the centre is detected with colour='red'."""
    img = _with_blob((255, 0, 0))
    frame = _make_frame(img)
    target = _target(color="red")
    detector = ColorDetector()

    dets = detector.detect(frame, target)

    assert len(dets) >= 1, "should detect the red blob"
    assert dets[0].area_frac > 0.005
    # Centre should be near (0, 0) since blob is at frame centre
    assert abs(dets[0].center_norm[0]) < 0.3
    assert abs(dets[0].center_norm[1]) < 0.3


@pytest.mark.unit
def test_true_positive_blue_blob() -> None:
    """A blue blob is detected with colour='blue'."""
    img = _with_blob((0, 0, 255))
    frame = _make_frame(img)
    target = _target(color="blue")
    detector = ColorDetector()

    dets = detector.detect(frame, target)

    assert len(dets) >= 1, "should detect the blue blob"


@pytest.mark.unit
def test_true_positive_green_blob() -> None:
    """A green blob is detected with colour='green'."""
    img = _with_blob((0, 200, 0))
    frame = _make_frame(img)
    detector = ColorDetector()

    dets = detector.detect(frame, _target(color="green"))

    assert len(dets) >= 1, "should detect the green blob"


@pytest.mark.unit
def test_true_positive_yellow_blob() -> None:
    """A yellow blob is detected with colour='yellow'."""
    img = _with_blob((255, 255, 0))
    frame = _make_frame(img)
    detector = ColorDetector()

    dets = detector.detect(frame, _target(color="yellow"))

    assert len(dets) >= 1, "should detect the yellow blob"


@pytest.mark.unit
def test_empty_room_no_detection() -> None:
    """A blank (all-black) frame produces no detections for any colour."""
    frame = _make_frame(_blank())
    detector = ColorDetector()
    for colour in ("red", "blue", "green", "yellow"):
        dets = detector.detect(frame, _target(color=colour))
        assert dets == [], f"expected no detections for {colour!r} in empty frame"


@pytest.mark.unit
def test_two_colours_correct_target_only() -> None:
    """Frame with red blob and blue blob: detect only the requested colour."""
    img = _blank()
    # Red blob on the left
    ys, xs = np.mgrid[0:96, 0:128]
    red_mask = (xs - 30) ** 2 + (ys - 48) ** 2 <= 15**2
    img[red_mask] = (255, 0, 0)
    # Blue blob on the right
    blue_mask = (xs - 98) ** 2 + (ys - 48) ** 2 <= 15**2
    img[blue_mask] = (0, 0, 255)

    frame = _make_frame(img)
    detector = ColorDetector()

    red_dets = detector.detect(frame, _target(color="red"))
    blue_dets = detector.detect(frame, _target(color="blue"))

    assert len(red_dets) >= 1, "should detect red"
    assert len(blue_dets) >= 1, "should detect blue"

    # Red detection centre should be left of frame centre (cx_norm < 0)
    assert red_dets[0].center_norm[0] < 0.0
    # Blue detection centre should be right of frame centre (cx_norm > 0)
    assert blue_dets[0].center_norm[0] > 0.0


@pytest.mark.unit
def test_partial_occlusion_at_edge() -> None:
    """A blob half cut off at the right edge is still detected."""
    img = _blank()
    # Place blob centre at x=118 (only ~10 px visible on a 128-wide frame)
    ys, xs = np.mgrid[0:96, 0:128]
    mask = (xs - 118) ** 2 + (ys - 48) ** 2 <= 20**2
    img[mask] = (255, 0, 0)

    frame = _make_frame(img)
    detector = ColorDetector()
    dets = detector.detect(frame, _target(color="red"))

    assert len(dets) >= 1, "should detect partially-occluded blob"
    # Centre should be on the right side of the frame
    assert dets[0].center_norm[0] > 0.0


@pytest.mark.unit
def test_unknown_colour_returns_empty() -> None:
    """Requesting an unknown colour returns an empty list gracefully."""
    img = _with_blob((128, 0, 128))  # purple — not in the colour map
    frame = _make_frame(img)
    detector = ColorDetector()

    dets = detector.detect(frame, _target(color="purple"))

    assert dets == []


@pytest.mark.unit
def test_no_colour_in_target_returns_empty() -> None:
    """A TargetSpec without a colour field returns an empty list."""
    frame = _make_frame(_with_blob((255, 0, 0)))
    detector = ColorDetector()

    dets = detector.detect(frame, TargetSpec(label="mystery", color=None))

    assert dets == []


@pytest.mark.unit
def test_detections_sorted_by_area_desc() -> None:
    """When multiple blobs are detected, they are sorted largest first."""
    img = _blank()
    ys, xs = np.mgrid[0:96, 0:128]
    # Large red blob left
    img[(xs - 30) ** 2 + (ys - 48) ** 2 <= 20**2] = (255, 0, 0)
    # Small red blob right
    img[(xs - 100) ** 2 + (ys - 48) ** 2 <= 8**2] = (255, 0, 0)

    frame = _make_frame(img)
    detector = ColorDetector()
    dets = detector.detect(frame, _target(color="red"))

    if len(dets) >= 2:
        assert dets[0].area_frac >= dets[1].area_frac
