"""ColorDetector — HSV colour-segmentation detector implementing the Perceiver protocol.

Supported target vocabulary (matches the builder's colour map):
    red · blue · green · yellow · orange · white

HSV notes (OpenCV convention: H 0–180, S 0–255, V 0–255):
    Red wraps around hue=0 so two ranges are needed.
    Saturation/Value floors are kept low enough to handle PyBullet's Phong shading,
    which can darken or partially desaturate faces not facing the light source.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from kormoran.types import Detection, Frame, TargetSpec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HSV range table  (lower_hsv, upper_hsv)
# ---------------------------------------------------------------------------

_HSV_RANGES: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "red": [
        ((0, 90, 40), (10, 255, 255)),
        ((168, 90, 40), (180, 255, 255)),
    ],
    "blue": [((100, 90, 40), (135, 255, 255))],
    "green": [((38, 70, 40), (88, 255, 255))],
    "yellow": [((18, 90, 40), (38, 255, 255))],
    "orange": [((10, 100, 40), (20, 255, 255))],
    "white": [((0, 0, 200), (180, 25, 255))],
}

# Contour area thresholds
_MIN_CONTOUR_AREA_PX: int = 30
_MIN_AREA_FRAC: float = 0.0005

# Morphological structuring element size
_MORPH_KERNEL_SIZE: int = 5


class ColorDetector:
    """Segment coloured targets in an RGB frame via HSV thresholding.

    Implements the ``Perceiver`` protocol (``detect(frame, target) -> list[Detection]``).

    Confidence is derived from blob *solidity* (contour_area / convex_hull_area) —
    a perfect convex blob returns 1.0, partially occluded or noisy blobs return lower.
    Very small blobs are penalised by an area scale factor so the agent treats
    distant/marginal detections conservatively.
    """

    def detect(self, frame: Frame, target: TargetSpec) -> list[Detection]:
        """Return all detections of *target* found in *frame*.

        Returns an empty list when the target colour is unknown or absent.
        Detections are sorted by ``area_frac`` descending.
        """
        if target.color is None:
            logger.debug("ColorDetector: no colour on target %r", target.label)
            return []

        colour = target.color.lower()
        ranges = _HSV_RANGES.get(colour)
        if ranges is None:
            logger.warning("ColorDetector: unknown colour %r", colour)
            return []

        img = frame.image  # H×W×3 uint8, RGB
        h_px, w_px = img.shape[:2]

        # RGB → BGR (OpenCV native) → HSV
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Union of all HSV ranges for this colour
        mask = np.zeros((h_px, w_px), dtype=np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))

        # Morphology: close small gaps, then remove isolated noise
        ks = _MORPH_KERNEL_SIZE
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_area = float(h_px * w_px)
        detections: list[Detection] = []

        for cnt in contours:
            area_px = float(cv2.contourArea(cnt))
            if area_px < _MIN_CONTOUR_AREA_PX:
                continue
            area_frac = area_px / frame_area
            if area_frac < _MIN_AREA_FRAC:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)

            # Normalised centre: (0, 0) = frame centre, ±1 = frame edge
            cx_norm = ((bx + bw / 2.0) - w_px / 2.0) / (w_px / 2.0)
            cy_norm = ((by + bh / 2.0) - h_px / 2.0) / (h_px / 2.0)

            # Confidence from solidity, scaled by apparent size
            hull = cv2.convexHull(cnt)
            hull_area = float(cv2.contourArea(hull))
            solidity = area_px / hull_area if hull_area > 1.0 else 0.5
            size_factor = min(1.0, area_frac / 0.002)
            confidence = float(min(1.0, solidity * max(0.3, size_factor)))

            detections.append(
                Detection(
                    label=target.label,
                    bbox_px=(int(bx), int(by), int(bw), int(bh)),
                    center_norm=(float(cx_norm), float(cy_norm)),
                    area_frac=float(min(1.0, area_frac)),
                    confidence=confidence,
                )
            )

        detections.sort(key=lambda d: d.area_frac, reverse=True)
        logger.debug(
            "ColorDetector: %d detection(s) for colour=%r label=%r",
            len(detections),
            colour,
            target.label,
        )
        return detections
