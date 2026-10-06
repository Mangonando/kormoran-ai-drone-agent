"""Ground-truth detector using PyBullet segmentation masks.

This module is **eval-only** — it produces reference detections from PyBullet's
segmentation output so that the colour detector's precision/recall can be measured.
It has no dependency on any drone backend; callers pass in the raw seg mask and
the body-ID mapping that SimDrone exposes.

PyBullet segmentation encoding:
    Each pixel stores ``objectUniqueId + (linkIndex + 1) * 65536``.
    For static single-body objects (linkIndex = -1 → base):
        pixel == objectUniqueId
    For articulated links: high bits encode the link.
    We extract the body ID with ``pixel & 0xFFFF`` to be safe.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from kormoran.types import Detection, TargetSpec


def detections_from_seg_mask(
    seg_mask: NDArray[np.int32],
    label_to_body_id: dict[str, int],
    target: TargetSpec,
) -> list[Detection]:
    """Extract ground-truth detections for *target* from a PyBullet seg mask.

    Parameters
    ----------
    seg_mask:
        H×W integer array returned by ``getCameraImage(segmentation=True)``.
    label_to_body_id:
        Mapping of target label → PyBullet body unique ID (from SceneBuilder).
    target:
        The target spec whose label is looked up in *label_to_body_id*.

    Returns
    -------
    list[Detection]
        Zero or one ``Detection`` with ``confidence=1.0`` (ground truth is exact).
    """
    body_id = label_to_body_id.get(target.label)
    if body_id is None:
        return []

    h_px, w_px = seg_mask.shape[:2]

    # Extract body ID from pixel value (mask off link-index bits).
    # Also accept an exact match for environments that return plain body IDs.
    body_mask = (seg_mask == body_id) | ((seg_mask & 0xFFFF) == (body_id & 0xFFFF))
    body_mask &= seg_mask > 0  # exclude background (0 or negative)

    if not body_mask.any():
        return []

    # Bounding box from pixel extents
    rows_with_object = np.any(body_mask, axis=1)
    cols_with_object = np.any(body_mask, axis=0)
    r_min = int(np.argmax(rows_with_object))
    r_max = int(len(rows_with_object) - 1 - np.argmax(rows_with_object[::-1]))
    c_min = int(np.argmax(cols_with_object))
    c_max = int(len(cols_with_object) - 1 - np.argmax(cols_with_object[::-1]))

    bw = c_max - c_min + 1
    bh = r_max - r_min + 1

    # Normalised centre
    cx_norm = ((c_min + c_max) / 2.0 - w_px / 2.0) / (w_px / 2.0)
    cy_norm = ((r_min + r_max) / 2.0 - h_px / 2.0) / (h_px / 2.0)

    area_frac = float(body_mask.sum()) / float(h_px * w_px)

    return [
        Detection(
            label=target.label,
            bbox_px=(c_min, r_min, bw, bh),
            center_norm=(float(cx_norm), float(cy_norm)),
            area_frac=float(min(1.0, area_frac)),
            confidence=1.0,
        )
    ]


def is_target_visible(
    seg_mask: NDArray[np.int32],
    label_to_body_id: dict[str, int],
    label: str,
) -> bool:
    """Return ``True`` when any pixel in *seg_mask* belongs to *label*'s body."""
    body_id = label_to_body_id.get(label)
    if body_id is None:
        return False
    return bool(np.any(seg_mask == body_id) or np.any((seg_mask & 0xFFFF) == (body_id & 0xFFFF)))
