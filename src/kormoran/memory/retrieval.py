"""Memory retrieval and staleness helpers (M5).

``retrieve_for_context`` fetches relevant notes from the SemanticStore
and formats them as a list of plain dicts that can be stored in
MissionState.recalled (and serialised to JSON).

``mark_stale_if_contradicted`` checks the recalled notes against a current
observation — if the drone is near a location note's claimed position but
sees no detection, confidence is halved.
"""

from __future__ import annotations

import logging

from kormoran.memory.semantic import SemanticStore

logger = logging.getLogger(__name__)

_RETRIEVAL_QUERY_TEMPLATE = (
    "searching for {target_label} in scene {scene_id}; drone at ({x:.1f},{y:.1f}); step {step}"
)


def retrieve_for_context(
    store: SemanticStore,
    scene_id: str,
    target_label: str,
    drone_x: float,
    drone_y: float,
    step: int,
    n: int = 5,
) -> list[dict]:
    """Return up to *n* ranked notes as plain dicts for the prompt context."""
    query = _RETRIEVAL_QUERY_TEMPLATE.format(
        target_label=target_label,
        scene_id=scene_id,
        x=drone_x,
        y=drone_y,
        step=step,
    )
    try:
        return store.retrieve(query, scene_id=scene_id, target_label=target_label, n=n)
    except Exception as exc:  # noqa: BLE001
        logger.warning("retrieve_for_context(): store query failed: %s", exc)
        return []


def mark_stale_if_contradicted(
    store: SemanticStore,
    recalled: list[dict],
    drone_x: float,
    drone_y: float,
    has_detection: bool,
    radius_m: float = 1.5,
) -> None:
    """Halve confidence of location notes that are contradicted by current observation.

    A location note is contradicted when:
    - Its kind is "location" AND
    - The drone is within *radius_m* of the claimed location AND
    - No detection is present (target not visible here).

    This is a best-effort heuristic; parse failures are silently ignored.
    """
    if has_detection or not recalled:
        return

    for note in recalled:
        if note.get("kind") != "location":
            continue
        note_id = note.get("note_id", "")
        if not note_id:
            continue
        # Try to parse coordinates from note text (e.g. "target found at (1.2,-0.5)")
        text: str = note.get("text", "")
        coords = _extract_coords(text)
        if coords is None:
            continue
        nx, ny = coords
        dist = ((drone_x - nx) ** 2 + (drone_y - ny) ** 2) ** 0.5
        if dist <= radius_m:
            logger.debug(
                "Halving confidence of note %s (drone near claimed location, no detection)", note_id
            )
            store.halve_confidence(note_id)


def supersede_not_found_notes(
    store: SemanticStore,
    scene_id: str,
    target_label: str,
    found_mission_id: str,
) -> int:
    """Halve confidence of notes from prior missions that ended 'not_found'.

    Called after a successful find so that stale "not found" notes don't
    drown out the new, accurate location information.

    Returns the number of notes whose confidence was halved.
    """
    notes = store.list_notes(scene_id=scene_id, target_label=target_label)
    count = 0
    for note in notes:
        # Skip notes from the current (successful) mission.
        if note.get("mission_id") == found_mission_id:
            continue
        # Only demote notes whose text suggests a negative observation.
        text_lower = note.get("text", "").lower()
        if any(kw in text_lower for kw in ("not found", "did not", "was not", "no ", "failed")):
            note_id = note.get("note_id", "")
            if note_id:
                store.halve_confidence(note_id)
                count += 1
                logger.debug("supersede: halved confidence of note %s", note_id)
    return count


def _extract_coords(text: str) -> tuple[float, float] | None:
    """Best-effort extraction of (x, y) from a note like '... at (1.2,-0.5) ...'."""
    import re

    m = re.search(r"\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)", text)
    if m:
        try:
            return float(m.group(1)), float(m.group(2))
        except ValueError:
            pass
    return None
