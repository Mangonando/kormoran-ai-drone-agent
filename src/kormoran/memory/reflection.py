"""LLM-driven post-mission reflection (M5).

After a mission ends, call ``reflect(record, llm)`` to generate 1–3
typed MemoryNote objects from the mission summary.  These are stored in
the SemanticStore and recalled on subsequent missions in the same scene.

The LLM is given a compact summary of the mission and asked for structured
notes.  Uses ``with_structured_output`` to return a list of MemoryNote.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from pydantic import BaseModel

from kormoran.memory.schemas import EpisodicRecord, MemoryNote

logger = logging.getLogger(__name__)

_REFLECT_PROMPT = """\
You are a memory encoder for an autonomous quadcopter drone.
After each mission you synthesise 2–4 short memory notes to help future missions.

Mission summary:
  scene_id      : {scene_id}
  target_label  : {target_label}
  outcome       : {outcome}
  steps taken   : {steps}
  found_pose    : {found_pose}
  found_from    : {found_from}
  path summary  : {path_summary}
  walls hit     : {wall_summary}

Write 2–4 MemoryNote objects.  You MUST include:
- "location"  note: WHERE the target was / was not (use found_pose or last position)
- "wall"      note: WHERE walls/obstacles blocked the drone (use the walls hit data).
                    Include specific coordinates and which direction was blocked.
                    If no walls were hit, write "No walls encountered."

You MAY also include:
- "strategy"  note: WHAT navigation approach worked or failed
- "scene"     note (optional): any structural observation about the room layout

Rules:
- Each note.text must be a concise, scene-agnostic natural-language sentence (≤ 120 chars).
- confidence: 0.9 for direct observations, 0.6 for inferences, 0.3 for guesses.
- Wall notes should always be 0.9 confidence (they are direct observations).
- Use the exact scene_id and target_label provided.
- Do NOT include raw pixel data, images, or frame references.
"""


class _NoteList(BaseModel):
    notes: list[MemoryNote]


def _path_summary(path: list[dict], max_pts: int = 5) -> str:
    """Summarise a path list to a short string."""
    if not path:
        return "no path recorded"
    step = max(1, len(path) // max_pts)
    pts = path[::step][:max_pts]
    return " → ".join(f"({p.get('x', 0):.1f},{p.get('y', 0):.1f})" for p in pts)


def _wall_summary(wall_hits: list[dict]) -> str:
    """Summarise wall/obstacle positions from clamped advances."""
    if not wall_hits:
        return "none"
    # Deduplicate to nearest 0.5m grid
    seen: set[tuple[float, float]] = set()
    unique: list[str] = []
    for hit in wall_hits:
        rx = round(hit.get("x", 0) * 2) / 2
        ry = round(hit.get("y", 0) * 2) / 2
        if (rx, ry) not in seen:
            seen.add((rx, ry))
            unique.append(f"({rx:.1f},{ry:.1f})")
    return "; ".join(unique[:8])


def reflect(
    record: EpisodicRecord, llm: object, wall_hits: list[dict] | None = None
) -> list[MemoryNote]:
    """Generate MemoryNote objects from a completed mission record.

    Parameters
    ----------
    record:
        The EpisodicRecord for the just-completed mission.
    llm:
        A LangChain ChatOpenAI instance (must support with_structured_output).

    Returns
    -------
    list[MemoryNote]
        Typically 1–3 notes; empty list on any LLM failure.
    """
    prompt = _REFLECT_PROMPT.format(
        scene_id=record.scene_id,
        target_label=record.target_label,
        outcome=record.outcome,
        steps=record.steps,
        found_pose=record.found_pose or "N/A",
        found_from=record.found_from_pose or "N/A",
        path_summary=_path_summary(record.path),
        wall_summary=_wall_summary(wall_hits or []),
    )

    now_iso = datetime.now(UTC).isoformat()
    chain = llm.with_structured_output(_NoteList)  # type: ignore[union-attr]
    try:
        result: _NoteList = chain.invoke([("human", prompt)])
        notes = result.notes
    except Exception as exc:  # noqa: BLE001
        logger.warning("reflect(): LLM call failed: %s", exc)
        return []

    # Stamp ids / timestamps that the LLM may not fill correctly.
    # Cap confidence for failed missions: "not_found" is not a direct
    # observation — notes from failures should carry less weight so they
    # don't drown out successful finds.
    # Exception: "wall" notes are always direct observations (the drone
    # physically hit the wall) regardless of mission outcome.
    max_conf = 0.4 if record.outcome != "found" else 1.0

    stamped: list[MemoryNote] = []
    for note in notes:
        cap = 1.0 if note.kind == "wall" else max_conf
        stamped.append(
            MemoryNote(
                note_id=str(uuid.uuid4()),
                scene_id=record.scene_id,
                target_label=record.target_label,
                kind=note.kind,
                text=note.text[:200],  # hard cap
                confidence=max(0.05, min(cap, note.confidence)),
                created_at=now_iso,
                mission_id=record.mission_id,
            )
        )
    return stamped
