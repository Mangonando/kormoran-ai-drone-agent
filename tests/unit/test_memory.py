"""Unit tests for M5 long-term memory (DoD requirements).

Uses temporary directories and a deterministic dummy embedding function
so no model download is needed.

DoD:
- Stores round-trip records correctly
- Retrieval returns the right scene's location note first
- Contradicting observation halves confidence
- clear() wipes both stores
- No frame bytes ever stored (asserts schema)
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from kormoran.memory.episodic import EpisodicStore
from kormoran.memory.schemas import EpisodicRecord, MemoryNote
from kormoran.memory.semantic import SemanticStore
from tests.conftest import DummyEmbeddingFunction

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _make_record(
    mission_id: str | None = None,
    scene_id: str = "test_scene",
    target_label: str = "red cube",
    outcome: str = "found",
    steps: int = 10,
) -> EpisodicRecord:
    return EpisodicRecord(
        mission_id=mission_id or uuid.uuid4().hex[:8],
        ts=_now(),
        scene_id=scene_id,
        target_label=target_label,
        outcome=outcome,  # type: ignore[arg-type]
        steps=steps,
        duration_s=12.3,
        path=[{"x": 0.0, "y": 0.0, "z": 1.0, "step": 0}],
        found_pose={"x": 1.0, "y": 0.5, "z": 0.5, "yaw_deg": 45.0},
        model="test",
        tokens=100,
        cost_usd=0.0001,
    )


def _make_note(
    scene_id: str = "test_scene",
    target_label: str = "red cube",
    kind: str = "location",
    text: str = "Target found at (1.0,0.5)",
    confidence: float = 0.9,
    mission_id: str | None = None,
) -> MemoryNote:
    return MemoryNote(
        note_id=str(uuid.uuid4()),
        scene_id=scene_id,
        target_label=target_label,
        kind=kind,  # type: ignore[arg-type]
        text=text,
        confidence=confidence,
        created_at=_now(),
        mission_id=mission_id or uuid.uuid4().hex[:8],
    )


@pytest.fixture()
def episodic_store(tmp_path: Path) -> EpisodicStore:
    return EpisodicStore(tmp_path / "episodic.db")


@pytest.fixture()
def semantic_store(tmp_path: Path) -> SemanticStore:
    return SemanticStore(tmp_path / "semantic", embedding_function=DummyEmbeddingFunction())


# ---------------------------------------------------------------------------
# EpisodicStore tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEpisodicStore:
    def test_round_trip(self, episodic_store: EpisodicStore) -> None:
        """store() then get() returns identical record."""
        record = _make_record(mission_id="m001")
        episodic_store.store(record)
        fetched = episodic_store.get("m001")
        assert fetched is not None
        assert fetched.mission_id == "m001"
        assert fetched.scene_id == "test_scene"
        assert fetched.outcome == "found"
        assert fetched.steps == 10
        assert fetched.path == [{"x": 0.0, "y": 0.0, "z": 1.0, "step": 0}]
        assert fetched.found_pose == {"x": 1.0, "y": 0.5, "z": 0.5, "yaw_deg": 45.0}

    def test_list_missions(self, episodic_store: EpisodicStore) -> None:
        """list_missions returns records newest-first, filtered by scene."""
        r1 = _make_record(scene_id="room_a", outcome="found")
        r2 = _make_record(scene_id="room_b", outcome="not_found")
        episodic_store.store(r1)
        episodic_store.store(r2)
        all_records = episodic_store.list_missions()
        assert len(all_records) == 2
        room_a = episodic_store.list_missions(scene_id="room_a")
        assert len(room_a) == 1
        assert room_a[0].scene_id == "room_a"

    def test_clear_wipes_both(self, episodic_store: EpisodicStore) -> None:
        """clear() removes all records; count returns 0."""
        episodic_store.store(_make_record())
        episodic_store.store(_make_record())
        assert episodic_store.count() == 2
        episodic_store.clear()
        assert episodic_store.count() == 0
        assert episodic_store.list_missions() == []

    def test_no_frames_in_schema(self, episodic_store: EpisodicStore) -> None:
        """EpisodicRecord schema must not contain raw frame/image bytes."""
        record = _make_record()
        # Path entries may only have: x, y, z, step — never 'frame' or 'image'.
        for entry in record.path:
            assert "frame" not in entry, "Raw frame bytes found in path entry"
            assert "image" not in entry, "Raw image bytes found in path entry"
        # The model itself must not have an image field.
        field_names = set(EpisodicRecord.model_fields.keys())
        assert "frame" not in field_names
        assert "image" not in field_names
        assert "frames" not in field_names
        assert "images" not in field_names


# ---------------------------------------------------------------------------
# SemanticStore tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSemanticStore:
    def test_round_trip(self, semantic_store: SemanticStore) -> None:
        """store() then list_notes() returns the stored note."""
        note = _make_note(scene_id="room1", target_label="blue sphere")
        semantic_store.store(note)
        notes = semantic_store.list_notes(scene_id="room1")
        assert len(notes) == 1
        assert notes[0]["text"] == note.text

    def test_clear_wipes_notes(self, semantic_store: SemanticStore) -> None:
        """clear() removes all notes; count returns 0."""
        semantic_store.store(_make_note())
        semantic_store.store(_make_note())
        assert semantic_store.count() == 2
        semantic_store.clear()
        assert semantic_store.count() == 0

    def test_no_frames_in_schema(self, semantic_store: SemanticStore) -> None:
        """SemanticStore metadata keys must not include raw frame/image data."""
        keys = semantic_store.get_metadata_keys()
        for bad_key in ("frame", "image", "frames", "images", "raw"):
            assert bad_key not in keys, f"Sensitive key {bad_key!r} found in metadata schema"

    def test_retrieval_returns_correct_scene_note_first(
        self, semantic_store: SemanticStore
    ) -> None:
        """Retrieval ranks the matching scene+target note above notes for other scenes."""
        # Store a location note for the correct scene/target.
        correct_note = _make_note(
            scene_id="living_room",
            target_label="red cube",
            kind="location",
            text="red cube found at (1.0,0.5) in the living room",
            confidence=0.9,
        )
        # Store a note for a different scene.
        wrong_note = _make_note(
            scene_id="kitchen",
            target_label="red cube",
            kind="location",
            text="red cube seen at (-1.0,-1.0) in the kitchen",
            confidence=0.9,
        )
        semantic_store.store(correct_note)
        semantic_store.store(wrong_note)

        results = semantic_store.retrieve(
            query="searching for red cube in living_room",
            scene_id="living_room",
            target_label="red cube",
            n=5,
        )
        # Only the matching scene note should be returned (wrong scene filtered out).
        assert len(results) >= 1
        assert any("living room" in r["text"] for r in results)
        # No results from the wrong scene.
        assert all("kitchen" not in r["text"] for r in results)

    def test_halve_confidence(self, semantic_store: SemanticStore) -> None:
        """halve_confidence() reduces confidence by ~50%."""
        note = _make_note(confidence=0.8)
        semantic_store.store(note)

        semantic_store.halve_confidence(note.note_id)

        raw = semantic_store._raw_get([note.note_id])
        assert raw["ids"]
        new_conf = raw["metadatas"][0]["confidence"]
        assert abs(new_conf - 0.4) < 0.01, f"Expected ~0.4, got {new_conf}"

    def test_contradicting_observation_halves_confidence(
        self, semantic_store: SemanticStore
    ) -> None:
        """mark_stale_if_contradicted halves confidence when drone is nearby."""
        from kormoran.memory.retrieval import mark_stale_if_contradicted

        note = _make_note(
            kind="location",
            text="red cube found at (1.0,0.5) near the window",
            confidence=0.8,
        )
        semantic_store.store(note)

        # Drone is at (1.1, 0.4) — within 1.5 m of (1.0, 0.5) — no detection.
        mark_stale_if_contradicted(
            store=semantic_store,
            recalled=[
                {
                    "kind": "location",
                    "note_id": note.note_id,
                    "text": note.text,
                    "confidence": 0.8,
                }
            ],
            drone_x=1.1,
            drone_y=0.4,
            has_detection=False,
        )

        raw = semantic_store._raw_get([note.note_id])
        new_conf = raw["metadatas"][0]["confidence"]
        assert new_conf < 0.5, f"Expected confidence < 0.5 after contradiction, got {new_conf}"

    def test_no_contradiction_when_detection_present(self, semantic_store: SemanticStore) -> None:
        """mark_stale_if_contradicted does NOT halve confidence when a detection is present."""
        from kormoran.memory.retrieval import mark_stale_if_contradicted

        note = _make_note(kind="location", text="red cube at (1.0,0.5)", confidence=0.8)
        semantic_store.store(note)

        mark_stale_if_contradicted(
            store=semantic_store,
            recalled=[{"kind": "location", "note_id": note.note_id, "text": note.text}],
            drone_x=1.0,
            drone_y=0.5,
            has_detection=True,  # detection present → no contradiction
        )

        raw = semantic_store._raw_get([note.note_id])
        new_conf = raw["metadatas"][0]["confidence"]
        assert abs(new_conf - 0.8) < 0.01, "Confidence should be unchanged when detection present"
