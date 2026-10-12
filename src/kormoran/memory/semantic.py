"""Semantic ChromaDB store for Kormoran (M5).

Stores MemoryNote objects and retrieves them by similarity.

Collection: "notes"
Document:   MemoryNote.text  (natural-language note, embedded by ChromaDB)
Metadata:   all scalar fields of MemoryNote except text
id:         MemoryNote.note_id

Retrieval ranking = similarity × recency_factor × confidence
  recency_factor = exp(-age_days / 30)  — decays to ~0.05 after 3 months
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

from kormoran.memory.schemas import MemoryNote

# chromadb import deferred to avoid slow startup when store is not used


class SemanticStore:
    """ChromaDB-backed semantic memory."""

    def __init__(
        self,
        persist_dir: str | Path,
        *,
        embedding_function: object | None = None,
    ) -> None:
        """
        Parameters
        ----------
        persist_dir:
            Directory where ChromaDB will persist data.
        embedding_function:
            Pass a custom ChromaDB EmbeddingFunction for tests (avoids model
            download).  None → ChromaDB default (sentence-transformers MiniLM).
        """
        import chromadb  # noqa: PLC0415
        from chromadb.config import Settings as ChromaSettings  # noqa: PLC0415

        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        kwargs: dict = {}
        if embedding_function is not None:
            kwargs["embedding_function"] = embedding_function
        self._col = self._client.get_or_create_collection("notes", **kwargs)

    # ------------------------------------------------------------------
    # Public API

    def store(self, note: MemoryNote) -> None:
        """Upsert a MemoryNote into the collection."""
        metadata = {
            "scene_id": note.scene_id,
            "target_label": note.target_label,
            "kind": note.kind,
            "confidence": note.confidence,
            "created_at": note.created_at,
            "mission_id": note.mission_id,
        }
        self._col.upsert(
            ids=[note.note_id],
            documents=[note.text],
            metadatas=[metadata],
        )

    def retrieve(
        self,
        query: str,
        scene_id: str,
        target_label: str,
        n: int = 5,
    ) -> list[dict]:
        """Return up to *n* notes ranked by similarity × recency × confidence.

        Returns list of dicts with keys: text, kind, confidence, score, note_id,
        mission_id, created_at.
        """
        total = self._col.count()
        if total == 0:
            return []

        # Query more candidates, re-rank, then truncate.
        k = min(total, max(n * 3, 20))
        where: dict = {"scene_id": scene_id}

        results = self._col.query(
            query_texts=[query],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        now = datetime.now(UTC)
        ranked: list[tuple[float, dict]] = []
        for doc, meta, dist in zip(docs, metas, distances, strict=True):
            # ChromaDB l2 distance → cosine-like similarity in [0,1]
            similarity = 1.0 / (1.0 + dist)
            confidence = float(meta.get("confidence", 1.0))
            created_at_str = meta.get("created_at", now.isoformat())
            try:
                created_at = datetime.fromisoformat(created_at_str)
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=UTC)
            except ValueError:
                created_at = now
            age_days = max(0.0, (now - created_at).total_seconds() / 86400)
            recency = math.exp(-age_days / 30.0)
            score = similarity * recency * confidence

            # Filter by target_label post-query (ChromaDB where only supports one key)
            if meta.get("target_label") != target_label:
                continue

            ranked.append(
                (
                    score,
                    {
                        "text": doc,
                        "kind": meta.get("kind"),
                        "confidence": confidence,
                        "score": round(score, 4),
                        "note_id": meta.get("note_id", ""),
                        "mission_id": meta.get("mission_id", ""),
                        "created_at": created_at_str,
                    },
                )
            )

        ranked.sort(key=lambda t: t[0], reverse=True)
        return [item for _, item in ranked[:n]]

    def halve_confidence(self, note_id: str) -> None:
        """Halve the confidence of a note (staleness signal)."""
        result = self._col.get(ids=[note_id], include=["documents", "metadatas"])
        if not result["ids"]:
            return
        meta = result["metadatas"][0]
        doc = result["documents"][0]
        new_conf = max(0.05, float(meta.get("confidence", 1.0)) * 0.5)
        meta["confidence"] = new_conf
        self._col.upsert(ids=[note_id], documents=[doc], metadatas=[meta])

    def prune(self, scene_id: str, target_label: str, keep: int = 5) -> int:
        """Keep only the top *keep* notes by confidence for a scene+target.

        Returns the number of notes deleted.
        """
        notes = self.list_notes(scene_id=scene_id, target_label=target_label, limit=200)
        if len(notes) <= keep:
            return 0
        # Sort by confidence descending, then by created_at descending (newest first).
        notes.sort(key=lambda n: (n.get("confidence", 0), n.get("created_at", "")), reverse=True)
        to_delete = [n["note_id"] for n in notes[keep:]]
        if to_delete:
            self._col.delete(ids=to_delete)
        return len(to_delete)

    def clear(self) -> None:
        """Delete all notes."""
        try:
            self._client.delete_collection("notes")
        except Exception:  # noqa: BLE001, S110
            pass
        self._col = self._client.get_or_create_collection("notes")

    def count(self) -> int:
        """Return total number of stored notes."""
        return self._col.count()

    def list_notes(
        self,
        scene_id: str | None = None,
        target_label: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Return notes as raw dicts (for CLI inspect / debugging)."""
        where: dict | None = None
        if scene_id is not None:
            where = {"scene_id": scene_id}

        result = self._col.get(
            where=where,
            include=["documents", "metadatas"],
            limit=limit,
        )
        out: list[dict] = []
        for note_id, doc, meta in zip(
            result["ids"], result["documents"], result["metadatas"], strict=True
        ):
            if target_label is not None and meta.get("target_label") != target_label:
                continue
            out.append({"note_id": note_id, "text": doc, **meta})
        return out

    def get_metadata_keys(self) -> list[str]:
        """Return metadata key names — used by tests to assert no frames stored."""
        return ["scene_id", "target_label", "kind", "confidence", "created_at", "mission_id"]

    def _raw_get(self, note_ids: list[str]) -> dict:
        """Fetch raw ChromaDB result by ids — for staleness tests."""
        return self._col.get(ids=note_ids, include=["documents", "metadatas"])
