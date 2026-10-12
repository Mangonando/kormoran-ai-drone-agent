"""Episodic SQLite store for Kormoran (M5).

One row per completed mission.  Uses stdlib sqlite3 — no ORM.

Schema (missions table):
    mission_id TEXT PRIMARY KEY
    ts         TEXT   — ISO-8601 UTC
    scene_id   TEXT
    target_label TEXT
    outcome    TEXT
    steps      INTEGER
    duration_s REAL
    path       TEXT   — JSON list of {x,y,z,step} dicts  (NO frames/images)
    found_pose TEXT   — JSON or NULL
    found_from_pose TEXT — JSON or NULL
    model      TEXT
    tokens     INTEGER
    cost_usd   REAL
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from kormoran.memory.schemas import EpisodicRecord

_DDL = """\
CREATE TABLE IF NOT EXISTS missions (
    mission_id      TEXT PRIMARY KEY,
    ts              TEXT NOT NULL,
    scene_id        TEXT NOT NULL,
    target_label    TEXT NOT NULL,
    outcome         TEXT NOT NULL,
    steps           INTEGER NOT NULL,
    duration_s      REAL NOT NULL,
    path            TEXT NOT NULL,
    found_pose      TEXT,
    found_from_pose TEXT,
    model           TEXT NOT NULL DEFAULT '',
    tokens          INTEGER NOT NULL DEFAULT 0,
    cost_usd        REAL NOT NULL DEFAULT 0.0
);
"""


class EpisodicStore:
    """SQLite-backed episodic memory.  Thread-safe via per-call connections."""

    def __init__(self, db_path: str | Path) -> None:
        self._db = Path(db_path)
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    # Internal helpers

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_DDL)

    # ------------------------------------------------------------------
    # Public API

    def store(self, record: EpisodicRecord) -> None:
        """Insert or replace a mission record."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO missions
                    (mission_id, ts, scene_id, target_label, outcome, steps,
                     duration_s, path, found_pose, found_from_pose, model, tokens, cost_usd)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    record.mission_id,
                    record.ts,
                    record.scene_id,
                    record.target_label,
                    record.outcome,
                    record.steps,
                    record.duration_s,
                    json.dumps(record.path),
                    json.dumps(record.found_pose) if record.found_pose else None,
                    json.dumps(record.found_from_pose) if record.found_from_pose else None,
                    record.model,
                    record.tokens,
                    record.cost_usd,
                ),
            )

    def list_missions(
        self,
        scene_id: str | None = None,
        target_label: str | None = None,
        limit: int = 50,
    ) -> list[EpisodicRecord]:
        """Return missions, optionally filtered, newest first."""
        where_clauses: list[str] = []
        params: list[object] = []
        if scene_id is not None:
            where_clauses.append("scene_id = ?")
            params.append(scene_id)
        if target_label is not None:
            where_clauses.append("target_label = ?")
            params.append(target_label)
        where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM missions {where} ORDER BY ts DESC LIMIT ?",  # noqa: S608
                params,
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get(self, mission_id: str) -> EpisodicRecord | None:
        """Fetch a single mission by id, or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM missions WHERE mission_id = ?", (mission_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def clear(self) -> None:
        """Delete all records (useful for testing / ablation)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM missions")

    def count(self) -> int:
        """Return total number of stored missions."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM missions").fetchone()[0]

    # ------------------------------------------------------------------
    # Conversion

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> EpisodicRecord:
        d = dict(row)
        d["path"] = json.loads(d["path"])
        d["found_pose"] = json.loads(d["found_pose"]) if d["found_pose"] else None
        d["found_from_pose"] = json.loads(d["found_from_pose"]) if d["found_from_pose"] else None
        return EpisodicRecord.model_validate(d)
