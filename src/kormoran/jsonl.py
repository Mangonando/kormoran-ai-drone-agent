"""JSONL file reader — shared helper used by eval, ui, and recording."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def read_jsonl(path: str | Path) -> list[dict]:
    """Parse a JSONL file and return a list of dicts.

    Silently skips blank and malformed lines.
    Returns an empty list if the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        return []
    records: list[dict] = []
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records
