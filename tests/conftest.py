"""Global pytest configuration and fixtures."""

from __future__ import annotations

import hashlib
import os

import pytest

# ---------------------------------------------------------------------------
# Shared ChromaDB test helpers (used by test_memory.py + test_memory_heuristic.py)
# ---------------------------------------------------------------------------


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Deterministic hash-based 32-dim embedding (no model download)."""
    return [[float(b - 128) / 128.0 for b in hashlib.sha256(t.encode()).digest()] for t in texts]


class DummyEmbeddingFunction:
    """ChromaDB-compatible deterministic embedding function for tests.

    ChromaDB 1.5 calls embed_query() for queries and embed_documents() (or
    __call__) for documents.
    """

    def name(self) -> str:  # noqa: D102
        return "dummy"

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        return _embed_texts(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        return _embed_texts(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        return _embed_texts(input)


def _openrouter_key_available() -> bool:
    """Return True if OPENROUTER_API_KEY is set in the environment or in .env."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return True
    try:
        from kormoran.config import Settings

        return Settings().openrouter_api_key is not None
    except Exception:  # noqa: BLE001
        return False


def pytest_collection_modifyitems(
    config: pytest.Config,  # noqa: ARG001
    items: list[pytest.Item],
) -> None:
    """Auto-skip llm-marked tests when OPENROUTER_API_KEY is absent."""
    if _openrouter_key_available():
        return
    skip = pytest.mark.skip(reason="OPENROUTER_API_KEY not set")
    for item in items:
        if "llm" in item.keywords:
            item.add_marker(skip)
