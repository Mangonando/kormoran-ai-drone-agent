"""Global pytest configuration and fixtures."""

from __future__ import annotations

import os

import pytest


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
