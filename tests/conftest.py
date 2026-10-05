"""Global pytest configuration and fixtures."""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config,  # noqa: ARG001
    items: list[pytest.Item],
) -> None:
    """Auto-skip llm-marked tests when OPENROUTER_API_KEY is absent."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return
    skip = pytest.mark.skip(reason="OPENROUTER_API_KEY not set")
    for item in items:
        if "llm" in item.keywords:
            item.add_marker(skip)
