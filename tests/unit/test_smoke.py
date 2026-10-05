"""Trivial smoke tests — no sim, no network, must pass in < 1 s."""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_import_kormoran() -> None:
    import kormoran

    assert kormoran.__version__ == "0.1.0"


@pytest.mark.unit
def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings can be instantiated without a real API key."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    # Clear lru_cache so monkeypatched env is picked up
    from kormoran.config import get_settings

    get_settings.cache_clear()
    s = get_settings()
    assert s.llm_model == "openai/gpt-4o-mini"
    assert s.max_llm_calls == 40
    assert s.max_steps == 50
    assert s.sim_gui is False
    get_settings.cache_clear()


@pytest.mark.unit
def test_settings_openrouter_key_not_logged(monkeypatch: pytest.MonkeyPatch) -> None:
    """The API key must never appear in the string representation of Settings."""
    secret = "sk-or-v1-supersecret"
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)
    from kormoran.config import get_settings

    get_settings.cache_clear()
    s = get_settings()
    assert secret not in repr(s)
    assert secret not in str(s)
    get_settings.cache_clear()
