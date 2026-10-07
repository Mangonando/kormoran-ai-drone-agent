"""Factory for the OpenRouter LLM client used by LLMPlanner."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_openai import ChatOpenAI

if TYPE_CHECKING:
    from kormoran.config import Settings


def make_llm(settings: Settings) -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at OpenRouter.

    Parameters
    ----------
    settings:
        Loaded Settings; ``openrouter_api_key`` must be non-None.
    """
    if settings.openrouter_api_key is None:
        msg = "OPENROUTER_API_KEY is not set — cannot create LLM client"
        raise RuntimeError(msg)

    return ChatOpenAI(
        model=settings.llm_model,
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key.get_secret_value(),
        timeout=settings.llm_timeout_s,
        temperature=0,
        max_tokens=500,
    )
