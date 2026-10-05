"""Kormoran CLI — entry point for all commands."""

from __future__ import annotations

import logging
import sys

import typer

from kormoran import __version__

app = typer.Typer(
    name="kormoran",
    help="Kormoran — autonomous drone agent for object search in simulation.",
    add_completion=False,
)

logger = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@app.command()
def version() -> None:
    """Print the package version and exit."""
    typer.echo(f"kormoran {__version__}")


@app.command(name="check-llm")
def check_llm() -> None:
    """Verify the OpenRouter connection with a cheap round-trip; exits non-zero on failure."""
    from langchain_openai import ChatOpenAI

    from kormoran.config import get_settings

    settings = get_settings()
    _configure_logging(settings.log_level)

    if settings.openrouter_api_key is None:
        typer.echo("Error: OPENROUTER_API_KEY is not set.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Connecting to OpenRouter with model: {settings.llm_model}")
    try:
        llm = ChatOpenAI(
            model=settings.llm_model,
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key.get_secret_value(),
            timeout=settings.llm_timeout_s,
            max_tokens=10,
            temperature=0,
        )
        response = llm.invoke("Reply with the single word: ok")
        usage = response.usage_metadata or {}
        typer.echo(f"Response   : {response.content!r}")
        typer.echo(f"Model      : {settings.llm_model}")
        typer.echo(f"Tokens in  : {usage.get('input_tokens', '?')}")
        typer.echo(f"Tokens out : {usage.get('output_tokens', '?')}")
        typer.echo("check-llm: PASS")
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"Error: {exc}", err=True)
        logger.debug("check-llm failure", exc_info=True)
        sys.exit(1)
