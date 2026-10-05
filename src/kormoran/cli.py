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


@app.command(name="run-sim-smoke")
def run_sim_smoke(
    gui: bool = typer.Option(False, "--gui", help="Show PyBullet GUI"),
    scenario: str = typer.Option("living_room_v1", "--scenario", help="Scenario id"),
    seed: int = typer.Option(1, "--seed", help="Random seed"),
) -> None:
    """Smoke test: load sim, take off, scan (8 rotations), land, save GIF."""
    from pathlib import Path

    from kormoran.config import get_settings
    from kormoran.drones.sim import SimDrone
    from kormoran.recording import save_gif

    settings = get_settings()
    _configure_logging(settings.log_level)

    scenarios_dir = Path(settings.scenarios_dir)
    scenario_path = scenarios_dir / f"{scenario}.yaml"
    if not scenario_path.exists():
        typer.echo(f"Error: scenario file not found: {scenario_path}", err=True)
        raise typer.Exit(code=1)

    from kormoran.config import ScenarioConfig

    scene = ScenarioConfig.from_yaml(scenario_path)
    drone = SimDrone(scenario=scene, gui=gui, seed=seed)

    typer.echo(f"Connecting to sim (scene={scenario}, seed={seed}, gui={gui})...")
    drone.connect()
    frames = []

    try:
        typer.echo("Taking off to 1.0 m...")
        result = drone.takeoff(altitude_m=1.0)
        typer.echo(f"  status={result.status} z={result.pose.z:.3f} m ({result.elapsed_s:.1f} s)")
        frames.append(drone.get_frame())

        typer.echo("Scanning (8 × 45°)...")
        for i in range(8):
            result = drone.rotate(yaw_deg=45.0, timeout_s=8.0)
            f = drone.get_frame()
            frames.append(f)
            state = drone.get_state()
            typer.echo(f"  step {i + 1}/8  yaw={state.pose.yaw_deg:.1f}°  status={result.status}")

        typer.echo("Landing...")
        result = drone.land()
        typer.echo(f"  status={result.status} z={result.pose.z:.3f} m")
        frames.append(drone.get_frame())

    finally:
        drone.disconnect()

    # Save GIF
    data_dir = Path(settings.data_dir)
    gif_path = data_dir / "recordings" / f"smoke_{scenario}_seed{seed}.gif"
    out = save_gif(frames, gif_path)
    typer.echo(f"GIF saved → {out} ({len(frames)} frames)")
    typer.echo("run-sim-smoke: PASS")
