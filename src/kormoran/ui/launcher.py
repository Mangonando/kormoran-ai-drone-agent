"""Subprocess launcher for mission runs (M7).

Launches ``uv run kormoran run-mission ...`` as a child process so the
Streamlit UI remains responsive while the agent runs headless.

The returned ``subprocess.Popen`` object is stored in Streamlit session_state
so callers can check ``proc.poll()`` (None = running) and read stdout.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def launch_mission(
    *,
    target: str,
    scenario: str,
    planner: str,
    seed: int,
    max_steps: int,
    memory: bool,
    data_dir: Path,
    mission_id: str = "",
) -> subprocess.Popen:  # type: ignore[type-arg]
    """Spawn a ``run-mission`` subprocess and return the Popen handle.

    stdout/stderr are captured as text so the UI can tail them.
    The process inherits the current environment (including any
    OPENROUTER_API_KEY set in the shell or .env file).
    """
    cmd = [
        sys.executable,
        "-m",
        "kormoran",
        "run-mission",
        "--target",
        target,
        "--scenario",
        scenario,
        "--planner",
        planner,
        "--seed",
        str(seed),
        "--max-steps",
        str(max_steps),
    ]
    if not memory:
        cmd.append("--no-memory")
    if mission_id:
        cmd.extend(["--mission-id", mission_id])

    # Try the installed script first; fall back to module invocation.
    kormoran_script = Path(sys.executable).parent / "kormoran"
    if kormoran_script.exists():
        cmd = [str(kormoran_script)] + cmd[3:]  # replace python -m kormoran with script

    return subprocess.Popen(  # noqa: S603
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def poll_output(proc: subprocess.Popen, max_lines: int = 200) -> list[str]:  # type: ignore[type-arg]
    """Non-blocking read of all available stdout lines."""
    lines: list[str] = []
    if proc.stdout is None:
        return lines
    # Read without blocking — only what's already buffered.
    import select

    ready, _, _ = select.select([proc.stdout], [], [], 0)
    if ready:
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            lines.append(line.rstrip())
            if len(lines) >= max_lines:
                break
    return lines
