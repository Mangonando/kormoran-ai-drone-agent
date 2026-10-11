"""Kormoran CLI — entry point for all commands."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

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


@app.command(name="eval-detector")
def eval_detector(
    scenario: str = typer.Option("living_room_v1", "--scenario", help="Scenario id"),
    seed: int = typer.Option(1, "--seed", help="Random seed for object placement"),
    target_label: str = typer.Option("red cube", "--target", help="Target label to evaluate"),
    n_positions: int = typer.Option(5, "--positions", help="Grid positions to visit"),
    yaws_per_pos: int = typer.Option(8, "--yaws", help="Yaw increments per position"),
    altitude: float = typer.Option(1.0, "--altitude", help="Flight altitude in metres"),
    save_results: bool = typer.Option(True, "--save/--no-save", help="Append to eval_results.md"),
) -> None:
    """Evaluate color detector precision/recall vs PyBullet segmentation ground truth.

    Flies the simulated drone to *n_positions* positions × *yaws_per_pos* yaws
    (total ≈ n_positions × yaws_per_pos frames) and compares ColorDetector output
    to ground-truth detections derived from PyBullet's segmentation mask.
    Results are printed and, by default, appended to docs/eval_results.md.
    """

    from kormoran.config import ScenarioConfig, get_settings
    from kormoran.drones.sim import SimDrone
    from kormoran.perception.color_detector import ColorDetector
    from kormoran.perception.ground_truth import detections_from_seg_mask
    from kormoran.types import TargetSpec

    settings = get_settings()
    _configure_logging(settings.log_level)

    scenarios_dir = Path(settings.scenarios_dir)
    scenario_path = scenarios_dir / f"{scenario}.yaml"
    if not scenario_path.exists():
        typer.echo(f"Error: scenario file not found: {scenario_path}", err=True)
        raise typer.Exit(code=1)

    scene = ScenarioConfig.from_yaml(scenario_path)
    # Match target spec from scenario
    target_spec: TargetSpec | None = None
    for t in scene.targets:
        if t.label == target_label:
            target_spec = TargetSpec(label=t.label, color=t.color, shape=t.shape)
            break
    if target_spec is None:
        typer.echo(f"Error: label {target_label!r} not in scenario {scenario!r}", err=True)
        raise typer.Exit(code=1)

    drone = SimDrone(scenario=scene, gui=False, seed=seed)
    detector = ColorDetector()

    typer.echo(f"eval-detector: scenario={scenario} target={target_label!r} seed={seed}")
    typer.echo(f"  positions={n_positions} yaws_per_pos={yaws_per_pos} altitude={altitude}m")

    drone.connect()
    tp = fp = fn = 0
    total_frames = 0

    try:
        drone.takeoff(altitude_m=altitude)

        # Build a 1-D grid of x positions within the geofence
        x_lo = scene.room.x[0] * 0.7
        x_hi = scene.room.x[1] * 0.7
        xs = [x_lo + (x_hi - x_lo) * i / max(1, n_positions - 1) for i in range(n_positions)]
        yaw_step = 360.0 / yaws_per_pos

        for i, x_target in enumerate(xs):
            # Move to this x, hold y=0
            curr = drone.get_state()
            dx = x_target - curr.pose.x
            drone.move_relative(dx, 0.0, 0.0, timeout_s=20.0)

            # Rotate through yaws_per_pos steps
            for _ in range(yaws_per_pos):
                frame, seg, body_ids = drone.get_seg_frame()
                gt_dets = detections_from_seg_mask(seg, body_ids, target_spec)
                pred_dets = detector.detect(frame, target_spec)

                gt_visible = len(gt_dets) > 0
                pred_visible = len(pred_dets) > 0

                if gt_visible and pred_visible:
                    tp += 1
                elif not gt_visible and pred_visible:
                    fp += 1
                elif gt_visible and not pred_visible:
                    fn += 1
                total_frames += 1

                drone.rotate(yaw_step, timeout_s=5.0)

            typer.echo(
                f"  position {i + 1}/{n_positions}  x={x_target:.1f}  TP={tp} FP={fp} FN={fn}"
            )

        drone.land()

    finally:
        drone.disconnect()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    typer.echo(f"\nResults over {total_frames} frames:")
    typer.echo(f"  TP={tp}  FP={fp}  FN={fn}")
    typer.echo(f"  Precision : {precision:.3f}  (target ≥ 0.95)")
    typer.echo(f"  Recall    : {recall:.3f}  (target ≥ 0.90)")

    passed = precision >= 0.95 and recall >= 0.90
    typer.echo(f"  Status    : {'PASS' if passed else 'FAIL'}")

    if save_results:
        results_path = Path("docs/eval_results.md")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("a") as fh:
            fh.write(
                f"\n## Detector eval — {scenario} / {target_label!r} (seed {seed})\n"
                f"- Frames: {total_frames}  TP={tp}  FP={fp}  FN={fn}\n"
                f"- Precision: {precision:.3f}  Recall: {recall:.3f}\n"
                f"- Status: {'PASS' if passed else 'FAIL'}\n"
            )
        typer.echo(f"  Results appended → {results_path}")

    if not passed:
        raise typer.Exit(code=1)


@app.command(name="run-mission")
def run_mission(
    target: str = typer.Option(..., "--target", help="Target label (e.g. 'red cube')"),
    scenario: str = typer.Option("living_room_v1", "--scenario", help="Scenario id"),
    seed: int = typer.Option(1, "--seed", help="Random seed for object placement"),
    planner_name: str = typer.Option(
        "heuristic",
        "--planner",
        help="Planner: heuristic | scripted | llm",
    ),
    gui: bool = typer.Option(False, "--gui", help="Show PyBullet GUI"),
    max_steps: int = typer.Option(50, "--max-steps", help="Max decision steps"),
) -> None:
    """Run a full LangGraph mission loop in the simulator."""
    import time

    from kormoran.agent.graph import build_mission_graph, make_initial_state, new_mission_id
    from kormoran.agent.planners import HeuristicPlanner, ScriptedPlanner
    from kormoran.config import ScenarioConfig, get_settings
    from kormoran.drones.sim import SimDrone
    from kormoran.perception.color_detector import ColorDetector
    from kormoran.recording import MissionLogger, save_gif
    from kormoran.types import TargetSpec

    settings = get_settings()
    _configure_logging(settings.log_level)

    scenarios_dir = Path(settings.scenarios_dir)
    scenario_path = scenarios_dir / f"{scenario}.yaml"
    if not scenario_path.exists():
        typer.echo(f"Error: scenario file not found: {scenario_path}", err=True)
        raise typer.Exit(code=1)

    scene = ScenarioConfig.from_yaml(scenario_path)
    target_spec: TargetSpec | None = None
    for t in scene.targets:
        if t.label == target:
            target_spec = TargetSpec(label=t.label, color=t.color, shape=t.shape)
            break
    if target_spec is None:
        typer.echo(f"Error: label {target!r} not in scenario {scenario!r}", err=True)
        raise typer.Exit(code=1)

    # Build planner.
    if planner_name == "heuristic":
        planner = HeuristicPlanner()
    elif planner_name == "scripted":
        from kormoran.types import Action

        planner = ScriptedPlanner(
            [
                Action(kind="scan", reasoning="scripted scan", confidence=1.0),
                Action(
                    kind="report_not_found",
                    reasoning="scripted end",
                    confidence=1.0,
                ),
            ]
        )
    elif planner_name == "llm":
        from kormoran.agent.llm import make_llm
        from kormoran.agent.planners import LLMPlanner

        if settings.openrouter_api_key is None:
            typer.echo("Error: OPENROUTER_API_KEY required for llm planner", err=True)
            raise typer.Exit(code=1)
        planner = LLMPlanner(make_llm(settings))
    else:
        typer.echo(f"Error: unknown planner {planner_name!r}", err=True)
        raise typer.Exit(code=1)

    drone = SimDrone(scenario=scene, gui=gui, seed=seed)
    perceiver = ColorDetector()
    mission_id = new_mission_id()

    typer.echo(
        f"run-mission: id={mission_id} target={target!r} scenario={scenario}"
        f" planner={planner_name} seed={seed}"
    )

    drone.connect()
    try:
        drone.takeoff(altitude_m=1.0)
        drone_state = drone.get_state()

        # Set up optional JSONL log before building the graph so the logger
        # is injected into act_node via closure.
        log_path = Path(settings.data_dir) / "missions" / f"{mission_id}.jsonl"
        mission_logger: MissionLogger | None = None
        if settings.record_missions:
            mission_logger = MissionLogger(log_path)

        graph = build_mission_graph(
            drone, planner, perceiver, settings, mission_logger=mission_logger
        )
        initial = make_initial_state(
            mission_id, scene.scene_id, target_spec, drone_state, max_steps
        )

        t0 = time.monotonic()
        thread_config = {"configurable": {"thread_id": mission_id}}
        final_state = graph.invoke(initial, config=thread_config)
        elapsed = time.monotonic() - t0

        status = final_state.get("status", "unknown")
        steps = final_state.get("step", 0)
        usage = final_state.get("usage", {})

        typer.echo(f"\nMission complete in {elapsed:.1f}s ({steps} steps)")
        typer.echo(f"  Status   : {status}")
        typer.echo(f"  LLM calls: {usage.get('llm_calls', 0)}")
        typer.echo(f"  Est. cost: ${usage.get('est_cost_usd', 0.0):.4f}")

        if mission_logger is not None:
            typer.echo(f"  Log      : {log_path}")
            if mission_logger.frames:
                gif_path = Path(settings.data_dir) / "missions" / f"{mission_id}.gif"
                save_gif(mission_logger.frames, gif_path, fps=2, scale=4, loop=1)
                typer.echo(f"  GIF      : {gif_path}")
            if mission_logger.top_frames:
                top_path = Path(settings.data_dir) / "missions" / f"{mission_id}_top.gif"
                save_gif(mission_logger.top_frames, top_path, fps=2, scale=4, loop=1)
                typer.echo(f"  Top GIF  : {top_path}")

        if status not in ("found",):
            raise typer.Exit(code=1)

    finally:
        try:
            drone.land()
        except Exception:  # noqa: BLE001, S110
            pass
        drone.disconnect()


@app.command(name="run-sim-smoke")
def run_sim_smoke(
    gui: bool = typer.Option(False, "--gui", help="Show PyBullet GUI"),
    scenario: str = typer.Option("living_room_v1", "--scenario", help="Scenario id"),
    seed: int = typer.Option(1, "--seed", help="Random seed"),
    pause: bool = typer.Option(
        False, "--pause", help="Pause after loading the scene (explore GUI before running)"
    ),
) -> None:
    """Smoke test: load sim, take off, scan (8 rotations), fly forward/back, land, save GIF."""
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

    if pause:
        typer.echo(
            "Scene loaded. Explore the PyBullet GUI, then press Enter to run the smoke test..."
        )
        input()

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

        cruise_m = 0.8
        for label, dx, dy in [
            ("forward", cruise_m, 0.0),
            ("right", 0.0, -cruise_m),
            ("forward", cruise_m, 0.0),
            ("left", 0.0, cruise_m),
            ("back", -cruise_m, 0.0),
            ("left", 0.0, cruise_m),
            ("back", -cruise_m, 0.0),
            ("right", 0.0, -cruise_m),
        ]:
            typer.echo(f"Flying {label} {cruise_m} m...")
            result = drone.move_relative(dx, dy, 0.0, timeout_s=15.0)
            frames.append(drone.get_frame())
            state = drone.get_state()
            pos = f"({state.pose.x:.2f}, {state.pose.y:.2f})"
            typer.echo(f"  {result.status}  {pos} {result.elapsed_s:.1f}s")

        typer.echo("Climbing to 1.8 m...")
        result = drone.move_relative(0.0, 0.0, 0.8, timeout_s=10.0)
        frames.append(drone.get_frame())
        typer.echo(f"  {result.status}  z={drone.get_state().pose.z:.2f} m")

        typer.echo("Scanning high (8 × 45°)...")
        for i in range(8):
            result = drone.rotate(yaw_deg=45.0, timeout_s=8.0)
            frames.append(drone.get_frame())
            state = drone.get_state()
            typer.echo(f"  step {i + 1}/8  yaw={state.pose.yaw_deg:.1f}°  status={result.status}")

        typer.echo("Landing...")
        result = drone.land()
        typer.echo(f"  status={result.status} z={result.pose.z:.3f} m")
        frames.append(drone.get_frame())

    finally:
        if pause:
            typer.echo("Flight done. Explore the GUI, then press Enter to close...")
            input()
        drone.disconnect()

    # Save GIF
    data_dir = Path(settings.data_dir)
    gif_path = data_dir / "recordings" / f"smoke_{scenario}_seed{seed}.gif"
    out = save_gif(frames, gif_path)
    typer.echo(f"GIF saved → {out} ({len(frames)} frames)")
    typer.echo("run-sim-smoke: PASS")
