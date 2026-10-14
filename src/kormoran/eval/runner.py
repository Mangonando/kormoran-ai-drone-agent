"""Headless evaluation runner (M6).

Runs each ``EvalCase`` as a subprocess via ``kormoran run-mission``, captures
the JSONL mission log, and writes one result line per case to a results JSONL
file.  Already-completed cases are skipped (resumable).
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

from kormoran.eval.scenarios import EvalCase

logger = logging.getLogger(__name__)


def _case_id(case: EvalCase) -> str:
    """Deterministic string key for dedup / resumption."""
    mem = "mem" if case["memory"] else "nomem"
    return f"{case['scenario']}__{case['target']}__{case['planner']}__s{case['seed']}__{mem}"


def _load_completed(results_path: Path) -> set[str]:
    """Load case IDs already present in the results file."""
    if not results_path.exists():
        return set()
    ids: set[str] = set()
    with results_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                ids.add(rec.get("case_id", ""))
            except json.JSONDecodeError:
                pass
    return ids


def run_case(
    case: EvalCase,
    *,
    data_dir: str = "./data",
    max_steps: int = 50,
    timeout_s: float = 300.0,
) -> dict:
    """Run one eval case as a subprocess and return a result dict.

    The subprocess calls ``kormoran run-mission`` in headless mode.
    After it finishes, the JSONL mission log is parsed to extract metrics.
    """
    case_id = _case_id(case)
    mission_id = f"eval_{case_id}"

    cmd = [
        sys.executable,
        "-m",
        "kormoran",
        "run-mission",
        "--target",
        case["target"],
        "--scenario",
        case["scenario"],
        "--seed",
        str(case["seed"]),
        "--planner",
        case["planner"],
        "--max-steps",
        str(max_steps),
        "--mission-id",
        mission_id,
    ]
    if not case["memory"]:
        cmd.append("--no-memory")

    t0 = time.monotonic()
    try:
        proc = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        exit_code = proc.returncode
        stderr = proc.stderr
    except subprocess.TimeoutExpired:
        exit_code = -1
        stderr = "TIMEOUT"
    elapsed = time.monotonic() - t0

    # Parse the JSONL log written by the mission
    log_path = Path(data_dir) / "missions" / f"{mission_id}.jsonl"
    steps_data = _parse_log(log_path)

    # Derive outcome
    outcome = _outcome_from_log(steps_data, exit_code)

    # Collect metrics from log
    n_steps = len(steps_data)
    usage = steps_data[-1].get("usage", {}) if steps_data else {}
    safety_events: list[dict] = []
    for s in steps_data:
        safety_events.extend(s.get("safety_events") or [])

    # Path length
    path_length = _path_length(steps_data)

    return {
        "case_id": case_id,
        "scenario": case["scenario"],
        "target": case["target"],
        "seed": case["seed"],
        "planner": case["planner"],
        "memory": case["memory"],
        "outcome": outcome,
        "steps": n_steps,
        "elapsed_s": round(elapsed, 2),
        "path_length_m": round(path_length, 3),
        "llm_calls": usage.get("llm_calls", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "est_cost_usd": usage.get("est_cost_usd", 0.0),
        "safety_violations": len(safety_events),
        "exit_code": exit_code,
        "stderr_tail": (stderr or "")[-500:],
    }


def _parse_log(log_path: Path) -> list[dict]:
    from kormoran.jsonl import read_jsonl

    return read_jsonl(log_path)


def _outcome_from_log(steps: list[dict], exit_code: int) -> str:
    if not steps:
        return "error" if exit_code != 0 else "no_log"
    last = steps[-1]
    action_kind = (last.get("action") or {}).get("kind", "")
    if action_kind == "land":
        return "found"
    if action_kind == "report_not_found":
        return "not_found"
    result_status = (last.get("result") or {}).get("status", "")
    if result_status == "aborted":
        return "aborted"
    if exit_code != 0:
        return "error"
    return "unknown"


def _path_length(steps: list[dict]) -> float:
    """Sum Euclidean distances between consecutive poses."""
    total = 0.0
    prev = None
    for s in steps:
        pose = s.get("pose")
        if pose is None:
            continue
        x, y, z = pose.get("x", 0), pose.get("y", 0), pose.get("z", 0)
        if prev is not None:
            dx = x - prev[0]
            dy = y - prev[1]
            dz = z - prev[2]
            total += (dx**2 + dy**2 + dz**2) ** 0.5
        prev = (x, y, z)
    return total


def run_matrix(
    cases: list[EvalCase],
    results_path: str | Path,
    *,
    data_dir: str = "./data",
    max_steps: int = 50,
    timeout_s: float = 300.0,
) -> list[dict]:
    """Run all cases in sequence, appending results to *results_path*.

    Already-completed cases (by case_id) are skipped for resumability.
    """
    rpath = Path(results_path)
    rpath.parent.mkdir(parents=True, exist_ok=True)

    completed = _load_completed(rpath)
    results: list[dict] = []

    for i, case in enumerate(cases):
        cid = _case_id(case)
        if cid in completed:
            logger.info("[%d/%d] SKIP %s (already done)", i + 1, len(cases), cid)
            continue

        logger.info("[%d/%d] RUN  %s", i + 1, len(cases), cid)
        result = run_case(
            case,
            data_dir=data_dir,
            max_steps=max_steps,
            timeout_s=timeout_s,
        )
        results.append(result)

        # Append immediately for resumability
        with rpath.open("a") as fh:
            fh.write(json.dumps(result) + "\n")

        status = result["outcome"]
        logger.info(
            "  → %s  steps=%d  cost=$%.4f  safety=%d",
            status,
            result["steps"],
            result["est_cost_usd"],
            result["safety_violations"],
        )

    return results
