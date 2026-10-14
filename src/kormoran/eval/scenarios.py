"""Scenario matrix for the evaluation harness (M6).

Generates a list of ``EvalCase`` dicts — each one is a single mission run.
The full matrix is: scenes × targets × seeds × planners × memory modes.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import yaml


class EvalCase(TypedDict):
    """One evaluation run."""

    scenario: str
    target: str
    seed: int
    planner: str
    memory: bool


# Default evaluation parameters
DEFAULT_SEEDS: int = 5
DEFAULT_PLANNERS: tuple[str, ...] = ("heuristic", "llm")
DEFAULT_MEMORY_MODES: tuple[bool, ...] = (True, False)


def _targets_from_yaml(path: Path) -> list[str]:
    """Extract target labels from a scenario YAML file."""
    with path.open() as fh:
        data = yaml.safe_load(fh)
    targets = data.get("targets") or []
    return [t["label"] for t in targets]


def build_matrix(
    scenarios_dir: str | Path = "configs/scenarios",
    *,
    scenarios: list[str] | None = None,
    seeds: int = DEFAULT_SEEDS,
    planners: list[str] | None = None,
    memory_modes: list[bool] | None = None,
) -> list[EvalCase]:
    """Build the full evaluation matrix.

    Parameters
    ----------
    scenarios_dir:
        Path to the directory containing scenario YAML files.
    scenarios:
        Scenario IDs to include.  ``None`` → all YAML files in *scenarios_dir*.
    seeds:
        Number of seeds (1 … seeds inclusive).
    planners:
        Planner names to include.  ``None`` → ``("heuristic", "llm")``.
    memory_modes:
        Memory on/off modes.  ``None`` → ``(True, False)``.
    """
    sdir = Path(scenarios_dir)
    if scenarios is None:
        scenarios = sorted(p.stem for p in sdir.glob("*.yaml"))

    planner_list = list(planners or DEFAULT_PLANNERS)
    mem_list = list(memory_modes if memory_modes is not None else DEFAULT_MEMORY_MODES)

    cases: list[EvalCase] = []
    for scenario in scenarios:
        yaml_path = sdir / f"{scenario}.yaml"
        targets = _targets_from_yaml(yaml_path) if yaml_path.exists() else []
        # For absent_target (no targets), use a synthetic target label
        if not targets:
            targets = ["red cube"]

        for target in targets:
            for seed in range(1, seeds + 1):
                for planner in planner_list:
                    for memory in mem_list:
                        cases.append(
                            EvalCase(
                                scenario=scenario,
                                target=target,
                                seed=seed,
                                planner=planner,
                                memory=memory,
                            )
                        )

    return cases
