"""Unit tests for the M6 evaluation harness.

Tests run without PyBullet, LLM, or subprocesses — they exercise the
scenarios matrix builder, metrics computation, and report generation
with fixture data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_result(
    scenario: str = "living_room_v1",
    target: str = "red cube",
    seed: int = 1,
    planner: str = "heuristic",
    memory: bool = False,
    outcome: str = "found",
    steps: int = 12,
    path_length_m: float = 3.5,
    llm_calls: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    est_cost_usd: float = 0.0,
    safety_violations: int = 0,
) -> dict:
    mem = "mem" if memory else "nomem"
    return {
        "case_id": f"{scenario}__{target}__{planner}__s{seed}__{mem}",
        "scenario": scenario,
        "target": target,
        "seed": seed,
        "planner": planner,
        "memory": memory,
        "outcome": outcome,
        "steps": steps,
        "elapsed_s": 10.0,
        "path_length_m": path_length_m,
        "llm_calls": llm_calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "est_cost_usd": est_cost_usd,
        "safety_violations": safety_violations,
        "exit_code": 0,
        "stderr_tail": "",
    }


@pytest.fixture()
def results_file(tmp_path: Path) -> Path:
    """Write a fixture results JSONL with 4 runs."""
    results = [
        _make_result(seed=1, memory=False, steps=15),
        _make_result(seed=1, memory=True, steps=10),
        _make_result(seed=2, memory=False, steps=18),
        _make_result(seed=2, memory=True, steps=8),
    ]
    path = tmp_path / "results.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in results) + "\n")
    return path


# ===========================================================================
# scenarios tests
# ===========================================================================


class TestScenarios:
    def test_build_matrix_default(self) -> None:
        from kormoran.eval.scenarios import build_matrix

        cases = build_matrix(
            scenarios=["living_room_v1"],
            seeds=2,
            planners=["heuristic"],
            memory_modes=[False],
        )
        # living_room has 2 targets × 2 seeds × 1 planner × 1 memory = 4
        assert len(cases) == 4

    def test_build_matrix_all_combos(self) -> None:
        from kormoran.eval.scenarios import build_matrix

        cases = build_matrix(
            scenarios=["living_room_v1"],
            seeds=1,
            planners=["heuristic", "llm"],
            memory_modes=[True, False],
        )
        # 2 targets × 1 seed × 2 planners × 2 memory = 8
        assert len(cases) == 8

    def test_absent_target_uses_synthetic_label(self) -> None:
        from kormoran.eval.scenarios import build_matrix

        cases = build_matrix(
            scenarios=["absent_target"],
            seeds=1,
            planners=["heuristic"],
            memory_modes=[False],
        )
        assert len(cases) == 1
        assert cases[0]["target"] == "red cube"

    def test_case_fields(self) -> None:
        from kormoran.eval.scenarios import build_matrix

        cases = build_matrix(
            scenarios=["living_room_v1"],
            seeds=1,
            planners=["heuristic"],
            memory_modes=[True],
        )
        case = cases[0]
        assert "scenario" in case
        assert "target" in case
        assert "seed" in case
        assert "planner" in case
        assert "memory" in case


# ===========================================================================
# metrics tests
# ===========================================================================


class TestMetrics:
    def test_load_results(self, results_file: Path) -> None:
        from kormoran.eval.metrics import load_results

        results = load_results(results_file)
        assert len(results) == 4

    def test_load_missing_file(self, tmp_path: Path) -> None:
        from kormoran.eval.metrics import load_results

        assert load_results(tmp_path / "nope.jsonl") == []

    def test_group_metrics(self, results_file: Path) -> None:
        from kormoran.eval.metrics import compute_group_metrics, load_results

        results = load_results(results_file)
        groups = compute_group_metrics(results)
        # 2 groups: memory on and memory off
        assert len(groups) == 2
        for g in groups:
            assert g["runs"] == 2
            assert g["success_rate"] == 1.0
            assert g["safety_violations"] == 0

    def test_memory_on_fewer_steps(self, results_file: Path) -> None:
        from kormoran.eval.metrics import compute_group_metrics, load_results

        results = load_results(results_file)
        groups = compute_group_metrics(results)
        mem_on = next(g for g in groups if g["memory"])
        mem_off = next(g for g in groups if not g["memory"])
        assert mem_on["mean_steps"] < mem_off["mean_steps"]

    def test_summary_table_markdown(self, results_file: Path) -> None:
        from kormoran.eval.metrics import compute_group_metrics, load_results, summary_table

        results = load_results(results_file)
        groups = compute_group_metrics(results)
        table = summary_table(groups)
        assert "| Scenario" in table
        assert "living_room_v1" in table
        assert table.count("\n") >= 3  # header + sep + 2 rows

    def test_absent_target_success(self) -> None:
        """absent_target: 'not_found' is the correct outcome."""
        from kormoran.eval.metrics import compute_group_metrics

        results = [
            _make_result(scenario="absent_target", outcome="not_found"),
            _make_result(scenario="absent_target", outcome="not_found", seed=2),
        ]
        groups = compute_group_metrics(results)
        assert groups[0]["success_rate"] == 1.0

    def test_false_found_rate(self) -> None:
        from kormoran.eval.metrics import compute_group_metrics

        results = [
            _make_result(scenario="absent_target", outcome="found"),
            _make_result(scenario="absent_target", outcome="not_found", seed=2),
        ]
        groups = compute_group_metrics(results)
        assert groups[0]["false_found_rate"] == 0.5


# ===========================================================================
# report tests
# ===========================================================================


class TestReport:
    def test_generate_report_creates_file(self, results_file: Path, tmp_path: Path) -> None:
        from kormoran.eval.report import generate_report

        report_path = generate_report(
            results_file,
            output_path=tmp_path / "report.md",
            plots_dir=tmp_path / "plots",
        )
        assert report_path.exists()
        content = report_path.read_text()
        assert "# Evaluation Results" in content
        assert "living_room_v1" in content
        assert "Safety violations" in content

    def test_generate_report_no_results(self, tmp_path: Path) -> None:
        from kormoran.eval.report import generate_report

        report_path = generate_report(
            tmp_path / "empty.jsonl",
            output_path=tmp_path / "report.md",
        )
        assert report_path.exists()
        assert "No results yet" in report_path.read_text()

    def test_generate_plots(self, results_file: Path, tmp_path: Path) -> None:
        from kormoran.eval.metrics import load_results
        from kormoran.eval.report import generate_plots

        results = load_results(results_file)
        paths = generate_plots(results, tmp_path / "plots")
        # Should produce at least the success_rates plot
        assert len(paths) >= 1
        for p in paths:
            assert p.exists()
            assert p.suffix == ".png"


# ===========================================================================
# runner helper tests (no subprocess)
# ===========================================================================


class TestRunnerHelpers:
    def test_case_id_deterministic(self) -> None:
        from kormoran.eval.runner import _case_id

        case = {
            "scenario": "living_room_v1",
            "target": "red cube",
            "seed": 1,
            "planner": "heuristic",
            "memory": False,
        }
        cid = _case_id(case)
        assert "living_room_v1" in cid
        assert "red cube" in cid
        assert "nomem" in cid
        assert _case_id(case) == cid  # deterministic

    def test_load_completed_empty(self, tmp_path: Path) -> None:
        from kormoran.eval.runner import _load_completed

        assert _load_completed(tmp_path / "missing.jsonl") == set()

    def test_load_completed_round_trip(self, results_file: Path) -> None:
        from kormoran.eval.runner import _load_completed

        completed = _load_completed(results_file)
        assert len(completed) == 4

    def test_outcome_from_log_found(self) -> None:
        from kormoran.eval.runner import _outcome_from_log

        steps = [{"action": {"kind": "land"}, "result": {"status": "ok"}}]
        assert _outcome_from_log(steps, 0) == "found"

    def test_outcome_from_log_not_found(self) -> None:
        from kormoran.eval.runner import _outcome_from_log

        steps = [{"action": {"kind": "report_not_found"}, "result": {"status": "ok"}}]
        assert _outcome_from_log(steps, 0) == "not_found"

    def test_outcome_from_log_error(self) -> None:
        from kormoran.eval.runner import _outcome_from_log

        assert _outcome_from_log([], exit_code=1) == "error"

    def test_path_length(self) -> None:
        from kormoran.eval.runner import _path_length

        steps = [
            {"pose": {"x": 0, "y": 0, "z": 1}},
            {"pose": {"x": 3, "y": 4, "z": 1}},  # distance = 5
        ]
        assert _path_length(steps) == pytest.approx(5.0)
