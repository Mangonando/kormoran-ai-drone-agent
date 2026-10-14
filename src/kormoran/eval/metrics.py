"""Metric computation from evaluation results (M6).

Reads the results JSONL written by ``runner.run_matrix`` and computes
aggregate statistics per (scenario, planner, memory) group.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import TypedDict


class GroupMetrics(TypedDict):
    """Aggregated metrics for one (scenario, target, planner, memory) group."""

    scenario: str
    target: str
    planner: str
    memory: bool
    runs: int
    success_rate: float
    false_found_rate: float
    mean_steps: float
    median_steps: float
    mean_path_length_m: float
    mean_elapsed_s: float
    total_llm_calls: int
    total_tokens: int
    total_cost_usd: float
    safety_violations: int


def load_results(results_path: str | Path) -> list[dict]:
    """Load all result records from a JSONL file."""
    from kormoran.jsonl import read_jsonl

    return read_jsonl(results_path)


def _group_key(r: dict) -> tuple[str, str, str, bool]:
    return (r["scenario"], r["target"], r["planner"], r["memory"])


def compute_group_metrics(results: list[dict]) -> list[GroupMetrics]:
    """Compute per-group aggregate metrics from flat result records."""
    groups: dict[tuple[str, str, str, bool], list[dict]] = {}
    for r in results:
        key = _group_key(r)
        groups.setdefault(key, []).append(r)

    out: list[GroupMetrics] = []
    for (scenario, target, planner, memory), runs in sorted(groups.items()):
        n = len(runs)

        # Success: "found" for scenes with targets, "not_found" for absent_target
        is_absent = scenario == "absent_target"
        success_outcome = "not_found" if is_absent else "found"
        successes = sum(1 for r in runs if r["outcome"] == success_outcome)
        success_rate = successes / n if n else 0.0

        # False-found: claimed "found" when target was absent
        false_founds = sum(1 for r in runs if is_absent and r["outcome"] == "found")
        false_found_rate = false_founds / n if n else 0.0

        steps_list = [r["steps"] for r in runs if r["steps"] > 0]
        mean_steps = statistics.mean(steps_list) if steps_list else 0.0
        median_steps = statistics.median(steps_list) if steps_list else 0.0

        path_lengths = [r.get("path_length_m", 0.0) for r in runs]
        mean_path = statistics.mean(path_lengths) if path_lengths else 0.0

        elapsed_list = [r.get("elapsed_s", 0.0) for r in runs]
        mean_elapsed = statistics.mean(elapsed_list) if elapsed_list else 0.0

        total_llm = sum(r.get("llm_calls", 0) for r in runs)
        total_tok = sum(r.get("prompt_tokens", 0) + r.get("completion_tokens", 0) for r in runs)
        total_cost = sum(r.get("est_cost_usd", 0.0) for r in runs)
        safety = sum(r.get("safety_violations", 0) for r in runs)

        out.append(
            GroupMetrics(
                scenario=scenario,
                target=target,
                planner=planner,
                memory=memory,
                runs=n,
                success_rate=round(success_rate, 3),
                false_found_rate=round(false_found_rate, 3),
                mean_steps=round(mean_steps, 2),
                median_steps=round(median_steps, 1),
                mean_path_length_m=round(mean_path, 3),
                mean_elapsed_s=round(mean_elapsed, 2),
                total_llm_calls=total_llm,
                total_tokens=total_tok,
                total_cost_usd=round(total_cost, 4),
                safety_violations=safety,
            )
        )

    return out


def summary_table(groups: list[GroupMetrics]) -> str:
    """Render a Markdown table of group metrics."""
    header = (
        "| Scenario | Target | Planner | Memory | Runs | Success | "
        "False-found | Mean steps | Median steps | Mean path (m) | "
        "LLM calls | Tokens | Cost ($) | Safety |\n"
    )
    sep = "|" + "|".join(["---"] * 14) + "|\n"
    rows = ""
    for g in groups:
        mem_str = "on" if g["memory"] else "off"
        rows += (
            f"| {g['scenario']} | {g['target']} | {g['planner']} | {mem_str} "
            f"| {g['runs']} | {g['success_rate']:.1%} | {g['false_found_rate']:.1%} "
            f"| {g['mean_steps']:.1f} | {g['median_steps']:.0f} "
            f"| {g['mean_path_length_m']:.2f} "
            f"| {g['total_llm_calls']} | {g['total_tokens']} "
            f"| {g['total_cost_usd']:.4f} | {g['safety_violations']} |\n"
        )
    return header + sep + rows
