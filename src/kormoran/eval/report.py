"""Report generator — Markdown + PNG plots → ``docs/eval_results.md`` (M6).

Reads the results JSONL, computes group metrics, generates plots, and writes
a self-contained Markdown report with embedded image references.
"""

from __future__ import annotations

import logging
from pathlib import Path

from kormoran.eval.metrics import compute_group_metrics, load_results, summary_table

logger = logging.getLogger(__name__)


def generate_plots(results: list[dict], plots_dir: str | Path) -> list[Path]:
    """Generate PNG plots and return their paths.

    Plots produced:
    1. Steps-to-find: memory on vs off (grouped bar, per scenario)
    2. LLM vs heuristic comparison (grouped bar)
    3. Steps distribution across seeds (box plot)
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plot generation")
        return []

    pdir = Path(plots_dir)
    pdir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    groups = compute_group_metrics(results)

    # --- Plot 1: Memory on vs off (mean steps, per scenario×target×planner) ---
    _plot_memory_comparison(groups, pdir, paths, plt)

    # --- Plot 2: LLM vs Heuristic (mean steps, per scenario×target×memory) ---
    _plot_planner_comparison(groups, pdir, paths, plt)

    # --- Plot 3: Success rate overview ---
    _plot_success_rates(groups, pdir, paths, plt)

    return paths


def _plot_memory_comparison(groups: list[dict], pdir: Path, paths: list[Path], plt: object) -> None:
    """Bar chart: mean steps with memory on vs off."""
    # Collect matching on/off pairs
    by_key: dict[tuple, dict] = {}
    for g in groups:
        base = (g["scenario"], g["target"], g["planner"])
        by_key.setdefault(base, {})[g["memory"]] = g["mean_steps"]

    labels = []
    mem_on = []
    mem_off = []
    for key, vals in sorted(by_key.items()):
        if True in vals and False in vals:
            labels.append(f"{key[0]}\n{key[1]}\n{key[2]}")
            mem_on.append(vals[True])
            mem_off.append(vals[False])

    if not labels:
        return

    import numpy as np

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2), 5))  # type: ignore[attr-defined]
    ax.bar(x - width / 2, mem_off, width, label="Memory OFF", color="#e74c3c", alpha=0.8)
    ax.bar(x + width / 2, mem_on, width, label="Memory ON", color="#2ecc71", alpha=0.8)
    ax.set_ylabel("Mean steps")
    ax.set_title("Memory ON vs OFF — Mean Steps to Completion")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.legend()
    fig.tight_layout()
    p = pdir / "memory_comparison.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)  # type: ignore[attr-defined]
    paths.append(p)


def _plot_planner_comparison(
    groups: list[dict], pdir: Path, paths: list[Path], plt: object
) -> None:
    """Bar chart: LLM vs heuristic mean steps."""
    by_key: dict[tuple, dict] = {}
    for g in groups:
        base = (g["scenario"], g["target"], g["memory"])
        by_key.setdefault(base, {})[g["planner"]] = g["mean_steps"]

    labels = []
    llm_vals = []
    heur_vals = []
    for key, vals in sorted(by_key.items()):
        if "llm" in vals and "heuristic" in vals:
            mem_str = "mem" if key[2] else "nomem"
            labels.append(f"{key[0]}\n{key[1]}\n{mem_str}")
            llm_vals.append(vals["llm"])
            heur_vals.append(vals["heuristic"])

    if not labels:
        return

    import numpy as np

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2), 5))  # type: ignore[attr-defined]
    ax.bar(x - width / 2, heur_vals, width, label="Heuristic", color="#3498db", alpha=0.8)
    ax.bar(x + width / 2, llm_vals, width, label="LLM", color="#9b59b6", alpha=0.8)
    ax.set_ylabel("Mean steps")
    ax.set_title("Heuristic vs LLM — Mean Steps to Completion")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.legend()
    fig.tight_layout()
    p = pdir / "planner_comparison.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)  # type: ignore[attr-defined]
    paths.append(p)


def _plot_success_rates(groups: list[dict], pdir: Path, paths: list[Path], plt: object) -> None:
    """Horizontal bar chart of success rates across all groups."""
    if not groups:
        return

    labels = []
    rates = []
    for g in sorted(groups, key=lambda g: g["success_rate"], reverse=True):
        mem_str = "mem" if g["memory"] else "nomem"
        labels.append(f"{g['scenario']} / {g['target']} / {g['planner']} / {mem_str}")
        rates.append(g["success_rate"] * 100)

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.4)))  # type: ignore[attr-defined]
    colors = ["#2ecc71" if r >= 80 else "#e67e22" if r >= 50 else "#e74c3c" for r in rates]
    ax.barh(range(len(labels)), rates, color=colors, alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Success rate (%)")
    ax.set_title("Success Rate by Configuration")
    ax.set_xlim(0, 105)
    ax.invert_yaxis()
    fig.tight_layout()
    p = pdir / "success_rates.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)  # type: ignore[attr-defined]
    paths.append(p)


def generate_report(
    results_path: str | Path,
    output_path: str | Path = "docs/eval_results.md",
    plots_dir: str | Path = "docs/plots",
) -> Path:
    """Generate the full evaluation report.

    Parameters
    ----------
    results_path:
        Path to the results JSONL file produced by ``runner.run_matrix``.
    output_path:
        Where to write the Markdown report.
    plots_dir:
        Directory for PNG plot files.

    Returns
    -------
    Path to the written report.
    """
    results = load_results(results_path)
    if not results:
        logger.warning("No results found at %s", results_path)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("# Evaluation Results\n\nNo results yet. Run `make eval` first.\n")
        return out

    groups = compute_group_metrics(results)
    table = summary_table(groups)

    # Generate plots
    plot_paths = generate_plots(results, plots_dir)

    # Total stats
    total_runs = len(results)
    total_cost = sum(r.get("est_cost_usd", 0.0) for r in results)
    total_safety = sum(r.get("safety_violations", 0) for r in results)
    total_tokens = sum(r.get("prompt_tokens", 0) + r.get("completion_tokens", 0) for r in results)

    # Assemble report
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    sections = [
        "# Evaluation Results\n",
        f"**Total runs:** {total_runs}  \n"
        f"**Total tokens:** {total_tokens:,}  \n"
        f"**Total cost:** ${total_cost:.4f}  \n"
        f"**Safety violations:** {total_safety}\n",
        "## Summary Table\n",
        table,
    ]

    # Plot references
    if plot_paths:
        sections.append("\n## Plots\n")
        for p in plot_paths:
            rel = (
                p.relative_to(Path(output_path).parent)
                if p.is_relative_to(Path(output_path).parent)
                else p
            )
            sections.append(f"### {p.stem.replace('_', ' ').title()}\n\n![{p.stem}]({rel})\n")

    # Per-run detail table
    sections.append("\n## Per-Run Details\n")
    detail_header = (
        "| Scenario | Target | Planner | Memory | Seed | "
        "Outcome | Steps | Path (m) | LLM calls | Cost ($) | Safety |\n"
    )
    detail_sep = "|" + "|".join(["---"] * 11) + "|\n"
    detail_rows = ""
    for r in sorted(
        results, key=lambda r: (r["scenario"], r["target"], r["planner"], r["memory"], r["seed"])
    ):
        mem_str = "on" if r["memory"] else "off"
        detail_rows += (
            f"| {r['scenario']} | {r['target']} | {r['planner']} | {mem_str} "
            f"| {r['seed']} | {r['outcome']} | {r['steps']} "
            f"| {r.get('path_length_m', 0):.2f} | {r.get('llm_calls', 0)} "
            f"| {r.get('est_cost_usd', 0):.4f} | {r.get('safety_violations', 0)} |\n"
        )
    sections.append(detail_header + detail_sep + detail_rows)

    out.write_text("\n".join(sections))
    logger.info("Report written → %s", out)
    return out
