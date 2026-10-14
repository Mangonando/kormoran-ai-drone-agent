"""Kormoran Streamlit dashboard (M7/M8).

Launch with::

    uv run kormoran dashboard
    # or
    streamlit run src/kormoran/ui/app.py

Tabs
----
Mission Control  — form + launch + emergency stop + current mission detail
Live Feed        — inspect any past mission in detail
Analytics        — cross-mission comparison charts (LLM vs heuristic, optimization)
Memory           — episodic table + semantic notes + clear button
Eval             — renders docs/eval_results.md
"""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Kormoran — Drone Agent Dashboard",
    page_icon=":helicopter:",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _settings():  # noqa: ANN202
    """Cached Settings object (reads .env)."""
    from kormoran.config import get_settings

    return get_settings()


def _data_dir() -> Path:
    p = Path(_settings().data_dir)
    if not p.is_absolute():
        # Resolve relative to project root (3 levels up from this file)
        p = Path(__file__).resolve().parents[3] / p
    return p


def _scenarios_dir() -> Path:
    p = Path(_settings().scenarios_dir)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[3] / p
    return p


def _list_scenarios() -> list[str]:
    d = _scenarios_dir()
    if not d.exists():
        return ["living_room_v1"]
    return sorted(p.stem for p in d.glob("*.yaml"))


def _mission_log_path(mission_id: str) -> Path:
    return _data_dir() / "missions" / f"{mission_id}.jsonl"


def _list_mission_logs() -> list[Path]:
    """Return mission JSONL logs sorted newest-first.

    Filters out redundant eval combinations (heuristic+mem, llm+nomem)
    since heuristic ignores memory and LLM is meant to run with memory.
    """
    missions_dir = _data_dir() / "missions"
    if not missions_dir.exists():
        return []
    logs = sorted(missions_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)

    def _is_redundant(name: str) -> bool:
        if not name.startswith("eval_"):
            return False
        is_heur_mem = "__heuristic__" in name and name.endswith("__mem")
        is_llm_nomem = "__llm__" in name and name.endswith("__nomem")
        return is_heur_mem or is_llm_nomem

    return [p for p in logs if not _is_redundant(p.stem)]


# ---------------------------------------------------------------------------
# Reusable mission detail renderer
# ---------------------------------------------------------------------------


def _render_mission_detail(mission_id: str, steps: list[dict]) -> None:
    """Render full mission detail: metrics, trajectory, GIFs, reasoning, safety."""
    from kormoran.ui.log_reader import cumulative_usage, latest_status_from_log, trajectory

    if not steps:
        st.caption("No step data available.")
        return

    # Metrics row
    usage = cumulative_usage(steps)
    status = latest_status_from_log(steps)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Steps", len(steps))
    c2.metric("Status", status)
    c3.metric("LLM calls", usage.get("llm_calls", 0))
    c4.metric("Est. cost", f"${usage.get('est_cost_usd', 0.0):.4f}")

    # Top-down trajectory map
    try:
        import matplotlib.pyplot as plt

        traj = trajectory(steps)
        if traj:
            xs, ys = zip(*traj, strict=True)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(xs, ys, "b-o", markersize=3, linewidth=1, label="path")
            ax.plot(xs[0], ys[0], "gs", markersize=8, label="start")
            end_color = "g" if status == "found" else "r"
            end_label = (
                "FOUND"
                if status == "found"
                else ("NOT FOUND" if status == "not_found" else status.upper())
            )
            ax.plot(xs[-1], ys[-1], marker="*", color=end_color, markersize=14, label=end_label)
            for s in steps:
                if (s.get("result") or {}).get("status") == "clamped":
                    px = s["pose"]["x"]
                    py = s["pose"]["y"]
                    ax.plot(px, py, "rx", markersize=6)
            ax.set_aspect("equal")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.legend(loc="upper right", fontsize="small")
            ax.set_title(f"Mission {mission_id} — {end_label}")
            st.pyplot(fig)
            plt.close(fig)
    except ImportError:
        st.caption("matplotlib not available — install it for the trajectory map.")

    # GIF playback
    top_gif = _data_dir() / "missions" / f"{mission_id}_top.gif"
    drone_gif = _data_dir() / "missions" / f"{mission_id}.gif"
    if top_gif.exists() or drone_gif.exists():
        st.subheader("Mission replay (GIF)")
        gif_col1, gif_col2 = st.columns(2)
        with gif_col1:
            if top_gif.exists():
                with top_gif.open("rb") as f:
                    st.image(f.read(), caption="Top-down view")
        with gif_col2:
            if drone_gif.exists():
                with drone_gif.open("rb") as f:
                    st.image(f.read(), caption="Drone camera view")

    # Reasoning stream
    st.subheader("Agent reasoning stream (last 20 steps)")
    rows = []
    for s in steps[-20:]:
        action = s.get("action") or {}
        result = s.get("result") or {}
        safety = s.get("safety_events") or []
        rows.append(
            {
                "step": s.get("step"),
                "action": action.get("kind", ""),
                "reasoning": (action.get("reasoning") or "")[:80],
                "status": result.get("status", ""),
                "detections": len(s.get("detections") or []),
                "safety": len(safety),
            }
        )
    st.dataframe(rows, use_container_width=True)

    # Safety events
    all_events = []
    for s in steps:
        for ev in s.get("safety_events") or []:
            all_events.append({"step": s["step"], **ev})
    if all_events:
        st.subheader(f"Safety events ({len(all_events)})")
        st.dataframe(all_events, use_container_width=True)
    else:
        st.caption("No safety events.")


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------


def _init_state() -> None:
    defaults: dict = {
        "proc": None,  # subprocess.Popen | None
        "mission_id": "",  # current mission id
        "log_lines": [],  # buffered stdout from subprocess
        "auto_refresh": False,
        "selected_mission": "",  # mission selected for Live Feed
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------

tab_mission, tab_live, tab_analytics, tab_memory, tab_eval = st.tabs(
    ["Mission Control", "Live Feed", "Analytics", "Memory", "Eval"]
)

# ===========================================================================
# TAB 1 — Mission Control
# ===========================================================================

with tab_mission:
    st.header("Mission Control")

    # ------------------------------------------------------------------
    # Mission form
    # ------------------------------------------------------------------
    with st.form("mission_form"):
        col1, col2 = st.columns(2)
        with col1:
            target = st.selectbox(
                "Target label",
                ["blue sphere", "yellow cylinder", "red cube"],
                index=0,
            )
            scenarios = _list_scenarios()
            scenario = st.selectbox(
                "Scenario",
                scenarios,
                index=scenarios.index("apartment_v1") if "apartment_v1" in scenarios else 0,
            )
            planner = st.selectbox("Planner", ["llm", "heuristic"], index=0)
        with col2:
            seed = st.text_input("Seed", value="2")
            max_steps = st.text_input("Max steps", value="50")
            memory = st.checkbox("Enable long-term memory", value=True)

        launch = st.form_submit_button("Launch Mission", type="primary")

    if launch:
        from kormoran.agent.graph import new_mission_id
        from kormoran.ui.launcher import launch_mission

        mid = new_mission_id()
        st.session_state["mission_id"] = mid
        st.session_state["log_lines"] = []
        proc = launch_mission(
            target=target,
            scenario=scenario,
            planner=planner,
            seed=int(seed),  # type: ignore[arg-type]
            max_steps=int(max_steps),  # type: ignore[arg-type]
            memory=memory,
            data_dir=_data_dir(),
            mission_id=mid,
        )
        st.session_state["proc"] = proc
        st.session_state["auto_refresh"] = True
        st.success(f"Mission {mid} launched (pid={proc.pid})")

    # ------------------------------------------------------------------
    # Running state + emergency stop
    # ------------------------------------------------------------------
    proc = st.session_state.get("proc")
    mid = st.session_state.get("mission_id", "")

    if proc is not None:
        from kormoran.ui.launcher import poll_output

        new_lines = poll_output(proc)
        st.session_state["log_lines"].extend(new_lines)

        exit_code = proc.poll()
        if exit_code is None:
            st.info(f"Running — mission {mid} (pid={proc.pid})")
            if st.button("Emergency Stop", type="primary"):
                from kormoran.ui.stop_flag import write_stop_flag

                write_stop_flag(_data_dir(), mid)
                st.warning(f"Stop flag written for mission {mid}.")
        else:
            status_label = "Success (exit 0)" if exit_code == 0 else f"Finished (exit {exit_code})"
            st.info(f"Mission {mid} — {status_label}")
            st.session_state["auto_refresh"] = False

    # ------------------------------------------------------------------
    # Current mission detail (or last mission if idle)
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Current Mission")

    display_mid = mid
    if not display_mid:
        # Show last mission if no current mission
        logs = _list_mission_logs()
        if logs:
            display_mid = logs[0].stem

    if display_mid:
        from kormoran.ui.log_reader import read_log

        current_steps = read_log(_mission_log_path(display_mid))
        if current_steps:
            _render_mission_detail(display_mid, current_steps)
        else:
            st.caption(f"Mission {display_mid} — waiting for data...")
    else:
        st.caption("No missions yet. Launch one above.")

    # ------------------------------------------------------------------
    # Auto-refresh
    # ------------------------------------------------------------------
    if st.session_state.get("auto_refresh"):
        time.sleep(1.5)
        st.rerun()

    # ------------------------------------------------------------------
    # Past missions list (clickable → opens in Live Feed)
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Past Missions")
    logs = _list_mission_logs()
    if logs:
        for log in logs[:15]:
            from kormoran.ui.log_reader import (
                cumulative_usage,
                latest_status_from_log,
                read_log,
            )

            steps = read_log(log)
            status = latest_status_from_log(steps)
            usage = cumulative_usage(steps)
            n = len(steps)
            cost = usage.get("est_cost_usd", 0.0)
            calls = usage.get("llm_calls", 0)

            col_info, col_btn = st.columns([5, 1])
            with col_info:
                status_icon = {"found": "🟢", "not_found": "🔴", "running": "🟡"}.get(status, "⚪")
                st.text(
                    f"{status_icon}  {log.stem}  steps={n}  "
                    f"status={status}  llm={calls}  cost=${cost:.4f}"
                )
            with col_btn:
                if st.button("View", key=f"view_{log.stem}"):
                    st.session_state["selected_mission"] = log.stem
    else:
        st.caption("No missions recorded yet.")


# ===========================================================================
# TAB 2 — Live Feed (mission detail viewer)
# ===========================================================================

with tab_live:
    st.header("Live Feed")

    logs = _list_mission_logs()

    if logs:
        from kormoran.ui.log_reader import latest_status_from_log, read_log

        # Build display labels with status
        _status_icons = {
            "found": "FOUND",
            "not_found": "NOT FOUND",
            "error": "ERROR",
            "running": "RUNNING",
        }
        log_labels = []
        log_stems = []
        for p in logs:
            s = read_log(p)
            st_label = _status_icons.get(latest_status_from_log(s), "?")
            log_labels.append(f"{p.stem}  [{st_label}]")
            log_stems.append(p.stem)

        preselect = st.session_state.get("selected_mission", "")
        default_idx = log_stems.index(preselect) if preselect in log_stems else 0
        selected_label = st.selectbox("Mission log", log_labels, index=default_idx)
        selected_id = log_stems[log_labels.index(selected_label)]

        steps = read_log(_mission_log_path(str(selected_id)))
        if steps:
            _render_mission_detail(str(selected_id), steps)
        else:
            st.caption("No step data in this log.")
    else:
        st.caption("No mission logs found. Launch a mission first.")


# ===========================================================================
# TAB 3 — Analytics (cross-mission comparison)
# ===========================================================================

with tab_analytics:
    st.header("Mission Analytics")

    from kormoran.ui.log_reader import cumulative_usage, latest_status_from_log, read_log

    logs = _list_mission_logs()
    if not logs:
        st.caption("No mission data. Run some missions first.")
    else:
        # Parse all mission logs into a flat table
        all_missions: list[dict] = []
        for log in logs:
            steps = read_log(log)
            if not steps:
                continue
            usage = cumulative_usage(steps)
            status = latest_status_from_log(steps)
            # Extract metadata from the log steps
            first = steps[0] if steps else {}
            mission_id = log.stem
            # Try to infer scenario/target/planner from mission_id or log content
            action_kinds = [(s.get("action") or {}).get("kind", "") for s in steps]
            has_llm = usage.get("llm_calls", 0) > 0

            all_missions.append(
                {
                    "mission_id": mission_id,
                    "steps": len(steps),
                    "status": status,
                    "llm_calls": usage.get("llm_calls", 0),
                    "cost": usage.get("est_cost_usd", 0.0),
                    "path_length": sum(
                        (
                            (steps[i]["pose"]["x"] - steps[i - 1]["pose"]["x"]) ** 2
                            + (steps[i]["pose"]["y"] - steps[i - 1]["pose"]["y"]) ** 2
                        )
                        ** 0.5
                        for i in range(1, len(steps))
                        if "pose" in steps[i] and "pose" in steps[i - 1]
                    ),
                    "planner": "llm" if has_llm else "heuristic",
                }
            )

        if not all_missions:
            st.caption("No valid mission data found.")
        else:
            import pandas as pd

            df = pd.DataFrame(all_missions)

            # ------------------------------------------------------------------
            # Overview metrics
            # ------------------------------------------------------------------
            st.subheader("Overview")
            ov1, ov2, ov3, ov4 = st.columns(4)
            ov1.metric("Total missions", len(df))
            ov2.metric("Found", int((df["status"] == "found").sum()))
            ov3.metric("Not found", int((df["status"] == "not_found").sum()))
            ov4.metric("Total cost", f"${df['cost'].sum():.4f}")

            # ------------------------------------------------------------------
            # Eval results analysis (from data/eval/results.jsonl)
            # ------------------------------------------------------------------
            eval_path = _data_dir() / "eval" / "results.jsonl"
            eval_records: list[dict] = []
            if eval_path.exists():
                import json

                with eval_path.open() as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            try:
                                eval_records.append(json.loads(line))
                            except Exception:  # noqa: BLE001, S110
                                pass

            if eval_records:
                eval_df = pd.DataFrame(eval_records)
                st.divider()
                st.subheader("Evaluation Results Breakdown")

                # ----------------------------------------------------------
                # Success rate: LLM vs Heuristic (per target)
                # ----------------------------------------------------------
                st.subheader("Success Rate: LLM vs Heuristic")

                try:
                    import matplotlib
                    import numpy as np

                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    # Filter to meaningful comparisons only:
                    # - Heuristic (memory doesn't matter, use nomem to avoid double-counting)
                    # - LLM with memory (the intended use case)
                    _TARGET_ORDER = ["blue sphere", "yellow cylinder", "red cube"]
                    meaningful = eval_df[
                        ((eval_df["planner"] == "heuristic") & (eval_df["memory"] == False))  # noqa: E712
                        | ((eval_df["planner"] == "llm") & (eval_df["memory"] == True))  # noqa: E712
                    ].copy()

                    target_df = meaningful[meaningful["scenario"] == "apartment_v1"].copy()
                    if not target_df.empty:
                        # Classify outcomes
                        target_df["status"] = target_df["outcome"].map(
                            lambda o: o if o in ("found", "not_found") else "error"
                        )
                        ordered = [t for t in _TARGET_ORDER if t in target_df["target"].unique()]
                        width = 0.35

                        def _grouped_bar(
                            data: object,
                            status: str,
                            title: str,
                            color_h: str,
                            color_l: str,
                            ylabel: str = "Runs",
                            is_pct: bool = False,
                        ) -> None:
                            vals: dict[str, list] = {"heuristic": [], "llm": []}
                            for t in ordered:
                                for p in ["heuristic", "llm"]:
                                    sub = data[(data["target"] == t) & (data["planner"] == p)]
                                    if is_pct:
                                        completed = sub[sub["status"].isin(["found", "not_found"])]
                                        if len(completed):
                                            v = (completed["status"] == "found").mean() * 100
                                        else:
                                            v = 0
                                    else:
                                        v = (sub["status"] == status).sum()
                                    vals[p].append(v)

                            x = np.arange(len(ordered))
                            fig, ax = plt.subplots(figsize=(7, 4))
                            ax.bar(
                                x - width / 2,
                                vals["heuristic"],
                                width,
                                label="Heuristic",
                                color=color_h,
                                alpha=0.85,
                            )
                            ax.bar(
                                x + width / 2,
                                vals["llm"],
                                width,
                                label="LLM",
                                color=color_l,
                                alpha=0.85,
                            )
                            ax.set_xticks(x)
                            ax.set_xticklabels(ordered, fontsize=9)
                            ax.set_ylabel(ylabel)
                            ax.legend()
                            if is_pct:
                                ax.set_ylim(0, 110)
                            ax.set_title(title)
                            fig.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)

                        # 4 charts in a 2×2 grid
                        _H = "#3498db"  # blue = heuristic
                        _L = "#9b59b6"  # purple = LLM
                        row1a, row1b = st.columns(2)
                        with row1a:
                            _grouped_bar(
                                target_df,
                                "found",
                                "Found",
                                _H,
                                _L,
                            )
                        with row1b:
                            _grouped_bar(
                                target_df,
                                "not_found",
                                "Not Found",
                                _H,
                                _L,
                            )
                        row2a, row2b = st.columns(2)
                        with row2a:
                            _grouped_bar(
                                target_df,
                                "error",
                                "Errors (timeout)",
                                _H,
                                _L,
                            )
                        with row2b:
                            _grouped_bar(
                                target_df,
                                "",
                                "Success Rate (excl. errors)",
                                _H,
                                _L,
                                ylabel="Success %",
                                is_pct=True,
                            )

                    # ----------------------------------------------------------
                    # Mean steps: LLM vs Heuristic (per target)
                    # ----------------------------------------------------------
                    st.subheader("Avg Steps to Find Target")
                    found_df = target_df[target_df["outcome"] == "found"].copy()
                    if not found_df.empty:
                        steps_grouped = (
                            found_df.groupby(["target", "planner"])["steps"]
                            .mean()
                            .unstack(fill_value=0)
                        )
                        ordered = [t for t in _TARGET_ORDER if t in steps_grouped.index]
                        steps_grouped = steps_grouped.reindex(ordered)
                        fig, ax = plt.subplots(figsize=(8, 4))
                        x = np.arange(len(steps_grouped.index))
                        if "heuristic" in steps_grouped.columns:
                            ax.bar(
                                x - width / 2,
                                steps_grouped["heuristic"],
                                width,
                                label="Heuristic",
                                color="#3498db",
                                alpha=0.85,
                            )
                        if "llm" in steps_grouped.columns:
                            ax.bar(
                                x + width / 2,
                                steps_grouped["llm"],
                                width,
                                label="LLM",
                                color="#9b59b6",
                                alpha=0.85,
                            )
                        ax.set_ylabel("Avg steps")
                        ax.set_xticks(x)
                        ax.set_xticklabels(steps_grouped.index, fontsize=9)
                        ax.legend()
                        ax.set_title("Avg Steps to Find Target (found missions only)")
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.caption("No found missions to compare steps.")

                    # ----------------------------------------------------------
                    # Memory ON vs OFF comparison
                    # ----------------------------------------------------------
                    # Absent target: false-found rate
                    # ----------------------------------------------------------
                    absent_df = meaningful[meaningful["scenario"] == "absent_target"].copy()
                    if not absent_df.empty:
                        st.subheader("Absent Target — Honesty Check")
                        total_absent = len(absent_df)
                        false_found = (absent_df["outcome"] == "found").sum()
                        correct = (absent_df["outcome"] == "not_found").sum()
                        h1, h2, h3 = st.columns(3)
                        h1.metric("Total runs", total_absent)
                        h2.metric("Correct (not_found)", int(correct))
                        h3.metric("False claims (found)", int(false_found))

                except ImportError:
                    st.caption("matplotlib not available — install it for charts.")

                # ----------------------------------------------------------
                # Raw eval data table
                # ----------------------------------------------------------
                st.subheader("Raw Evaluation Data")
                display_cols = [
                    c
                    for c in [
                        "scenario",
                        "target",
                        "planner",
                        "seed",
                        "outcome",
                        "steps",
                        "path_length_m",
                        "llm_calls",
                        "est_cost_usd",
                        "safety_violations",
                    ]
                    if c in meaningful.columns
                ]
                st.dataframe(
                    meaningful[display_cols].sort_values(["scenario", "target", "planner", "seed"]),
                    use_container_width=True,
                )
            else:
                st.caption(
                    "No eval results found. Run `make eval` or "
                    "`uv run kormoran eval --seeds 2 --scenarios apartment_v1` to generate data."
                )


# ===========================================================================
# TAB 4 — Memory
# ===========================================================================

with tab_memory:
    st.header("Memory Browser")

    col_ep, col_sem = st.columns(2)

    # ------------------------------------------------------------------
    # Episodic store
    # ------------------------------------------------------------------
    with col_ep:
        st.subheader("Episodic (SQLite)")
        try:
            from kormoran.memory.episodic import EpisodicStore

            ep_store = EpisodicStore(_data_dir() / "memory" / "episodic.db")
            records = ep_store.list_missions(limit=50)
            if records:
                rows = [
                    {
                        "id": r.mission_id,
                        "ts": r.ts[:19],
                        "scene": r.scene_id,
                        "target": r.target_label,
                        "outcome": r.outcome,
                        "steps": r.steps,
                        "cost": f"${r.cost_usd:.4f}",
                    }
                    for r in records
                ]
                st.dataframe(rows, use_container_width=True)
                st.caption(f"{len(records)} mission(s) recorded.")
            else:
                st.caption("No missions in episodic memory.")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not read episodic store: {exc}")

    # ------------------------------------------------------------------
    # Semantic store
    # ------------------------------------------------------------------
    with col_sem:
        st.subheader("Semantic (ChromaDB)")
        try:
            from kormoran.memory.semantic import SemanticStore

            sem_store = SemanticStore(_data_dir() / "memory" / "semantic")
            notes = sem_store.list_notes(limit=50)
            if notes:
                rows = [
                    {
                        "kind": n.get("kind"),
                        "conf": f"{n.get('confidence', 1.0):.2f}",
                        "scene": n.get("scene_id"),
                        "target": n.get("target_label"),
                        "text": (n.get("text") or "")[:80],
                    }
                    for n in notes
                ]
                st.dataframe(rows, use_container_width=True)
                st.caption(f"{len(notes)} note(s) stored.")
            else:
                st.caption("No semantic notes stored.")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not read semantic store: {exc}")

    # ------------------------------------------------------------------
    # Clear memory
    # ------------------------------------------------------------------
    st.divider()
    if st.button("Clear all memory", type="secondary"):
        try:
            from kormoran.memory.episodic import EpisodicStore as _EpStore
            from kormoran.memory.semantic import SemanticStore as _SemStore

            _EpStore(_data_dir() / "memory" / "episodic.db").clear()
            _SemStore(_data_dir() / "memory" / "semantic").clear()
            st.success("Memory cleared.")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Clear failed: {exc}")


# ===========================================================================
# TAB 5 — Eval
# ===========================================================================

with tab_eval:
    st.header("Evaluation Results")
    eval_path = Path("docs/eval_results.md")
    if eval_path.exists():
        st.markdown(eval_path.read_text())
    else:
        st.caption(
            "No eval_results.md found. Run `make eval` or "
            "`uv run kormoran eval-detector` to generate results."
        )
