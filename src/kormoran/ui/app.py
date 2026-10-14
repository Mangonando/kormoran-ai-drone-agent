"""Kormoran Streamlit dashboard (M7).

Launch with::

    uv run kormoran dashboard
    # or
    streamlit run src/kormoran/ui/app.py

Tabs
----
Mission Control  — form + launch + emergency stop + live status
Live Feed        — top-down trajectory map + reasoning stream + cost counter
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


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------


def _init_state() -> None:
    defaults: dict = {
        "proc": None,  # subprocess.Popen | None
        "mission_id": "",  # current mission id
        "log_lines": [],  # buffered stdout from subprocess
        "auto_refresh": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------

tab_mission, tab_live, tab_memory, tab_eval = st.tabs(
    ["Mission Control", "Live Feed", "Memory", "Eval"]
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
                ["yellow cylinder", "red cube", "blue sphere"],
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
        # Collect stdout lines
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

        # Show stdout tail
        lines = st.session_state.get("log_lines", [])
        if lines:
            st.text_area("Mission output", "\n".join(lines[-50:]), height=200)

    # ------------------------------------------------------------------
    # Auto-refresh
    # ------------------------------------------------------------------
    if st.session_state.get("auto_refresh"):
        time.sleep(1.5)
        st.rerun()

    # ------------------------------------------------------------------
    # Past missions list
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Past missions")
    missions_dir = _data_dir() / "missions"
    if missions_dir.exists():
        logs = sorted(missions_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if logs:
            for log in logs[:10]:
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
                st.text(
                    f"{log.stem}  steps={n}  status={status}  llm_calls={calls}  cost=${cost:.4f}"
                )
        else:
            st.caption("No missions recorded yet.")
    else:
        st.caption("No missions directory found.")


# ===========================================================================
# TAB 2 — Live Feed
# ===========================================================================

with tab_live:
    st.header("Live Feed")

    mid = st.session_state.get("mission_id", "")

    # Optionally pick a specific mission log to inspect
    missions_dir = _data_dir() / "missions"
    log_files = (
        sorted(missions_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if missions_dir.exists()
        else []
    )
    log_stems = [p.stem for p in log_files]

    if log_stems:
        default_idx = 0 if not mid else (log_stems.index(mid) if mid in log_stems else 0)
        selected_id = st.selectbox("Mission log", log_stems, index=default_idx)
        log_path = _mission_log_path(str(selected_id))
        from kormoran.ui.log_reader import (
            cumulative_usage,
            latest_status_from_log,
            read_log,
            trajectory,
        )

        steps = read_log(log_path)
    else:
        st.caption("No mission logs found. Launch a mission first.")
        steps = []
        selected_id = ""

    if steps:
        # ------------------------------------------------------------------
        # Metrics row
        # ------------------------------------------------------------------
        usage = cumulative_usage(steps)
        status = latest_status_from_log(steps)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Steps", len(steps))
        c2.metric("Status", status)
        c3.metric("LLM calls", usage.get("llm_calls", 0))
        c4.metric("Est. cost", f"${usage.get('est_cost_usd', 0.0):.4f}")

        # ------------------------------------------------------------------
        # Top-down trajectory map
        # ------------------------------------------------------------------
        st.subheader("Top-down trajectory")
        try:
            import matplotlib.pyplot as plt

            traj = trajectory(steps)
            if traj:
                xs, ys = zip(*traj, strict=True)
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.plot(xs, ys, "b-o", markersize=3, linewidth=1, label="path")
                ax.plot(xs[0], ys[0], "gs", markersize=8, label="start")
                # End marker colour depends on outcome
                end_color = "g" if status == "found" else "r"
                end_label = (
                    "FOUND"
                    if status == "found"
                    else ("NOT FOUND" if status == "not_found" else status.upper())
                )
                ax.plot(xs[-1], ys[-1], marker="*", color=end_color, markersize=14, label=end_label)
                # Colour clamped steps red
                for s in steps:
                    if (s.get("result") or {}).get("status") == "clamped":
                        px = s["pose"]["x"]
                        py = s["pose"]["y"]
                        ax.plot(px, py, "rx", markersize=6)
                ax.set_aspect("equal")
                ax.set_xlabel("x (m)")
                ax.set_ylabel("y (m)")
                ax.legend(loc="upper right", fontsize="small")
                ax.set_title(f"Mission {selected_id} — {end_label}")
                st.pyplot(fig)
                plt.close(fig)
        except ImportError:
            st.caption("matplotlib not available — install it for the trajectory map.")

        # ------------------------------------------------------------------
        # GIF playback
        # ------------------------------------------------------------------
        top_gif = _data_dir() / "missions" / f"{selected_id}_top.gif"
        drone_gif = _data_dir() / "missions" / f"{selected_id}.gif"
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

        # ------------------------------------------------------------------
        # Reasoning stream
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Safety events
        # ------------------------------------------------------------------
        all_events = []
        for s in steps:
            for ev in s.get("safety_events") or []:
                all_events.append({"step": s["step"], **ev})
        if all_events:
            st.subheader(f"Safety events ({len(all_events)})")
            st.dataframe(all_events, use_container_width=True)
        else:
            st.caption("No safety events.")


# ===========================================================================
# TAB 3 — Memory
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
# TAB 4 — Eval
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
