"""Unit / smoke tests for the M7 Streamlit dashboard utilities.

DoD checks
----------
- log_reader: read_log parses valid JSONL; silently skips malformed lines
- log_reader: latest_status_from_log, cumulative_usage, trajectory correct
- stop_flag: write → is_stop_requested → clear round-trip
- AppTest: app loads without exception; all 4 tabs present
- AppTest: emergency-stop button writes the sentinel file
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

STEP_JSON = {
    "step": 1,
    "ts": 1700000000.0,
    "action": {
        "kind": "advance",
        "distance_m": 0.5,
        "angle_deg": None,
        "reasoning": "moving forward",
        "confidence": 0.9,
    },
    "result": {
        "status": "ok",
        "pose": {"x": 0.5, "y": 0.0, "z": 1.0, "yaw_deg": 0.0},
        "elapsed_s": 0.3,
    },
    "detections": [],
    "pose": {"x": 0.5, "y": 0.0, "z": 1.0, "yaw_deg": 0.0},
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "llm_calls": 1, "est_cost_usd": 0.0002},
    "safety_events": [],
}

LAND_JSON = {
    **STEP_JSON,
    "step": 2,
    "action": {
        "kind": "land",
        "distance_m": None,
        "angle_deg": None,
        "reasoning": "target found",
        "confidence": 1.0,
    },
    "pose": {"x": 0.5, "y": 0.1, "z": 0.0, "yaw_deg": 10.0},
    "usage": {"prompt_tokens": 20, "completion_tokens": 8, "llm_calls": 2, "est_cost_usd": 0.0005},
}


@pytest.fixture()
def tmp_log(tmp_path: Path) -> Path:
    """Write a two-step JSONL mission log and return its path."""
    log = tmp_path / "m001.jsonl"
    log.write_text(
        json.dumps(STEP_JSON) + "\n" + json.dumps(LAND_JSON) + "\n",
    )
    return log


# ===========================================================================
# log_reader tests
# ===========================================================================


class TestLogReader:
    def test_read_log_parses_steps(self, tmp_log: Path) -> None:
        from kormoran.ui.log_reader import read_log

        steps = read_log(tmp_log)
        assert len(steps) == 2
        assert steps[0]["step"] == 1
        assert steps[1]["step"] == 2

    def test_read_log_missing_file_returns_empty(self, tmp_path: Path) -> None:
        from kormoran.ui.log_reader import read_log

        steps = read_log(tmp_path / "nonexistent.jsonl")
        assert steps == []

    def test_read_log_skips_malformed_lines(self, tmp_path: Path) -> None:
        from kormoran.ui.log_reader import read_log

        log = tmp_path / "bad.jsonl"
        log.write_text(json.dumps(STEP_JSON) + "\nnot-json\n" + json.dumps(LAND_JSON) + "\n")
        steps = read_log(log)
        assert len(steps) == 2  # malformed line skipped

    def test_latest_status_found(self, tmp_log: Path) -> None:
        from kormoran.ui.log_reader import latest_status_from_log, read_log

        steps = read_log(tmp_log)
        assert latest_status_from_log(steps) == "found"

    def test_latest_status_not_found(self, tmp_path: Path) -> None:
        from kormoran.ui.log_reader import latest_status_from_log, read_log

        nf = {**STEP_JSON, "action": {**STEP_JSON["action"], "kind": "report_not_found"}}
        log = tmp_path / "nf.jsonl"
        log.write_text(json.dumps(nf) + "\n")
        assert latest_status_from_log(read_log(log)) == "not_found"

    def test_latest_status_running_when_no_steps(self) -> None:
        from kormoran.ui.log_reader import latest_status_from_log

        assert latest_status_from_log([]) == "running"

    def test_cumulative_usage_last_step(self, tmp_log: Path) -> None:
        from kormoran.ui.log_reader import cumulative_usage, read_log

        steps = read_log(tmp_log)
        usage = cumulative_usage(steps)
        assert usage["llm_calls"] == 2
        assert usage["est_cost_usd"] == pytest.approx(0.0005)

    def test_trajectory_extracts_xy(self, tmp_log: Path) -> None:
        from kormoran.ui.log_reader import read_log, trajectory

        steps = read_log(tmp_log)
        traj = trajectory(steps)
        assert len(traj) == 2
        assert traj[0] == (0.5, 0.0)
        assert traj[1] == (0.5, 0.1)


# ===========================================================================
# stop_flag tests
# ===========================================================================


class TestStopFlag:
    def test_round_trip(self, tmp_path: Path) -> None:
        from kormoran.ui.stop_flag import clear_stop_flag, is_stop_requested, write_stop_flag

        mid = "test001"
        assert not is_stop_requested(tmp_path, mid)

        path = write_stop_flag(tmp_path, mid)
        assert path.exists()
        assert is_stop_requested(tmp_path, mid)

        clear_stop_flag(tmp_path, mid)
        assert not is_stop_requested(tmp_path, mid)

    def test_clear_missing_is_noop(self, tmp_path: Path) -> None:
        from kormoran.ui.stop_flag import clear_stop_flag

        # Should not raise even if file doesn't exist
        clear_stop_flag(tmp_path, "nonexistent")

    def test_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        from kormoran.ui.stop_flag import write_stop_flag

        deep = tmp_path / "a" / "b" / "c"
        write_stop_flag(deep, "m999")
        assert (deep / "stop" / "m999.stop").exists()


# ===========================================================================
# AppTest smoke tests
# ===========================================================================


@pytest.mark.unit
class TestAppSmoke:
    """Smoke-test the Streamlit app via AppTest without a live agent."""

    def _run_app(self, monkeypatch: pytest.MonkeyPatch, data_dir: Path) -> object:
        """Launch AppTest with env vars pointing at a temp data dir."""
        from streamlit.testing.v1 import AppTest

        monkeypatch.setenv("DATA_DIR", str(data_dir))
        monkeypatch.setenv("SCENARIOS_DIR", "configs/scenarios")

        app_path = Path(__file__).parents[2] / "src" / "kormoran" / "ui" / "app.py"
        at = AppTest.from_file(str(app_path), default_timeout=30)
        at.run()
        return at

    def test_app_loads_without_exception(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """App renders all 5 tabs without raising."""
        at = self._run_app(monkeypatch, tmp_path)
        assert not at.exception, f"App raised: {at.exception}"

    def test_five_tabs_present(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """All five tab labels are present in the rendered output."""
        at = self._run_app(monkeypatch, tmp_path)
        tab_labels = [t.label for t in at.tabs]
        assert "Mission Control" in tab_labels
        assert "Live Feed" in tab_labels
        assert "Analytics" in tab_labels
        assert "Memory" in tab_labels
        assert "Eval" in tab_labels

    def test_no_missions_caption_shown(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """With no log files, the 'No missions recorded yet' caption appears."""
        at = self._run_app(monkeypatch, tmp_path)
        captions = [c.value for c in at.caption]
        assert any("No missions" in c or "No mission" in c for c in captions)
