# Architecture Decision Log

Decisions are recorded here when the plan in ROADMAP.md changes or when a VERIFY item is confirmed.
Format: `YYYY-MM-DD — <title>` followed by context, decision, and consequences.

---

## 2026-10-05 — M0 bootstrap

**Context:** Initial project scaffold.

**Decisions made:**
- Build backend: `hatchling` (default for `uv init`).
- `openrouter_api_key` is `SecretStr | None` (not required at import time) so that unit tests and CI run without a live key; `check-llm` validates its presence at runtime.
- `cli.py` uses `BLE001` noqa on the broad `except Exception` in `check-llm` because the command's only job is to surface *any* connection failure gracefully — a narrow except would silently swallow unexpected SDK errors.
- ruff `ANN` rule is enabled for `src/` but suppressed in `tests/` to keep test code readable.

**Open VERIFY items (confirm in M1):**
- gym-pybullet-drones installs cleanly under `uv` + Python 3.12.
- `get_frame()` returns non-black frames in PyBullet `DIRECT` mode.
- `with_structured_output(Action)` works for the chosen OpenRouter model; fallback to `method="function_calling"` if not.
