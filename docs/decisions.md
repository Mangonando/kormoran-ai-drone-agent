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

---

## 2026-10-05 — M1: pybullet 3.2.7 macOS arm64 + Xcode 21 build failure

**Context:** pybullet 3.2.7 fails to build from source on macOS arm64 with Apple clang 21.x.
The error is in the vendored `examples/ThirdPartyLibs/zlib/zutil.h`: when `TARGET_OS_MAC` is
defined, the file does `#define fdopen(fd, mode) NULL`.  Newer macOS SDK `stdio.h` declares
`fdopen` with `__DARWIN_ALIAS_STARTING(…, __DARWIN_ALIAS(fdopen))`, which expands the `fdopen`
macro to `NULL` and produces a parse error.

**Decision:** Patch the one line in `zutil.h` (replace the `#define fdopen` with a comment),
build a local wheel (`pybullet-3.2.7-cp312-cp312-macosx_11_0_arm64.whl`), store it in
`wheels/`, and add `[tool.uv] find-links = ["wheels"]` to `pyproject.toml`.  On Linux (CI,
Docker) the macOS wheel is incompatible and uv falls back to the PyPI sdist, which builds
cleanly with gcc.

**Consequences:** The `wheels/` directory is committed; the patch is not upstreamed (would
need a pybullet fork PR).  If pybullet releases 3.2.8+ with the fix, remove the local wheel
and the find-links entry.

---

## 2026-10-05 — M1: VERIFY items resolved

**VERIFY 1 — gym-pybullet-drones under uv + Python 3.12:**
✅ CONFIRMED after the pybullet wheel fix above.  All 7 sim integration tests pass in 1.3 s.

**VERIFY 2 — CtrlAviary vision_attributes:**
`CtrlAviary.__init__` does not expose `vision_attributes`.  The parameter lives in
`BaseAviary.__init__` but is not forwarded by `CtrlAviary`.  Workaround: post-hoc set
`env.IMG_RES`, `env.rgb`, `env.dep`, `env.seg` after `reset()`.  This is implemented in
`drones/sim.py:connect()`.

**VERIFY 3 — Frames non-black in DIRECT mode:**
✅ CONFIRMED.  `frame.image.var() > 0` passes in CI-style headless mode.  The renderer
renders the PyBullet plane + coloured objects without a display server.

**VERIFY 4 — `with_structured_output(Action)` (M4):**
Still open — deferred to M4 when the agent calls the LLM for the first time.

---

## 2026-10-06 — M3: Perception layer

**Context:** HSV-based colour detector for coloured primitives (red/blue/green/yellow × cube/sphere/cylinder).

**Decisions made:**
- `ColorDetector` is a class implementing the `Perceiver` protocol via `detect(frame, target)`.  HSV
  ranges are hardcoded constants in `color_detector.py`; changing them is a breaking change (update
  decisions.md and snapshot tests).  Range floors: S ≥ 90, V ≥ 40 to handle PyBullet Phong shading.
- Confidence metric = blob solidity × size_factor.  Solidity (contour_area / convex_hull_area) rewards
  compact blobs; size_factor penalises blobs covering < 0.2 % of the frame so distant/edge detections
  don't over-state certainty.
- `ground_truth.py` is pure (no backend imports).  It takes a raw numpy seg mask + label→body_id map;
  callers (CLI/eval) supply those from `SimDrone.get_seg_frame()`.  Seg mask decoding uses both exact
  match and `& 0xFFFF` masking to handle PyBullet's link-index encoding for static single-body objects.
- `summarizer.py` uses deterministic thresholds (small ≤ 3 %, large ≥ 12 %, left/right > ±0.25 cx_norm,
  high/low > ±0.25 cy_norm).  These are snapshot-tested; any change to the output format requires
  updating `tests/unit/test_summarizer.py`.
- `SimDrone.get_seg_frame()` returns `(Frame, seg_mask_int32, label_to_body_id)`.  The method is not
  part of the `DroneInterface` protocol (eval-only).
- `eval-detector` CLI command: 1-D grid of x positions × 8 yaws per position.  Default 5 × 8 = 40
  frames (fast; ~ 3 min in headless sim).  Scale with `--positions 25 --yaws 8` for ~200 frames.
  Results appended to `docs/eval_results.md`.
