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

---

## 2026-10-06 — M4: Agent core (LangGraph + short-term memory)

**Context:** Implements the full mission loop: recall_memory → perceive → plan → validate → act →
check → reflect_and_store.

**Decisions made:**
- `MemorySaver` (in-process) used for M4; `SqliteSaver` deferred to M5 when the `memory/` package
  is built.  `langgraph>=0.2` (installed as 1.2.13) added to dependencies.
- All node functions are returned by factory functions (`make_*_node`) so backends are injected via
  closure — no global state, trivially testable with `MockDrone` + `MockPerceiver`.
- `MissionState` uses `Annotated[list[T], operator.add]` for `recent_steps`, `visited`, and
  `safety_events` (accumulate across steps); plain `list[dict]` for `last_detections` (replaced
  each step by perceive / act).
- Found criterion: `area_frac >= 0.01` AND `|center_norm[0]| <= 0.6` for 2 consecutive frames.
  These are constants in `agent/state.py`; changing them requires updating decisions.md.
- Scan action handling in `act_node`: runs detector on all 8 `ScanResult` frames, rotates drone to
  face the best detection yaw (if any), stores that detection in `last_detections`.
- `LLMPlanner` retries once on `Exception` then falls back to `Action(kind="scan")`.  The fallback
  is a constant at module level so it is testable without hitting the LLM.
- `HeuristicPlanner`: scan at step 0, scan every 4 steps, approach on detection, advance otherwise;
  rotate after 3 consecutive advances to explore a new direction.
- System prompt is `agent/prompts.py::SYSTEM_PROMPT` at `PROMPT_VERSION = "v1"`.  Any change to
  the prompt text requires incrementing `PROMPT_VERSION` and adding a new entry here.
- `recording.py` v1: `MissionLogger` appends one JSONL line per step (step, ts, action, result,
  detections, pose, usage, safety_events); path = `data/missions/<mission_id>.jsonl`.
- CLI command `run-mission` added; exits non-zero unless status == "found" so CI can gate on it.

**VERIFY item resolved (from M0):**
- VERIFY 4 — `with_structured_output(Action)`: Confirmed working for `openai/gpt-4o-mini` on
  OpenRouter.  No need for `method="function_calling"` fallback (LangChain handles it
  transparently).

---

## 2026-10-06 — M4: System prompt v2 (exploration enforcement)

**Context:** LLM test run (10 seeds, `living_room_v1`, `gpt-4o-mini`) showed 3/10 success.
Seeds 1–4 and 9 all terminated at 7–8 steps with `report_not_found`.  Root cause: prompt v1
rule 4 read "After ≥ 5 scans with no detection, consider `report_not_found`."  The model
interpreted this as a green light to scan from the starting position five times then quit,
without ever advancing to explore new areas.

**Decision:** Bump `PROMPT_VERSION` to `"v2"` with the following changes to the search strategy:
- Rule 4 replaced: `report_not_found` now requires ≥ 6 advance moves, ≥ 3 scans from
  different positions, AND being past 60 % of the step budget — all three conditions must hold.
- Rule 5 added: explicit hard floor — NEVER call `report_not_found` in the first 15 steps.
- Rule 6 added: explicit statement that scanning from the same position is not exploration.
- Honesty rule amended to reinforce exhausting the search area before declaring not_found.

**Consequences:** The absent_target test should still pass (the LLM will now explore the full
room before admitting the target is gone).  The found-rate on `living_room_v1` should improve
significantly because the model will now advance and cover the room before giving up.

---

## 2026-10-06 — M4: FOUND_AREA_FRAC lowered from 0.01 to 0.003; FOUND_CONSECUTIVE from 2 to 1

**Context:** HeuristicPlanner achieved 0/3 genuine finds after both geofence bugs were fixed.
Analysis of camera geometry revealed the root cause:

- Camera: forward-facing, 60° horizontal × 45° vertical FOV, drone at altitude 1.0 m.
- Cube: 0.2 m size at z=0.1 m (floor level), 0.9 m below the camera.
- Minimum horizontal viewing distance: `0.9 / tan(22.5°) ≈ 2.17 m` (below this the cube
  falls outside the vertical FOV entirely).
- At 2.17 m: cube area ≈ 11×11 px / (128×96 px) ≈ 0.9 % of frame — barely at threshold.
- `FOUND_AREA_FRAC = 0.01` (1 %) was therefore unreachable: the cube only approaches 1 %
  at the instant it disappears from the bottom of the frame.

Additionally, `FOUND_CONSECUTIVE = 2` could never be met when geofence rotation was triggered
on the approach step: the drone rotated away from the cube, breaking the consecutive streak.

**Decisions:**
- Lower `FOUND_AREA_FRAC` from 0.01 to 0.003 (0.3 %): matches the typical blob size at
  ~3 m viewing distance, well within the camera's vertical FOV.
- Lower `FOUND_CONSECUTIVE` from 2 to 1: a single qualifying frame (area ≥ 0.3 %, centred)
  is sufficient to declare found.

**False-positive risk:** Low.  `ColorDetector` requires S ≥ 90 saturation and a minimum
contour area of 30 px.  In the absent-target scenario (empty room), no red blobs exist.

---



**Context:** After fixing the geofence supervisor to force `rotate` instead of `land`, the
HeuristicPlanner regressed to 0/3 finds.  Root cause: when `approach` is geofence-rejected and
replaced with `rotate(90°)`, the drone faces away from the cube.  `check_node` sees the preserved
detection (area_frac OK) and increments `consecutive_found` to 1, but the next `perceive_node`
runs with the drone now facing away — no detection — resets `consecutive_found` to 0.  The
cycle repeats and `FOUND_CONSECUTIVE = 2` is never reached.

**Decision:** Lower `FOUND_CONSECUTIVE` from 2 to 1.  A single frame with `area_frac >= 0.01`
AND `|center_norm[0]| <= 0.6` is a strong enough signal: the target covers ≥ 1 % of the frame
and is horizontally centred.  The two-frame requirement was overly conservative and incompatible
with the rotate-on-reject safety response.

**Consequences:** The absent-target false-found rate is unaffected (ColorDetector never fires
in an empty room).  The found criterion is slightly more permissive but still specific enough
to avoid spurious detections.

---

## 2026-10-08 — M4: executor drift recovery after scan/rotate

**Context:** After fixing the supervisor's "already outside" path, advances from y=-2.99 were
allowed but still timed out (`status="clamped"`).  Root cause: the room has physical PyBullet
walls at y=±3.0 (built by `SceneBuilder._build_walls`).  The geofence margin is 0.3 m
(geofence at y=±2.7), but scan operations cause up to 0.3 m of PID drift, pushing the drone
against the physical wall at y=-2.99.  Once physically touching the wall, the PID controller
cannot stabilise the drone enough to complete any move → all moves time out.

**Decision:** Add `SkillExecutor._recover_from_drift(pose)` and call it at the end of every
`execute()` call, after the skill returns.  If the post-action pose is outside the geofence,
the method computes the nearest in-bounds target (geofence edge + 0.2 m buffer), converts the
world-frame delta to body-frame, and calls `drone.move_relative()` directly.  The recovery is
silent (no safety event added) and transparent to the agent — the next `perceive_node` call
sees the recovered position.

**Consequences:** Scan/rotate drift no longer traps the drone against the physical wall.
The 0.2 m inward buffer gives clearance from the geofence boundary after recovery.

---

## 2026-10-08 — M4: supervisor geofence drift recovery

**Context:** The supervisor only intercepts `advance`/`strafe`/`climb`/`descend` actions; it does
NOT check `rotate` or `scan`.  Both skills cause small physical displacements (PID overshoot,
8-frame sweep), which can push the drone slightly outside the geofence.  Once outside
(e.g. y=-2.99 when y_min=-2.7), `_max_scale_2d` misbehaves: if the heading has any tiny
southward component (floating-point sin(≈0°) < 0), the formula gives
`scale = (y_min - y) / tiny_negative = large_negative → max(0, negative) = 0`, so
`clamped_dist = 0 < _MIN_CLAMPED_M` → REJECT.  The drone is then trapped: every advance is
rejected and the planner guards loop endlessly.

**Decision:** Add a "drone already outside" early-exit in `_check_geofence_horizontal`:
- Detect if `pose.x` or `pose.y` is already outside the bounds.
- If the proposed move does **not** worsen the violation (neutral or inward), allow it outright.
- If the move pushes further outside, reject and rotate as before.

This recovers from physical drift without changing the normal (inside → attempted-outside)
clamping behaviour.

**Consequences:** Drones that drift outside the geofence during scan/rotate can now self-rescue
by advancing parallel to or away from the violated boundary.  The normal scale-clamping path is
unchanged for moves that start inside the geofence.

---

## 2026-10-08 — M4: LLM rejection loop fix (prompt v3 + consecutive-rejection guard)

**Context:** LLM run on `living_room_v1` (seed 2, 50 steps, 38 LLM calls, $0.006) showed the
drone stuck near the south wall for ~900 s.  The supervisor rejected the same advance direction
~30 times; the LLM never changed strategy because:
1. `_build_user_message` did not include safety events, so the LLM was unaware its advance was
   being rejected and replaced with a supervisor-forced rotate.
2. No programmatic guard existed to break the loop when the same direction was repeatedly blocked.

**Decisions:**
- `LLMPlanner._build_user_message`: append `"Safety alerts (recent): <last 3 safety events>"`
  so the LLM always sees rejection reason + kind in its context.
- `LLMPlanner.plan()`: consecutive-rejection guard — if `ctx.safety_events[-3:]` contains ≥ 2
  geofence events, skip the LLM call and return `rotate(135°)` directly.  This is a hard
  deterministic escape hatch; the drone will face a new direction before the LLM is consulted.
- `SYSTEM_PROMPT` bumped to v3: safety constraints section updated to explain that rejections
  appear in "Safety alerts", and a bold rule added: never repeat the same advance after a
  geofence rejection — rotate ≥ 90° first.

**Consequences:** The loop is broken at most 2 rejections deep (deterministic guard) and the LLM
has enough context to avoid the same direction next time.  The 135° rotation ensures the drone
faces well away from the blocking wall, not just 90° (which could still be into a corner).

---

## 2026-10-06 — M4: supervisor geofence rejection changed from land to rotate

**Context:** After fixing the false-found bug (see entry below), all seeds returned `aborted`
because the supervisor forced a `land` action on every horizontal geofence hit.  The drone would
hit a wall, get forced to land, `check_node` correctly returned `aborted`, and the mission ended.
Root cause: the old rejection response was `_make_land_action()` for both battery AND geofence
hits — geofence boundary contact is a soft error, not a safety-critical one.

**Decision:** Create `_make_rotate_action(reason)` that returns `Action(kind="rotate", angle_deg=90.0)`.
`_check_geofence_horizontal` now returns this rotate action instead of a land action when the drone
has no safe forward distance.  The `check` function still marks the verdict as `"reject"` (so
`check_node` does not treat it as a voluntary land), but the drone simply rotates 90° away from the
wall and continues searching.  Battery and altitude out-of-bounds still force a land.

**Consequences:** Mission throughput improved from ~665 s/mission to ~89 s/mission (rotate takes
~1 s vs. ~10 s for a full scan).  The supervisor warning message updated from "scan to reorient"
to "rotating to find a clear direction".

---

## 2026-10-06 — M4 bug: supervisor-forced land incorrectly declared "found"

**Context:** LLM test run revealed false-found rate 100% on `absent_target`.  Root cause:
`check_node` contained `if action_kind == "land": return {"status": "found"}`.  When the
safety supervisor rejects an action (geofence hit with no safe distance) it replaces it with
a forced `land`.  `act_node` executes the land, then `check_node` sees `action_kind == "land"`
and declares `found` — even in an empty room with zero detections.  This also corrupted the
`living_room_v1` results: the 10/10 "found" with prompt v2 in 6–8 steps were mostly geofence
false positives, not genuine visual finds.

**Decision:** Add `last_verdict_kind: str` to `MissionState` (values: `"allow"`, `"clamp"`,
`"reject"`).  `validate_node` sets it from `verdict.kind` each step.  `check_node` uses it:
- `action_kind == "land"` AND `last_verdict_kind == "reject"` → `status = "aborted"`
- `action_kind == "land"` AND verdict was `"allow"` or `"clamp"` → `status = "found"` (LLM chose to land)

**Consequences:** Safety-forced landings now correctly produce `"aborted"`.  Genuine `found`
still works via both the consecutive-detection criterion and voluntary LLM `land` calls.
The `absent_target` false-found rate drops to near 0.  The `living_room_v1` success rate will
reflect real visual detection rather than geofence collisions.

---

## 2026-10-09 — M4: fix grid cell jitter (floor → round in _cell_from_pose)

**Context:** With the direction-aware burst, seed 3 still returned `not_found`.  The burst fired
toward SW (-135°) instead of NW (135°, where the cube at x=-1.31, y=0.22 is).  Log analysis
showed that after the step-4 scan, the drone's pose was (-0.0000, 0.0000) — a tiny negative
float from PID drift.  `math.floor(-ε / 0.5)` = -1, recording cell (-1, 0) as NW visited.
At burst time: NW=1, NE=1, SW=0, SE=0 → `min()` picked SW (first zero-count key) → burst
headed -135° (opposite direction from the cube).

**Root cause:** `_cell_from_pose` used `math.floor`, so any sub-millimetre drift past the x=0
or y=0 cell boundary falsely registers the neighbouring cell as visited, corrupting the
coverage quadrant counts used by `_least_visited_quadrant_yaw`.

**Decision:** Change `_cell_from_pose` to use `round()`.  A drone at x=-0.0001 (1/10mm west)
rounds to cell 0 (NE) rather than cell -1 (NW).  Real exploration (moving ≥ 0.25 m into a
half-space) correctly maps to the expected cell index.  The ±0.25 m snap zone is well below
any deliberate move (minimum 0.2 m) and well above typical PID drift (< 5 mm).

**Consequences:** Sub-millimetre origin drift no longer contaminates the coverage signal.
`_least_visited_quadrant_yaw` will correctly identify NW as unexplored when the drone has
only scanned and rotated at the origin, directing the burst toward the right half of the room.

---

## 2026-10-09 — M4: direction-aware burst

**Context:** Seed 3 run with v5 prompt + 3-step burst returned `not_found` in 33.8 s / 50 steps.
After the initial scan+rotate sequence, the drone's yaw was 53° (pointing NE).  The no-advance
guard fired and sent 3×1.0m advances at 53° → drone reached (2.07, 2.86), deep in the NE corner.
The cube was at x=-1.31, y=0.22 (NW).  The burst drove the drone the opposite direction.  Once
in the NE corner, every advance hit the geofence; the geo-rejection loop consumed the rest of
the budget.  The drone never visited x<0.

**Root cause:** The burst fired in the drone's current heading, which was set by the most recent
scan (360° sweep → final yaw unpredictable).  3m in an arbitrary direction can push the drone
into a corner on the wrong side of the room.

**Decision:** Make the burst direction-aware:
- `LLMPlanner.__init__`: add `_burst_target_yaw: float | None = None`.
- New static method `_least_visited_quadrant_yaw(visited)`: inspects `visited` grid cells,
  counts per quadrant (NW/NE/SW/SE), picks the least-visited, and returns the corresponding
  world yaw (NW→135°, NE→45°, SW→-135°, SE→-45°, using dx=cos(yaw), dy=sin(yaw) convention).
- No-advance guard: compute `_burst_target_yaw` from coverage, set `_committed_advances = 3`,
  then return a **rotate** toward the target yaw rather than the first advance.  The burst-drain
  fires the 3 advances on subsequent steps.
- Burst-drain: before each advance, check if current yaw is >45° off `_burst_target_yaw`.
  If so, return a rotate (no burst step consumed) until aligned; then advance.
- Consecutive-geo guard: resets `_committed_advances = 0` and `_burst_target_yaw = None` to
  abort a burst that keeps hitting walls.

**Consequences:** Bursts now travel toward the least-explored quadrant regardless of what yaw
the previous scan left the drone at.  Alignment rotates are free (don't consume burst steps).
A burst that hits the geofence is aborted by the geo guard after 2 rejections.

---

## 2026-10-08 — M4: committed advance burst + prompt v5

**Context:** Seed 3 run (`living_room_v1`, red cube at x=-1.31, y=0.22 — NW quadrant) returned
`not_found` in 57.3 s / 50 steps / 31 LLM calls.  Two failure modes observed:

1. Step 1: LLM wrote "advance to NW" but the drone's heading after the scan pointed SE.
   `advance` always moves in the current yaw — the LLM didn't rotate first.
2. Steps 5–50: LLM saw "UNEXPLORED: NW" each step and chose `rotate` repeatedly.  The
   no-advance guard fired every ~6 steps and forced a single 0.5 m advance.  From x=2.5,
   one 0.5 m advance only reaches x=2.0 — still SE.  The drone needed ~8 advances to cross
   to negative-x territory; it never got more than one in a row.

**Decisions:**
- `LLMPlanner.__init__`: add `_committed_advances: int = 0`.
- `LLMPlanner.plan()`: before consulting the LLM, drain `_committed_advances` by returning
  `advance(1.0 m)` and decrementing the counter.  When the no-advance guard fires, set
  `_committed_advances = 2` (this step + 2 more = **3 × 1.0 m = 3 m** total burst).
  3 m is enough to cross from any SE corner to the NW half of a 6 m room.
- `SYSTEM_PROMPT` bumped to v5: new "Heading and movement rules" section explaining that
  `advance` follows current yaw (rotate first if needed) and that reaching a new quadrant
  requires **at least 3 consecutive 1.0 m advances** without interruption.  Coverage rule 4
  updated with the same 3-advance commitment requirement.

**Consequences:** The drone now travels in committed bursts rather than one-advance-then-rotate
cycles.  The deterministic burst guard is the hard backstop; the prompt change reduces how often
the guard is needed by teaching the LLM the correct multi-advance pattern.

---

## 2026-10-08 — M4: prompt v4 — coverage-guided exploration

**Context:** After fixing the geofence loop and drift recovery, a 61.7 s run showed the drone
moving freely but still returning `not_found` for `living_room_v1` seed 2 (target at
(-1.19, -1.01)).  Log analysis showed the drone stayed in the SW quadrant for the majority of
steps, never venturing into NE or SE.  The `visited` grid data was already being computed
(M4 state, GRID_CELL_SIZE_M=0.5) and exposed in `MissionState["visited"]`, but neither the
user message nor the prompt told the LLM to use it.

**Decisions:**
- `PlanContext`: add `visited: list[dict]` field (list of `{x_idx, y_idx}` grid cell dicts).
- `make_plan_node` in `nodes.py`: pass `visited=list(state["visited"])` when constructing
  `PlanContext`.
- `LLMPlanner._coverage_summary(visited)`: new static method that computes unique cell count
  and per-quadrant breakdown (NW/NE/SW/SE relative to origin), flagging any quadrant with
  zero visits as "UNEXPLORED".  Example: `18 cells (NW=3 NE=2 SW=12 SE=1) — UNEXPLORED: NE`.
- `LLMPlanner._build_user_message`: append `Coverage: {coverage}` line from the summary.
- `SYSTEM_PROMPT` bumped to v4: new rule in "Search strategy" — if any quadrant is UNEXPLORED,
  rotate to face it and advance there before re-scanning already-visited areas.

**Consequences:** The LLM now has explicit spatial feedback on where it has and hasn't been.
UNEXPLORED flags create a strong directional signal to break the tendency to scan in circles.
The coverage summary is O(n) over visited cells and adds negligible overhead.
