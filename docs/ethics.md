# Ethics & Responsible AI

This document covers the ethical considerations for Kormoran, an LLM-driven autonomous drone agent operating in simulation. Each concern maps to a concrete mitigation implemented in the codebase.

---

## 1. Dual Use

**Risk:** An autonomous search drone is surveillance- and military-adjacent. Object-search capabilities could be repurposed for tracking people, unauthorized surveillance, or weaponized autonomy.

**Mitigations:**
- **Intended scope is explicit:** indoor, hobby/research, finding inanimate objects in simulation. The README and all documentation state this.
- **No person detection:** The perception module (`ColorDetector`) only recognises coloured geometric primitives via HSV thresholding. It cannot identify, track, or classify people.
- **Simulator-first stance:** The graded build runs entirely in PyBullet simulation. Hardware support (M9) is architectured for but not required and is restricted to a toy-class DJI Tello indoors.
- **No outdoor autonomy:** Geofence limits are hard-coded in the safety supervisor and cannot be disabled by the LLM.

## 2. Privacy

**Risk:** The drone camera captures images of indoor spaces that may contain personal items or people.

**Mitigations:**
- **Memory never stores raw frames.** The `EpisodicStore` and `SemanticStore` schemas are tested to confirm no image bytes are persisted (`test_no_frames_in_schema`).
- **Recordings are explicit.** GIF recordings are only saved when `RECORD_MISSIONS=true` (configurable). On hardware, this defaults to off.
- **No face or person detection** is implemented anywhere in the codebase.
- **VLM guardrail (if added):** Any future vision-language model detector is restricted to outputting structured `Detection` objects. Free-text VLM output never reaches the planner prompt, preventing indirect exfiltration of visual information.

## 3. Safety

**Risk:** An autonomous drone making decisions from an LLM could crash, fly out of bounds, or behave unpredictably.

**Mitigations:**
- **Independent safety supervisor** (`safety/supervisor.py`): Runs on every action before execution. The LLM cannot bypass it. It enforces geofence bounds, altitude limits, and battery reserves.
- **The LLM is never inside the real-time control loop.** It emits one validated discrete `Action` per decision step; the deterministic PID controller executes it at simulator rate.
- **Battery reserve:** 15% battery floor forces a landing before power is exhausted.
- **Emergency stop:** The Streamlit dashboard writes a sentinel file; the agent's `check_node` polls it each step and aborts cleanly.
- **Hardware kill switch (M9):** A human-held keyboard kill switch triggers `emergency_stop()` instantly.
- **Budget limits:** Per-mission caps on steps (`max_steps`), LLM calls (`max_llm_calls`), cost (`max_cost_usd`), and wall-clock time (`max_mission_s`) prevent runaway behaviour.
- **Zero safety violations** is a hard evaluation requirement. The eval harness counts every supervisor reject and the DoD requires the count to be zero across the entire test suite.

## 4. Honesty & Hallucination

**Risk:** The LLM might claim to have found a target that isn't there, or navigate based on hallucinated spatial reasoning.

**Mitigations:**
- **`report_not_found` is a first-class outcome.** The agent is explicitly trained (via prompt) and tested to report honestly when a target cannot be found.
- **False-found rate is measured.** The `absent_target` scenario has no targets; the eval harness tracks `false_found_rate` — LLM claims of "found" when nothing exists.
- **Detection-based found criterion.** The agent cannot self-declare "found" — the `check_node` requires the camera to actually see the target (area fraction + centring thresholds) for `FOUND_CONSECUTIVE` frames before triggering the land reflex.
- **Memory notes are labelled "may be stale."** The prompt section "Prior knowledge" explicitly warns the LLM that recalled memories might be outdated. Contradicting observations halve note confidence.

## 5. Prompt Injection via the Environment

**Risk:** An attacker could place text or adversarial patterns in the physical environment to hijack the LLM's reasoning.

**Mitigations:**
- **Perception output is structured data.** The `ColorDetector` produces `Detection` objects (label, bounding box, area fraction, confidence) — not free text. The LLM never sees raw pixel data or OCR output.
- **The planner prompt is code-controlled.** `agent/prompts.py` is versioned; changes require a `decisions.md` entry. The prompt template does not interpolate arbitrary external strings.
- **VLM guardrail (if added):** VLM output is parsed into `Detection` objects only, never into free text that reaches the planner. This prevents an adversary from injecting instructions via camera-visible text.

## 6. Cost & Energy

**Risk:** Uncontrolled LLM usage wastes money and compute resources.

**Mitigations:**
- **Per-mission budget caps:** `max_llm_calls=40`, `max_cost_usd=0.50`, `max_tokens` capped in the LLM config.
- **Decision count, not flight time, drives LLM usage.** The LLM is called once per decision step, not continuously.
- **`temperature=0`** for deterministic, minimal-token responses.
- **Cost tracking:** Every mission logs prompt tokens, completion tokens, and estimated cost. The eval harness aggregates this across all runs.
- **OpenRouter spend limit:** The project uses a dedicated API key with a spend cap set on the OpenRouter dashboard.

## 7. Data Provenance

**Risk:** Training data, assets, or dependencies may have unclear licensing or contain personal data.

**Mitigations:**
- **Simulator assets are self-made.** All scene objects (cubes, spheres, cylinders, obstacles) are programmatically generated — no external 3D models or datasets.
- **No external datasets of people** are used anywhere.
- **Dependencies are pinned** (`uv.lock` committed) and auditable. The CI workflow uses `--frozen` to ensure reproducibility.
- **No pre-trained models are fine-tuned.** The LLM is used via API (OpenRouter) with no custom training data.

---

## Summary

| Concern | Key mitigation | Evidence |
|---------|---------------|----------|
| Dual use | HSV-only perception, no person detection, indoor sim scope | `perception/color_detector.py` |
| Privacy | No frames in memory stores, recordings configurable | `test_no_frames_in_schema` |
| Safety | Independent supervisor, budget caps, kill switch, zero-violation eval | `safety/supervisor.py`, `docs/eval_results.md` |
| Honesty | `report_not_found`, false-found metric, detection-based found | `eval/metrics.py`, `absent_target` scenario |
| Injection | Structured perception output, code-controlled prompt | `agent/prompts.py`, `perception/` |
| Cost | Per-mission caps, cost tracking, spend limit | `config.py`, eval cost column |
| Provenance | Self-made assets, pinned deps, no external datasets | `configs/scenarios/`, `uv.lock` |
