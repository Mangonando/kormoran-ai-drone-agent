# Presentation Notes — Kormoran

## SCR Narrative (10 minutes)

### Situation
Drones are transforming inspection, search-and-rescue, and delivery — but they need either a skilled human pilot or a pre-programmed flight script. A pilot is expensive and can't scale. A script breaks the moment the room layout changes, the target moves, or an obstacle appears.

### Complication
LLMs can reason about goals, plans, and observations — but they're too slow and unreliable for real-time flight control. A single bad token could crash a drone. And without memory of past flights, the LLM rediscovers the same walls and dead ends every mission.

### Resolution
Kormoran separates the problem into two loops:
- **Outer loop (LLM):** Decides *what* to do next — scan, advance, approach, report not found — based on camera observations, memory, and reasoning. One decision per step, validated by an independent safety supervisor before execution.
- **Inner loop (PID):** Executes *how* to do it — deterministic flight control at 240 Hz inside PyBullet simulation. The LLM never touches motor commands.

Long-term memory (SQLite + ChromaDB) stores what the agent learned: target locations, wall positions, gap crossings, search strategies. Mission 3 is measurably faster than mission 1 because the agent navigates directly to known gaps instead of rediscovering them.

---

## Six Required Points

### 1. Problem
Finding a named object in an unknown indoor environment using a camera-equipped drone, without a pre-scripted route or human pilot. The agent must search, navigate around obstacles, and report honestly when a target doesn't exist.

### 2. Solution
A LangGraph agent loop with six nodes (recall → perceive → plan → validate → act → check) running above a deterministic PID controller. An independent safety supervisor vetoes unsafe actions. Long-term memory (episodic + semantic) enables cross-mission learning.

**Key architectural choice:** The LLM is never inside the control loop. It emits one validated `Action` per decision step. This makes the system safe by construction — a hallucinated action is caught by the supervisor before it reaches the motors.

### 3. Data
- **Scenarios:** 4 YAML-defined environments (living room, absent target, apartment with wall dividers, penthouse labyrinth) with randomised target placement via seeds.
- **Memory data:** SQLite episodic records + ChromaDB semantic notes, accumulated across missions. No external datasets — all training data is self-generated through flight experience.
- **No personal data:** The camera sees coloured primitives in simulation. No faces, no text, no real-world images.

### 4. Evaluation
Reproducible evaluation harness (`make eval SEEDS=5`) running a scenario matrix:
- **Success rate:** ≥ 70% LLM on known targets; 100% on absent-target (no false claims)
- **Zero safety violations** across all runs
- **Memory gain:** Mean steps for mission 3 < mission 1 with memory on
- **Cost tracking:** Tokens, LLM calls, estimated USD per mission

Results: 38 runs, $0.064 total cost, zero safety violations. See `docs/eval_results.md`.

### 5. Challenges
1. **LLM spatial reasoning is weak.** Small models struggle to mentally track walls and gaps. Solution: deterministic gap inference in code, injected as navigation hints into the prompt.
2. **Memory pollution.** Too many notes with slightly different coordinates confuse the LLM. Solution: wipe-on-success, deduplication, max 5 notes per scene+target, confidence decay.
3. **Cold start.** First mission in a new scene usually fails — the agent must discover walls before it can navigate. This is by design; the value is in the learning loop.
4. **PyBullet build issues.** pybullet 3.2.7 fails to compile on macOS arm64 + Xcode 21. Solution: patched wheel committed to `wheels/`.

### 6. Next Steps
- **Hardware (M9):** `TelloDrone` backend via `djitellopy` — same agent code, real flight. Kill switch, connection heartbeat, tighter safety limits.
- **VLM perception:** Vision-language model detector for real-world objects ("blue chair") instead of HSV colour matching.
- **SLAM integration:** Replace LLM spatial reasoning with proper SLAM + path planning for obstacle avoidance, keeping the LLM for high-level goal selection.
- **Multi-drone coordination:** Multiple agents sharing a semantic memory store, dividing search areas.

---

## Rubric Self-Check

| Criterion | Evidence | Location |
|-----------|----------|----------|
| **Outcome** — Working AI agent with tool integration, memory, multi-step reasoning | Full LangGraph loop: perceive → plan → validate → act → check. LLM planner + safety supervisor + long-term memory. Finds targets across 4 scenarios. | `src/kormoran/agent/`, `src/kormoran/memory/` |
| **Outcome** — Streamlit prototype | 4-tab dashboard: launch missions, live feed, memory browser, eval results. Emergency stop. AppTest smoke tests. | `src/kormoran/ui/app.py`, `tests/unit/test_ui.py` |
| **Outcome** — Evaluation with metrics | Reproducible harness: 38+ runs, success rate, false-found rate, steps, cost, safety violations. Markdown report + PNG plots. | `src/kormoran/eval/`, `docs/eval_results.md` |
| **Learning Application** — LangGraph agent loop | StateGraph with 6 nodes, `SqliteSaver` checkpointer, crash-and-resume tested. | `src/kormoran/agent/graph.py`, `tests/unit/test_agent_graph.py` |
| **Learning Application** — Tool integration | 10 action types (scan, advance, strafe, climb, descend, rotate, approach, land, takeoff, report_not_found) as validated Pydantic models. | `src/kormoran/types.py`, `src/kormoran/control/skills.py` |
| **Learning Application** — Long-term memory | SQLite episodic + ChromaDB semantic stores. Post-mission reflection. Cross-mission learning measured. | `src/kormoran/memory/`, `tests/unit/test_memory.py` |
| **Ethics** — Responsible AI considerations | 7-section ethics document covering dual use, privacy, safety, honesty, injection, cost, provenance. Each with concrete mitigations. | `docs/ethics.md` |
| **Presentation** — Clear narrative | SCR structure + 6 required points in this document. | `docs/presentation.md` |
| **Showcase** — Deployable demo | `make dashboard` launches the Streamlit UI. Docker multi-stage build. README quickstart. | `Makefile`, `Dockerfile`, `README.md` |
