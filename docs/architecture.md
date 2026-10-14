# Architecture

Kormoran uses a **two-loop architecture**: an LLM-driven outer loop for high-level decisions and a deterministic PID inner loop for flight control.

```
                    ┌─────────────────────────────────────────────┐
                    │              LangGraph Outer Loop            │
                    │                                             │
                    │  recall → perceive → plan → validate → act  │
                    │     ↑                                  │    │
                    │     └──────────── check ←──────────────┘    │
                    └───────────────────┬─────────────────────────┘
                                        │ Action (validated)
                                        ▼
                    ┌─────────────────────────────────────────────┐
                    │         Deterministic Inner Loop             │
                    │                                             │
                    │  PID Controller → Simulator (PyBullet)      │
                    │  DSLPIDControl    CtrlAviary @ 240 Hz       │
                    └─────────────────────────────────────────────┘
```

## Package Layout

```
src/kormoran/
├── agent/              # M4: LangGraph graph, nodes, planners, prompts
│   ├── graph.py        # StateGraph: recall → perceive → plan → validate → act → check
│   ├── nodes.py        # Node factories (make_*_node), route_check()
│   ├── planners.py     # ScriptedPlanner, HeuristicPlanner, LLMPlanner
│   ├── prompts.py      # SYSTEM_PROMPT (versioned, changes need decisions.md)
│   ├── llm.py          # make_llm() → ChatOpenAI on OpenRouter
│   └── state.py        # MissionState TypedDict + constants
├── control/            # M2: Deterministic skills + executor
│   ├── skills.py       # takeoff, land, scan, advance, strafe, climb, descend, rotate, approach
│   └── executor.py     # SkillExecutor (supervisor + watchdog)
├── drones/             # M1: Backend implementations
│   ├── mock.py         # MockDrone (unit tests, no PyBullet)
│   └── sim.py          # SimDrone (CtrlAviary + DSLPIDControl)
├── eval/               # M6: Evaluation harness
│   ├── scenarios.py    # EvalCase, build_matrix()
│   ├── runner.py       # Subprocess-per-mission runner (resumable)
│   ├── metrics.py      # GroupMetrics, compute_group_metrics()
│   └── report.py       # Markdown + PNG plots → docs/eval_results.md
├── memory/             # M5: Long-term memory
│   ├── episodic.py     # EpisodicStore (SQLite)
│   ├── semantic.py     # SemanticStore (ChromaDB)
│   ├── reflection.py   # Post-mission LLM reflection → MemoryNote
│   ├── retrieval.py    # Recall + staleness update
│   └── schemas.py      # EpisodicRecord, MemoryNote
├── perception/         # M3: Camera → Detection
│   ├── color_detector.py   # HSV thresholding (Perceiver protocol)
│   ├── ground_truth.py     # Segmentation mask → Detection (eval only)
│   └── summarizer.py       # Detection list → natural language
├── safety/             # M2: Independent supervisor
│   ├── supervisor.py   # check(action, state, config) → Verdict
│   └── battery.py      # Battery reserve logic
├── ui/                 # M7: Streamlit dashboard
│   ├── app.py          # 4 tabs: Mission Control, Live Feed, Memory, Eval
│   ├── launcher.py     # Subprocess mission launcher
│   ├── log_reader.py   # JSONL log parser
│   └── stop_flag.py    # Emergency stop sentinel file
├── cli.py              # Typer commands: run-mission, eval, dashboard, memory, etc.
├── config.py           # Settings (pydantic-settings), ScenarioConfig (YAML)
├── interfaces.py       # DroneInterface, Perceiver, Planner protocols
├── recording.py        # MissionLogger (JSONL) + save_gif()
└── types.py            # Pose, DroneState, Action, Detection, etc.
```

## Key Design Decisions

### 1. Two-Loop Separation
The LLM emits one `Action` per decision step. The deterministic controller executes it at 240 Hz. The LLM never controls motors directly — it cannot cause a crash by emitting bad control signals.

### 2. Independent Safety Supervisor
The supervisor (`safety/supervisor.py`) runs on every action *after* the LLM decides but *before* execution. It can:
- **Allow** the action unchanged
- **Clamp** distances to stay within geofence/altitude
- **Reject** the action entirely and force a landing

The supervisor is a pure function with no LLM dependency — it cannot be prompt-injected.

### 3. Backend Abstraction
The `DroneInterface` protocol defines the contract: `connect`, `takeoff`, `move`, `rotate`, `land`, `get_state`, `get_frame`. Implementations:
- `MockDrone` — unit tests (no physics)
- `SimDrone` — PyBullet simulation
- `TelloDrone` — DJI Tello hardware (M9, architectured for)

Agent, memory, perception, and UI never import a backend.

### 4. Memory Architecture
- **Episodic** (SQLite): Full mission records (outcome, steps, cost). Used for the eval harness and the dashboard history table.
- **Semantic** (ChromaDB): Typed notes (location, wall, scene, strategy) with confidence scores. Retrieved by scene+target similarity at mission start. Contradicting observations halve confidence; stale notes are pruned.

### 5. Subprocess Isolation
Both the Streamlit dashboard and the eval harness launch missions as subprocesses (`kormoran run-mission`). This ensures:
- Clean PyBullet environment per mission
- UI crash doesn't kill the agent (and vice versa)
- Results are read from JSONL logs, not shared memory
