# Kormoran

**Goal:** An LLM-driven agent that takes a plain-language mission ("find the red cube") and completes it inside a visible 3D quadcopter simulator with no pre-scripted route.

**Problem it solves:** Traditional drone scripts break the moment the environment changes. A human-controlled drone requires constant attention. Kormoran bridges both: it uses an LLM to make high-level navigation decisions from camera observations, while a deterministic inner loop safely executes each move — so the same code works across different room layouts and target objects.

**How it works:** A LangGraph loop — *recall → perceive → plan → validate → act → check* — lets the model pick the next macro-action (scan, advance, climb, approach…) from what the camera currently shows. A deterministic PID controller flies each action inside gym-pybullet-drones. An independent safety supervisor vetoes any action that would violate geofence, altitude, or battery limits. Long-term memory (SQLite + ChromaDB) lets the agent recall where objects were found across missions, so search gets faster over time. An evaluation harness proves that claim with numbers.

## Quickstart

```bash
# 1. Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and set up
git clone <repo-url> && cd kormoran
uv sync
uv run pre-commit install

# 3. Copy env template and add your OpenRouter key
cp .env.example .env
# Edit .env: OPENROUTER_API_KEY=sk-or-...

# 4. Verify the LLM connection
uv run kormoran check-llm

# 5. Run tests
make test
```

## Architecture

See [ROADMAP.md](ROADMAP.md) for the full two-loop design, package layout, and milestone plan.

## Development

```bash
make setup       # uv sync + pre-commit install
make lint fmt    # ruff check + format
make test        # unit tests (no sim, no LLM)
make test-sim    # headless simulator tests
make test-llm    # LLM round-trip tests (needs OPENROUTER_API_KEY)
make docker-build
```

## Ethics

See [docs/ethics.md](docs/ethics.md) *(added in M8)*.
