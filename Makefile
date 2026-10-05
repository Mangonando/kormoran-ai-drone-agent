.PHONY: setup lint fmt test test-sim test-llm docker-build run eval dashboard

# --- Setup ---
setup:
	uv sync
	uv run pre-commit install

# --- Quality ---
lint:
	uv run ruff check .

fmt:
	uv run ruff format .

# --- Tests ---
test:
	uv run pytest -m "not llm and not sim" -v

test-sim:
	uv run pytest -m sim -v

test-llm:
	uv run pytest -m llm -v

# --- Docker ---
docker-build:
	docker build -t kormoran .

# --- Runtime (placeholders until M1+) ---
run:
	uv run kormoran run-mission \
		--target "$(TARGET)" \
		--scenario $(or $(SCENARIO),living_room_v1) \
		--seed $(or $(SEED),1) \
		$(if $(GUI),--gui,)

eval:
	uv run kormoran eval --seeds $(or $(SEEDS),5)

dashboard:
	uv run kormoran dashboard
