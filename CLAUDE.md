# Kormoran

Read [ROADMAP.md](ROADMAP.md) first; work one milestone at a time; a milestone is done only when its Definition of Done passes.

- Python 3.12, `uv`, `ruff` (pre-commit), `pytest` markers `unit`/`sim`/`llm`/`slow`.
- Never commit secrets; never log the API key (`SecretStr` everywhere).
- Pydantic v2 models at every boundary. LLM output is a validated `Action`; the safety supervisor has the final say.
- Backends (`sim`/`mock`/`tello`) are only chosen in `cli.py`/config. Agent, memory, perception, UI never import them.
- Log architectural changes in `docs/decisions.md`. Update ROADMAP.md checkboxes as milestones complete.
- Every commit: `uv run pre-commit run --all-files` clean → `uv run pytest -m "not llm"` green.
