# Castle Steward Engine

Castle Steward Engine coordinates coding jobs in a worker checkout. The intended
workflow scopes a task, runs implementation and review, and pauses when human
input is needed. Job inspection shows the saved state of a job.

Repository setup is landing first. The engine is under development. Live agent
execution requires Agent Layer to accept per-invocation MCP configuration on
both start and continue.

## Development

Install uv, then run `make bootstrap` to create the virtual environment and
install Git hooks. uv selects the Python version pinned in `pyproject.toml`.

- `make check` runs Ruff and, when engine code or tests are present, mypy and pytest.
- `make hooks-check` runs both hook stages across tracked and untracked project files.
