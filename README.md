# Castle Steward Engine

Durable, crash-resumable workflow engine driven through the Agent Layer CLI (`al dispatch`),
paired with the sibling `castle_steward_coding` workflow package registered via Python entry
points (`castle_steward_engine.workflows`).

## Architecture

- `castle_steward_engine`: Generic workflow SDK, SQLite state store + LangGraph checkpointing,
  thin `al -q dispatch` client, per-session stdio MCP server (session files + completion checks;
  never touches SQLite), and `stew` CLI. Never imports workflow packages.
- `castle_steward_coding`: Declarative coding workflow (`resolve_setup -> coordinate ->
  assign_agents -> plan? -> implement -> parallel review -> ship -> merge gate ->
  exact-head merge -> verify -> cleanup`). Consolidates `coordinate`, `triage_repair`, and
  `merge_gate` under a fresh-session `orchestrator` role paired with deterministic, quota-aware
  coder and cross-family reviewer routing. Separates internal state (`models.py`), deterministic
  routing (`routing.py`), agent-facing MCP tool contracts (`tools.py`), and pure agent-facing
  prompt templates (`prompt_templates.py`). Imports only top-level `castle_steward_engine` public
  exports.
- Every agent turn completes either by submitting a validated Pydantic payload through a
  per-session MCP tool (or `ask`) verified by `check_done`, or via a check-only `check_done`
  callback that verifies live external state (such as Git commits and GitHub PRs); agent text
  output is never parsed.

## Public SDK

Workflows import exclusively from `castle_steward_engine`:
`Workflow`, `Graph`, `Role`, `Turn`, `Wait`, `JobInfo`, `AgentSpec`, `AvailableAgent`,
`QuotaReading`, `QuotaSnapshot`, `AppendList`, `END`, `load_workflows`, `load_workflow`.

For deterministic in-process workflow tests without real LLMs or `al` subprocesses, import the
public test harness (`FakeAgentLayer`, `FakeTurnContext`, `FakeTurnOutcome`, `SimulatedCrash`)
explicitly from `castle_steward_engine.testing`.

## CLI

```bash
stew workflows
stew start coding --input '{"task": "...", "checkout": "/path/to/repo"}'
stew run <job_id>
stew inspect <job_id>
stew reply <job_id> <wait_id> --value '{"approved": true}'
stew stop <job_id>
stew mcp --session-dir <session_dir>
```

## Coding Workflow Configuration & Routing

- **Environment defaults:** `CodingSettings` reads `STEW_ENGINE_CODING_*` environment variables
  (e.g., `STEW_ENGINE_CODING_ALLOW_UNATTENDED_MERGE`, `STEW_ENGINE_CODING_MAX_AUTO_REPAIRS`) once
  in `resolve_setup` and checkpoints the resolved `CodingConfig` on the job.
- **Orchestrator:** Defaults to `antigravity` / `gemini-4-neon-high` on `CodingConfig.orchestrator`;
  per-job `CodingInput.orchestrator` overrides `CodingConfig.orchestrator`. Runs fresh sessions for
  `coordinate` (`submit_assessment`), `triage_repair` (`submit_repair_triage`), and each
  `merge_gate` attempt (`submit_gate_decision`, including `escalate`).
- **Check-only coder completion:** `implement`, `address_review`, `repair`, and `ship_pr` verify
  live Git lineage and GitHub PR state directly inside `check_done` (no `submit_commit` or
  `submit_ship` MCP tools); the `coder` role exposes only `submit_plan`, `ask`, and `check_done`.
- **Immutable artifact mirrors:** Plans (`plan-vN.md`) and merge-gate context files
  (`merge-context-aN.md`) are written canonically under `<state_dir>/artifacts/<job_id>/` and
  mirrored read-only (`0o444`) under `CodingConfig.artifact_mirror_dir` (configurable via
  `STEW_ENGINE_CODING_ARTIFACT_MIRROR_DIR`, default `.agent-layer/cache/steward-engine`) inside
  `<checkout>/<artifact_mirror_dir>/<job_id>/` so every provider reads workspace-relative paths
  cleanly. `resolve_setup` requires the configured mirror directory to be gitignored in the target
  checkout (the default in Agent Layer repositories). Post-merge `cleanup` removes the job's
  checkout mirror directory; aborted, declined, or failed runs leave their gitignored cache
  directory in place, and canonical copies always remain in `state_dir`.
- **Deterministic routing:** `RoutingPolicy` and `AgentProfiles` in
  `castle_steward_coding/routing.py` map `(difficulty, needs_plan)` plus host availability and
  quota telemetry to a coder and distinct cross-family reviewers. Until `conn-castle/agent-layer#575`
  ships, `AgentLayer.quota()` returns an empty `QuotaSnapshot`, so burn-rate pace thresholds stay
  inactive and reviewer pairs pick randomly.

## Forthcoming Agent Layer Dependencies

Live execution depends on four upcoming Agent Layer features:
- **#563** — Session-scoped MCP configuration passed on `al dispatch start` and reported in `al dispatch options`.
- **#564** — Continuation reservations (`al dispatch reserve --continue` / `al dispatch continue --reservation`).
- **#575** — Quota and burn-rate telemetry consumed via `AgentLayer.quota()` / `JobInfo.agent_quota()` for deterministic routing.
- **#576** — Disabling nested `al dispatch` inside dispatched workflow agent turns.

## Development

```bash
make bootstrap    # create .venv, install editable package + dev deps, install hooks
make check        # ruff format --check, ruff check, mypy, pytest
make hooks-check  # run pre-commit and pre-push hooks across all project files
```

### Database Migrations

To generate a new Alembic migration after updating `castle_steward_engine/store.py` (against a DB upgraded to `head`):

```bash
.venv/bin/alembic -x sqlalchemy.url=sqlite:///path/to/engine.db revision --autogenerate -m "<msg>"
```
