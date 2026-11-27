"""Thin CLI: `stew [--state-dir PATH] {workflows, start, run, reply, stop, inspect, mcp}`.

Command / execution split:
- `workflows` lists installed workflow entry points (name + version).
- `start`, `reply`, and `stop` validate and record durable facts in SQLite without executing
  any workflow nodes.
- `run` is the sole command that acquires the exclusive per-job lock and advances a job until
  completion, pause (human wait/ask), cooperative stop, or error.
- `inspect` returns a non-blocking, read-only snapshot of phase, pending waits, active turns,
  and checkpointed state (safe to run concurrently with `run`).
- `mcp` serves the per-session stdio MCP protocol over stdin/stdout (session files and
  completion checks; never imports LangGraph or SQLite).

I/O & exit-code contract:
- All non-`mcp` subcommands print structured JSON to stdout and exit `0` on success.
- On runtime or validation errors, non-`mcp` subcommands print
  `{"error": "<ExceptionType>", "message": "..."}` JSON to stderr and exit `1`
  (argparse argument errors exit `2`).
- Environment defaults (`STEW_ENGINE_STATE_DIR`, `STEW_ENGINE_AGENT_LAYER_EXECUTABLE`) are read
  via `EngineSettings` here; `--state-dir` overrides `STEW_ENGINE_STATE_DIR`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic_settings import BaseSettings, SettingsConfigDict

from castle_steward_engine.agent_layer import DEFAULT_AGENT_LAYER_EXECUTABLE
from castle_steward_engine.workflow import load_workflows


class EngineSettings(BaseSettings):
    """Environment-backed defaults for CLI-constructed `Engine` instances.

    `STEW_ENGINE_*` environment variables (`STEW_ENGINE_STATE_DIR` and
    `STEW_ENGINE_AGENT_LAYER_EXECUTABLE`) are read only in the CLI layer; programmatic
    `Engine(state_dir=None)` callers never consult `STEW_ENGINE_*` variables and fall back
    only to the standard XDG `default_state_dir()`.
    """

    model_config = SettingsConfigDict(env_prefix="STEW_ENGINE_", extra="forbid", frozen=True)

    state_dir: Path | None = None
    agent_layer_executable: str = DEFAULT_AGENT_LAYER_EXECUTABLE


def _load_json_arg(raw: str | None, file_path: str | None) -> Any:
    """Parse a mutually exclusive JSON string or file path argument."""
    if file_path is not None:
        return json.loads(Path(file_path).read_text(encoding="utf-8"))
    assert raw is not None
    return json.loads(raw)


def _build_parser() -> argparse.ArgumentParser:
    """Construct the `stew` argument parser covering all seven subcommands."""
    parser = argparse.ArgumentParser(prog="stew", description="Castle Steward Engine CLI")
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help="Engine state directory (overrides STEW_ENGINE_STATE_DIR)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("workflows", help="List installed workflow entry points")

    p_start = sub.add_parser("start", help="Record a new pending job without running it")
    p_start.add_argument("workflow", help="Workflow name")
    start_input = p_start.add_mutually_exclusive_group(required=True)
    start_input.add_argument("--input", dest="input_json", default=None, help="JSON input object")
    start_input.add_argument("--input-file", default=None, help="Path to JSON input file")

    p_run = sub.add_parser("run", help="Run or resume a job until completion, pause, or stop")
    p_run.add_argument("job_id", help="Job UUID")

    p_reply = sub.add_parser("reply", help="Record a validated reply to a pending wait")
    p_reply.add_argument("job_id", help="Job UUID")
    p_reply.add_argument("wait_id", help="Pending wait ID from inspect")
    reply_value = p_reply.add_mutually_exclusive_group(required=True)
    reply_value.add_argument("--value", dest="value_json", default=None, help="JSON reply value")
    reply_value.add_argument("--value-file", default=None, help="Path to JSON reply file")

    p_stop = sub.add_parser("stop", help="Request cooperative stop for a job")
    p_stop.add_argument("job_id", help="Job UUID")

    p_inspect = sub.add_parser("inspect", help="Inspect job status without blocking")
    p_inspect.add_argument("job_id", help="Job UUID")

    p_mcp = sub.add_parser("mcp", help="Run the per-session stdio MCP server")
    p_mcp.add_argument("--session-dir", type=Path, required=True, help="Session directory path")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch a `stew` CLI invocation and return its process exit code (`0` or `1`).

    Imports heavy engine/SQLite/LangGraph modules lazily so `stew mcp` and `stew workflows`
    start cleanly without loading the runtime or database stack.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "mcp":
        from castle_steward_engine.mcp_server import serve_stdio

        serve_stdio(args.session_dir)
        return 0

    try:
        if args.command == "workflows":
            payload = [
                {"name": wf.name, "version": wf.version}
                for _, wf in sorted(load_workflows().items())
            ]
            print(json.dumps(payload, indent=2))
            return 0

        from castle_steward_engine.agent_layer import AgentLayer
        from castle_steward_engine.engine import Engine
        from castle_steward_engine.store import default_state_dir

        settings = EngineSettings()
        state_dir = args.state_dir or settings.state_dir or default_state_dir()
        with Engine(
            state_dir=state_dir,
            agent_layer=AgentLayer(settings.agent_layer_executable),
        ) as engine:
            if args.command == "start":
                raw_input = _load_json_arg(args.input_json, args.input_file)
                job_id = engine.start(args.workflow, raw_input)
                print(json.dumps({"job_id": str(job_id)}, indent=2))
            elif args.command == "run":
                print(engine.run(UUID(args.job_id)).model_dump_json(indent=2))
            elif args.command == "reply":
                val = _load_json_arg(args.value_json, args.value_file)
                engine.reply(UUID(args.job_id), args.wait_id, val)
                print(
                    json.dumps(
                        {"ok": True, "job_id": args.job_id, "wait_id": args.wait_id}, indent=2
                    )
                )
            elif args.command == "stop":
                engine.stop(UUID(args.job_id))
                print(json.dumps({"ok": True, "job_id": args.job_id}, indent=2))
            elif args.command == "inspect":
                print(engine.inspect(UUID(args.job_id)).model_dump_json(indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 1
