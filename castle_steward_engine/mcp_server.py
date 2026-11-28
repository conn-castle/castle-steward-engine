"""Per-session stdio MCP server (session files + completion checks; never touches SQLite)."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp import types as mcp_types
from mcp.server import Server
from mcp.server.stdio import stdio_server
from pydantic import (
    BaseModel,
    ConfigDict,
    JsonValue,
    StringConstraints,
    ValidationError,
    model_validator,
)

from castle_steward_engine.workflow import (
    ASK_TOOL_NAME,
    CHECK_DONE_TOOL_NAME,
    Role,
    load_workflow,
)

_NonBlankStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

_SESSION_META_FILENAME = "session.json"
_MCP_CONFIG_FILENAME = "mcp-config.json"
_ACTIVE_TURN_FILENAME = "active_turn.json"
_TURNS_DIRNAME = "turns"
_PROMPT_FILENAME = "prompt.md"
_QUESTION_FILENAME = "question.json"
_HISTORY_FILENAME = "history.jsonl"
SUBMISSION_FILENAME = "submission.json"
_MCP_SERVER_NAME = "castle-steward"


class SessionMeta(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    workflow: str
    role: str


class ActiveTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    turn_key: str
    attempt: int
    expects_tool: str | None = None
    expects_check: str | None = None
    context: dict[str, JsonValue]
    turn_dir: str

    @model_validator(mode="after")
    def _require_exactly_one_expectation(self) -> ActiveTurn:
        if (self.expects_tool is None) == (self.expects_check is None):
            raise ValueError("ActiveTurn must set exactly one of expects_tool or expects_check")
        return self


class AskQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    question: _NonBlankStr


class _EmptyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolCallResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    is_error: bool = False


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically write UTF-8 text to `path` via a sibling temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def _safe_turn_dirname(turn_key: str, attempt: int) -> str:
    return f"{re.sub(r'[^A-Za-z0-9_.-]', '_', turn_key)}_a{attempt}"


class SessionDir:
    """Stateless filesystem protocol for one agent session directory.

    Holds only `self.path` and caches no mutable turn state in RAM because the engine
    process and the child stdio MCP server process communicate strictly through atomic
    disk files (`session.json`, `mcp-config.json`, `active_turn.json`, and turn files).
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    @property
    def meta_path(self) -> Path:
        return self.path / _SESSION_META_FILENAME

    @property
    def mcp_config_path(self) -> Path:
        return self.path / _MCP_CONFIG_FILENAME

    @property
    def active_turn_path(self) -> Path:
        return self.path / _ACTIVE_TURN_FILENAME

    def init(self, *, workflow: str, role: str) -> Path:
        """Create `session.json` and `mcp-config.json` inside `self.path`."""
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / _TURNS_DIRNAME).mkdir(exist_ok=True)
        meta = SessionMeta(workflow=workflow, role=role)
        atomic_write_text(self.meta_path, meta.model_dump_json(indent=2))
        mcp_env = {"PATH": os.environ.get("PATH", "")}
        if pythonpath := os.environ.get("PYTHONPATH"):
            mcp_env["PYTHONPATH"] = pythonpath
        mcp_config = {
            "mcpServers": {
                _MCP_SERVER_NAME: {
                    "command": sys.executable,
                    "args": ["-m", "castle_steward_engine", "mcp", "--session-dir", str(self.path)],
                    "env": mcp_env,
                }
            }
        }
        atomic_write_text(self.mcp_config_path, json.dumps(mcp_config, indent=2))
        return self.mcp_config_path

    def prepare_turn(
        self,
        *,
        turn_key: str,
        attempt: int,
        expects_tool: str | None = None,
        expects_check: str | None = None,
        context: dict[str, JsonValue],
        prompt: str,
    ) -> tuple[Path, Path]:
        """Create turn directory, write `prompt.md`, and atomically update `active_turn.json`."""
        turn_dir = self.path / _TURNS_DIRNAME / _safe_turn_dirname(turn_key, attempt)
        turn_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = turn_dir / _PROMPT_FILENAME
        atomic_write_text(prompt_path, prompt)
        self.activate_turn(
            turn_key=turn_key,
            attempt=attempt,
            expects_tool=expects_tool,
            expects_check=expects_check,
            context=context,
            turn_dir=turn_dir,
        )
        return turn_dir, prompt_path

    def activate_turn(
        self,
        *,
        turn_key: str,
        attempt: int,
        expects_tool: str | None = None,
        expects_check: str | None = None,
        context: dict[str, JsonValue],
        turn_dir: Path,
    ) -> None:
        """Atomically replace `active_turn.json` pointing at `turn_dir`."""
        active = ActiveTurn(
            turn_key=turn_key,
            attempt=attempt,
            expects_tool=expects_tool,
            expects_check=expects_check,
            context=context,
            turn_dir=str(turn_dir),
        )
        atomic_write_text(self.active_turn_path, active.model_dump_json(indent=2))

    def read_meta(self) -> SessionMeta:
        """Read and validate `session.json`."""
        return SessionMeta.model_validate_json(self.meta_path.read_text(encoding="utf-8"))

    def read_active_turn(self) -> ActiveTurn:
        """Read and validate `active_turn.json`."""
        return ActiveTurn.model_validate_json(self.active_turn_path.read_text(encoding="utf-8"))

    @staticmethod
    def read_outcome[R: BaseModel](
        turn_dir: Path,
        *,
        expects: type[R],
        context: dict[str, JsonValue],
        check_fn: Callable[[dict[str, JsonValue]], BaseModel] | None = None,
    ) -> (
        tuple[Literal["question"], str]
        | tuple[Literal["submission"], R]
        | tuple[Literal["missing"], None]
    ):
        """Read and validate the outcome artifacts recorded in `turn_dir`."""
        q_path = turn_dir / _QUESTION_FILENAME
        if q_path.exists():
            asked = AskQuestion.model_validate_json(q_path.read_text(encoding="utf-8"))
            return ("question", asked.question)
        s_path = turn_dir / SUBMISSION_FILENAME
        if not s_path.exists():
            return ("missing", None)
        raw = json.loads(s_path.read_text(encoding="utf-8"))
        if check_fn is None:
            result = expects.model_validate(raw, context=context)
            return ("submission", result)
        recorded = expects.model_validate(raw)
        live_raw = check_fn(context)
        live = expects.model_validate(live_raw.model_dump())
        if live != recorded:
            raise ValueError(
                f"Live state changed after {CHECK_DONE_TOOL_NAME}: "
                f"recorded {recorded.model_dump()}, live {live.model_dump()}"
            )
        return ("submission", live)


def format_validation_error(exc: ValidationError) -> str:
    """Render concise, actionable field-level Pydantic validation feedback."""
    lines = [
        f"- {'.'.join(str(part) for part in err['loc']) or '(root)'}: {err['msg']}"
        for err in exc.errors()
    ]
    return "\n".join(lines)


class SessionToolHost:
    """Fixed per-session MCP tool registry reading `active_turn.json` on every call."""

    def __init__(self, session_dir: Path, role: Role) -> None:
        self._session = SessionDir(session_dir)
        self.role = role
        self._models_by_tool = role.result_models_by_tool()
        self._tools = self._build_tools()

    @classmethod
    def from_session_dir(cls, session_dir: Path) -> SessionToolHost:
        meta = SessionDir(session_dir).read_meta()
        workflow = load_workflow(meta.workflow)
        if meta.role not in workflow.roles:
            raise KeyError(f"Role {meta.role!r} not found in workflow {meta.workflow!r}")
        return cls(session_dir, workflow.roles[meta.role])

    def _build_tools(self) -> list[mcp_types.Tool]:
        tools = [
            mcp_types.Tool(
                name=tool_name,
                description=(model.__doc__ or f"Submit {model.__name__} result.").strip(),
                input_schema=model.model_json_schema(),
            )
            for tool_name, model in self._models_by_tool.items()
        ]
        if self.role.can_ask_human:
            tools.append(
                mcp_types.Tool(
                    name=ASK_TOOL_NAME,
                    description="Pause the turn and ask the human a clarifying question.",
                    input_schema=AskQuestion.model_json_schema(),
                )
            )
        tools.append(
            mcp_types.Tool(
                name=CHECK_DONE_TOOL_NAME,
                description=(
                    "Verify whether this turn is complete (valid submission, passing completion "
                    "check, or pending question)."
                ),
                input_schema=_EmptyArgs.model_json_schema(),
            )
        )
        return tools

    def list_tools(self) -> list[mcp_types.Tool]:
        return list(self._tools)

    def _append_history(self, turn_dir: Path, *, tool: str, ok: bool, detail: str) -> None:
        entry = {"ts": datetime.now(UTC).isoformat(), "tool": tool, "ok": ok, "detail": detail}
        with (turn_dir / _HISTORY_FILENAME).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def _fail(self, turn_dir: Path, *, tool: str, msg: str) -> ToolCallResult:
        self._append_history(turn_dir, tool=tool, ok=False, detail=msg)
        return ToolCallResult(text=msg, is_error=True)

    def _ok(self, turn_dir: Path, *, tool: str, detail: str, text: str) -> ToolCallResult:
        self._append_history(turn_dir, tool=tool, ok=True, detail=detail)
        return ToolCallResult(text=text)

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> ToolCallResult:
        args = {} if arguments is None else arguments
        turn = self._session.read_active_turn()
        turn_dir = Path(turn.turn_dir)

        if name == CHECK_DONE_TOOL_NAME:
            return self._handle_check_done(turn, turn_dir, args)
        if name == ASK_TOOL_NAME:
            return self._handle_ask(turn_dir, args)
        if name in self._models_by_tool:
            return self._handle_submission(turn, turn_dir, name, args)
        valid = ", ".join(tool.name for tool in self._tools)
        return self._fail(
            turn_dir, tool=name, msg=f"Unknown tool '{name}'. Available tools: {valid}."
        )

    def _handle_ask(self, turn_dir: Path, args: dict[str, Any]) -> ToolCallResult:
        if not self.role.can_ask_human:
            return self._fail(
                turn_dir,
                tool=ASK_TOOL_NAME,
                msg=f"Tool '{ASK_TOOL_NAME}' is not enabled for role '{self.role.name}'.",
            )
        if (turn_dir / SUBMISSION_FILENAME).exists():
            return self._fail(
                turn_dir,
                tool=ASK_TOOL_NAME,
                msg=f"This turn is already marked complete; cannot also call '{ASK_TOOL_NAME}'.",
            )
        try:
            asked = AskQuestion.model_validate(args)
        except ValidationError as exc:
            return self._fail(
                turn_dir,
                tool=ASK_TOOL_NAME,
                msg=f"Invalid '{ASK_TOOL_NAME}' arguments:\n{format_validation_error(exc)}",
            )
        atomic_write_text(turn_dir / _QUESTION_FILENAME, asked.model_dump_json(indent=2))
        return self._ok(
            turn_dir,
            tool=ASK_TOOL_NAME,
            detail=asked.question,
            text=(
                f"Question recorded. Call {CHECK_DONE_TOOL_NAME} and end your turn to wait for a "
                "reply."
            ),
        )

    def _handle_submission(
        self, turn: ActiveTurn, turn_dir: Path, name: str, args: dict[str, Any]
    ) -> ToolCallResult:
        if name != turn.expects_tool:
            ask_hint = f" or '{ASK_TOOL_NAME}'" if self.role.can_ask_human else ""
            if turn.expects_tool is not None:
                msg = (
                    f"Tool '{name}' is not active for this turn. "
                    f"Call '{turn.expects_tool}'{ask_hint} instead."
                )
            else:
                msg = (
                    f"Tool '{name}' is not active for this turn. "
                    f"Call '{CHECK_DONE_TOOL_NAME}'{ask_hint} when finished."
                )
            return self._fail(turn_dir, tool=name, msg=msg)
        if (turn_dir / _QUESTION_FILENAME).exists():
            return self._fail(
                turn_dir,
                tool=name,
                msg=(
                    "A question is already recorded for this turn; "
                    "end your turn to wait for a reply instead of submitting."
                ),
            )
        model = self._models_by_tool[name]
        try:
            validated = model.model_validate(args, context=turn.context)
        except ValidationError as exc:
            return self._fail(
                turn_dir,
                tool=name,
                msg=f"Validation failed for '{name}':\n{format_validation_error(exc)}",
            )
        atomic_write_text(turn_dir / SUBMISSION_FILENAME, validated.model_dump_json(indent=2))
        return self._ok(
            turn_dir,
            tool=name,
            detail="accepted",
            text=f"Accepted '{name}'. Call {CHECK_DONE_TOOL_NAME} before ending your turn.",
        )

    def _handle_check_only_done(self, turn: ActiveTurn, turn_dir: Path) -> ToolCallResult:
        assert turn.expects_check is not None
        if (turn_dir / _QUESTION_FILENAME).exists():
            return ToolCallResult(
                text="Waiting for human input (question recorded). You may end your turn now."
            )
        check_fn = self.role.checks.get(turn.expects_check)
        if check_fn is None:
            return self._fail(
                turn_dir,
                tool=CHECK_DONE_TOOL_NAME,
                msg=f"Expected check '{turn.expects_check}' is not registered on this role.",
            )
        (turn_dir / SUBMISSION_FILENAME).unlink(missing_ok=True)
        try:
            raw_result = check_fn(turn.context)
        except (ValueError, ValidationError) as exc:
            detail = format_validation_error(exc) if isinstance(exc, ValidationError) else str(exc)
            return self._fail(
                turn_dir,
                tool=CHECK_DONE_TOOL_NAME,
                msg=f"Incomplete: {detail}\nFix this, then call '{CHECK_DONE_TOOL_NAME}' again.",
            )
        atomic_write_text(turn_dir / SUBMISSION_FILENAME, raw_result.model_dump_json(indent=2))
        return self._ok(
            turn_dir,
            tool=CHECK_DONE_TOOL_NAME,
            detail=f"check:{turn.expects_check}",
            text=f"Complete: '{turn.expects_check}' check passed. You may end your turn now.",
        )

    def _handle_tool_check_done(self, turn: ActiveTurn, turn_dir: Path) -> ToolCallResult:
        assert turn.expects_tool is not None
        model = self._models_by_tool.get(turn.expects_tool)
        if model is None:
            return self._fail(
                turn_dir,
                tool=CHECK_DONE_TOOL_NAME,
                msg=f"Expected tool '{turn.expects_tool}' is not registered on this role.",
            )
        try:
            kind, _payload = self._session.read_outcome(
                turn_dir, expects=model, context=turn.context
            )
        except (json.JSONDecodeError, ValidationError) as exc:
            detail = format_validation_error(exc) if isinstance(exc, ValidationError) else str(exc)
            return self._fail(
                turn_dir,
                tool=CHECK_DONE_TOOL_NAME,
                msg=(
                    f"Incomplete: your submission is no longer valid:\n{detail}\n"
                    f"Fix this, then call '{turn.expects_tool}' again."
                ),
            )
        if kind == "question":
            return ToolCallResult(
                text="Waiting for human input (question recorded). You may end your turn now."
            )
        if kind == "submission":
            return ToolCallResult(
                text=(
                    f"Complete: '{turn.expects_tool}' submission is valid. "
                    "You may end your turn now."
                )
            )
        ask_hint = f" or '{ASK_TOOL_NAME}'" if self.role.can_ask_human else ""
        return ToolCallResult(
            text=f"Incomplete: call '{turn.expects_tool}'{ask_hint} before ending your turn."
        )

    def _handle_check_done(
        self, turn: ActiveTurn, turn_dir: Path, args: dict[str, Any]
    ) -> ToolCallResult:
        try:
            _EmptyArgs.model_validate(args)
        except ValidationError as exc:
            return self._fail(
                turn_dir,
                tool=CHECK_DONE_TOOL_NAME,
                msg=f"Invalid '{CHECK_DONE_TOOL_NAME}' arguments:\n{format_validation_error(exc)}",
            )
        if turn.expects_check is not None:
            return self._handle_check_only_done(turn, turn_dir)
        return self._handle_tool_check_done(turn, turn_dir)

    async def handle_list_tools(
        self, _ctx: Any, _params: mcp_types.PaginatedRequestParams | None
    ) -> mcp_types.ListToolsResult:
        return mcp_types.ListToolsResult(tools=self.list_tools())

    async def handle_call_tool(
        self, _ctx: Any, params: mcp_types.CallToolRequestParams
    ) -> mcp_types.CallToolResult:
        result = self.call_tool(params.name, params.arguments)
        return mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text=result.text)],
            is_error=result.is_error,
        )


def serve_stdio(session_dir: Path) -> None:
    """Run the per-session MCP stdio server until stdin closes."""

    async def _run() -> None:
        host = SessionToolHost.from_session_dir(session_dir)
        server = Server(
            _MCP_SERVER_NAME,
            on_list_tools=host.handle_list_tools,
            on_call_tool=host.handle_call_tool,
        )
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    asyncio.run(_run())
