"""Workflow engine, private LangGraph compiler, and step/agent/wait runners."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, cast
from uuid import UUID, uuid4

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphDrained
from langgraph.graph import END as LG_END
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import RunControl, Runtime
from langgraph.types import Command, Send, interrupt
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    TypeAdapter,
    ValidationError,
)

from castle_steward_engine.agent_layer import (
    AgentLayer,
    AgentLayerClient,
    NoProviderConversationError,
)
from castle_steward_engine.mcp_server import SUBMISSION_FILENAME, SessionDir
from castle_steward_engine.store import (
    SQLITE_BUSY_TIMEOUT_SECONDS,
    SessionRecord,
    Store,
    TurnRecord,
    TurnStatus,
)
from castle_steward_engine.workflow import (
    ASK_TOOL_NAME,
    CHECK_DONE_TOOL_NAME,
    END,
    AgentNode,
    BranchEdge,
    JobInfo,
    StepNode,
    Turn,
    WaitNode,
    Workflow,
    load_workflow,
)

Phase = Literal["pending", "running", "paused", "stopping", "stopped", "completed", "failed"]
_NonBlankStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_JSON_ADAPTER: TypeAdapter[Any] = TypeAdapter(Any)


class EngineError(RuntimeError):
    """Base error for workflow execution issues."""


class WorkflowVersionMismatchError(EngineError):
    """Raised when a job's recorded workflow version differs from the installed workflow."""


class TurnFailed(EngineError):
    """Raised when an agent turn terminates with status other than 'completed'."""


class IncompleteWork(EngineError):
    """Raised when an agent finishes without calling its expected submission tool after a nudge."""


class UnknownWaitError(EngineError):
    """Raised when replying to a wait ID that is not currently pending."""


class PendingWait(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    wait_id: str
    kind: str
    prompt: str
    node: str
    payload: dict[str, JsonValue] = Field(default_factory=dict)
    answered: bool = False


class ActiveTurnInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    turn_key: str
    attempt: int
    node: str
    session_name: str
    al_handle: str
    al_invocation_id: str
    status: TurnStatus


class JobStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    job_id: UUID
    workflow: str
    workflow_version: str
    phase: Phase
    last_error: str | None = None
    next_nodes: list[str] = Field(default_factory=list)
    pending_waits: list[PendingWait] = Field(default_factory=list)
    active_turns: list[ActiveTurnInfo] = Field(default_factory=list)
    state: dict[str, JsonValue] | None = None


class _AskReply(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    answer: _NonBlankStr


@dataclass(frozen=True)
class _RunContext:
    engine: Engine
    workflow: Workflow
    job_id: UUID
    artifacts_dir: Path


class _StoreRunControl(RunControl):
    """Cooperative drain signal backed by `jobs.stop_seq > jobs.ack_seq` in SQLite."""

    def __init__(self, store: Store, job_id: UUID) -> None:
        super().__init__()
        self._store = store
        self._job_id = job_id

    @property
    def drain_requested(self) -> bool:
        return super().drain_requested or self._store.is_stop_pending(self._job_id)

    @property
    def drain_reason(self) -> str | None:
        if self._store.is_stop_pending(self._job_id):
            return "stop_requested"
        return super().drain_reason


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _check_drain(runtime: Runtime[_RunContext]) -> None:
    if runtime.drain_requested:
        raise GraphDrained(runtime.drain_reason or "stop_requested")


def _validate_ask_reply(value: Any) -> str:
    if isinstance(value, str):
        return _AskReply(answer=value).answer
    return _AskReply.model_validate(value).answer


def _dump_json_state(values: Any) -> dict[str, JsonValue] | None:
    if not values:
        return None
    dumped = _JSON_ADAPTER.dump_python(values, mode="json")
    return dumped if isinstance(dumped, dict) else None


def _require_session(store: Store, job_id: UUID, session_name: str, turn_key: str) -> SessionRecord:
    session = store.get_session(job_id, session_name)
    if session is None:
        raise EngineError(f"Missing session row {session_name!r} for turn {turn_key}")
    return session


def _check_session_match(session: SessionRecord, node: AgentNode, turn: Turn) -> None:
    if turn.agent is not None and turn.agent != session.agent:
        raise EngineError(
            f"Session {session.session_name!r} uses {session.agent}, cannot switch to {turn.agent}"
        )
    if session.role_name != node.role.name:
        raise EngineError(
            f"Session {session.session_name!r} has role {session.role_name!r}, "
            f"cannot run node {node.name!r} with role {node.role.name!r}"
        )


def _node_turn_expectations(node: AgentNode) -> tuple[str | None, str | None]:
    if node.check is not None:
        return None, node.check
    return node.role.tool_name(node.expects), None


def _missing_outcome_nudge(expects_tool: str | None) -> str:
    if expects_tool is not None:
        return (
            f"Your turn ended without calling {expects_tool}. "
            f"Continue until you are done, as verified by {CHECK_DONE_TOOL_NAME}."
        )
    return (
        f"Your turn ended without verifying completion with {CHECK_DONE_TOOL_NAME}. "
        f"Continue until you are done, as verified by {CHECK_DONE_TOOL_NAME}."
    )


def _prepare_reserved_turn(
    ctx: _RunContext,
    node: AgentNode,
    turn: Turn,
    *,
    session_dir: SessionDir,
    turn_key: str,
    attempt: int,
    prompt: str,
    al_handle: str,
    al_invocation_id: str,
    created_at: str | None = None,
) -> TurnRecord:
    expects_tool, expects_check = _node_turn_expectations(node)
    _, prompt_path = session_dir.prepare_turn(
        turn_key=turn_key,
        attempt=attempt,
        expects_tool=expects_tool,
        expects_check=expects_check,
        context=turn.context,
        prompt=prompt,
    )
    return TurnRecord(
        job_id=ctx.job_id,
        turn_key=turn_key,
        attempt=attempt,
        node=node.name,
        session_name=turn.session,
        al_handle=al_handle,
        al_invocation_id=al_invocation_id,
        status="reserved",
        prompt_path=str(prompt_path),
        created_at=created_at or _utc_now(),
    )


def _reserve_new_session(
    ctx: _RunContext,
    node: AgentNode,
    turn: Turn,
    *,
    turn_key: str,
    prompt: str,
) -> tuple[SessionRecord, TurnRecord]:
    if turn.agent is None:
        raise EngineError(
            f"New session {turn.session!r} on node {node.name!r} requires an AgentSpec"
        )
    assert turn.workdir is not None
    store = ctx.engine.store
    res = ctx.engine.agent_layer.reserve_start(turn.workdir)
    session_id = str(uuid4())
    session_dir = SessionDir(store.session_dir(session_id))
    session_dir.init(workflow=ctx.workflow.name, role=node.role.name)
    now = _utc_now()
    session_rec = SessionRecord(
        job_id=ctx.job_id,
        session_name=turn.session,
        session_id=session_id,
        role_name=node.role.name,
        al_handle=res.handle,
        agent_json=turn.agent.model_dump_json(),
        workdir=str(turn.workdir),
        created_at=now,
    )
    turn_rec = _prepare_reserved_turn(
        ctx,
        node,
        turn,
        session_dir=session_dir,
        turn_key=turn_key,
        attempt=1,
        prompt=prompt,
        al_handle=res.handle,
        al_invocation_id=res.invocation_id,
        created_at=now,
    )
    store.create_session_and_turn(session_rec, turn_rec)
    return session_rec, turn_rec


def _reserve_existing_session(
    ctx: _RunContext,
    node: AgentNode,
    turn: Turn,
    session: SessionRecord,
    *,
    turn_key: str,
    prompt: str,
    prev: TurnRecord | None,
) -> tuple[SessionRecord, TurnRecord]:
    _check_session_match(session, node, turn)
    store = ctx.engine.store
    al = ctx.engine.agent_layer
    if prev is None:
        attempt, turn_prompt = 1, prompt
    else:
        attempt, turn_prompt = prev.attempt + 1, prev.prompt_file.read_text(encoding="utf-8")
    replaced_handle = False
    try:
        res = al.reserve_continue(session.workdir_path, session.al_handle)
    except NoProviderConversationError:
        if prev is None or not store.is_first_turn_of_session(ctx.job_id, turn.session, turn_key):
            raise
        # Keep immutable session_id/session.json/mcp-config.json; replace only al_handle.
        res = al.reserve_start(session.workdir_path)
        replaced_handle = True
    session_dir = SessionDir(store.session_dir(session.session_id))
    turn_rec = _prepare_reserved_turn(
        ctx,
        node,
        turn,
        session_dir=session_dir,
        turn_key=turn_key,
        attempt=attempt,
        prompt=turn_prompt,
        al_handle=res.handle,
        al_invocation_id=res.invocation_id,
    )
    if replaced_handle:
        updated = store.update_session_handle_and_create_turn(
            ctx.job_id, turn.session, res.handle, turn_rec
        )
        return updated, turn_rec
    store.create_turn(turn_rec)
    return session, turn_rec


def _reserve_turn(
    ctx: _RunContext,
    node: AgentNode,
    turn: Turn,
    *,
    turn_key: str,
    prompt: str,
    prev: TurnRecord | None,
) -> tuple[SessionRecord, TurnRecord]:
    """Reserve a new, continued, or retried turn and persist its files + SQLite row."""
    existing = ctx.engine.store.get_session(ctx.job_id, turn.session)
    if existing is None:
        if prev is not None:
            raise EngineError(f"Missing session row {turn.session!r} for turn {turn_key}")
        return _reserve_new_session(ctx, node, turn, turn_key=turn_key, prompt=prompt)
    return _reserve_existing_session(
        ctx, node, turn, existing, turn_key=turn_key, prompt=prompt, prev=prev
    )


def _wait_and_finalize(
    store: Store,
    al: AgentLayerClient,
    workdir: Path,
    node_name: str,
    turn: TurnRecord,
) -> Path:
    outcome = al.wait(workdir, turn.al_invocation_id)
    final_status: TurnStatus = (
        "completed"
        if outcome.state == "completed"
        else ("cancelled" if outcome.state == "cancelled" else "failed")
    )
    store.update_turn_status(turn.job_id, turn.turn_key, turn.attempt, final_status, completed=True)
    if final_status != "completed":
        detail = f": {outcome.error}" if outcome.error else ""
        raise TurnFailed(
            f"Turn {turn.turn_key} on node {node_name!r} ended with status {final_status!r}{detail}"
        )
    return turn.prompt_file.parent


def _execute_turn(
    ctx: _RunContext,
    node: AgentNode,
    initial_turn: Turn,
    *,
    turn_key: str,
    prompt: str,
) -> Path:
    """Drive one `(job_id, turn_key)` to `completed` and return its turn directory."""
    store = ctx.engine.store
    al = ctx.engine.agent_layer
    job_id = ctx.job_id
    session_name = initial_turn.session
    prev = store.get_latest_turn(job_id, turn_key)

    if prev is not None and prev.status == "completed":
        return prev.prompt_file.parent
    if prev is not None and prev.status == "running":
        session = _require_session(store, job_id, session_name, turn_key)
        return _wait_and_finalize(store, al, session.workdir_path, node.name, prev)

    if prev is not None and prev.status == "reserved":
        session = _require_session(store, job_id, session_name, turn_key)
        turn = prev
        expects_tool, expects_check = _node_turn_expectations(node)
        SessionDir(store.session_dir(session.session_id)).activate_turn(
            turn_key=turn_key,
            attempt=turn.attempt,
            expects_tool=expects_tool,
            expects_check=expects_check,
            context=initial_turn.context,
            turn_dir=turn.prompt_file.parent,
        )
    elif prev is None or prev.status in ("failed", "cancelled"):
        session, turn = _reserve_turn(
            ctx, node, initial_turn, turn_key=turn_key, prompt=prompt, prev=prev
        )
    else:
        raise EngineError(f"Unexpected turn status {prev.status!r} for {turn_key}")

    if store.is_initial_session_turn(job_id, turn.al_handle, turn.turn_key, turn.attempt):
        inv = al.start(
            session.workdir_path,
            reservation_handle=turn.al_handle,
            invocation_id=turn.al_invocation_id,
            agent=session.agent,
            prompt_file=turn.prompt_file,
            mcp_config=SessionDir(store.session_dir(session.session_id)).mcp_config_path,
            role=node.role.name,
        )
    else:
        inv = al.continue_session(
            session.workdir_path,
            handle=turn.al_handle,
            reservation_id=turn.al_invocation_id,
            prompt_file=turn.prompt_file,
        )
    running_turn = store.update_turn_status(
        job_id,
        turn.turn_key,
        turn.attempt,
        "running",
        al_handle=inv.handle,
        al_invocation_id=inv.invocation_id,
    )
    return _wait_and_finalize(store, al, session.workdir_path, node.name, running_turn)


def _mark_turn_failed(ctx: _RunContext, turn_key: str) -> None:
    latest = ctx.engine.store.get_latest_turn(ctx.job_id, turn_key)
    if latest is not None:
        ctx.engine.store.update_turn_status(ctx.job_id, turn_key, latest.attempt, "failed")


def _run_agent_node(node: AgentNode, state: Any, runtime: Runtime[_RunContext]) -> dict[str, Any]:
    ctx = runtime.context
    assert runtime.execution_info is not None
    task_id = runtime.execution_info.task_id

    raw_turn = node.ask(state)
    raw_workdir = raw_turn.workdir or ctx.engine.default_workdir
    if raw_workdir is None:
        raise EngineError(
            f"Node {node.name!r} requires Turn.workdir or Engine(default_workdir=...)"
        )
    initial_turn = replace(raw_turn, workdir=raw_workdir.resolve())
    expects_tool, expects_check = _node_turn_expectations(node)
    check_fn = node.role.check_fn(expects_check) if expects_check is not None else None

    round_index = 0
    prompt = initial_turn.prompt
    nudged = False

    while True:
        _check_drain(runtime)
        turn_key = f"{task_id}:{round_index}"
        turn_dir = _execute_turn(ctx, node, initial_turn, turn_key=turn_key, prompt=prompt)

        try:
            kind, payload = SessionDir.read_outcome(
                turn_dir, expects=node.expects, context=initial_turn.context, check_fn=check_fn
            )
        except (ValueError, ValidationError) as exc:
            if check_fn is None:
                raise
            (turn_dir / SUBMISSION_FILENAME).unlink(missing_ok=True)
            if not nudged:
                nudged = True
                prompt = (
                    f"Your turn ended with an invalid state after {CHECK_DONE_TOOL_NAME}: {exc}\n"
                    f"Fix this and confirm with {CHECK_DONE_TOOL_NAME}."
                )
                round_index += 1
                continue
            _mark_turn_failed(ctx, turn_key)
            raise IncompleteWork(
                f"Node {node.name!r} ended with invalid state after {CHECK_DONE_TOOL_NAME} "
                f"nudge: {exc}"
            ) from exc

        if kind == "question":
            raw_answer = interrupt(
                {
                    "kind": ASK_TOOL_NAME,
                    "prompt": payload,
                    "node": node.name,
                    "session": initial_turn.session,
                }
            )
            answer = _validate_ask_reply(raw_answer)
            prompt = f"Human reply to your question:\n\n{answer}"
            round_index += 1
            continue
        if kind == "submission":
            return node.then(state, payload)

        if not nudged:
            nudged = True
            prompt = _missing_outcome_nudge(expects_tool)
            round_index += 1
            continue
        _mark_turn_failed(ctx, turn_key)
        target = expects_tool or CHECK_DONE_TOOL_NAME
        raise IncompleteWork(f"Node {node.name!r} ended without calling {target} after nudge.")


def _build_field_validators(state_cls: type[BaseModel]) -> dict[str, TypeAdapter[Any]]:
    return {
        name: TypeAdapter(cast(Any, Annotated)[field.annotation, field])
        for name, field in state_cls.model_fields.items()
    }


def _validate_state_update(
    node_name: str, validators: dict[str, TypeAdapter[Any]], update: Any
) -> dict[str, Any]:
    """Validate each returned field against its `FieldInfo`.

    Nested models that configure `revalidate_instances="always"` are also revalidated.
    """
    if not isinstance(update, dict):
        raise EngineError(
            f"Node {node_name!r} must return a dict state update, got {type(update).__name__}"
        )
    if unknown := sorted(str(k) for k in set(update) - set(validators)):
        raise EngineError(f"Node {node_name!r} returned unknown state key(s): {', '.join(unknown)}")
    validated: dict[str, Any] = {}
    for key, value in update.items():
        try:
            validated[key] = validators[key].validate_python(value)
        except ValidationError as exc:
            raise EngineError(
                f"Node {node_name!r} returned invalid value for state key {key!r}: {exc}"
            ) from exc
    return validated


def _compile_workflow(
    workflow: Workflow, checkpointer: SqliteSaver
) -> CompiledStateGraph[Any, _RunContext, Any, Any]:
    graph = workflow.graph
    builder: StateGraph[Any, _RunContext, Any, Any] = StateGraph(
        workflow.state, context_schema=_RunContext
    )
    fan_out_inputs = {fo.dst: fo.item_type for fo in graph.fan_outs}
    validators = _build_field_validators(workflow.state)

    for node in graph.nodes.values():
        input_schema = fan_out_inputs.get(node.name)
        if isinstance(node, StepNode):
            fn = _make_step_runner(node, validators)
        elif isinstance(node, WaitNode):
            fn = _make_wait_runner(node, validators)
        else:
            fn = _make_agent_runner(node, validators)
        if input_schema is not None:
            builder.add_node(node.name, cast(Any, fn), input_schema=cast(Any, input_schema))
        else:
            builder.add_node(node.name, cast(Any, fn))

    assert graph.entry_node is not None
    builder.add_edge(START, graph.entry_node)

    for src, dst in graph.edges:
        builder.add_edge(src, LG_END if dst == END else dst)

    for branch in graph.branches:
        path_map = {
            key: (LG_END if target == END else target) for key, target in branch.targets.items()
        }
        builder.add_conditional_edges(branch.src, _make_branch_router(branch), cast(Any, path_map))

    for fo in graph.fan_outs:
        builder.add_conditional_edges(
            fo.src, _make_fan_out_router(fo.src, fo.dst, fo.items), [fo.dst]
        )

    return builder.compile(checkpointer=checkpointer)


def _make_step_runner(
    node: StepNode, validators: dict[str, TypeAdapter[Any]]
) -> Callable[[Any, Runtime[_RunContext]], dict[str, Any]]:
    def _run(state: Any, runtime: Runtime[_RunContext]) -> dict[str, Any]:
        _check_drain(runtime)
        ctx = runtime.context
        job_info = JobInfo(
            job_id=ctx.job_id,
            artifacts_dir=ctx.artifacts_dir,
            _options_fn=ctx.engine.agent_layer.options,
            _quota_fn=ctx.engine.agent_layer.quota,
            _check_stop_fn=lambda: _check_drain(runtime),
        )
        return _validate_state_update(node.name, validators, node.fn(state, job_info))

    return _run


def _make_wait_runner(
    node: WaitNode, validators: dict[str, TypeAdapter[Any]]
) -> Callable[[Any, Runtime[_RunContext]], dict[str, Any]]:
    def _run(state: Any, runtime: Runtime[_RunContext]) -> dict[str, Any]:
        _check_drain(runtime)
        wait_spec = node.ask(state)
        raw_reply = interrupt(
            {
                "kind": wait_spec.kind,
                "prompt": wait_spec.prompt,
                "payload": wait_spec.payload,
                "node": node.name,
            }
        )
        return _validate_state_update(
            node.name, validators, node.then(state, node.reply.model_validate(raw_reply))
        )

    return _run


def _make_agent_runner(
    node: AgentNode, validators: dict[str, TypeAdapter[Any]]
) -> Callable[[Any, Runtime[_RunContext]], dict[str, Any]]:
    def _run(state: Any, runtime: Runtime[_RunContext]) -> dict[str, Any]:
        _check_drain(runtime)
        return _validate_state_update(node.name, validators, _run_agent_node(node, state, runtime))

    return _run


def _make_branch_router(branch: BranchEdge) -> Callable[[Any], str]:
    # Engine owns the concrete workflow.state schema; avoid exposing branch.route's annotations.
    def _route(state: Any) -> str:
        key = branch.route(state)
        if key not in branch.targets:
            valid = ", ".join(sorted(branch.targets))
            raise EngineError(
                f"Branch on {branch.src!r} returned unknown route key {key!r} "
                f"(valid targets: {valid})"
            )
        return key

    return _route


def _make_fan_out_router(
    src: str, dst: str, items_fn: Callable[[Any], Sequence[Any]]
) -> Callable[[Any], list[Send]]:
    def _route(state: Any) -> list[Send]:
        items = list(items_fn(state))
        if not items:
            raise EngineError(f"Fan-out from {src!r} to {dst!r} produced no items")
        return [Send(dst, item) for item in items]

    return _route


def open_checkpointer_conn(db_path: Path) -> sqlite3.Connection:
    """Open a dedicated SQLite connection for LangGraph's SqliteSaver."""
    conn = sqlite3.connect(
        str(db_path),
        timeout=SQLITE_BUSY_TIMEOUT_SECONDS,
        check_same_thread=False,
        isolation_level="DEFERRED",
    )
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


class Engine:
    """Durable workflow runner backed by SQLite + LangGraph checkpointing."""

    def __init__(
        self,
        *,
        state_dir: Path | None = None,
        agent_layer: AgentLayerClient | None = None,
        workflows: dict[str, Workflow] | None = None,
        default_workdir: Path | None = None,
    ) -> None:
        self.store = Store(state_dir)
        self.agent_layer: AgentLayerClient = agent_layer or AgentLayer()
        self.default_workdir = default_workdir.resolve() if default_workdir is not None else None
        self._workflows: dict[str, Workflow] = dict(workflows) if workflows is not None else {}
        self._allow_entry_points = workflows is None
        self._cp_conn = open_checkpointer_conn(self.store.db_path)
        self._checkpointer = SqliteSaver(
            self._cp_conn, serde=JsonPlusSerializer(allowed_msgpack_modules=True)
        )
        self._compiled: dict[str, CompiledStateGraph[Any, _RunContext, Any, Any]] = {}

    def close(self) -> None:
        self._cp_conn.close()
        self.store.close()

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _get_workflow(self, name: str) -> Workflow:
        wf = self._workflows.get(name)
        if wf is None:
            if not self._allow_entry_points:
                raise KeyError(f"Unknown workflow: {name!r}")
            wf = load_workflow(name)
            self._workflows[name] = wf
        return wf

    def _get_compiled(self, wf: Workflow) -> CompiledStateGraph[Any, _RunContext, Any, Any]:
        compiled = self._compiled.get(wf.name)
        if compiled is None:
            compiled = _compile_workflow(wf, self._checkpointer)
            self._compiled[wf.name] = compiled
        return compiled

    def _thread_config(self, job_id: UUID) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": str(job_id)}}

    def start(self, workflow: str, raw_input: BaseModel | dict[str, Any]) -> UUID:
        """Validate input and record a pending job without executing it."""
        wf = self._get_workflow(workflow)
        validated = (
            raw_input if isinstance(raw_input, wf.input) else wf.input.model_validate(raw_input)
        )
        job_id = uuid4()
        self.store.create_job(
            job_id,
            workflow=wf.name,
            workflow_version=wf.version,
            input_json=validated.model_dump_json(),
        )
        return job_id

    def reply(self, job_id: UUID, wait_id: str, value: JsonValue) -> None:
        """Validate and record a reply for a currently pending wait/ask interrupt."""
        job = self.store.get_job(job_id)
        wf = self._get_workflow(job.workflow)
        snap = self._get_compiled(wf).get_state(cast(Any, self._thread_config(job_id)))

        target = next((intr for intr in snap.interrupts if intr.id == wait_id), None)
        if target is None:
            raise UnknownWaitError(f"Wait {wait_id!r} is not pending on job {job_id}")

        meta = target.value if isinstance(target.value, dict) else {}
        node_name = meta.get("node")
        if not isinstance(node_name, str) or node_name not in wf.graph.nodes:
            raise EngineError(f"Interrupt {wait_id!r} references unknown node {node_name!r}")

        node = wf.graph.nodes[node_name]
        if isinstance(node, WaitNode):
            serialized = node.reply.model_validate(value).model_dump_json()
        elif isinstance(node, AgentNode):
            serialized = json.dumps(_validate_ask_reply(value))
        else:
            raise EngineError(f"Node {node_name!r} does not accept replies")

        self.store.record_reply(job_id, wait_id, serialized)

    def stop(self, job_id: UUID) -> None:
        """Request cooperative stop for `job_id`."""
        self.store.request_stop(job_id)

    def run(self, job_id: UUID) -> JobStatus:
        """Start or resume a job until completion, pause, cooperative stop, or error."""
        job = self.store.get_job(job_id)
        wf = self._get_workflow(job.workflow)
        if wf.version != job.workflow_version:
            raise WorkflowVersionMismatchError(
                f"Job {job_id} was created with {wf.name}@{job.workflow_version}, "
                f"cannot run with {wf.name}@{wf.version}"
            )
        compiled = self._get_compiled(wf)
        config = cast(Any, self._thread_config(job_id))

        with self.store.job_lock(job_id):
            snap = compiled.get_state(config)
            graph_input: Any = None
            should_invoke = True
            if snap.created_at is None:
                graph_input = wf.initial_state(json.loads(job.input_json))
            elif snap.interrupts:
                answered = self.store.get_replies(job_id, [intr.id for intr in snap.interrupts])
                if not answered:
                    should_invoke = False
                else:
                    graph_input = Command(
                        resume={wid: json.loads(raw) for wid, raw in answered.items()}
                    )
            elif not snap.next:
                should_invoke = False

            if should_invoke:
                self.store.ack_stop(job_id)
                self.store.set_last_error(job_id, None)
                ctx = _RunContext(
                    engine=self,
                    workflow=wf,
                    job_id=job_id,
                    artifacts_dir=self.store.artifacts_dir(job_id),
                )
                control = _StoreRunControl(self.store, job_id)
                try:
                    compiled.invoke(
                        graph_input, config, context=ctx, durability="sync", control=control
                    )
                except GraphDrained:
                    pass
                except Exception as exc:
                    self.store.set_last_error(job_id, f"{type(exc).__name__}: {exc}")
                    raise

        return self.inspect(job_id)

    def inspect(self, job_id: UUID) -> JobStatus:
        """Non-blocking snapshot of job phase, pending waits, active turns, and state."""
        job = self.store.get_job(job_id)
        wf = self._get_workflow(job.workflow)
        running = self.store.is_running(job_id)
        snap = self._get_compiled(wf).get_state(cast(Any, self._thread_config(job_id)))
        active_turns = [
            ActiveTurnInfo.model_validate(t, from_attributes=True)
            for t in self.store.list_active_turns(job_id)
        ]

        answered_ids = set(
            self.store.get_replies(job_id, [intr.id for intr in snap.interrupts]).keys()
        )
        pending_waits = [
            PendingWait(
                wait_id=intr.id,
                kind=str(meta.get("kind", "wait")),
                prompt=str(meta.get("prompt", "")),
                node=str(meta.get("node", "")),
                payload=meta["payload"] if isinstance(meta.get("payload"), dict) else {},
                answered=intr.id in answered_ids,
            )
            for intr in snap.interrupts
            for meta in [intr.value if isinstance(intr.value, dict) else {}]
        ]

        phase = _derive_phase(
            running=running,
            stop_pending=job.stop_pending,
            last_error=job.last_error,
            has_checkpoint=snap.created_at is not None,
            has_next=bool(snap.next),
            has_interrupts=bool(snap.interrupts),
        )
        return JobStatus(
            job_id=job.job_id,
            workflow=job.workflow,
            workflow_version=job.workflow_version,
            phase=phase,
            last_error=job.last_error,
            next_nodes=list(snap.next),
            pending_waits=pending_waits,
            active_turns=active_turns,
            state=_dump_json_state(snap.values),
        )


def _derive_phase(
    *,
    running: bool,
    stop_pending: bool,
    last_error: str | None,
    has_checkpoint: bool,
    has_next: bool,
    has_interrupts: bool,
) -> Phase:
    if running:
        return "stopping" if stop_pending else "running"
    if last_error is not None:
        return "failed"
    if has_checkpoint and not has_next and not has_interrupts:
        return "completed"
    if stop_pending:
        return "stopped"
    if has_interrupts:
        return "paused"
    return "pending"
