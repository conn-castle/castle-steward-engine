"""Public in-process test harness for workflows built on `castle_steward_engine`.

Analogous to `django.test` or `click.testing`, this module lives inside the
`castle_steward_engine` package so external downstream workflow packages installed
against `castle-steward-engine` can test their workflows deterministically through
the real MCP `SessionToolHost` without real LLMs or `al` subprocesses.

Import `castle_steward_engine.testing` explicitly from test suites; it is kept
separate from `castle_steward_engine.__init__` so production runtime never imports
test helpers.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from castle_steward_engine.agent_layer import (
    AgentLayerError,
    DispatchResult,
    NoProviderConversationError,
)
from castle_steward_engine.mcp_server import (
    ActiveTurn,
    SessionDir,
    SessionToolHost,
    ToolCallResult,
)
from castle_steward_engine.workflow import AgentSpec, AvailableAgent, QuotaSnapshot, Workflow


class SimulatedCrash(RuntimeError):
    """Test-injected crash to exercise reservation/running recovery paths."""


@dataclass(frozen=True)
class FakeTurnOutcome:
    state: str = "completed"
    provider_started: bool = True
    error: str | None = None


@dataclass(frozen=True)
class FakeCallRecord:
    handle: str
    invocation_id: str
    role: str
    agent: AgentSpec
    expects_tool: str | None
    turn_key: str
    attempt: int
    is_continue: bool
    prompt: str
    expects_check: str | None = None


@dataclass
class FakeTurnContext:
    handle: str
    invocation_id: str
    role: str
    agent: AgentSpec
    workdir: Path
    prompt: str
    session_dir: Path
    active_turn: ActiveTurn
    is_continue: bool
    _host: SessionToolHost = field(repr=False)

    def list_tools(self) -> list[str]:
        return [tool.name for tool in self._host.list_tools()]

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> ToolCallResult:
        return self._host.call_tool(name, arguments)


@dataclass
class _SessionInfo:
    session_dir: Path
    role: str
    agent: AgentSpec
    provider_started: bool = False


class FakeAgentLayer:
    """Thread-safe fake AgentLayer executing turns through real MCP validation and files."""

    def __init__(
        self,
        workflows: dict[str, Workflow],
        handler: Callable[[FakeTurnContext], FakeTurnOutcome | str | None],
        *,
        available_agents: dict[str, AvailableAgent] | None = None,
        quota: QuotaSnapshot | Callable[[Path | None], QuotaSnapshot] | None = None,
        crash_before_launch: Callable[[str, bool], bool] | None = None,
        crash_after_launch: Callable[[FakeTurnContext], bool] | None = None,
        crash_on_wait: Callable[[str], bool] | None = None,
    ) -> None:
        self._workflows = workflows
        self._handler = handler
        self._available = (
            dict(available_agents)
            if available_agents is not None
            else {
                "codex": AvailableAgent(agent="codex", default_model="gpt-test", session_mcp=True),
                "claude": AvailableAgent(
                    agent="claude", default_model="opus-test", session_mcp=True
                ),
            }
        )
        self._quota = quota
        self._crash_before_launch = crash_before_launch
        self._crash_after_launch = crash_after_launch
        self._crash_on_wait = crash_on_wait
        self._lock = threading.Lock()
        self._counter = 0
        self._sessions: dict[str, _SessionInfo] = {}
        self._outcomes: dict[str, DispatchResult] = {}
        self._launched: dict[str, DispatchResult] = {}
        self.calls: list[FakeCallRecord] = []
        self.adopted: list[str] = []

    def _next_id(self, prefix: str) -> str:
        with self._lock:
            self._counter += 1
            return f"{prefix}-{self._counter}"

    def options(self, workdir: Path | None = None) -> dict[str, AvailableAgent]:
        return dict(self._available)

    def quota(self, workdir: Path | None = None) -> QuotaSnapshot:
        if self._quota is None:
            return QuotaSnapshot()
        if callable(self._quota):
            return self._quota(workdir)
        return self._quota

    def reserve_start(self, workdir: Path) -> DispatchResult:
        return DispatchResult(
            handle=self._next_id("h"), invocation_id=self._next_id("inv"), state="reserved"
        )

    def start(
        self,
        workdir: Path,
        *,
        reservation_handle: str,
        invocation_id: str,
        agent: AgentSpec,
        prompt_file: Path,
        mcp_config: Path,
        role: str,
    ) -> DispatchResult:
        with self._lock:
            if invocation_id in self._launched:
                self.adopted.append(invocation_id)
                return self._outcomes[invocation_id]
        if self._crash_before_launch is not None and self._crash_before_launch(
            invocation_id, False
        ):
            raise SimulatedCrash(f"Simulated crash before launching {invocation_id}")

        info = _SessionInfo(session_dir=mcp_config.parent, role=role, agent=agent)
        with self._lock:
            self._sessions[reservation_handle] = info
        return self._execute_turn(
            handle=reservation_handle,
            invocation_id=invocation_id,
            info=info,
            workdir=workdir,
            prompt_file=prompt_file,
            is_continue=False,
        )

    def reserve_continue(self, workdir: Path, handle: str) -> DispatchResult:
        with self._lock:
            info = self._sessions.get(handle)
        if info is None or not info.provider_started:
            raise NoProviderConversationError(
                f"Handle {handle!r} has no provider conversation to continue"
            )
        return DispatchResult(handle=handle, invocation_id=self._next_id("inv"), state="reserved")

    def continue_session(
        self,
        workdir: Path,
        *,
        handle: str,
        reservation_id: str,
        prompt_file: Path,
    ) -> DispatchResult:
        with self._lock:
            if reservation_id in self._launched:
                self.adopted.append(reservation_id)
                return self._outcomes[reservation_id]
            info = self._sessions.get(handle)
        if info is None:
            raise AgentLayerError(f"Unknown session handle: {handle!r}")
        if self._crash_before_launch is not None and self._crash_before_launch(
            reservation_id, True
        ):
            raise SimulatedCrash(f"Simulated crash before continuing {reservation_id}")
        return self._execute_turn(
            handle=handle,
            invocation_id=reservation_id,
            info=info,
            workdir=workdir,
            prompt_file=prompt_file,
            is_continue=True,
        )

    def _execute_turn(
        self,
        *,
        handle: str,
        invocation_id: str,
        info: _SessionInfo,
        workdir: Path,
        prompt_file: Path,
        is_continue: bool,
    ) -> DispatchResult:
        sdir = SessionDir(info.session_dir)
        meta = sdir.read_meta()
        host = SessionToolHost(info.session_dir, self._workflows[meta.workflow].roles[meta.role])
        active_turn = sdir.read_active_turn()
        prompt = prompt_file.read_text(encoding="utf-8")
        ctx = FakeTurnContext(
            handle=handle,
            invocation_id=invocation_id,
            role=info.role,
            agent=info.agent,
            workdir=workdir,
            prompt=prompt,
            session_dir=info.session_dir,
            active_turn=active_turn,
            is_continue=is_continue,
            _host=host,
        )
        inv = DispatchResult(handle=handle, invocation_id=invocation_id, state="running")
        with self._lock:
            self._launched[invocation_id] = inv
            self.calls.append(
                FakeCallRecord(
                    handle=handle,
                    invocation_id=invocation_id,
                    role=info.role,
                    agent=info.agent,
                    expects_tool=active_turn.expects_tool,
                    expects_check=active_turn.expects_check,
                    turn_key=active_turn.turn_key,
                    attempt=active_turn.attempt,
                    is_continue=is_continue,
                    prompt=prompt,
                )
            )

        raw = self._handler(ctx)
        spec = (
            FakeTurnOutcome()
            if raw is None
            else (FakeTurnOutcome(state=raw) if isinstance(raw, str) else raw)
        )
        with self._lock:
            if spec.provider_started:
                info.provider_started = True
            self._outcomes[invocation_id] = DispatchResult(
                handle=handle,
                invocation_id=invocation_id,
                state=spec.state,
                condition_met=True,
                termination_confirmed=True,
                error=spec.error,
            )

        if self._crash_after_launch is not None and self._crash_after_launch(ctx):
            raise SimulatedCrash(f"Simulated crash after launching {invocation_id}")
        return inv

    def wait(self, workdir: Path, invocation_id: str) -> DispatchResult:
        if self._crash_on_wait is not None and self._crash_on_wait(invocation_id):
            raise SimulatedCrash(f"Simulated crash while waiting on {invocation_id}")
        return self.inspect(workdir, invocation_id)

    def inspect(self, workdir: Path, invocation_id: str) -> DispatchResult:
        with self._lock:
            outcome = self._outcomes.get(invocation_id)
        if outcome is None:
            raise AgentLayerError(f"Unknown invocation: {invocation_id!r}")
        return outcome
