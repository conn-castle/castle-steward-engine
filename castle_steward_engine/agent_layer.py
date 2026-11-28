"""Thin subprocess client over `al -q dispatch`."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Annotated, Protocol

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
)

from castle_steward_engine.workflow import AgentSpec, AvailableAgent, QuotaReading, QuotaSnapshot

DEFAULT_AGENT_LAYER_EXECUTABLE = "al"
DEFAULT_DISPATCH_MAX_DEPTH = 1
ALREADY_STARTED_EXIT_CODE = 82
EXIT_QUOTA_PARTIAL = 84
EXIT_QUOTA_NONE = 85
_NO_PROVIDER_CONVERSATION_PATTERNS = (
    "no provider conversation",
    "no conversation to continue",
    "no resumable conversation",
    "conversation does not exist",
    "never started",
)
_NonBlankStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class AgentLayerError(RuntimeError):
    """Raised when an `al dispatch` subprocess fails."""

    def __init__(
        self,
        message: str,
        *,
        returncode: int | None = None,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class NoProviderConversationError(AgentLayerError):
    """Raised when `reserve_continue` fails because no provider conversation exists."""


class DispatchResult(BaseModel):
    """JSON Result payload returned by `al dispatch` commands (`agentdispatch.Result`).

    Uses `extra="ignore"` so additive fields from newer `agent-layer` CLI versions remain
    forward-compatible while the required fields the engine reads stay strictly validated.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    handle: _NonBlankStr
    invocation_id: _NonBlankStr
    state: _NonBlankStr
    error: str | None = None
    termination_confirmed: bool = False
    condition_met: bool | None = None


class AgentLayerClient(Protocol):
    """Protocol implemented by `AgentLayer` and in-process test doubles."""

    def options(self, workdir: Path | None = None) -> dict[str, AvailableAgent]: ...

    def quota(self, workdir: Path | None = None) -> QuotaSnapshot: ...

    def reserve_start(self, workdir: Path) -> DispatchResult: ...

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
    ) -> DispatchResult: ...

    def reserve_continue(self, workdir: Path, handle: str) -> DispatchResult: ...

    def continue_session(
        self,
        workdir: Path,
        *,
        handle: str,
        reservation_id: str,
        prompt_file: Path,
    ) -> DispatchResult: ...

    def wait(self, workdir: Path, invocation_id: str) -> DispatchResult: ...

    def inspect(self, workdir: Path, invocation_id: str) -> DispatchResult: ...


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)


class _QuotaWindow(_WireModel):
    used_percent: float | int | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    duration_seconds: int | None = Field(default=None, ge=0)
    starts_at: AwareDatetime | None = None
    resets_at: AwareDatetime | None = None
    state: str = ""
    disabled: bool | None = None


class _QuotaPool(_WireModel):
    id: str = ""
    scope: str = ""
    windows: list[_QuotaWindow] | None = None


class _QuotaModel(_WireModel):
    id: _NonBlankStr
    pools: list[str] | None = None
    blocked: bool | None = None


class _QuotaResult(_WireModel):
    agent: _NonBlankStr
    status: _NonBlankStr
    attempted_at: AwareDatetime | None = None
    observed_at: AwareDatetime | None = None
    blocked: bool | None = None
    pools: list[_QuotaPool] | None = None
    models: list[_QuotaModel] | None = None


class _QuotaResponse(_WireModel):
    schema_version: int = Field(ge=1, le=1)
    results: list[_QuotaResult]


def _looks_like_no_provider_conversation(text: str) -> bool:
    lower = text.lower()
    return any(needle in lower for needle in _NO_PROVIDER_CONVERSATION_PATTERNS)


def _window_pace(w: _QuotaWindow, now: datetime | None) -> float | None:
    if w.used_percent is None or w.duration_seconds is None or w.duration_seconds <= 0:
        return None
    if w.state == "rolling":
        return w.used_percent / 100.0
    if w.state != "fixed":
        return None
    elapsed: float | None = None
    if now is not None:
        if w.starts_at is not None:
            elapsed = (now - w.starts_at).total_seconds()
        elif w.resets_at is not None:
            elapsed = w.duration_seconds - (w.resets_at - now).total_seconds()
    if elapsed is not None and elapsed <= 0.0:
        return None
    if w.used_percent == 0.0:
        return 0.0
    if elapsed is None:
        return None
    return (w.used_percent / 100.0) / max(1 / 7, min(1.0, elapsed / w.duration_seconds))


def _pool_pace(pool: _QuotaPool, now: datetime | None) -> float | None:
    active = [
        (w.duration_seconds, _window_pace(w, now))
        for w in (pool.windows or [])
        if w.disabled is not True
        and w.state != "not_started"
        and w.used_percent is not None
        and w.duration_seconds is not None
        and w.duration_seconds > 0
    ]
    eligible = [c for c in active if c[0] >= 86_400] or active
    if not eligible:
        return None
    return max(
        eligible, key=lambda c: (c[0], c[1] is not None, c[1] if c[1] is not None else -1.0)
    )[1]


def _to_quota_snapshot(response: _QuotaResponse) -> QuotaSnapshot:
    by_agent: dict[str, QuotaReading] = {}
    by_model: dict[str, dict[str, QuotaReading]] = {}
    for res in response.results:
        obs = res.observed_at.isoformat() if res.observed_at is not None else None
        if res.status != "ok":
            if res.blocked is True:
                by_agent[res.agent] = QuotaReading(blocked=True, observed_at=obs)
            continue
        now = res.observed_at or res.attempted_at
        pools = [p for p in (res.pools or []) if p.scope != "other_surface"]
        paces = [(p, _pool_pace(p, now)) for p in pools]
        by_id = {p.id: pace for p, pace in paces if p.id and pace is not None}
        all_paces = [pace for p, pace in paces if p.scope == "all" and pace is not None]
        agent_blocked = bool(res.blocked)
        by_agent[res.agent] = QuotaReading(
            blocked=agent_blocked,
            pace_ratio=max(all_paces) if all_paces else None,
            observed_at=obs,
        )
        models: dict[str, QuotaReading] = {}
        for m in res.models or []:
            m_paces = [by_id[pid] for pid in (m.pools or []) if pid in by_id]
            models[m.id] = QuotaReading(
                blocked=m.blocked if m.blocked is not None else agent_blocked,
                pace_ratio=max(m_paces) if m_paces else None,
                observed_at=obs,
            )
        if models:
            by_model[res.agent] = models
    return QuotaSnapshot(by_agent=by_agent, by_model=by_model)


class AgentLayer:
    """Thin `al -q dispatch` wrapper."""

    def __init__(self, executable: str = DEFAULT_AGENT_LAYER_EXECUTABLE) -> None:
        self._executable = executable

    def _run(
        self,
        args: list[str],
        *,
        workdir: Path | None,
        allow_exit_codes: tuple[int, ...] = (),
    ) -> subprocess.CompletedProcess[str]:
        cmd = [self._executable, "-q", "dispatch", *args]
        proc = subprocess.run(
            cmd,
            cwd=str(workdir) if workdir is not None else None,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0 and proc.returncode not in allow_exit_codes:
            detail = (proc.stderr or proc.stdout).strip() or f"exit code {proc.returncode}"
            raise AgentLayerError(
                f"`{' '.join(cmd)}` failed: {detail}",
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
        return proc

    def options(self, workdir: Path | None = None) -> dict[str, AvailableAgent]:
        proc = self._run(["options"], workdir=workdir)
        data = json.loads(proc.stdout)
        if not isinstance(data, dict):
            raise AgentLayerError("Expected JSON object from al dispatch options")
        result: dict[str, AvailableAgent] = {}
        for entry in data.get("agents", []):
            if not isinstance(entry, dict) or not entry.get("available"):
                continue
            name = str(entry["agent"])
            model_cfg = entry.get("model")
            default_model = (
                model_cfg.get("configured") or None if isinstance(model_cfg, dict) else None
            )
            effort_cfg = entry.get("reasoning_effort")
            default_effort = (
                effort_cfg.get("configured") or None if isinstance(effort_cfg, dict) else None
            )
            mcp_cfg = entry.get("session_mcp_servers")
            session_mcp = bool(isinstance(mcp_cfg, dict) and mcp_cfg.get("supported"))
            result[name] = AvailableAgent(
                agent=name,
                default_model=default_model,
                default_reasoning_effort=default_effort,
                session_mcp=session_mcp,
            )
        return result

    def quota(self, workdir: Path | None = None) -> QuotaSnapshot:
        """Parse `al -q dispatch quota` JSON telemetry into a `QuotaSnapshot`."""
        proc = self._run(
            ["quota"],
            workdir=workdir,
            allow_exit_codes=(EXIT_QUOTA_PARTIAL, EXIT_QUOTA_NONE),
        )
        try:
            response = _QuotaResponse.model_validate_json(proc.stdout)
        except ValidationError as exc:
            raise AgentLayerError(
                f"Invalid al dispatch quota payload: {exc}",
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
            ) from exc
        return _to_quota_snapshot(response)

    def reserve_start(self, workdir: Path) -> DispatchResult:
        return DispatchResult.model_validate_json(self._run(["reserve"], workdir=workdir).stdout)

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
        args = [
            "start",
            "--reservation",
            reservation_handle,
            "--agent",
            agent.agent,
            "--prompt-file",
            str(prompt_file),
            "--mcp-config",
            str(mcp_config),
            "--role",
            role,
            "--max-depth",
            str(DEFAULT_DISPATCH_MAX_DEPTH),
        ]
        if agent.model:
            args.extend(["--model", agent.model])
        if agent.reasoning_effort:
            args.extend(["--reasoning-effort", agent.reasoning_effort])
        proc = self._run(args, workdir=workdir, allow_exit_codes=(ALREADY_STARTED_EXIT_CODE,))
        if proc.returncode == ALREADY_STARTED_EXIT_CODE:
            return self.inspect(workdir, invocation_id)
        return DispatchResult.model_validate_json(proc.stdout)

    def reserve_continue(self, workdir: Path, handle: str) -> DispatchResult:
        try:
            proc = self._run(["reserve", "--continue", handle], workdir=workdir)
        except AgentLayerError as exc:
            if _looks_like_no_provider_conversation(f"{exc} {exc.stderr} {exc.stdout}"):
                raise NoProviderConversationError(
                    str(exc), returncode=exc.returncode, stdout=exc.stdout, stderr=exc.stderr
                ) from exc
            raise
        return DispatchResult.model_validate_json(proc.stdout)

    def continue_session(
        self,
        workdir: Path,
        *,
        handle: str,
        reservation_id: str,
        prompt_file: Path,
    ) -> DispatchResult:
        args = [
            "continue",
            handle,
            "--reservation",
            reservation_id,
            "--prompt-file",
            str(prompt_file),
        ]
        proc = self._run(args, workdir=workdir, allow_exit_codes=(ALREADY_STARTED_EXIT_CODE,))
        if proc.returncode == ALREADY_STARTED_EXIT_CODE:
            return self.inspect(workdir, reservation_id)
        return DispatchResult.model_validate_json(proc.stdout)

    def wait(self, workdir: Path, invocation_id: str) -> DispatchResult:
        while True:
            proc = self._run(
                ["wait", invocation_id, "--condition", "termination_confirmed"],
                workdir=workdir,
            )
            outcome = DispatchResult.model_validate_json(proc.stdout)
            if outcome.condition_met is not False:
                return outcome

    def inspect(self, workdir: Path, invocation_id: str) -> DispatchResult:
        proc = self._run(["inspect", invocation_id], workdir=workdir)
        return DispatchResult.model_validate_json(proc.stdout)
