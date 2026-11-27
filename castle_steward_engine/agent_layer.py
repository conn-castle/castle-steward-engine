"""Thin subprocess client over `al -q dispatch`."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Annotated, Protocol

from pydantic import BaseModel, ConfigDict, StringConstraints

from castle_steward_engine.workflow import AgentSpec, AvailableAgent, QuotaSnapshot

DEFAULT_AGENT_LAYER_EXECUTABLE = "al"
ALREADY_STARTED_EXIT_CODE = 82
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


def _looks_like_no_provider_conversation(text: str) -> bool:
    lower = text.lower()
    return any(needle in lower for needle in _NO_PROVIDER_CONVERSATION_PATTERNS)


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
        """Return quota telemetry once conn-castle/agent-layer#575 ships; empty until then."""
        _ = workdir
        return QuotaSnapshot()

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
