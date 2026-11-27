"""Public SDK surface (eagerly loads only stdlib + pydantic workflow primitives).

Runtime symbols (`Engine`, `AgentLayer`, `EngineSettings`, exceptions, etc.) are exposed via
PEP 562 `__getattr__` lazy loading so stdio MCP server startup (`stew mcp`) never imports
LangGraph or SQLite when loading workflow definitions.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from castle_steward_engine.workflow import (
    END,
    AgentSpec,
    AppendList,
    AvailableAgent,
    Graph,
    JobInfo,
    QuotaReading,
    QuotaSnapshot,
    Role,
    Turn,
    Wait,
    Workflow,
    WorkflowNotFoundError,
    load_workflow,
    load_workflows,
)

if TYPE_CHECKING:
    from castle_steward_engine.agent_layer import (
        AgentLayer,
        AgentLayerError,
        NoProviderConversationError,
    )
    from castle_steward_engine.cli import EngineSettings
    from castle_steward_engine.engine import (
        ActiveTurnInfo,
        Engine,
        EngineError,
        IncompleteWork,
        JobStatus,
        PendingWait,
        TurnFailed,
        UnknownWaitError,
        WorkflowVersionMismatchError,
    )
    from castle_steward_engine.store import JobBusy, JobNotFoundError

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ActiveTurnInfo": ("castle_steward_engine.engine", "ActiveTurnInfo"),
    "AgentLayer": ("castle_steward_engine.agent_layer", "AgentLayer"),
    "AgentLayerError": ("castle_steward_engine.agent_layer", "AgentLayerError"),
    "Engine": ("castle_steward_engine.engine", "Engine"),
    "EngineError": ("castle_steward_engine.engine", "EngineError"),
    "EngineSettings": ("castle_steward_engine.cli", "EngineSettings"),
    "IncompleteWork": ("castle_steward_engine.engine", "IncompleteWork"),
    "JobBusy": ("castle_steward_engine.store", "JobBusy"),
    "JobNotFoundError": ("castle_steward_engine.store", "JobNotFoundError"),
    "JobStatus": ("castle_steward_engine.engine", "JobStatus"),
    "NoProviderConversationError": (
        "castle_steward_engine.agent_layer",
        "NoProviderConversationError",
    ),
    "PendingWait": ("castle_steward_engine.engine", "PendingWait"),
    "TurnFailed": ("castle_steward_engine.engine", "TurnFailed"),
    "UnknownWaitError": ("castle_steward_engine.engine", "UnknownWaitError"),
    "WorkflowVersionMismatchError": (
        "castle_steward_engine.engine",
        "WorkflowVersionMismatchError",
    ),
}

__all__ = [
    "END",
    "ActiveTurnInfo",
    "AgentLayer",
    "AgentLayerError",
    "AgentSpec",
    "AppendList",
    "AvailableAgent",
    "Engine",
    "EngineError",
    "EngineSettings",
    "Graph",
    "IncompleteWork",
    "JobBusy",
    "JobInfo",
    "JobNotFoundError",
    "JobStatus",
    "NoProviderConversationError",
    "PendingWait",
    "QuotaReading",
    "QuotaSnapshot",
    "Role",
    "Turn",
    "TurnFailed",
    "UnknownWaitError",
    "Wait",
    "Workflow",
    "WorkflowNotFoundError",
    "WorkflowVersionMismatchError",
    "load_workflow",
    "load_workflows",
]


def __getattr__(name: str) -> Any:
    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr_name = spec
    value = getattr(importlib.import_module(mod_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
