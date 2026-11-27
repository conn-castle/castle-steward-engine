"""Workflow definition SDK (stdlib + pydantic only)."""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Final, Self, TypeVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue

# --- Constants & reducer alias ---

# Keep TypeVar + Annotated alias (not PEP 695 `type AppendList[T] = ...`):
# LangGraph inspects `Annotated` reducers directly and does not unwrap `TypeAliasType`.
T = TypeVar("T")

END: Final[str] = "__end__"
WORKFLOW_ENTRY_POINT_GROUP: Final[str] = "castle_steward_engine.workflows"
ASK_TOOL_NAME: Final[str] = "ask"
CHECK_DONE_TOOL_NAME: Final[str] = "check_done"

AppendList = Annotated[list[T], operator.add]


# --- Public author-facing SDK ---


class AgentSpec(BaseModel):
    """Target agent configuration for a session."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    agent: str
    model: str | None = None
    reasoning_effort: str | None = None


class AvailableAgent(BaseModel):
    """Agent availability reported by Agent Layer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent: str
    default_model: str | None = None
    default_reasoning_effort: str | None = None
    session_mcp: bool = False


class QuotaReading(BaseModel):
    """Point-in-time block status and weekly burn-rate pace for an agent or model pool."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    blocked: bool = False
    pace_ratio: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    observed_at: str | None = None


class QuotaSnapshot(BaseModel):
    """Quota telemetry keyed by provider agent and model pool."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    by_agent: dict[str, QuotaReading] = Field(default_factory=dict)
    by_model: dict[str, dict[str, QuotaReading]] = Field(default_factory=dict)

    def for_spec(self, spec: AgentSpec) -> QuotaReading:
        """Return merged model-pool and agent-level quota reading for `spec`."""
        agent_reading = self.by_agent.get(spec.agent)
        model_reading = (
            models.get(spec.model)
            if spec.model is not None and (models := self.by_model.get(spec.agent)) is not None
            else None
        )
        if model_reading is None:
            return agent_reading if agent_reading is not None else QuotaReading()
        if agent_reading is None:
            return model_reading
        return QuotaReading(
            blocked=model_reading.blocked or agent_reading.blocked,
            pace_ratio=(
                model_reading.pace_ratio
                if model_reading.pace_ratio is not None
                else agent_reading.pace_ratio
            ),
            observed_at=(
                model_reading.observed_at
                if model_reading.observed_at is not None
                else agent_reading.observed_at
            ),
        )


@dataclass(frozen=True)
class JobInfo:
    """Read-only job metadata and host capabilities passed to step nodes."""

    job_id: UUID
    artifacts_dir: Path
    _options_fn: Callable[[Path | None], dict[str, AvailableAgent]] = field(
        repr=False, compare=False
    )
    _quota_fn: Callable[[Path | None], QuotaSnapshot] = field(repr=False, compare=False)
    _check_stop_fn: Callable[[], None] = field(repr=False, compare=False)

    def available_agents(self, workdir: Path | None = None) -> dict[str, AvailableAgent]:
        """Return available agents reported by Agent Layer for `workdir`."""
        return self._options_fn(workdir)

    def agent_quota(self, workdir: Path | None = None) -> QuotaSnapshot:
        """Return current agent quota telemetry reported by Agent Layer for `workdir`."""
        return self._quota_fn(workdir)

    def check_stop(self) -> None:
        """Raise cooperatively to stop the step if a stop has been requested."""
        self._check_stop_fn()


@dataclass(frozen=True)
class Role:
    """Fixed MCP tool and completion-check contract for an agent session."""

    name: str
    tools: Mapping[str, type[BaseModel]] = field(default_factory=dict)
    checks: Mapping[str, Callable[[dict[str, JsonValue]], BaseModel]] = field(default_factory=dict)
    can_ask_human: bool = True

    def _validate_tools(self) -> dict[str, type[BaseModel]]:
        seen_models: dict[type[BaseModel], str] = {}
        validated: dict[str, type[BaseModel]] = {}
        for tname, model in self.tools.items():
            if not tname or not tname.strip() or any(ch.isspace() for ch in tname):
                raise ValueError(f"Invalid tool name {tname!r} in role {self.name!r}")
            if tname in (ASK_TOOL_NAME, CHECK_DONE_TOOL_NAME):
                raise ValueError(f"Tool name {tname!r} collides with built-in MCP tool")
            if not (isinstance(model, type) and issubclass(model, BaseModel)):
                raise TypeError(
                    f"Role tool {tname!r} must map to a BaseModel subclass, got {model!r}"
                )
            if model in seen_models:
                raise ValueError(
                    f"Model {model.__name__} mapped to multiple tools in role {self.name!r}: "
                    f"{seen_models[model]!r} and {tname!r}"
                )
            seen_models[model] = tname
            validated[tname] = model
        return validated

    def _validate_checks(self) -> dict[str, Callable[[dict[str, JsonValue]], BaseModel]]:
        validated: dict[str, Callable[[dict[str, JsonValue]], BaseModel]] = {}
        for cname, fn in self.checks.items():
            if not cname or not cname.strip() or any(ch.isspace() for ch in cname):
                raise ValueError(f"Invalid check name {cname!r} in role {self.name!r}")
            if cname in (ASK_TOOL_NAME, CHECK_DONE_TOOL_NAME):
                raise ValueError(f"Check name {cname!r} collides with built-in MCP tool")
            if not callable(fn):
                raise TypeError(f"Role check {cname!r} must be callable, got {fn!r}")
            validated[cname] = fn
        return validated

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip() or any(ch.isspace() for ch in self.name):
            raise ValueError(f"Invalid role name: {self.name!r}")
        object.__setattr__(self, "tools", MappingProxyType(self._validate_tools()))
        object.__setattr__(self, "checks", MappingProxyType(self._validate_checks()))

    def __hash__(self) -> int:
        return hash(
            (
                self.name,
                self.can_ask_human,
                tuple(sorted(self.tools.items())),
                tuple(sorted(self.checks.items())),
            )
        )

    def tool_name(self, model: type[BaseModel]) -> str:
        """Return the registered MCP tool name for `model`."""
        for tname, registered in self.tools.items():
            if registered is model:
                return tname
        raise ValueError(f"Model {model.__name__} is not registered on role {self.name!r}")

    def check_fn(self, check_name: str) -> Callable[[dict[str, JsonValue]], BaseModel]:
        """Return the registered completion-check callback for `check_name`."""
        fn = self.checks.get(check_name)
        if fn is None:
            raise ValueError(f"Check {check_name!r} is not registered on role {self.name!r}")
        return fn

    def result_models_by_tool(self) -> dict[str, type[BaseModel]]:
        """Map submission tool names to their Pydantic models."""
        return dict(self.tools)


@dataclass(frozen=True)
class Turn:
    """Prompt and session parameters returned by an agent node's `ask` callback."""

    session: str
    prompt: str
    context: dict[str, JsonValue] = field(default_factory=dict)
    agent: AgentSpec | None = None
    workdir: Path | None = None


@dataclass(frozen=True)
class Wait:
    """Human wait specification returned by a wait node's `ask` callback."""

    kind: str
    prompt: str
    payload: dict[str, JsonValue] = field(default_factory=dict)


class Graph[S: BaseModel]:
    """Declarative workflow graph compiled privately by the engine."""

    def __init__(self) -> None:
        self.nodes: dict[str, GraphNode] = {}
        self.entry_node: str | None = None
        self.edges: list[tuple[str, str]] = []
        self.branches: list[BranchEdge] = []
        self.fan_outs: list[FanOutEdge] = []

    def _register_node(self, node: GraphNode) -> None:
        if not node.name or node.name == END:
            raise ValueError(f"Invalid node name: {node.name!r}")
        if node.name in self.nodes:
            raise ValueError(f"Duplicate node name: {node.name!r}")
        self.nodes[node.name] = node

    def step(self, name: str, fn: Callable[[S, JobInfo], dict[str, Any]]) -> Self:
        self._register_node(StepNode(name=name, fn=fn))
        return self

    def agent[R: BaseModel](
        self,
        name: str,
        *,
        role: Role,
        expects: type[R],
        ask: Callable[[Any], Turn],
        then: Callable[[Any, R], dict[str, Any]],
        check: str | None = None,
    ) -> Self:
        self._register_node(
            AgentNode(name=name, role=role, expects=expects, ask=ask, then=then, check=check)
        )
        return self

    def wait[ReplyT: BaseModel](
        self,
        name: str,
        *,
        reply: type[ReplyT],
        ask: Callable[[S], Wait],
        then: Callable[[S, ReplyT], dict[str, Any]],
    ) -> Self:
        self._register_node(WaitNode(name=name, reply=reply, ask=ask, then=then))
        return self

    def entry(self, name: str) -> Self:
        self.entry_node = name
        return self

    def edge(self, src: str, dst: str) -> Self:
        self.edges.append((src, dst))
        return self

    def branch(self, src: str, route: Callable[[S], str], targets: dict[str, str]) -> Self:
        self.branches.append(BranchEdge(src=src, route=route, targets=dict(targets)))
        return self

    def fan_out[ItemT: BaseModel](
        self,
        src: str,
        dst: str,
        item_type: type[ItemT],
        items: Callable[[S], Sequence[ItemT]],
    ) -> Self:
        self.fan_outs.append(FanOutEdge(src=src, dst=dst, item_type=item_type, items=items))
        return self

    def _validate_roles(self) -> None:
        roles: dict[str, Role] = {}
        for node in self.nodes.values():
            if not isinstance(node, AgentNode):
                continue
            if roles.setdefault(node.role.name, node.role) != node.role:
                raise ValueError(f"Inconsistent Role {node.role.name!r} on node {node.name!r}")
            if node.check is None:
                if node.expects not in node.role.tools.values():
                    raise ValueError(
                        f"Node {node.name!r} expects {node.expects.__name__}, "
                        f"not in role {node.role.name!r}"
                    )
            else:
                if node.check not in node.role.checks:
                    raise ValueError(
                        f"Node {node.name!r} references check {node.check!r}, "
                        f"not in role {node.role.name!r}"
                    )
                if not (isinstance(node.expects, type) and issubclass(node.expects, BaseModel)):
                    raise TypeError(
                        f"Node {node.name!r} expects must be a BaseModel subclass, "
                        f"got {node.expects!r}"
                    )

    def validate(self) -> None:  # noqa: C901
        """Validate entry point, roles, expectations, routing invariants, and reachability."""
        nodes, entry = self.nodes, self.entry_node
        if not entry or entry not in nodes:
            raise ValueError(f"Entry node {entry!r} is missing from graph")
        self._validate_roles()

        outgoing: dict[str, str] = {}
        adjacency: dict[str, set[str]] = {name: set() for name in nodes}
        plain_targets: set[str] = set()
        fan_out_types: dict[str, type[BaseModel]] = {}

        def _claim(src: str, kind: str) -> None:
            if src not in nodes:
                raise ValueError(f"{kind} source {src!r} is not a registered node")
            if prev := outgoing.get(src):
                raise ValueError(f"Node {src!r} has multiple outgoing routes ({prev} and {kind})")
            outgoing[src] = kind

        for fo in self.fan_outs:
            _claim(fo.src, "fan_out")
            if fo.dst not in nodes:
                raise ValueError(f"Fan-out target {fo.dst!r} is not a registered node")
            if fan_out_types.setdefault(fo.dst, fo.item_type) is not fo.item_type:
                raise ValueError(f"Conflicting fan_out item_type for {fo.dst!r}")
            adjacency[fo.src].add(fo.dst)

        routes: list[tuple[str, str, Sequence[str]]] = [
            *((src, "edge", (dst,)) for src, dst in self.edges),
            *((b.src, "branch", tuple(b.targets.values())) for b in self.branches),
        ]
        for src, kind, targets in routes:
            _claim(src, kind)
            if kind == "branch":
                if src in fan_out_types:
                    raise ValueError(f"Cannot branch directly off fan-out target {src!r}")
                if not targets:
                    raise ValueError(f"Branch on {src!r} has no targets")
            for dst in targets:
                if dst != END:
                    if dst not in nodes:
                        raise ValueError(f"{kind} target {dst!r} is not a registered node or END")
                    adjacency[src].add(dst)
                    plain_targets.add(dst)

        for dst in fan_out_types:
            if dst == entry:
                raise ValueError(f"Fan-out target {dst!r} cannot also be the entry node")
            if dst in plain_targets:
                raise ValueError(f"Fan-out target {dst!r} cannot be targeted by an edge or branch")

        if dead_ends := sorted(set(nodes) - set(outgoing)):
            raise ValueError(f"Nodes have no outgoing route: {', '.join(dead_ends)}")

        reachable: set[str] = set()
        stack = [entry]
        while stack:
            if (curr := stack.pop()) not in reachable:
                reachable.add(curr)
                stack.extend(adjacency[curr])
        if unreachable := sorted(set(nodes) - reachable):
            raise ValueError(f"Nodes unreachable from {entry!r}: {', '.join(unreachable)}")


@dataclass(frozen=True)
class Workflow:
    """Registered workflow definition."""

    name: str
    version: str
    input: type[BaseModel]
    state: type[BaseModel]
    graph: Graph[Any]

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Workflow name must be non-empty")
        if not self.version or not self.version.strip():
            raise ValueError("Workflow version must be non-empty")
        if "input" not in self.state.model_fields:
            raise ValueError(f"Workflow state {self.state.__name__} must declare an 'input' field")
        self.graph.validate()

    @property
    def roles(self) -> dict[str, Role]:
        """Derive role map directly from agent nodes in the graph."""
        return {
            node.role.name: node.role
            for node in self.graph.nodes.values()
            if isinstance(node, AgentNode)
        }

    def initial_state(self, raw_input: BaseModel | dict[str, Any]) -> BaseModel:
        """Validate workflow input and construct initial state."""
        validated = (
            raw_input if isinstance(raw_input, self.input) else self.input.model_validate(raw_input)
        )
        return self.state.model_validate({"input": validated})


# --- Workflow entry-point discovery ---


class WorkflowNotFoundError(KeyError):
    """Raised when a requested workflow entry point is not installed."""


def _load_entry_point(ep: EntryPoint) -> Workflow:
    loaded = ep.load()
    if not isinstance(loaded, Workflow):
        raise TypeError(f"Entry point {ep.name!r} did not resolve to a Workflow")
    if loaded.name != ep.name:
        raise ValueError(
            f"Entry point name {ep.name!r} does not match workflow.name {loaded.name!r}"
        )
    return loaded


def load_workflow(name: str) -> Workflow:
    """Load only the named workflow entry point."""
    matches = list(entry_points(group=WORKFLOW_ENTRY_POINT_GROUP, name=name))
    if not matches:
        raise WorkflowNotFoundError(f"Unknown workflow: {name!r}")
    if len(matches) > 1:
        raise ValueError(f"Multiple workflow entry points registered for {name!r}")
    return _load_entry_point(matches[0])


def load_workflows() -> dict[str, Workflow]:
    """Load all registered workflows from entry points."""
    return {ep.name: _load_entry_point(ep) for ep in entry_points(group=WORKFLOW_ENTRY_POINT_GROUP)}


# --- Internal graph AST records ---


@dataclass(frozen=True)
class StepNode:
    name: str
    fn: Callable[[Any, JobInfo], dict[str, Any]]


@dataclass(frozen=True)
class AgentNode:
    name: str
    role: Role
    expects: type[BaseModel]
    ask: Callable[[Any], Turn]
    then: Callable[[Any, Any], dict[str, Any]]
    check: str | None = None


@dataclass(frozen=True)
class WaitNode:
    name: str
    reply: type[BaseModel]
    ask: Callable[[Any], Wait]
    then: Callable[[Any, Any], dict[str, Any]]


GraphNode = StepNode | AgentNode | WaitNode


@dataclass(frozen=True)
class BranchEdge:
    src: str
    route: Callable[[Any], str]
    targets: dict[str, str]


@dataclass(frozen=True)
class FanOutEdge:
    src: str
    dst: str
    item_type: type[BaseModel]
    items: Callable[[Any], Sequence[Any]]
