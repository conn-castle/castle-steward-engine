"""End-to-end tests for AgentLayer, Engine, Store, Graph validation, MCP tools, and recovery."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import threading
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import pytest
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from castle_steward_engine import (
    END,
    AgentLayer,
    AgentLayerError,
    AgentSpec,
    AppendList,
    Engine,
    EngineError,
    Graph,
    IncompleteWork,
    JobBusy,
    JobInfo,
    JobNotFoundError,
    QuotaReading,
    QuotaSnapshot,
    Role,
    Turn,
    TurnFailed,
    UnknownWaitError,
    Wait,
    Workflow,
    WorkflowVersionMismatchError,
)
from castle_steward_engine.agent_layer import EXIT_QUOTA_NONE, EXIT_QUOTA_PARTIAL
from castle_steward_engine.mcp_server import SessionDir
from castle_steward_engine.store import METADATA, Store, TurnRecord, include_name
from castle_steward_engine.testing import (
    FakeAgentLayer,
    FakeTurnContext,
    FakeTurnOutcome,
    SimulatedCrash,
)

SPEC = AgentSpec(agent="codex", model="gpt-test")
QUOTA_NOW = "2026-11-27T12:00:00Z"
QUOTA_WEEK_START = "2026-11-24T00:00:00Z"  # 3.5 days (50%) elapsed at QUOTA_NOW
QUOTA_WEEK_RESET = "2026-12-01T00:00:00Z"
QUOTA_WEEK_SECONDS = 604800
QUOTA_SHORT_SECONDS = 18000


class DemoInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    items: list[str] = Field(default_factory=lambda: ["a", "b"])
    loops: int = 1


class StepChoice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    go: bool
    note: str


class WorkItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    label: str


class WorkResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    value: str


class FinalReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    summary: str


class ConfirmReply(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ok: bool


class DemoState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: DemoInput
    iteration: int = 0
    note: str = ""
    results: AppendList[str] = Field(default_factory=list)
    approved: bool = False
    summary: str = ""


LEAD = Role(
    name="lead",
    tools={"submit_choice": StepChoice, "submit_report": FinalReport},
    can_ask_human=True,
)
WORKER = Role(name="worker", tools={"submit_work": WorkResult}, can_ask_human=True)


def _make_workflow(
    version: str = "1.0.0", on_step: Any = None, workdir: Path | None = None
) -> Workflow:
    g: Graph[DemoState] = Graph()

    def _prep(s: DemoState, info: JobInfo) -> dict[str, Any]:
        if on_step:
            on_step(info)
        return {"iteration": s.iteration + 1}

    g.step("prep", _prep)
    g.agent(
        "decide",
        role=LEAD,
        expects=StepChoice,
        ask=lambda s: Turn(
            session="lead", prompt=f"iter {s.iteration}", agent=SPEC, workdir=workdir
        ),
        then=lambda _s, r: {"note": r.note},
    )
    g.agent(
        "work",
        role=WORKER,
        expects=WorkResult,
        ask=lambda item: Turn(
            session=f"w-{item.label}", prompt=item.label, agent=SPEC, workdir=workdir
        ),
        then=lambda _item, r: {"results": [r.value]},
    )
    g.wait(
        "confirm",
        reply=ConfirmReply,
        ask=lambda s: Wait(
            kind="confirm", prompt=f"ok {len(s.results)}?", payload={"n": len(s.results)}
        ),
        then=lambda _s, r: {"approved": r.ok},
    )
    g.agent(
        "finish",
        role=LEAD,
        expects=FinalReport,
        ask=lambda s: Turn(
            session="lead", prompt=f"finish {s.results}", agent=SPEC, workdir=workdir
        ),
        then=lambda _s, r: {"summary": r.summary},
    )
    g.entry("prep")
    g.edge("prep", "decide")
    g.fan_out("decide", "work", WorkItem, lambda s: [WorkItem(label=x) for x in s.input.items])
    g.edge("work", "confirm")
    g.branch(
        "confirm",
        lambda s: "loop" if s.iteration < s.input.loops else "done",
        {"loop": "prep", "done": "finish"},
    )
    g.edge("finish", END)
    return Workflow(name="demo", version=version, input=DemoInput, state=DemoState, graph=g)


def test_role_and_graph_validation() -> None:
    with pytest.raises(ValueError, match="built-in"):
        Role(name="r", tools={"ask": StepChoice})
    with pytest.raises(ValueError, match="multiple tools"):
        Role(name="r", tools={"a": StepChoice, "b": StepChoice})
    with pytest.raises(ValueError, match="Invalid tool name"):
        Role(name="r", tools={"bad name": StepChoice})
    with pytest.raises(ValueError, match="built-in"):
        Role(name="r", checks={"check_done": lambda _c: StepChoice(go=True, note="x")})
    with pytest.raises(ValueError, match="Invalid check name"):
        Role(name="r", checks={"bad check": lambda _c: StepChoice(go=True, note="x")})
    with pytest.raises(TypeError, match="callable"):
        Role(name="r", checks={"c": "not-callable"})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="not registered"):
        WORKER.tool_name(StepChoice)
    with pytest.raises(ValueError, match="not registered"):
        WORKER.check_fn("missing")
    with pytest.raises(TypeError):
        LEAD.tools["x"] = WorkResult  # type: ignore[index]
    lead_copy = Role(
        name="lead",
        tools={"submit_report": FinalReport, "submit_choice": StepChoice},
        can_ask_human=True,
    )
    assert lead_copy == LEAD and hash(lead_copy) == hash(LEAD)
    assert len({LEAD, lead_copy, WORKER}) == 2

    # Branch directly off a fan_out target is rejected
    g_fo: Graph[DemoState] = Graph()
    g_fo.step("src", lambda _s, _j: {})
    g_fo.step("dst", lambda _s, _j: {})
    g_fo.entry("src")
    g_fo.fan_out("src", "dst", WorkItem, lambda _s: [WorkItem(label="a")])
    g_fo.branch("dst", lambda _s: "end", {"end": END})
    with pytest.raises(ValueError, match="fan-out target"):
        g_fo.validate()

    # Multiple outgoing routing declarations from one node are rejected
    for setup in (
        lambda g: g.edge("a", "b").edge("a", END),
        lambda g: g.edge("a", "b").branch("a", lambda _s: "x", {"x": END}),
        lambda g: g.branch("a", lambda _s: "x", {"x": END}).fan_out(
            "a", "b", WorkItem, lambda _s: []
        ),
    ):
        g_dup: Graph[DemoState] = Graph()
        g_dup.step("a", lambda _s, _j: {}).step("b", lambda _s, _j: {}).entry("a")
        setup(g_dup)
        with pytest.raises(ValueError, match="multiple outgoing"):
            g_dup.validate()

    # AgentNode.expects must be in role.tools.values()
    g_bad_exp: Graph[DemoState] = Graph()
    g_bad_exp.agent(
        "a",
        role=WORKER,
        expects=StepChoice,
        ask=lambda _s: Turn(session="w", prompt="p"),
        then=lambda _s, _r: {},
    ).entry("a").edge("a", END)
    with pytest.raises(ValueError, match="not in role"):
        g_bad_exp.validate()

    g_bad_chk: Graph[DemoState] = Graph()
    g_bad_chk.agent(
        "a",
        role=WORKER,
        expects=WorkResult,
        check="missing",
        ask=lambda _s: Turn(session="w", prompt="p"),
        then=lambda _s, _r: {},
    ).entry("a").edge("a", END)
    with pytest.raises(ValueError, match="references check"):
        g_bad_chk.validate()

    # Dead-end node without outgoing route is rejected
    g_dead: Graph[DemoState] = Graph()
    g_dead.step("a", lambda _s, _j: {}).step("b", lambda _s, _j: {}).entry("a").edge("a", "b")
    with pytest.raises(ValueError, match="no outgoing route"):
        g_dead.validate()

    # Unreachable node is rejected
    g_unreach: Graph[DemoState] = Graph()
    (
        g_unreach.step("a", lambda _s, _j: {})
        .step("orphan", lambda _s, _j: {})
        .entry("a")
        .edge("a", END)
        .edge("orphan", END)
    )
    with pytest.raises(ValueError, match="unreachable"):
        g_unreach.validate()

    # Plain edge, branch, or entry targeting a fan_out destination is rejected
    g_fo_plain: Graph[DemoState] = Graph()
    (
        g_fo_plain.step("a", lambda _s, _j: {})
        .step("b", lambda _s, _j: {})
        .step("w", lambda _s, _j: {})
        .entry("a")
        .edge("a", "w")
        .fan_out("b", "w", WorkItem, lambda _s: [])
        .edge("w", END)
    )
    with pytest.raises(ValueError, match="edge or branch"):
        g_fo_plain.validate()

    g_fo_branch: Graph[DemoState] = Graph()
    (
        g_fo_branch.step("a", lambda _s, _j: {})
        .step("b", lambda _s, _j: {})
        .step("w", lambda _s, _j: {})
        .entry("a")
        .branch("a", lambda _s: "x", {"x": "b", "y": "w"})
        .fan_out("b", "w", WorkItem, lambda _s: [])
        .edge("w", END)
    )
    with pytest.raises(ValueError, match="edge or branch"):
        g_fo_branch.validate()

    g_fo_entry: Graph[DemoState] = Graph()
    g_fo_entry.step("a", lambda _s, _j: {}).entry("a").fan_out("a", "a", WorkItem, lambda _s: [])
    with pytest.raises(ValueError, match="entry node"):
        g_fo_entry.validate()


def test_real_agent_layer_cli_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    responses: list[tuple[int, dict[str, Any]]] = [
        (
            0,
            {
                "agents": [
                    {
                        "agent": "codex",
                        "available": True,
                        "model": {"configured": "gpt-x"},
                        "reasoning_effort": {"configured": "high"},
                        "session_mcp_servers": {"supported": True},
                    },
                    {
                        "agent": "legacy",
                        "available": True,
                        "model": {"configured": ""},
                        "session_mcp_servers": {"supported": False},
                    },
                ]
            },
        ),
        (
            0,
            {
                "handle": "h1",
                "invocation_id": "inv1",
                "state": "reserved",
                "reservation_expires_at": "2026-11-26T00:00:00Z",
            },
        ),
        (82, {}),  # start already started -> inspect by invocation_id
        (0, {"handle": "h1", "invocation_id": "inv1", "state": "running"}),
        (0, {"handle": "h1", "invocation_id": "inv2", "state": "reserved"}),
        (82, {}),  # continue already started -> inspect by reservation_id
        (0, {"handle": "h1", "invocation_id": "inv2", "state": "running"}),
        (
            0,
            {
                "handle": "h1",
                "invocation_id": "inv2",
                "state": "running",
                "condition_met": False,
                "last_activity_at": "2026-11-26T00:00:01Z",
            },
        ),
        (
            0,
            {
                "handle": "h1",
                "invocation_id": "inv2",
                "state": "running",
                "result_path": "/tmp/result.json",
                "error": None,
                "termination_confirmed": True,
                "condition_met": True,
                "future_additive_field": {"nested": True},
            },
        ),
        (0, {"handle": "h1", "invocation_id": "inv2"}),  # missing required `state`
    ]

    def fake_run(cmd: list[str], **_kw: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        rc, payload = responses.pop(0)
        return subprocess.CompletedProcess(cmd, rc, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    al = AgentLayer("al")
    snap = QuotaSnapshot(
        by_agent={"claude": QuotaReading(pace_ratio=0.7, observed_at="2026-11-26T00:00:00Z")},
        by_model={"claude": {"fable": QuotaReading(blocked=True, pace_ratio=0.2)}},
    )
    assert snap.for_spec(AgentSpec(agent="claude", model="fable")) == QuotaReading(
        blocked=True, pace_ratio=0.2, observed_at="2026-11-26T00:00:00Z"
    )
    assert snap.for_spec(AgentSpec(agent="claude", model="opus")).pace_ratio == 0.7
    assert snap.for_spec(AgentSpec(agent="codex")) == QuotaReading()
    # Model-level blocked is authoritative (e.g., credit-bypass model stays unblocked);
    # unlisted models still inherit by_agent.blocked.
    provider_blocked = QuotaSnapshot(
        by_agent={
            "claude": QuotaReading(blocked=True, pace_ratio=0.9, observed_at="2026-11-26T01:00:00Z")
        },
        by_model={
            "claude": {
                "fable": QuotaReading(blocked=False, pace_ratio=0.1),
                "opus": QuotaReading(blocked=False, pace_ratio=None),
            }
        },
    )
    assert provider_blocked.for_spec(AgentSpec(agent="claude", model="fable")) == QuotaReading(
        blocked=False, pace_ratio=0.1, observed_at="2026-11-26T01:00:00Z"
    )
    assert provider_blocked.for_spec(AgentSpec(agent="claude", model="opus")) == QuotaReading(
        blocked=False, pace_ratio=0.9, observed_at="2026-11-26T01:00:00Z"
    )
    assert provider_blocked.for_spec(AgentSpec(agent="claude", model="sonnet")) == QuotaReading(
        blocked=True, pace_ratio=0.9, observed_at="2026-11-26T01:00:00Z"
    )
    with pytest.raises(ValidationError):
        QuotaReading(pace_ratio=-0.1)
    opts = al.options(tmp_path)
    assert opts["codex"].session_mcp is True and opts["legacy"].session_mcp is False
    res = al.reserve_start(tmp_path)
    assert res.handle == "h1" and res.invocation_id == "inv1" and res.state == "reserved"
    started = al.start(
        tmp_path,
        reservation_handle="h1",
        invocation_id="inv1",
        agent=SPEC,
        prompt_file=tmp_path / "p.md",
        mcp_config=tmp_path / "m.json",
        role="lead",
    )
    start_cmd = calls[-2]
    assert start_cmd[:4] == ["al", "-q", "dispatch", "start"]
    max_depth_idx = start_cmd.index("--max-depth")
    assert start_cmd[max_depth_idx : max_depth_idx + 2] == ["--max-depth", "1"]
    assert started.invocation_id == "inv1" and calls[-1][-2:] == ["inspect", "inv1"]
    cres = al.reserve_continue(tmp_path, "h1")
    cont = al.continue_session(
        tmp_path, handle="h1", reservation_id=cres.invocation_id, prompt_file=tmp_path / "p.md"
    )
    assert cont.invocation_id == "inv2" and calls[-1][-2:] == ["inspect", "inv2"]
    waited = al.wait(tmp_path, "inv2")
    assert waited.condition_met is True and waited.termination_confirmed is True
    with pytest.raises(ValidationError):
        al.inspect(tmp_path, "inv2")
    assert not responses


def _win(used: Any, dur: int = QUOTA_WEEK_SECONDS, **kw: Any) -> dict[str, Any]:
    return {"used_percent": used, "duration_seconds": dur, "state": "fixed", **kw}


def _pool(pid: str, windows: list[dict[str, Any]], scope: str = "models") -> dict[str, Any]:
    return {"id": pid, "scope": scope, "windows": windows}


def test_agent_layer_quota_parsing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agy_cases = (
        (
            "neon",
            [
                _win(20, starts_at=QUOTA_WEEK_START),
                _win(80, QUOTA_SHORT_SECONDS, starts_at="2026-11-27T09:30:00Z"),
            ],
            0.4,
        ),
        (
            "zero",
            [
                _win(90, starts_at=QUOTA_WEEK_START, state="not_started"),
                _win(0, QUOTA_SHORT_SECONDS),
            ],
            0.0,
        ),
        ("rolling", [_win(35, state="rolling", resets_at=QUOTA_WEEK_RESET)], 0.35),
        ("unk", [_win(40, state="unknown", starts_at=QUOTA_WEEK_START)], None),
        ("early", [_win(1, starts_at="2026-11-27T11:59:00Z")], 0.07),
        ("future", [_win(0, starts_at="2026-11-27T12:05:00Z")], None),
        ("past", [_win(40, resets_at="2026-11-27T11:00:00Z")], 0.4),
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "results": [
            {
                "agent": "claude",
                "status": "ok",
                "observed_at": QUOTA_NOW,
                "blocked": False,
                "pools": [
                    _pool(
                        "claude",
                        [
                            _win(30, starts_at=QUOTA_WEEK_START, resets_at=QUOTA_WEEK_RESET),
                            _win(100, starts_at=QUOTA_WEEK_START, disabled=True),
                        ],
                        "all",
                    ),
                    _pool("fable_wk", [_win(45, resets_at=QUOTA_WEEK_RESET)]),
                    _pool("web", [_win(100, starts_at=QUOTA_WEEK_START)], "other_surface"),
                ],
                "models": [
                    {"id": "fable", "pools": ["claude", "fable_wk", "web"]},
                    {"id": "opus", "pools": ["claude"]},
                ],
            },
            {
                "agent": "antigravity",
                "status": "ok",
                "observed_at": QUOTA_NOW,
                "pools": [_pool(mid, wins) for mid, wins, _ in agy_cases],
                "models": [{"id": mid, "pools": [mid]} for mid, _, _ in agy_cases],
            },
            {
                "agent": "codex",
                "status": "ok",
                "attempted_at": QUOTA_NOW,
                "blocked": True,
                "pools": [
                    _pool(
                        "codex",
                        [_win(25, QUOTA_SHORT_SECONDS, starts_at="2026-11-27T09:30:00Z")],
                        "all",
                    )
                ],
                "models": [{"id": "gpt-credits", "pools": ["codex"], "blocked": False}],
            },
            {
                "agent": "nil",
                "status": "ok",
                "observed_at": QUOTA_NOW,
                "pools": None,
                "models": None,
            },
            {"agent": "muse", "status": "unavailable", "observed_at": QUOTA_NOW, "blocked": None},
            {"agent": "unblk", "status": "error", "observed_at": QUOTA_NOW, "blocked": False},
            {"agent": "grok", "status": "error", "observed_at": QUOTA_NOW, "blocked": True},
        ],
    }
    responses = [
        (EXIT_QUOTA_PARTIAL, json.dumps(payload)),
        (EXIT_QUOTA_NONE, '{"schema_version": 1, "results": []}'),
        (1, "{}"),
        (0, "not-json"),
        (0, '{"schema_version": 2, "results": []}'),
    ]
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **_kw: subprocess.CompletedProcess(cmd, *responses.pop(0), stderr=""),
    )
    al = AgentLayer("al")
    snap = al.quota(tmp_path)
    obs = "2026-11-27T12:00:00+00:00"

    assert snap.for_spec(AgentSpec(agent="claude", model="fable")).pace_ratio == pytest.approx(0.9)
    assert snap.for_spec(AgentSpec(agent="claude", model="opus")).pace_ratio == pytest.approx(0.6)
    assert snap.by_agent["antigravity"].pace_ratio is None
    for mid, _, expected in agy_cases:
        actual = snap.for_spec(AgentSpec(agent="antigravity", model=mid)).pace_ratio
        assert actual is None if expected is None else actual == pytest.approx(expected)

    credit_model = snap.for_spec(AgentSpec(agent="codex", model="gpt-credits"))
    unlisted = snap.for_spec(AgentSpec(agent="codex", model="gpt-6.2-sol"))
    assert credit_model.blocked is False and credit_model.pace_ratio == pytest.approx(0.5)
    assert unlisted.blocked is True and unlisted.pace_ratio == pytest.approx(0.5)
    assert snap.by_agent["nil"] == QuotaReading(blocked=False, observed_at=obs)
    assert "muse" not in snap.by_agent and "unblk" not in snap.by_agent
    assert snap.for_spec(AgentSpec(agent="grok")) == QuotaReading(blocked=True, observed_at=obs)

    assert al.quota(tmp_path) == QuotaSnapshot()
    for _ in range(3):
        with pytest.raises(AgentLayerError):
            al.quota(tmp_path)
    assert not responses


def test_session_dir_forwards_pythonpath_when_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PYTHONPATH", raising=False)
    s1 = SessionDir(tmp_path / "s1")
    s1.init(workflow="demo", role="lead")
    env1 = json.loads(s1.mcp_config_path.read_text(encoding="utf-8"))["mcpServers"][
        "castle-steward"
    ]["env"]
    assert "PATH" in env1 and "PYTHONPATH" not in env1

    monkeypatch.setenv("PYTHONPATH", "/custom/workflows")
    s2 = SessionDir(tmp_path / "s2")
    s2.init(workflow="demo", role="lead")
    env2 = json.loads(s2.mcp_config_path.read_text(encoding="utf-8"))["mcpServers"][
        "castle-steward"
    ]["env"]
    assert env2["PYTHONPATH"] == "/custom/workflows"


def test_fanout_parallel_ask_partial_reply_exclusivity_and_loop(tmp_path: Path) -> None:
    seen_quota: list[QuotaSnapshot] = []
    wf = _make_workflow(
        workdir=tmp_path, on_step=lambda info: seen_quota.append(info.agent_quota())
    )
    asked: set[str] = set()
    lead_tools: list[str] = []

    def handler(ctx: FakeTurnContext) -> None:
        tool = ctx.active_turn.expects_tool
        if tool == "submit_choice":
            lead_tools[:] = ctx.list_tools()
            assert ctx.call_tool("submit_report", {"summary": "early"}).is_error
            assert "Incomplete" in ctx.call_tool("check_done", {}).text
            assert not ctx.call_tool("submit_choice", {"go": True, "note": "ok"}).is_error
            assert ctx.call_tool("ask", {"question": "too late?"}).is_error
            assert "Complete" in ctx.call_tool("check_done", {}).text
        elif tool == "submit_work":
            label = ctx.prompt if not ctx.is_continue else ("x" if "from-x" in ctx.prompt else "y")
            if ctx.handle not in asked and not ctx.is_continue:
                asked.add(ctx.handle)
                assert not ctx.call_tool("ask", {"question": f"need {label}?"}).is_error
                assert ctx.call_tool("submit_work", {"value": "bad"}).is_error
                assert "Waiting for human input" in ctx.call_tool("check_done", {}).text
                return
            ctx.call_tool("submit_work", {"value": ctx.prompt.splitlines()[-1]})
        else:
            assert ctx.is_continue
            ctx.call_tool("submit_report", {"summary": "all done"})

    expected_quota = QuotaSnapshot(by_agent={"codex": QuotaReading(pace_ratio=0.4)})
    al = FakeAgentLayer({"demo": wf}, handler, quota=lambda _wd: expected_quota)
    with Engine(state_dir=tmp_path, agent_layer=al, workflows={"demo": wf}) as eng:
        jid = eng.start("demo", {"items": ["x", "y"], "loops": 1})
        st = eng.run(jid)
        assert seen_quota == [expected_quota]
        assert st.phase == "paused" and len(st.pending_waits) == 2
        w1, w2 = st.pending_waits
        with pytest.raises(ValueError):
            eng.reply(jid, w1.wait_id, "   ")
        eng.reply(jid, w1.wait_id, f"from-{'x' if 'need x' in w1.prompt else 'y'}")
        st = eng.run(jid)
        assert st.phase == "paused" and len(st.pending_waits) == 1
        rem = st.pending_waits[0]
        assert rem.wait_id == w2.wait_id
        eng.reply(jid, rem.wait_id, f"from-{'x' if 'need x' in rem.prompt else 'y'}")

        st = eng.run(jid)
        assert st.phase == "paused" and st.pending_waits[0].kind == "confirm"
        wid = st.pending_waits[0].wait_id
        with pytest.raises(ValidationError):
            eng.reply(jid, wid, {"ok": True, "extra": 1})
        with pytest.raises(UnknownWaitError):
            eng.reply(jid, "missing", {"ok": True})
        eng.reply(jid, wid, {"ok": True})

        st = eng.run(jid)
        assert st.phase == "completed" and st.state is not None
        assert sorted(str(v) for v in st.state["results"]) == ["from-x", "from-y"]  # type: ignore[union-attr]
        assert lead_tools == ["submit_choice", "submit_report", "ask", "check_done"]


def test_errors_nudge_empty_fanout_and_missing_workdir(tmp_path: Path) -> None:
    wf_no_wd = _make_workflow(workdir=None)
    al = FakeAgentLayer({"demo": wf_no_wd}, lambda _c: None)
    with Engine(state_dir=tmp_path / "s1", agent_layer=al, workflows={"demo": wf_no_wd}) as eng:
        jid = eng.start("demo", {"items": ["a"]})
        with pytest.raises(EngineError, match="workdir"):
            eng.run(jid)

    mode = {"nudge_submit": True}

    def handler(ctx: FakeTurnContext) -> FakeTurnOutcome:
        if ctx.active_turn.expects_tool == "submit_choice":
            if not ctx.is_continue:
                return FakeTurnOutcome()
            if mode["nudge_submit"]:
                ctx.call_tool("submit_choice", {"go": True, "note": "nudged"})
            return FakeTurnOutcome()
        ctx.call_tool("submit_work", {"value": ctx.prompt})
        return FakeTurnOutcome()

    wf = _make_workflow(workdir=tmp_path)
    al2 = FakeAgentLayer({"demo": wf}, handler)
    with Engine(state_dir=tmp_path / "s2", agent_layer=al2, workflows={"demo": wf}) as eng:
        j1 = eng.start("demo", {"items": ["a"]})
        st = eng.run(j1)
        assert st.phase == "paused" and st.state is not None and st.state["note"] == "nudged"

        j_empty = eng.start("demo", {"items": []})
        with pytest.raises(EngineError, match="no items"):
            eng.run(j_empty)

        mode["nudge_submit"] = False
        calls_before = len(al2.calls)
        j2 = eng.start("demo", {"items": ["a"]})
        with pytest.raises(IncompleteWork):
            eng.run(j2)
        assert [c.is_continue for c in al2.calls[calls_before:]] == [False, True]
        mode["nudge_submit"] = True
        st2 = eng.run(j2)
        assert st2.phase == "paused" and st2.state is not None and st2.state["note"] == "nudged"
        decide_calls = [c for c in al2.calls[calls_before:] if c.expects_tool == "submit_choice"]
        assert [c.is_continue for c in decide_calls] == [False, True, True]


def test_crash_recovery_all_boundaries_and_immutable_session_id(tmp_path: Path) -> None:
    wf = _make_workflow(workdir=tmp_path)
    counts = {"decide": 0, "finish": 0}
    flags = {"pre": True, "after_start": True, "on_wait": True, "after_cont": True}

    def handler(ctx: FakeTurnContext) -> FakeTurnOutcome:
        tool = ctx.active_turn.expects_tool
        if tool == "submit_choice":
            counts["decide"] += 1
            if counts["decide"] == 1:
                return FakeTurnOutcome(state="running", provider_started=False)
            ctx.call_tool("submit_choice", {"go": True, "note": "ok"})
            return FakeTurnOutcome()
        if tool == "submit_work":
            ctx.call_tool("submit_work", {"value": ctx.prompt})
            return FakeTurnOutcome()
        counts["finish"] += 1
        if counts["finish"] == 1:
            return FakeTurnOutcome(state="failed", provider_started=True, error="provider exploded")
        ctx.call_tool("submit_report", {"summary": "done"})
        return FakeTurnOutcome()

    def crash_before(_inv: str, is_cont: bool) -> bool:
        if not is_cont and flags["pre"]:
            flags["pre"] = False
            return True
        return False

    def crash_after(ctx: FakeTurnContext) -> bool:
        if (
            ctx.active_turn.expects_tool == "submit_choice"
            and counts["decide"] == 2
            and flags["after_start"]
        ):
            flags["after_start"] = False
            return True
        if (
            ctx.active_turn.expects_tool == "submit_report"
            and counts["finish"] == 2
            and flags["after_cont"]
        ):
            flags["after_cont"] = False
            return True
        return False

    def crash_wait(_inv: str) -> bool:
        if counts["decide"] == 2 and not flags["after_start"] and flags["on_wait"]:
            flags["on_wait"] = False
            return True
        return False

    al = FakeAgentLayer(
        {"demo": wf},
        handler,
        crash_before_launch=crash_before,
        crash_after_launch=crash_after,
        crash_on_wait=crash_wait,
    )
    with Engine(state_dir=tmp_path, agent_layer=al, workflows={"demo": wf}) as eng:
        jid = eng.start("demo", {"items": ["a"]})
        with pytest.raises(SimulatedCrash):
            eng.run(jid)
        sess1 = eng.store.get_session(jid, "lead")
        assert sess1 is not None and not al.adopted

        with pytest.raises(TurnFailed):
            eng.run(jid)

        with pytest.raises(SimulatedCrash):
            eng.run(jid)
        sess2 = eng.store.get_session(jid, "lead")
        assert sess2 is not None and sess2.session_id == sess1.session_id
        assert sess2.al_handle != sess1.al_handle

        with pytest.raises(SimulatedCrash):
            eng.run(jid)
        assert len(al.adopted) == 1

        st = eng.run(jid)
        assert st.phase == "paused" and counts["decide"] == 2
        eng.reply(jid, st.pending_waits[0].wait_id, {"ok": True})

        with pytest.raises(TurnFailed, match="provider exploded"):
            eng.run(jid)
        with pytest.raises(SimulatedCrash):
            eng.run(jid)
        st = eng.run(jid)
        assert st.phase == "completed" and counts["finish"] == 2 and len(al.adopted) == 2


def test_stop_drain_unanswered_pause_nonblocking_inspect_and_version(tmp_path: Path) -> None:
    entered = threading.Event()
    proceed = threading.Event()

    def on_step(_info: JobInfo) -> None:
        entered.set()
        proceed.wait(timeout=5)

    wf_v1 = _make_workflow(version="1.0.0", on_step=on_step, workdir=tmp_path)

    def handler(ctx: FakeTurnContext) -> None:
        tool = ctx.active_turn.expects_tool
        if tool == "submit_choice":
            ctx.call_tool("submit_choice", {"go": True, "note": "ok"})
        elif tool == "submit_work":
            ctx.call_tool("submit_work", {"value": ctx.prompt})
        else:
            ctx.call_tool("submit_report", {"summary": "done"})

    al = FakeAgentLayer({"demo": wf_v1}, handler)
    with Engine(state_dir=tmp_path, agent_layer=al, workflows={"demo": wf_v1}) as eng:
        jid = eng.start("demo", {"items": ["a"]})
        holder: dict[str, Any] = {}
        t = threading.Thread(target=lambda: holder.setdefault("st", eng.run(jid)))
        t.start()
        assert entered.wait(timeout=5)
        assert eng.inspect(jid).phase == "running"
        with pytest.raises(JobBusy):
            eng.run(jid)
        eng.stop(jid)
        assert eng.inspect(jid).phase == "stopping"
        proceed.set()
        t.join(timeout=5)
        assert holder["st"].phase == "stopped"

        st = eng.run(jid)
        assert st.phase == "paused"
        eng.stop(jid)
        assert eng.run(jid).phase == "stopped"

        eng.reply(jid, st.pending_waits[0].wait_id, {"ok": True})
        assert eng.run(jid).phase == "completed"

    wf_v2 = _make_workflow(version="2.0.0", workdir=tmp_path)
    with (
        Engine(state_dir=tmp_path, agent_layer=al, workflows={"demo": wf_v2}) as eng2,
        pytest.raises(WorkflowVersionMismatchError),
    ):
        eng2.run(jid)


def test_alembic_metadata_parity_ignores_langgraph_tables(tmp_path: Path) -> None:
    store = Store(tmp_path)
    try:
        cp_conn = sqlite3.connect(str(store.db_path))
        try:
            cp_conn.execute("CREATE TABLE checkpoints (thread_id TEXT PRIMARY KEY)")
            cp_conn.commit()
        finally:
            cp_conn.close()

        with store._read_engine.connect() as conn:
            ctx = MigrationContext.configure(conn, opts={"include_name": include_name})
            assert compare_metadata(ctx, METADATA) == []
    finally:
        store.close()


def test_store_reads_do_not_block_during_write_lock(tmp_path: Path) -> None:
    store = Store(tmp_path)
    try:
        jid = uuid4()
        store.create_job(jid, workflow="demo", workflow_version="1.0.0", input_json="{}")
        writer = sqlite3.connect(str(store.db_path), timeout=0.1, isolation_level=None)
        try:
            writer.execute("BEGIN IMMEDIATE")
            assert store.get_job(jid).job_id == jid
            reader = Store(tmp_path)
            try:
                assert reader.get_job(jid).job_id == jid
            finally:
                reader.close()
        finally:
            writer.execute("ROLLBACK")
            writer.close()

        # Loud errors on missing rows (and no orphan turn inserted on failed session update)
        missing = uuid4()
        orphan_turn = TurnRecord(
            job_id=missing,
            turn_key="k:0",
            attempt=1,
            node="decide",
            session_name="lead",
            al_handle="h1",
            al_invocation_id="inv1",
            status="reserved",
            prompt_path=str(tmp_path / "p.md"),
            created_at="2026-11-26T00:00:00+00:00",
        )
        with pytest.raises(JobNotFoundError):
            store.request_stop(missing)
        with pytest.raises(JobNotFoundError):
            store.set_last_error(missing, "boom")
        with pytest.raises(LookupError):
            store.update_session_handle_and_create_turn(missing, "lead", "h2", orphan_turn)
        assert store.get_latest_turn(missing, "k:0") is None
        with pytest.raises(LookupError):
            store.update_turn_status(missing, "k:0", 1, "completed")
    finally:
        store.close()


class _StrictInner(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")
    count: int = Field(ge=0)


class _TagA(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["a"] = "a"


class _TagB(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["b"] = "b"


class _StrictState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")
    input: DemoInput
    inner: _StrictInner = Field(default_factory=lambda: _StrictInner(count=0))
    n: int = Field(default=0, ge=0)
    tagged: _TagA | _TagB = Field(default_factory=_TagA, discriminator="kind")


def test_node_state_update_validation_rejects_unknown_keys_and_invalid_values(
    tmp_path: Path,
) -> None:
    bad_updates: list[tuple[Any, str]] = [
        ("not-a-dict", r"'bad'.*dict"),
        ({"bogus": 1}, r"'bad'.*bogus"),
        ({"n": -1}, r"'bad'.*'n'"),
        ({"inner": _StrictInner(count=0).model_copy(update={"count": -1})}, r"'bad'.*'inner'"),
        ({"tagged": {"kind": "c"}}, r"'bad'.*'tagged'[\s\S]*union_tag_invalid"),
    ]
    for i, (payload, pattern) in enumerate(bad_updates):
        g: Graph[_StrictState] = Graph()

        def _ok(_s: _StrictState, _j: JobInfo) -> dict[str, Any]:
            return {"n": 2, "tagged": {"kind": "b"}}

        def _bad(_s: _StrictState, _j: JobInfo, p: Any = payload) -> Any:
            return p

        g.step("ok", _ok).step("bad", _bad).entry("ok").edge("ok", "bad").edge("bad", END)
        wf = Workflow(name=f"bad{i}", version="1.0.0", input=DemoInput, state=_StrictState, graph=g)
        al = FakeAgentLayer({wf.name: wf}, lambda _c: None)
        with Engine(state_dir=tmp_path / f"s{i}", agent_layer=al, workflows={wf.name: wf}) as eng:
            jid = eng.start(wf.name, {})
            with pytest.raises(EngineError, match=pattern):
                eng.run(jid)
            st = eng.inspect(jid)
            assert st.phase == "failed" and st.last_error is not None
            assert "EngineError" in st.last_error and st.state is not None
            assert st.state["n"] == 2
            assert st.state["inner"] == {"count": 0}
            assert st.state["tagged"] == {"kind": "b"}
            assert "bogus" not in st.state


def test_check_only_completion_nudge_invalidation_and_inactive_tool(tmp_path: Path) -> None:
    live = {"ok": False, "value": "v1"}

    def _check(ctx: dict[str, Any]) -> WorkResult:
        if not live["ok"]:
            raise ValueError(f"not ready for {ctx['tag']}")
        return WorkResult(value=str(live["value"]))

    role = Role(
        name="checker",
        tools={"submit_work": WorkResult},
        checks={"done": _check},
        can_ask_human=True,
    )
    g: Graph[DemoState] = Graph()
    g.agent(
        "do_work",
        role=role,
        expects=WorkResult,
        check="done",
        ask=lambda _s: Turn(
            session="checker",
            prompt="run check",
            context={"tag": "x"},
            agent=SPEC,
            workdir=tmp_path,
        ),
        then=lambda _s, r: {"summary": r.value},
    ).entry("do_work").edge("do_work", END)
    wf = Workflow(name="chk", version="1.0.0", input=DemoInput, state=DemoState, graph=g)

    # Inactive tool error, check failure clears stale submission.json, missing check_done nudges
    turn_idx = 0

    def handler_ok(ctx: FakeTurnContext) -> None:
        nonlocal turn_idx
        turn_idx += 1
        assert ctx.list_tools() == ["submit_work", "ask", "check_done"]
        if turn_idx == 1:
            bad_tool = ctx.call_tool("submit_work", {"value": "nope"})
            assert bad_tool.is_error and "check_done" in bad_tool.text
            live["ok"] = True
            passed = ctx.call_tool("check_done", {})
            assert not passed.is_error and "Complete" in passed.text
            sub_file = Path(ctx.active_turn.turn_dir) / "submission.json"
            assert sub_file.is_file()
            live["ok"] = False
            failed = ctx.call_tool("check_done", {})
            assert (
                failed.is_error
                and "Incomplete:" in failed.text
                and "not ready for x" in failed.text
            )
            assert not sub_file.exists()
            return
        assert "without verifying completion with check_done" in ctx.prompt
        live["ok"] = True
        live["value"] = "final"
        assert not ctx.call_tool("check_done", {}).is_error

    al = FakeAgentLayer({"chk": wf}, handler_ok)
    with Engine(state_dir=tmp_path / "s1", agent_layer=al, workflows={"chk": wf}) as eng:
        jid = eng.start("chk", {})
        st = eng.run(jid)
        assert st.phase == "completed" and st.state is not None and st.state["summary"] == "final"
        assert [c.expects_check for c in al.calls] == ["done", "done"]
        assert [c.expects_tool for c in al.calls] == [None, None]

    # Post-check_done live drift triggers nudge; persistent drift raises IncompleteWork & recovers
    drift_round = 0

    def handler_drift(ctx: FakeTurnContext) -> None:
        nonlocal drift_round
        drift_round += 1
        live["ok"] = True
        live["value"] = f"r{drift_round}"
        assert not ctx.call_tool("check_done", {}).is_error
        if drift_round == 1:
            live["value"] = "mutated-after-check"
        elif drift_round == 2:
            assert "invalid state after check_done" in ctx.prompt
            live["ok"] = False
        else:
            assert "invalid state after check_done" in ctx.prompt

    al2 = FakeAgentLayer({"chk": wf}, handler_drift)
    with Engine(state_dir=tmp_path / "s2", agent_layer=al2, workflows={"chk": wf}) as eng2:
        j2 = eng2.start("chk", {})
        with pytest.raises(IncompleteWork, match="invalid state after check_done"):
            eng2.run(j2)
        recovered = eng2.run(j2)
        assert recovered.phase == "completed" and recovered.state is not None
        assert recovered.state["summary"] == "r3"
        assert [c.is_continue for c in al2.calls] == [False, True, True]

    # Drift on round 0 -> nudge -> round 1 restores state & asks human -> resume delivers reply
    ask_round = 0

    def handler_drift_then_ask(ctx: FakeTurnContext) -> None:
        nonlocal ask_round
        ask_round += 1
        if ask_round == 1:
            live["ok"] = True
            live["value"] = "r0"
            assert not ctx.call_tool("check_done", {}).is_error
            live["value"] = "drifted"
        elif ask_round == 2:
            assert "invalid state after check_done" in ctx.prompt
            live["value"] = "r0"  # Restored to round-0 value; stale submission must NOT replay!
            assert not ctx.call_tool("ask", {"question": "confirm value?"}).is_error
        else:
            assert "use-human-value" in ctx.prompt
            live["value"] = "use-human-value"
            assert not ctx.call_tool("check_done", {}).is_error

    al3 = FakeAgentLayer({"chk": wf}, handler_drift_then_ask)
    with Engine(state_dir=tmp_path / "s3", agent_layer=al3, workflows={"chk": wf}) as eng3:
        j3 = eng3.start("chk", {})
        paused = eng3.run(j3)
        assert paused.phase == "paused" and paused.pending_waits[0].prompt == "confirm value?"
        eng3.reply(j3, paused.pending_waits[0].wait_id, "use-human-value")
        done = eng3.run(j3)
        assert done.phase == "completed" and done.state is not None
        assert done.state["summary"] == "use-human-value"


class _BranchIn(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _BranchBaseState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    input: _BranchIn
    flag: bool = False


class _BranchSubState(_BranchBaseState):
    extra_val: str = "kept"


def test_branch_router_preserves_subclass_state(tmp_path: Path) -> None:
    seen: list[tuple[type[BaseModel], str]] = []
    bad = False

    def route(state: _BranchBaseState) -> str:
        assert isinstance(state, _BranchSubState)
        seen.append((type(state), state.extra_val))
        if bad:
            return "bogus"
        return "done" if state.flag else "go"

    g: Graph[_BranchSubState] = Graph()
    g.step("init", lambda _s, _j: {"flag": True, "extra_val": "updated"})
    g.step("finish", lambda _s, _j: {})
    g.entry("init")
    g.branch("init", route, {"go": "init", "done": "finish"})
    g.edge("finish", END)
    wf = Workflow(name="sub", version="1.0.0", input=_BranchIn, state=_BranchSubState, graph=g)
    al = FakeAgentLayer({"sub": wf}, lambda _c: None)
    with Engine(state_dir=tmp_path / "st", agent_layer=al, workflows={"sub": wf}) as eng:
        st = eng.run(eng.start("sub", {}))
        assert st.phase == "completed" and st.state == {
            "input": {},
            "flag": True,
            "extra_val": "updated",
        }
        assert seen == [(_BranchSubState, "updated")]
        bad = True
        with pytest.raises(EngineError, match=r"'init'.*'bogus'.*done, go"):
            eng.run(eng.start("sub", {}))
