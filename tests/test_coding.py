"""End-to-end tests for `castle_steward_coding` against real isolated Git repos + fake `gh`."""

from __future__ import annotations

import ast
import json
import os
import stat
import string
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from castle_steward_coding import (
    CODE_AND_SHIP_WORKFLOW,
    CODE_WORKFLOW,
    SHIP_WORKFLOW,
    CodeAndShipState,
    CodeInput,
    CodeState,
    CodingConfig,
    CodingSettings,
    ShipInput,
    ShipState,
    prompt_templates,
)
from castle_steward_coding.github import (
    GitHubError,
    MergeResult,
    PrReadiness,
    is_ignored,
    merge_pr_exact_head,
    pr_readiness,
    verify_commit_lineage,
    verify_shipped_pr,
    wait_for_readiness,
)
from castle_steward_coding.models import (
    CodeRevision,
    EscalationRecord,
    MergeApprovalReply,
    MergeGateVerdict,
    PlanVersion,
    PrAttempt,
    PullRequest,
    RepairTriageVerdict,
    ReviewItem,
    ReviewRecord,
    RunSetup,
    ShipStateBase,
    VerifiedPr,
    WorkflowState,
)
from castle_steward_coding.prompts import (
    address_review_prompt,
    coordinate_prompt,
    implement_prompt,
    merge_gate_prompt,
    render_merge_gate_context,
    repair_prompt,
    review_code_prompt,
    review_plan_prompt,
    revise_plan_prompt,
    ship_pr_prompt,
    task_summary,
    triage_repair_prompt,
    write_plan_prompt,
)
from castle_steward_coding.tools import TaskAssessment, WrittenPlan
from castle_steward_engine import AgentSpec, AvailableAgent, Engine, QuotaReading, QuotaSnapshot
from castle_steward_engine.mcp_server import SessionToolHost
from castle_steward_engine.testing import FakeAgentLayer, FakeTurnContext

_ALL_AGENTS: dict[str, AvailableAgent] = {
    "antigravity": AvailableAgent(agent="antigravity", session_mcp=True),
    "claude": AvailableAgent(agent="claude", default_model="opus", session_mcp=True),
    "codex": AvailableAgent(agent="codex", default_model="gpt-6.2-sol", session_mcp=True),
    "grok": AvailableAgent(agent="grok", default_model="grok-4.7", session_mcp=True),
    "muse": AvailableAgent(agent="muse", default_model="muse-wonder-1.0", session_mcp=True),
}


_WORKFLOWS = {
    "code-and-ship": CODE_AND_SHIP_WORKFLOW,
    "code": CODE_WORKFLOW,
    "ship": SHIP_WORKFLOW,
}


def _fake_al(
    handler: Any,
    *,
    available_agents: dict[str, AvailableAgent] | None = None,
    quota: QuotaSnapshot | None = None,
) -> FakeAgentLayer:
    return FakeAgentLayer(
        _WORKFLOWS,
        handler,
        available_agents=_ALL_AGENTS if available_agents is None else available_agents,
        quota=quota,
    )


@pytest.fixture(autouse=True)
def _isolate_git_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith(("GIT_", "STEW_ENGINE_CODING_")):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Test User")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "test@example.com")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Test User")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "test@example.com")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _init_repo(
    tmp_path: Path,
    name: str = "repo",
    *,
    gitignore_path: str | None = ".gitignore",
    gitignore: str = "/.agent-layer/\n",
) -> Path:
    repo = tmp_path / name
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    (repo / "README.md").write_text("init\n", encoding="utf-8")
    files = ["README.md"]
    if gitignore_path is not None:
        gi = repo / gitignore_path
        gi.parent.mkdir(parents=True, exist_ok=True)
        gi.write_text(gitignore, encoding="utf-8")
        files.append(gitignore_path)
    _git(repo, "add", *files)
    _git(repo, "commit", "-m", "initial")
    _git(repo, "checkout", "-b", "feat/task")
    return repo


def _commit(repo: Path, filename: str, content: str, msg: str) -> str:
    (repo / filename).write_text(content, encoding="utf-8")
    _git(repo, "add", filename)
    _git(repo, "commit", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


def _install_fake_gh(gh_dir: Path, monkeypatch: pytest.MonkeyPatch, state: dict[str, Any]) -> Path:
    gh_dir.mkdir(parents=True, exist_ok=True)
    state_file = gh_dir / "gh_state.json"
    state_file.write_text(json.dumps(state), encoding="utf-8")
    bin_dir = gh_dir / "bin"
    bin_dir.mkdir(exist_ok=True)
    gh = bin_dir / "gh"
    gh.write_text(
        f"""#!{sys.executable}
import json, sys
from pathlib import Path

sf = Path({str(state_file)!r})
st = json.loads(sf.read_text())
args = sys.argv[1:]
def save(data):
    tmp = sf.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(sf)
if args[:2] == ["pr", "view"]:
    if st.get("gh_error"):
        sys.stderr.write(f"{{st['gh_error']}}\\n")
        sys.exit(1)
    if len(args) > 2 and not args[2].startswith("--") and int(args[2]) != st.get("number"):
        sys.stderr.write("pull request not found\\n")
        sys.exit(1)
    if st.get("no_pr"):
        branch = st.get("headRefName", "feat/task")
        sys.stderr.write(f'no pull requests found for branch "{{branch}}"\\n')
        sys.exit(1)
    st["views"] = st.get("views", 0) + 1
    v = st["views"]
    if st.get("pending_views", 0) >= v:
        checks = st.get(
            "pending_checks",
            [{{"name": "ci", "status": "IN_PROGRESS", "conclusion": ""}}],
        )
        mstatus = st.get("pending_merge_state", "UNKNOWN")
    else:
        checks = st.get("checks", [])
        mstatus = st.get("mergeStateStatus", "CLEAN")
    flips = st.get("flip_dirty_on_views", [])
    if st.get("flip_dirty_on_view") == v or v in flips:
        mstatus = "DIRTY"
        st["mergeStateStatus"] = "DIRTY"
    save(st)
    print(json.dumps({{
        "number": st["number"],
        "url": st["url"],
        "state": st["state"],
        "isDraft": st.get("isDraft", False),
        "mergeStateStatus": mstatus,
        "headRefOid": st["headRefOid"],
        "headRefName": st["headRefName"],
        "baseRefName": st["baseRefName"],
        "statusCheckRollup": checks,
        "mergeCommit": {{"oid": st.get("mergeCommit", "deadbeef")}},
    }}))
elif args[:2] == ["pr", "merge"]:
    methods = [m for m in ("--squash", "--merge", "--rebase") if m in args]
    if len(methods) != 1:
        sys.stderr.write("merge method flag required\\n")
        sys.exit(2)
    idx = args.index("--match-head-commit") + 1
    if args[idx] != st["headRefOid"]:
        sys.stderr.write("head sha mismatch\\n")
        sys.exit(1)
    st["state"] = "MERGED"
    st["merge_method"] = methods[0]
    st["merges"] = st.get("merges", 0) + 1
    save(st)
    print("merged")
else:
    sys.stderr.write(f"unsupported gh args: {{args}}\\n")
    sys.exit(2)
""",
        encoding="utf-8",
    )
    gh.chmod(gh.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    return state_file


def _base_input(
    repo: Path, *, config: dict[str, Any] | None = None, **extra: Any
) -> dict[str, Any]:
    cfg: dict[str, Any] = {"readiness_poll_seconds": 0.01, "readiness_timeout_seconds": 1.0}
    if config:
        cfg.update(config)
    return {"task": "Add feature", "checkout": repo, "config": cfg, **extra}


def _sync_gh_head(gh_file: Path, sha: str, *, merge_state: str | None = None) -> None:
    st = json.loads(gh_file.read_text(encoding="utf-8"))
    st["headRefOid"] = sha
    st.pop("no_pr", None)
    if merge_state is not None:
        st["mergeStateStatus"] = merge_state
    tmp = gh_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(st), encoding="utf-8")
    tmp.replace(gh_file)


def test_setup_filters_session_mcp_and_idempotent_already_merged_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path)
    al = _fake_al(
        lambda _c: None,
        available_agents={
            "legacy": AvailableAgent(agent="legacy", session_mcp=False),
            "codex": AvailableAgent(agent="codex", default_model="m1", session_mcp=True),
        },
    )
    with Engine(state_dir=tmp_path / "st", agent_layer=al, workflows=_WORKFLOWS) as eng:
        jid = eng.start("code-and-ship", _base_input(repo, orchestrator=AgentSpec(agent="legacy")))
        with pytest.raises(ValueError, match="session_mcp"):
            eng.run(jid)

    unignored = _init_repo(tmp_path, "unignored", gitignore_path=None)
    al_unignored = _fake_al(
        lambda _c: (_ for _ in ()).throw(AssertionError("agent must not run on unignored checkout"))
    )
    with Engine(
        state_dir=tmp_path / "st-unignored",
        agent_layer=al_unignored,
        workflows=_WORKFLOWS,
    ) as eng:
        jid_unignored = eng.start("code-and-ship", _base_input(unignored))
        with pytest.raises(ValueError, match=r"not gitignored.*\.agent-layer/cache/"):
            eng.run(jid_unignored)
    assert not (unignored / ".agent-layer").exists()
    assert _git(unignored, "status", "--porcelain") == ""
    assert al_unignored.calls == []

    sha = _git(repo, "rev-parse", "HEAD")
    gh_file = _install_fake_gh(
        tmp_path / "gh",
        monkeypatch,
        {
            "number": 5,
            "url": "https://example.test/pr/5",
            "state": "MERGED",
            "headRefOid": sha,
            "headRefName": "feat/task",
            "baseRefName": "main",
            "mergeCommit": "abc123",
        },
    )
    res = merge_pr_exact_head(repo, 5, sha)
    assert res.status == "merged" and res.merge_commit_sha == "abc123"
    assert json.loads(gh_file.read_text()).get("merges", 0) == 0


def test_plan_artifacts_validations_pending_ci_session_continuity_and_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path)
    gh_file = _install_fake_gh(
        tmp_path / "gh",
        monkeypatch,
        {
            "number": 42,
            "url": "https://example.test/pr/42",
            "state": "OPEN",
            "pending_views": 4,
            "headRefOid": "",
            "headRefName": "feat/task",
            "baseRefName": "main",
            "checks": [{"name": "ci", "status": "COMPLETED", "conclusion": "SUCCESS"}],
        },
    )
    checks_done = {
        "dirty": False,
        "rewind": False,
        "ship_mismatch": False,
        "gate_mismatch": False,
        "tampered_restored": False,
    }
    baseline_sha = _git(repo, "rev-parse", "HEAD")
    sibling_file = repo / ".agent-layer/cache/steward-engine/other-job/plan-v1.md"
    sibling_file.parent.mkdir(parents=True, exist_ok=True)
    sibling_file.write_text("keep\n", encoding="utf-8")
    plan_writes = 0
    plan_reviews_seen = 0
    review_lock = threading.Lock()
    job_ctx: dict[str, Any] = {}

    def _assert_mirror(rel_ref: str, name: str) -> Path:
        assert rel_ref == f".agent-layer/cache/steward-engine/{job_ctx['jid']}/{name}"
        mirror = repo / rel_ref
        canonical = job_ctx["artifacts_dir"] / name
        assert mirror.is_file() and canonical.is_file()
        assert stat.S_IMODE(mirror.stat().st_mode) == 0o444
        assert mirror.read_text(encoding="utf-8") == canonical.read_text(encoding="utf-8")
        assert _git(repo, "status", "--porcelain") == ""
        return mirror

    def handler(ctx: FakeTurnContext) -> None:
        nonlocal plan_writes, plan_reviews_seen
        tool = ctx.active_turn.expects_tool
        check = ctx.active_turn.expects_check
        if tool == "submit_assessment":
            assert ctx.role == "orchestrator"
            assert ctx.agent == AgentSpec(agent="antigravity", model="gemini-4-neon-high")
            assert ctx.call_tool(
                "submit_assessment",
                {"difficulty": "extreme", "needs_plan": True, "rationale": "bad"},
            ).is_error
            assert not ctx.call_tool(
                "submit_assessment",
                {
                    "difficulty": "medium",
                    "needs_plan": True,
                    "rationale": "multi-module change benefits from a written plan",
                },
            ).is_error
        elif tool == "submit_plan":
            assert ctx.agent == AgentSpec(agent="claude", model="opus", reasoning_effort="high")
            plan_writes += 1
            if plan_writes == 2:
                plan_ref = ctx.prompt.split("Plan file:\n", 1)[1].splitlines()[0]
                _assert_mirror(plan_ref, "plan-v1.md")
            plan_body = (
                "# Plan v1\n- step 1\n"
                if plan_writes == 1
                else "# Plan v2\n- step 1\n- tightened\n"
            )
            assert not ctx.call_tool("submit_plan", {"plan": plan_body}).is_error
        elif tool == "submit_review":
            plan_ref = ctx.prompt.split("Plan file:\n", 1)[1].splitlines()[0]
            if "git diff" in ctx.prompt:
                _assert_mirror(plan_ref, "plan-v2.md")
                checks_done["tampered_restored"] = True
            else:
                with review_lock:
                    mirror = _assert_mirror(plan_ref, "plan-v1.md")
                    plan_reviews_seen += 1
                    if plan_reviews_seen == 2:
                        mirror.unlink()
            # Emit a concrete finding so revise_plan and address_review both execute
            assert not ctx.call_tool(
                "submit_review", {"approved": True, "summary": "nit", "findings": ["tighten check"]}
            ).is_error
        elif check == "commit":
            if not checks_done["dirty"]:
                checks_done["dirty"] = True
                plan_ref = ctx.prompt.split("Plan file:\n", 1)[1].splitlines()[0]
                mirror_plan = _assert_mirror(plan_ref, "plan-v2.md")
                os.chmod(mirror_plan, 0o644)
                mirror_plan.write_text("tampered\n", encoding="utf-8")
                assert _git(repo, "status", "--porcelain") == ""
                (repo / "feature.py").write_text("x = 1\n", encoding="utf-8")
                bad = ctx.call_tool("check_done", {})
                assert bad.is_error and "uncommitted changes" in bad.text.lower()
                _commit(repo, "feature.py", "x = 1\n", "impl")
            else:
                sha = _git(repo, "rev-parse", "HEAD")
                if not checks_done["rewind"]:
                    checks_done["rewind"] = True
                    _git(repo, "reset", "--hard", baseline_sha)
                    rewound = ctx.call_tool("check_done", {})
                    assert rewound.is_error and "not a descendant" in rewound.text.lower()
                    _git(repo, "reset", "--hard", sha)
            assert not ctx.call_tool("check_done", {}).is_error
        elif check == "pr":
            sha = _git(repo, "rev-parse", "HEAD")
            if not checks_done["ship_mismatch"]:
                checks_done["ship_mismatch"] = True
                bad_ship = ctx.call_tool("check_done", {})
                assert bad_ship.is_error and "does not match expected commit" in bad_ship.text
            _sync_gh_head(gh_file, sha)
            assert not ctx.call_tool("check_done", {}).is_error
        elif tool == "submit_gate_decision":
            ctx_ref = ctx.prompt.split("Context file:\n", 1)[1].splitlines()[0]
            _assert_mirror(ctx_ref, "merge-context-a1.md")
            sha = _git(repo, "rev-parse", "HEAD")
            if not checks_done["gate_mismatch"]:
                checks_done["gate_mismatch"] = True
                assert ctx.call_tool(
                    "submit_gate_decision",
                    {"decision": "merge", "head_sha": "0" * 40, "reason": "wrong sha"},
                ).is_error
            assert not ctx.call_tool(
                "submit_gate_decision", {"decision": "merge", "head_sha": sha, "reason": "ready"}
            ).is_error

    al = _fake_al(
        handler,
        quota=QuotaSnapshot(
            by_agent={
                "codex": QuotaReading(pace_ratio=0.3),
                "antigravity": QuotaReading(pace_ratio=0.7),
                "grok": QuotaReading(pace_ratio=0.2),
                "muse": QuotaReading(pace_ratio=0.6),
            }
        ),
    )
    with Engine(state_dir=tmp_path / "st", agent_layer=al, workflows=_WORKFLOWS) as eng:
        jid = eng.start("code-and-ship", _base_input(repo, config={"allow_unattended_merge": True}))
        job_ctx["jid"] = jid
        job_ctx["artifacts_dir"] = eng.store.artifacts_dir(jid)
        st = eng.run(jid)
        assert st.phase == "completed" and st.state is not None
        assert all(checks_done.values())
        data = cast(dict[str, Any], st.state)
        assert data["coordination"]["difficulty"] == "medium"
        assert data["assignment"]["coder_profile"] == "opus_high"
        assert [r["profile"] for r in data["assignment"]["reviewers"]] == ["sol_high", "grok"]
        assert [p["ref"] for p in data["plans"]] == [
            f".agent-layer/cache/steward-engine/{jid}/plan-v1.md",
            f".agent-layer/cache/steward-engine/{jid}/plan-v2.md",
        ]
        assert (eng.store.artifacts_dir(jid) / "plan-v1.md").read_text(
            encoding="utf-8"
        ) == "# Plan v1\n- step 1"
        assert (eng.store.artifacts_dir(jid) / "plan-v2.md").read_text(
            encoding="utf-8"
        ) == "# Plan v2\n- step 1\n- tightened"
        attempt = data["pr"]["attempts"][-1]
        assert attempt["context_ref"] == (
            f".agent-layer/cache/steward-engine/{jid}/merge-context-a1.md"
        )
        assert (eng.store.artifacts_dir(jid) / "merge-context-a1.md").is_file()
        assert not (repo / ".agent-layer/cache/steward-engine" / str(jid)).exists()
        assert sibling_file.read_text(encoding="utf-8") == "keep\n"
        assert attempt["merge"]["status"] == "merged"
        assert attempt["merge"]["cleaned_branch"] == "feat/task"
        assert json.loads(gh_file.read_text())["merge_method"] == "--squash"
        assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"
        assert "feat/task" not in _git(repo, "branch")
        rev0 = data["revisions"][0]
        assert len(data["revisions"]) == 1 and rev0["base_sha"] == baseline_sha
        assert all(r["head_sha"] is None for r in data["reviews"] if r["stage"] == "plan")
        assert all(
            r["head_sha"] == rev0["head_sha"] for r in data["reviews"] if r["stage"] == "code"
        )

        coder_calls = [c for c in al.calls if c.role == "coder"]
        assert len({c.handle for c in coder_calls}) == 1
        assert [c.is_continue for c in coder_calls] == [False, True, True, True, True]
        rev_calls = [c for c in al.calls if c.role == "reviewer"]
        assert len({c.handle for c in rev_calls}) == 2
        for h in {c.handle for c in rev_calls}:
            handle_calls = [c for c in rev_calls if c.handle == h]
            assert [c.is_continue for c in handle_calls] == [False, True]
            assert f"Plan file:\n{data['plans'][0]['ref']}" in handle_calls[0].prompt
            assert f"Plan file:\n{data['plans'][-1]['ref']}" in handle_calls[1].prompt
            assert "plan-v2.md" in handle_calls[1].prompt
        orch_calls = [c for c in al.calls if c.role == "orchestrator"]
        assert [c.expects_tool for c in orch_calls] == [
            "submit_assessment",
            "submit_gate_decision",
        ]
        assert all(not c.is_continue for c in orch_calls)
        assert len({c.handle for c in orch_calls}) == 2


def test_clean_plan_and_code_reviews_skip_revise_and_address(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path, gitignore_path=".agent-layer/.gitignore", gitignore="/cache/\n")
    baseline = _git(repo, "rev-parse", "HEAD")
    gh_file = _install_fake_gh(
        tmp_path / "gh",
        monkeypatch,
        {
            "number": 3,
            "url": "https://example.test/pr/3",
            "state": "OPEN",
            "headRefOid": "",
            "headRefName": "feat/task",
            "baseRefName": "main",
        },
    )
    deleted_restored = False

    def handler(ctx: FakeTurnContext) -> None:
        nonlocal deleted_restored
        tool = ctx.active_turn.expects_tool
        check = ctx.active_turn.expects_check
        if tool == "submit_assessment":
            ctx.call_tool(
                "submit_assessment",
                {"difficulty": "medium", "needs_plan": True, "rationale": "planned work"},
            )
        elif tool == "submit_plan":
            ctx.call_tool("submit_plan", {"plan": "# Clean Plan\n"})
        elif tool == "submit_review":
            plan_ref = ctx.prompt.split("Plan file:\n", 1)[1].splitlines()[0]
            if "git diff" in ctx.prompt:
                mirror = repo / plan_ref
                assert mirror.read_text(encoding="utf-8") == "# Clean Plan"
                assert stat.S_IMODE(mirror.stat().st_mode) == 0o444
                deleted_restored = True
            else:
                (repo / plan_ref).unlink(missing_ok=True)
            ctx.call_tool("submit_review", {"approved": True, "summary": "lgtm", "findings": []})
        elif check == "commit":
            plan_ref = ctx.prompt.split("Plan file:\n", 1)[1].splitlines()[0]
            mirror = repo / plan_ref
            assert mirror.read_text(encoding="utf-8") == "# Clean Plan"
            assert stat.S_IMODE(mirror.stat().st_mode) == 0o444
            mirror.unlink()
            _commit(repo, "clean.py", "ok = True\n", "impl")
            assert not ctx.call_tool("check_done", {}).is_error
        elif check == "pr":
            sha = _git(repo, "rev-parse", "HEAD")
            _sync_gh_head(gh_file, sha)
            assert not ctx.call_tool("check_done", {}).is_error
        elif tool == "submit_gate_decision":
            sha = _git(repo, "rev-parse", "HEAD")
            ctx.call_tool(
                "submit_gate_decision", {"decision": "merge", "head_sha": sha, "reason": "clean"}
            )

    override_orch = AgentSpec(agent="claude", model="opus", reasoning_effort="high")
    al = _fake_al(handler)
    with Engine(state_dir=tmp_path / "st", agent_layer=al, workflows=_WORKFLOWS) as eng:
        jid = eng.start(
            "code-and-ship",
            _base_input(repo, orchestrator=override_orch, config={"allow_unattended_merge": True}),
        )
        st = eng.run(jid)
        assert st.phase == "completed" and st.state is not None
        assert deleted_restored is True
        data = cast(dict[str, Any], st.state)
        assert len(data["plans"]) == 1
        assert (eng.store.artifacts_dir(jid) / "plan-v1.md").is_file()
        assert not (eng.store.artifacts_dir(jid) / "plan-v2.md").exists()
        assert not (repo / ".agent-layer/cache/steward-engine" / str(jid)).exists()
        assert (repo / ".agent-layer/.gitignore").is_file()
        assert len(data["revisions"]) == 1 and data["revisions"][0]["base_sha"] == baseline
        assert [(c.expects_tool, c.expects_check) for c in al.calls if c.role == "coder"] == [
            ("submit_plan", None),
            (None, "commit"),
            (None, "pr"),
        ]
        # Explicit orchestrator override wins over default antigravity even when available
        orch_calls = [c for c in al.calls if c.role == "orchestrator"]
        assert [c.expects_tool for c in orch_calls] == [
            "submit_assessment",
            "submit_gate_decision",
        ]
        assert all(c.agent == override_orch for c in orch_calls)


def test_approval_request_human_approval_and_readiness_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _run(
        name: str,
        *,
        unattended: bool,
        decision: str,
        approve: bool,
        flip_dirty: bool,
        second_decision: str | None = None,
    ) -> Any:
        repo = _init_repo(tmp_path, name)
        gh_file = _install_fake_gh(
            tmp_path / f"gh-{name}",
            monkeypatch,
            {
                "number": 7,
                "url": "https://example.test/pr/7",
                "state": "OPEN",
                "headRefOid": "",
                "headRefName": "feat/task",
                "baseRefName": "main",
                # ship_pr uses 2 views (check_done + read_outcome), check_readiness 3rd, merge 4th
                "flip_dirty_on_view": 4 if flip_dirty else 0,
            },
        )
        ships = 0

        def handler(ctx: FakeTurnContext) -> None:
            nonlocal ships
            tool = ctx.active_turn.expects_tool
            check = ctx.active_turn.expects_check
            if tool == "submit_assessment":
                ctx.call_tool(
                    "submit_assessment",
                    {"difficulty": "low", "needs_plan": False, "rationale": "small change"},
                )
            elif check == "commit":
                _commit(repo, "a.txt", "v1\n", "impl")
                assert not ctx.call_tool("check_done", {}).is_error
            elif tool == "submit_review":
                ctx.call_tool("submit_review", {"approved": True, "summary": "ok"})
            elif check == "pr":
                ships += 1
                sha = _git(repo, "rev-parse", "HEAD")
                if ships == 2:
                    assert "existing PR #7" in ctx.prompt
                    assert "Previous merge aborted because PR readiness changed" in ctx.prompt
                _sync_gh_head(gh_file, sha, merge_state="CLEAN")
                assert not ctx.call_tool("check_done", {}).is_error
            elif tool == "submit_gate_decision":
                sha = _git(repo, "rev-parse", "HEAD")
                use_dec = second_decision if (second_decision and ships >= 2) else decision
                reason = "architecture mismatch" if use_dec == "escalate" else "gate"
                ctx.call_tool(
                    "submit_gate_decision",
                    {"decision": use_dec, "head_sha": sha, "reason": reason},
                )

        al = _fake_al(handler)
        with Engine(state_dir=tmp_path / f"st-{name}", agent_layer=al, workflows=_WORKFLOWS) as eng:
            jid = eng.start(
                "code-and-ship", _base_input(repo, config={"allow_unattended_merge": unattended})
            )
            st = eng.run(jid)
            if second_decision == "escalate":
                gate_calls = [c for c in al.calls if c.expects_tool == "submit_gate_decision"]
                assert len(gate_calls) == 2 and all(not c.is_continue for c in gate_calls)
                assert len({c.handle for c in gate_calls}) == 2
                assert st.phase == "paused" and st.pending_waits[0].kind == "escalation"
                eng.reply(jid, st.pending_waits[0].wait_id, {"action": "abort"})
                aborted = eng.run(jid)
                assert aborted.phase == "completed"
                return st, json.loads(gh_file.read_text())
            assert st.phase == "paused" and st.pending_waits[0].kind == "merge_approval"
            eng.reply(jid, st.pending_waits[0].wait_id, {"approved": approve})
            st = eng.run(jid)
            if flip_dirty:
                assert st.phase == "paused" and ships == 2
                gate_calls = [c for c in al.calls if c.expects_tool == "submit_gate_decision"]
                assert len(gate_calls) == 2 and all(not c.is_continue for c in gate_calls)
                assert len({c.handle for c in gate_calls}) == 2
                eng.reply(jid, st.pending_waits[0].wait_id, {"approved": True})
                st = eng.run(jid)
            return st, json.loads(gh_file.read_text())

    st_no, gh_no = _run("no", unattended=False, decision="merge", approve=False, flip_dirty=False)
    assert st_no.phase == "completed" and gh_no["state"] == "OPEN"

    st_yes, gh_yes = _run(
        "yes", unattended=True, decision="request_human_approval", approve=True, flip_dirty=True
    )
    assert st_yes.phase == "completed" and gh_yes["state"] == "MERGED"
    yes_attempts = cast(dict[str, Any], st_yes.state)["pr"]["attempts"]
    assert len(yes_attempts) == 2
    assert yes_attempts[0]["merge"]["status"] == "readiness_changed"
    assert yes_attempts[1]["merge"]["status"] == "merged"

    st_esc, gh_esc = _run(
        "restart",
        unattended=True,
        decision="merge",
        approve=True,
        flip_dirty=True,
        second_decision="escalate",
    )
    assert gh_esc["state"] == "OPEN"
    assert st_esc.phase == "paused" and st_esc.pending_waits[0].kind == "escalation"
    assert st_esc.state is not None
    esc_data = cast(dict[str, Any], st_esc.state)
    assert esc_data["retries"]["readiness_checks"] == 1
    assert len(esc_data["pr"]["attempts"]) == 2
    assert "architecture mismatch" in st_esc.pending_waits[0].prompt
    assert "PR readiness changed" not in st_esc.pending_waits[0].prompt
    assert st_esc.pending_waits[0].payload["reason"] == "architecture mismatch"


def test_repair_triage_escalation_retry_and_reviewer_continuity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path)
    initial_baseline = _git(repo, "rev-parse", "HEAD")
    gh_file = _install_fake_gh(
        tmp_path / "gh",
        monkeypatch,
        {
            "number": 9,
            "url": "https://example.test/pr/9",
            "state": "OPEN",
            "headRefOid": "",
            "headRefName": "feat/task",
            "baseRefName": "main",
            "checks": [{"name": "ci", "status": "COMPLETED", "conclusion": "FAILURE"}],
        },
    )
    commits: list[str] = []
    gates = 0
    triages = 0

    def handler(ctx: FakeTurnContext) -> None:
        nonlocal gates, triages
        tool = ctx.active_turn.expects_tool
        check = ctx.active_turn.expects_check
        if tool == "submit_assessment":
            ctx.call_tool(
                "submit_assessment",
                {"difficulty": "low", "needs_plan": False, "rationale": "small change"},
            )
        elif check == "commit":
            n = len(commits) + 1
            if n == 2:
                assert "PR not ready: check 'ci' concluded FAILURE" in ctx.prompt
                assert "stale-gate-1" not in ctx.prompt
            elif n == 3:
                assert "gate-2" in ctx.prompt and "add boundary guard" in ctx.prompt
                assert "PR not ready" not in ctx.prompt
            elif n == 5:
                assert "gate-3" in ctx.prompt
                assert "add boundary guard" not in ctx.prompt and "gate-2" not in ctx.prompt
            sha = _commit(repo, "f.txt", f"v{n}\n", f"commit {n}")
            commits.append(sha)
            assert not ctx.call_tool("check_done", {}).is_error
        elif tool == "submit_review":
            findings = ["tighten"] if triages == 2 else []
            ctx.call_tool(
                "submit_review", {"approved": True, "summary": "ok", "findings": findings}
            )
        elif tool == "submit_repair_triage":
            triages += 1
            sha = _git(repo, "rev-parse", "HEAD")
            assert f"{commits[-2]}..{commits[-1]}" in ctx.prompt
            assert ctx.call_tool(
                "submit_repair_triage",
                {"needs_code_review": False, "head_sha": "0" * 40, "reason": "wrong"},
            ).is_error
            if triages == 1:
                # Receives exact non-ready PR repair_reason, not stale gate verdict.reason
                assert "PR not ready: check 'ci' concluded FAILURE" in ctx.prompt
                assert "stale-gate-1" not in ctx.prompt
                assert not ctx.call_tool(
                    "submit_repair_triage",
                    {"needs_code_review": False, "head_sha": sha, "reason": "tiny fix"},
                ).is_error
            elif triages == 2:
                # After escalation retry, receives repair_reason + human_instructions
                assert "gate-2" in ctx.prompt and "add boundary guard" in ctx.prompt
                assert "PR not ready" not in ctx.prompt
                assert not ctx.call_tool(
                    "submit_repair_triage",
                    {"needs_code_review": True, "head_sha": sha, "reason": "substantive"},
                ).is_error
            else:
                # Neither prior repair_reason nor human_instructions leaks into later triage
                assert "gate-3" in ctx.prompt
                assert "add boundary guard" not in ctx.prompt and "gate-2" not in ctx.prompt
                assert "PR not ready" not in ctx.prompt
                assert not ctx.call_tool(
                    "submit_repair_triage",
                    {"needs_code_review": False, "head_sha": sha, "reason": "followup"},
                ).is_error
        elif check == "pr":
            sha = _git(repo, "rev-parse", "HEAD")
            st = json.loads(gh_file.read_text(encoding="utf-8"))
            st["headRefOid"] = sha
            if triages >= 1:
                st["checks"] = [{"name": "ci", "status": "COMPLETED", "conclusion": "SUCCESS"}]
            gh_file.write_text(json.dumps(st), encoding="utf-8")
            assert not ctx.call_tool("check_done", {}).is_error
        elif tool == "submit_gate_decision":
            gates += 1
            sha = _git(repo, "rev-parse", "HEAD")
            ctx_line = ctx.prompt.split("Context file:\n", 1)[1].splitlines()[0]
            assert not Path(ctx_line).is_absolute()
            assert Path(ctx_line).name == f"merge-context-a{gates}.md"
            ctx_doc = (repo / ctx_line).read_text(encoding="utf-8")
            assert "Prior Merge Gate Verdicts" not in ctx.prompt
            if gates == 1:
                dec, reason = "merge", "stale-gate-1"
            elif gates == 2:
                assert "repair triage skipped re-review" in ctx_doc and "tiny fix" in ctx_doc
                assert "repair triage skipped re-review" not in ctx.prompt
                dec, reason = "escalate", "gate-2"
            elif gates == 3:
                assert "before post-review updates advanced PR head" in ctx_doc
                assert "add boundary guard" in ctx_doc
                assert "Prior Merge Gate Verdicts" in ctx_doc
                dec, reason = "request_changes", "gate-3"
            else:
                dec, reason = "merge", "gate-4"
            ctx.call_tool(
                "submit_gate_decision", {"decision": dec, "head_sha": sha, "reason": reason}
            )

    al = _fake_al(handler)
    with Engine(state_dir=tmp_path / "st", agent_layer=al, workflows=_WORKFLOWS) as eng:
        jid = eng.start(
            "code-and-ship",
            _base_input(repo, config={"allow_unattended_merge": True, "max_auto_repairs": 2}),
        )
        st = eng.run(jid)
        assert st.phase == "paused" and st.pending_waits[0].kind == "escalation"
        assert st.state is not None and len(cast(dict[str, Any], st.state)["revisions"]) == 2
        assert triages == 1

        eng.reply(
            jid,
            st.pending_waits[0].wait_id,
            {"action": "retry", "instructions": "add boundary guard"},
        )
        st = eng.run(jid)
        assert st.phase == "completed" and st.state is not None
        data = cast(dict[str, Any], st.state)
        revs = data["revisions"]
        assert len(revs) == 4 and revs[0]["base_sha"] == initial_baseline
        # address_review on rev 2 updates head_sha while preserving base_sha and triage
        assert revs[2]["base_sha"] == commits[1] and revs[2]["head_sha"] == commits[3]
        assert revs[2]["triage"] == {
            "head_sha": commits[2],
            "needs_code_review": True,
            "reason": "substantive",
        }
        assert [r["triage"]["needs_code_review"] if r["triage"] else None for r in revs] == [
            None,
            False,
            True,
            False,
        ]
        assert data["pr"]["attempts"][-1]["merge"]["status"] == "merged"
        assert len(data["pr"]["attempts"]) == 4
        assert len(data["escalations"]) == 1 and data["escalations"][0]["action"] == "retry"
        assert sorted({r["revision"] for r in data["reviews"]}) == [0, 2]
        assert len(data["reviews"]) == 4

        # Both cross-family reviewer sessions continued across rev 0 -> rev 2 (rev 1 & 3 skipped)
        rev_calls = [c for c in al.calls if c.role == "reviewer"]
        assert len({c.handle for c in rev_calls}) == 2
        for h in {c.handle for c in rev_calls}:
            handle_calls = [c for c in rev_calls if c.handle == h]
            assert [c.is_continue for c in handle_calls] == [False, True]
            assert all("Plan file:" not in c.prompt for c in handle_calls)
        # Orchestrator runs fresh sessions (never continued) with distinct handles across all steps
        orch_calls = [c for c in al.calls if c.role == "orchestrator"]
        assert [c.expects_tool for c in orch_calls] == [
            "submit_assessment",
            "submit_gate_decision",
            "submit_repair_triage",
            "submit_gate_decision",
            "submit_repair_triage",
            "submit_gate_decision",
            "submit_repair_triage",
            "submit_gate_decision",
        ]
        assert all(not c.is_continue for c in orch_calls)
        assert len({c.handle for c in orch_calls}) == len(orch_calls)


def test_settings_env_checkpoint_immunity_max_repairs_zero_and_bounded_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path, gitignore="/.custom-cache/\n")
    monkeypatch.setenv("STEW_ENGINE_CODING_ALLOW_UNATTENDED_MERGE", "true")
    monkeypatch.setenv("STEW_ENGINE_CODING_MAX_AUTO_REPAIRS", "0")
    monkeypatch.setenv("STEW_ENGINE_CODING_MAX_READINESS_RETRIES", "1")
    monkeypatch.setenv("STEW_ENGINE_CODING_READINESS_POLL_SECONDS", "0.01")
    monkeypatch.setenv("STEW_ENGINE_CODING_READINESS_TIMEOUT_SECONDS", "1.0")
    monkeypatch.setenv("STEW_ENGINE_CODING_ARTIFACT_MIRROR_DIR", ".custom-cache/steward")
    cfg = CodingSettings().to_config()
    assert type(cfg) is CodingConfig and cfg.max_auto_repairs == 0
    assert cfg.artifact_mirror_dir == Path(".custom-cache/steward")
    inp_from_settings = CodeInput(task="Env test", checkout=repo, config=CodingSettings())
    assert type(inp_from_settings.config) is CodingConfig
    assert inp_from_settings.config.max_auto_repairs == 0

    gh_file = _install_fake_gh(
        tmp_path / "gh",
        monkeypatch,
        {
            "number": 15,
            "url": "https://example.test/pr/15",
            "state": "OPEN",
            "headRefOid": "",
            "headRefName": "feat/task",
            "baseRefName": "main",
            # Flip DIRTY on both merge_exact_head views (7 & 11; each ship_pr uses 2 views)
            "flip_dirty_on_views": [7, 11],
        },
    )
    gates = 0
    commits = 0

    def handler(ctx: FakeTurnContext) -> None:
        nonlocal gates, commits
        tool = ctx.active_turn.expects_tool
        check = ctx.active_turn.expects_check
        if tool == "submit_assessment":
            ctx.call_tool(
                "submit_assessment",
                {"difficulty": "low", "needs_plan": False, "rationale": "small change"},
            )
        elif check == "commit":
            commits += 1
            _commit(repo, "e.txt", f"v{commits}\n", f"c{commits}")
            assert not ctx.call_tool("check_done", {}).is_error
        elif tool == "submit_review":
            ctx.call_tool("submit_review", {"approved": True, "summary": "ok"})
        elif tool == "submit_repair_triage":
            sha = _git(repo, "rev-parse", "HEAD")
            ctx.call_tool(
                "submit_repair_triage",
                {"needs_code_review": False, "head_sha": sha, "reason": "small"},
            )
        elif check == "pr":
            sha = _git(repo, "rev-parse", "HEAD")
            _sync_gh_head(gh_file, sha, merge_state="CLEAN")
            assert not ctx.call_tool("check_done", {}).is_error
        elif tool == "submit_gate_decision":
            gates += 1
            sha = _git(repo, "rev-parse", "HEAD")
            dec = "request_changes" if gates == 1 else "merge"
            ctx.call_tool(
                "submit_gate_decision", {"decision": dec, "head_sha": sha, "reason": f"g{gates}"}
            )

    al = _fake_al(handler)
    with Engine(state_dir=tmp_path / "st", agent_layer=al, workflows=_WORKFLOWS) as eng:
        # Omit explicit `config` so resolve_setup loads CodingSettings().to_config()
        jid = eng.start("code-and-ship", {"task": "Env test", "checkout": repo})
        st = eng.run(jid)
        # max_auto_repairs=0 escalates immediately on first request_changes
        assert st.phase == "paused" and st.pending_waits[0].kind == "escalation"
        assert st.state is not None
        assert cast(dict[str, Any], st.state)["retries"]["auto_repairs"] == 0

        # Mutating env vars mid-run must NOT alter checkpointed state.setup.config
        monkeypatch.setenv("STEW_ENGINE_CODING_ALLOW_UNATTENDED_MERGE", "false")
        monkeypatch.setenv("STEW_ENGINE_CODING_MAX_AUTO_REPAIRS", "99")
        monkeypatch.setenv("STEW_ENGINE_CODING_MAX_READINESS_RETRIES", "99")
        monkeypatch.setenv("STEW_ENGINE_CODING_ARTIFACT_MIRROR_DIR", ".other-cache/steward")
        eng.reply(jid, st.pending_waits[0].wait_id, {"action": "retry", "instructions": "fix"})
        st = eng.run(jid)
        # After 1 readiness_changed retry (max_readiness_retries=1), the 2nd escalates
        assert st.phase == "paused" and st.pending_waits[0].kind == "escalation"
        assert "PR readiness changed" in st.pending_waits[0].prompt
        assert st.state is not None
        env_data = cast(dict[str, Any], st.state)
        persisted_cfg = env_data["setup"]["config"]
        assert persisted_cfg["allow_unattended_merge"] is True
        assert persisted_cfg["max_auto_repairs"] == 0
        assert persisted_cfg["max_readiness_retries"] == 1
        assert persisted_cfg["artifact_mirror_dir"] == ".custom-cache/steward"
        assert env_data["retries"]["readiness_checks"] == 2
        attempts = env_data["pr"]["attempts"]
        assert all(a["context_ref"].startswith(f".custom-cache/steward/{jid}/") for a in attempts)
        latest_mirror = repo / attempts[-1]["context_ref"]
        assert latest_mirror.is_file() and stat.S_IMODE(latest_mirror.stat().st_mode) == 0o444
        assert not (repo / ".other-cache").exists()

        eng.reply(jid, st.pending_waits[0].wait_id, {"action": "retry", "instructions": "ship"})
        done = eng.run(jid)
        assert done.phase == "completed" and json.loads(gh_file.read_text())["state"] == "MERGED"
        assert not (repo / ".custom-cache/steward" / str(jid)).exists()
        assert not (repo / ".other-cache").exists()
        assert (eng.store.artifacts_dir(jid) / "merge-context-a1.md").is_file()


def test_readiness_blocked_pending_transition_and_git_failure_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")

    gh_file = _install_fake_gh(
        tmp_path / "gh-r1",
        monkeypatch,
        {
            "number": 11,
            "url": "https://example.test/pr/11",
            "state": "OPEN",
            "pending_views": 1,
            "pending_merge_state": "BLOCKED",
            "mergeStateStatus": "CLEAN",
            "headRefOid": sha,
            "headRefName": "feat/task",
            "baseRefName": "main",
            "checks": [{"name": "ci", "status": "COMPLETED", "conclusion": "SUCCESS"}],
        },
    )
    ready = wait_for_readiness(repo, head=sha, pr_number=11, poll_seconds=0.01, timeout_seconds=1.0)
    assert ready.ready is True and ready.pending is False and ready.reasons == ()
    assert json.loads(gh_file.read_text())["views"] == 2

    _install_fake_gh(
        tmp_path / "gh-r1-fail",
        monkeypatch,
        {
            "number": 12,
            "url": "https://example.test/pr/12",
            "state": "OPEN",
            "pending_views": 1,
            "pending_merge_state": "UNSTABLE",
            "pending_checks": [{"name": "ci", "status": "QUEUED", "conclusion": ""}],
            "mergeStateStatus": "BLOCKED",
            "headRefOid": sha,
            "headRefName": "feat/task",
            "baseRefName": "main",
            "checks": [{"name": "ci", "status": "COMPLETED", "conclusion": "SUCCESS"}],
        },
    )
    unstable_pending = pr_readiness(repo, sha, pr_number=12)
    assert unstable_pending.ready is False and unstable_pending.pending is True
    blocked = pr_readiness(repo, sha, pr_number=12)
    assert blocked.ready is False and blocked.pending is False
    assert any("BLOCKED" in reason for reason in blocked.reasons)

    non_git = tmp_path / "not-a-repo"
    non_git.mkdir()
    with pytest.raises(ValueError, match="git status"):
        verify_commit_lineage(non_git, baseline_sha=sha, expected_head=sha)
    with pytest.raises(GitHubError, match="git check-ignore"):
        is_ignored(non_git, ".agent-layer/cache/probe.md")

    # verify_shipped_pr validates live GitHub PR state, draft flag, PR number, head SHA, and errors
    from castle_steward_coding import github as coding_github

    base_pr = {
        "number": 11,
        "state": "OPEN",
        "isDraft": False,
        "headRefOid": sha,
        "headRefName": "feat/task",
    }
    monkeypatch.setattr(coding_github, "_view_pr", lambda _c, _n=None: base_pr)
    assert verify_shipped_pr(repo, expected_head=sha, expected_pr=11) == (11, sha)
    for override, pattern in (
        ({"isDraft": True}, "draft"),
        ({"state": "CLOSED"}, "expected OPEN"),
        ({"headRefOid": "0" * 40}, "does not match expected commit"),
        ({"headRefName": "other-branch"}, "does not match checked-out branch"),
        ({"number": 99}, "does not match expected PR"),
    ):
        monkeypatch.setattr(
            coding_github, "_view_pr", lambda _c, _n=None, o=override: {**base_pr, **o}
        )
        with pytest.raises(ValueError, match=pattern):
            verify_shipped_pr(repo, expected_head=sha, expected_pr=11)
    monkeypatch.setattr(
        coding_github,
        "_view_pr",
        lambda _c, _n=None: (_ for _ in ()).throw(GitHubError("no pull requests found")),
    )
    with pytest.raises(ValueError, match="Could not read pull request via gh"):
        verify_shipped_pr(repo, expected_head=sha)


def test_repair_rejects_noop_after_address_review_and_stop_interrupts_ci_polling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path)
    assert CodingConfig().readiness_timeout_seconds == 1800.0

    gh_file = _install_fake_gh(
        tmp_path / "gh",
        monkeypatch,
        {
            "number": 21,
            "url": "https://example.test/pr/21",
            "state": "OPEN",
            "pending_views": 0,
            "pending_merge_state": "BLOCKED",
            "mergeStateStatus": "CLEAN",
            "headRefOid": "",
            "headRefName": "feat/task",
            "baseRefName": "main",
            "checks": [{"name": "ci", "status": "COMPLETED", "conclusion": "SUCCESS"}],
        },
    )
    commit_step = 0
    addressed_sha = ""
    noop_rejected = False
    gates = 0

    def handler(ctx: FakeTurnContext) -> None:
        nonlocal commit_step, addressed_sha, noop_rejected, gates
        tool = ctx.active_turn.expects_tool
        check = ctx.active_turn.expects_check
        if tool == "submit_assessment":
            ctx.call_tool(
                "submit_assessment",
                {"difficulty": "low", "needs_plan": False, "rationale": "small change"},
            )
        elif check == "commit":
            commit_step += 1
            if commit_step == 1:
                _commit(repo, "f.txt", "impl\n", "implement")
            elif commit_step == 2:
                addressed_sha = _commit(repo, "f.txt", "addressed\n", "address review")
            else:
                assert addressed_sha in ctx.prompt
                noop = ctx.call_tool("check_done", {})
                assert noop.is_error and "equals baseline" in noop.text.lower()
                noop_rejected = True
                _commit(repo, "f.txt", "repaired\n", "repair")
            assert not ctx.call_tool("check_done", {}).is_error
        elif tool == "submit_review":
            # Emit a finding so address_review executes and advances head
            ctx.call_tool(
                "submit_review", {"approved": True, "summary": "nit", "findings": ["polish"]}
            )
        elif tool == "submit_repair_triage":
            sha = _git(repo, "rev-parse", "HEAD")
            ctx.call_tool(
                "submit_repair_triage",
                {"needs_code_review": False, "head_sha": sha, "reason": "small"},
            )
        elif check == "pr":
            sha = _git(repo, "rev-parse", "HEAD")
            st = json.loads(gh_file.read_text())
            st["headRefOid"] = sha
            if gates == 1:
                st["pending_views"] = 100
                st["views"] = 0
            tmp = gh_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(st), encoding="utf-8")
            tmp.replace(gh_file)
            assert not ctx.call_tool("check_done", {}).is_error
        elif tool == "submit_gate_decision":
            gates += 1
            sha = _git(repo, "rev-parse", "HEAD")
            dec = "request_changes" if gates == 1 else "merge"
            ctx.call_tool(
                "submit_gate_decision",
                {"decision": dec, "head_sha": sha, "reason": f"gate-{gates}"},
            )

    al = _fake_al(handler)
    with Engine(state_dir=tmp_path / "st", agent_layer=al, workflows=_WORKFLOWS) as eng:
        jid = eng.start(
            "code-and-ship",
            {
                "task": "Add feature",
                "checkout": repo,
                "config": {
                    "allow_unattended_merge": True,
                    "readiness_poll_seconds": 10.0,
                    "readiness_timeout_seconds": 60.0,
                },
            },
        )
        holder: dict[str, Any] = {}
        t = threading.Thread(target=lambda: holder.setdefault("st", eng.run(jid)))
        t.start()

        deadline = time.monotonic() + 15.0
        entered_polling = False
        while time.monotonic() < deadline and t.is_alive():
            data = json.loads(gh_file.read_text())
            # ship_pr uses 2 views (check_done + read_outcome); view 3 is check_readiness polling
            if data.get("pending_views") == 100 and data.get("views", 0) >= 3:
                entered_polling = True
                break
            time.sleep(0.02)
        if not entered_polling:
            eng.stop(jid)
            t.join(timeout=6.0)
            pytest.fail("check_readiness never entered pending CI polling")

        stop_started = time.monotonic()
        eng.stop(jid)
        t.join(timeout=6.0)
        assert not t.is_alive()
        assert time.monotonic() - stop_started < 5.0
        assert holder["st"].phase == "stopped"
        assert noop_rejected is True

        st_data = json.loads(gh_file.read_text())
        st_data["pending_views"] = 0
        gh_file.write_text(json.dumps(st_data))
        final = eng.run(jid)
        assert final.phase == "completed" and final.state is not None
        final_data = cast(dict[str, Any], final.state)
        assert final_data["pr"]["attempts"][-1]["merge"]["status"] == "merged"


def test_prompts_omit_checkout_use_plan_ref_and_isolate_retry_instructions(
    tmp_path: Path,
) -> None:
    spec = AgentSpec(agent="codex")
    plan_ref = ".agent-layer/cache/steward-engine/job-1/plan-v1.md"
    ctx_ref_1 = ".agent-layer/cache/steward-engine/job-1/merge-context-a1.md"
    ctx_ref_2 = ".agent-layer/cache/steward-engine/job-1/merge-context-a2.md"
    base_item = ReviewItem(
        stage="plan",
        reviewer="opus_high",
        agent=spec,
        checkout=tmp_path,
        task="Do task",
        revision=0,
        plan_ref=plan_ref,
        baseline="aaa",
        reviewed_head="bbb",
    )

    full_sha = "0123456789abcdef0123456789abcdef01234567"
    state = CodeAndShipState(
        input=CodeInput(task="Do task", checkout=tmp_path),
        setup=RunSetup(config=CodingConfig(), orchestrator=spec),
        revisions=(CodeRevision(base_sha="aaa", head_sha="aaa"),),
        reviews=[
            ReviewRecord(
                stage="plan", revision=0, reviewer="opus_high", approved=True, summary="ok"
            ),
            ReviewRecord(
                stage="code",
                revision=0,
                reviewer="grok",
                approved=False,
                summary="fix edge",
                head_sha=full_sha,
            ),
        ],
    )

    coord = coordinate_prompt(state)
    assert all(t in coord for t in ("`low`", "`medium`", "`high`", "submit_assessment"))
    assert "Planning adds extra turns and latency" in coord
    assert "with your `difficulty` and `needs_plan` determination" in coord
    assert "includes, but is not limited to," in coord
    no_edits = (
        "Do not edit, create, or delete any files in the repository; "
        "using `submit_plan` is all that is required."
    )
    wp = write_plan_prompt(state)
    assert wp.startswith("Write an implementation plan") and no_edits in wp
    assert "where appropriate" in wp
    with_plan = state.model_copy(update={"plans": (PlanVersion(text="# P", ref=plan_ref),)})
    rp = revise_plan_prompt(with_plan)
    assert (
        "Read your current plan in the plan file below and revise it to address "
        "the review findings below." in rp
    )
    assert no_edits in rp and f"Plan file:\n{plan_ref}" in rp and "Review findings:\n" in rp
    assert "Call `submit_plan` with the full updated plan" in rp and "(head " not in rp
    plan_p = review_plan_prompt(base_item)
    assert "Review the implementation plan in the plan file below" in plan_p
    assert "- anything in the plan that doesn't match the code around it" in plan_p
    assert f"Plan file:\n{plan_ref}\n\nCall `submit_review`" in plan_p
    with pytest.raises(AssertionError):
        review_plan_prompt(base_item.model_copy(update={"plan_ref": None}))
    ar = address_review_prompt(state)
    assert f"- grok [CHANGES REQUESTED] (head {full_sha}): fix edge" in ar
    assert "Review findings:\n" in ar and "reuse valid results when the tree has not changed." in ar
    assert "You may defer worthwhile new features" in ar
    assert "You may disagree with unsupported, harmful, or non-beneficial findings." in ar
    assert "Commit any changes locally (do not push), then call `check_done`" in ar
    impl, impl_plan = implement_prompt(state), implement_prompt(with_plan)
    assert impl.startswith("Implement the task below.")
    assert "Reuse existing code and established patterns where appropriate." in impl
    assert "Run appropriate checks for your final changes and fix any failures" in impl
    assert "Check completion against every requirement in the task" in impl
    assert "Do not amend, rebase, or rewind." in impl
    assert "Commit your changes locally (do not push), then call `check_done`" in impl
    assert "Implement the task below, following the plan file below." in impl_plan
    assert "Reuse existing code and established patterns where appropriate." in impl_plan
    assert "Check completion against both the task and the plan file" in impl_plan
    assert (
        "Do not amend, rebase, or rewind." in impl_plan and f"Plan file:\n{plan_ref}" in impl_plan
    )
    focus = "In addition to a general review, pay particular attention to:"
    rc = review_code_prompt(base_item)
    assert f"Plan file:\n{plan_ref}" in rc and focus in rc
    assert "- anything that doesn't match the code around it" in rc
    no_ref_code = review_code_prompt(base_item.model_copy(update={"plan_ref": None}))
    assert focus in no_ref_code and "Plan file:" not in no_ref_code
    assert "None" not in no_ref_code and "\n\n\n" not in no_ref_code
    ship_init = ship_pr_prompt(state)
    assert (
        "Push `aaa` as-is with a clean working tree. "
        "Do not edit, create, or delete any files in the repository or change `HEAD`." in ship_init
    )
    all_p = (coord, wp, plan_p, rp, impl, impl_plan, rc, ar, ship_init)
    assert all(str(tmp_path) not in p and "\n\n\n" not in p for p in all_p)

    # Escalation retry instructions flow to repair + triage and expire on subsequent revisions;
    # readiness_changed warning clears once a repair advances head; merge_gate_prompt points at
    # context_ref while render_merge_gate_context captures verdicts, reviews, and escalations.
    readiness_fail = PrReadiness(
        pr_number=5,
        url="https://github.com/org/repo/pull/5",
        head_sha=full_sha,
        head_branch="feat/task",
        base_branch="main",
        ready=False,
        reasons=("check 'ci' concluded FAILURE",),
    )
    pr_bounced = PullRequest(
        number=5,
        attempts=(
            PrAttempt(
                head_sha=full_sha,
                readiness=readiness_fail,
                context_ref=ctx_ref_1,
                verdict=MergeGateVerdict(
                    decision="request_changes", head_sha=full_sha, reason="fix bug"
                ),
                approval=MergeApprovalReply(approved=True),
                merge=MergeResult(
                    status="readiness_changed", pr_number=5, head_sha=full_sha, reasons=("DIRTY",)
                ),
            ),
        ),
    )
    esc = EscalationRecord(revision=0, reason="fix bug", action="retry", instructions="retry-note")
    st_rev0 = CodeAndShipState(
        input=CodeInput(task="Do task", checkout=tmp_path),
        revisions=(CodeRevision(base_sha="b0", head_sha=full_sha),),
        pr=pr_bounced,
        escalations=[esc],
    )
    gate_p = merge_gate_prompt(st_rev0)
    assert f"PR #5 at commit `{full_sha}`" in gate_p and "via `gh`" in gate_p
    assert f"Context file:\n{ctx_ref_1}" in gate_p
    assert gate_p.count(ctx_ref_1) == 1
    assert "Execution difficulty alone is not a reason to escalate." in gate_p
    assert f"Call `submit_gate_decision` for commit `{full_sha}`" in gate_p
    assert "Human Escalation History" not in gate_p and "Prior Merge Gate Verdicts" not in gate_p
    assert str(tmp_path) not in gate_p and "\n\n\n" not in gate_p

    ctx0 = render_merge_gate_context(st_rev0, pr_bounced)
    assert "#5 (https://github.com/org/repo/pull/5)" in ctx0
    assert "check 'ci' concluded FAILURE" in ctx0 and "Prior Merge Gate Verdicts" not in ctx0
    assert "- Revision 0 (retry): retry-note (original issue: fix bug)" in ctx0
    assert "Human instructions outrank earlier verdicts and reviews." in ctx0
    assert "Human Escalation History" not in render_merge_gate_context(
        st_rev0.model_copy(update={"escalations": []}), pr_bounced
    )
    no_instr = EscalationRecord(revision=0, reason="fix bug", action="retry", instructions="")
    no_instr_ctx = render_merge_gate_context(
        st_rev0.model_copy(update={"escalations": [no_instr]}), pr_bounced
    )
    assert "- Revision 0 (retry): fix bug\n" in no_instr_ctx
    assert "(original issue:" not in no_instr_ctx

    rev0_repair, rev0_ship = repair_prompt(st_rev0), ship_pr_prompt(st_rev0)
    assert "Fix what is blocking PR #5 from merging." in rev0_repair and "retry-note" in rev0_repair
    assert "Blocking issue:\n" in rev0_repair
    assert "reuse valid results when the tree has not changed." in rev0_repair
    assert (
        f"Push `{full_sha}` as-is with a clean working tree. "
        "Do not edit, create, or delete any files in the repository or change `HEAD`." in rev0_ship
    )
    assert "Previous merge aborted because PR readiness changed" in rev0_ship
    assert str(tmp_path) not in rev0_repair and str(tmp_path) not in rev0_ship

    revs1 = (
        CodeRevision(base_sha="b0", head_sha=full_sha),
        CodeRevision(base_sha=full_sha, head_sha="h2"),
    )
    st_rev1 = st_rev0.model_copy(update={"revisions": revs1})
    rev1_triage, rev1_repair, rev1_ship = (
        triage_repair_prompt(st_rev1),
        repair_prompt(st_rev1),
        ship_pr_prompt(st_rev1),
    )
    assert "Blocking issue:\n" in rev1_triage
    assert "retry-note" in rev1_triage and "retry-note" not in rev1_repair
    assert "Previous merge aborted because PR readiness changed" not in rev1_ship
    assert str(tmp_path) not in rev1_triage
    assert "\n\n\n" not in rev1_repair and "\n\n\n" not in rev1_ship

    pr_attempt2 = PullRequest(
        number=5,
        attempts=(
            pr_bounced.attempts[0],
            PrAttempt(
                head_sha="h3",
                readiness=readiness_fail.model_copy(
                    update={"head_sha": "h3", "ready": True, "reasons": ()}
                ),
                context_ref=ctx_ref_2,
            ),
        ),
    )
    rev2_triage_v = RepairTriageVerdict(head_sha="h3", needs_code_review=False, reason="lint-only")
    rev0_code_review = ReviewRecord(
        stage="code",
        revision=0,
        reviewer="opus_high",
        approved=True,
        summary="solid",
        head_sha=full_sha,
    )
    st_rev2 = st_rev1.model_copy(
        update={
            "revisions": (*revs1, CodeRevision(base_sha="h2", head_sha="h3", triage=rev2_triage_v)),
            "reviews": [rev0_code_review],
            "pr": pr_attempt2,
        }
    )
    rev2_triage, rev2_gate = triage_repair_prompt(st_rev2), merge_gate_prompt(st_rev2)
    assert "retry-note" not in rev2_triage and "\n\n\n" not in rev2_triage
    assert f"Context file:\n{ctx_ref_2}" in rev2_gate and "\n\n\n" not in rev2_gate
    ctx2 = render_merge_gate_context(st_rev2, pr_attempt2)
    assert "repair triage skipped re-review" in ctx2 and "lint-only" in ctx2
    assert "## Latest Code Reviews (from before the repair)" in ctx2
    assert f"- opus_high [APPROVED] (head {full_sha}): solid" in ctx2
    assert "## Prior Merge Gate Verdicts" in ctx2
    assert f"- Attempt 1 (head {full_sha}): request_changes — fix bug" in ctx2
    assert "## Human Escalation History" in ctx2
    assert "- Revision 0 (retry): retry-note (original issue: fix bug)" in ctx2
    assert "\n\n\n" not in ctx2

    # 1:1 constant + placeholder parity between prompt_templates.py and prompts.py
    defined = {k: v for k, v in vars(prompt_templates).items() if not k.startswith("_")}
    prm_path = Path(__file__).resolve().parent.parent / "castle_steward_coding" / "prompts.py"
    prm_tree = ast.parse(prm_path.read_text(encoding="utf-8"))

    def _tpl(e: ast.AST) -> str | None:
        return (
            e.attr
            if isinstance(e, ast.Attribute)
            and isinstance(e.value, ast.Name)
            and e.value.id == "tpl"
            else None
        )

    assert set(defined) == {n for node in ast.walk(prm_tree) if (n := _tpl(node))}
    fmt = string.Formatter()
    for call in (c for c in ast.walk(prm_tree) if isinstance(c, ast.Call)):
        fn = call.func
        name = (
            _tpl(call.args[0])
            if isinstance(fn, ast.Name) and fn.id == "_render" and call.args
            else (_tpl(fn.value) if isinstance(fn, ast.Attribute) and fn.attr == "format" else None)
        )
        if name is not None:
            expected_fields = {f for _, f, _, _ in fmt.parse(defined[name]) if f is not None}
            assert expected_fields == {kw.arg for kw in call.keywords}, f"tpl.{name}"


def test_input_checkout_and_submission_constraints(tmp_path: Path) -> None:
    for rel in (Path("."), "relative/repo"):
        with pytest.raises(ValidationError, match="absolute path"):
            CodeInput.model_validate({"task": "t", "checkout": rel})
    assert CodeInput(task="t", checkout=tmp_path / "a" / "..").checkout == tmp_path.resolve()
    for bad_mirror in (Path("/tmp/cache"), Path("."), Path("../escape"), Path("a/../b")):
        with pytest.raises(ValidationError, match="artifact_mirror_dir"):
            CodingConfig.model_validate({"artifact_mirror_dir": bad_mirror})
    with pytest.raises(ValidationError, match="greater than 0"):
        VerifiedPr(pr_number=0, head_sha="a")
    with pytest.raises(ValidationError, match="at least 1 character"):
        WrittenPlan(plan="  \n\t ")
    with pytest.raises(ValidationError):
        TaskAssessment(difficulty="low", needs_plan=False, rationale="   ")
    assert WrittenPlan.model_json_schema()["properties"]["plan"]["minLength"] == 1
    assert set(CODE_AND_SHIP_WORKFLOW.roles) == {"orchestrator", "coder", "reviewer"}
    coder_tools = [
        t.name
        for t in SessionToolHost(
            tmp_path / "mcp-coder-list", CODE_AND_SHIP_WORKFLOW.roles["coder"]
        ).list_tools()
    ]
    assert coder_tools == ["submit_plan", "ask", "check_done"]
    submission_tools = 0
    for role in CODE_AND_SHIP_WORKFLOW.roles.values():
        host = SessionToolHost(tmp_path / f"mcp-{role.name}", role)
        for tool in host.list_tools():
            if tool.name in ("ask", "check_done"):
                continue
            submission_tools += 1
            assert tool.description and tool.description.strip()
            props = tool.input_schema["properties"]
            assert props and all(
                isinstance(prop.get("description"), str) and prop["description"].strip()
                for prop in props.values()
            )
    assert submission_tools == 5
    with pytest.raises(ValueError, match="Cannot change PR number"):
        PullRequest.ship(PullRequest.ship(None, number=7, head_sha="a"), number=8, head_sha="b")


def test_composable_state_schemas_and_input_invariants(tmp_path: Path) -> None:
    inp = CodeInput(task="t", checkout=tmp_path)
    for abstract_cls in (WorkflowState, ShipStateBase):
        with pytest.raises(TypeError):
            abstract_cls(input=inp)  # type: ignore[abstract]
    assert set(CodeState.model_fields) == {
        "input",
        "setup",
        "assignment",
        "revisions",
        "reviews",
        "coordination",
        "plans",
    }
    assert CodeState.model_fields["input"].annotation is CodeInput
    assert set(ShipState.model_fields) == {
        "input",
        "setup",
        "assignment",
        "revisions",
        "reviews",
        "pr",
        "escalations",
        "retries",
    }
    assert ShipState.model_fields["input"].annotation is ShipInput
    assert len(CodeAndShipState.model_fields) == 10
    assert CodeAndShipState.model_fields["input"].annotation is CodeInput
    with pytest.raises(ValidationError, match="task"):
        CodeInput.model_validate({"checkout": tmp_path})
    with pytest.raises(ValidationError, match="difficulty"):
        ShipInput.model_validate({"checkout": tmp_path})
    with pytest.raises(ValidationError, match="Extra inputs"):
        ShipInput.model_validate({"checkout": tmp_path, "difficulty": "low", "task": "extra"})
    with pytest.raises(ValidationError):
        ShipInput.model_validate({"checkout": tmp_path, "difficulty": "low", "pr_number": True})


def test_code_workflow_completes_locally_and_cleans_mirror(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, "code")
    commits = 0

    def handler(ctx: FakeTurnContext) -> None:
        nonlocal commits
        t, chk = ctx.active_turn.expects_tool, ctx.active_turn.expects_check
        assert chk != "pr" and t != "submit_gate_decision"
        if t == "submit_assessment":
            ctx.call_tool(t, {"difficulty": "medium", "needs_plan": True, "rationale": "p"})
        elif t == "submit_plan":
            ctx.call_tool(t, {"plan": "# Plan\n- step\n"})
        elif t == "submit_review":
            ctx.call_tool(t, {"approved": True, "summary": "nit", "findings": ["f"]})
        elif chk == "commit":
            commits += 1
            _commit(repo, "feat.py", f"v = {commits}\n", f"c{commits}")
            assert not ctx.call_tool("check_done", {}).is_error

    al = _fake_al(handler)
    with Engine(state_dir=tmp_path / "st", agent_layer=al, workflows=_WORKFLOWS) as eng:
        jid = eng.start("code", _base_input(repo))
        st = eng.run(jid)
        assert st.phase == "completed" and st.state is not None
        d = cast(dict[str, Any], st.state)
        assert "pr" not in d and "escalations" not in d and "retries" not in d
        assert len(d["plans"]) == 2 and len(d["revisions"]) == 1
        assert not (repo / ".agent-layer/cache/steward-engine" / str(jid)).exists()
        assert (eng.store.artifacts_dir(jid) / "plan-v2.md").is_file()
        assert _git(repo, "rev-parse", "HEAD") == d["revisions"][0]["head_sha"]
        assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "feat/task"
        assert [c.expects_tool or c.expects_check for c in al.calls if c.role == "coder"] == [
            "submit_plan",
            "submit_plan",
            "commit",
            "commit",
        ]
        reloaded = CodeState.model_validate(d)
        assert reloaded.input.task == "Add feature" and len(reloaded.plans) == 2


def test_ship_workflow_flows_resume_and_fail_fast_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _base_input(tmp_path, config={"allow_unattended_merge": True})["config"]
    gh_base: dict[str, Any] = {
        "state": "OPEN",
        "headRefName": "feat/task",
        "baseRefName": "main",
        "checks": [{"name": "ci", "status": "COMPLETED", "conclusion": "SUCCESS"}],
    }

    # 1. No-PR branch -> ship_pr opens PR -> pause at ask_approval -> fresh Engine resume + merge
    r_new = _init_repo(tmp_path, "no-pr")
    h_new = _commit(r_new, "a.py", "a = 1\n", "init")
    gh_new = _install_fake_gh(
        tmp_path / "gh-new",
        monkeypatch,
        {**gh_base, "number": 33, "url": "u/33", "headRefOid": "", "no_pr": True},
    )

    def h_open(ctx: FakeTurnContext) -> None:
        t, chk = ctx.active_turn.expects_tool, ctx.active_turn.expects_check
        if chk == "pr":
            assert "existing PR" not in ctx.prompt
            _sync_gh_head(gh_new, h_new)
            assert not ctx.call_tool("check_done", {}).is_error
        elif t == "submit_gate_decision":
            ctx.call_tool(t, {"decision": "merge", "head_sha": h_new, "reason": "ok"})

    st_dir = tmp_path / "st-resume"
    al_new = _fake_al(h_open)
    with Engine(state_dir=st_dir, agent_layer=al_new, workflows=_WORKFLOWS) as eng:
        jid = eng.start(
            "ship",
            {
                "checkout": r_new,
                "difficulty": "low",
                "config": {**cfg, "allow_unattended_merge": False},
            },
        )
        paused = eng.run(jid)
        assert paused.phase == "paused" and paused.pending_waits[0].kind == "merge_approval"
        wait_id = paused.pending_waits[0].wait_id
    with Engine(state_dir=st_dir, agent_layer=al_new, workflows=_WORKFLOWS) as eng2:
        eng2.reply(jid, wait_id, {"approved": True})
        done = eng2.run(jid)
        assert done.phase == "completed" and done.state is not None
        d1 = cast(dict[str, Any], done.state)
        assert d1["pr"]["number"] == 33 and d1["pr"]["attempts"][0]["merge"]["status"] == "merged"

    # 2. Explicit pr_number=55 on unpushed branch -> repair -> triage -> review -> coder continuity
    r_rep = _init_repo(tmp_path, "unpushed")
    head0 = _commit(r_rep, "auth.py", "ok = 1\n", "Add auth fix")
    gh_rep = _install_fake_gh(
        tmp_path / "gh-rep", monkeypatch, {**gh_base, "number": 55, "url": "u/55", "headRefOid": ""}
    )
    ships = gates = rep_commits = 0

    def h_rep(ctx: FakeTurnContext) -> None:
        nonlocal ships, gates, rep_commits
        t, chk = ctx.active_turn.expects_tool, ctx.active_turn.expects_check
        sha = _git(r_rep, "rev-parse", "HEAD")
        assert t != "submit_assessment"
        if chk == "pr":
            ships += 1
            assert "existing PR #55" in ctx.prompt
            _sync_gh_head(gh_rep, sha)
            assert not ctx.call_tool("check_done", {}).is_error
        elif t == "submit_gate_decision":
            gates += 1
            assert prompt_templates.SHIP_DEFAULT_TASK in ctx.prompt
            dec = "request_changes" if gates == 1 else "merge"
            ctx.call_tool(t, {"decision": dec, "head_sha": sha, "reason": f"g{gates}"})
        elif chk == "commit":
            rep_commits += 1
            _commit(r_rep, "auth.py", f"ok = {rep_commits + 1}\n", f"fix {rep_commits}")
            assert not ctx.call_tool("check_done", {}).is_error
        elif t == "submit_repair_triage":
            assert f"{head0}.." in ctx.prompt
            ctx.call_tool(t, {"needs_code_review": True, "head_sha": sha, "reason": "logic"})
        elif t == "submit_review":
            ctx.call_tool(t, {"approved": True, "summary": "nit", "findings": ["p"]})

    al_rep = _fake_al(h_rep)
    ship_in = {"checkout": r_rep, "difficulty": "low", "pr_number": 55, "config": cfg}
    with Engine(state_dir=tmp_path / "st-rep", agent_layer=al_rep, workflows=_WORKFLOWS) as eng:
        jid2 = eng.start("ship", ship_in)
        st2 = eng.run(jid2)
        assert st2.phase == "completed" and st2.state is not None
        d2 = cast(dict[str, Any], st2.state)
        a2 = d2["pr"]["attempts"]
        assert ships == gates == rep_commits == 2 and d2["pr"]["number"] == 55
        assert len(a2) == 2 and a2[-1]["merge"]["status"] == "merged"
        assert not (r_rep / ".agent-layer/cache/steward-engine" / str(jid2)).exists()
        coder = [c for c in al_rep.calls if c.role == "coder"]
        assert len({c.handle for c in coder}) == 1
        assert [c.is_continue for c in coder] == [False, True, True, True]
        assert [c.expects_check for c in coder] == ["pr", "commit", "commit", "pr"]

    # 3. Existing open PR at HEAD -> skips ship_pr, uses SHIP_DEFAULT_TASK
    r_open = _init_repo(tmp_path, "open-pr")
    h_open_sha = _commit(r_open, "w.py", "w = 1\n", "Fix widget")
    _install_fake_gh(
        tmp_path / "gh-open",
        monkeypatch,
        {**gh_base, "number": 77, "url": "u/77", "headRefOid": h_open_sha},
    )

    def h_skip(ctx: FakeTurnContext) -> None:
        t = ctx.active_turn.expects_tool
        assert t == "submit_gate_decision" and prompt_templates.SHIP_DEFAULT_TASK in ctx.prompt
        ctx.call_tool(t, {"decision": "merge", "head_sha": h_open_sha, "reason": "ok"})

    al_skip = _fake_al(h_skip)
    with Engine(state_dir=tmp_path / "st-open", agent_layer=al_skip, workflows=_WORKFLOWS) as eng:
        st3 = eng.run(
            eng.start("ship", {"checkout": r_open, "difficulty": "medium", "config": cfg})
        )
        assert st3.phase == "completed" and st3.state is not None
        d3 = cast(dict[str, Any], st3.state)
        assert "coordination" not in d3 and "plans" not in d3
        assert [c.role for c in al_skip.calls] == ["orchestrator"] and d3["pr"]["number"] == 77
        assert d3["pr"]["attempts"][0]["merge"]["status"] == "merged"
        r3 = ShipState.model_validate(d3)
        assert r3.task_text is None and task_summary(r3) == prompt_templates.SHIP_DEFAULT_TASK

    # 4. Fail-fast setup rejections (0 agent calls)
    r_fail = _init_repo(tmp_path, "fail-fast")
    h_fail = _commit(r_fail, "x.py", "x = 1\n", "c")
    al_none = _fake_al(lambda _c: (_ for _ in ()).throw(AssertionError("agent must not run")))
    cases: list[tuple[str, dict[str, Any], int | None, type[Exception], str]] = [
        (
            "err",
            {**gh_base, "number": 99, "url": "u/99", "headRefOid": h_fail, "gh_error": "auth"},
            None,
            GitHubError,
            "auth",
        ),
        (
            "closed",
            {**gh_base, "number": 99, "url": "u/99", "headRefOid": h_fail, "state": "CLOSED"},
            99,
            ValueError,
            "expected OPEN",
        ),
        (
            "branch",
            {**gh_base, "number": 99, "url": "u/99", "headRefOid": h_fail, "headRefName": "other"},
            99,
            ValueError,
            "branch",
        ),
    ]
    for name, gh_st, pr_num, exc_cls, pat in cases:
        _install_fake_gh(tmp_path / f"gh-{name}", monkeypatch, gh_st)
        raw: dict[str, Any] = {"checkout": r_fail, "difficulty": "low", "config": cfg}
        if pr_num is not None:
            raw["pr_number"] = pr_num
        with (
            Engine(
                state_dir=tmp_path / f"st-{name}", agent_layer=al_none, workflows=_WORKFLOWS
            ) as eng,
            pytest.raises(exc_cls, match=pat),
        ):
            eng.run(eng.start("ship", raw))
    _git(r_fail, "checkout", "--detach", "HEAD")
    with (
        Engine(state_dir=tmp_path / "st-detach", agent_layer=al_none, workflows=_WORKFLOWS) as eng,
        pytest.raises(ValueError, match="detached HEAD"),
    ):
        eng.run(eng.start("ship", {"checkout": r_fail, "difficulty": "low", "config": cfg}))
    assert al_none.calls == []
