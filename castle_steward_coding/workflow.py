"""Declarative coding workflows built strictly on top-level `castle_steward_engine` exports."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from pydantic import JsonValue

from castle_steward_coding import github, models, prompts, routing, tools
from castle_steward_engine import (
    END,
    AgentSpec,
    AvailableAgent,
    Graph,
    JobInfo,
    Role,
    Turn,
    Wait,
    Workflow,
)

_READ_ONLY_FILE_MODE = 0o444
_PROBE_ARTIFACT_NAME = "probe.md"
_PLAN_ARTIFACT_TEMPLATE = "plan-v{version}.md"
_MERGE_CONTEXT_ARTIFACT_TEMPLATE = "merge-context-a{attempt}.md"

# --- Completion checks ---


def _require_check_ctx(context: dict[str, JsonValue], key: str) -> JsonValue:
    val = context.get(key)
    if val is None:
        raise ValueError(f"Completion check context missing required key {key!r}")
    return val


def _check_commit(context: dict[str, JsonValue]) -> models.VerifiedCommit:
    checkout = Path(str(_require_check_ctx(context, "checkout")))
    baseline = str(_require_check_ctx(context, "baseline"))
    require_new = bool(_require_check_ctx(context, "require_new_commit"))
    head = github.verify_commit_lineage(
        checkout, baseline_sha=baseline, require_new_commit=require_new
    )
    return models.VerifiedCommit(head_sha=head)


def _check_pr(context: dict[str, JsonValue]) -> models.VerifiedPr:
    checkout = Path(str(_require_check_ctx(context, "checkout")))
    expected_head = str(_require_check_ctx(context, "expected_head"))
    raw_pr = context.get("expected_pr")
    if raw_pr is not None and type(raw_pr) is not int:
        raise ValueError(f"Completion check context 'expected_pr' must be int, got {raw_pr!r}")
    pr_number, head = github.verify_shipped_pr(
        checkout, expected_head=expected_head, expected_pr=raw_pr
    )
    return models.VerifiedPr(pr_number=pr_number, head_sha=head)


# --- Roles ---


ORCHESTRATOR = Role(
    name="orchestrator",
    tools={
        "submit_assessment": tools.TaskAssessment,
        "submit_repair_triage": tools.RepairTriage,
        "submit_gate_decision": tools.MergeGateDecision,
    },
    can_ask_human=True,
)
CODER = Role(
    name="coder",
    tools={"submit_plan": tools.WrittenPlan},
    checks={"commit": _check_commit, "pr": _check_pr},
    can_ask_human=True,
)
REVIEWER = Role(name="reviewer", tools={"submit_review": tools.ReviewFindings}, can_ask_human=False)


# --- Step & routing helpers ---


def _setup(state: models.WorkflowState) -> models.RunSetup:
    assert state.setup is not None
    return state.setup


def _pr(state: models.ShipStateBase) -> models.PullRequest:
    assert state.pr is not None
    return state.pr


def _readiness(state: models.ShipStateBase) -> github.PrReadiness:
    readiness = _pr(state).current.readiness
    assert readiness is not None
    return readiness


def _revision(state: models.WorkflowState) -> models.CodeRevision:
    rev = state.current_revision
    assert rev is not None
    return rev


def _replace_latest_revision(state: models.WorkflowState, **changes: Any) -> dict[str, Any]:
    updated = models.validated_replace(_revision(state), **changes)
    return {"revisions": (*state.revisions[:-1], updated)}


def _artifact_mirror_rel_path(mirror_root: Path, job_id: UUID, name: str) -> Path:
    return mirror_root / str(job_id) / name


def _require_ignored_artifact_path(checkout: Path, mirror_root: Path, rel_path: Path) -> None:
    if not github.is_ignored(checkout, rel_path):
        raise ValueError(
            f"Artifact cache path {rel_path.as_posix()!r} is not gitignored in {checkout}; "
            f"ensure {mirror_root.as_posix()!r} is ignored by Git."
        )


def _atomic_write_text(path: Path, content: str, *, read_only: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp_path = Path(fh.name)
            fh.write(content)
        if read_only:
            os.chmod(tmp_path, _READ_ONLY_FILE_MODE)
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def _mirror_artifact(
    checkout: Path, mirror_root: Path, job: JobInfo, name: str, content: str
) -> str:
    rel_path = _artifact_mirror_rel_path(mirror_root, job.job_id, name)
    _require_ignored_artifact_path(checkout, mirror_root, rel_path)
    _atomic_write_text(checkout / rel_path, content, read_only=True)
    return rel_path.as_posix()


def _write_artifact(
    checkout: Path, mirror_root: Path, job: JobInfo, name: str, content: str
) -> str:
    _atomic_write_text(job.artifacts_dir / name, content, read_only=False)
    return _mirror_artifact(checkout, mirror_root, job, name, content)


def _sync_latest_plan_mirror(state: models.WorkflowState, job: JobInfo) -> None:
    if (plan := state.latest_plan) is not None and plan.ref is not None:
        _mirror_artifact(
            state.input.checkout,
            _setup(state).config.artifact_mirror_dir,
            job,
            Path(plan.ref).name,
            plan.text,
        )


def _remove_mirror_dir(checkout: Path, mirror_root: Path, job_id: UUID) -> None:
    mirror_dir = checkout / mirror_root / str(job_id)
    if mirror_dir.exists():
        shutil.rmtree(mirror_dir)


def _step_noop(_state: models.WorkflowState, _job: JobInfo) -> dict[str, Any]:
    return {}


def _resolve_common_setup(
    state: models.WorkflowState, job: JobInfo
) -> tuple[Path, models.CodingConfig, dict[str, AvailableAgent], AgentSpec]:
    checkout = github.verify_checkout_root(state.input.checkout)
    config = state.input.config or models.CodingSettings().to_config()
    _require_ignored_artifact_path(
        checkout,
        config.artifact_mirror_dir,
        _artifact_mirror_rel_path(config.artifact_mirror_dir, job.job_id, _PROBE_ARTIFACT_NAME),
    )
    avail = job.available_agents(checkout)
    eligible = {name: info for name, info in avail.items() if info.session_mcp}
    if not eligible:
        raise ValueError("No available agents support session-scoped MCP (session_mcp=True)")
    orchestrator = routing.resolve_agent(state.input.orchestrator or config.orchestrator, eligible)
    return checkout, config, eligible, orchestrator


def _step_resolve_setup(state: models.CodeState, job: JobInfo) -> dict[str, Any]:
    _checkout, config, _eligible, orchestrator = _resolve_common_setup(state, job)
    return {"setup": models.RunSetup(config=config, orchestrator=orchestrator)}


def _step_resolve_ship_setup(state: models.ShipState, job: JobInfo) -> dict[str, Any]:
    checkout, config, eligible, orchestrator = _resolve_common_setup(state, job)
    head = github.head_sha(checkout)
    probed = github.probe_ship_pr(checkout, expected_head=head, expected_pr=state.input.pr_number)
    assignment = routing.assign_agents(
        difficulty=state.input.difficulty,
        needs_plan=False,
        policy=config.routing,
        eligible=eligible,
        quota=job.agent_quota(checkout),
    )
    update: dict[str, Any] = {
        "setup": models.RunSetup(config=config, orchestrator=orchestrator),
        "assignment": assignment,
        "revisions": (models.CodeRevision(base_sha=head, head_sha=head),),
    }
    if probed is not None:
        pr_num, pr_head = probed
        update["pr"] = models.PullRequest.ship(None, number=pr_num, head_sha=pr_head)
    return update


def _step_assign_agents(state: models.CodeState, job: JobInfo) -> dict[str, Any]:
    setup = _setup(state)
    coord = state.coordination
    assert coord is not None
    avail = job.available_agents(state.input.checkout)
    eligible = {name: info for name, info in avail.items() if info.session_mcp}
    quota = job.agent_quota(state.input.checkout)
    assignment = routing.assign_agents(
        difficulty=coord.difficulty,
        needs_plan=coord.needs_plan,
        policy=setup.config.routing,
        eligible=eligible,
        quota=quota,
    )
    return {"assignment": assignment}


def _step_save_plan(state: models.CodeState, job: JobInfo) -> dict[str, Any]:
    plan = state.latest_plan
    assert plan is not None
    ref = _write_artifact(
        state.input.checkout,
        _setup(state).config.artifact_mirror_dir,
        job,
        _PLAN_ARTIFACT_TEMPLATE.format(version=len(state.plans)),
        plan.text,
    )
    updated = models.validated_replace(plan, ref=ref)
    return {"plans": (*state.plans[:-1], updated)}


def _step_collect_plan_reviews(state: models.CodeState, job: JobInfo) -> dict[str, Any]:
    _sync_latest_plan_mirror(state, job)
    return {}


def _step_record_baseline(state: models.CodeState, job: JobInfo) -> dict[str, Any]:
    _sync_latest_plan_mirror(state, job)
    sha = github.head_sha(state.input.checkout)
    return {"revisions": (models.CodeRevision(base_sha=sha, head_sha=sha),)}


def _step_dispatch_code_review(state: models.WorkflowState, job: JobInfo) -> dict[str, Any]:
    _sync_latest_plan_mirror(state, job)
    return {}


def _step_check_readiness(state: models.ShipStateBase, job: JobInfo) -> dict[str, Any]:
    pr = _pr(state)
    cfg = _setup(state).config
    readiness = github.wait_for_readiness(
        state.input.checkout,
        head=pr.current.head_sha,
        pr_number=pr.number,
        poll_seconds=cfg.readiness_poll_seconds,
        timeout_seconds=cfg.readiness_timeout_seconds,
        check_stop=job.check_stop,
    )
    updated_pr = pr.update_current(readiness=readiness)
    context_ref = _write_artifact(
        state.input.checkout,
        cfg.artifact_mirror_dir,
        job,
        _MERGE_CONTEXT_ARTIFACT_TEMPLATE.format(attempt=len(updated_pr.attempts)),
        prompts.render_merge_gate_context(state, updated_pr),
    )
    return {"pr": updated_pr.update_current(context_ref=context_ref)}


def _step_merge_exact_head(state: models.ShipStateBase, _job: JobInfo) -> dict[str, Any]:
    pr = _pr(state)
    readiness = pr.current.readiness
    cfg = _setup(state).config
    assert readiness is not None
    result = github.merge_pr_exact_head(
        state.input.checkout,
        readiness.pr_number,
        readiness.head_sha,
        merge_method=cfg.merge_method,
    )
    update: dict[str, Any] = {"pr": pr.update_current(merge=result)}
    if result.status == "readiness_changed":
        update["retries"] = models.validated_replace(
            state.retries, readiness_checks=state.retries.readiness_checks + 1
        )
    return update


def _step_verify_merge(state: models.ShipStateBase, _job: JobInfo) -> dict[str, Any]:
    pr = _pr(state)
    readiness = pr.current.readiness
    assert readiness is not None
    verified = github.verify_pr_merged(
        state.input.checkout, readiness.pr_number, readiness.head_sha
    )
    return {"pr": pr.update_current(merge=verified)}


def _step_cleanup(state: models.ShipStateBase, job: JobInfo) -> dict[str, Any]:
    pr = _pr(state)
    readiness = pr.current.readiness
    merge = pr.current.merge
    assert readiness is not None and merge is not None
    cleaned = github.cleanup_merged_branch(state.input.checkout, readiness.pr_number)
    _remove_mirror_dir(state.input.checkout, _setup(state).config.artifact_mirror_dir, job.job_id)
    updated_merge = models.validated_replace(merge, cleaned_branch=cleaned)
    return {"pr": pr.update_current(merge=updated_merge)}


def _step_cleanup_mirror(state: models.WorkflowState, job: JobInfo) -> dict[str, Any]:
    _remove_mirror_dir(state.input.checkout, _setup(state).config.artifact_mirror_dir, job.job_id)
    return {}


def _review_items(
    state: models.WorkflowState, stage: Literal["plan", "code"]
) -> list[models.ReviewItem]:
    assert state.assignment is not None
    plan = state.latest_plan
    return [
        models.ReviewItem(
            stage=stage,
            reviewer=r.profile,
            agent=r.agent,
            checkout=state.input.checkout,
            task=prompts.task_summary(state),
            revision=state.revision,
            plan_ref=plan.ref if plan else None,
            baseline=state.diff_base if stage == "code" else None,
            reviewed_head=state.head if stage == "code" else None,
        )
        for r in state.assignment.reviewers
    ]


def _ask_approval(state: models.ShipStateBase) -> Wait:
    readiness = _readiness(state)
    verdict = _pr(state).current.verdict
    assert verdict is not None
    return Wait(
        kind="merge_approval",
        prompt=(
            f"Approve merging PR #{readiness.pr_number} ({github.short_sha(readiness.head_sha)})?"
        ),
        payload={
            "pr_number": readiness.pr_number,
            "url": readiness.url,
            "head_sha": readiness.head_sha,
            "reason": verdict.reason,
        },
    )


def _ask_escalation(state: models.ShipStateBase) -> Wait:
    reason = prompts.escalation_reason(state)
    return Wait(
        kind="escalation",
        prompt=f"Workflow escalated after revision {state.revision}: {reason}",
        payload={"revision": state.revision, "reason": reason},
    )


def _route_coordination(state: models.CodeState) -> str:
    return "plan" if (state.coordination and state.coordination.needs_plan) else "implement"


def _route_plan_reviews(state: models.CodeState) -> str:
    return "skip" if state.reviews_clean("plan") else "revise"


def _route_repair_triage(state: models.ShipStateBase) -> str:
    rev = state.current_revision
    assert rev is not None and rev.triage is not None and rev.triage.head_sha == rev.head_sha
    return "review" if rev.triage.needs_code_review else "skip"


def _route_code_reviews(state: models.WorkflowState) -> str:
    return "skip" if state.reviews_clean("code", head_sha=state.head) else "address"


def _route_merge_gate(state: models.ShipStateBase) -> str:
    attempt = _pr(state).current
    cfg = _setup(state).config
    assert attempt.verdict is not None
    ready = bool(attempt.readiness and attempt.readiness.ready)
    if attempt.verdict.decision == "escalate":
        return "escalate"
    if attempt.verdict.decision == "request_changes" or not ready:
        return "repair" if state.retries.auto_repairs < cfg.max_auto_repairs else "escalate"
    if attempt.verdict.decision == "request_human_approval":
        return "approve"
    return "merge" if cfg.allow_unattended_merge else "approve"


def _route_approval(state: models.ShipStateBase) -> str:
    return (
        "merge"
        if (state.pr and state.pr.current.approval and state.pr.current.approval.approved)
        else "stop"
    )


def _route_escalation(state: models.ShipStateBase) -> str:
    return "stop" if (state.escalations and state.escalations[-1].action == "abort") else "repair"


def _route_merge_result(state: models.ShipStateBase) -> str:
    merge = _pr(state).current.merge
    cfg = _setup(state).config
    assert merge is not None
    if merge.status == "merged":
        return "merged"
    if state.retries.readiness_checks <= cfg.max_readiness_retries:
        return "readiness_changed"
    return "escalate"


def _route_ship_entry(state: models.ShipStateBase) -> str:
    return "check" if state.pr is not None else "ship"


# --- Declarative graph ---


def _add_code_review_loop[S: models.WorkflowState](g: Graph[S], *, after_review: str) -> None:
    g.step("dispatch_code_review", _step_dispatch_code_review)
    g.agent(
        "review_code",
        role=REVIEWER,
        expects=tools.ReviewFindings,
        ask=lambda item: Turn(
            session=f"reviewer:{item.reviewer}",
            prompt=prompts.review_code_prompt(item),
            agent=item.agent,
            workdir=item.checkout,
        ),
        then=lambda item, r: {"reviews": [item.to_review(r)]},
    )
    g.step("collect_code_reviews", _step_noop)
    g.agent(
        "address_review",
        role=CODER,
        expects=models.VerifiedCommit,
        check="commit",
        ask=lambda s: Turn(
            session="coder",
            prompt=prompts.address_review_prompt(s),
            context={
                "checkout": str(s.input.checkout),
                "baseline": s.head,
                "require_new_commit": False,
            },
            agent=s.coder_agent(),
            workdir=s.input.checkout,
        ),
        then=lambda s, r: _replace_latest_revision(s, head_sha=r.head_sha),
    )
    g.fan_out(
        "dispatch_code_review", "review_code", models.ReviewItem, lambda s: _review_items(s, "code")
    )
    g.edge("review_code", "collect_code_reviews")
    g.branch(
        "collect_code_reviews",
        _route_code_reviews,
        {"skip": after_review, "address": "address_review"},
    )
    g.edge("address_review", after_review)


def _add_code_stage[S: models.CodeState](g: Graph[S], *, after_review: str) -> None:
    g.step("resolve_setup", _step_resolve_setup)
    g.agent(
        "coordinate",
        role=ORCHESTRATOR,
        expects=tools.TaskAssessment,
        ask=lambda s: Turn(
            session="orchestrator:coordinate",
            prompt=prompts.coordinate_prompt(s),
            agent=_setup(s).orchestrator,
            workdir=s.input.checkout,
        ),
        then=lambda _s, r: {
            "coordination": models.TaskClassification(
                difficulty=r.difficulty, needs_plan=r.needs_plan, rationale=r.rationale
            )
        },
    )
    g.step("assign_agents", _step_assign_agents)
    g.agent(
        "write_plan",
        role=CODER,
        expects=tools.WrittenPlan,
        ask=lambda s: Turn(
            session="coder",
            prompt=prompts.write_plan_prompt(s),
            agent=s.coder_agent(),
            workdir=s.input.checkout,
        ),
        then=lambda s, r: {"plans": (*s.plans, models.PlanVersion(text=r.plan))},
    )
    g.step("save_plan", _step_save_plan)
    g.agent(
        "review_plan",
        role=REVIEWER,
        expects=tools.ReviewFindings,
        ask=lambda item: Turn(
            session=f"reviewer:{item.reviewer}",
            prompt=prompts.review_plan_prompt(item),
            agent=item.agent,
            workdir=item.checkout,
        ),
        then=lambda item, r: {"reviews": [item.to_review(r)]},
    )
    g.step("collect_plan_reviews", _step_collect_plan_reviews)
    g.agent(
        "revise_plan",
        role=CODER,
        expects=tools.WrittenPlan,
        ask=lambda s: Turn(
            session="coder",
            prompt=prompts.revise_plan_prompt(s),
            agent=s.coder_agent(),
            workdir=s.input.checkout,
        ),
        then=lambda s, r: {"plans": (*s.plans, models.PlanVersion(text=r.plan))},
    )
    g.step("save_revised_plan", _step_save_plan)
    g.step("record_baseline", _step_record_baseline)
    g.agent(
        "implement",
        role=CODER,
        expects=models.VerifiedCommit,
        check="commit",
        ask=lambda s: Turn(
            session="coder",
            prompt=prompts.implement_prompt(s),
            context={
                "checkout": str(s.input.checkout),
                "baseline": s.head,
                "require_new_commit": True,
            },
            agent=s.coder_agent(),
            workdir=s.input.checkout,
        ),
        then=lambda s, r: _replace_latest_revision(s, head_sha=r.head_sha),
    )
    _add_code_review_loop(g, after_review=after_review)

    g.entry("resolve_setup")
    g.edge("resolve_setup", "coordinate")
    g.edge("coordinate", "assign_agents")
    g.branch(
        "assign_agents",
        _route_coordination,
        {"plan": "write_plan", "implement": "record_baseline"},
    )
    g.edge("write_plan", "save_plan")
    g.fan_out("save_plan", "review_plan", models.ReviewItem, lambda s: _review_items(s, "plan"))
    g.edge("review_plan", "collect_plan_reviews")
    g.branch(
        "collect_plan_reviews",
        _route_plan_reviews,
        {"skip": "record_baseline", "revise": "revise_plan"},
    )
    g.edge("revise_plan", "save_revised_plan")
    g.edge("save_revised_plan", "record_baseline")
    g.edge("record_baseline", "implement")
    g.edge("implement", "dispatch_code_review")


def _add_ship_stage[S: models.ShipStateBase](g: Graph[S]) -> None:
    g.agent(
        "triage_repair",
        role=ORCHESTRATOR,
        expects=tools.RepairTriage,
        ask=lambda s: Turn(
            session=f"orchestrator:triage:r{s.revision}",
            prompt=prompts.triage_repair_prompt(s),
            context={"expected_head": s.head},
            agent=_setup(s).orchestrator,
            workdir=s.input.checkout,
        ),
        then=lambda s, r: _replace_latest_revision(
            s,
            triage=models.RepairTriageVerdict(
                head_sha=r.head_sha, needs_code_review=r.needs_code_review, reason=r.reason
            ),
        ),
    )
    g.agent(
        "ship_pr",
        role=CODER,
        expects=models.VerifiedPr,
        check="pr",
        ask=lambda s: Turn(
            session="coder",
            prompt=prompts.ship_pr_prompt(s),
            context={
                "checkout": str(s.input.checkout),
                "expected_head": s.head,
                "expected_pr": s.pr.number if s.pr else s.initial_pr_number,
            },
            agent=s.coder_agent(),
            workdir=s.input.checkout,
        ),
        then=lambda s, r: {
            "pr": models.PullRequest.ship(s.pr, number=r.pr_number, head_sha=r.head_sha)
        },
    )
    g.step("check_readiness", _step_check_readiness)
    g.agent(
        "merge_gate",
        role=ORCHESTRATOR,
        expects=tools.MergeGateDecision,
        ask=lambda s: Turn(
            session=f"orchestrator:merge_gate:a{len(_pr(s).attempts)}",
            prompt=prompts.merge_gate_prompt(s),
            context={"expected_head": _readiness(s).head_sha},
            agent=_setup(s).orchestrator,
            workdir=s.input.checkout,
        ),
        then=lambda s, r: {
            "pr": _pr(s).update_current(
                verdict=models.MergeGateVerdict(
                    decision=r.decision, head_sha=r.head_sha, reason=r.reason
                )
            )
        },
    )
    g.wait(
        "ask_approval",
        reply=models.MergeApprovalReply,
        ask=_ask_approval,
        then=lambda s, reply: {"pr": _pr(s).update_current(approval=reply)},
    )
    g.agent(
        "repair",
        role=CODER,
        expects=models.VerifiedCommit,
        check="commit",
        ask=lambda s: Turn(
            session="coder",
            prompt=prompts.repair_prompt(s),
            context={
                "checkout": str(s.input.checkout),
                "baseline": s.head,
                "require_new_commit": True,
            },
            agent=s.coder_agent(),
            workdir=s.input.checkout,
        ),
        then=lambda s, r: {
            "revisions": (
                *s.revisions,
                models.CodeRevision(base_sha=_revision(s).head_sha, head_sha=r.head_sha),
            ),
            "retries": models.validated_replace(s.retries, auto_repairs=s.retries.auto_repairs + 1),
        },
    )
    g.wait(
        "escalate",
        reply=models.EscalationReply,
        ask=_ask_escalation,
        then=lambda s, reply: {
            "escalations": [
                models.EscalationRecord(
                    revision=s.revision,
                    reason=prompts.escalation_reason(s),
                    action=reply.action,
                    instructions=reply.instructions,
                )
            ],
            **({"retries": models.RetryCounters()} if reply.action == "retry" else {}),
        },
    )
    g.step("merge_exact_head", _step_merge_exact_head)
    g.step("verify_merge", _step_verify_merge)
    g.step("cleanup", _step_cleanup)

    g.branch(
        "triage_repair",
        _route_repair_triage,
        {"review": "dispatch_code_review", "skip": "ship_pr"},
    )
    g.edge("ship_pr", "check_readiness")
    g.edge("check_readiness", "merge_gate")
    g.branch(
        "merge_gate",
        _route_merge_gate,
        {
            "merge": "merge_exact_head",
            "approve": "ask_approval",
            "repair": "repair",
            "escalate": "escalate",
        },
    )
    g.branch("ask_approval", _route_approval, {"merge": "merge_exact_head", "stop": END})
    g.edge("repair", "triage_repair")
    g.branch("escalate", _route_escalation, {"repair": "repair", "stop": END})
    g.branch(
        "merge_exact_head",
        _route_merge_result,
        {"readiness_changed": "ship_pr", "merged": "verify_merge", "escalate": "escalate"},
    )
    g.edge("verify_merge", "cleanup")
    g.edge("cleanup", END)


def _build_code_and_ship_graph() -> Graph[models.CodeAndShipState]:
    g: Graph[models.CodeAndShipState] = Graph()
    _add_code_stage(g, after_review="ship_pr")
    _add_ship_stage(g)
    return g


def _build_code_graph() -> Graph[models.CodeState]:
    g: Graph[models.CodeState] = Graph()
    _add_code_stage(g, after_review="cleanup_mirror")
    g.step("cleanup_mirror", _step_cleanup_mirror)
    g.edge("cleanup_mirror", END)
    return g


def _build_ship_graph() -> Graph[models.ShipState]:
    g: Graph[models.ShipState] = Graph()
    g.step("resolve_ship_setup", _step_resolve_ship_setup)
    _add_code_review_loop(g, after_review="ship_pr")
    _add_ship_stage(g)
    g.entry("resolve_ship_setup")
    g.branch(
        "resolve_ship_setup",
        _route_ship_entry,
        {"ship": "ship_pr", "check": "check_readiness"},
    )
    return g


CODE_AND_SHIP_WORKFLOW = Workflow(
    name="code-and-ship",
    version="2.0.0",
    input=models.CodeInput,
    state=models.CodeAndShipState,
    graph=_build_code_and_ship_graph(),
)
CODE_WORKFLOW = Workflow(
    name="code",
    version="2.0.0",
    input=models.CodeInput,
    state=models.CodeState,
    graph=_build_code_graph(),
)
SHIP_WORKFLOW = Workflow(
    name="ship",
    version="2.0.0",
    input=models.ShipInput,
    state=models.ShipState,
    graph=_build_ship_graph(),
)
