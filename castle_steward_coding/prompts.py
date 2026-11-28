"""Prompt assembly logic rendering templates from `prompt_templates.py`."""

from __future__ import annotations

import re
import textwrap
from typing import Any

from castle_steward_coding import prompt_templates as tpl
from castle_steward_coding.models import (
    CodeState,
    PullRequest,
    ReviewItem,
    ShipStateBase,
    WorkflowState,
)

_STANDALONE_FIELD_RE = re.compile(r"\{(\w+)\}")


def _render(template: str, **fields: Any) -> str:
    """Dedent template, omit empty standalone placeholder lines, and format values verbatim."""
    lines = textwrap.dedent(template).strip().splitlines()
    kept = [
        line
        for line in lines
        if not (
            (match := _STANDALONE_FIELD_RE.fullmatch(line.strip()))
            and match.group(1) in fields
            and not str(fields[match.group(1)])
        )
    ]
    cleaned_template = re.sub(r"\n{3,}", "\n\n", "\n".join(kept))
    return cleaned_template.format(**fields)


def _format_reviews(state: WorkflowState, *, stage: str, revision: int | None = None) -> str:
    items = [
        r
        for r in state.reviews
        if r.stage == stage and (revision is None or r.revision == revision)
    ]
    if not items:
        return tpl.NO_REVIEWS
    lines: list[str] = []
    for r in items:
        verdict = (
            tpl.REVIEW_VERDICT_APPROVED if r.approved else tpl.REVIEW_VERDICT_CHANGES_REQUESTED
        )
        if r.head_sha:
            lines.append(
                tpl.REVIEW_LINE_CODE.format(
                    reviewer=r.reviewer,
                    verdict=verdict,
                    head_sha=r.head_sha,
                    summary=r.summary,
                )
            )
        else:
            lines.append(
                tpl.REVIEW_LINE_PLAN.format(
                    reviewer=r.reviewer,
                    verdict=verdict,
                    summary=r.summary,
                )
            )
        for finding in r.findings:
            lines.append(tpl.REVIEW_FINDING_LINE.format(finding=finding))
    return "\n".join(lines)


def _latest_code_review_revision(state: WorkflowState) -> int | None:
    code_revs = [r.revision for r in state.reviews if r.stage == "code"]
    if not code_revs:
        return None
    return state.revision if state.revision in code_revs else max(code_revs)


def _format_review_context_note(
    state: WorkflowState, pr_head_sha: str, reviewed_revision: int | None
) -> str:
    rev = state.current_revision
    if rev is not None and rev.triage is not None and not rev.triage.needs_code_review:
        return tpl.MERGE_CONTEXT_TRIAGE_SKIPPED.format(
            pr_head_sha=pr_head_sha,
            reason=rev.triage.reason,
        )
    if reviewed_revision is None:
        return ""
    rev_reviews = [
        r for r in state.reviews if r.stage == "code" and r.revision == reviewed_revision
    ]
    reviewed_shas = sorted({r.head_sha for r in rev_reviews if r.head_sha})
    if pr_head_sha and reviewed_shas and reviewed_shas != [pr_head_sha]:
        listed = ", ".join(f"`{sha}`" for sha in reviewed_shas)
        return tpl.MERGE_CONTEXT_POST_REVIEW_UPDATES.format(
            reviewed_shas=listed,
            pr_head_sha=pr_head_sha,
        )
    return ""


def _render_pr_section(pr: PullRequest, pr_head_sha: str) -> str:
    readiness = pr.current.readiness
    url_suffix = (
        tpl.MERGE_CONTEXT_PR_URL.format(url=readiness.url) if readiness and readiness.url else ""
    )
    readiness_lines: list[str] = []
    if readiness is not None:
        if readiness.ready:
            status = tpl.MERGE_CONTEXT_READINESS_READY
        elif readiness.pending:
            status = tpl.MERGE_CONTEXT_READINESS_PENDING
        else:
            status = tpl.MERGE_CONTEXT_READINESS_NOT_READY
        readiness_lines.append(tpl.MERGE_CONTEXT_READINESS_LINE.format(status=status))
        if readiness.reasons:
            readiness_lines.append(tpl.MERGE_CONTEXT_REASONS_HEADING)
            readiness_lines.extend(
                tpl.MERGE_CONTEXT_REASON_LINE.format(reason=reason) for reason in readiness.reasons
            )
    return _render(
        tpl.MERGE_CONTEXT_PR_SECTION,
        pr_number=pr.number,
        url_suffix=url_suffix,
        head_sha=pr_head_sha,
        readiness_lines="\n".join(readiness_lines),
    )


def _render_escalations_section(state: ShipStateBase) -> str:
    if not state.escalations:
        return ""
    entries: list[str] = []
    for e in state.escalations:
        if e.instructions is not None:
            entries.append(
                tpl.MERGE_CONTEXT_ESCALATION_LINE_WITH_INSTRUCTIONS.format(
                    revision=e.revision,
                    action=e.action,
                    instructions=e.instructions,
                    reason=e.reason,
                )
            )
        else:
            entries.append(
                tpl.MERGE_CONTEXT_ESCALATION_LINE.format(
                    revision=e.revision,
                    action=e.action,
                    reason=e.reason,
                )
            )
    return _render(tpl.MERGE_CONTEXT_ESCALATIONS_SECTION, entries="\n".join(entries))


def _render_prior_verdicts_section(pr: PullRequest) -> str:
    entries = [
        tpl.MERGE_CONTEXT_PRIOR_VERDICT_LINE.format(
            attempt=i,
            head_sha=a.head_sha,
            decision=a.verdict.decision,
            reason=a.verdict.reason,
        )
        for i, a in enumerate(pr.attempts[:-1], start=1)
        if a.verdict is not None
    ]
    return (
        _render(tpl.MERGE_CONTEXT_PRIOR_VERDICTS_SECTION, entries="\n".join(entries))
        if entries
        else ""
    )


def _render_reviews_section(state: WorkflowState, pr_head_sha: str) -> str:
    reviewed_rev = _latest_code_review_revision(state)
    heading = (
        tpl.MERGE_CONTEXT_REVIEWS_HEADING_PRIOR_REV
        if reviewed_rev is not None and reviewed_rev != state.revision
        else tpl.MERGE_CONTEXT_REVIEWS_HEADING
    )
    reviews = (
        _format_reviews(state, stage="code", revision=reviewed_rev)
        if reviewed_rev is not None
        else tpl.NO_REVIEWS
    )
    return _render(
        tpl.MERGE_CONTEXT_REVIEWS_SECTION,
        heading=heading,
        review_note=_format_review_context_note(state, pr_head_sha, reviewed_rev),
        reviews=reviews,
    )


def render_merge_gate_context(state: ShipStateBase, pr: PullRequest) -> str:
    """Render the per-attempt Markdown context artifact read by `merge_gate`."""
    readiness = pr.current.readiness
    pr_head_sha = readiness.head_sha if readiness else pr.current.head_sha
    sections = [
        tpl.MERGE_CONTEXT_HEADER.format(pr_number=pr.number, attempt=len(pr.attempts)),
        _render_pr_section(pr, pr_head_sha),
        _render_escalations_section(state),
        _render_prior_verdicts_section(pr),
        _render_reviews_section(state, pr_head_sha),
    ]
    return "\n\n".join(section for section in sections if section) + "\n"


def _format_human_instructions(raw: str | None) -> str:
    return tpl.HUMAN_INSTRUCTIONS_SECTION.format(instructions=raw) if raw is not None else ""


def task_summary(state: WorkflowState) -> str:
    return state.task_text or tpl.SHIP_DEFAULT_TASK


def coordinate_prompt(state: CodeState) -> str:
    return _render(tpl.COORDINATE, task=state.input.task)


def write_plan_prompt(state: CodeState) -> str:
    return _render(tpl.WRITE_PLAN, task=state.input.task)


def review_plan_prompt(item: ReviewItem) -> str:
    assert item.plan_ref is not None
    return _render(
        tpl.REVIEW_PLAN,
        task=item.task,
        plan_ref=item.plan_ref,
    )


def revise_plan_prompt(state: CodeState) -> str:
    plan = state.latest_plan
    assert plan is not None and plan.ref is not None
    return _render(
        tpl.REVISE_PLAN,
        plan_ref=plan.ref,
        reviews=_format_reviews(state, stage="plan"),
    )


def implement_prompt(state: CodeState) -> str:
    plan_ref = state.latest_plan.ref if state.latest_plan else None
    if plan_ref:
        return _render(
            tpl.IMPLEMENT_WITH_PLAN,
            plan_ref=plan_ref,
            implement_guidance=tpl.IMPLEMENT_GUIDANCE,
            head=state.head,
            task=state.input.task,
        )
    return _render(
        tpl.IMPLEMENT,
        implement_guidance=tpl.IMPLEMENT_GUIDANCE,
        head=state.head,
        task=state.input.task,
    )


def review_code_prompt(item: ReviewItem) -> str:
    if item.plan_ref:
        return _render(
            tpl.REVIEW_CODE_WITH_PLAN,
            baseline=item.baseline,
            reviewed_head=item.reviewed_head,
            review_focus=tpl.CODE_REVIEW_FOCUS,
            task=item.task,
            plan_ref=item.plan_ref,
        )
    return _render(
        tpl.REVIEW_CODE,
        baseline=item.baseline,
        reviewed_head=item.reviewed_head,
        review_focus=tpl.CODE_REVIEW_FOCUS,
        task=item.task,
    )


def address_review_prompt(state: WorkflowState) -> str:
    return _render(
        tpl.ADDRESS_REVIEW,
        head=state.head,
        reviews=_format_reviews(state, stage="code", revision=state.revision),
    )


def ship_pr_prompt(state: ShipStateBase) -> str:
    if state.pr is None:
        if state.initial_pr_number is not None:
            return _render(
                tpl.SHIP_PR_UPDATE,
                head=state.head,
                pr_number=state.initial_pr_number,
                readiness_note="",
            )
        return _render(tpl.SHIP_PR_INITIAL, head=state.head)
    readiness_note = ""
    if (
        (last := state.pr.current).merge is not None
        and last.merge.status == "readiness_changed"
        and last.head_sha == state.head
    ):
        readiness_note = tpl.READINESS_CHANGED_NOTE.format(reasons="; ".join(last.merge.reasons))
    return _render(
        tpl.SHIP_PR_UPDATE,
        head=state.head,
        pr_number=state.pr.number,
        readiness_note=readiness_note,
    )


def merge_gate_prompt(state: ShipStateBase) -> str:
    assert state.pr is not None and state.pr.current.context_ref is not None
    attempt = state.pr.current
    pr_head_sha = attempt.readiness.head_sha if attempt.readiness else attempt.head_sha
    return _render(
        tpl.MERGE_GATE,
        pr_number=state.pr.number,
        pr_head_sha=pr_head_sha,
        task=task_summary(state),
        context_ref=attempt.context_ref,
    )


def format_repair_reason(state: ShipStateBase, *, default: str = "") -> str:
    cause = state.repair_cause()
    if cause is None:
        return default
    joined = "; ".join(cause.details)
    if cause.kind == "readiness_changed":
        return tpl.REPAIR_CAUSE_READINESS_CHANGED.format(reasons=joined)
    if cause.kind == "pr_not_ready":
        return tpl.REPAIR_CAUSE_PR_NOT_READY.format(reasons=joined)
    return joined or default


def escalation_reason(state: ShipStateBase) -> str:
    return format_repair_reason(state, default=tpl.DEFAULT_ESCALATION_REASON)


def repair_prompt(state: ShipStateBase) -> str:
    assert state.pr is not None
    return _render(
        tpl.REPAIR,
        pr_number=state.pr.number,
        head=state.head,
        reason=format_repair_reason(state, default=tpl.DEFAULT_REPAIR_REASON),
        human_instructions=_format_human_instructions(state.repair_instructions()),
    )


def triage_repair_prompt(state: ShipStateBase) -> str:
    return _render(
        tpl.TRIAGE_REPAIR,
        diff_range=f"{state.diff_base}..{state.head}",
        task=task_summary(state),
        reason=format_repair_reason(state, default=tpl.EMPTY_REASON),
        human_instructions=_format_human_instructions(state.triage_instructions()),
        head=state.head,
    )
