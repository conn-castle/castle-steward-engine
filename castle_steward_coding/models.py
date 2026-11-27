"""Internal config, input, durable state, and human wait-reply models for `coding`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from castle_steward_coding.github import MergeMethod, MergeResult, PrReadiness
from castle_steward_coding.routing import AgentProfiles, Assignment, Difficulty, RoutingPolicy
from castle_steward_coding.tools import MergeGateOutcome, ReviewFindings
from castle_steward_engine import AgentSpec, AppendList

_NonBlankStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_STATE_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")


def validated_replace[M: BaseModel](model: M, **changes: Any) -> M:
    """Return a revalidated copy of a frozen Pydantic model with updated fields."""
    return type(model).model_validate({**dict(model), **changes})


# --- Configuration & Input ---


class CodingConfig(BaseModel):
    """Plain frozen runtime configuration persisted inside workflow checkpoints."""

    model_config = _STATE_MODEL_CONFIG

    orchestrator: AgentSpec = Field(default_factory=lambda: AgentProfiles().neon_high)
    routing: RoutingPolicy = Field(default_factory=RoutingPolicy)
    artifact_mirror_dir: Path = Path(".agent-layer/cache/steward-engine")
    allow_unattended_merge: bool = False
    max_auto_repairs: int = Field(default=1, ge=0)
    max_readiness_retries: int = Field(default=3, ge=0)
    merge_method: MergeMethod = "squash"
    readiness_poll_seconds: float = Field(default=5.0, ge=0.0)
    readiness_timeout_seconds: float = Field(default=1800.0, ge=0.0)

    @field_validator("artifact_mirror_dir")
    @classmethod
    def _validate_artifact_mirror_dir(cls, value: Path) -> Path:
        if value.is_absolute():
            raise ValueError(f"artifact_mirror_dir must be a relative path, got {str(value)!r}")
        if not value.parts:
            raise ValueError(f"artifact_mirror_dir must not be empty or '.', got {str(value)!r}")
        if ".." in value.parts:
            raise ValueError(
                f"artifact_mirror_dir must not contain '..' segments, got {str(value)!r}"
            )
        return value


class CodingSettings(BaseSettings, CodingConfig):
    """Environment-backed loader (`STEW_ENGINE_CODING_*`) for default `CodingConfig`.

    Never stored in `CodingInput` or `CodingState`; call `to_config()` once in
    `resolve_setup` so resumed jobs keep their checkpointed configuration.
    """

    model_config = SettingsConfigDict(env_prefix="STEW_ENGINE_CODING_", extra="forbid", frozen=True)

    def to_config(self) -> CodingConfig:
        """Convert loaded settings into a plain frozen `CodingConfig` model."""
        return CodingConfig.model_validate(self.model_dump())


class CodingInput(BaseModel):
    """Validated input accepted when starting a `coding` job."""

    model_config = _STATE_MODEL_CONFIG

    task: str
    checkout: Path
    orchestrator: AgentSpec | None = None
    config: CodingConfig | None = None

    @field_validator("checkout")
    @classmethod
    def _require_absolute_checkout(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError(f"checkout must be an absolute path, got {str(value)!r}")
        return value.resolve()


# --- State & domain records ---


class RunSetup(BaseModel):
    """Resolved runtime configuration and orchestrator agent captured once at job start."""

    model_config = _STATE_MODEL_CONFIG

    config: CodingConfig  # Resolved runtime config snapshot (input.config or STEW_ENGINE_CODING_*)
    orchestrator: AgentSpec  # Resolved agent for coordinate, triage_repair, and merge_gate


class TaskClassification(BaseModel):
    """Orchestrator classification: difficulty level, plan requirement, and rationale."""

    model_config = _STATE_MODEL_CONFIG

    difficulty: Difficulty  # Classified task difficulty (low, medium, or high)
    needs_plan: bool  # Whether to run the written plan phase before implementing
    rationale: _NonBlankStr  # Concise justification for difficulty and planning choices


class PlanVersion(BaseModel):
    """Submitted implementation plan version and its saved artifact path."""

    model_config = _STATE_MODEL_CONFIG

    text: _NonBlankStr  # Markdown plan text submitted by the coder
    ref: str | None = None  # Workspace-relative mirror path set by save_plan/save_revised_plan


class ReviewRecord(BaseModel):
    """Single reviewer verdict for a plan or code revision."""

    model_config = _STATE_MODEL_CONFIG

    stage: Literal["plan", "code"]  # Review stage (plan or code)
    revision: int  # Code revision index active during review
    reviewer: _NonBlankStr  # Assigned reviewer profile name (e.g., opus_high, grok)
    approved: bool  # Whether the reviewer approved without blocking concerns
    summary: str  # Concise review summary
    findings: tuple[str, ...] = ()  # Concrete actionable review findings
    head_sha: str | None = None  # Reviewed commit SHA (None for plan reviews)


class RepairTriageVerdict(BaseModel):
    """Post-repair triage verdict for a repair commit."""

    model_config = _STATE_MODEL_CONFIG

    head_sha: str  # Exact repair commit SHA that was triaged
    needs_code_review: bool  # Whether the repair commit requires another code review pass
    reason: _NonBlankStr  # Repair triage rationale


class CodeRevision(BaseModel):
    """Git revision lineage (base..head) and optional post-repair triage verdict."""

    model_config = _STATE_MODEL_CONFIG

    base_sha: str  # Pre-revision commit SHA (lineage & diff base for verify/review/triage)
    head_sha: str  # Latest verified commit SHA (== base_sha at initial baseline before implement)
    triage: RepairTriageVerdict | None = None  # Post-repair triage (populated on revisions >= 1)


class MergeGateVerdict(BaseModel):
    """Merge-gate decision recorded for a PR shipment attempt."""

    model_config = _STATE_MODEL_CONFIG

    decision: MergeGateOutcome  # Gate decision for this shipment
    head_sha: str  # Exact PR head commit SHA evaluated by merge_gate
    reason: str  # Rationale for the merge-gate decision


class MergeApprovalReply(BaseModel):
    """Human reply to an `ask_approval` wait, persisted on the PR shipment attempt."""

    model_config = _STATE_MODEL_CONFIG

    approved: bool  # Whether the human approved merging this shipment


class PrAttempt(BaseModel):
    """Chronological record of one PR shipment through readiness, gate, approval, and merge."""

    model_config = _STATE_MODEL_CONFIG

    head_sha: str  # Exact commit SHA pushed by this ship_pr attempt
    readiness: PrReadiness | None = None  # GitHub check/mergeability snapshot for this shipment
    context_ref: str | None = None  # Workspace-relative mirror path for merge-gate context
    verdict: MergeGateVerdict | None = None  # merge_gate decision for this shipment
    approval: MergeApprovalReply | None = None  # Human ask_approval reply for this shipment
    merge: MergeResult | None = None  # merge_exact_head / verify_merge / cleanup outcome


class PullRequest(BaseModel):
    """PR number and chronological shipment attempts (append-only, never overwritten)."""

    model_config = _STATE_MODEL_CONFIG

    number: int = Field(gt=0)  # GitHub PR number opened or updated by ship_pr
    attempts: tuple[PrAttempt, ...] = Field(min_length=1)  # Chronological PR shipment attempts

    @property
    def current(self) -> PrAttempt:
        return self.attempts[-1]

    @classmethod
    def ship(cls, existing: PullRequest | None, *, number: int, head_sha: str) -> PullRequest:
        if existing is not None and existing.number != number:
            raise ValueError(f"Cannot change PR number from #{existing.number} to #{number}")
        attempt = PrAttempt(head_sha=head_sha)
        attempts = (attempt,) if existing is None else (*existing.attempts, attempt)
        return cls(number=number, attempts=attempts)

    def update_current(self, **changes: Any) -> PullRequest:
        updated = validated_replace(self.current, **changes)
        return PullRequest(number=self.number, attempts=(*self.attempts[:-1], updated))


class EscalationRecord(BaseModel):
    """Recorded human escalation pause and reply."""

    model_config = _STATE_MODEL_CONFIG

    revision: int = Field(ge=0)  # Revision index active when escalation paused
    reason: str  # Failure reason presented to the human
    action: Literal["retry", "abort"]  # Human escalation decision
    instructions: str = ""  # Optional live human retry instructions for repair and triage


class RetryCounters(BaseModel):
    """Retry budgets reset to zero on human escalation retry."""

    model_config = _STATE_MODEL_CONFIG

    # Repairs since start or last human retry (includes the retry repair itself)
    auto_repairs: int = Field(default=0, ge=0)
    readiness_checks: int = Field(default=0, ge=0)  # Merge readiness bounces since start or retry


class RepairCause(BaseModel):
    """Structured cause explaining why the latest PR attempt requires repair or escalation."""

    model_config = _STATE_MODEL_CONFIG

    kind: Literal["readiness_changed", "gate_verdict", "pr_not_ready"]
    details: tuple[str, ...] = ()


class VerifiedCommit(BaseModel):
    """Verified clean Git commit SHA captured by the `commit` completion check."""

    model_config = _STATE_MODEL_CONFIG

    head_sha: _NonBlankStr


class VerifiedPr(BaseModel):
    """Verified open non-draft GitHub PR captured by the `pr` completion check."""

    model_config = _STATE_MODEL_CONFIG

    pr_number: int = Field(gt=0)
    head_sha: _NonBlankStr


class CodingState(BaseModel):
    """Durable checkpoint state for `coding` (10 top-level LangGraph channels)."""

    model_config = _STATE_MODEL_CONFIG

    # --- Setup (written once) ---
    input: CodingInput  # Validated job request (engine-required)
    setup: RunSetup | None = None  # Resolved config and orchestrator agent

    # --- Coordination & assignment (each written once) ---
    coordination: TaskClassification | None = None  # Task difficulty, needs_plan, and rationale
    assignment: Assignment | None = None  # Deterministic coder and reviewer assignments

    # --- Append-only domain records (no `latest_*` overwrite loss) ---
    plans: tuple[PlanVersion, ...] = ()  # Plan versions (v1, v2); len(plans) is current version
    revisions: tuple[CodeRevision, ...] = ()  # Git revisions (base..head + optional triage)
    reviews: AppendList[ReviewRecord] = Field(default_factory=list)  # Parallel plan & code reviews
    pr: PullRequest | None = None  # PR number + chronological shipment attempts
    escalations: AppendList[EscalationRecord] = Field(default_factory=list)  # Human escalations

    # --- Retry budgets (reset to 0 on human escalation retry) ---
    retries: RetryCounters = Field(default_factory=RetryCounters)  # Auto-repair & readiness counts

    @property
    def head(self) -> str | None:
        return self.revisions[-1].head_sha if self.revisions else None

    @property
    def diff_base(self) -> str | None:
        return self.revisions[-1].base_sha if self.revisions else None

    @property
    def current_revision(self) -> CodeRevision | None:
        return self.revisions[-1] if self.revisions else None

    @property
    def revision(self) -> int:
        return max(0, len(self.revisions) - 1)

    @property
    def latest_plan(self) -> PlanVersion | None:
        return self.plans[-1] if self.plans else None

    def coder_agent(self) -> AgentSpec:
        assert self.assignment is not None
        return self.assignment.coder

    def repair_cause(self) -> RepairCause | None:
        if self.pr is None:
            return None
        attempt = self.pr.current
        if attempt.merge and attempt.merge.status == "readiness_changed":
            return RepairCause(kind="readiness_changed", details=attempt.merge.reasons)
        if attempt.verdict and attempt.verdict.decision in (
            "request_changes",
            "escalate",
        ):
            return RepairCause(kind="gate_verdict", details=(attempt.verdict.reason,))
        if attempt.readiness and not attempt.readiness.ready:
            return RepairCause(kind="pr_not_ready", details=attempt.readiness.reasons)
        return None

    def _retry_instructions(self, *, for_revision: int) -> str:
        if (
            self.escalations
            and (last := self.escalations[-1]).action == "retry"
            and last.revision == for_revision
        ):
            return last.instructions
        return ""

    # `repair` runs before `repair.then` appends the new revision; `triage_repair` runs after.
    def repair_instructions(self) -> str:
        return self._retry_instructions(for_revision=self.revision)

    def triage_instructions(self) -> str:
        return self._retry_instructions(for_revision=self.revision - 1)

    def reviews_clean(self, stage: Literal["plan", "code"], *, head_sha: str | None = None) -> bool:
        assert self.assignment is not None
        reviewers = {r.profile for r in self.assignment.reviewers}
        if not reviewers or (stage == "code" and head_sha is None):
            return False
        latest = {
            r.reviewer: r
            for r in self.reviews
            if r.stage == stage and (head_sha is None or r.head_sha == head_sha)
        }
        return reviewers.issubset(latest) and all(
            latest[name].approved and not latest[name].findings for name in reviewers
        )


class ReviewItem(BaseModel):
    """Per-reviewer fan-out item passed into `review_plan` and `review_code` nodes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: Literal["plan", "code"]
    reviewer: _NonBlankStr
    agent: AgentSpec
    checkout: Path
    task: str
    revision: int
    plan_ref: str | None = None
    baseline: str | None = None
    reviewed_head: str | None = None

    def to_review(self, findings: ReviewFindings) -> ReviewRecord:
        return ReviewRecord(
            stage=self.stage,
            revision=self.revision,
            reviewer=self.reviewer,
            approved=findings.approved,
            summary=findings.summary,
            findings=findings.findings,
            head_sha=self.reviewed_head if self.stage == "code" else None,
        )


# --- Human wait reply models ---


class EscalationReply(BaseModel):
    """Human reply to an `escalate` wait pause."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["retry", "abort"]
    instructions: str = ""
