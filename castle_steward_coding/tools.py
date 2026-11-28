"""Agent-facing MCP submission tool schemas and validators for coding workflows.

Every class docstring and field `description` in this module is surfaced directly to agents
as the MCP tool specification (`description` and JSON Schema `properties`).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationInfo,
    model_validator,
)

from castle_steward_coding.routing import Difficulty

MergeGateOutcome = Literal["merge", "request_human_approval", "request_changes", "escalate"]
_NonBlankStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def _require_ctx(info: ValidationInfo, key: str) -> Any:
    if not isinstance(info.context, dict) or key not in info.context or info.context[key] is None:
        raise ValueError(f"Validation context missing required key {key!r}")
    return info.context[key]


class TaskAssessment(BaseModel):
    """Submit the task's difficulty and whether it needs a written plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    difficulty: Difficulty = Field(description="How hard the task is to get right.")
    needs_plan: bool = Field(description="Whether to write and review a plan before writing code.")
    rationale: _NonBlankStr = Field(
        description="Short explanation for your difficulty and needs_plan choices (1-2 sentences)."
    )


class WrittenPlan(BaseModel):
    """Submit the implementation plan as Markdown text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan: _NonBlankStr = Field(
        description="The full Markdown text of the plan (pass the text itself, not a file path)."
    )


class ReviewFindings(BaseModel):
    """Submit your review verdict and any problems to fix."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    approved: bool = Field(
        description=(
            "Set to true if this is ready as-is with no problems to fix, "
            "or false if changes are needed."
        )
    )
    summary: _NonBlankStr = Field(description="Short summary of your review (1-2 sentences).")
    findings: tuple[str, ...] = Field(
        default=(),
        description=(
            "List of specific problems to fix, one per item "
            "(include file and line when reviewing code). Leave empty when approved is true."
        ),
    )


class MergeGateDecision(BaseModel):
    """Submit the merge gate decision for the pull request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: MergeGateOutcome = Field(description="Next action for the pull request.")
    head_sha: str = Field(
        description="Full 40-character Git commit SHA of the pull request head you evaluated."
    )
    reason: _NonBlankStr = Field(
        description=(
            "Clear explanation of what needs to be fixed, what risk needs human sign-off, "
            "or why human input is needed."
        )
    )

    @model_validator(mode="after")
    def _check_decision(self, info: ValidationInfo) -> MergeGateDecision:
        expected_head = str(_require_ctx(info, "expected_head"))
        if self.head_sha != expected_head:
            raise ValueError(
                f"head_sha {self.head_sha!r} must be the pull request head {expected_head!r}"
            )
        return self


class RepairTriage(BaseModel):
    """Submit whether the repair commit needs another code review."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    needs_code_review: bool = Field(
        description=(
            "Set to true if the repair could change behavior or weaken tests "
            "(such as logic, interface, dependency, config, or merge-conflict changes). "
            "Set to false for safe mechanical fixes like formatting, lint, types, or docs."
        )
    )
    head_sha: str = Field(
        description="Full 40-character Git commit SHA of the repair commit you evaluated."
    )
    reason: _NonBlankStr = Field(
        description="Short explanation for your needs_code_review choice (1-2 sentences)."
    )

    @model_validator(mode="after")
    def _check_triage(self, info: ValidationInfo) -> RepairTriage:
        expected_head = str(_require_ctx(info, "expected_head"))
        if self.head_sha != expected_head:
            raise ValueError(
                f"head_sha {self.head_sha!r} must be the repair commit {expected_head!r}"
            )
        return self
