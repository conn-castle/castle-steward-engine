"""Pure agent-facing prompt templates and text snippets for coding workflows.

This module contains only string constants (no functions, imports, or control flow) so all
agent-facing prose can be reviewed in one place.
"""

# --- Shared prompt blocks ---

IMPLEMENT_GUIDANCE = (
    "Reuse existing code and established patterns where appropriate. Before committing:\n"
    "- Simplify the changes without altering required behavior or broadening scope: remove unnecessary complexity, duplication, dead code, premature abstractions, and tests that cannot detect a realistic product defect.\n"
    "- Run appropriate checks for your final changes and fix any failures; reuse valid results when the tree has not changed."
)

CODE_REVIEW_FOCUS = (
    "In addition to a general review, pay particular attention to:\n"
    "- unmet requirements, out-of-scope changes, or broken public/internal contracts\n"
    "- correctness defects, missed edge cases or error paths, and regressions across affected integration paths\n"
    "- security, safety, reliability, concurrency, data-integrity, or rollback hazards\n"
    "- unnecessary complexity, duplication, dead code, or duplicate state\n"
    "- anything that doesn't match the code around it\n"
    "- missing behavioral tests for key paths and failures, or tautological tests that would pass despite a realistic defect"
)


SHIP_DEFAULT_TASK = "Ship the existing branch; judge the PR on its description and diff."

# --- Full prompt templates ---

# Placeholders:
#   task = "Add --dry-run flag to the deploy CLI"
COORDINATE = """
You are the workflow orchestrator, the first agent in a multi-step coding workflow. Your sole job is to classify the following task's `difficulty` and decide whether it `needs_plan` before implementation. Do not edit, create, or delete any files in the repository.

`difficulty` measures how hard the change is to get right — independent of diff size, file count, or `needs_plan`. It determines coder capability and reviewer count:
- `low` — Unambiguous requirements and an obvious mechanism following established local patterns, with no subtle invariant, public contract, persistence, concurrency, or security impact, and straightforward rollback. Realistic mistakes surface immediately in local checks or basic tests.
- `medium` — Non-trivial control flow, cross-module coordination, interface updates, or edge cases and failure paths that require deliberate handling and targeted behavioral tests, within well-understood boundaries and straightforward rollback.
- `high` — High-risk or subtly coupled work where defects can escape basic tests or cause serious harm: concurrency, ordering, or lifecycle hazards; security, auth, or trust boundaries; persistence invariants, schema migrations, or irreversible external effects; complex state machines or cross-cutting architectural changes; or non-obvious root causes and interactions.

`needs_plan` is independent of `difficulty` and adds a written plan and independent plan review before any code is written. Planning adds extra turns and latency; choose `true` only when the task justifies that cost:
- `true` — A reviewed design and verification checklist are needed before coding: the task leaves substantive architectural or design tradeoffs open where a wrong direction is costly to undo, or coordinates enough coupled changes that an upfront plan prevents missed requirements or broken invariants.
- `false` — Implement directly without a separate planning phase. This includes, but is not limited to, tasks where the steps are obvious, already specified, or where the act of planning requires implementing first (such as debugging).

Task:
{task}

Call `submit_assessment` with your `difficulty` and `needs_plan` determination, then confirm with `check_done` before ending your turn.
"""

# Placeholders:
#   task = "Add --dry-run flag to the deploy CLI"
WRITE_PLAN = """
Write an implementation plan for the task below. Independent reviewers will check the plan before implementation begins.

Reuse existing code and established patterns where appropriate, and resolve substantive design tradeoffs in the plan. Include the codebase context needed to execute the plan, the concrete changes to make, and a verification checklist covering every requirement and the behavioral tests to run.

Do not edit, create, or delete any files in the repository; using `submit_plan` is all that is required.

Task:
{task}

Call `submit_plan`, then confirm with `check_done` before ending your turn.
"""

# Placeholders:
#   task     = "Add --dry-run flag to the deploy CLI"
#   plan_ref = ".agent-layer/cache/steward-engine/<job-id>/plan-v1.md"
REVIEW_PLAN = """
Review the implementation plan in the plan file below against the task and the repository's existing code.

In addition to a general review, pay particular attention to:
- unmet task requirements, out-of-scope additions, or incorrect assumptions about the codebase
- flawed strategy that treats symptoms instead of root causes or misses existing code and patterns to reuse
- correctness, edge-case, error-handling, security, reliability, data-safety, or reversibility risks
- unnecessary complexity, duplication, or premature abstractions
- anything in the plan that doesn't match the code around it
- verification gaps, or planned tests that are tautological instead of testing real behavior

Report only concrete, actionable in-scope problems in `findings`; leave `findings` empty when the plan is ready to implement. Do not edit, create, or delete any files in the repository.

Task:
{task}

Plan file:
{plan_ref}

Call `submit_review`, then confirm with `check_done` before ending your turn.
"""

# Placeholders:
#   plan_ref = ".agent-layer/cache/steward-engine/<job-id>/plan-v1.md"
#   reviews  = "- opus_high [CHANGES REQUESTED]: Missing error path\n  * Handle invalid flag combinations"
REVISE_PLAN = """
Read your current plan in the plan file below and revise it to address the review findings below.

Do not edit, create, or delete any files in the repository; using `submit_plan` is all that is required.

Plan file:
{plan_ref}

Review findings:
{reviews}

Call `submit_plan` with the full updated plan, then confirm with `check_done` before ending your turn.
"""

# Placeholders:
#   implement_guidance = IMPLEMENT_GUIDANCE
#   head               = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
#   task               = "Add --dry-run flag to the deploy CLI"
IMPLEMENT = """
Implement the task below.

{implement_guidance}
- Check completion against every requirement in the task; passing tests are supporting evidence, not proof of completion.

Finish with a clean working tree (no uncommitted or untracked files) and at least one new commit descending from `{head}`. Do not amend, rebase, or rewind.

Task:
{task}

Commit your changes locally (do not push), then call `check_done` before ending your turn.
"""

# Placeholders:
#   implement_guidance = IMPLEMENT_GUIDANCE
#   head               = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
#   task               = "Add --dry-run flag to the deploy CLI"
#   plan_ref           = ".agent-layer/cache/steward-engine/<job-id>/plan-v1.md"
IMPLEMENT_WITH_PLAN = """
Implement the task below, following the plan file below.

{implement_guidance}
- Check completion against both the task and the plan file; passing tests are supporting evidence, not proof of completion.

Finish with a clean working tree (no uncommitted or untracked files) and at least one new commit descending from `{head}`. Do not amend, rebase, or rewind.

Task:
{task}

Plan file:
{plan_ref}

Commit your changes locally (do not push), then call `check_done` before ending your turn.
"""

# Placeholders:
#   baseline      = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
#   reviewed_head = "f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5"
#   review_focus  = CODE_REVIEW_FOCUS
#   task          = "Add --dry-run flag to the deploy CLI"
REVIEW_CODE = """
Review the changes in `git diff {baseline}..{reviewed_head}` against the task below.

{review_focus}

Report only concrete, actionable in-scope defects in `findings`; leave `findings` empty when the changes are ready to ship. Do not edit, create, or delete any files in the repository.

Task:
{task}

Call `submit_review`, then confirm with `check_done` before ending your turn.
"""

# Placeholders:
#   baseline      = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
#   reviewed_head = "f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5"
#   review_focus  = CODE_REVIEW_FOCUS
#   task          = "Add --dry-run flag to the deploy CLI"
#   plan_ref      = ".agent-layer/cache/steward-engine/<job-id>/plan-v2.md"
REVIEW_CODE_WITH_PLAN = """
Review the changes in `git diff {baseline}..{reviewed_head}` against the task and plan file below.

{review_focus}

Report only concrete, actionable in-scope defects in `findings`; leave `findings` empty when the changes are ready to ship. Do not edit, create, or delete any files in the repository.

Task:
{task}

Plan file:
{plan_ref}

Call `submit_review`, then confirm with `check_done` before ending your turn.
"""

# Placeholders:
#   head    = "f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5"
#   reviews = "- opus_high [CHANGES REQUESTED] (head f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5): Unvalidated flag\n  * Reject mutually exclusive flags"
ADDRESS_REVIEW = """
Evaluate the code review findings below against the task and current codebase, and fix every valid in-scope problem.

- Fix findings that identify a material in-scope correctness, security, safety, reliability, maintainability, verification, or contract-completion defect.
- You may defer worthwhile new features, pre-existing issues, or unrelated refactors, but never defer a defect introduced by your changes.
- You may disagree with unsupported, harmful, or non-beneficial findings. Check any finding tied to an explicit requirement against the task before rejecting it, and never dismiss a valid finding to avoid work.
- If you modify any files, run appropriate checks for your final changes and fix any failures before committing; reuse valid results when the tree has not changed.

Finish with a clean working tree (no uncommitted or untracked files), either still at `{head}` (if no changes were needed) or with new commits descending from `{head}`. Do not amend, rebase, or rewind.

Review findings:
{reviews}

Commit any changes locally (do not push), then call `check_done` before ending your turn.
"""

# Placeholders:
#   head = "f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5"
SHIP_PR_INITIAL = """
Push commit `{head}` and open a non-draft pull request for it with a clear title and summary derived from the changes. If this branch already has an open pull request, update that one instead (and mark it ready for review if it is a draft).

Push `{head}` as-is with a clean working tree. Do not edit, create, or delete any files in the repository or change `HEAD`.

Call `check_done` before ending your turn.
"""

# Placeholders:
#   head           = "f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5"
#   pr_number      = 42
#   readiness_note = "Previous merge aborted because PR readiness changed: PR mergeStateStatus is DIRTY. Do not fix readiness in this step; later steps handle it."  # optional, blank when unset
SHIP_PR_UPDATE = """
Push commit `{head}` to existing PR #{pr_number}. Do not open a new pull request; ensure PR #{pr_number} is open and non-draft.

Push `{head}` as-is with a clean working tree. Do not edit, create, or delete any files in the repository or change `HEAD`.

{readiness_note}

Call `check_done` before ending your turn.
"""

# Placeholders:
#   pr_number   = 42
#   pr_head_sha = "f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5"
#   task        = "Add --dry-run flag to the deploy CLI"
#   context_ref = ".agent-layer/cache/steward-engine/<job-id>/merge-context-a1.md"
MERGE_GATE = """
Decide the next action for PR #{pr_number} at commit `{pr_head_sha}`. Inspect live PR checks and mergeability on GitHub via `gh`, and read the context file below for local review findings, prior merge gate verdicts, and any human escalation instructions (human instructions outrank earlier verdicts and reviews). Do not edit, create, or delete any files in the repository.

- `merge` — The PR is mergeable at its exact current head, every required check and repository gate is green, any repo-specific PR policy criteria are met, the task and any human instructions are satisfied, and no valid in-scope review finding remains unaddressed.
- `request_human_approval` — Technically ready to merge, but involves judgment or risk the agent cannot responsibly approve on the user's behalf (such as architectural choices with lasting consequences, subjective end-user changes, security/data/operational risk, or irreversible actions).
- `request_changes` — A concrete, recoverable defect blocks merging that the coder can fix in a follow-up commit (failing required check, merge conflict, unmet requirement, policy gate violation, or unaddressed valid review finding). Execution difficulty alone is not a reason to escalate.
- `escalate` — Another incremental commit will not resolve the problem or requires human judgment/authority: the approach is unsound, requirements or tradeoffs need human input, an external or human-only gate blocks merging, or repeated repairs are not converging.

Task:
{task}

Context file:
{context_ref}

Call `submit_gate_decision` for commit `{pr_head_sha}`, then confirm with `check_done` before ending your turn.
"""

# Placeholders:
#   pr_number          = 42
#   head               = "f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5"
#   reason             = "PR not ready: check 'ci' concluded FAILURE"
#   human_instructions = "Human instructions:\nFix the timeout in test_deploy_cli instead of skipping the test."  # optional, blank when unset
REPAIR = """
Fix what is blocking PR #{pr_number} from merging.

- Diagnose and fix the root cause of the problem below rather than working around it.
- If a CI or GitHub check failed, inspect its logs on GitHub before choosing a fix.
- If the branch has merge conflicts, fetch and merge the PR's base branch rather than rebasing so `{head}` stays an ancestor.
- Run appropriate checks for your final changes and fix any failures before committing; reuse valid results when the tree has not changed.

Finish with a clean working tree (no uncommitted or untracked files) and at least one new commit descending from `{head}`. Do not amend, rebase, or rewind.

Blocking issue:
{reason}

{human_instructions}

Commit your changes locally (do not push), then call `check_done` before ending your turn.
"""

# Placeholders:
#   diff_range         = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2..f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5"
#   task               = "Add --dry-run flag to the deploy CLI"
#   reason             = "PR not ready: check 'ci' concluded FAILURE"
#   human_instructions = "Human instructions:\nFix the timeout in test_deploy_cli instead of skipping the test."  # optional, blank when unset
#   head               = "f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5"
TRIAGE_REPAIR = """
Decide whether the repair in `git diff {diff_range}` requires another code review pass before shipping. Do not edit, create, or delete any files in the repository.

- `true` — The repair could change behavior or weaken tests (such as logic, interface, dependency, config, or merge-conflict changes).
- `false` — The repair is a safe mechanical fix that cannot change behavior or weaken tests (such as formatting, lint, types, docs, or straightforward test fixes).

Task:
{task}

Blocking issue:
{reason}

{human_instructions}

Call `submit_repair_triage` for commit `{head}`, then confirm with `check_done` before ending your turn.
"""


# --- Conditional and formatted text snippets ---

READINESS_CHANGED_NOTE = (
    "Previous merge aborted because PR readiness changed: {reasons}. "
    "Do not fix readiness in this step; later steps handle it."
)
REPAIR_CAUSE_READINESS_CHANGED = "PR readiness changed: {reasons}"
REPAIR_CAUSE_PR_NOT_READY = "PR not ready: {reasons}"
HUMAN_INSTRUCTIONS_SECTION = "Human instructions:\n{instructions}"
DEFAULT_REPAIR_REASON = "(no reason recorded)"
DEFAULT_ESCALATION_REASON = "needs input"
EMPTY_REASON = "(none)"

NO_REVIEWS = "(none)"
REVIEW_VERDICT_APPROVED = "APPROVED"
REVIEW_VERDICT_CHANGES_REQUESTED = "CHANGES REQUESTED"
REVIEW_LINE_PLAN = "- {reviewer} [{verdict}]: {summary}"
REVIEW_LINE_CODE = "- {reviewer} [{verdict}] (head {head_sha}): {summary}"
REVIEW_FINDING_LINE = "  * {finding}"


# --- Merge gate context artifact ---

MERGE_CONTEXT_HEADER = "# Merge Gate Context — PR #{pr_number} (Attempt {attempt})"
MERGE_CONTEXT_PR_SECTION = """
## Pull Request
- PR: #{pr_number}{url_suffix}
- Head commit: `{head_sha}`
{readiness_lines}
"""
MERGE_CONTEXT_PR_URL = " ({url})"
MERGE_CONTEXT_READINESS_READY = "Ready to merge"
MERGE_CONTEXT_READINESS_PENDING = "Pending (checks or mergeability still in progress)"
MERGE_CONTEXT_READINESS_NOT_READY = "Not ready"
MERGE_CONTEXT_READINESS_LINE = "- Readiness snapshot: {status}"
MERGE_CONTEXT_REASONS_HEADING = "- Blocking or pending reasons:"
MERGE_CONTEXT_REASON_LINE = "  - {reason}"

MERGE_CONTEXT_ESCALATIONS_SECTION = """
## Human Escalation History
Human instructions outrank earlier verdicts and reviews.
{entries}
"""
MERGE_CONTEXT_ESCALATION_LINE = "- Revision {revision} ({action}): {reason}"
MERGE_CONTEXT_ESCALATION_LINE_WITH_INSTRUCTIONS = (
    "- Revision {revision} ({action}): {instructions} (original issue: {reason})"
)

MERGE_CONTEXT_PRIOR_VERDICTS_SECTION = """
## Prior Merge Gate Verdicts
{entries}
"""
MERGE_CONTEXT_PRIOR_VERDICT_LINE = "- Attempt {attempt} (head {head_sha}): {decision} — {reason}"

MERGE_CONTEXT_REVIEWS_SECTION = """
{heading}
{review_note}
{reviews}
"""
MERGE_CONTEXT_REVIEWS_HEADING = "## Latest Code Reviews"
MERGE_CONTEXT_REVIEWS_HEADING_PRIOR_REV = "## Latest Code Reviews (from before the repair)"
MERGE_CONTEXT_TRIAGE_SKIPPED = (
    "Review context: repair triage skipped re-review of commit `{pr_head_sha}`: {reason}"
)
MERGE_CONTEXT_POST_REVIEW_UPDATES = (
    "Review context: the reviews below evaluated commit {reviewed_shas}, before post-review "
    "updates advanced PR head to `{pr_head_sha}`. Check whether their findings were addressed."
)
