"""Git lineage verification (`git -C`) and `gh` CLI PR readiness/merge/cleanup."""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

MergeMethod = Literal["squash", "merge", "rebase"]
SHORT_SHA_LENGTH = 12
_REPO_LOCAL_GIT_ENV = frozenset(
    {
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_PREFIX",
        "GIT_SUPER_PREFIX",
        "GIT_QUARANTINE_PATH",
    }
)
# GitHub GraphQL CheckStatusState (REQUESTED/QUEUED/IN_PROGRESS/WAITING/PENDING vs COMPLETED)
# and MergeStateStatus (BLOCKED/UNSTABLE/UNKNOWN remain transient while checks are still pending).
_PENDING_STATUSES = frozenset({"IN_PROGRESS", "QUEUED", "PENDING", "WAITING", "REQUESTED"})
_PENDING_MERGE_STATES = frozenset({"BLOCKED", "UNSTABLE", "UNKNOWN"})
_STOP_CHECK_INTERVAL = 0.2
_PR_VIEW_FIELDS = (
    "number,url,state,isDraft,mergeStateStatus,headRefOid,headRefName,"
    "baseRefName,statusCheckRollup,mergeCommit"
)


def short_sha(sha: str) -> str:
    """Return a shortened Git commit SHA prefix for human-facing display."""
    return sha[:SHORT_SHA_LENGTH]


class GitHubError(RuntimeError):
    """Raised when a git or `gh` CLI operation fails unexpectedly."""


class PrReadiness(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    pr_number: int
    url: str
    head_sha: str
    head_branch: str
    base_branch: str
    ready: bool
    pending: bool = False
    reasons: tuple[str, ...] = ()


class MergeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    status: Literal["merged", "readiness_changed"]
    pr_number: int
    head_sha: str
    merge_commit_sha: str | None = None
    cleaned_branch: str | None = None
    reasons: tuple[str, ...] = ()


def _scrubbed_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in _REPO_LOCAL_GIT_ENV}


def run_git(checkout: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(checkout), *args],
            env=_scrubbed_env(),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise GitHubError(f"git executable not found while running `git {' '.join(args)}`") from exc
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip() or f"exit code {proc.returncode}"
        raise GitHubError(f"git {' '.join(args)} failed in {checkout}: {detail}")
    return proc


def _run_gh(checkout: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(
            ["gh", *args],
            cwd=str(checkout),
            env=_scrubbed_env(),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise GitHubError(f"gh executable not found while running `gh {' '.join(args)}`") from exc
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip() or f"exit code {proc.returncode}"
        raise GitHubError(f"gh {' '.join(args)} failed in {checkout}: {detail}")
    return proc


def verify_checkout_root(checkout: Path) -> Path:
    """Verify `checkout` exists and is the top-level directory of a Git working tree."""
    resolved = checkout.resolve()
    if not resolved.is_dir():
        raise ValueError(f"Checkout directory does not exist: {resolved}")
    proc = run_git(resolved, "rev-parse", "--show-toplevel", check=False)
    if proc.returncode != 0:
        raise ValueError(f"Checkout is not a Git repository: {resolved}")
    toplevel = Path(proc.stdout.strip()).resolve()
    if toplevel != resolved:
        raise ValueError(f"Checkout {resolved} must be the Git repository root (found {toplevel})")
    return resolved


def head_sha(checkout: Path) -> str:
    return run_git(checkout, "rev-parse", "HEAD").stdout.strip()


def current_branch(checkout: Path) -> str:
    return run_git(checkout, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()


def dirty_status(checkout: Path) -> list[str]:
    return [
        line
        for line in run_git(checkout, "status", "--porcelain").stdout.splitlines()
        if line.strip()
    ]


def is_ignored(checkout: Path, rel_path: str | Path) -> bool:
    """Return True if `rel_path` is ignored by Git inside `checkout`."""
    proc = run_git(checkout, "check-ignore", "-q", "--", str(rel_path), check=False)
    if proc.returncode == 0:
        return True
    if proc.returncode == 1:
        return False
    detail = (proc.stderr or proc.stdout).strip() or f"exit code {proc.returncode}"
    raise GitHubError(f"git check-ignore failed for {rel_path!s} in {checkout}: {detail}")


def is_ancestor(checkout: Path, ancestor: str, descendant: str) -> bool:
    return (
        run_git(
            checkout, "merge-base", "--is-ancestor", ancestor, descendant, check=False
        ).returncode
        == 0
    )


def _require_clean_checkout(checkout: Path) -> None:
    dirty = dirty_status(checkout)
    if dirty:
        preview = ", ".join(dirty[:5])
        raise ValueError(
            f"Checkout has uncommitted changes ({len(dirty)} file(s): {preview}). "
            "Commit or discard all changes first."
        )


def verify_commit_lineage(
    checkout: Path,
    *,
    baseline_sha: str,
    expected_head: str | None = None,
    require_new_commit: bool = True,
) -> str:
    """Validate clean checkout and descendant Git lineage from `baseline_sha`."""
    try:
        _require_clean_checkout(checkout)
        head = head_sha(checkout)
        if expected_head is not None and head != expected_head:
            raise ValueError(f"Expected HEAD {expected_head!r}, but checkout HEAD is {head!r}.")
        if not is_ancestor(checkout, baseline_sha, head):
            raise ValueError(
                f"Checkout HEAD {head} is not a descendant of baseline {baseline_sha}."
            )
        if require_new_commit and head == baseline_sha:
            raise ValueError(
                f"Checkout HEAD {head} equals baseline {baseline_sha}; create a commit first."
            )
        return head
    except GitHubError as exc:
        raise ValueError(str(exc)) from exc


def _require_branch(checkout: Path) -> str:
    branch = current_branch(checkout)
    if branch == "HEAD":
        raise ValueError("Checkout is in detached HEAD state; check out a branch first.")
    return branch


def _parse_pr_json(stdout: str, args: Sequence[str], checkout: Path) -> dict[str, Any]:
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise GitHubError(f"Invalid JSON from gh {' '.join(args)} in {checkout}: {exc}") from exc
    if not isinstance(data, dict):
        raise GitHubError(f"Expected JSON object from gh CLI, got {type(data).__name__}")
    return data


def _validate_open_branch_pr(
    data: dict[str, Any], *, branch: str, expected_pr: int | None = None
) -> int:
    raw_number = data.get("number")
    if type(raw_number) is not int or raw_number <= 0:
        raise ValueError(f"Pull request has invalid number: {raw_number!r}")
    if expected_pr is not None and raw_number != expected_pr:
        raise ValueError(f"PR #{raw_number} does not match expected PR #{expected_pr}")
    state = str(data.get("state") or "")
    if state != "OPEN":
        raise ValueError(f"PR #{raw_number} state is {state or 'UNKNOWN'}, expected OPEN")
    head_branch = str(data.get("headRefName") or "")
    if head_branch != branch:
        raise ValueError(
            f"PR #{raw_number} branch {head_branch!r} does not match checked-out branch {branch!r}"
        )
    return raw_number


def verify_shipped_pr(
    checkout: Path,
    *,
    expected_head: str,
    expected_pr: int | None = None,
) -> tuple[int, str]:
    """Verify clean checkout at `expected_head` and an open non-draft PR on GitHub at that SHA."""
    verify_commit_lineage(
        checkout,
        baseline_sha=expected_head,
        expected_head=expected_head,
        require_new_commit=False,
    )
    branch = _require_branch(checkout)
    try:
        data = _view_pr(checkout, expected_pr)
    except GitHubError as exc:
        raise ValueError(f"Could not read pull request via gh: {exc}") from exc

    pr_number = _validate_open_branch_pr(data, branch=branch, expected_pr=expected_pr)
    if bool(data.get("isDraft", False)):
        raise ValueError(f"PR #{pr_number} is still a draft; mark it ready for review")

    remote_head = str(data.get("headRefOid") or "")
    if remote_head != expected_head:
        raise ValueError(
            f"PR #{pr_number} head {remote_head!r} does not match expected commit "
            f"{expected_head!r}; push {expected_head} to the PR branch"
        )
    return pr_number, remote_head


def probe_ship_pr(
    checkout: Path,
    *,
    expected_head: str,
    expected_pr: int | None = None,
) -> tuple[int, str] | None:
    """Validate branch cleanliness and probe GitHub for an existing ready PR at `expected_head`."""
    _require_clean_checkout(checkout)
    branch = _require_branch(checkout)
    args = ["pr", "view"]
    if expected_pr is not None:
        args.append(str(expected_pr))
    args.extend(["--json", _PR_VIEW_FIELDS])
    proc = _run_gh(checkout, *args, check=False)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip() or f"exit code {proc.returncode}"
        if expected_pr is None and "no pull requests found for branch" in detail.lower():
            return None
        raise GitHubError(f"gh {' '.join(args)} failed in {checkout}: {detail}")
    data = _parse_pr_json(proc.stdout, args, checkout)

    pr_number = _validate_open_branch_pr(data, branch=branch, expected_pr=expected_pr)
    remote_head = str(data.get("headRefOid") or "")
    if bool(data.get("isDraft", False)) or remote_head != expected_head:
        return None
    return pr_number, remote_head


def _view_pr(checkout: Path, pr_number: int | None = None) -> dict[str, Any]:
    args = ["pr", "view"]
    if pr_number is not None:
        args.append(str(pr_number))
    args.extend(["--json", _PR_VIEW_FIELDS])
    return _parse_pr_json(_run_gh(checkout, *args).stdout, args, checkout)


def _merge_commit_sha(data: dict[str, Any]) -> str | None:
    mc = data.get("mergeCommit")
    oid = mc.get("oid") if isinstance(mc, dict) else None
    return str(oid) if oid else None


def _classify_rollup_item(item: dict[str, Any]) -> tuple[Literal["ok", "pending", "failed"], str]:
    """Normalize a GitHub CheckRun or StatusContext rollup entry into `(kind, reason)`."""
    name = str(item.get("name") or item.get("context") or "check")
    status = str(item.get("status") or "").upper()
    conclusion = str(item.get("conclusion") or "").upper()
    check_state = str(item.get("state") or "").upper()
    if status in _PENDING_STATUSES or check_state in ("PENDING", "EXPECTED"):
        return ("pending", f"check {name!r} is {status or check_state}")
    if status and status != "COMPLETED":
        return ("failed", f"check {name!r} is {status}")
    if conclusion and conclusion not in ("SUCCESS", "NEUTRAL", "SKIPPED"):
        return ("failed", f"check {name!r} concluded {conclusion}")
    if check_state and check_state != "SUCCESS":
        return ("failed", f"status {name!r} is {check_state}")
    return ("ok", "")


def _eval_readiness(data: dict[str, Any], head: str) -> PrReadiness:
    number = int(data["number"])
    pr_head = str(data.get("headRefOid") or "")
    state = str(data.get("state") or "")
    is_draft = bool(data.get("isDraft", False))
    merge_state = str(data.get("mergeStateStatus") or "").upper()

    failures: list[str] = []
    pending_reasons: list[str] = []
    if state != "OPEN":
        failures.append(f"PR #{number} state is {state or 'UNKNOWN'}, expected OPEN")
    if is_draft:
        failures.append(f"PR #{number} is still a draft")
    if pr_head != head:
        failures.append(f"PR #{number} head {pr_head!r} does not match expected {head!r}")

    buckets: dict[Literal["pending", "failed"], list[str]] = {
        "pending": pending_reasons,
        "failed": failures,
    }
    rollup = data.get("statusCheckRollup")
    if isinstance(rollup, list):
        for item in rollup:
            if isinstance(item, dict):
                kind, reason = _classify_rollup_item(item)
                if kind != "ok":
                    buckets[kind].append(reason)

    has_pending_checks = bool(pending_reasons)
    if merge_state == "UNKNOWN" or (has_pending_checks and merge_state in _PENDING_MERGE_STATES):
        pending_reasons.append(f"PR #{number} mergeStateStatus is {merge_state}")
    elif merge_state and merge_state not in ("CLEAN", "HAS_HOOKS"):
        failures.append(f"PR #{number} mergeStateStatus is {merge_state}")

    reasons = tuple(failures + pending_reasons)
    return PrReadiness(
        pr_number=number,
        url=str(data.get("url") or ""),
        head_sha=pr_head,
        head_branch=str(data.get("headRefName") or ""),
        base_branch=str(data.get("baseRefName") or ""),
        ready=not reasons,
        pending=bool(pending_reasons) and not failures,
        reasons=reasons,
    )


def pr_readiness(checkout: Path, head: str, pr_number: int | None = None) -> PrReadiness:
    """Inspect PR state, head SHA, mergeability, and checks via `gh pr view`."""
    return _eval_readiness(_view_pr(checkout, pr_number), head)


def wait_for_readiness(
    checkout: Path,
    *,
    head: str,
    pr_number: int | None = None,
    poll_seconds: float,
    timeout_seconds: float,
    check_stop: Callable[[], None] | None = None,
) -> PrReadiness:
    """Poll PR readiness while checks or mergeability are pending, up to timeout."""
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    if check_stop is not None:
        check_stop()
    readiness = pr_readiness(checkout, head, pr_number=pr_number)
    while readiness.pending:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        sleep_until = time.monotonic() + min(max(0.0, poll_seconds), remaining)
        while True:
            if check_stop is not None:
                check_stop()
            nap = sleep_until - time.monotonic()
            if nap <= 0:
                break
            time.sleep(min(nap, _STOP_CHECK_INTERVAL))
        readiness = pr_readiness(checkout, head, pr_number=pr_number)
    return readiness


def _reconcile_pr_view(data: dict[str, Any], pr: int, head: str) -> MergeResult | None:
    """Return a merged/readiness_changed MergeResult from `gh pr view`, or None if ready."""
    if str(data.get("state") or "") == "MERGED" and str(data.get("headRefOid") or "") == head:
        return MergeResult(
            status="merged",
            pr_number=pr,
            head_sha=head,
            merge_commit_sha=_merge_commit_sha(data),
        )
    readiness = _eval_readiness(data, head)
    if not readiness.ready:
        return MergeResult(
            status="readiness_changed",
            pr_number=pr,
            head_sha=head,
            reasons=readiness.reasons,
        )
    return None


def merge_pr_exact_head(
    checkout: Path,
    pr: int,
    head: str,
    *,
    merge_method: MergeMethod = "squash",
) -> MergeResult:
    """Idempotently verify readiness and merge exact head via `gh pr merge`."""
    if result := _reconcile_pr_view(_view_pr(checkout, pr), pr, head):
        return result

    proc = _run_gh(
        checkout,
        "pr",
        "merge",
        str(pr),
        f"--{merge_method}",
        "--match-head-commit",
        head,
        check=False,
    )
    if proc.returncode != 0:
        if result := _reconcile_pr_view(_view_pr(checkout, pr), pr, head):
            return result
        detail = (proc.stderr or proc.stdout).strip() or f"exit code {proc.returncode}"
        raise GitHubError(f"gh pr merge #{pr} failed: {detail}")
    return MergeResult(status="merged", pr_number=pr, head_sha=head)


def verify_pr_merged(checkout: Path, pr: int, head: str) -> MergeResult:
    """Confirm via `gh pr view` that `pr` is MERGED at exact `head`."""
    data = _view_pr(checkout, pr)
    state = str(data.get("state") or "")
    pr_head = str(data.get("headRefOid") or "")
    if state != "MERGED":
        raise GitHubError(f"PR #{pr} expected state MERGED, got {state!r}")
    if pr_head != head:
        raise GitHubError(f"PR #{pr} merged head {pr_head!r} does not match expected {head!r}")
    return MergeResult(
        status="merged", pr_number=pr, head_sha=head, merge_commit_sha=_merge_commit_sha(data)
    )


def cleanup_merged_branch(checkout: Path, pr: int) -> str | None:
    """Safely delete the local PR head branch after verifying merge and exact local tip."""
    data = _view_pr(checkout, pr)
    if str(data.get("state") or "") != "MERGED":
        raise GitHubError(f"Refusing to clean branch: PR #{pr} is not MERGED")
    if dirty_status(checkout):
        raise GitHubError(f"Refusing to clean branch: checkout {checkout} is dirty")

    head_branch = str(data.get("headRefName") or "")
    base_branch = str(data.get("baseRefName") or "")
    merged_head_sha = str(data.get("headRefOid") or "")
    if not head_branch or head_branch == base_branch:
        return None

    ref = f"refs/heads/{head_branch}"
    if run_git(checkout, "show-ref", "--verify", "--quiet", ref, check=False).returncode != 0:
        return None
    local_tip = run_git(checkout, "rev-parse", ref).stdout.strip()
    if merged_head_sha and local_tip != merged_head_sha:
        raise GitHubError(
            f"Refusing to delete local branch {head_branch!r}: "
            f"tip {local_tip} != merged PR head {merged_head_sha}"
        )

    if current_branch(checkout) == head_branch:
        base_ref = f"refs/heads/{base_branch}"
        has_base = (
            bool(base_branch)
            and run_git(
                checkout, "show-ref", "--verify", "--quiet", base_ref, check=False
            ).returncode
            == 0
        )
        run_git(checkout, "checkout", *(base_branch,) if has_base else ("--detach", "HEAD"))

    if run_git(checkout, "branch", "-d", head_branch, check=False).returncode != 0:
        # Safe for squash-merged branches: PR is verified MERGED and local tip matches.
        run_git(checkout, "branch", "-D", head_branch)
    return head_branch
