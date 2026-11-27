"""Pure deterministic coder and cross-family reviewer routing for `coding`."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from castle_steward_engine import AgentSpec, AvailableAgent, QuotaSnapshot

Difficulty = Literal["low", "medium", "high"]
_NonBlankStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_ROUTING_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")


class AgentProfiles(BaseModel):
    """Developer-editable default agent profiles for deterministic routing."""

    model_config = _ROUTING_MODEL_CONFIG

    sol_high: AgentSpec = AgentSpec(agent="codex", model="gpt-6.2-sol", reasoning_effort="high")
    sol_xhigh: AgentSpec = AgentSpec(agent="codex", model="gpt-6.2-sol", reasoning_effort="xhigh")
    opus_high: AgentSpec = AgentSpec(agent="claude", model="opus", reasoning_effort="high")
    opus_xhigh: AgentSpec = AgentSpec(agent="claude", model="opus", reasoning_effort="xhigh")
    fable_high: AgentSpec = AgentSpec(agent="claude", model="fable", reasoning_effort="high")
    neon_high: AgentSpec = AgentSpec(agent="antigravity", model="gemini-4-neon-high")
    neon_xhigh: AgentSpec = AgentSpec(agent="antigravity", model="gemini-4-neon-xhigh")
    hydrogen_xhigh: AgentSpec = AgentSpec(agent="antigravity", model="gemini-4-hydrogen-xhigh")
    grok: AgentSpec = AgentSpec(agent="grok", model="grok-4.7", reasoning_effort="xhigh")
    muse: AgentSpec = AgentSpec(agent="muse", model="muse-wonder-1.0", reasoning_effort="high")


class RoutingPolicy(BaseModel):
    """Checkpointed routing profiles and burn-rate thresholds."""

    model_config = _ROUTING_MODEL_CONFIG

    profiles: AgentProfiles = Field(default_factory=AgentProfiles)
    low_grok_max_ratio: float = Field(default=0.4, gt=0.0, allow_inf_nan=False)
    low_codex_min_ratio: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)
    high_fable_max_ratio: float = Field(default=0.5, gt=0.0, allow_inf_nan=False)


class AssignedReviewer(BaseModel):
    """Resolved reviewer profile and agent specification."""

    model_config = _ROUTING_MODEL_CONFIG

    profile: _NonBlankStr
    agent: AgentSpec


class Assignment(BaseModel):
    """Deterministic coder and cross-family reviewer assignment for a job."""

    model_config = _ROUTING_MODEL_CONFIG

    coder_profile: _NonBlankStr
    coder: AgentSpec
    reviewers: tuple[AssignedReviewer, ...] = Field(min_length=2)
    summary: _NonBlankStr


def _fmt_ratio(ratio: float | None) -> str:
    return "unknown" if ratio is None else f"{ratio:.2f}"


def resolve_agent(spec: AgentSpec, eligible: Mapping[str, AvailableAgent]) -> AgentSpec:
    """Validate `spec.agent` is eligible (`session_mcp=True`) and fill omitted defaults."""
    avail = eligible.get(spec.agent)
    if avail is None or not avail.session_mcp:
        names = ", ".join(sorted(k for k, v in eligible.items() if v.session_mcp)) or "(none)"
        raise ValueError(
            f"Agent {spec.agent!r} is unavailable or lacks session_mcp support (eligible: {names})"
        )
    return AgentSpec(
        agent=avail.agent,
        model=spec.model if spec.model is not None else avail.default_model,
        reasoning_effort=(
            spec.reasoning_effort
            if spec.reasoning_effort is not None
            else avail.default_reasoning_effort
        ),
    )


def is_usable(
    spec: AgentSpec,
    eligible: Mapping[str, AvailableAgent],
    quota: QuotaSnapshot,
) -> bool:
    """Return True iff `spec.agent` is eligible (`session_mcp=True`) and not quota-blocked."""
    avail = eligible.get(spec.agent)
    return avail is not None and avail.session_mcp and not quota.for_spec(spec).blocked


def lowest_ratio(
    candidates: Sequence[tuple[str, AgentSpec]],
    *,
    eligible: Mapping[str, AvailableAgent],
    quota: QuotaSnapshot,
    rng: random.Random | None = None,
) -> tuple[str, AgentSpec] | None:
    """Pick the usable candidate with lowest known pace_ratio (random tie-break).

    Known pace ratios beat unknown (`None`); if all usable candidates have unknown ratios,
    choose randomly via `rng`. Returns `None` if no candidate is usable.
    """
    usable = [(name, spec) for name, spec in candidates if is_usable(spec, eligible, quota)]
    if not usable:
        return None
    picker = rng if rng is not None else random.Random()
    known = [
        (name, spec, ratio)
        for name, spec in usable
        if (ratio := quota.for_spec(spec).pace_ratio) is not None
    ]
    if known:
        min_ratio = min(ratio for _, _, ratio in known)
        tied = [(name, spec) for name, spec, ratio in known if ratio == min_ratio]
        return tied[0] if len(tied) == 1 else picker.choice(tied)
    return usable[0] if len(usable) == 1 else picker.choice(usable)


def _select_first_usable(
    label: str,
    candidates: Sequence[tuple[str, AgentSpec]],
    eligible: Mapping[str, AvailableAgent],
    quota: QuotaSnapshot,
) -> tuple[str, AgentSpec, str]:
    skipped: list[str] = []
    for name, spec in candidates:
        if is_usable(spec, eligible, quota):
            ratio = _fmt_ratio(quota.for_spec(spec).pace_ratio)
            detail = (
                f"{label} fallback after {', '.join(skipped)}" if skipped else f"{label} primary"
            )
            return name, resolve_agent(spec, eligible), f"coder={name} ({detail}; pace={ratio})"
        skipped.append(name)
    tried = ", ".join(name for name, _ in candidates)
    raise ValueError(f"No usable coder available for {label} task (tried {tried})")


def _select_low_coder(
    policy: RoutingPolicy,
    eligible: Mapping[str, AvailableAgent],
    quota: QuotaSnapshot,
) -> tuple[str, AgentSpec, str]:
    p = policy.profiles
    sol_ok = is_usable(p.sol_high, eligible, quota)
    grok_ok = is_usable(p.grok, eligible, quota)
    codex_r = quota.for_spec(p.sol_high).pace_ratio
    grok_r = quota.for_spec(p.grok).pace_ratio
    if sol_ok:
        if (
            grok_ok
            and grok_r is not None
            and codex_r is not None
            and grok_r < policy.low_grok_max_ratio
            and codex_r > policy.low_codex_min_ratio
        ):
            return (
                "grok",
                resolve_agent(p.grok, eligible),
                f"coder=grok (low pace swap: grok={grok_r:.2f} < {policy.low_grok_max_ratio:g}, "
                f"codex={codex_r:.2f} > {policy.low_codex_min_ratio:g})",
            )
        return (
            "sol_high",
            resolve_agent(p.sol_high, eligible),
            f"coder=sol_high (low default; codex={_fmt_ratio(codex_r)}, grok={_fmt_ratio(grok_r)})",
        )
    if grok_ok:
        return (
            "grok",
            resolve_agent(p.grok, eligible),
            f"coder=grok (low fallback; sol_high unavailable/blocked; grok={_fmt_ratio(grok_r)})",
        )
    raise ValueError("No usable coder available for low-difficulty task (tried sol_high, grok)")


def _select_high_coder(
    policy: RoutingPolicy,
    eligible: Mapping[str, AvailableAgent],
    quota: QuotaSnapshot,
) -> tuple[str, AgentSpec, str]:
    p = policy.profiles
    fable_ok = is_usable(p.fable_high, eligible, quota)
    fable_r = quota.for_spec(p.fable_high).pace_ratio
    if fable_ok and fable_r is not None and fable_r < policy.high_fable_max_ratio:
        return (
            "fable_high",
            resolve_agent(p.fable_high, eligible),
            f"coder=fable_high (high pace gate: fable={fable_r:.2f} < "
            f"{policy.high_fable_max_ratio:g})",
        )
    if not fable_ok:
        fable_note = "fable unavailable/blocked"
    elif fable_r is None:
        fable_note = "fable pace unknown"
    else:
        fable_note = f"fable={fable_r:.2f} >= {policy.high_fable_max_ratio:g}"

    fallbacks = (
        ("neon_xhigh", p.neon_xhigh),
        ("opus_xhigh", p.opus_xhigh),
        ("sol_xhigh", p.sol_xhigh),
    )
    skipped = ["fable_high"]
    for name, spec in fallbacks:
        if is_usable(spec, eligible, quota):
            ratio = _fmt_ratio(quota.for_spec(spec).pace_ratio)
            detail = (
                f"high; {fable_note}"
                if len(skipped) == 1
                else f"high fallback after {', '.join(skipped)}; {fable_note}"
            )
            return name, resolve_agent(spec, eligible), f"coder={name} ({detail}; pace={ratio})"
        skipped.append(name)
    raise ValueError(
        "No usable coder available for high-difficulty task "
        "(tried fable_high, neon_xhigh, opus_xhigh, sol_xhigh)"
    )


def select_coder(
    *,
    difficulty: Difficulty,
    needs_plan: bool,
    policy: RoutingPolicy,
    eligible: Mapping[str, AvailableAgent],
    quota: QuotaSnapshot,
) -> tuple[str, AgentSpec, str]:
    """Select the coder profile and resolved `AgentSpec` with a summary explanation."""
    if difficulty == "low":
        return _select_low_coder(policy, eligible, quota)
    if difficulty == "medium":
        p = policy.profiles
        if needs_plan:
            return _select_first_usable(
                "medium/plan",
                (("opus_high", p.opus_high), ("neon_high", p.neon_high), ("sol_high", p.sol_high)),
                eligible,
                quota,
            )
        return _select_first_usable(
            "medium/no-plan",
            (
                ("sol_high", p.sol_high),
                ("opus_high", p.opus_high),
                ("hydrogen_xhigh", p.hydrogen_xhigh),
            ),
            eligible,
            quota,
        )
    return _select_high_coder(policy, eligible, quota)


def _select_primary_reviewers(
    *,
    difficulty: Difficulty,
    coder_agent: str,
    grok_agent: str,
    top_tier: Sequence[tuple[str, AgentSpec]],
    secondary_pool: Sequence[tuple[str, AgentSpec]],
    eligible: Mapping[str, AvailableAgent],
    quota: QuotaSnapshot,
    picker: random.Random,
) -> tuple[list[tuple[str, AgentSpec]], set[str]]:
    chosen: list[tuple[str, AgentSpec]] = []
    used_agents: set[str] = {coder_agent}

    def _add(entry: tuple[str, AgentSpec]) -> None:
        chosen.append(entry)
        used_agents.add(entry[1].agent)

    if coder_agent == grok_agent:
        for entry in top_tier[:2]:
            if entry[1].agent not in used_agents and is_usable(entry[1], eligible, quota):
                _add(entry)
        return chosen, used_agents

    other_top = [entry for entry in top_tier if entry[1].agent not in used_agents]
    if difficulty == "high":
        for entry in other_top:
            if entry[1].agent not in used_agents and is_usable(entry[1], eligible, quota):
                _add(entry)
    elif pick := lowest_ratio(other_top, eligible=eligible, quota=quota, rng=picker):
        _add(pick)

    secondary = [entry for entry in secondary_pool if entry[1].agent not in used_agents]
    if pick := lowest_ratio(secondary, eligible=eligible, quota=quota, rng=picker):
        _add(pick)
    return chosen, used_agents


def _fill_reviewer_shortfall(
    *,
    target: int,
    chosen: list[tuple[str, AgentSpec]],
    used_agents: set[str],
    pool: Sequence[tuple[str, AgentSpec]],
    eligible: Mapping[str, AvailableAgent],
    quota: QuotaSnapshot,
    picker: random.Random,
) -> list[str]:
    added: list[str] = []
    while len(chosen) < target:
        remaining = [entry for entry in pool if entry[1].agent not in used_agents]
        pick = lowest_ratio(remaining, eligible=eligible, quota=quota, rng=picker)
        if pick is None:
            break
        chosen.append(pick)
        used_agents.add(pick[1].agent)
        added.append(pick[0])
    return added


def select_reviewers(
    *,
    difficulty: Difficulty,
    coder: AgentSpec,
    policy: RoutingPolicy,
    eligible: Mapping[str, AvailableAgent],
    quota: QuotaSnapshot,
    rng: random.Random | None = None,
) -> tuple[tuple[AssignedReviewer, ...], str]:
    """Select distinct cross-family reviewers with shortfall substitution and >= 2 guarantee."""
    p = policy.profiles
    top_tier: tuple[tuple[str, AgentSpec], ...] = (
        ("sol_high", p.sol_high),
        ("opus_high", p.opus_high),
        ("neon_high", p.neon_high),
    )
    secondary_pool: tuple[tuple[str, AgentSpec], ...] = (
        ("grok", p.grok),
        ("muse", p.muse),
    )
    picker = rng if rng is not None else random.Random()
    target = 3 if difficulty == "high" else 2

    chosen, used_agents = _select_primary_reviewers(
        difficulty=difficulty,
        coder_agent=coder.agent,
        grok_agent=p.grok.agent,
        top_tier=top_tier,
        secondary_pool=secondary_pool,
        eligible=eligible,
        quota=quota,
        picker=picker,
    )
    shortfall_picks = _fill_reviewer_shortfall(
        target=target,
        chosen=chosen,
        used_agents=used_agents,
        pool=(*top_tier, *secondary_pool),
        eligible=eligible,
        quota=quota,
        picker=picker,
    )
    if len(chosen) < 2:
        found = ", ".join(name for name, _ in chosen) or "none"
        raise ValueError(
            f"Need at least 2 distinct cross-family reviewers for coder {coder.agent!r}, "
            f"found {len(chosen)} ({found})"
        )

    reviewers = tuple(
        AssignedReviewer(profile=name, agent=resolve_agent(spec, eligible)) for name, spec in chosen
    )
    parts = [f"{name}({_fmt_ratio(quota.for_spec(spec).pace_ratio)})" for name, spec in chosen]
    notes: list[str] = []
    if shortfall_picks:
        notes.append(f"shortfall filled by {', '.join(shortfall_picks)}")
    if len(reviewers) < target:
        notes.append(f"degraded from {target} to {len(reviewers)}")
    suffix = f" ({'; '.join(notes)})" if notes else ""
    return reviewers, f"reviewers=[{', '.join(parts)}]{suffix}"


def assign_agents(
    *,
    difficulty: Difficulty,
    needs_plan: bool,
    policy: RoutingPolicy,
    eligible: Mapping[str, AvailableAgent],
    quota: QuotaSnapshot,
    rng: random.Random | None = None,
) -> Assignment:
    """Deterministically assign coder and cross-family reviewers from task classification."""
    coder_profile, coder, coder_summary = select_coder(
        difficulty=difficulty,
        needs_plan=needs_plan,
        policy=policy,
        eligible=eligible,
        quota=quota,
    )
    reviewers, reviewer_summary = select_reviewers(
        difficulty=difficulty,
        coder=coder,
        policy=policy,
        eligible=eligible,
        quota=quota,
        rng=rng,
    )
    return Assignment(
        coder_profile=coder_profile,
        coder=coder,
        reviewers=reviewers,
        summary=f"{coder_summary}; {reviewer_summary}",
    )
