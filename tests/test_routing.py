"""Behavioral unit tests for deterministic coder and reviewer routing."""

from __future__ import annotations

import random

import pytest
from pydantic import ValidationError

from castle_steward_coding.routing import (
    AgentProfiles,
    Assignment,
    Difficulty,
    RoutingPolicy,
    assign_agents,
    lowest_ratio,
    select_reviewers,
)
from castle_steward_engine import AgentSpec, AvailableAgent, QuotaReading, QuotaSnapshot


def _eligible(*excluded: str, no_mcp: tuple[str, ...] = ()) -> dict[str, AvailableAgent]:
    all_agents = ("antigravity", "claude", "codex", "grok", "muse")
    return {
        name: AvailableAgent(agent=name, session_mcp=name not in no_mcp)
        for name in all_agents
        if name not in excluded
    }


def _quota(
    agents: dict[str, tuple[bool, float | None]] | None = None,
    models: dict[str, dict[str, tuple[bool, float | None]]] | None = None,
) -> QuotaSnapshot:
    return QuotaSnapshot(
        by_agent={
            agent: QuotaReading(blocked=blocked, pace_ratio=ratio)
            for agent, (blocked, ratio) in (agents or {}).items()
        },
        by_model={
            agent: {
                model: QuotaReading(blocked=blocked, pace_ratio=ratio)
                for model, (blocked, ratio) in entries.items()
            }
            for agent, entries in (models or {}).items()
        },
    )


def _assert_assignment_invariants(assignment: Assignment, *, expected_reviewers: int) -> None:
    assert len(assignment.reviewers) == expected_reviewers
    families = [r.agent.agent for r in assignment.reviewers]
    assert assignment.coder.agent not in families
    assert len(set(families)) == len(families)
    if assignment.coder.agent == "antigravity":
        assert assignment.coder.reasoning_effort is None
    for reviewer in assignment.reviewers:
        if reviewer.agent.agent == "antigravity":
            assert reviewer.agent.reasoning_effort is None
    assert assignment.summary.strip()


def test_low_difficulty_coder_default_pace_swap_fallback_and_exhaustion() -> None:
    policy = RoutingPolicy()
    elig = _eligible()

    # Default sol_high when ratios unknown
    default = assign_agents(
        difficulty="low", needs_plan=False, policy=policy, eligible=elig, quota=_quota()
    )
    assert default.coder_profile == "sol_high"
    assert default.coder == policy.profiles.sol_high
    _assert_assignment_invariants(default, expected_reviewers=2)

    # Pace swap to grok when grok < 0.4 and codex > 1.0
    swapped = assign_agents(
        difficulty="low",
        needs_plan=False,
        policy=policy,
        eligible=elig,
        quota=_quota({"grok": (False, 0.39), "codex": (False, 1.01)}),
    )
    assert swapped.coder_profile == "grok"
    assert [r.profile for r in swapped.reviewers] == ["sol_high", "opus_high"]
    _assert_assignment_invariants(swapped, expected_reviewers=2)

    # Strict inequality boundaries, unknown ratios, and blocked grok do not swap
    non_swap_cases = (
        _quota({"grok": (False, 0.4), "codex": (False, 1.5)}),
        _quota({"grok": (False, 0.2), "codex": (False, 1.0)}),
        _quota({"grok": (False, 0.2), "codex": (False, None)}),
        _quota({"grok": (False, None), "codex": (False, 1.5)}),
        _quota({"grok": (True, 0.1), "codex": (False, 1.5)}),
    )
    for snap in non_swap_cases:
        res = assign_agents(
            difficulty="low", needs_plan=False, policy=policy, eligible=elig, quota=snap
        )
        assert res.coder_profile == "sol_high"

    # Blocked or unavailable codex falls back to grok regardless of grok pace ratio
    for el, snap in (
        (elig, _quota({"codex": (True, 0.1), "grok": (False, 1.8)})),
        (_eligible("codex"), _quota({"grok": (False, None)})),
        (_eligible(no_mcp=("codex",)), _quota()),
    ):
        fb = assign_agents(
            difficulty="low", needs_plan=False, policy=policy, eligible=el, quota=snap
        )
        assert fb.coder_profile == "grok"
        _assert_assignment_invariants(fb, expected_reviewers=2)

    # Total exhaustion fails loudly
    with pytest.raises(ValueError, match="No usable coder"):
        assign_agents(
            difficulty="low",
            needs_plan=False,
            policy=policy,
            eligible=elig,
            quota=_quota({"codex": (True, None), "grok": (True, None)}),
        )


@pytest.mark.parametrize(
    ("needs_plan", "chain"),
    [
        (False, ("sol_high", "opus_high", "hydrogen_xhigh")),
        (True, ("opus_high", "neon_high", "sol_high")),
    ],
)
def test_medium_difficulty_primary_fallbacks_and_exhaustion(
    needs_plan: bool, chain: tuple[str, str, str]
) -> None:
    policy = RoutingPolicy()
    agent_of = {name: getattr(policy.profiles, name).agent for name in chain}

    # Primary
    r0 = assign_agents(
        difficulty="medium",
        needs_plan=needs_plan,
        policy=policy,
        eligible=_eligible(),
        quota=_quota(),
    )
    assert r0.coder_profile == chain[0]
    _assert_assignment_invariants(r0, expected_reviewers=2)

    # First fallback (via blocked)
    r1 = assign_agents(
        difficulty="medium",
        needs_plan=needs_plan,
        policy=policy,
        eligible=_eligible(),
        quota=_quota({agent_of[chain[0]]: (True, None)}),
    )
    assert r1.coder_profile == chain[1]
    _assert_assignment_invariants(r1, expected_reviewers=2)

    # Second fallback (mixing host-unavailable and blocked)
    r2 = assign_agents(
        difficulty="medium",
        needs_plan=needs_plan,
        policy=policy,
        eligible=_eligible(agent_of[chain[0]]),
        quota=_quota({agent_of[chain[1]]: (True, None)}),
    )
    assert r2.coder_profile == chain[2]
    _assert_assignment_invariants(r2, expected_reviewers=2)

    # Total exhaustion fails loudly
    with pytest.raises(ValueError, match="No usable coder"):
        assign_agents(
            difficulty="medium",
            needs_plan=needs_plan,
            policy=policy,
            eligible=_eligible(agent_of[chain[0]]),
            quota=_quota({agent_of[chain[1]]: (True, None), agent_of[chain[2]]: (True, None)}),
        )


def test_high_difficulty_fable_gate_fallbacks_and_model_pool_granularity() -> None:
    policy = RoutingPolicy()
    elig = _eligible()

    # fable_high chosen when pace < 0.5 (regardless of needs_plan)
    for needs_plan in (False, True):
        res = assign_agents(
            difficulty="high",
            needs_plan=needs_plan,
            policy=policy,
            eligible=elig,
            quota=_quota(models={"claude": {"fable": (False, 0.49)}}),
        )
        assert res.coder_profile == "fable_high"
        assert res.coder == policy.profiles.fable_high
        _assert_assignment_invariants(res, expected_reviewers=3)

    # fable_high skipped at >= 0.5, unknown, or blocked -> neon_xhigh
    for snap in (
        _quota({"claude": (False, 0.5)}),
        _quota({"claude": (False, 0.9)}),
        _quota(),
        _quota(models={"claude": {"fable": (True, 0.1)}}),
    ):
        res = assign_agents(
            difficulty="high", needs_plan=True, policy=policy, eligible=elig, quota=snap
        )
        assert res.coder_profile == "neon_xhigh"
        assert res.coder == policy.profiles.neon_xhigh
        _assert_assignment_invariants(res, expected_reviewers=3)

    # Model-level fable block + antigravity block still allows opus_xhigh
    opus_res = assign_agents(
        difficulty="high",
        needs_plan=True,
        policy=policy,
        eligible=elig,
        quota=_quota(
            {"antigravity": (True, None)},
            models={"claude": {"fable": (True, 0.1), "opus": (False, 0.8)}},
        ),
    )
    assert opus_res.coder_profile == "opus_xhigh"
    _assert_assignment_invariants(opus_res, expected_reviewers=3)

    # Final fallback to sol_xhigh when claude & antigravity are out
    sol_res = assign_agents(
        difficulty="high",
        needs_plan=False,
        policy=policy,
        eligible=_eligible("antigravity"),
        quota=_quota({"claude": (True, None)}),
    )
    assert sol_res.coder_profile == "sol_xhigh"
    # Only grok & muse remain as cross-family reviewers -> degrades cleanly to 2
    assert {r.profile for r in sol_res.reviewers} == {"grok", "muse"}
    _assert_assignment_invariants(sol_res, expected_reviewers=2)

    # Total exhaustion fails loudly
    with pytest.raises(ValueError, match="No usable coder"):
        assign_agents(
            difficulty="high",
            needs_plan=True,
            policy=policy,
            eligible=_eligible("antigravity", "claude", "codex"),
            quota=_quota(),
        )


def test_reviewer_selection_pace_ordering_known_beats_unknown_and_seeded_ties() -> None:
    policy = RoutingPolicy()
    elig = _eligible()

    # Known ratio beats unknown (even when known ratio is high)
    known_wins = lowest_ratio(
        [("grok", policy.profiles.grok), ("muse", policy.profiles.muse)],
        eligible=elig,
        quota=_quota({"grok": (False, None), "muse": (False, 2.5)}),
        rng=random.Random(0),
    )
    assert known_wins is not None and known_wins[0] == "muse"

    # Low/medium reviewer selection picks lowest ratio from each pool for each coder family
    q = _quota(
        {
            "codex": (False, 0.2),
            "claude": (False, 0.7),
            "antigravity": (False, 0.4),
            "grok": (False, 0.8),
            "muse": (False, 0.3),
        }
    )
    for coder, expected_top in (
        (policy.profiles.sol_high, "neon_high"),  # antigravity 0.4 < claude 0.7
        (policy.profiles.opus_high, "sol_high"),  # codex 0.2 < antigravity 0.4
        (policy.profiles.neon_high, "sol_high"),  # codex 0.2 < claude 0.7
    ):
        revs, _ = select_reviewers(
            difficulty="medium", coder=coder, policy=policy, eligible=elig, quota=q
        )
        assert [r.profile for r in revs] == [expected_top, "muse"]

    # High difficulty picks BOTH non-coder top-tier families + lowest of {grok, muse}
    high_revs, _ = select_reviewers(
        difficulty="high", coder=policy.profiles.fable_high, policy=policy, eligible=elig, quota=q
    )
    assert [r.profile for r in high_revs] == ["sol_high", "neon_high", "muse"]

    # Exact ties and all-unknown break randomly via injectable rng (reproducible per seed)
    for tie_q in (_quota({"grok": (False, 0.5), "muse": (False, 0.5)}), _quota()):
        picks_a = [
            select_reviewers(
                difficulty="high",
                coder=policy.profiles.fable_high,
                policy=policy,
                eligible=elig,
                quota=tie_q,
                rng=random.Random(seed),
            )[0][2].profile
            for seed in range(20)
        ]
        picks_b = [
            select_reviewers(
                difficulty="high",
                coder=policy.profiles.fable_high,
                policy=policy,
                eligible=elig,
                quota=tie_q,
                rng=random.Random(seed),
            )[0][2].profile
            for seed in range(20)
        ]
        assert picks_a == picks_b
        assert set(picks_a) == {"grok", "muse"}


def test_reviewer_shortfall_substitution_degradation_and_minimum_guarantee() -> None:
    policy = RoutingPolicy()
    elig = _eligible()

    # Medium with both grok & muse blocked -> shortfall picks the other top-tier reviewer
    med_revs, med_summary = select_reviewers(
        difficulty="medium",
        coder=policy.profiles.sol_high,
        policy=policy,
        eligible=elig,
        quota=_quota({"grok": (True, None), "muse": (True, None)}),
    )
    assert {r.profile for r in med_revs} == {"opus_high", "neon_high"}
    assert "shortfall" in med_summary

    # High with one top-tier blocked -> shortfall picks BOTH grok and muse (still 3 reviewers)
    high_sub, _ = select_reviewers(
        difficulty="high",
        coder=policy.profiles.fable_high,
        policy=policy,
        eligible=elig,
        quota=_quota({"codex": (True, None)}),
    )
    assert {r.profile for r in high_sub} == {"neon_high", "grok", "muse"}

    # High with both grok & muse blocked -> cleanly degrades to 2 cross-family top-tier reviewers
    high_deg, deg_summary = select_reviewers(
        difficulty="high",
        coder=policy.profiles.neon_xhigh,
        policy=policy,
        eligible=elig,
        quota=_quota({"grok": (True, None), "muse": (True, None)}),
    )
    assert [r.profile for r in high_deg] == ["sol_high", "opus_high"]
    assert "degraded from 3 to 2" in deg_summary

    # Grok coder with codex blocked -> opus_high + lowest of {neon_high, muse}
    grok_sub, _ = select_reviewers(
        difficulty="low",
        coder=policy.profiles.grok,
        policy=policy,
        eligible=elig,
        quota=_quota({"codex": (True, None), "antigravity": (False, 0.2), "muse": (False, 0.9)}),
    )
    assert [r.profile for r in grok_sub] == ["opus_high", "neon_high"]

    # Fewer than 2 distinct cross-family reviewers fails loudly
    for diff in ("low", "medium", "high"):
        d: Difficulty = diff
        with pytest.raises(ValueError, match="at least 2"):
            select_reviewers(
                difficulty=d,
                coder=policy.profiles.sol_high,
                policy=policy,
                eligible=_eligible("antigravity", "grok", "muse"),
                quota=_quota(),
            )

    # Custom grok profile agent family triggers grok-coder primary reviewer selection
    custom_policy = RoutingPolicy(
        profiles=AgentProfiles(grok=AgentSpec(agent="xai", model="grok-custom"))
    )
    custom_elig = {**_eligible(), "xai": AvailableAgent(agent="xai", session_mcp=True)}
    custom_revs, _ = select_reviewers(
        difficulty="low",
        coder=custom_policy.profiles.grok,
        policy=custom_policy,
        eligible=custom_elig,
        quota=_quota(),
    )
    assert [r.profile for r in custom_revs] == ["sol_high", "opus_high"]

    # Strict validation on RoutingPolicy and AgentProfiles
    for bad in (0.0, float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            RoutingPolicy(low_grok_max_ratio=bad)
    with pytest.raises(ValidationError):
        AgentProfiles.model_validate({"unknown_profile": AgentSpec(agent="codex")})
