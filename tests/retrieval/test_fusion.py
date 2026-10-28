import math
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from typing import Any

import pytest

from botc.retrieval.contracts import (
    Boost,
    ChunkId,
    ConstantBoost,
    DenseScore,
    FusedScore,
    Mass,
    MassFloor,
    Temperature,
)
from botc.retrieval.fusion import bonus, fuse, rank


def dense(**scores: float) -> dict[ChunkId, DenseScore]:
    return {ChunkId(id): DenseScore(score) for id, score in scores.items()}


def ids(*names: str) -> set[ChunkId]:
    return {ChunkId(name) for name in names}


# a0, a1: chunks of a detected role (boosted); b0, b1, b2: other chunks. The best chunk overall is not boosted.
SCORES = dense(a0=0.80, a1=0.50, b0=0.90, b1=0.70, b2=0.60)
BOOSTED = ids("a0", "a1")


def mass(scores: Mapping[ChunkId, float], boosted: AbstractSet[ChunkId], tau: float) -> float:
    """Total probability of the boosted chunks under softmax(scores / tau), computed stably (shift by the max)."""
    top = max(scores.values())
    weights = {id: math.exp((score - top) / tau) for id, score in scores.items()}
    return sum(weights[id] for id in boosted) / sum(weights.values())


def logit(p: float) -> float:
    return math.log(p / (1 - p))


# --- bonus: constant boost ---

def test_a_constant_boost_is_beta_whatever_the_scores():
    """ConstantBoost ignores the scores: the bonus is beta, for any scores and any set of boosted chunks."""
    rule = ConstantBoost(beta=Boost(0.3))
    assert bonus(SCORES, BOOSTED, rule) == 0.3
    assert bonus(dense(a0=-0.5, b0=0.9), ids("a0"), rule) == 0.3


# --- bonus: mass floor ---

def test_no_bonus_when_a_already_has_the_mass_floor():
    """If the boosted chunks already have a total probability >= alpha under softmax(scores / tau), no bonus.

    Here the boosted chunks are the two best ones."""
    scores = dense(a0=0.95, a1=0.90, b0=0.50, b1=0.40)
    assert mass(scores, ids("a0", "a1"), 0.05) > 0.3
    assert bonus(scores, ids("a0", "a1"), MassFloor(alpha=Mass(0.3), tau=Temperature(0.05))) == 0


def test_the_mass_floor_bonus_gives_a_exactly_alpha():
    """Below alpha, the bonus is the one that brings the boosted chunks to exactly alpha: after fuse, their total
    probability under softmax(fused / tau) is alpha.

    Here it is about 0.12 before the bonus (tau = 0.05), below alpha = 0.3."""
    rule = MassFloor(alpha=Mass(0.3), tau=Temperature(0.05))
    assert mass(SCORES, BOOSTED, 0.05) < 0.3
    fused = fuse(SCORES, BOOSTED, bonus(SCORES, BOOSTED, rule))
    assert mass(fused, BOOSTED, 0.05) == pytest.approx(0.3, rel=1e-9)


def test_the_cap_bounds_the_mass_floor_bonus():
    """With a cap, the bonus is min(uncapped bonus, cap): a cap below the uncapped bonus gives the cap, a cap above it
    changes nothing, a cap of 0 gives 0.

    Here the uncapped bonus is about 0.059 (alpha = 0.3, tau = 0.05)."""
    uncapped = bonus(SCORES, BOOSTED, MassFloor(alpha=Mass(0.3), tau=Temperature(0.05)))
    assert bonus(SCORES, BOOSTED, MassFloor(alpha=Mass(0.3), tau=Temperature(0.05), cap=Boost(0.01))) == 0.01
    assert bonus(SCORES, BOOSTED, MassFloor(alpha=Mass(0.3), tau=Temperature(0.05), cap=Boost(1.0))) == uncapped
    assert bonus(SCORES, BOOSTED, MassFloor(alpha=Mass(0.3), tau=Temperature(0.05), cap=Boost(0.0))) == 0


@pytest.mark.parametrize("boosted", [set[ChunkId](), set(SCORES)], ids=["nothing boosted", "everything boosted"])
def test_no_mass_floor_bonus_when_nothing_or_everything_is_boosted(boosted: set[ChunkId]):
    """Without any detected role, or when every chunk is boosted, the ranking cannot change: bonus 0.

    Neither case divides by zero nor takes the logit of 0 or 1."""
    assert bonus(SCORES, boosted, MassFloor(alpha=Mass(0.3), tau=Temperature(0.05))) == 0


def test_the_mass_floor_bonus_is_exact_when_the_probability_underflows():
    """At low tau the total probability of the boosted chunks can be 0 in floating point (here e^-1000):
    the bonus must stay finite and exact.

    Scores 0 (boosted) and 1 (other), tau = 0.001, alpha = 0.5: logit of the probability = -1000, bonus = 1."""
    scores = dense(a0=0.0, b0=1.0)
    rule = MassFloor(alpha=Mass(0.5), tau=Temperature(0.001))
    assert bonus(scores, ids("a0"), rule) == pytest.approx(1.0, rel=1e-9)


def test_at_low_tau_the_best_boosted_chunk_rises_to_the_best_score_plus_tau_logit_alpha():
    """At low tau the mass floor lifts the best boosted chunk next to the best chunk overall: its fused score is
    about best score + tau * logit(alpha), whatever the gap between them.

    tau = 0.01, alpha = 0.3: the best boosted chunk goes from 0.80 to about 0.90 - 0.0085."""
    rule = MassFloor(alpha=Mass(0.3), tau=Temperature(0.01))
    fused = fuse(SCORES, BOOSTED, bonus(SCORES, BOOSTED, rule))
    assert fused[ChunkId("a0")] == pytest.approx(0.90 + 0.01 * logit(0.3), abs=1e-4)


def test_bonus_rejects_a_boosted_chunk_without_score():
    """A boosted chunk missing from the scores means the scores do not cover the whole index: ValueError."""
    with pytest.raises(ValueError):
        bonus(SCORES, ids("a0", "zz"), MassFloor(alpha=Mass(0.3), tau=Temperature(0.05)))


# --- fuse ---

def test_fuse_adds_the_bonus_to_the_boosted_chunks_only():
    """Every boosted chunk gets exactly its dense score plus the bonus; every other chunk keeps its dense score.

    So the order among the boosted chunks and among the other chunks is unchanged."""
    assert fuse(SCORES, BOOSTED, Boost(0.25)) == pytest.approx(
        {"a0": 1.05, "a1": 0.75, "b0": 0.90, "b1": 0.70, "b2": 0.60})


def test_fuse_rejects_a_boosted_chunk_without_score():
    """A boosted chunk missing from the scores would be silently dropped from the ranking: ValueError."""
    with pytest.raises(ValueError):
        fuse(SCORES, ids("a0", "zz"), Boost(0.1))


# --- rank ---

def test_rank_orders_by_decreasing_fused_score_then_by_id():
    """The context is filled in this order: best fused score first; equal scores by increasing id (deterministic)."""
    fused = {ChunkId(id): FusedScore(score) for id, score in [("b", 0.5), ("c", 0.9), ("a", 0.5), ("d", -0.1)]}
    assert rank(fused) == ["c", "a", "b", "d"]


# --- contracts ---

def test_a_constant_boost_is_not_negative():
    """A negative bonus would rank the chunks of the detected roles lower, the opposite of the goal: ValueError.

    beta = 0 is valid."""
    assert ConstantBoost(beta=Boost(0.0)).beta == 0
    with pytest.raises(ValueError):
        ConstantBoost(beta=Boost(-0.1))


@pytest.mark.parametrize("fields", [
    {"alpha": Mass(0.0)},
    {"alpha": Mass(1.0)},
    {"alpha": Mass(-0.1)},
    {"tau": Temperature(0.0)},
    {"tau": Temperature(-0.05)},
    {"cap": Boost(-0.1)},
], ids=["alpha 0", "alpha 1", "negative alpha", "tau 0", "negative tau", "negative cap"])
def test_an_invalid_mass_floor_raises(fields: dict[str, Any]):
    """A mass floor has 0 < alpha < 1 (alpha = 1 needs an infinite bonus), tau > 0 and a cap None or >= 0.

    Each case changes one field of a valid rule (alpha 0.3, tau 0.05, no cap)."""
    with pytest.raises(ValueError):
        MassFloor(**{"alpha": Mass(0.3), "tau": Temperature(0.05), **fields})
