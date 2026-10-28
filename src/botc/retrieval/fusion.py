"""Fusion of the dense retrieval with the roles detected in the question.

The chunks of the pages of the detected roles ("boosted" chunks) get a bonus b >= 0 on their dense score; the other
chunks keep theirs. A FusionRule says how b is chosen: a constant, or the smallest bonus that gives the boosted
chunks a minimal total probability under a softmax (MassFloor). `bonus` computes b, `fuse` adds it, `rank` orders
the chunks."""

import math
from collections.abc import Mapping
from collections.abc import Set as AbstractSet

import numpy as np
from numpy.typing import NDArray

from botc.retrieval.contracts import (
    Boost,
    ChunkId,
    ConstantBoost,
    DenseScore,
    FusedScore,
    FusionRule,
)


def bonus(scores: Mapping[ChunkId, DenseScore], boosted: AbstractSet[ChunkId], rule: FusionRule) -> Boost:
    """Returns the bonus to add to the dense score of every boosted chunk.

    `scores` must hold the dense score of every chunk of the index, not only the top k; this function cannot
    check it (it does not know the index), the caller does.
    ConstantBoost: beta. MassFloor: let m be the total probability of the boosted chunks under softmax(scores / tau);
    if m >= alpha the bonus is 0, else it is tau * (logit(alpha) - logit(m)), the value for which
    softmax(fused scores / tau) gives them exactly alpha; then min(bonus, cap) if there is a cap.
    0 if no chunk is boosted or if every chunk is. Raises ValueError if a boosted id has no score."""
    _check_boosted(scores, boosted)
    if isinstance(rule, ConstantBoost):
        return rule.beta
    inside = np.array([score for id, score in scores.items() if id in boosted], dtype=np.float64)
    outside = np.array([score for id, score in scores.items() if id not in boosted], dtype=np.float64)
    if inside.size == 0 or outside.size == 0:
        return Boost(0.0)
    # logit(m) = log(sum of exp over boosted) - log(sum of exp over the others), never m itself: it underflows at low tau.
    logit_mass = _logsumexp(inside / rule.tau) - _logsumexp(outside / rule.tau)
    logit_alpha = math.log(rule.alpha / (1 - rule.alpha))
    value = max(0.0, rule.tau * (logit_alpha - logit_mass))
    return Boost(value if rule.cap is None else min(value, rule.cap))


def fuse(scores: Mapping[ChunkId, DenseScore], boosted: AbstractSet[ChunkId], bonus: Boost) -> dict[ChunkId, FusedScore]:
    """Returns the fused score of every chunk: its dense score, plus `bonus` if it is boosted.

    The order among the boosted chunks and among the other chunks is unchanged.
    Raises ValueError if a boosted id has no score."""
    _check_boosted(scores, boosted)
    return {id: FusedScore(score + bonus if id in boosted else score) for id, score in scores.items()}


def rank(fused: Mapping[ChunkId, FusedScore]) -> list[ChunkId]:
    """Returns the chunk ids by decreasing fused score, ties broken by increasing id (as `sort_key` in store.py)."""
    return sorted(fused, key=lambda id: (-fused[id], id))


def _check_boosted(scores: Mapping[ChunkId, DenseScore], boosted: AbstractSet[ChunkId]) -> None:
    missing = sorted(boosted - scores.keys())
    if missing:
        raise ValueError(f"{len(missing)} boosted chunks have no dense score, first ones: {missing[:5]}")


def _logsumexp(values: NDArray[np.float64]) -> float:
    """log(sum(exp(values))) without overflow nor underflow: the max is factored out."""
    top = float(values.max())
    return top + math.log(float(np.exp(values - top).sum()))
