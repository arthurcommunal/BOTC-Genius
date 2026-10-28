"""Fusion of the dense retrieval with the roles detected in the question.

The chunks of the pages of the detected roles ("boosted" chunks) get a bonus b >= 0 on their dense score; the other
chunks keep theirs. A FusionRule says how b is chosen: a constant, or the smallest bonus that gives the boosted
chunks a minimal total probability under a softmax (MassFloor). `bonus` computes b, `fuse` adds it, `rank` orders
the chunks."""

from collections.abc import Mapping
from collections.abc import Set as AbstractSet

from botc.retrieval.contracts import Boost, ChunkId, DenseScore, FusedScore, FusionRule


def bonus(scores: Mapping[ChunkId, DenseScore], boosted: AbstractSet[ChunkId], rule: FusionRule) -> Boost:
    """Returns the bonus to add to the dense score of every boosted chunk.

    `scores` holds the dense score of every chunk of the index, not only the top k.
    ConstantBoost: beta. MassFloor: let m be the total probability of the boosted chunks under softmax(scores / tau);
    if m >= alpha the bonus is 0, else it is tau * (logit(alpha) - logit(m)), the value for which
    softmax(fused scores / tau) gives them exactly alpha; then min(bonus, cap) if there is a cap.
    0 if no chunk is boosted or if every chunk is. Raises ValueError if a boosted id has no score."""
    raise NotImplementedError


def fuse(scores: Mapping[ChunkId, DenseScore], boosted: AbstractSet[ChunkId], bonus: Boost) -> dict[ChunkId, FusedScore]:
    """Returns the fused score of every chunk: its dense score, plus `bonus` if it is boosted.

    The order among the boosted chunks and among the other chunks is unchanged.
    Raises ValueError if a boosted id has no score."""
    raise NotImplementedError


def rank(fused: Mapping[ChunkId, FusedScore]) -> list[ChunkId]:
    """Returns the chunk ids by decreasing fused score, ties broken by increasing id (as `sort_key` in store.py)."""
    raise NotImplementedError
