"""Contracts of the retrieval module: semantic types and public data structures, without implementation.

Read this file before calling the module. Changing it is an architecture decision."""

from dataclasses import dataclass
from typing import NewType

### --- entities.py --- ###

ChunkId = NewType("ChunkId", str)                # "Imp#3": page file + position in the page
RoleName = NewType("RoleName", str)              # title of a role page: "Fortune Teller"
GlossaryTerm = NewType("GlossaryTerm", str)      # raw term of the glossary: "Each night*", "Demon, The"
Question = NewType("Question", str)              # question of the user, as typed
NormalizedText = NewType("NormalizedText", str)  # output of entities.normalize: [a-z0-9*] words, single spaces
Similarity = NewType("Similarity", float)        # cosine in [0, 1]; a matching threshold is a Similarity too


@dataclass(frozen=True)
class GlossaryEntry:
    """An entry of the glossary: a term and the paragraph that defines it.

    Valid if the term is not blank and the text contains the "Term: definition" separator ": "."""

    term: GlossaryTerm  # raw, as written
    text: str           # the whole paragraph: "Term: definition", put as is in the context of the LLM

    def __post_init__(self) -> None:
        if not self.term.strip():
            raise ValueError(f"Blank glossary term for {self.text[:80]!r}")
        if ": " not in self.text:
            raise ValueError(f"Glossary text without 'Term: ' for {self.term!r}")


@dataclass(frozen=True)
class Match[N: (RoleName, GlossaryTerm)]:
    """An entity recognized in a question: a role or a glossary term, never a mix of both.

    Valid if the span is not empty and 0 < score <= 1 (a score of 0 means no shared n-gram, never a match)."""

    name: N               # original form: "Fortune Teller"
    span: NormalizedText  # segment of the normalized question: "fortune teller"
    score: Similarity     # cosine between the segment and the name

    def __post_init__(self) -> None:
        if not self.span:
            raise ValueError(f"Empty span for {self.name!r}")
        if not 0 < self.score <= 1:
            raise ValueError(f"Score of {self.name!r} out of ]0, 1]: {self.score}")


### --- fusion.py --- ###
# The chunks of the pages of the roles detected in the question are "boosted": their dense score gets a bonus,
# so that they rank higher without excluding the other chunks.

DenseScore = NewType("DenseScore", float)    # cosine between the question and a chunk (vector store), in [-1, 1]
FusedScore = NewType("FusedScore", float)    # dense score, plus the bonus for a boosted chunk; only its order matters
Temperature = NewType("Temperature", float)  # tau of softmax(score / tau), in cosine units, > 0
Mass = NewType("Mass", float)                # total probability of a set of chunks under softmax(score / tau), in [0, 1]
Boost = NewType("Boost", float)              # bonus added to the dense score of every boosted chunk, in cosine units, >= 0


@dataclass(frozen=True)
class ConstantBoost:
    """Fusion rule: the same bonus beta for the boosted chunks, whatever the scores.

    Valid if beta >= 0. beta = 0 leaves the dense ranking unchanged; beta = 2 puts every boosted chunk at or above
    every other chunk (cosines are in [-1, 1])."""

    beta: Boost

    def __post_init__(self) -> None:
        if self.beta < 0:
            raise ValueError(f"Negative constant boost: {self.beta}")


@dataclass(frozen=True)
class MassFloor:
    """Fusion rule: the smallest bonus that gives the boosted chunks a total probability of at least alpha
    under softmax(score / tau), optionally capped.

    The bonus adapts to the question: none when the boosted chunks already score high, a large one when they
    score far below the best chunk. The cap bounds it, so that a chunk far from the question never rises to the top.
    Valid if 0 < alpha < 1 (alpha = 1 would need an infinite bonus: use ConstantBoost(beta=2)), tau > 0
    and cap is None or >= 0."""

    alpha: Mass               # minimal total probability of the boosted chunks
    tau: Temperature
    cap: Boost | None = None  # upper bound of the bonus; None: no bound

    def __post_init__(self) -> None:
        if not 0 < self.alpha < 1:
            raise ValueError(f"Mass floor alpha out of ]0, 1[: {self.alpha}")
        if self.tau <= 0:
            raise ValueError(f"Non-positive temperature: {self.tau}")
        if self.cap is not None and self.cap < 0:
            raise ValueError(f"Negative cap: {self.cap}")


FusionRule = ConstantBoost | MassFloor
