"""Contracts of the retrieval module: semantic types and public data structures, without implementation.

Read this file before calling the module. Changing it is an architecture decision."""

from dataclasses import dataclass
from typing import NewType, Protocol

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


### --- budget.py --- ###
# The context given to the generator holds the detected glossary definitions and the best chunks, within a budget
# of tokens counted with the tokenizer of the generator.

ModelName = NewType("ModelName", str)    # Hugging Face id of the generator, whose tokenizer counts the budget
TokenCount = NewType("TokenCount", int)  # number of tokens of the generator's tokenizer, >= 0
ChunkText = NewType("ChunkText", str)    # text of a chunk as put in the context: header + "\n\n" + text


class TokenCounter(Protocol):
    """Counts the tokens of a text with the tokenizer of the generator, without special tokens."""

    def __call__(self, text: str) -> TokenCount: ...


### --- guided.py --- ###

DEFAULT_BUDGET = TokenCount(2000)
DEFAULT_THRESHOLD_ROLES = Similarity(0.9)  # calibrated on wiki sentences (src/botc/experiments/threshold.py),
DEFAULT_THRESHOLD_TERMS = Similarity(0.8)  # lowered from 1.0 for roles to tolerate typos in questions


@dataclass(frozen=True)
class GuidedConfig:
    """Settings of the guided retriever: how to fuse, how many tokens, and the matching thresholds.

    Valid if budget >= 0 and both thresholds are in [0, 1]. The budget covers the glossary definitions
    and the chunks, not the prompt template of the generator."""

    rule: FusionRule
    budget: TokenCount = DEFAULT_BUDGET
    threshold_roles: Similarity = DEFAULT_THRESHOLD_ROLES
    threshold_terms: Similarity = DEFAULT_THRESHOLD_TERMS

    def __post_init__(self) -> None:
        if self.budget < 0:
            raise ValueError(f"Negative budget: {self.budget}")
        for name, threshold in [("roles", self.threshold_roles), ("terms", self.threshold_terms)]:
            if not 0 <= threshold <= 1:
                raise ValueError(f"Threshold of {name} out of [0, 1]: {threshold}")


@dataclass(frozen=True)
class Retrieval:
    """What the guided retriever found for one question, and what it puts in the context.

    Valid if tokens >= 0, and neither chunks nor definition texts contain duplicates. That tokens is the cost of the
    definitions plus the chunks is guaranteed by the retriever, which knows the costs."""

    question: Question
    roles: tuple[Match[RoleName], ...]      # detected roles, as returned by EntityMatcher.match
    terms: tuple[Match[GlossaryTerm], ...]  # detected glossary terms, as returned by EntityMatcher.match
    definitions: tuple[GlossaryEntry, ...]  # in the context; one per paragraph (synonyms share it), order of `terms`
    bonus: Boost                            # bonus given to the chunks of the pages of the detected roles
    chunks: tuple[ChunkId, ...]             # in the context, by decreasing fused score
    tokens: TokenCount                      # definitions + chunks; above the budget only if the definitions alone are

    def __post_init__(self) -> None:
        if self.tokens < 0:
            raise ValueError(f"Negative token count: {self.tokens}")
        if len(set(self.chunks)) != len(self.chunks):
            raise ValueError(f"Duplicate chunks in the context of {self.question!r}")
        texts = [entry.text for entry in self.definitions]
        if len(set(texts)) != len(texts):
            raise ValueError(f"Duplicate definitions in the context of {self.question!r}")
