"""Contracts of the experiments module: semantic types and public data structures, without implementation.

Read this file before calling the module. Changing it is an architecture decision.

Threshold calibration exchanges two JSONL files per entity kind (data/experiments/threshold/{roles,terms}/)
with the judging session:
    candidates.jsonl  one line per match: {"candidate_id", "sentence_id", "chunk_id", "sentence", "span",
                      "entity", "score", "description"}; a sentence with several matches has several lines
    judgments.jsonl   one line per candidate, in any order: {"candidate_id": int, "correct": bool}"""

from dataclasses import dataclass
from typing import NewType

from botc.retrieval.contracts import ChunkId, GlossaryTerm, Match, RoleName, Similarity

SentenceId = NewType("SentenceId", int)    # position of the sentence in the sample: 0..n-1
Sentence = NewType("Sentence", str)        # a sentence of a chunk text, as written on the wiki
CandidateId = NewType("CandidateId", int)  # line of the candidate in its candidates.jsonl: 0..m-1
Description = NewType("Description", str)  # what the judge reads about the entity: role page start, "Term: definition"
Seed = NewType("Seed", int)
WordCount = NewType("WordCount", int)      # length in words (split on whitespace), not in tokens
Precision = NewType("Precision", float)    # in [0, 1]
Recall = NewType("Recall", float)          # relative recall: correct matches >= t / correct matches >= t_min, in [0, 1]
F1 = NewType("F1", float)                  # harmonic mean of precision and relative recall, in [0, 1]


@dataclass(frozen=True)
class SampledSentence:
    """A sentence drawn from the text of a chunk, to be matched and judged.

    Valid if the id is >= 0 and the text is not blank."""

    id: SentenceId
    chunk_id: ChunkId  # where it comes from: traceability for the judge and the manual control
    text: Sentence

    def __post_init__(self) -> None:
        if self.id < 0:
            raise ValueError(f"Negative sentence id: {self.id}")
        if not self.text.strip():
            raise ValueError(f"Blank sentence {self.id} from {self.chunk_id!r}")


@dataclass(frozen=True)
class Candidate[N: (RoleName, GlossaryTerm)]:
    """A match found in a sampled sentence, to be judged correct or not.

    Valid if the id is >= 0. The kind of entity is carried by N only: roles and terms are never mixed."""

    id: CandidateId
    sentence: SampledSentence
    match: Match[N]

    def __post_init__(self) -> None:
        if self.id < 0:
            raise ValueError(f"Negative candidate id: {self.id}")


@dataclass(frozen=True)
class CurvePoint:
    """Precision, relative recall and F1 of a matcher at one threshold.

    Valid if 0 <= correct <= matches, if precision is None exactly when matches == 0 (0/0 is undefined),
    if f1 is None exactly when precision is None, and if precision, recall and f1 are in [0, 1]."""

    threshold: Similarity
    matches: int                 # candidates with score >= threshold
    correct: int                 # among them, judged correct
    precision: Precision | None  # correct / matches
    recall: Recall
    f1: F1 | None                # 0 when precision + recall == 0

    def __post_init__(self) -> None:
        if not 0 <= self.correct <= self.matches:
            raise ValueError(f"Need 0 <= correct <= matches, got {self.correct} and {self.matches}")
        if (self.precision is None) != (self.matches == 0):
            raise ValueError(f"Precision must be None exactly when there is no match, got {self.precision}")
        if (self.f1 is None) != (self.precision is None):
            raise ValueError(f"F1 must be None exactly when precision is None, got {self.f1}")
        for name, value in [("precision", self.precision), ("recall", self.recall), ("f1", self.f1)]:
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"{name} out of [0, 1]: {value}")
