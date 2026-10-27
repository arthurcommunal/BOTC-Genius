"""Contracts of the retrieval module: semantic types and public data structures, without implementation.

Read this file before calling the module. Changing it is an architecture decision."""

from dataclasses import dataclass
from typing import NewType

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
