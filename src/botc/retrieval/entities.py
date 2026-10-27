"""Detect roles and specific BOTC terms in a question.

The objective is to detect roles or specific terms defined in the Glossary
in the question of the user to create a direct route towards role's page
or glossary during the retrieval."""

import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

# scikit-learn ships no type stubs: the ignores below are confined to the lines that call it,
# and its results are stored in variables typed explicitly.
from sklearn.feature_extraction.text import TfidfVectorizer  # pyright: ignore[reportMissingTypeStubs]
from sklearn.metrics.pairwise import (  # pyright: ignore[reportMissingTypeStubs]
    cosine_similarity,  # pyright: ignore[reportUnknownVariableType]
)

from botc.retrieval.contracts import (
    ChunkId,
    GlossaryEntry,
    GlossaryTerm,
    Match,
    NormalizedText,
    Question,
    RoleName,
    Similarity,
)

SCORE_DECIMALS = 6           # exact matches must tie at 1.0 despite float errors
TERM_SEPARATOR = ": "        # "Term: definition" in Glossary.md
SYNONYM_SEPARATOR = " / "    # "Resurrected / Reborn: ..." defines several terms at once

POSSESSIVE = re.compile(r"'s\b")
NOT_KEPT = re.compile(r"[^a-z0-9*]+")  # the star is BOTC vocabulary ("Each night*")


def normalize(text: str) -> NormalizedText:
    """Returns the shared canonical form for text, applied both to questions and to entity names.

    Lower case a-z, 0-9 and "*" only, words separated by single spaces: accents are stripped,
    the possessive "'s" is removed (straight or typographic apostrophe), any other character is a space."""
    decomposed = unicodedata.normalize("NFKD", text)
    without_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    lowered = without_accents.lower().replace("’", "'")
    return NormalizedText(" ".join(NOT_KEPT.sub(" ", POSSESSIVE.sub("", lowered)).split()))


def parse_glossary(path: Path) -> list[GlossaryEntry]:
    """Reads Glossary.md and returns the list of all the entries, in the order of the file.

    The term is what precedes the first ": "; a paragraph "A / B: ..." gives one entry per term,
    all with the same text. Headings and blank lines are skipped.
    Raises ValueError on a paragraph without ": ", which means the format of the glossary changed,
    or on a blank term (checked by GlossaryEntry).
    Hypothèse : the separator is ": " rather than ":", so that a colon glued to a word is not a term."""
    entries: list[GlossaryEntry] = []
    for paragraph in path.read_text("utf-8").split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph or paragraph.startswith("#"):
            continue
        if TERM_SEPARATOR not in paragraph:
            raise ValueError(f"Glossary paragraph without 'Term: ' in {path}: {paragraph[:80]!r}")
        terms = paragraph.split(TERM_SEPARATOR, 1)[0]
        entries.extend(GlossaryEntry(term=GlossaryTerm(term.strip()), text=paragraph)
                       for term in terms.split(SYNONYM_SEPARATOR))
    return entries


def role_chunks(chunks: list[dict[str, Any]]) -> dict[RoleName, list[ChunkId]]:
    """To every role, associates the chunks' ids of the role's page.

    `chunks` are the lines of chunks.jsonl; a role is a page whose `character_type` is not None.
    The ids keep the order of the input (the page order), they are not sorted."""
    roles: dict[RoleName, list[ChunkId]] = {}
    for chunk in chunks:
        if chunk["character_type"] is not None:
            roles.setdefault(RoleName(chunk["title"]), []).append(ChunkId(chunk["id"]))
    return roles


class EntityMatcher[N: (RoleName, GlossaryTerm)]:
    """Constructs the TF-IDF vectorizer (n-grams and a max_span_words).
    One EntityMatcher for the roles, on for the glossary: EntityMatcher[RoleName], EntityMatcher[GlossaryTerm].

    Names and segments become TF-IDF vectors of character n-grams, built word by word ("char_wb"),
    with a vocabulary fitted on the normalized names only: n-grams unknown to the names are ignored.
    Hypothèse : TfidfVectorizer with its default settings (smooth idf, l2 norm); the thresholds of the tests rely on it."""

    def __init__(self, names: Sequence[N], ngram: int = 3, max_span_words: int = 3):
        if not names:
            raise ValueError("EntityMatcher needs at least one name")
        if ngram < 1 or max_span_words < 1:
            raise ValueError(f"ngram and max_span_words must be >= 1, got {ngram} and {max_span_words}")
        self._names: list[N] = list(names)
        self._max_span_words = max_span_words
        self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(ngram, ngram), lowercase=False)
        # Sparse matrix (n_names, vocabulary size), rows l2-normalized.
        self._name_vectors: Any = self._vectorizer.fit_transform(  # pyright: ignore[reportUnknownMemberType]
            [normalize(name) for name in self._names])

    def _spans(self, question: Question) -> list[NormalizedText]:
        """Every segment of 1 to max_span_words consecutive words of the normalized question."""
        words = normalize(question).split()
        return [NormalizedText(" ".join(words[start:end]))
                for start in range(len(words))
                for end in range(start + 1, min(start + self._max_span_words, len(words)) + 1)]

    def match(self, question: Question, threshold: Similarity) -> list[Match[N]]:
        """Returns the list of entities detected in the question.
        At most one per span of words, sorted in decreasing order.

        Each segment keeps its best entity if its score is > 0 and >= threshold. An entity found in several
        segments is returned once: best score, then shortest segment. Overlaps are not resolved.
        Sorted by decreasing score, then by name.
        Hypothèse : a segment equally close to two names keeps the first one of `names`; on a full tie
        (same score, same length), the first segment of the question is kept."""
        spans = self._spans(question)
        if not spans:
            return []
        # Shape (n_spans, n_names). A segment without any known n-gram is a zero vector: cosine 0 with every name.
        span_vectors = cast(Any, self._vectorizer.transform(spans))  # pyright: ignore[reportUnknownMemberType]
        scores = cast(NDArray[np.float64], cosine_similarity(span_vectors, self._name_vectors))
        best: dict[N, Match[N]] = {}
        for span, row in zip(spans, scores):
            k = int(row.argmax())
            score = Similarity(round(float(row[k]), SCORE_DECIMALS))
            if score <= 0 or score < threshold:
                continue
            name = self._names[k]
            current = best.get(name)
            if current is None or (score, -len(span.split())) > (current.score, -len(current.span.split())):
                best[name] = Match(name=name, span=span, score=score)
        return sorted(best.values(), key=lambda m: (-m.score, m.name))
