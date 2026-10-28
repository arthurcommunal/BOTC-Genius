import dataclasses
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from botc.retrieval.contracts import (
    Boost,
    ChunkId,
    ConstantBoost,
    DenseScore,
    GlossaryEntry,
    GlossaryTerm,
    GuidedConfig,
    Mass,
    MassFloor,
    Question,
    Retrieval,
    RoleName,
    Similarity,
    Temperature,
    TokenCount,
)
from botc.retrieval.entities import EntityMatcher
from botc.retrieval.fusion import bonus
from botc.retrieval.guided import GuidedRetriever
from botc.retrieval.store import FlatVectorStore, Hit

DIM = 4

# Mini corpus: (id, title, character type, dense score with any question). Every chunk costs 10 words.
CORPUS = [
    ("Glossary#0", "Glossary", None, 0.95),     # the best dense score, yet replaced by the definitions
    ("Abilities#0", "Abilities", None, 0.80),   # a rules page
    ("Abilities#1", "Abilities", None, 0.70),
    ("Poisoner#0", "Poisoner", "Minion", 0.60),
    ("Drunk#0", "Drunk", "Outsider", 0.50),
    ("Poisoner#1", "Poisoner", "Minion", 0.30),
]
DENSE = {ChunkId(id): DenseScore(score) for id, _, _, score in CORPUS}

ALIVE = "Alive: A player who has not died."                           # 7 words
RESURRECTED = "Resurrected / Reborn: A dead player becomes alive again."  # 9 words, two terms
DRUNK = "Drunk: A player whose ability does not work."                 # 8 words
GLOSSARY = [GlossaryEntry(GlossaryTerm("Alive"), ALIVE),
            GlossaryEntry(GlossaryTerm("Resurrected"), RESURRECTED),
            GlossaryEntry(GlossaryTerm("Reborn"), RESURRECTED),
            GlossaryEntry(GlossaryTerm("Drunk"), DRUNK)]

DENSE_ONLY = ConstantBoost(beta=Boost(0.0))
NO_ENTITY = Question("How does voting work during the day?")
ABOUT_POISONER = Question("Can the Poisoner change their mind?")


def chunk(id: str, title: str, character_type: str | None) -> dict[str, Any]:
    """A line of chunks.jsonl: a header of 2 words and a text of 8 words."""
    return {"id": id, "title": title, "character_type": character_type, "header": f"{title} page",
            "text": f"Text of the chunk {id} on this page."}


CHUNKS = [chunk(id, title, character_type) for id, title, character_type, _ in CORPUS]


class FixedQueryEmbedder:
    """Embeds every question as e0, so that the dense score of a chunk is the first coordinate of its vector."""

    @property
    def dim(self) -> int:
        return DIM

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.tile(np.eye(DIM, dtype=np.float32)[0], (len(texts), 1))

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        raise AssertionError("a question must be embedded with embed, not embed_documents")


def store(scores: dict[ChunkId, DenseScore]) -> FlatVectorStore:
    """A store where the cosine of each chunk with e0 is exactly its score: vector (s, sqrt(1 - s^2), 0, 0)."""
    flat = FlatVectorStore(DIM)
    flat.add(list(scores), np.array([[s, math.sqrt(1 - s * s), 0, 0] for s in scores.values()]))
    return flat


class TruncatingStore(FlatVectorStore):
    """A store whose search forgets its last hit, as a store that would cap k."""

    def search(self, query: np.ndarray, k: int) -> list[Hit]:
        return super().search(query, k)[:-1]


def count(text: str) -> TokenCount:
    return TokenCount(len(text.split()))


def retriever(rule: Any = DENSE_ONLY, budget: int = 30, **config: Any) -> GuidedRetriever:
    return GuidedRetriever(CHUNKS, store(DENSE), FixedQueryEmbedder(), GLOSSARY, count,
                           GuidedConfig(rule=rule, budget=TokenCount(budget), **config))


# --- ranking ---

def test_without_entity_the_context_is_the_top_of_the_dense_ranking():
    """No role nor term in the question: no bonus, no definition, and the chunks follow the dense scores.

    Budget 30 = 3 chunks of 10 words; Glossary#0 has the best score but is not a candidate."""
    found = retriever().retrieve(NO_ENTITY)
    assert (found.roles, found.terms, found.definitions, found.bonus) == ((), (), (), 0)
    assert found.chunks == ("Abilities#0", "Abilities#1", "Poisoner#0")


def test_the_chunks_of_the_glossary_page_never_enter_the_context():
    """The glossary page is replaced by the definitions of the detected terms: even with the best dense score and a
    budget for every chunk, Glossary#0 is not taken."""
    found = retriever(budget=100).retrieve(NO_ENTITY)
    assert set(found.chunks) == {"Abilities#0", "Abilities#1", "Poisoner#0", "Drunk#0", "Poisoner#1"}


def test_a_detected_role_brings_its_page_to_the_top():
    """With a bonus of 2 (cosines are in [-1, 1]), the chunks of the page of the detected role come first, in their
    dense order, then the other chunks.

    Poisoner#1 has the lowest dense score but enters the context."""
    found = retriever(rule=ConstantBoost(beta=Boost(2.0))).retrieve(ABOUT_POISONER)
    assert [m.name for m in found.roles] == ["Poisoner"]
    assert found.chunks == ("Poisoner#0", "Poisoner#1", "Abilities#0")


def test_the_bonus_is_the_fusion_bonus_on_the_page_of_the_role_without_the_glossary_page():
    """The reported bonus is fusion.bonus applied to the dense scores of every chunk except the glossary page, with the
    chunks of the page of the detected role boosted.

    Checks the boosted set and that the glossary page is removed before the fusion (it changes the softmax)."""
    rule = MassFloor(alpha=Mass(0.5), tau=Temperature(0.05))
    without_glossary = {id: score for id, score in DENSE.items() if id != "Glossary#0"}
    expected = bonus(without_glossary, {ChunkId("Poisoner#0"), ChunkId("Poisoner#1")}, rule)
    assert expected > 0
    assert retriever(rule=rule).retrieve(ABOUT_POISONER).bonus == pytest.approx(expected, rel=1e-4)


# --- detection ---

def test_detected_entities_are_the_matcher_outputs_at_the_configured_thresholds():
    """Roles and terms are what EntityMatcher returns on the role pages and the glossary terms, at the thresholds of
    the config."""
    question = Question("Can the Poisoner choose an alive player?")
    found = retriever(threshold_roles=Similarity(0.9), threshold_terms=Similarity(0.8)).retrieve(question)
    roles = EntityMatcher([RoleName("Poisoner"), RoleName("Drunk")])
    terms = EntityMatcher([entry.term for entry in GLOSSARY])
    assert found.roles == tuple(roles.match(question, Similarity(0.9)))
    assert found.terms == tuple(terms.match(question, Similarity(0.8)))


def test_a_higher_role_threshold_drops_an_approximate_match():
    """"poisoned" matches the role Poisoner below 1 (about 0.94): kept at threshold 0.9, dropped at 1.0."""
    question = Question("Is the player poisoned?")
    assert [m.name for m in retriever(threshold_roles=Similarity(0.9)).retrieve(question).roles] == ["Poisoner"]
    assert retriever(threshold_roles=Similarity(1.0)).retrieve(question).roles == ()


# --- definitions and budget ---

def test_synonyms_of_one_paragraph_give_one_definition():
    """"resurrected" and "reborn" are two terms of the same glossary paragraph: it enters the context once."""
    found = retriever().retrieve(Question("Can a resurrected player be reborn?"))
    assert [m.name for m in found.terms] == ["Reborn", "Resurrected"]
    assert [entry.text for entry in found.definitions] == [RESURRECTED]


def test_definitions_are_paid_from_the_budget_and_chunks_fill_the_rest():
    """The definitions are always in the context and their cost is taken from the budget; the chunks are the longest
    prefix of the ranking within what is left. tokens counts both.

    Definition of Alive: 7 words; budget 26 leaves 19: one chunk of 10 words, not two."""
    found = retriever(rule=ConstantBoost(beta=Boost(2.0)), budget=26).retrieve(
        Question("Can the Poisoner choose an alive player?"))
    assert [entry.text for entry in found.definitions] == [ALIVE]
    assert found.chunks == ("Poisoner#0",)
    assert found.tokens == 7 + 10


def test_definitions_longer_than_the_budget_leave_no_room_for_chunks():
    """Definitions are never cut: when they alone exceed the budget, they all stay, no chunk is added, and tokens is
    above the budget.

    Alive (7 words) and Drunk (8 words) with a budget of 10."""
    found = retriever(budget=10).retrieve(Question("Is the Drunk alive?"))
    assert [entry.text for entry in found.definitions] == [ALIVE, DRUNK]
    assert (found.chunks, found.tokens) == ((), 15)


def test_tokens_is_the_cost_of_the_definitions_plus_the_text_of_the_chunks():
    """tokens is what the generator receives: the definitions and the chunks as returned by text()."""
    gr = retriever(budget=50)
    found = gr.retrieve(Question("Is the Drunk alive?"))
    assert found.tokens == (sum(count(entry.text) for entry in found.definitions)
                            + sum(count(gr.text(id)) for id in found.chunks))


# --- text ---

def test_the_text_of_a_chunk_is_its_header_a_blank_line_and_its_text():
    """The context shows where a chunk comes from: its header, then its text."""
    assert retriever().text(ChunkId("Poisoner#0")) == "Poisoner page\n\nText of the chunk Poisoner#0 on this page."


def test_the_text_of_an_unknown_chunk_raises():
    with pytest.raises(KeyError):
        retriever().text(ChunkId("Imp#0"))


# --- errors ---

def test_a_retriever_needs_chunks():
    """Without any chunk there is nothing to retrieve: ValueError at construction."""
    with pytest.raises(ValueError):
        GuidedRetriever([], FlatVectorStore(DIM), FixedQueryEmbedder(), GLOSSARY, count,
                        GuidedConfig(rule=DENSE_ONLY))


@pytest.mark.parametrize("scores", [
    {id: s for id, s in DENSE.items() if id != "Drunk#0"},
    {**DENSE, ChunkId("Imp#0"): DenseScore(0.1)},
], ids=["missing chunk", "extra chunk"])
def test_the_store_must_hold_exactly_the_chunks(scores: dict[ChunkId, DenseScore]):
    """The store and the chunks come from the same index: a chunk missing from the store, or a vector without chunk,
    means they do not match. ValueError at construction."""
    with pytest.raises(ValueError):
        GuidedRetriever(CHUNKS, store(scores), FixedQueryEmbedder(), GLOSSARY, count,
                        GuidedConfig(rule=DENSE_ONLY))


def test_a_store_returning_fewer_scores_than_chunks_raises():
    """The bonus needs the dense score of every chunk: a search that returns fewer hits than the store holds
    raises ValueError instead of fusing on part of the index."""
    truncating = TruncatingStore(DIM)
    truncating.add([id for id, *_ in CORPUS], np.array([[s, math.sqrt(1 - s * s), 0, 0] for *_, s in CORPUS]))
    gr = GuidedRetriever(CHUNKS, truncating, FixedQueryEmbedder(), GLOSSARY, count,
                         GuidedConfig(rule=DENSE_ONLY))
    with pytest.raises(ValueError):
        gr.retrieve(NO_ENTITY)


# --- contracts ---

@pytest.mark.parametrize("fields", [
    {"budget": TokenCount(-1)},
    {"threshold_roles": Similarity(1.1)},
    {"threshold_terms": Similarity(-0.1)},
], ids=["negative budget", "role threshold above 1", "negative term threshold"])
def test_an_invalid_config_raises(fields: dict[str, Any]):
    """A config has a budget >= 0 and thresholds in [0, 1]."""
    with pytest.raises(ValueError):
        GuidedConfig(rule=DENSE_ONLY, **fields)


@pytest.mark.parametrize("fields", [
    {"tokens": TokenCount(-1)},
    {"chunks": (ChunkId("Imp#0"), ChunkId("Imp#0"))},
    {"definitions": (GlossaryEntry(GlossaryTerm("Resurrected"), RESURRECTED),
                     GlossaryEntry(GlossaryTerm("Reborn"), RESURRECTED))},
], ids=["negative tokens", "duplicate chunk", "duplicate definition"])
def test_an_invalid_retrieval_raises(fields: dict[str, Any]):
    """A retrieval has tokens >= 0 and puts no chunk nor definition text twice in the context."""
    valid = Retrieval(question=NO_ENTITY, roles=(), terms=(), definitions=(), bonus=Boost(0.0), chunks=(),
                      tokens=TokenCount(0))
    with pytest.raises(ValueError):
        dataclasses.replace(valid, **fields)
