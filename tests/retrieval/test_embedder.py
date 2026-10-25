import numpy as np
import pytest

from botc.retrieval.embedder import Embedder, HashEmbedder, SentenceTransformerEmbedder

DIM = 8


class FakeModel:
    """Stands in for SentenceTransformer: records what it is asked to encode."""

    def __init__(self):
        self.seen: list[str] = []

    def get_embedding_dimension(self) -> int:
        return DIM

    def encode(self, texts, **kwargs):
        self.seen.extend(texts)
        vectors = np.random.default_rng(len(self.seen)).standard_normal((len(texts), DIM))
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


# Static check: Pyright / Pylance reports an error here if a class drifts from the Protocol.
_hash: Embedder = HashEmbedder(dim=DIM)
_sentence: Embedder = SentenceTransformerEmbedder("intfloat/e5-small-v2", model=FakeModel())


# Contract tests: the behaviour promised by the Protocol docstring, for every implementation.

@pytest.fixture(params=["hash", "sentence"])
def embedder(request) -> Embedder:
    if request.param == "hash":
        return HashEmbedder(dim=DIM, fixed_cost_s=0, per_item_cost=0)
    return SentenceTransformerEmbedder("intfloat/e5-small-v2", model=FakeModel())


@pytest.fixture(params=["embed", "embed_documents"])
def method(request) -> str:
    return request.param


def test_one_float32_row_per_text(embedder, method):
    vectors = getattr(embedder, method)(["a", "b", "c"])
    assert vectors.shape == (3, embedder.dim)
    assert vectors.dtype == np.float32


def test_rows_are_normalized(embedder, method):
    vectors = getattr(embedder, method)(["a", "b", "c"])
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-5)


def test_empty_input_gives_an_empty_matrix(embedder, method):
    assert getattr(embedder, method)([]).shape == (0, embedder.dim)


def test_a_single_string_is_refused(embedder, method):
    with pytest.raises(TypeError):
        getattr(embedder, method)("not a list")


# Specific to SentenceTransformerEmbedder.

def test_queries_and_documents_get_their_own_prefix():
    model = FakeModel()
    embedder = SentenceTransformerEmbedder("intfloat/e5-small-v2", model=model)
    embedder.embed(["Can the Imp die?"])
    embedder.embed_documents(["The Imp kills each night."])
    assert model.seen == ["query: Can the Imp die?", "passage: The Imp kills each night."]


def test_empty_input_does_not_call_the_model():
    model = FakeModel()
    SentenceTransformerEmbedder("intfloat/e5-small-v2", model=model).embed([])
    assert model.seen == []


def test_unknown_model_is_refused():
    with pytest.raises(ValueError):
        SentenceTransformerEmbedder("some/unknown-model", model=FakeModel())
