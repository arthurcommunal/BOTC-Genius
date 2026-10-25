"""Embedders: shared interface, fake implementation for testing, real sentence-transformers model"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from hashlib import sha256
from time import sleep
from typing import Protocol

import numpy as np

DTYPE = np.float32


class Embedder(Protocol):
    """Tranforms text into vector of fixed dimension

    Retrieval is asymmetric: short questions are compared to longer passages, and a retrieval
    model may encode each side differently. Hence two methods:
    - `embed(texts)`: questions (what the API receives).
    - `embed_documents(texts)`: passages of the corpus (what the index stores).
    A symmetric embedder implements both the same way.

    For both methods:
    - returns a np.ndarray of shape (len(texts), dim) and of type float32, line[i] <-> texts[i].
    - Each line is l2-normalized: the embedder guaranties the norm.
    - An empy list returns a np.ndarray of shape (0, dim).
    - Embeddings are mutually indepedent within the batch"""

    @property
    def dim(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

def seed_from_text(text: str) -> int:
    """Returns a stable 8-bytes seed between processes for a text: SHA-256 on UTF-8 bytes."""
    digest = sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big")


class HashEmbedder:
    """Fake embedder laying on SHA-256 hashing
    - Deterministic: seed = sha256(text) then Numpy generator.
    - Isotropic: normal coordinates then normalized.
    - Latency: a call with n texts rests `fixed_cost_s + n * per_item_cost`.
    The method is atomic and blocking."""

    def __init__(self, dim: int = 384,
                 fixed_cost_s: float = 0.005,
                 per_item_cost: float = 0.0002,
                 sleep: Callable[[float], None] = sleep):
        if dim <= 0:
            raise ValueError(f"dim is {dim} whereas it should be >0")
        if fixed_cost_s < 0 or per_item_cost < 0:
            raise ValueError("Latency costs must be >= 0")
        self._dim = dim
        self._fixed_cost_s = fixed_cost_s
        self._per_item_cost = per_item_cost
        self._sleep = sleep


    @property
    def dim(self) -> int: return self._dim

    def _embed_one(self, text: str) -> np.ndarray:
        seed = seed_from_text(text)
        rng = np.random.default_rng(seed)
        embed = rng.standard_normal(self._dim)
        normalized = (embed / np.linalg.norm(embed)).astype(DTYPE)
        return normalized

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, str):
            raise TypeError("embed takes a Sequence of strings as input, not a string")
        n = len(texts)
        if n == 0:
            return np.empty(shape=(0, self._dim), dtype=DTYPE)
        self._sleep(self._fixed_cost_s + n * self._per_item_cost)
        embeddings = np.empty(shape=(n, self._dim), dtype=DTYPE)
        for i, text in enumerate(texts):
            emb = self._embed_one(text)
            embeddings[i] = emb
        return embeddings

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Symmetric: a passage is embedded exactly like a question."""
        return self.embed(texts)


# model name -> (query prefix, passage prefix), as given on each model card.
MODEL_PREFIXES = {
    "BAAI/bge-small-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "BAAI/bge-base-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "intfloat/e5-small-v2": ("query: ", "passage: "),
    "intfloat/e5-base-v2": ("query: ", "passage: "),
}
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class SentenceTransformerEmbedder:
    """Real embedder: a sentence-transformers model, used asymmetrically.

    Retrieval models are trained on (question, passage) pairs, each side marked by a prefix.
    The same prefixes must be used at inference, otherwise the scores drop.
    `model` lets the tests pass a fake model instead of downloading the real one."""

    def __init__(self, model_name: str = DEFAULT_MODEL, model=None, batch_size: int = 32):
        if model_name not in MODEL_PREFIXES:
            raise ValueError(f"Unknown prefixes for {model_name}, add them to MODEL_PREFIXES")
        self.model_name = model_name
        self.query_prefix, self.passage_prefix = MODEL_PREFIXES[model_name]
        if model is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
        dim = model.get_embedding_dimension()
        if dim is None:  
            raise ValueError(f"{model_name} does not declare its embedding dimension")
        self._model = model
        self._dim = dim
        self._batch_size = batch_size

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.query_prefix)

    def embed_documents(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        return self._encode(texts, self.passage_prefix, show_progress)

    def _encode(self, texts: Sequence[str], prefix: str, show_progress: bool = False) -> np.ndarray:
        if isinstance(texts, str):
            raise TypeError("embed takes a Sequence of strings as input, not a string")
        if len(texts) == 0:
            return np.empty(shape=(0, self.dim), dtype=DTYPE)
        vectors = self._model.encode([prefix + t for t in texts], batch_size=self._batch_size,
                                     normalize_embeddings=True, convert_to_numpy=True,
                                     show_progress_bar=show_progress)
        return vectors.astype(DTYPE)

