"""Embedders: shared interface and fake impementation for testing"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from hashlib import sha256
from time import sleep
from typing import Protocol

import numpy as np

DTYPE = np.float32


class Embedder(Protocol):
    """Tranforms text into vector of fixed dimension
    
    - `embed(texts)` returns a np.ndarray of shape (len(texts), dim) and of type float32, line[i] <-> texts[i].
    - Each line is l2-normalized: the embedder guaranties the norm.
    - An empy list returns a np.ndarray of shape (0, dim).
    - Embeddings are mutually indepedent within the batch"""

    @property
    def dim(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> np.ndarray: ... 

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

