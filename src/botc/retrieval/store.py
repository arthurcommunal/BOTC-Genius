from __future__ import annotations

import heapq
from collections.abc import Callable, Iterable, Sequence
from hashlib import sha256
from itertools import islice
from typing import NamedTuple, Protocol

import numpy as np

from botc.retrieval.embedder import DTYPE


class Hit(NamedTuple):
    id: str
    score: float


def sort_key(hit: Hit) -> tuple[float, str]:
    """Global sort key: decreasing by score then increasing by id"""
    return (-hit.score, hit.id)



class VectorStore(Protocol):
    """Index of vectors consultable through k-nearest-neighbors
    - `add(ids, vectors)`: vectors' shape is (len(ids), dim). If an entry is invalid,
    nothing is added.
    - `search(query, k)`: at most k `Hit`, sorted according to `sort_key`. Length is
     min(len(store), k). 
    """

    @property
    def dim(self) -> int: ...

    def __len__(self) -> int: ...

    def __contains__(self, id: object) -> bool: ...

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None: ...

    def search(self, query: np.ndarray, k: int) -> list[Hit]: ...


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("There is a zero vector in the array")
    if not np.isfinite(norms).all():
        raise ValueError("There is non finite vector in the array")
    return (x / norms).astype(DTYPE)


def _validate_batch(ids: Sequence[str], vectors: np.ndarray, dim: int, existing: Callable[[str], bool]) -> np.ndarray:
    vectors = np.asarray(vectors)
    if vectors.ndim != 2 or vectors.shape[1] != dim:
        raise ValueError(f"vectors's shape is {vectors.shape} whereas it should be (n, {dim})")
    if len(ids) != len(vectors):
        raise ValueError(f"Size mismatch between ids ({len(ids)}) and vectors ({len(vectors)})")
    if len(set(ids)) != len(ids):
        raise ValueError("There is at least one duplicate in ids")
    dupes = [i for i in ids if existing(i)]
    if dupes:
        raise ValueError(f"ids already in the vector store: {dupes[:5]}")
    return _normalize_rows(vectors)

class FlatVectorStore:
    """One shard: does a search on a N x dim matrix""" 

    def __init__(self, dim: int, initial_capacity: int = 1024) -> None:
        if dim <= 0:
            raise ValueError(f"dim is {dim} wheras it should be >0")
        if initial_capacity <= 0:
            raise ValueError(f"initial capacity is {initial_capacity} wheras it should be >0")
        self._dim = dim
        self._vectors = np.empty(shape=(initial_capacity, self._dim), dtype=DTYPE)
        self._ids: list[str] = []
        self._ids_set: set[str] = set()


    @property
    def dim(self) -> int: return self._dim

    def __len__(self) -> int: return len(self._ids)

    def __contains__(self, id: object) -> bool: 
        return id in self._ids_set

    def _ensure_capacity(self, needed: int) -> None:
        capacity = self._vectors.shape[0]
        if needed <= capacity:
            return
        while capacity < needed:
            capacity *= 2
        grown = np.empty(shape=(capacity, self._dim), dtype=DTYPE)
        grown[:len(self)] = self._vectors[:len(self)]
        self._vectors = grown

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        ids = list(ids)
        new_vectors = _validate_batch(ids=ids, vectors=vectors, dim=self._dim, existing=self.__contains__)
        needed = len(self) + len(new_vectors)
        self._ensure_capacity(needed)
        self._vectors[len(self) : needed] = new_vectors
        self._ids.extend(ids)
        self._ids_set.update(ids)

    def search(self, query: np.ndarray, k: int) -> list[Hit]:
        if k <= 0:
            raise ValueError(f"k is {k} wheras it should be >0")
        if query.shape[0] != self._dim:
            raise ValueError(f"query's dimension is {query.shape[0]} whereas it should be {self._dim}")
        q = _normalize_rows(np.asarray(query).reshape(1, -1))[0]

        n = len(self)
        if n == 0:
            return []

        scores = (self._vectors[:n] * q).sum(axis=1)
        if k >= n:
            candidates = np.arange(n)
        else:
            kth = n - k
            treshold = scores[np.argpartition(scores, kth=kth)[kth]]
            candidates = np.flatnonzero(scores >= treshold)

        hits = [Hit(self._ids[i], float(scores[i])) for i in candidates]
        hits.sort(key=sort_key)
        return hits[:k]


def shard_index(id: str, num_shards: int) -> int:
    """Routing the document towards the correct shard using sha256(id)"""
    digest = sha256(id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % num_shards

def merge_top_k(hits_list: Iterable[list[Hit]], k: int) -> list[Hit]:
    """Merges the sorted lists for each shard and returns the k hits with the best score"""
    return list(islice(heapq.merge(*hits_list, key=sort_key), k))


class ShardedVectorStore:
    """Container of sharded vector stores"""

    def __init__(self, shards: Sequence[VectorStore]) -> None:
        if not shards:
            raise ValueError("Initialization needs at least one shard")
        dims = {s.dim for s in shards}
        if len(dims) != 1:
            raise ValueError(f"All shards need to have the same dimension, received {dims}")
        self._shards = list(shards)
        self._dim = dims.pop()

    @classmethod
    def create(cls, dim: int, num_shards: int = 2) -> ShardedVectorStore:
        return cls([FlatVectorStore(dim) for _ in range(num_shards)])

    @property
    def dim(self) -> int: return self._dim

    @property
    def num_shards(self) -> int: return len(self._shards)

    def __len__(self) -> int:
        return sum(len(s) for s in self._shards)

    def _shard_for(self, id: str) -> VectorStore:
        return self._shards[shard_index(id, num_shards=len(self._shards))]

    def __contains__(self, id: object) -> bool:
        return isinstance(id, str) and id in self._shard_for(id)

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        ids = list(ids)
        _validate_batch(ids=ids, vectors=vectors, dim=self._dim, existing=self.__contains__)
        groups: dict[int, list[int]] = {}
        for row, id in enumerate(ids):
            groups.setdefault(shard_index(id, len(self._shards)), []).append(row)
        for s_index, rows in groups.items():
            self._shards[s_index].add(ids=[ids[r] for r in rows], vectors=vectors[rows])

    def search(self, query: np.ndarray, k: int):
        if k <= 0:
            raise ValueError(f"k is {k} wheras it should be >0")
        local = [shard.search(query, k) for shard in self._shards]
        return merge_top_k(local, k)