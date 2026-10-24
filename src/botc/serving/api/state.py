"""Shared state of the application and the lifespan."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from time import monotonic
from typing import TypedDict

from fastapi import FastAPI

from botc.retrieval.corpus import load_corpus
from botc.retrieval.embedder import HashEmbedder
from botc.retrieval.store import Hit, ShardedVectorStore
from botc.serving.batcher import Batcher
from botc.serving.cache import LRUCache
from botc.serving.metrics import MetricsRegistry
from botc.serving.ratelimit import Limit, RateLimiter

from .auth import KeyRegistry
from .schemas import Section
from .settings import Settings


class AppState(TypedDict):
    """Object exposed by the lifespan in `request.state` at every request."""

    settings: Settings
    registry: KeyRegistry
    buckets: RateLimiter
    cache: LRUCache[tuple[str, int], list[Section]]
    metrics: MetricsRegistry
    batcher: Batcher



def build_state(settings: Settings) -> AppState:
    """Makes all the shared objects, does not start anything."""
    clock = settings.clock if settings.clock is not None else monotonic

    registry = KeyRegistry.from_json(settings.tenants_path)
    tenants = registry.get_tenants()
    overrides = {tenant.tenant_id: Limit(rate=tenant.rate, capacity=tenant.capacity) for tenant in tenants}
    rate_limiter = RateLimiter(overrides=overrides, clock=clock)
    
    cache = LRUCache[tuple[str, int], list[Section]](capacity=settings.cache_capacity)

    embedder = HashEmbedder(dim=settings.embedding_dim)
    store = ShardedVectorStore.create(dim=settings.embedding_dim, num_shards=settings.n_shards)
    corpus = load_corpus(settings.files_path)
    ids, texts = zip(*corpus)
    vectors = embedder.embed(texts)
    store.add(ids=ids, vectors=vectors)
    index_corpus = dict(corpus)

    metrics = MetricsRegistry(clock=clock)
    metrics.gauge(name="queue_size")
    metrics.counter(name="503_errors")
    metrics.counter(name="429_errors")

    def emb_search(items: Sequence[tuple[str, int]]) -> list[list[Hit]]:
        if not items: return []
        questions, ks = zip(*items)
        embeddings = embedder.embed(questions)
        nearest_neighs = [store.search(emb, k) for emb, k in zip(embeddings, ks)]
        return nearest_neighs
  
    async def process_fn(items: Sequence[tuple[str, int]]) -> list[list[Section]]:
        nearest_neighs = await asyncio.to_thread(emb_search, items)
        return [[Section(id=hit.id, text=index_corpus[hit.id], score=hit.score) for hit in row] for row in nearest_neighs]
    
    batcher = Batcher(process_fn=process_fn,
                      max_batch_size=settings.max_batch_size, 
                      max_wait_ms=settings.max_wait_ms, 
                      max_queue_size=settings.max_queue_size)

    return {"settings": settings,
            "registry": registry,
            "buckets": rate_limiter,
            "cache": cache,
            "metrics": metrics,
            "batcher": batcher}

async def start_state(state: AppState) -> None:
    """Start the worker as a background task."""
    batcher = state["batcher"]
    await batcher.start()



async def stop_state(state: AppState) -> None:
    """Stops cleanly the background task of the worker."""
    batcher = state["batcher"]
    await batcher.stop()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[AppState, None]:
    """Lifecycle of the app."""
    settings = app.state.settings
    lifespan_state = build_state(settings)
    await start_state(lifespan_state)
    try:
        yield lifespan_state 
    finally:
        await stop_state(lifespan_state)
