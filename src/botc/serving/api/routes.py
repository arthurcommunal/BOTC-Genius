"""API endpoints."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter

from botc.serving.batcher import QueueFullError

from .deps import AdminDep, RateLimitedTenantDep, StateDep
from .errors import service_unavailable, tenant_not_found
from .schemas import HealthResponse, InvalidateResponse, QueryRequest, QueryResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(state: StateDep) -> HealthResponse:
    """Health probe."""
    return HealthResponse(status="ok")


@router.post("/v1/query", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    tenant: RateLimitedTenantDep,
    state: StateDep,
) -> QueryResponse:
    """Returns the top-k sections that are the closest to the question."""
    batcher = state["batcher"]
    cache = state["cache"]
    metrics = state["metrics"]

    q = body.question.lower()
    k = body.top_k if body.top_k is not None else state["settings"].top_k
    tenant_id = tenant.tenant_id
    cache_response = cache.get(tenant_id, (q, k))
    if cache_response is not None:
        return QueryResponse(sections=cache_response, cached=True)
    try:
        response = await batcher.submit((q, k))
    except QueueFullError:
        metrics.counter(name="503_errors").inc()
        raise service_unavailable(retry_after_s=1)
    cache.put(tenant=tenant_id,
              key=(q, k),
              value=response)
    return QueryResponse(sections=response, cached=False)


@router.get("/metrics")
async def metrics(state: StateDep) -> dict:
    """Returns the metrics."""
    metrics = state["metrics"]
    batcher = state["batcher"]
    cache = state["cache"]

    metrics.gauge(name="queue_size").set(batcher.qsize())
    snapshot = metrics.snapshot([50.0, 99.0])
    cache_stats = asdict(cache.stats())
    hits, misses = cache_stats["hits"], cache_stats["misses"]
    if hits + misses > 0:
        cache_stats["hit_ratio"] = hits / (hits + misses)
    else:
        cache_stats["hit_ratio"] = None
    snapshot["cache"] = cache_stats
    return snapshot


@router.post("/v1/admin/tenants/{tenant_id}/cache/invalidate", response_model=InvalidateResponse)
async def invalidate_tenant_cache(
    tenant_id: str,
    admin: AdminDep,
    state: StateDep,
) -> InvalidateResponse:
    """Empties the cache entries of a tenant.
    Returns 404 status code if the tenant is unknown.
    """
    cache = state["cache"]
    registry = state["registry"]
    if not registry.has_tenant(tenant_id=tenant_id):
        raise tenant_not_found()
    removed = cache.invalidate_tenant(tenant=tenant_id)
    return InvalidateResponse(tenant_id=tenant_id, removed=removed)
