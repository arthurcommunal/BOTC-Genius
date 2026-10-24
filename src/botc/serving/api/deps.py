"""FastAPI dependencies: state, authentification, authorization, quota."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import APIKeyHeader

from .auth import Tenant
from .errors import API_KEY_HEADER_NAME, forbidden, too_many_requests, unauthorized
from .state import AppState

api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


async def get_state(request: Request) -> AppState:
    """Returns the state of the lifespan for the current request."""
    try:
        settings = request.state.settings
        registry = request.state.registry
        buckets = request.state.buckets
        cache = request.state.cache
        metrics = request.state.metrics
        batcher = request.state.batcher
    except AttributeError as e:
        raise RuntimeError("Missing state: has the lifespan been executed ?") from e
    
    return AppState(settings=settings,
                    registry=registry,
                    buckets=buckets,
                    cache=cache,
                    metrics=metrics,
                    batcher=batcher)


StateDep = Annotated[AppState, Depends(get_state)]


async def get_tenant(
    api_key: Annotated[str | None, Depends(api_key_header)],
    state: StateDep,
) -> Tenant:
    """Authenticate the request and returns its tenant."""
    registry = state["registry"]
    tenant = registry.authenticate(api_key=api_key)
    if tenant is None:
        raise unauthorized()
    return tenant




TenantDep = Annotated[Tenant, Depends(get_tenant)]


async def enforce_rate_limit(tenant: TenantDep, state: StateDep) -> Tenant:
    """Consumes a token from the tenant's bucket or raise a 429."""
    buckets = state["buckets"]
    metrics = state["metrics"]
    decision = buckets.check(tenant_id=tenant.tenant_id, cost=1.0)
    if not decision.allowed:
        wait = decision.retry_after
        metrics.counter(name="429_errors").inc()
        raise too_many_requests(retry_after_s=wait)
    return tenant


RateLimitedTenantDep = Annotated[Tenant, Depends(enforce_rate_limit)]


async def require_admin(tenant: TenantDep) -> Tenant:
    """Authorizes only admin tenants otherwise raises a 403."""
    if not tenant.is_admin:
        raise forbidden("Admin rights required.")
    return tenant


AdminDep = Annotated[Tenant, Depends(require_admin)]
