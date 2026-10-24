"""Latency middleware"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from time import perf_counter

from fastapi import FastAPI, Request, Response

CallNext = Callable[[Request], Awaitable[Response]]

UNMATCHED_ROUTE = "<unmatched>"
EXCLUDED_PATHS = frozenset({"/metrics"})


def latency_name(route: str, status_code: int) -> str:
    """Returns the name of the latency window."""
    return f"latency:{route}:{status_code // 100}xx"


def route_label(request: Request) -> str:
    """Returns the template of the resolved route or `UNMATCHED_ROUTE`."""
    template = request.scope.get("route")
    if template is None:
        return UNMATCHED_ROUTE
    return template.path



async def latency_middleware(request: Request, call_next: CallNext) -> Response:
    """Measures the duration of the processing and records it im the metrics."""
    if request.url.path in EXCLUDED_PATHS:
        response = await call_next(request)
        return response
    start = perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
    finally:
        duration = (perf_counter() - start)*1000
        label = route_label(request=request)
        request.state.metrics.latency_window(latency_name(route=label, status_code=status)).record(duration)
    response.headers["X-Process-Time-Ms"] = str(duration)
    return response

def install_middleware(app: FastAPI) -> None:
    """Saves `latency_middleware` on the app."""
    app.middleware("http")(latency_middleware)
