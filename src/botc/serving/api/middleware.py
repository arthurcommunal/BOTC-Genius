"""Latency middleware"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response

CallNext = Callable[[Request], Awaitable[Response]]

UNMATCHED_ROUTE = "<unmatched>"
EXCLUDED_PATHS = frozenset({"/metrics"})


def route_label(request: Request) -> str:
    """Returns the template of the resolved route or `UNMATCHED_ROUTE`."""
    raise NotImplementedError


async def latency_middleware(request: Request, call_next: CallNext) -> Response:
    """Measures the duration of the processing and records it im the metrics."""
    raise NotImplementedError


def install_middleware(app: FastAPI) -> None:
    """Saves `latency_middleware` on the app."""
    raise NotImplementedError
