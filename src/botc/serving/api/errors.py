"""Fabriques d'erreurs HTTP du serveur.

Concepts : Cours 5. Centraliser la construction des erreurs garantit que
chaque code porte les bons en-têtes et un message cohérent.
"""

from __future__ import annotations

import math

from fastapi import HTTPException, status

API_KEY_HEADER_NAME = "X-API-Key"


def unauthorized() -> HTTPException:
    """401 for missing or unkown key."""
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing key",
        headers={"WWW-Authenticate": f'ApiKey header="{API_KEY_HEADER_NAME}"'},
    )


def forbidden(reason: str) -> HTTPException:
    """403: tenants is identified but does not have the sufficient rights."""
    return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=reason)


def too_many_requests(retry_after_s: float) -> HTTPException:
    """429: the tenant's bucket is empty.

    `retry_after_s` est le temps d'attente réel (flottant) avant le prochain
    jeton. Contrat : l'en-tête `Retry-After` est un entier de secondes, jamais
    inférieur au temps réel (sens de l'arrondi ?), et au moins 1.
    """
    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail="Rate limit exceeded for this API key",
        headers={"Retry-After": str(max(1, math.ceil(retry_after_s)))},
    )


def service_unavailable(retry_after_s: int = 1) -> HTTPException:
    """503: the batcher's queue is full."""
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="The server is saturated, please try again later",
        headers={"Retry-After": f"{retry_after_s}"},
    )

def tenant_not_found() -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                         detail="Tenant id not found")
