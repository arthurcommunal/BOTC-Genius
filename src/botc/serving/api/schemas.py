"""Input and output schemas for the API (Pydantic models)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

MIN_QUESTION_LENGTH = 1
MAX_QUESTION_LENGTH = 1000 # Environ 150/200 mots pour limiter la latence du lot
TOP_K_MIN = 1
TOP_K_MAX = 20

class QueryRequest(BaseModel):
    """Body POST /v1/query."""

    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=MIN_QUESTION_LENGTH, max_length=MAX_QUESTION_LENGTH) 
    top_k: int | None = Field(default=None, ge=TOP_K_MIN, le=TOP_K_MAX)


class Section(BaseModel):
    """A section returned by the store : identifiant, texte et score cosinus."""

    id: str
    text: str
    score: float


class QueryResponse(BaseModel):
    """Responses of POST /v1/query."""

    sections: list[Section]
    cached: bool


class InvalidateResponse(BaseModel):
    """Response of a tenant's cache invalidation admin endpoint."""

    tenant_id: str
    removed: int


class HealthResponse(BaseModel):
    """Response of GET /health."""
    status: str
