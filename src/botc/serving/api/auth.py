"""API keys: generation, hash and key registry."""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

KEY_PREFIX = "botc_"

MIN_RATE = 0.0
MIN_CAPACITY = 1

class TenantsJsonLine(BaseModel):
    """Schema for a line of /tenants.json"""

    tenant_id: str
    key_hash: str
    rate: float = Field(gt=MIN_RATE)
    capacity: int = Field(ge=MIN_CAPACITY)
    is_admin: bool = False


@dataclass(frozen=True)
class Tenant:
    """Id and rights for a client.    """

    tenant_id: str
    rate: float
    capacity: int
    is_admin: bool = False


def generate_api_key() -> str:
    """Returns a new key: `KEY_PREFIX` + random secret."""
    secret = secrets.token_urlsafe(32)
    return f"{KEY_PREFIX}{secret}"


def hash_api_key(api_key: str) -> str:
    """Returns the SHA-256 hexadecimal print of the key."""
    key_print = sha256(api_key.encode("utf-8")).hexdigest()
    return key_print


class KeyRegistry:
    """`key hash -> tenant` table, loaded once at the start of the app."""

    def __init__(self, tenants_by_hash: dict[str, Tenant]) -> None:
        """Creates the registry from a dictionnary."""
        self._tenants_by_hash = dict(tenants_by_hash)
        tenants = self._tenants_by_hash.values()
        tenant_ids = [t.tenant_id for t in tenants]
        self._tenant_ids = set(tenant_ids)
        if len(self._tenants_by_hash) != len(self._tenant_ids):
            raise ValueError("At least one tenant has a duplicate in tenants_by_hash")


    @classmethod
    def from_json(cls, path: Path) -> KeyRegistry:
        """Loads the file containing the tenants.
        Then makes the registry.
        
        Expected format is a list of objects with `tenant_id`, `key_hash`, `rate`, `capacity`,
        `is_admin`.
        """
        with open(path, "r", encoding="utf-8") as file:
            registry = json.load(file)

        if not isinstance(registry, list):
            raise TypeError(f"The registry inside {path} is not a list")

        tmp_registry = {}
        ids_added = set()

        for i, row in enumerate(registry):
            try:
                line = TenantsJsonLine.model_validate(row)
            except ValidationError as e:
                raise ValueError(f"row {i+1} of registry at {path} is invalid") from e 
            tenant_id = line.tenant_id
            if tenant_id in ids_added:
                raise ValueError(f"There is a duplicate for {line.tenant_id} in {path} at line {i+1}")
            key_hash = line.key_hash
            if key_hash in tmp_registry:
                raise ValueError(f"There is a duplicate for at least one key_hash in {path} at line {i+1}")
            tenant_args = line.model_dump(exclude={"key_hash"})
            tenant = Tenant(**tenant_args)
            tmp_registry[key_hash] = tenant
            ids_added.add(tenant_id)

        return cls(tmp_registry)


    def get_tenants(self) -> list[Tenant]:
        """Returns all the tenants in the key registry (tenant_id, rate, capacity, is_admin)"""
        return [tenant for tenant in self._tenants_by_hash.values()]
    
    def has_tenant(self, tenant_id: str) -> bool:
        """Return true if the tenant_id is in the registry, otherwise returns false."""
        return tenant_id in self._tenant_ids


    def authenticate(self, api_key: str | None) -> Tenant | None:
        """Returns the tenant associated with the key."""
        if api_key is None or len(api_key) == 0:
            return None
        return self._tenants_by_hash.get(hash_api_key(api_key))
    

    def __len__(self) -> int:
        """Number of registered tenants"""
        return len(self._tenants_by_hash)
