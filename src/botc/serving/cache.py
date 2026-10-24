"""LRU Cache: bounded, multi-tenant"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable
from dataclasses import dataclass
from typing import overload

_MISSING = object()


@dataclass(frozen=True)
class CacheStats:
    """Immutable snapshot of the counters."""
    hits: int
    misses: int
    evictions: int
    invalidations: int
    size: int
    capacity: int


class LRUCache[K: Hashable, V]:
    """Bounded LRU cache: each entry belongs to a tenant"""
    
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity is {capacity} whereas it should be > 0")
        self._capacity = capacity
        self._data: OrderedDict[tuple[str, K], V] = OrderedDict()
        self._by_tenant: dict[str, set[K]] = {}
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._invalidations = 0

    @overload
    def get(self, tenant: str, key: K) -> V | None: ...
    @overload
    def get[D](self, tenant: str, key: K, default: D) -> V | D: ... 
    
    def get(self, tenant: str, key: K, default: object = None) -> object:
        """Returns the value and makes it recent.
        On a miss, returns `default`.
        """
        full_key = (tenant, key)
        value = self._data.get(full_key, _MISSING)
        if value is _MISSING:   
            self._misses += 1
            return default
        self._data.move_to_end(full_key)
        self._hits += 1
        return value


    def __contains__(self, item: object) -> bool:
        """To check if `(tenant_key)` is in cache."""
        return item in self._data

    def __len__(self) -> int:
          return len(self._data)

    def keys(self) -> list[tuple[str, K]]:
        return list(self._data)


    def put(self, tenant: str, key: K, value: V) -> tuple[str, K] | None:
        """Inserts or updated then returns None or the evicted key if a key has
        been evicted."""

        full_key = (tenant, key)
        if full_key in self:
            self._data[full_key] = value
            self._data.move_to_end(full_key)
            return None
        
        self._data[full_key] = value
        self._by_tenant.setdefault(tenant, set()).add(key)
        if len(self) > self._capacity:
            return self._evict_oldest()
        return None


    def _forget(self, tenant: str, key: K) -> None:
        """Removes the key of the tenant entry in the by_tenants index. Removes the tenant if its list of keys is empty."""
        keys = self._by_tenant.get(tenant)
        if keys is None:
            return
        keys.remove(key)
        if not keys:
            del self._by_tenant[tenant]

    def _evict_oldest(self) -> tuple[str, K]:
        """Evicts the oldest key (the first one) and returns it"""
        (tenant, key), _ = self._data.popitem(last=False)
        self._forget(tenant, key)
        self._evictions += 1
        return (tenant, key)


    def invalidate_tenant(self, tenant: str) -> int:
        """Removes all the entries of the tenant and returns thier number."""
        keys = self._by_tenant.pop(tenant, set())
        for key in keys:
            del self._data[(tenant, key)]
        self._invalidations += len(keys)
        return len(keys)


    def clear(self) -> None:
        """Empties the content."""
        self._data.clear()
        self._by_tenant.clear()


    def stats(self) -> CacheStats:
         return CacheStats(
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            invalidations=self._invalidations,
            size=len(self),
            capacity=self._capacity)

    def reset_stats(self) -> None:
        """Put stats back to zero."""
        self._hits = self._misses = self._evictions = self._invalidations = 0