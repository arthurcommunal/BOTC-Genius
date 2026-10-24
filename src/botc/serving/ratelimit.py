"""Gives the TokenBucket and RateLimiter classes"""

from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic


class TokenBucket:
    """Each tenant has a token bucket that keeps track of the number of requests that he can succesfully
    send at the moment of the sending."""
    def __init__(self, rate: float, capacity: float, clock: Callable[[], float] = monotonic):
        if capacity <= 0:
            raise ValueError("Capacity must be striclty greater than 0")
        if rate <= 0:
            raise ValueError("Rate must be striclty greater than 0")
        self._rate = rate
        self._capacity = capacity
        self._clock = clock
        self._n_tokens = capacity
        self._last = self._clock()

    @property
    def capacity(self) -> float: return self._capacity

    def tokens(self) -> float:
        now = self._clock()
        delta = max(0, now - self._last)
        self._n_tokens = min(self._capacity, self._n_tokens + delta * self._rate)
        self._last = now
        if self._n_tokens < 0 or self._n_tokens > self._capacity:
            raise ValueError("The number of tokens in this bucket does not respect the contract.")
        return self._n_tokens
    
    def _validate_cost(self, cost: float) -> None:
        if cost <= 0:
            raise ValueError("The cost can't be negative or 0")
        if cost > self._capacity:
            raise ValueError("The cost can't be higher than the capacity")
        
    def try_acquire(self, cost: float = 1.0) -> bool:
        self._validate_cost(cost)
        tokens_now = self.tokens()
        if tokens_now < cost:
            return False
        self._n_tokens -= cost
        return True

    def retry_after(self, cost: float = 1.0) -> float:
        self._validate_cost(cost)
        tokens_now = self.tokens()
        if tokens_now >= cost:
            return 0.0
        return (cost - tokens_now) / self._rate

    def check(self, cost: float = 1.0):
        self._validate_cost(cost)
        tokens_now = self.tokens()
        if cost <= tokens_now:
            allowed = True
            self._n_tokens -= cost
            tokens_now = self._n_tokens
            wait = 0.0
        else:
            allowed = False
            wait = (cost - tokens_now) / self._rate
        return (allowed, wait, tokens_now)


@dataclass(frozen=True)
class Limit:
    capacity: float
    rate: float

@dataclass(frozen=True)
class Decision:
    allowed: bool
    retry_after: float
    remaining: float


class RateLimiter:
    """The rate limiter makes the link between tenants and their token bucket"""

    def __init__(self, default: Limit | None = None, overrides: dict[str, Limit] | None = None, clock: Callable[[], float] | None = monotonic):
        self._default = default
        self._overrides = dict(overrides or {})
        if clock is None:
            self._clock = monotonic
        else:
            self._clock = clock
        self._buckets = {}

    def _get_bucket(self, tenant_id: str) -> TokenBucket:
        bucket = self._buckets.get(tenant_id)
        if bucket is None:
            limit = self._overrides.get(tenant_id, self._default) 
            if limit is None:
                raise KeyError(f"{tenant_id} has no override and default limit is None")
            bucket = TokenBucket(rate=limit.rate, capacity=limit.capacity, clock=self._clock)
            self._buckets[tenant_id] = bucket
        return bucket
    
    def check(self, tenant_id: str, cost: float = 1.0) -> Decision:
        bucket = self._get_bucket(tenant_id)
        a, w, t = bucket.check(cost)
        return Decision(a, w, t)

    def evict(self) -> int:
        to_remove = []
        for tenant_id, bucket in self._buckets.items():
            if bucket.tokens() == bucket.capacity:
                to_remove.append(tenant_id)
        for tenant_id in to_remove:
            self._buckets.pop(tenant_id)
        return len(to_remove)