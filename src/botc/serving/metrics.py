"""Metrics for the server."""

from __future__ import annotations

import bisect
import math
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence

Clock = Callable[[], float]

SNAPSHOT_PERCENTILES: tuple[float, ...] = (50.0, 95.0, 99.0)



def _check_p(p: float) -> None:
    if not 0.0 <= p <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {p!r}")


def percentile_sorted(sorted_values: Sequence[float], p: float) -> float | None:
    """Percentile ``p`` of a sorted sequence cmoputed by linear interpolation."""
    _check_p(p)
    n = len(sorted_values)
    if n == 0:
        return None
    h = (n - 1) * (p / 100.0)
    lo = math.floor(h)
    hi = min(lo + 1, n - 1)
    frac = h - lo
    x_lo = sorted_values[lo]
    return x_lo + frac * (sorted_values[hi] - x_lo)


def percentile(values: Iterable[float], p: float) -> float | None:
    """Percentile ``p`` of an iterable."""
    return percentile_sorted(sorted(values), p)


def _p_key(p: float) -> str:
    """50 -> 'p50', 99.9 -> 'p99.9'."""
    return f"p{p:g}"




class Counter:
    """Cumulative counter that only increases (requests, hits, misses, 429...)."""

    __slots__ = ("_value",)

    def __init__(self) -> None:
        self._value: float = 0

    def inc(self, n: float = 1) -> None:
        if n < 0:
            raise ValueError(f"a counter can only increase, got inc({n!r})")
        self._value += n

    @property
    def value(self) -> float:
        return self._value


class Gauge:
    """Snapshot value that increases or decreases (queue depth, cache size...)."""

    __slots__ = ("_value",)

    def __init__(self) -> None:
        self._value: float = 0

    def set(self, v: float) -> None:
        self._value = v

    def inc(self, n: float = 1) -> None:
        self._value += n

    def dec(self, n: float = 1) -> None:
        self._value -= n

    @property
    def value(self) -> float:
        return self._value




class LatencyWindow:
    """Observations of the ``window_s`` last seconds, exact percentiles."""

    def __init__(
        self,
        window_s: float = 60.0,
        clock: Clock = time.monotonic,
        max_samples: int | None = None,
    ) -> None:
        if window_s <= 0:
            raise ValueError(f"window_s must be > 0, got {window_s!r}")
        if max_samples is not None and max_samples <= 0:
            raise ValueError(f"max_samples must be > 0, got {max_samples!r}")
        self._window_s = window_s
        self._clock = clock
        self._max_samples = max_samples
        self._samples: deque[tuple[float, float]] = deque()
        self._dropped = 0

    @property
    def window_s(self) -> float:
        return self._window_s

    @property
    def dropped(self) -> int:
        """Observations removed because of ``max_samples``."""
        return self._dropped

    def _evict(self, now: float) -> None:
        cutoff = now - self._window_s
        samples = self._samples
        # Les éléments périmés sont tous à gauche : on s'arrête au premier valide.
        while samples and samples[0][0] <= cutoff:
            samples.popleft()

    def record(self, value: float) -> None:
        if not math.isfinite(value):
            raise ValueError(f"observation must be finite, got {value!r}")
        now = self._clock()
        self._evict(now)
        if self._max_samples is not None and len(self._samples) >= self._max_samples:
            self._samples.popleft()
            self._dropped += 1
        self._samples.append((now, value))

    def _sorted_values(self) -> list[float]:
        self._evict(self._clock())
        return sorted(v for _, v in self._samples)

    def count(self) -> int:
        self._evict(self._clock())
        return len(self._samples)

    def percentile(self, p: float) -> float | None:
        return percentile_sorted(self._sorted_values(), p)

    def percentiles(self, ps: Iterable[float] = SNAPSHOT_PERCENTILES) -> dict[float, float | None]:
        """Several percentiles with on sorting."""
        values = self._sorted_values()
        return {p: percentile_sorted(values, p) for p in ps}

    def summary(self, ps: Iterable[float] = SNAPSHOT_PERCENTILES) -> dict[str, float | int | None]:
        values = self._sorted_values()
        out: dict[str, float | int | None] = {"count": len(values)}
        for p in ps:
            out[_p_key(p)] = percentile_sorted(values, p)
        out["dropped"] = self._dropped
        out["window_s"] = self._window_s
        return out




class Histogram:
    """Histogram"""

    def __init__(self, bounds: Sequence[float]) -> None:
        b = [float(x) for x in bounds]
        if len(b) < 2:
            raise ValueError("at least two bounds are required")
        if any(not math.isfinite(x) for x in b):
            raise ValueError("bounds must be finite (overflow bucket is implicit)")
        if any(b[i] >= b[i + 1] for i in range(len(b) - 1)):
            raise ValueError("bounds must be strictly increasing")
        self._bounds: tuple[float, ...] = tuple(b)
        self._counts: list[int] = [0] * len(b)
        self._total = 0

    @classmethod
    def exponential(cls, start: float, factor: float, count: int) -> Histogram:
        """Bornes 0, start, start*factor, ..., start*factor**(count-1)."""
        if start <= 0 or factor <= 1 or count < 1:
            raise ValueError("need start > 0, factor > 1, count >= 1")
        return cls([0.0] + [start * factor**i for i in range(count)])

    @property
    def bounds(self) -> tuple[float, ...]:
        return self._bounds

    @property
    def count(self) -> int:
        return self._total

    def bucket_counts(self) -> list[int]:
        return list(self._counts)

    def observe(self, value: float) -> None:
        if not math.isfinite(value):
            raise ValueError(f"observation must be finite, got {value!r}")
        if value < self._bounds[0]:
            raise ValueError(f"value {value!r} is below the lowest bound {self._bounds[0]!r}")
        i = bisect.bisect_right(self._bounds, value) - 1
        self._counts[i] += 1
        self._total += 1

    def quantile(self, p: float) -> float | None:
        _check_p(p)
        if self._total == 0:
            return None
        rank = (p / 100.0) * self._total
        overflow = len(self._counts) - 1
        cumulative = 0
        for i, c in enumerate(self._counts):
            if c and cumulative + c >= rank:
                if i == overflow:
                    return self._bounds[-1]
                lo, hi = self._bounds[i], self._bounds[i + 1]
                return lo + (hi - lo) * (rank - cumulative) / c
            cumulative += c
        return self._bounds[-1] 

    def merge(self, other: Histogram) -> Histogram:
        """New histogram = sum by buckets. Same bounds."""
        if self._bounds != other._bounds:
            raise ValueError("cannot merge histograms with different bounds")
        merged = Histogram(self._bounds)
        merged._counts = [a + b for a, b in zip(self._counts, other._counts)]
        merged._total = self._total + other._total
        return merged

    def summary(self, ps: Iterable[float] = SNAPSHOT_PERCENTILES) -> dict[str, float | int | None]:
        out: dict[str, float | int | None] = {"count": self._total}
        for p in ps:
            out[_p_key(p)] = self.quantile(p)
        return out




class MetricsRegistry:
    """Sole access point to named metrics."""

    def __init__(self, clock: Clock | None = time.monotonic, default_window_s: float = 60.0) -> None:
        if clock is None:
            self._clock = time.monotonic
        else:
            self._clock = clock
        self._default_window_s = default_window_s
        self._counters: dict[str, Counter] = {}
        self._gauges: dict[str, Gauge] = {}
        self._windows: dict[str, LatencyWindow] = {}
        self._histograms: dict[str, Histogram] = {}
        self._ratios: dict[str, tuple[str, tuple[str, ...]]] = {}

    def _check_free(self, name: str, own: dict) -> None:
        for store in (self._counters, self._gauges, self._windows, self._histograms, self._ratios):
            if store is not own and name in store:
                raise ValueError(f"metric name {name!r} is already used by another metric type")

    def counter(self, name: str) -> Counter:
        if name not in self._counters:
            self._check_free(name, self._counters)
            self._counters[name] = Counter()
        return self._counters[name]

    def gauge(self, name: str) -> Gauge:
        if name not in self._gauges:
            self._check_free(name, self._gauges)
            self._gauges[name] = Gauge()
        return self._gauges[name]

    def latency_window(
        self,
        name: str,
        window_s: float | None = None,
        max_samples: int | None = None,
    ) -> LatencyWindow:
        if name not in self._windows:
            self._check_free(name, self._windows)
            self._windows[name] = LatencyWindow(
                window_s=self._default_window_s if window_s is None else window_s,
                clock=self._clock,
                max_samples=max_samples,
            )
        elif window_s is not None and window_s != self._windows[name].window_s:
            raise ValueError(f"window {name!r} already exists with another window_s")
        return self._windows[name]

    def histogram(self, name: str, bounds: Sequence[float]) -> Histogram:
        if name not in self._histograms:
            self._check_free(name, self._histograms)
            self._histograms[name] = Histogram(bounds)
        elif tuple(float(b) for b in bounds) != self._histograms[name].bounds:
            raise ValueError(f"histogram {name!r} already exists with other bounds")
        return self._histograms[name]

    def register_ratio(self, name: str, numerator: str, denominator: Sequence[str]) -> None:
        """Derived ratio : numerator / sum(denominator), computed at the snapshot time.        """
        self._check_free(name, self._ratios)
        self.counter(numerator)
        for d in denominator:
            self.counter(d)
        self._ratios[name] = (numerator, tuple(denominator))

    def _ratio_value(self, name: str) -> float | None:
        num, den = self._ratios[name]
        total = sum(self._counters[d].value for d in den)
        if total == 0:
            return None
        return self._counters[num].value / total

    def snapshot(self, ps: Iterable[float] = SNAPSHOT_PERCENTILES) -> dict:
        ps = tuple(ps)
        return {
            "counters": {k: c.value for k, c in self._counters.items()},
            "gauges": {k: g.value for k, g in self._gauges.items()},
            "ratios": {k: self._ratio_value(k) for k in self._ratios},
            "latency": {k: w.summary(ps) for k, w in self._windows.items()},
            "histograms": {k: h.summary(ps) for k, h in self._histograms.items()},
        }