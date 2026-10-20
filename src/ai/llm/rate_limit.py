"""Async sliding-window rate limiter shared by provider clients."""

import asyncio
from collections import deque
from time import monotonic

from ai.core.logging import get_logger

logger = get_logger(__name__)


class SlidingWindowRateLimiter:
    """Allow at most `max_units` units per `period_seconds` across concurrent tasks.

    Weighted: acquire(n) reserves n units at once, so a batch of n items
    consumes n units of quota.
    """

    def __init__(
        self,
        *,
        max_units: int,
        period_seconds: float = 60.0,
        name: str = "limiter",
    ) -> None:
        if max_units < 1:
            raise ValueError("max_units must be >= 1")
        if period_seconds <= 0:
            raise ValueError("period_seconds must be > 0")
        self.max_units = max_units
        self.period_seconds = period_seconds
        self.name = name
        self._events: deque[tuple[float, int]] = deque()
        self._used = 0
        self._lock = asyncio.Lock()

    def _evict(self, now: float) -> None:
        cutoff = now - self.period_seconds
        while self._events and self._events[0][0] <= cutoff:
            _, units = self._events.popleft()
            self._used -= units

    def _seconds_until_available(self, now: float, units: int) -> float:
        needed = self._used + units - self.max_units
        freed = 0
        for timestamp, used in self._events:
            freed += used
            if freed >= needed:
                return max(timestamp + self.period_seconds - now, 0.0)
        return self.period_seconds

    async def acquire(self, units: int = 1) -> None:
        if units < 1:
            raise ValueError("units must be >= 1")
        if units > self.max_units:
            raise ValueError(
                f"{self.name}: requested {units} units, limit is {self.max_units}"
            )
        waited = 0.0
        while True:
            async with self._lock:
                now = monotonic()
                self._evict(now)
                if self._used + units <= self.max_units:
                    self._events.append((now, units))
                    self._used += units
                    break
                delay = self._seconds_until_available(now, units)
            delay = max(delay, 0.05)
            waited += delay
            await asyncio.sleep(delay)  # sleep outside the lock
        if waited >= 1.0:
            logger.info(
                "rate_limit.waited",
                limiter=self.name,
                units=units,
                waited_seconds=round(waited, 2),
            )