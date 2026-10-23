"""One rate limiter per model, shared by every role and every parallel search.

Gemini's free tier allows 15 requests per minute PER MODEL. The planner and
the reviewer both use gemini-3.1-flash-lite, and the agent runs up to three
searches in parallel threads, so the limiter must be:
  - shared: keyed by model name, not by role;
  - sliding: "at most N calls in ANY 60 s", so no burst at a minute boundary;
  - thread-safe: parallel graph branches acquire it concurrently.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable

logger = logging.getLogger("app.rate_limit")

WINDOW_S = 60.0


class SlidingWindowLimiter:
    def __init__(
        self,
        max_calls: int,
        window_s: float = WINDOW_S,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be at least 1")
        self.max_calls = max_calls
        self._window_s = window_s
        self._clock = clock  # injectable so tests don't really wait
        self._sleep = sleep
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Block until one more call fits in the window, record it, and return
        how long we waited (seconds)."""
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                while self._calls and now - self._calls[0] >= self._window_s:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return waited
                wait = self._window_s - (now - self._calls[0]) + 0.05
            logger.info(f"Rate limit: waiting {wait:.1f}s")
            self._sleep(wait)  # outside the lock, so other threads can check too
            waited += wait


_registry: dict[str, SlidingWindowLimiter] = {}
_registry_lock = threading.Lock()


def shared_limiter(name: str, max_calls_per_minute: int) -> SlidingWindowLimiter:
    """The process-wide limiter for `name` (a model name, or "cohere")."""
    with _registry_lock:
        limiter = _registry.get(name)
        if limiter is None:
            limiter = _registry[name] = SlidingWindowLimiter(max_calls_per_minute)
        return limiter
