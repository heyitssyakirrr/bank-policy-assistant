"""Client-side rate limiting: stay under an API's per-minute quotas instead of
hitting them and waiting on 429 errors.

Gemini's free tier counts EVERY TEXT in a batch call as one request: one call
with 50 texts uses 50 of the 100 requests allowed per minute. So we track
texts and tokens per minute and sleep before a batch that would cross either.
Used by offline indexing (batches); live calls use app/llm/rate_limit.py.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

logger = logging.getLogger("app.indexing.rate_limit")


class MinuteRateLimiter:
    def __init__(
        self,
        max_items_per_minute: int,
        max_tokens_per_minute: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.max_items = max_items_per_minute
        self.max_tokens = max_tokens_per_minute
        self._clock = clock  # injectable so tests don't really wait
        self._sleep = sleep
        self._window_start = clock()
        self._items = 0
        self._tokens = 0

    def acquire(self, items: int, tokens: int) -> None:
        """Block until a call of this size fits in the current minute, then count it."""
        if items > self.max_items or tokens > self.max_tokens:
            raise ValueError("a single call is larger than the per-minute limit; use smaller batches")
        elapsed = self._clock() - self._window_start
        if elapsed >= 60:
            self._reset()
        elif self._items + items > self.max_items or self._tokens + tokens > self.max_tokens:
            wait = 60 - elapsed + 1  # +1s margin: the server's minute and ours aren't perfectly aligned
            logger.info(f"Rate limit: waiting {wait:.0f}s for the next minute window")
            self._sleep(wait)
            self._reset()
        self._items += items
        self._tokens += tokens

    def _reset(self) -> None:
        self._window_start = self._clock()
        self._items = 0
        self._tokens = 0
