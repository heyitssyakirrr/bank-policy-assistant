"""Retry policies for every external API (DeepSeek, Gemini, Cohere).

Free tiers return 429 "too many requests" and the occasional 500/503, and
networks time out. Those are temporary: waiting and trying again works.
Anything else (bad API key, bad request) is permanent, and retrying would
just hide the bug.

Two policies:
  retry_transient  offline indexing: 6 attempts, waits up to 60 s (it can afford to wait)
  retry_live       a user is waiting: 3 attempts, ~1 s then ~2 s (+ jitter), capped at 8 s
"""

from __future__ import annotations

import logging

import httpx
import openai
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_exponential_jitter,
)

logger = logging.getLogger("app.retry")

TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
LIVE_ATTEMPTS = 3


def is_transient(exc: BaseException) -> bool:
    """Works for SDKs that put the HTTP status on .code or .status_code, and for
    LangChain's model errors, which say themselves whether a retry can help.
    The OpenAI SDK (DeepSeek) wraps network failures in APIConnectionError
    (APITimeoutError is one), which carries no status."""
    if isinstance(exc, TimeoutError | httpx.TimeoutException | httpx.TransportError | openai.APIConnectionError):
        return True
    if getattr(exc, "is_retryable", False):
        return True
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return status in TRANSIENT_STATUS_CODES


def retry_transient(max_attempts: int = 6):
    """Decorator: retry transient API errors with exponential backoff
    (2s, 4s, 8s ... capped at 60s), logging each retry."""
    return retry(
        retry=retry_if_exception(is_transient),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(max_attempts),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


def retry_live(max_attempts: int = LIVE_ATTEMPTS):
    """Decorator for calls made while a user waits (PROJECT_PLAN §5.5)."""
    return retry(
        retry=retry_if_exception(is_transient),
        wait=wait_exponential_jitter(initial=1, max=8, jitter=0.5),
        stop=stop_after_attempt(max_attempts),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
