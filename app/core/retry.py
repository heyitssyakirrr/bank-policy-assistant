"""One retry policy for every external API (Gemini now, Cohere later).

Free tiers return 429 "too many requests" and the occasional 500/503. Those
are temporary: waiting and trying again works. Anything else (bad API key,
bad request) is permanent, and retrying would just hide the bug.
"""

from __future__ import annotations

import logging

from tenacity import before_sleep_log, retry, retry_if_exception, stop_after_attempt, wait_exponential

logger = logging.getLogger("app.retry")

TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def is_transient(exc: BaseException) -> bool:
    """Works for SDKs that put the HTTP status on .code or .status_code."""
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
