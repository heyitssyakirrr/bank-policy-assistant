"""Gemini embeddings (native google-genai SDK: it exposes task_type and
output_dimensionality, which the OpenAI-compatible endpoint doesn't document).

  RETRIEVAL_DOCUMENT  for chunks, at index time
  RETRIEVAL_QUERY     for user questions, at query time
The model embeds the two differently on purpose: a question and the passage
that answers it are phrased differently, and task types bring them closer.
"""

from __future__ import annotations

from collections.abc import Callable

from google import genai
from google.genai import types

from app.core.config import settings
from app.core.logging import logger
from app.core.rate_limit import MinuteRateLimiter
from app.core.retry import retry_transient

EMBEDDING_MODEL = "gemini-embedding-001"
OUTPUT_DIMENSIONALITY = 768
BATCH_SIZE = 50  # texts per API call (Gemini allows up to 100)
# Free-tier quotas. Gemini counts each TEXT in a batch as one request, so
# 100 "requests" per minute means 100 texts per minute, i.e. 2 batches of 50.
TEXTS_PER_MINUTE = 100
TOKENS_PER_MINUTE = 30_000

_client = genai.Client(api_key=settings.gemini_api_key)


def _estimate_tokens(text: str) -> int:
    """Rough estimate (~0.75 words per token): good enough for pacing."""
    return int(len(text.split()) / 0.75)


@retry_transient()
def embed_batch(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Embed a batch of texts in a single API call (retried on 429/5xx)."""
    response = _client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            output_dimensionality=OUTPUT_DIMENSIONALITY,
            task_type=task_type,
        ),
    )
    return [e.values for e in response.embeddings]


def embed_query(text: str) -> list[float]:
    return embed_batch([text], task_type="RETRIEVAL_QUERY")[0]


def embed_all(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    on_batch: Callable[[list[str], list[list[float]]], None] | None = None,
) -> list[list[float]]:
    """Embed many texts in batches, paced to stay under the per-minute quotas.
    `on_batch` is called after every batch so the caller can save progress:
    if the run stops at batch 20 (e.g. the DAILY quota runs out), batches
    1-19 are kept and the next run continues from there."""
    limiter = MinuteRateLimiter(TEXTS_PER_MINUTE, TOKENS_PER_MINUTE)
    all_embeddings: list[list[float]] = []
    total_batches = -(-len(texts) // BATCH_SIZE)

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        limiter.acquire(items=len(batch), tokens=sum(_estimate_tokens(t) for t in batch))
        embeddings = embed_batch(batch, task_type=task_type)
        all_embeddings.extend(embeddings)
        if on_batch:
            on_batch(batch, embeddings)
        logger.info(f"Embedded batch {i // BATCH_SIZE + 1}/{total_batches} ({len(batch)} texts)")

    return all_embeddings
