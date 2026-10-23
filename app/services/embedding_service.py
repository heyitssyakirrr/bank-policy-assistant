"""Gemini embeddings (native google-genai SDK).

Two kinds of text get embedded, and the model is told which is which:
  RETRIEVAL_DOCUMENT  chunks, at index time
  RETRIEVAL_QUERY     user questions, at query time
A question and the passage that answers it are phrased differently; telling
the model which side each text is on brings them closer together.

Supported models (set EMBEDDING_MODEL in .env):
  gemini-embedding-001  says "which side" via the task_type parameter.
  gemini-embedding-2    has no task_type: the side goes into the TEXT as a
                        prefix, and each text must be sent as its own Content,
                        otherwise the API merges a list into ONE embedding.
The two models' vectors are incompatible, so each model gets its own cache
file and its own Qdrant collection (the indexer enforces this).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from google import genai
from google.genai import types

from app.core.config import settings
from app.core.logging import logger
from app.core.rate_limit import MinuteRateLimiter
from app.core.retry import retry_live, retry_transient
from app.llm.cache import ResponseCache, cache_key
from app.llm.rate_limit import shared_limiter

EMBEDDING_MODEL = settings.embedding_model
OUTPUT_DIMENSIONALITY = settings.embedding_dim
BATCH_SIZE = 50  # texts per API call (Gemini allows up to 100)
# Free-tier quotas. Gemini counts each TEXT in a batch as one request, so
# 100 "requests" per minute means 100 texts per minute, i.e. 2 batches of 50.
TEXTS_PER_MINUTE = 100
TOKENS_PER_MINUTE = 30_000

DOCUMENT, QUERY = "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"
_V2_PREFIX = {QUERY: "task: search result | query: ", DOCUMENT: "title: none | text: "}

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Created on first use, so importing this module (e.g. to index from the
    committed cache on a machine with no API key) never needs a key."""
    global _client
    if _client is None:
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set (add it to .env)")
        timeout_ms = int(settings.llm_timeout_s * 1000)
        _client = genai.Client(api_key=settings.gemini_api_key, http_options=types.HttpOptions(timeout=timeout_ms))
    return _client


def uses_task_prefixes(model: str) -> bool:
    return model.startswith("gemini-embedding-2")


def build_request(
    texts: list[str], task_type: str, model: str = EMBEDDING_MODEL
) -> tuple[Any, types.EmbedContentConfig]:
    """(contents, config) for one embed_content call. Kept separate from the
    network call so the model differences can be unit-tested."""
    if uses_task_prefixes(model):
        contents = [types.Content(parts=[types.Part(text=_V2_PREFIX[task_type] + text)]) for text in texts]
        return contents, types.EmbedContentConfig(output_dimensionality=OUTPUT_DIMENSIONALITY)
    return texts, types.EmbedContentConfig(output_dimensionality=OUTPUT_DIMENSIONALITY, task_type=task_type)


def _estimate_tokens(text: str) -> int:
    """Rough estimate (~0.75 words per token): good enough for pacing."""
    return int(len(text.split()) / 0.75)


def _embed(texts: list[str], task_type: str) -> list[list[float]]:
    contents, config = build_request(texts, task_type)
    response = _get_client().models.embed_content(model=EMBEDDING_MODEL, contents=contents, config=config)
    vectors = [e.values for e in response.embeddings]
    if len(vectors) != len(texts):  # guards against the "merged into one embedding" trap
        raise RuntimeError(f"Asked for {len(texts)} embeddings, got {len(vectors)}")
    return vectors


@retry_transient()
def embed_batch(texts: list[str], task_type: str = DOCUMENT) -> list[list[float]]:
    """Embed a batch of texts in a single API call (offline: patient retries)."""
    return _embed(texts, task_type)


@retry_live()
def embed_query(text: str) -> list[float]:
    """Embed one user query (live: a user is waiting, so short retries). The
    limiter is shared: the agent's parallel searches embed concurrently."""
    shared_limiter(EMBEDDING_MODEL, TEXTS_PER_MINUTE).acquire()
    return _embed([text], QUERY)[0]


def cached_embed_query(cache: ResponseCache) -> Callable[[str], list[float]]:
    """embed_query, but a query already embedded costs nothing (development
    and evaluation runs only)."""

    def embed(text: str) -> list[float]:
        key = cache_key("embed_query", f"{EMBEDDING_MODEL}-{OUTPUT_DIMENSIONALITY}", text)
        if (vector := cache.get(key)) is None:
            vector = embed_query(text)
            cache.put(key, vector)
        return vector

    return embed


def embed_all(
    texts: list[str],
    task_type: str = DOCUMENT,
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
