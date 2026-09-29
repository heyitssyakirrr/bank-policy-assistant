import time

from google import genai
from google.genai import types

from app.core.config import settings
from app.core.logging import logger

EMBEDDING_MODEL = "gemini-embedding-001"
OUTPUT_DIMENSIONALITY = 768
BATCH_SIZE = 50          # texts per API call — under Gemini's 100/request cap, with margin
TPM_LIMIT = 30_000        # tokens/minute, free tier
SECONDS_PER_MINUTE = 60

_client = genai.Client(api_key=settings.gemini_api_key)


def _estimate_tokens(text: str) -> int:
    """Rough estimate (~0.75 tokens/word) — good enough for pacing
    against the TPM limit, not meant to be exact."""
    return int(len(text.split()) / 0.75)


def embed_batch(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Embed a batch of texts in a single API call."""
    response = _client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            output_dimensionality=OUTPUT_DIMENSIONALITY,
            task_type=task_type,
        ),
    )
    return [e.values for e in response.embeddings]


def embed_all(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Embed many texts: batched under BATCH_SIZE per call, paced to
    stay under the free-tier TPM limit."""
    all_embeddings: list[list[float]] = []
    minute_token_count = 0
    minute_start = time.monotonic()

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        batch_tokens = sum(_estimate_tokens(t) for t in batch)

        if minute_token_count + batch_tokens > TPM_LIMIT:
            elapsed = time.monotonic() - minute_start
            wait = max(0, SECONDS_PER_MINUTE - elapsed) + 1
            logger.info(f"Pausing {wait:.0f}s to stay under the embedding TPM limit")
            time.sleep(wait)
            minute_token_count = 0
            minute_start = time.monotonic()

        embeddings = embed_batch(batch, task_type=task_type)
        all_embeddings.extend(embeddings)
        minute_token_count += batch_tokens
        logger.info(f"Embedded batch {i // BATCH_SIZE + 1}/{-(-len(texts)//BATCH_SIZE)} ({len(batch)} texts)")

    return all_embeddings