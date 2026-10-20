"""chunks.jsonl -> Qdrant (dense + BM25 vectors). Offline, run after the pipeline.

    python -m app.ingestion.indexer              # sync Qdrant with chunks.jsonl
    python -m app.ingestion.indexer --recreate   # drop and rebuild the collection

What one run does
-----------------
1. Load the chunks the ingestion pipeline wrote.
2. Embeddings: look each chunk up in the EMBEDDING CACHE by its content_hash.
   Only chunks whose text is new or changed are sent to Gemini. Re-running
   after a small rule change costs a handful of API calls, not 1,200, and a
   wiped Qdrant can be rebuilt without spending any quota.
3. BM25: compute every chunk's sparse vector locally (free, takes a second).
4. Upsert every chunk into Qdrant under its stable chunk_id: existing points
   are overwritten, never duplicated.
5. Delete points whose chunk no longer exists (e.g. a paragraph a rule fix
   merged away), so Qdrant always mirrors chunks.jsonl exactly.

This "sync" design (cache + idempotent upsert + stale delete) is the same
shape production indexing jobs use; only the scale differs.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings
from app.core.logging import setup_logging
from app.db.vector_repository import MODEL_FIELD, StoredPoint, VectorRepository
from app.services.bm25_encoder import Bm25Encoder

logger = logging.getLogger("app.ingestion.indexer")

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CHUNKS_PATH = DATA_DIR / "processed" / "chunks.jsonl"
CACHE_DIR = DATA_DIR / "interim" / "embeddings"

EmbedFn = Callable[[list[str], Callable[[list[str], list[list[float]]], None]], list[list[float]]]


class EmbeddingCache:
    """content_hash -> vector, stored as append-only JSONL.

    One file per model + dimension: switching models can never mix vectors
    from two different embedding spaces. Append-only means a crash mid-run
    loses at most the line being written, never earlier work."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._vectors: dict[str, list[float]] = {}
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                        self._vectors[row["h"]] = row["v"]
                    except (json.JSONDecodeError, KeyError):
                        continue  # a half-written last line from a crash

    def __contains__(self, content_hash: str) -> bool:
        return content_hash in self._vectors

    def __getitem__(self, content_hash: str) -> list[float]:
        return self._vectors[content_hash]

    def add_many(self, hashes: list[str], vectors: list[list[float]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for content_hash, vector in zip(hashes, vectors, strict=True):
                self._vectors[content_hash] = vector
                handle.write(json.dumps({"h": content_hash, "v": vector}) + "\n")


def load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} not found. Run the ingestion pipeline first: python -m app.ingestion.pipeline")
    with path.open(encoding="utf-8") as handle:
        chunks = [json.loads(line) for line in handle if line.strip()]
    if chunks and "citation" not in chunks[0]:
        raise SystemExit(f"{path} was written by an older pipeline. Re-run: python -m app.ingestion.pipeline")
    return chunks


@dataclass(frozen=True, slots=True)
class IndexReport:
    chunks: int
    embedded_now: int
    not_yet_embedded: int  # > 0 only if embedding stopped early; re-run to finish
    stopped_early: str
    reused_from_cache: int
    upserted: int
    deleted_stale: int
    collection_count: int


def index_chunks(
    chunks: list[dict],
    repo: VectorRepository,
    cache: EmbeddingCache,
    embed_fn: EmbedFn,
    dense_size: int,
    embedding_model: str = "",
    recreate: bool = False,
) -> IndexReport:
    # 0. Fail fast: check Qdrant (and that this collection belongs to this
    #    embedding model) BEFORE spending any embedding quota.
    repo.ensure_collection(dense_size, embedding_model=embedding_model, recreate=recreate)

    # 1. Embed only what the cache doesn't have (identical texts are embedded once).
    reused = sum(1 for c in chunks if c["content_hash"] in cache)
    missing = {c["content_hash"]: c["embed_text"] for c in chunks if c["content_hash"] not in cache}
    stopped_early = ""
    if missing:
        logger.info(f"Embedding {len(missing)} new/changed chunks ({reused} cached)")
        hashes, texts = list(missing), list(missing.values())
        done = 0

        def save_batch(batch_texts: list[str], vectors: list[list[float]]) -> None:
            nonlocal done
            cache.add_many(hashes[done : done + len(batch_texts)], vectors)
            done += len(batch_texts)

        try:
            embed_fn(texts, save_batch)
        except Exception as exc:  # e.g. the free tier's DAILY quota ran out
            stopped_early = f"{type(exc).__name__}: {str(exc)[:200]}"
            logger.warning(f"Embedding stopped after {done}/{len(texts)} texts ({stopped_early}). Progress is cached.")
    else:
        logger.info(f"All {len(chunks)} chunks already embedded: no API calls needed")

    # 2. BM25 statistics come from the whole corpus, so recompute them every run.
    bm25 = Bm25Encoder.fit(c["embed_text"] for c in chunks)

    # 3. Upsert every chunk that has an embedding; 4. delete what no longer exists.
    ready = [c for c in chunks if c["content_hash"] in cache]
    points = (
        StoredPoint(
            id=c["chunk_id"],
            dense=cache[c["content_hash"]],
            sparse=bm25.encode_document(c["embed_text"]),
            payload={**c, MODEL_FIELD: embedding_model} if embedding_model else c,
        )
        for c in ready
    )
    upserted = repo.upsert(points)
    stale = sorted(repo.stored_ids() - {c["chunk_id"] for c in chunks})
    repo.delete(stale)

    return IndexReport(
        chunks=len(chunks),
        embedded_now=len(missing) - sum(1 for h in missing if h not in cache),
        not_yet_embedded=len(chunks) - len(ready),
        stopped_early=stopped_early,
        reused_from_cache=reused,
        upserted=upserted,
        deleted_stale=len(stale),
        collection_count=repo.count(),
    )


def main() -> None:
    cli = argparse.ArgumentParser(description="Embed chunks.jsonl and sync it into Qdrant.")
    cli.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    cli.add_argument("--recreate", action="store_true", help="drop and rebuild the Qdrant collection")
    args = cli.parse_args()
    setup_logging(settings.log_level)

    # Imported here so tests can use index_chunks() without the Gemini/Qdrant clients.
    from qdrant_client import QdrantClient

    from app.services import embedding_service as emb

    started = time.perf_counter()
    chunks = load_chunks(args.chunks)
    cache = EmbeddingCache(CACHE_DIR / f"{emb.EMBEDDING_MODEL}-{emb.OUTPUT_DIMENSIONALITY}.jsonl")
    client = QdrantClient(url=settings.qdrant_url)
    try:
        client.get_collections()  # cheap reachability check
    except Exception as exc:
        raise SystemExit(
            f"Can't reach Qdrant at {settings.qdrant_url} ({type(exc).__name__}).\n"
            "Start it with: docker compose up -d   then check "
            f"{settings.qdrant_url}/dashboard opens in a browser."
        ) from exc
    repo = VectorRepository(client, settings.qdrant_collection)

    report = index_chunks(
        chunks,
        repo,
        cache,
        embed_fn=lambda texts, on_batch: emb.embed_all(texts, "RETRIEVAL_DOCUMENT", on_batch=on_batch),
        dense_size=emb.OUTPUT_DIMENSIONALITY,
        embedding_model=emb.EMBEDDING_MODEL,
        recreate=args.recreate,
    )
    if report.not_yet_embedded:
        print(
            f"\nINCOMPLETE: {report.not_yet_embedded} chunks have no embedding yet and are not searchable.\n"
            f"Reason: {report.stopped_early}\n"
            "Progress is saved. If this is the free tier's daily quota, re-run tomorrow: "
            "only the missing chunks will be sent."
        )
    print(
        f"\nchunks: {report.chunks}\n"
        f"embedded now: {report.embedded_now} (Gemini calls: {-(-report.embedded_now // emb.BATCH_SIZE)})\n"
        f"reused from cache: {report.reused_from_cache}\n"
        f"upserted: {report.upserted}\n"
        f"deleted stale: {report.deleted_stale}\n"
        f"points in '{settings.qdrant_collection}': {report.collection_count}\n"
        f"took {time.perf_counter() - started:.0f}s\n"
        f"Browse it: {settings.qdrant_url}/dashboard"
    )


if __name__ == "__main__":
    main()
