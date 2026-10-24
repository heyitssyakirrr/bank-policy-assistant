"""Tests for BM25 encoding and the Qdrant indexer.

They use Qdrant's in-memory mode and a fake embedder, so they need no
Docker, no network and no API key, and still exercise the real Qdrant
client: the same calls the indexer makes against the server.
"""

import gzip
import hashlib
import math

import pytest
from qdrant_client import QdrantClient

from app.db.vector_repository import VectorRepository
from app.ingestion.indexer import EmbeddingCache, index_chunks, uncached_count, write_snapshot
from app.services.bm25_encoder import Bm25Encoder, term_id, tokenize
from tests.test_ingestion import APPENDIX, FTFC_CALLOUT, FTFC_SPEC, MAIN_BODY, SPEC, chunks_for, chunks_for_spec

DIM = 8

# ---------------------------------------------------------------- tokenizer


def test_paragraph_ids_stay_whole():
    assert "14a.9.6" in tokenize("as required under paragraph 14A.9.6 of the policy")
    assert "8.10" in tokenize("G 8.10 In identifying")


def test_compound_terms_are_kept_and_split():
    tokens = tokenize("the AML/CFT/CPF and TFS for FIs policy, through e-KYC")
    assert {"aml/cft/cpf", "aml", "cft", "cpf", "e-kyc", "kyc"} <= set(tokens)


def test_stopwords_dropped_but_modal_verbs_kept():
    tokens = tokenize("The Board shall approve and the institution may consider")
    assert "the" not in tokens and "and" not in tokens
    assert "shall" in tokens and "may" in tokens


def test_light_stemming_matches_plurals():
    assert tokenize("requirements")[0] == tokenize("requirement")[0] == "requirement"
    assert tokenize("business") == ["business"]  # "ss" endings untouched


def test_term_ids_are_stable_across_runs():
    # Python's hash() would differ per process; blake2b must not.
    expected = int.from_bytes(hashlib.blake2b(b"far", digest_size=4).digest(), "big")
    assert term_id("far") == expected


# ---------------------------------------------------------------- BM25 math


def test_bm25_weight_formula():
    encoder = Bm25Encoder(avg_doc_len=2.0)  # doc below has exactly 2 tokens = average length
    vec = encoder.encode_document("liveness detection")
    # tf=1 at average length: 1 * (k1+1) / (1 + k1) = 1.0
    assert all(math.isclose(v, 1.0) for v in vec.values)


def test_repeated_terms_saturate():
    encoder = Bm25Encoder(avg_doc_len=10.0)

    def fraud_weight(text: str) -> float:
        vec = encoder.encode_document(text)
        return dict(zip(vec.indices, vec.values, strict=True))[term_id("fraud")]

    once = fraud_weight("fraud " + "risk " * 9)  # both texts are 10 tokens long
    five_times = fraud_weight("fraud " * 5 + "risk " * 5)
    assert once < five_times < 5 * once  # more mentions help, but with diminishing returns


# ---------------------------------------------------------------- indexer against in-memory Qdrant


def fake_embed(text: str) -> list[float]:
    """Deterministic stand-in for Gemini: same text -> same vector."""
    digest = hashlib.sha256(text.encode()).digest()
    return [b / 255 for b in digest[:DIM]]


class CountingEmbedder:
    def __init__(self):
        self.calls = 0
        self.texts = 0

    def __call__(self, texts, on_batch):
        self.calls += 1
        self.texts += len(texts)
        vectors = [fake_embed(t) for t in texts]
        on_batch(texts, vectors)
        return vectors


@pytest.fixture
def corpus():
    ekyc, _ = chunks_for(MAIN_BODY + APPENDIX)
    ftfc, _ = chunks_for_spec(FTFC_CALLOUT, FTFC_SPEC)
    return [c.to_dict() for c in ekyc + ftfc]


@pytest.fixture
def repo():
    return VectorRepository(QdrantClient(":memory:"), "test_chunks")


def test_index_then_bm25_finds_the_right_chunks(corpus, repo, tmp_path):
    embedder = CountingEmbedder()
    report = index_chunks(corpus, repo, EmbeddingCache(tmp_path / "cache.jsonl"), embedder, DIM)
    assert report.collection_count == len(corpus) == report.upserted

    top = repo.search_sparse(Bm25Encoder.encode_query("FAR thresholds Level 3"), limit=3)[0]
    assert top.payload["chunk_type"] == "table" and "Level 3" in top.payload["text"]

    top = repo.search_sparse(Bm25Encoder.encode_query("paragraph 8.10"), limit=3)[0]
    assert top.payload["paragraph_id"] == "8.10"
    assert top.payload["citation"].startswith(SPEC.title)


def test_second_run_costs_no_api_calls(corpus, repo, tmp_path):
    cache_path = tmp_path / "cache.jsonl"
    first = CountingEmbedder()
    index_chunks(corpus, repo, EmbeddingCache(cache_path), first, DIM)
    second = CountingEmbedder()
    report = index_chunks(corpus, repo, EmbeddingCache(cache_path), second, DIM)  # cache re-read from disk
    assert first.texts > 0 and second.calls == 0
    assert report.embedded_now == 0 and report.collection_count == len(corpus)  # overwritten, not duplicated


def test_removed_chunks_are_deleted_from_qdrant(corpus, repo, tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.jsonl")
    index_chunks(corpus, repo, cache, CountingEmbedder(), DIM)
    report = index_chunks(corpus[:-2], repo, cache, CountingEmbedder(), DIM)
    assert report.deleted_stale == 2 and report.collection_count == len(corpus) - 2


def test_form_templates_are_hidden_from_search_by_default(corpus, repo, tmp_path):
    corpus[0]["tags"] = ["form_template"]
    index_chunks(corpus, repo, EmbeddingCache(tmp_path / "cache.jsonl"), CountingEmbedder(), DIM)
    hidden_id = corpus[0]["chunk_id"]
    query = fake_embed(corpus[0]["embed_text"])  # the exact vector of the hidden chunk
    assert hidden_id not in {h.id for h in repo.search_dense(query, limit=50)}
    assert hidden_id in {h.id for h in repo.search_dense(query, limit=50, exclude_tags=())}


def test_changing_embedding_size_is_refused(corpus, repo, tmp_path):
    index_chunks(corpus, repo, EmbeddingCache(tmp_path / "a.jsonl"), CountingEmbedder(), DIM)
    with pytest.raises(RuntimeError, match="--recreate"):
        repo.ensure_collection(dense_size=DIM * 2)


# ---------------------------------------------------------------- quota handling


class QuotaLimitedEmbedder:
    """Embeds in batches of 3 and 'runs out of daily quota' after `allowed` batches."""

    def __init__(self, allowed: int):
        self.allowed = allowed
        self.texts = 0

    def __call__(self, texts, on_batch):
        out = []
        for start in range(0, len(texts), 3):
            if self.allowed == 0:
                raise RuntimeError("429 RESOURCE_EXHAUSTED (daily quota)")
            self.allowed -= 1
            batch = texts[start : start + 3]
            vectors = [fake_embed(t) for t in batch]
            on_batch(batch, vectors)
            self.texts += len(batch)
            out += vectors
        return out


def test_quota_stop_keeps_progress_and_next_run_finishes(corpus, repo, tmp_path):
    cache_path = tmp_path / "cache.jsonl"
    first = index_chunks(corpus, repo, EmbeddingCache(cache_path), QuotaLimitedEmbedder(allowed=2), DIM)
    assert first.stopped_early and first.not_yet_embedded == len(corpus) - 6
    assert first.collection_count == 6  # what was embedded is already searchable

    resumed = QuotaLimitedEmbedder(allowed=100)
    second = index_chunks(corpus, repo, EmbeddingCache(cache_path), resumed, DIM)
    assert resumed.texts == len(corpus) - 6  # only the missing ones were sent
    assert second.not_yet_embedded == 0 and second.collection_count == len(corpus)


def test_rate_limiter_waits_for_the_next_minute():
    from app.core.rate_limit import MinuteRateLimiter

    now = [0.0]
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    limiter = MinuteRateLimiter(100, 30_000, clock=lambda: now[0], sleep=sleep)
    limiter.acquire(items=50, tokens=1000)
    now[0] += 1
    limiter.acquire(items=50, tokens=1000)  # 100 texts used: still fine
    assert slept == []
    now[0] += 1
    limiter.acquire(items=50, tokens=1000)  # 150 > 100 -> must wait for the window to reset
    assert slept and 58 <= slept[0] <= 60


def test_require_cached_counts_chunks_that_would_need_gemini(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.jsonl")
    cache.add_many(["h1"], [[0.1] * DIM])
    chunks = [{"content_hash": "h1"}, {"content_hash": "h2"}, {"content_hash": "h2"}]
    assert uncached_count(chunks, cache) == 2
    assert uncached_count(chunks[:1], cache) == 0


def test_snapshot_is_a_gzip_copy_docker_can_index_from(tmp_path):
    source = tmp_path / "chunks.jsonl"
    source.write_text('{"chunk": 1}\n', encoding="utf-8")
    target = write_snapshot(source)
    assert target.name == "chunks.jsonl.gz" and gzip.decompress(target.read_bytes()) == source.read_bytes()
