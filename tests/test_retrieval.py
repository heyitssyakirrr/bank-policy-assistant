"""Tests for hybrid retrieval: RRF maths, paragraph-reference detection, and
the full service on in-memory Qdrant (no Docker, no API key)."""

import math

import pytest
from qdrant_client import QdrantClient

from app.db.vector_repository import SearchHit, VectorRepository
from app.ingestion.indexer import EmbeddingCache, index_chunks
from app.services.embedding_service import build_request
from app.services.retrieval_service import (
    ParagraphLookup,
    RetrievalService,
    find_paragraph_refs,
    named_document,
    named_document_first,
    rrf_fuse,
)
from tests.test_indexing import DIM, CountingEmbedder, fake_embed
from tests.test_ingestion import APPENDIX, MAIN_BODY, chunks_for

# ---------------------------------------------------------------- RRF


def test_rrf_scores_match_the_formula():
    fused = dict(rrf_fuse([["a", "b"], ["b", "c"]], k=60))
    assert math.isclose(fused["b"], 1 / 62 + 1 / 61)  # #2 in list 1, #1 in list 2
    assert math.isclose(fused["a"], 1 / 61)
    assert math.isclose(fused["c"], 1 / 62)


def test_rrf_rewards_agreement_between_lists():
    # The FAR question: the table was #3 dense and #4 BM25, para 5 was #2 in BM25 only.
    dense = ["para13", "para10", "table", "para11", "p821"]
    bm25 = ["para13", "para5", "para4", "table", "p821"]
    order = [item for item, _ in rrf_fuse([dense, bm25])]
    assert order[:3] == ["para13", "table", "p821"]  # in both lists -> on top


def test_rrf_is_deterministic_on_ties():
    first = rrf_fuse([["x", "y"], ["y", "x"]])
    assert first == rrf_fuse([["x", "y"], ["y", "x"]])
    assert [item for item, _ in first] == ["x", "y"]  # equal score and best rank -> by id


# ---------------------------------------------------------------- paragraph references


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("14A.9.6", ["14A.9.6"]),
        ("what does para 8.10 say", ["8.10"]),
        ("compare 14a.9.6 and 14A.9.7", ["14A.9.6", "14A.9.7"]),  # lower case normalised
        ("What must a bank do if FAR exceeds 5%?", []),
        ("a 3.5% threshold", []),  # a percentage, not a paragraph
        ("an RM3.5 million limit", []),  # an amount
        ("8.10 and again 8.10", ["8.10"]),  # no duplicates
    ],
)
def test_find_paragraph_refs(question, expected):
    assert find_paragraph_refs(question) == expected


@pytest.mark.parametrize(
    ("question", "document"),
    [
        ("What does paragraph 9.1 of the Fair Treatment of Financial Consumers policy require?", "ftfc"),
        ("What does e-KYC paragraph 8.21 say?", "ekyc"),
        ("What does paragraph 14A.9.6 of the AML/CFT policy document require?", "amlcft"),
        ("Electronic Know-Your-Customer paragraph 8.4", "ekyc"),
        ("e KYC paragraph 8.4", "ekyc"),
        ("anti money laundering paragraph 14A.9.6", "amlcft"),
        ("Fair-treatment paragraph 9.1", "ftfc"),
        ("What does paragraph 14A.9.6 say about e-KYC onboarding?", "ekyc"),  # a topic: only orders, see below
        ("What does para 8.2 say?", None),  # no document named
        ("Compare e-KYC 8.2 with FTFC 8.2", None),  # two named: don't guess
        ("What does KYC paragraph 8.4 say?", None),  # plain KYC is a practice, not the e-KYC document
        ("What does paragraph 13 require?", None),
    ],
)
def test_the_document_a_question_names(question, document):
    assert named_document(question) == document


def hit(doc_code, paragraph_id):
    return SearchHit(id=f"{doc_code}:{paragraph_id}", score=0.0, payload={"doc_code": doc_code})


def test_the_named_documents_matches_come_first_and_none_are_lost():
    hits = [hit("ekyc", "9.1"), hit("amlcft", "9.1"), hit("ftfc", "9.1")]
    assert [h.id for h in named_document_first(hits, "ftfc")] == ["ftfc:9.1", "ekyc:9.1", "amlcft:9.1"]
    assert named_document_first(hits, None) == hits
    assert named_document_first(hits, "nowhere") == hits  # a wrong guess changes nothing


# ---------------------------------------------------------------- the service on real-shaped chunks


@pytest.fixture
def service(tmp_path):
    chunks, _ = chunks_for(MAIN_BODY + APPENDIX)
    repo = VectorRepository(QdrantClient(":memory:"), "t")
    index_chunks([c.to_dict() for c in chunks], repo, EmbeddingCache(tmp_path / "c.jsonl"), CountingEmbedder(), DIM)
    return RetrievalService(repo, embed_query=fake_embed, candidates=10)


def test_exact_paragraph_is_returned_first(service):
    result = service.retrieve("what is required under paragraph 8.10")
    top = result.chunks[0]
    assert top.exact_match and top.payload["paragraph_id"] == "8.10"
    assert result.paragraph_refs == [ParagraphLookup(None, "8.10")]


def test_a_document_named_as_a_topic_never_loses_the_exact_match(service):
    # Only e-KYC is indexed here: "of the FTFC policy" must not hide e-KYC 8.10.
    result = service.retrieve("What does paragraph 8.10 of the FTFC policy say?")
    top = result.chunks[0]
    assert top.exact_match and top.payload["doc_code"] == "ekyc" and top.payload["paragraph_id"] == "8.10"


def test_paragraph_lookup_respects_the_document(service):
    assert [h.payload["paragraph_id"] for h in service.lookup([ParagraphLookup("ekyc", "8.10")])] == ["8.10"]
    assert service.lookup([ParagraphLookup("ftfc", "8.10")]) == []  # same number, other document: not found


def test_explicit_empty_refs_skip_the_regex_lookup(service):
    result = service.retrieve("what is required under paragraph 8.10", paragraph_refs=[])
    assert not any(c.exact_match for c in result.chunks) and result.paragraph_refs == []


def test_embedding_failure_falls_back_to_bm25(service):
    def broken(_):
        raise TimeoutError("embedding quota")

    service._embed_query = broken
    result = service.retrieve("FAR thresholds Level 3 notify the Bank")
    assert not result.dense_used and result.dense == [] and result.chunks


def test_document_filter_limits_every_search(service):
    assert service.retrieve("FAR thresholds", documents=["ftfc"]).chunks == []
    assert service.retrieve("FAR thresholds", documents=["ekyc"]).chunks


def test_fused_results_have_no_duplicates_and_carry_ranks(service):
    result = service.retrieve("FAR thresholds Level 3 notify the Bank")
    ids = [c.id for c in result.chunks]
    assert len(ids) == len(set(ids))
    assert all(c.dense_rank or c.sparse_rank for c in result.chunks if not c.exact_match)
    # Fake dense vectors are random here, so only BM25's view is meaningful:
    # it ranks the FAR table first, and fusion must keep the table in the candidates.
    table = next(c for c in result.chunks if "Level 3" in c.payload["text"])
    assert table.sparse_rank == 1


def test_both_searches_run_and_are_reported(service):
    result = service.retrieve("liveness detection biometric")
    assert result.dense and result.sparse
    assert set(result.timings_ms) == {"embed", "search"}


def test_question_of_only_stopwords_still_returns_dense_results(service):
    result = service.retrieve("what can you tell me about it")
    assert result.sparse == [] and result.dense  # BM25 has no terms; dense still works


# ---------------------------------------------------------------- embedding model differences


def test_embedding_001_uses_task_type():
    contents, config = build_request(["a", "b"], "RETRIEVAL_QUERY", model="gemini-embedding-001")
    assert contents == ["a", "b"] and config.task_type == "RETRIEVAL_QUERY"


def test_embedding_2_uses_prefixes_and_one_content_per_text():
    contents, config = build_request(["q1", "q2"], "RETRIEVAL_QUERY", model="gemini-embedding-2")
    assert config.task_type is None
    assert len(contents) == 2  # one Content each: otherwise the API merges them into ONE vector
    assert contents[0].parts[0].text == "task: search result | query: q1"
    docs, _ = build_request(["chunk"], "RETRIEVAL_DOCUMENT", model="gemini-embedding-2")
    assert docs[0].parts[0].text == "title: none | text: chunk"


def test_collection_refuses_a_different_embedding_model(tmp_path):
    chunks, _ = chunks_for(MAIN_BODY)
    repo = VectorRepository(QdrantClient(":memory:"), "t")
    rows = [c.to_dict() for c in chunks]
    index_chunks(rows, repo, EmbeddingCache(tmp_path / "a.jsonl"), CountingEmbedder(), DIM, embedding_model="model-a")
    with pytest.raises(RuntimeError, match="QDRANT_COLLECTION"):
        index_chunks(
            rows, repo, EmbeddingCache(tmp_path / "b.jsonl"), CountingEmbedder(), DIM, embedding_model="model-b"
        )
