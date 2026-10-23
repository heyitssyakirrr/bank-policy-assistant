"""Tests for reranking and final selection. No Cohere key needed: a fake
reranker and a fake Cohere client stand in for the API."""

from types import SimpleNamespace

import pytest

from app.ingestion.manifest import DocumentSpec
from app.llm.cache import ResponseCache
from app.llm.rate_limit import SlidingWindowLimiter
from app.services import search_service
from app.services.rerank_service import CachedReranker, CohereReranker, NoReranker, RerankScore
from app.services.retrieval_service import ParagraphLookup, RankedChunk, RetrievalResult
from app.services.search_service import SearchService
from app.services.selection import Candidate, collapse_sector_duplicates, select_final

NON_FTF = "Reporting institutions may establish non face-to-face (non-FTF) business relationships with its customers."


def chunk(cid: str, text: str, sector: str | None = None, exact: bool = False, pid: str | None = None) -> RankedChunk:
    payload = {"text": text, "embed_text": f"[header] {text}", "sector": sector, "paragraph_id": pid, "citation": cid}
    return RankedChunk(id=cid, payload=payload, rrf_score=0.0, dense_rank=None, sparse_rank=None, exact_match=exact)


def cands(*chunks: RankedChunk) -> list[Candidate]:
    return [Candidate(c, fused_rank=i) for i, c in enumerate(chunks, start=1)]


# The real case: one rule repeated in AML 14A/14B/14C/14D, plus an e-KYC paragraph.
INSURANCE = chunk("14B.17.1", f"G 14B.17.1 {NON_FTF}", "insurance_takaful", pid="14B.17.1")
MSB = chunk("14C.16.1", f"G 14C.16.1 {NON_FTF}", "money_services", pid="14C.16.1")
BANKING = chunk("14A.15.1", f"G 14A.15.1 {NON_FTF}", "banking", pid="14A.15.1")
EMONEY = chunk("14D.16.1", f"G 14D.16.1 {NON_FTF}", "non_bank_e_money", pid="14D.16.1")
EKYC = chunk("8.26", "G 8.26 A financial institution may also perform e-KYC where verification is conducted by video.")

# ---------------------------------------------------------------- sector duplicates


def test_repeated_rule_collapses_to_the_banking_copy():
    kept, collapsed = collapse_sector_duplicates(cands(INSURANCE, MSB, BANKING, EMONEY, EKYC), "banking")
    assert [c.id for c in kept] == ["14A.15.1", "8.26"]  # one copy, at the best position any copy had
    assert len(collapsed) == 3


def test_preferred_sector_keeps_that_sectors_copy():
    kept, _ = collapse_sector_duplicates(cands(BANKING, INSURANCE, EKYC), "insurance_takaful")
    assert [c.id for c in kept] == ["14B.17.1", "8.26"]


def test_different_rules_in_different_sectors_are_both_kept():
    other = chunk(
        "14B.9.1",
        "S 14B.9.1 Insurers shall verify the identity of the policy owner before issuance.",
        "insurance_takaful",
    )
    kept, collapsed = collapse_sector_duplicates(cands(BANKING, other), "banking")
    assert len(kept) == 2 and not collapsed


def test_collapsing_frees_slots_for_other_sources():
    final = select_final(cands(INSURANCE, MSB, BANKING, EMONEY, EKYC), top_n=2, preferred_sector="banking").final
    assert [c.id for c in final] == ["14A.15.1", "8.26"]  # e-KYC 8.26 now makes the cut


def test_no_preferred_sector_keeps_the_best_ranked_copy_and_demotes_nothing():
    selection = select_final(cands(MSB_ONLY, INSURANCE, BANKING, EKYC), top_n=3, preferred_sector=None)
    assert [c.id for c in selection.final] == ["14C.16.12", "14B.17.1", "8.26"]
    assert not selection.demoted


# ---------------------------------------------------------------- named paragraphs


def test_named_paragraph_is_never_dropped():
    others = [chunk(f"x{i}", f"unrelated text number {i}") for i in range(6)]
    named = chunk("14A.9.6", "14A.9.6 Reporting institutions are required to identify beneficial owners.", exact=True)
    final = select_final(cands(*others, named), top_n=3, preferred_sector="banking").final  # named ranked last
    assert "14A.9.6" in [c.id for c in final] and len(final) == 3


def test_named_insurance_paragraph_beats_the_banking_copy():
    named = chunk("14B.17.1", f"G 14B.17.1 {NON_FTF}", "insurance_takaful", exact=True, pid="14B.17.1")
    kept, _ = collapse_sector_duplicates(cands(BANKING, named), "banking")
    assert [c.id for c in kept] == ["14B.17.1"]  # the user asked for 14B.17.1 by number


# ---------------------------------------------------------------- the search service


class FakeRetrieval:
    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, question, documents=(), paragraph_refs=None):
        self.documents = documents
        return RetrievalResult(question, self.chunks, [], [], [], {"embed": 1.0, "search": 1.0})


class ScoreByKeyword:
    """Fake reranker: a document scores 1.0 if it contains the keyword."""

    def __init__(self, keyword):
        self.keyword = keyword

    def rerank(self, query, documents):
        scores = [RerankScore(i, 1.0 if self.keyword in d else 0.1) for i, d in enumerate(documents)]
        return sorted(scores, key=lambda s: -s.score)


class BrokenReranker:
    def rerank(self, query, documents):
        raise RuntimeError("429 quota exhausted")


def test_reranker_reorders_candidates():
    retrieval = FakeRetrieval([chunk("a", "sampling approach"), chunk("b", "Level 3 notify the Bank")])
    result = SearchService(retrieval, ScoreByKeyword("Level 3"), top_n=2).search("FAR above 5%", sector="banking")
    assert [c.id for c in result.final] == ["b", "a"]
    assert result.final[0].fused_rank == 2 and result.final[0].rerank_score == 1.0  # moved from #2 to #1
    assert result.reranked


def test_reranker_failure_falls_back_to_fused_order():
    retrieval = FakeRetrieval([chunk("a", "first"), chunk("b", "second")])
    result = SearchService(retrieval, BrokenReranker(), top_n=2).search("anything", sector="banking")
    assert not result.reranked and [c.id for c in result.final] == ["a", "b"]


def test_search_passes_the_document_filter_and_records_it():
    retrieval = FakeRetrieval([chunk("a", "first")])
    result = SearchService(retrieval, ScoreByKeyword("x"), top_n=2).search("q", sector=None, documents=["ekyc"])
    assert retrieval.documents == ["ekyc"] and result.documents == ["ekyc"] and result.preferred_sector is None


def test_cached_reranker_calls_the_api_once_per_query(tmp_path):
    calls = []

    class Counting(ScoreByKeyword):
        def rerank(self, query, documents):
            calls.append(query)
            return super().rerank(query, documents)

    reranker = CachedReranker(Counting("b"), ResponseCache(tmp_path / "c.sqlite"), "m")
    first = reranker.rerank("q", ["a", "b"])
    assert reranker.rerank("q", ["a", "b"]) == first and calls == ["q"]


# ---------------------------------------------------------------- Cohere adapter (fake client)


def test_cohere_reranker_sends_all_documents_and_maps_scores():
    calls = []

    class FakeClient:
        def rerank(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                results=[SimpleNamespace(index=1, relevance_score=0.9), SimpleNamespace(index=0, relevance_score=0.2)]
            )

    reranker = CohereReranker(
        api_key="", model="rerank-v4.0-pro", calls_per_minute=10, client=FakeClient(), limiter=SlidingWindowLimiter(10)
    )
    scores = reranker.rerank("q", ["doc a", "doc b"])
    assert scores == [RerankScore(1, 0.9), RerankScore(0, 0.2)]
    assert calls[0]["top_n"] == 2 and calls[0]["model"] == "rerank-v4.0-pro"


def test_missing_cohere_key_fails_clearly():
    with pytest.raises(ValueError, match="COHERE_API_KEY"):
        CohereReranker(api_key="", model="m", calls_per_minute=10)


# ---------------------------------------------------------------- sector tagging at ingestion


def test_sector_comes_from_paragraph_number_or_heading():
    spec = DocumentSpec(code="aml", file="a.pdf", title="AML", sectors={"14A": "banking", "14B": "insurance_takaful"})
    assert spec.sector_for("14A.15.1", ()) == "banking"
    assert spec.sector_for(None, ("PART B", "14B CDD: Insurance and Takaful")) == "insurance_takaful"  # a table
    assert spec.sector_for("10.2.1", ("PART B", "10 Application of Risk-Based Approach")) is None
    assert spec.sector_for("14A.15.1", ()) is not None and spec.sector_for("114A.1", ()) is None


def test_copies_differing_only_in_cross_references_still_collapse():
    # AML 14A.7 vs 14C.8: identical except each cites its own sector's paragraph
    banking = chunk(
        "14A.7", "S 14A.7 Reporting institutions shall comply with paragraph 14A.5 on verification.", "banking"
    )
    msb = chunk(
        "14C.8", "S 14C.8 Reporting institutions shall comply with paragraph 14C.6 on verification.", "money_services"
    )
    kept, _ = collapse_sector_duplicates(cands(msb, banking), "banking")
    assert [c.id for c in kept] == ["14A.7"]


def test_genuinely_different_sector_rules_are_kept_apart():
    # AML 14A.8 vs 14B.9: banking adds "for conducting an occasional transaction"
    base = "Reporting institutions are required to conduct CDD on the customer and the person acting on behalf of the customer"
    banking = chunk(
        "14A.8",
        f"S 14A.8 {base} when establishing business relations or for conducting an occasional transaction.",
        "banking",
    )
    insurer = chunk("14B.9", f"S 14B.9 {base} when establishing business relations.", "insurance_takaful")
    kept, collapsed = collapse_sector_duplicates(cands(banking, insurer), "banking")
    assert len(kept) == 2 and not collapsed


# ---------------------------------------------------------------- other-sector rules rank lower

MSB_ONLY = chunk(
    "14C.16.12", "G 14C.16.12 Money services businesses may verify identity through video calls.", "money_services"
)


def test_rule_for_another_sector_is_ranked_below_rules_for_the_user():
    # The real case: 14C.16.12 (money services only, no banking twin) was reranked #1
    final = select_final(cands(MSB_ONLY, BANKING, EKYC), top_n=3, preferred_sector="banking")
    assert [c.id for c in final.final] == ["14A.15.1", "8.26", "14C.16.12"]  # demoted, not deleted
    assert [c.id for c in final.demoted] == ["14C.16.12"]


def test_preferring_another_sector_keeps_its_rules_on_top():
    final = select_final(cands(MSB_ONLY, EKYC), top_n=2, preferred_sector="money_services")
    assert [c.id for c in final.final] == ["14C.16.12", "8.26"]


def test_named_paragraph_from_another_sector_is_not_demoted():
    named = chunk("14C.16.12", MSB_ONLY.payload["text"], "money_services", exact=True)
    final = select_final(cands(named, EKYC), top_n=2, preferred_sector="banking")
    assert final.final[0].id == "14C.16.12" and not final.demoted


def test_sector_phrase_tags_paragraphs_in_documents_without_sector_parts():
    spec = DocumentSpec(
        code="ekyc",
        file="e.pdf",
        title="e-KYC",
        sector_phrases={"For the money services business sector": "money_services"},
    )
    assert (
        spec.sector_for("8.11", ("PART B",), "S 8.11 For the money services business sector, in meeting requirements")
        == "money_services"
    )
    assert spec.sector_for("8.10", ("PART B",), "G 8.10 In identifying and verifying an individual's identity") is None


def test_search_lookup_returns_named_paragraphs_as_exact_matches():
    class Lookup:
        def lookup(self, refs):
            return [SimpleNamespace(id="x", payload={"paragraph_id": refs[0].paragraph_id})]

    found = SearchService(Lookup(), BrokenReranker()).lookup([ParagraphLookup("ekyc", "8.4")])
    assert [c.id for c in found] == ["x"] and found[0].exact_match


def test_no_cohere_key_means_no_reranking_not_a_crash(monkeypatch):
    monkeypatch.setattr(search_service.settings, "cohere_api_key", "")
    service = search_service.build_search_service()
    assert isinstance(service._reranker, NoReranker)
