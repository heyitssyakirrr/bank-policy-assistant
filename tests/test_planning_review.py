"""Code checks on the planner's and reviewer's output, and the routing."""

import pytest

from app.agent.evidence import merge_evidence, to_evidence
from app.agent.planning import clean_paragraph_id, recent_history, sanitise_plan
from app.agent.review import sanitise_review
from app.agent.routing import route_after_planning, route_after_review
from app.agent.schemas import EvidenceReview, ParagraphRef, QueryPlan, SearchRequest
from tests.fakes import candidate, payload


def plan(route="search", searches=(), refs=(), clarifying=None, standalone="Full question?"):
    return QueryPlan(
        route=route,
        standalone_question=standalone,
        searches=[SearchRequest(query=q) for q in searches],
        paragraph_refs=[ParagraphRef(document=d, paragraph_id=p) for d, p in refs],
        clarifying_question=clarifying,
        reasoning=" one sentence ",
    )


def test_more_than_three_searches_are_cut_to_three():
    assert len(sanitise_plan(plan(searches=["a", "b", "c", "d", "e"]), "q").searches) == 3


def test_empty_searches_are_dropped_and_replaced_by_the_question():
    cleaned = sanitise_plan(plan(searches=["  "]), "q")
    assert [s.query for s in cleaned.searches] == ["Full question?"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("8.2", "8.2"),
        ("14a.9.6", "14A.9.6"),
        ("para 8.10", "8.10"),
        ("paragraph 13", "13"),
        ("8.4(b)", None),
        ("FAR", None),
    ],
)
def test_paragraph_ids_are_validated(raw, expected):
    assert clean_paragraph_id(raw) == expected


def test_lookup_with_only_malformed_refs_becomes_a_search():
    cleaned = sanitise_plan(plan(route="lookup", refs=[("ekyc", "section four")]), "q")
    assert cleaned.route == "search" and cleaned.paragraph_refs == [] and cleaned.searches


def test_clarify_without_a_question_becomes_a_search():
    assert sanitise_plan(plan(route="clarify", clarifying="  "), "q").route == "search"


def test_refusal_carries_no_searches_or_refs():
    cleaned = sanitise_plan(plan(route="out_of_scope", searches=["x"], refs=[("ekyc", "8.2")]), "q")
    assert cleaned.searches == [] and cleaned.paragraph_refs == []


def test_duplicate_refs_are_merged():
    cleaned = sanitise_plan(plan(route="lookup", refs=[("ekyc", "8.4"), ("ekyc", "para 8.4")]), "q")
    assert len(cleaned.paragraph_refs) == 1


def test_planner_sees_only_the_recent_history_before_the_question():
    messages = list(range(10))
    assert recent_history(messages) == [3, 4, 5, 6, 7, 8]


EVIDENCE = merge_evidence(
    [],
    to_evidence(
        [
            candidate("e810", payload("8.10", "G 8.10 as required under paragraph 8.4", provision="G")),
            candidate("e84", payload("8.4", "S 8.4 shall verify")),
        ],
        "search",
        "q",
    ),
)


def review(**kwargs):
    defaults = {"relevant_evidence": ["E1"], "is_sufficient": False}
    return EvidenceReview(**(defaults | kwargs))


def test_unknown_evidence_ids_are_dropped():
    cleaned = sanitise_review(review(relevant_evidence=["E1", "[E2]", "E7", "8.4"]), EVIDENCE, 4)
    assert cleaned.relevant_evidence == ["E1", "E2"]


def test_sufficient_review_carries_no_follow_ups():
    cleaned = sanitise_review(review(is_sufficient=True, follow_up_searches=[SearchRequest(query="x")]), EVIDENCE, 4)
    assert cleaned.follow_up_searches == [] and cleaned.cross_references == []


def test_sufficient_with_nothing_relevant_keeps_all_evidence():
    assert sanitise_review(review(is_sufficient=True, relevant_evidence=[]), EVIDENCE, 4).relevant_evidence == [
        "E1",
        "E2",
    ]


def test_cross_reference_already_in_the_evidence_is_dropped():
    refs = [ParagraphRef(document="ekyc", paragraph_id="8.4"), ParagraphRef(document="ekyc", paragraph_id="8.6")]
    cleaned = sanitise_review(review(cross_references=refs), EVIDENCE, 4)
    assert [r.paragraph_id for r in cleaned.cross_references] == ["8.6"]


def test_follow_ups_are_capped_by_the_search_budget():
    many = [SearchRequest(query=q) for q in "abc"]
    refs = [ParagraphRef(document="ekyc", paragraph_id="8.6")]
    assert len(sanitise_review(review(follow_up_searches=many), EVIDENCE, 4).follow_up_searches) == 2
    tight = sanitise_review(review(follow_up_searches=many, cross_references=refs), EVIDENCE, 2)
    assert len(tight.follow_up_searches) == 2 and tight.cross_references == []  # a lookup needs 2 slots
    lookup_only = sanitise_review(review(cross_references=refs), EVIDENCE, 2)
    assert [r.paragraph_id for r in lookup_only.cross_references] == ["8.6"]


def test_cross_reference_without_a_document_gets_the_relevant_evidences_document():
    refs = [ParagraphRef(document=None, paragraph_id="8.6")]
    cleaned = sanitise_review(review(relevant_evidence=["E1"], cross_references=refs), EVIDENCE, 4)
    assert cleaned.cross_references[0].document == "ekyc"


def test_same_number_in_another_document_does_not_count_as_present():
    refs = [ParagraphRef(document="ftfc", paragraph_id="8.4")]  # e-KYC 8.4 is present, FTFC 8.4 isn't
    assert sanitise_review(review(cross_references=refs), EVIDENCE, 4).cross_references == refs


def test_routing_after_planning():
    assert route_after_planning({"plan": plan(route="out_of_scope")}) == "respond_directly"
    assert route_after_planning({"plan": plan(route="small_talk")}) == "respond_directly"
    assert route_after_planning({"plan": plan(route="clarify", clarifying="?")}) == "ask_clarification"
    sends = route_after_planning(
        {"plan": plan(route="search_and_lookup", searches=["a", "b"], refs=[("ekyc", "8.4")]), "sector": "banking"}
    )
    assert [s.node for s in sends] == ["search_policies", "search_policies", "lookup_paragraphs"]
    assert sends[0].arg["sector"] == "banking"


def test_routing_after_review():
    current_plan = plan(searches=["a"])
    assert route_after_review({"review": review(is_sufficient=True), "plan": current_plan}) == "generate_answer"
    assert route_after_review({"review": review(), "plan": current_plan}) == "generate_answer"  # nothing to try
    more = review(
        follow_up_searches=[SearchRequest(query="x")],
        cross_references=[ParagraphRef(document="ekyc", paragraph_id="8.6")],
    )
    sends = route_after_review({"review": more, "sector": None, "plan": current_plan})
    assert [s.node for s in sends] == ["search_policies", "lookup_paragraphs"]
    assert sends[1].arg["question"] == "Full question?"


def test_follow_up_search_for_a_paragraph_number_is_dropped():
    searches = [SearchRequest(query="paragraph 8.4"), SearchRequest(query="liveness detection")]
    cleaned = sanitise_review(review(follow_up_searches=searches), EVIDENCE, 4)
    assert [s.query for s in cleaned.follow_up_searches] == ["liveness detection"]
