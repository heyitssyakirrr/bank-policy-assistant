"""Every agentic graph path, run deterministically with a scripted LLM
(PROJECT_PLAN §8.1). The corpus mirrors real cases from the PDFs: e-KYC
8.10 refers to 8.4, and AML repeats a rule in 14A (banking) and 14B
(insurance/takaful)."""

from langgraph.checkpoint.memory import InMemorySaver

from app.agent.budget import Budget
from app.agent.checkpoint import state_serializer
from app.agent.graph import build_agentic_graph
from app.agent.nodes import UNAVAILABLE_SUMMARY
from app.agent.responses import GREETING, OUT_OF_SCOPE
from app.agent.schemas import EvidenceReview, GroundedAnswer, ParagraphRef, QueryPlan, SearchRequest, Statement
from app.agent.state import AgentContext, turn_input
from app.agent.steps import describe_step
from app.llm.models import LLMUnavailableError
from tests.fakes import FakeSearch, ScriptedLLM, payload

RULE = "may establish non face-to-face business relationships with customers"
CORPUS = {
    "e810": payload(
        "8.10", "G 8.10 In verifying identity through biometric matching as required under paragraph 8.4", provision="G"
    ),
    "e84": payload("8.4", "S 8.4 A financial institution shall verify identity using biometric technology"),
    "a14a": payload("14A.15.1", f"G 14A.15.1 Reporting institutions {RULE}", "amlcft", "G", sector="banking"),
    "a14b": payload("14B.17.1", f"G 14B.17.1 Reporting institutions {RULE}", "amlcft", "G", sector="insurance_takaful"),
    "f82": payload("8.2", "G 8.2 Appendix 1 lists examples of fair treatment outcomes", "ftfc", "G"),
}


def plan(route="search", searches=(), refs=(), question="Q?", clarifying=None, reasoning="why"):
    return QueryPlan(
        route=route,
        standalone_question=question,
        searches=[SearchRequest(**s) if isinstance(s, dict) else SearchRequest(query=s) for s in searches],
        paragraph_refs=[ParagraphRef(document=d, paragraph_id=p) for d, p in refs],
        clarifying_question=clarifying,
        reasoning=reasoning,
    )


def review(sufficient=True, relevant=("E1",), follow_ups=(), cross=()):
    return EvidenceReview(
        relevant_evidence=list(relevant),
        is_sufficient=sufficient,
        missing_information=None if sufficient else "the rule it refers to",
        follow_up_searches=[SearchRequest(query=q) for q in follow_ups],
        cross_references=[ParagraphRef(document=d, paragraph_id=p) for d, p in cross],
    )


def answer(evidence_id="E1", quote="verify identity using biometric technology", obligation="must"):
    statement = Statement(text="t", obligation=obligation, evidence_ids=[evidence_id], quote=quote, topic="x")
    return GroundedAnswer(status="answered", summary="s", statements=[statement])


def run(llm, question="Q?", sector="banking", budget=None, graph=None, thread="t"):
    search = FakeSearch(CORPUS)
    context = AgentContext(llm=llm, search=search, budget=budget or Budget())
    graph = graph or build_agentic_graph()
    config = {"configurable": {"thread_id": thread}}
    return graph.invoke(turn_input(question, sector), config, context=context), search


def test_simple_search_path_costs_three_llm_calls():
    llm = ScriptedLLM(planner=[plan(searches=["technology"])], reviewer=[review(relevant=["E1"])], generator=[answer()])
    state, search = run(llm)
    assert llm.roles() == ["planner", "reviewer", "generator"] and state["llm_calls"] == 3
    assert state["verifications"][-1].passed and state["answer"].status == "answered"
    assert search.searches[0]["query"] == "technology"


def test_comparison_runs_one_filtered_search_per_part_in_parallel():
    searches = [
        {"query": "biometric identity", "documents": ["ekyc"]},
        {"query": "face-to-face", "documents": ["amlcft"]},
    ]
    llm = ScriptedLLM(
        planner=[plan(searches=searches)],
        reviewer=[review(relevant=["E1", "E2", "E3"])],
        generator=[answer(evidence_id="E2")],
    )
    state, search = run(llm)
    assert sorted(s["documents"][0] for s in search.searches) == ["amlcft", "ekyc"]
    assert {e.document for e in state["evidence"]} == {"ekyc", "amlcft"}
    assert len({e.id for e in state["evidence"]}) == len(state["evidence"])  # labels unique after the merge


def test_lookup_path_fetches_the_paragraph_in_the_named_document():
    llm = ScriptedLLM(
        planner=[plan(route="lookup", refs=[("ftfc", "8.2")])],
        reviewer=[review()],
        generator=[answer(quote="lists examples of fair treatment", obligation="information")],
    )
    state, search = run(llm)
    assert [e.chunk_id for e in state["evidence"]] == ["f82"] and state["evidence"][0].source == "lookup"
    assert search.searches == []  # found by number: no search needed


def test_cross_reference_is_followed_before_answering():
    # e-KYC 8.10 says "as required under paragraph 8.4": the reviewer asks for 8.4.
    llm = ScriptedLLM(
        planner=[plan(searches=["matching"])],  # finds only 8.10
        reviewer=[review(sufficient=False, relevant=["E1"], cross=[("ekyc", "8.4")]), review(relevant=["E1", "E2"])],
        generator=[answer(evidence_id="E2")],
    )
    state, search = run(llm, question="What does 8.10 require?")
    assert [e.paragraph_id for e in state["evidence"]] == ["8.10", "8.4"]
    assert search.lookups[0][0].paragraph_id == "8.4"
    assert state["rounds"] == 2 and state["verifications"][-1].passed


def test_review_loop_stops_after_two_rounds():
    always_more = review(sufficient=False, relevant=["E1"], follow_ups=["technology again"])
    llm = ScriptedLLM(
        planner=[plan(searches=["technology"])], reviewer=[always_more, always_more], generator=[answer()]
    )
    state, search = run(llm)
    assert llm.roles() == ["planner", "reviewer", "reviewer", "generator"]
    assert state["rounds"] == 2 and len(search.searches) == 2
    assert any("review round limit" in note for note in state["notes"])


def test_llm_budget_stops_the_loop_early():
    llm = ScriptedLLM(
        planner=[plan(searches=["technology"])],
        reviewer=[review(sufficient=False, relevant=["E1"], follow_ups=["more"])],
        generator=[answer()],
    )
    state, _ = run(llm, budget=Budget(max_llm_calls=3))
    assert llm.roles() == ["planner", "reviewer", "generator"]
    assert any("LLM call budget" in note for note in state["notes"])


def test_out_of_scope_is_refused_before_any_search():
    llm = ScriptedLLM(planner=[plan(route="out_of_scope")])
    state, search = run(llm, question="What is the interest rate cap on credit cards?")
    assert state["outcome"] == "out_of_scope" and state["answer"].summary == OUT_OF_SCOPE
    assert search.searches == [] and state["llm_calls"] == 1


def test_small_talk_gets_a_greeting():
    state, _ = run(ScriptedLLM(planner=[plan(route="small_talk")]), question="hi")
    assert state["outcome"] == "small_talk" and state["answer"].summary == GREETING


def test_ambiguous_paragraph_asks_which_document():
    llm = ScriptedLLM(planner=[plan(route="clarify", clarifying="Which document's para 8.2: e-KYC, FTFC or AML?")])
    state, search = run(llm, question="What does para 8.2 say?")
    assert state["outcome"] == "clarify" and state["answer"].summary.startswith("Which document")
    assert search.searches == [] and search.lookups == []


def test_planner_failure_falls_back_to_searching_the_question():
    llm = ScriptedLLM(planner=[LLMUnavailableError("quota", 2)], reviewer=[review()], generator=[answer()])
    state, search = run(llm, question="technology rules")
    assert search.searches[0]["query"] == "technology rules"
    assert state["answer"].status == "answered" and "Planner unavailable" in state["notes"][0]


def test_reviewer_failure_answers_from_all_evidence():
    llm = ScriptedLLM(
        planner=[plan(searches=["technology"])], reviewer=[LLMUnavailableError("quota", 2)], generator=[answer()]
    )
    state, _ = run(llm)
    assert state["answer"].status == "answered" and any("Reviewer unavailable" in n for n in state["notes"])


def test_paragraph_not_in_the_index_falls_back_to_a_search():
    llm = ScriptedLLM(
        planner=[plan(route="lookup", refs=[("ekyc", "99.9")])], reviewer=[review(sufficient=False, relevant=[])]
    )
    state, search = run(llm)
    assert search.searches[0]["query"] == "Q?" and search.searches[0]["documents"] == ["ekyc"]  # the question
    assert state["search_log"][0]["missing"] == ["ekyc:99.9"]
    assert state["answer"].status == "not_covered"  # nothing relevant: refused without a generator call


def test_sector_named_in_the_question_applies_to_the_whole_answer():
    takaful = {"query": "non face-to-face business relationships", "sector": "insurance_takaful"}
    llm = ScriptedLLM(
        planner=[plan(searches=[takaful])],
        reviewer=[review()],
        generator=[answer(quote=RULE, obligation="should_or_may")],
    )
    state, search = run(llm, sector="banking")
    assert search.searches[0]["sector"] == "insurance_takaful" and state["sector"] == "insurance_takaful"
    assert state["evidence"][0].chunk_id == "a14b"  # the takaful copy, not the banking one
    assert "insurers and takaful operators" in llm.calls[-1][1][-1].content


def test_follow_up_question_sees_the_conversation():
    graph = build_agentic_graph(checkpointer=InMemorySaver(serde=state_serializer()))
    llm = ScriptedLLM(
        planner=[plan(searches=["technology"]), plan(route="small_talk")],
        reviewer=[review()],
        generator=[answer()],
    )
    run(llm, question="Can banks use biometrics?", graph=graph)
    state, _ = run(llm, question="what about takaful operators?", graph=graph)
    second_planner_prompt = llm.calls[-1][1][-1].content
    assert "User: Can banks use biometrics?" in second_planner_prompt and "Assistant: s" in second_planner_prompt
    assert state["llm_calls"] == 1  # counters restarted for the new question


def test_failed_verification_regenerates_in_agentic_mode_too():
    llm = ScriptedLLM(
        planner=[plan(searches=["technology"])],
        reviewer=[review()],
        generator=[answer(quote="words not in the passage"), answer()],
    )
    state, _ = run(llm)
    assert llm.roles() == ["planner", "reviewer", "generator", "generator"] and state["llm_calls"] == 4


def test_every_step_can_be_described():
    llm = ScriptedLLM(
        planner=[plan(searches=["technology", "face-to-face"])],
        reviewer=[review(relevant=["E1"])],
        generator=[answer()],
    )
    graph = build_agentic_graph()
    context = AgentContext(llm=llm, search=FakeSearch(CORPUS))
    lines = [
        line
        for chunk in graph.stream(turn_input("Q?", "banking"), context=context, stream_mode="updates")
        for node, update in chunk.items()
        for line in describe_step(node, update)
    ]
    assert lines[0] == "Understanding question: why" and lines[1] == "Split into 2 searches"
    assert sum(line.startswith("Searched all documents") for line in lines) == 2
    assert lines[-1] == "Citations verified"


def test_generator_reads_the_rewritten_follow_up_question():
    llm = ScriptedLLM(
        planner=[plan(searches=["technology"], question="Can takaful operators use biometric technology?")],
        reviewer=[review()],
        generator=[answer()],
    )
    run(llm, question="what about takaful operators?")
    assert "Question: Can takaful operators use biometric technology?" in llm.calls[-1][1][-1].content


def test_one_failing_search_branch_keeps_the_others_evidence():
    class FlakySearch(FakeSearch):
        def search(self, question, sector, documents=(), paragraph_refs=None):
            if question == "face-to-face":
                raise TimeoutError("qdrant")
            return super().search(question, sector, documents, paragraph_refs)

    llm = ScriptedLLM(
        planner=[plan(searches=["technology", "face-to-face"])], reviewer=[review()], generator=[answer()]
    )
    context = AgentContext(llm=llm, search=FlakySearch(CORPUS))
    state = build_agentic_graph().invoke(turn_input("Q?", "banking"), context=context)
    assert state["answer"].status == "answered" and state["evidence"]
    assert any("failed" in note for note in state["notes"])
    assert any(record.get("error") == "TimeoutError" for record in state["search_log"])


def test_a_search_without_a_sector_keeps_the_users_sector():
    # "How do CDD rules for banks differ from takaful operators?": one side names no sector.
    searches = [{"query": "face-to-face"}, {"query": "face-to-face takaful", "sector": "insurance_takaful"}]
    llm = ScriptedLLM(
        planner=[plan(searches=searches)],
        reviewer=[review()],
        generator=[answer(quote=RULE, obligation="should_or_may")],
    )
    state, search = run(llm, sector="banking")
    assert state["sector"] == "banking"
    assert sorted(s["sector"] for s in search.searches) == ["banking", "insurance_takaful"]


def test_cross_sector_comparison_keeps_both_sectors_copies():
    searches = [
        {"query": "face-to-face", "sector": "banking"},
        {"query": "face-to-face", "sector": "insurance_takaful"},
    ]
    llm = ScriptedLLM(
        planner=[plan(searches=searches)],
        reviewer=[review()],
        generator=[answer(quote=RULE, obligation="should_or_may")],
    )
    state, _ = run(llm)
    assert {"a14a", "a14b"} <= {e.chunk_id for e in state["evidence"]}


def test_search_and_lookup_run_together():
    llm = ScriptedLLM(
        planner=[plan(route="search_and_lookup", searches=["technology"], refs=[("ftfc", "8.2")])],
        reviewer=[review()],
        generator=[answer()],
    )
    state, search = run(llm)
    assert len(search.searches) == 1 and search.lookups[0][0].paragraph_id == "8.2"
    assert {e.source for e in state["evidence"]} == {"search", "lookup"}


def test_generator_down_in_agentic_mode_returns_a_notice():
    llm = ScriptedLLM(
        planner=[plan(searches=["technology"])], reviewer=[review()], generator=[LLMUnavailableError("quota", 2)]
    )
    state, _ = run(llm)
    assert state["answer"].summary == UNAVAILABLE_SUMMARY and state["llm_calls"] == 4  # failed attempts count


def test_time_budget_stops_the_loop():
    more = review(sufficient=False, relevant=["E1"], follow_ups=["face-to-face"])
    llm = ScriptedLLM(planner=[plan(searches=["technology"])], reviewer=[more], generator=[answer()])
    context = AgentContext(llm=llm, search=FakeSearch(CORPUS), clock=lambda: 1000.0)
    state = build_agentic_graph().invoke(turn_input("Q?", "banking", started_at=100.0), context=context)
    assert llm.roles() == ["planner", "reviewer", "generator"]
    assert any("time budget" in note for note in state["notes"])


def test_search_ceiling_holds_with_lookup_fallbacks():
    three = ["technology", "face-to-face", "matching"]
    lookup_more = review(sufficient=False, relevant=["E1"], cross=[("ekyc", "77.7")])
    llm = ScriptedLLM(
        planner=[plan(route="search_and_lookup", searches=three, refs=[("ekyc", "99.9")])],
        reviewer=[lookup_more, review()],
        generator=[answer()],
    )
    state, _ = run(llm)
    assert len(state["search_log"]) <= Budget().max_searches


LINKED_CORPUS = {"e810": CORPUS["e810"], "e84": CORPUS["e84"]}
ONLY_810 = Statement(
    text="May use biometric matching.",
    obligation="should_or_may",
    evidence_ids=["E1"],
    quote="verifying identity through biometric matching",
    topic="8.10",
)


def linked_run(generator_script, budget=None):
    llm = ScriptedLLM(
        planner=[plan(searches=["matching"])],
        reviewer=[review(sufficient=False, relevant=["E1"], cross=[("ekyc", "8.4")]), review(relevant=["E1", "E2"])],
        generator=generator_script,
    )
    context = AgentContext(llm=llm, search=FakeSearch(LINKED_CORPUS), budget=budget or Budget())
    return build_agentic_graph().invoke(turn_input("What does 8.10 require?", "banking"), context=context), llm


def test_an_incomplete_answer_is_regenerated_as_the_fifth_llm_call():
    complete = GroundedAnswer(
        status="answered",
        summary="s",
        statements=[ONLY_810, answer(evidence_id="E2").statements[0]],
    )
    state, llm = linked_run([GroundedAnswer(status="answered", summary="s", statements=[ONLY_810]), complete])
    assert llm.roles() == ["planner", "reviewer", "reviewer", "generator", "generator"] and state["llm_calls"] == 5
    assert state["verifications"][-1].complete and "E2" in state["answer"].statements[1].evidence_ids


def test_a_worse_regeneration_never_replaces_a_correct_first_answer():
    first = GroundedAnswer(status="answered", summary="first", statements=[ONLY_810])
    worse = GroundedAnswer(
        status="answered", summary="worse", statements=[answer(quote="words not in 8.4").statements[0]]
    )
    state, _ = linked_run([first, worse])
    assert state["answer"].summary == "first" and state["answer"].status == "answered"
    assert "Kept the first answer" in state["notes"][-1]


def test_an_incomplete_answer_is_kept_when_the_budget_is_spent():
    first = GroundedAnswer(status="answered", summary="first", statements=[ONLY_810])
    state, llm = linked_run([first], budget=Budget(max_llm_calls=4))
    assert llm.roles().count("generator") == 1 and state["answer"].summary == "first"
