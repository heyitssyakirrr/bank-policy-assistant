"""The traditional graph with a scripted LLM and an in-memory search."""

from langgraph.checkpoint.memory import InMemorySaver

from app.agent.budget import Budget
from app.agent.checkpoint import state_serializer
from app.agent.nodes import UNAVAILABLE_SUMMARY
from app.agent.schemas import GroundedAnswer, Statement
from app.agent.state import AgentContext, turn_input
from app.agent.traditional import build_traditional_graph
from app.agent.verification import NOT_COVERED_SUMMARY, PARTIAL_SUMMARY_PREFIX
from app.llm.models import LLMUnavailableError
from tests.fakes import FakeSearch, ScriptedLLM, payload

CORPUS = {
    "c13": payload("13", "13. Where FAR exceeds 5%, the financial institution shall notify the Bank.", provision=None),
    "c84": payload("8.4", "S 8.4 A financial institution shall obtain Board approval for e-KYC."),
}
GOOD = Statement(
    text="Notify BNM when FAR exceeds 5%.",
    obligation="must",
    evidence_ids=["E1"],
    quote="shall notify the Bank",
    topic="FAR",
)
BAD_QUOTE = GOOD.model_copy(update={"text": "Stop using e-KYC.", "quote": "must immediately stop using e-KYC"})
QUESTION = "What happens when FAR exceeds 5%?"


def run(llm, question=QUESTION, sector="banking", budget=None, graph=None, thread="t"):
    search = FakeSearch(CORPUS)
    context = AgentContext(llm=llm, search=search, budget=budget or Budget())
    graph = graph or build_traditional_graph()
    state = graph.invoke(turn_input(question, sector), {"configurable": {"thread_id": thread}}, context=context)
    return state, search


def answered(*statements):
    return GroundedAnswer(status="answered", summary="Notify the Bank.", statements=list(statements))


def test_retrieve_generate_verify_in_one_pass():
    llm = ScriptedLLM(generator=[answered(GOOD)])
    state, search = run(llm)
    assert state["verifications"][-1].passed and state["answer"].status == "answered"
    assert state["evidence"][0].paragraph_id == "13"
    assert search.searches == [{"query": QUESTION, "sector": "banking", "documents": []}]
    assert state["llm_calls"] == 1 and state["generations"] == 1
    assert state["search_log"][0]["chunks"][0]["id"] == "c13"
    assert state["messages"][-1].content == "Notify the Bank."  # the reply joins the conversation


def test_failed_verification_regenerates_once_with_feedback():
    llm = ScriptedLLM(generator=[answered(BAD_QUOTE), answered(GOOD)])
    state, _ = run(llm)
    assert state["generations"] == 2 and state["llm_calls"] == 2
    assert [v.passed for v in state["verifications"]] == [False, True]  # both reports kept for the audit
    assert "failed the citation check" in llm.calls[1][1][-1].content


def test_persistent_failure_drops_the_bad_statement_and_refuses():
    llm = ScriptedLLM(generator=[answered(BAD_QUOTE), answered(BAD_QUOTE)])
    state, _ = run(llm)
    assert state["generations"] == 2 and not state["verifications"][-1].passed
    assert state["answer"].status == "not_covered" and state["answer"].summary == NOT_COVERED_SUMMARY


def test_no_regeneration_when_the_llm_budget_is_spent():
    llm = ScriptedLLM(generator=[answered(GOOD, BAD_QUOTE)])
    state, _ = run(llm, budget=Budget(max_llm_calls=1))
    assert state["generations"] == 1 and state["answer"].status == "partially_answered"
    assert state["answer"].summary == PARTIAL_SUMMARY_PREFIX + GOOD.text
    assert state["notes"] == ["Regeneration skipped: request budget reached."]


def test_nothing_retrieved_refuses_without_an_llm_call():
    llm = ScriptedLLM()
    state, _ = run(llm, question="credit card interest cap")
    assert state["answer"].status == "not_covered" and llm.calls == []


def test_sector_none_means_all_sectors():
    llm = ScriptedLLM(generator=[answered(GOOD)])
    _, search = run(llm, sector=None)
    assert search.searches[0]["sector"] is None
    assert "all sectors" in llm.calls[0][1][-1].content


def test_generator_down_returns_a_notice_not_an_exception():
    llm = ScriptedLLM(generator=[LLMUnavailableError("quota", 2)])
    state, _ = run(llm)
    assert state["answer"].summary == UNAVAILABLE_SUMMARY and state["notes"]


def test_generator_down_on_regeneration_keeps_the_verified_part():
    llm = ScriptedLLM(generator=[answered(GOOD, BAD_QUOTE), LLMUnavailableError("quota", 2)])
    state, _ = run(llm)
    assert [s.text for s in state["answer"].statements] == [GOOD.text]
    assert state["answer"].status == "partially_answered"


def test_second_turn_starts_fresh_but_keeps_the_conversation():
    graph = build_traditional_graph(checkpointer=InMemorySaver(serde=state_serializer()))
    llm = ScriptedLLM(generator=[answered(GOOD), answered(BAD_QUOTE), answered(GOOD)])
    run(llm, graph=graph)
    state, _ = run(llm, graph=graph)  # same thread: turn 2 regenerates once, like turn 1 could
    assert state["generations"] == 2 and state["llm_calls"] == 2 and len(state["search_log"]) == 1
    assert [e.id for e in state["evidence"]] == ["E1"]
    assert [m.type for m in state["messages"]] == ["human", "ai", "human", "ai"]


def test_long_input_is_cut():
    assert len(turn_input("x" * 10_000, None)["question"]) == 2000
