"""Evaluation metrics and the golden set's integrity (no LLM, no network)."""

from argparse import Namespace

import pytest

from app.agent.evidence import merge_evidence, to_evidence
from app.agent.schemas import GroundedAnswer, Statement
from app.agent.state import AgentContext
from app.core.config import settings
from app.llm.models import LLMUnavailableError, quota_name, role_models
from app.services.rerank_service import RerankScore
from app.services.search_service import SearchParts
from evals.answer_eval import TurnResult, evaluate_answers
from evals.golden import GoldenItem, chunk_texts, load_golden, missing_labels, paragraph_key
from evals.judge import AnswerJudgement, JudgeMismatchError, StatementVerdict, agreement, judge_answer, verdicts
from evals.metrics import (
    coverage,
    faithfulness,
    hit_at_k,
    mean,
    observed_outcome,
    percentile,
    recall_at_k,
    reciprocal_rank,
)
from evals.report import build_report, mode_summary
from evals.retrieval_eval import STAGES, evaluate_retrieval, score
from evals.run import estimate, judge_turns, select_items, stratified
from tests.fakes import FakeSearch, ScriptedLLM, candidate, payload

RANKED = ["ekyc:8.10", "ekyc:8.10", "amlcft:14A.15.1", "ekyc:8.4", "ftfc:9.1", "ekyc:8.2", "ekyc:8.26"]


def test_hit_recall_and_mrr():
    gold = ["ekyc:8.4", "ekyc:8.26"]
    assert hit_at_k(RANKED, gold, k=5) == 1.0 and hit_at_k(RANKED, ["ftfc:15.4"]) == 0.0
    # duplicate chunks of 8.10 count once, so 8.26 (rank 6 after dedupe) is outside the top 5
    assert recall_at_k(RANKED, gold, k=5) == 0.5 and recall_at_k(RANKED, gold, k=6) == 1.0
    assert reciprocal_rank(RANKED, gold) == pytest.approx(1 / 3)
    assert reciprocal_rank(RANKED, ["nowhere"]) == 0.0


def test_coverage_counts_each_gold_paragraph_once():
    assert coverage(["ekyc:8.4"], ["ekyc:8.4", "ekyc:8.4", "ekyc:8.10"]) == 0.5
    assert coverage(["x"], []) == 0.0


@pytest.mark.parametrize(
    ("outcome", "status", "observed"),
    [
        ("answer", "answered", "answer"),
        ("answer", "partially_answered", "answer"),
        ("answer", "not_covered", "refuse"),
        ("out_of_scope", "not_covered", "refuse"),
        ("clarify", "not_covered", "clarify"),
        ("answer", "error", "error"),  # a crash is never a correct refusal
        ("answer", "cancelled", "error"),
    ],
)
def test_observed_outcome(outcome, status, observed):
    assert observed_outcome(outcome, status) == observed


def test_an_outage_answer_is_an_error_not_a_refusal():
    assert observed_outcome("answer", "not_covered", degraded=True) == "error"


def test_faithfulness_mean_and_percentile():
    assert faithfulness(["yes", "partly", "no", "yes"]) == 0.625 and faithfulness([]) is None
    assert mean([1.0, None, 0.0]) == 0.5 and mean([None]) is None
    assert percentile([1, 2, 3, 4, 100], 50) == 3 and percentile([1, 2, 3, 4, 100], 95) == 100
    assert percentile([], 50) is None


def test_paragraph_keys_distinguish_appendices():
    assert paragraph_key({"doc_code": "ekyc", "scope": "main", "paragraph_id": "8.4"}) == "ekyc:8.4"
    assert paragraph_key({"doc_code": "ekyc", "scope": "Appendix 2", "paragraph_id": "13"}) == "ekyc:Appendix 2:13"


def test_every_golden_label_exists_in_the_chunks():
    items = load_golden()
    assert len(items) >= 40 and {i.category for i in items} >= {"comparison", "follow_up", "ambiguous"}
    assert missing_labels(items, chunk_texts()) == []


def test_unanswerable_items_have_no_gold_paragraphs():
    for item in load_golden():
        assert bool(item.expected_paragraphs) == item.answerable, item.id


def test_missing_labels_are_reported():
    item = GoldenItem(
        id="x", category="paraphrased", question="q", expected_paragraphs=["ekyc:99.9"], expected_outcome="answer"
    )
    assert missing_labels([item], {"ekyc:8.4": ["t"]}) == ["x: ekyc:99.9"]


# ---------------------------------------------------------------- judge, report, estimate


def turn(item_id="fact-6", mode="agentic", correct=True, recall=0.5, cached=False, **extra):
    base = {
        "item_id": item_id,
        "category": "paraphrased",
        "mode": mode,
        "conversation": "User: q",
        "standalone_question": "q",
        "expected_outcome": "answer",
        "observed_outcome": "answer" if correct else "refuse",
        "outcome_correct": correct,
        "status": "answered" if correct else "not_covered",
        "answer": GroundedAnswer(status="answered", summary="s"),
        "evidence": [],
        "cited_keys": ["ekyc:Appendix 2:10"],
        "evidence_keys": ["ekyc:Appendix 2:10", "ekyc:Appendix 2:11"],
        "citation_recall": recall,
        "evidence_recall": 1.0,
        "verified": True,
        "notes": [],
        "llm_calls": 4,
        "input_tokens": 100,
        "output_tokens": 50,
        "latency_ms": 2000,
        "cached": cached,
    }
    return TurnResult(**(base | extra))


def judgement(*numbers):
    return AnswerJudgement(
        statements=[StatementVerdict(statement=n, reason="r", supported="yes") for n in numbers],
        relevancy_reason="r",
        relevancy=5,
    )


def test_verdicts_need_exactly_one_per_statement_numbered_from_one():
    assert verdicts(judgement(1, 2), 2) == ["yes", "yes"]
    for numbers in [(1,), (0, 1), (1, 2, 3)]:
        with pytest.raises(JudgeMismatchError):
            verdicts(judgement(*numbers), 2)
    assert agreement(["yes", "no"], ["yes", "partly"]) == 0.5 and agreement([], []) is None


def test_latency_ignores_cached_replays_and_errors_are_counted_apart():
    summary = mode_summary(
        [turn(latency_ms=2000), turn(latency_ms=0, cached=True), turn(observed_outcome="error", correct=False)]
    )
    assert summary["latency_p50"] == 2.0 and summary["latency_n"] == 1
    assert summary["errors"] == 1 and summary["outcome_accuracy"] == 1.0


def test_report_explains_where_each_gold_paragraph_was_lost():
    items = load_golden()
    by_id = {i.id: i for i in items}
    before = score(by_id["fact-4"], "hybrid (RRF)", ["ftfc:9.1"])
    after = score(by_id["fact-4"], "+ exact lookup", ["ekyc:9.1", "amlcft:9.1", "ftfc:9.1"])
    report, data = build_report(
        "test",
        {
            "date": "d",
            "commit": "c",
            "prompt_version": "p",
            "golden_size": 40,
            "models": {"generator": "m"},
            "review": {"claude_code": ["a"], "syakir": []},
        },
        items,
        [before, after],
        [turn(recall=0.5), turn(mode="traditional", correct=False)],
        {"llm_requests": {}, "embeddings": 0, "reranks": 0, "cache_hits": 0},
    )
    assert "| hybrid (RRF) | 1 | 100% | 100% | 1.00 |" in report
    assert "fact-4 (paraphrased): hybrid (RRF) → + exact lookup: MRR@5 1.00 → 0.33" in report
    assert "retrieved but not cited: ['ekyc:Appendix 2:11']" in report
    assert "expected answer, got refuse" in report
    assert "Syakir spot-checked 0 item(s)" in report and "How to read this" in report
    assert set(data["answers_by_mode"]) == {"agentic", "traditional"}


def test_estimate_counts_follow_up_turns_and_the_judge():
    items = {i.id: i for i in load_golden()}
    one = estimate([items["fact-6"]], ["agentic"], judge=True, cross_check=0)
    follow_up = estimate([items["followup-1"]], ["agentic"], judge=True, cross_check=0)
    key = f"expected {quota_name(role_models('planner')[0])}"  # whichever model the planner uses here
    assert follow_up[key] > one[key] and follow_up["worst case, all models"] == 2 * 5 + 1


def test_cross_check_sample_covers_every_category():
    turns = [
        turn(item_id=f"{c}-{i}", category=c) for c in ("scenario", "comparison", "cross_reference") for i in range(5)
    ]
    assert [t.category for t in stratified(turns, 4)] == [
        "scenario",
        "comparison",
        "cross_reference",
        "scenario",
    ]


def test_only_rejects_unknown_ids_and_ignores_repeats():
    golden = load_golden()
    retrieval, answers = select_items(golden, Namespace(smoke=False, full=False, only=["fact-6", "fact-6"]))
    assert [i.id for i in answers] == ["fact-6"]
    with pytest.raises(SystemExit, match="Unknown item id"):
        select_items(golden, Namespace(smoke=False, full=False, only=["nope"]))


# ---------------------------------------------------------------- harness with fakes (no API)

GOOD = GroundedAnswer(
    status="answered",
    summary="s",
    statements=[
        Statement(text="t", obligation="must", evidence_ids=["E1"], quote="shall verify identity using", topic="x")
    ],
)
EVAL_CORPUS = {"e84": payload("8.4", "S 8.4 A financial institution shall verify identity using biometric technology")}


class CountingLLM(ScriptedLLM):
    def __init__(self, **scripts):
        super().__init__(**scripts)
        self.sent = 0

    def invoke(self, role, messages, schema):
        self.sent += 1
        return super().invoke(role, messages, schema)


def test_answer_eval_runs_follow_ups_in_one_thread_and_counts_only_the_last_turn():
    item = GoldenItem(
        id="f",
        category="follow_up",
        previous_turns=["biometric technology for banks?"],
        question="and technology",
        expected_paragraphs=["ekyc:8.4"],
        expected_outcome="answer",
    )
    llm = CountingLLM(generator=[GOOD, GOOD])
    context = AgentContext(llm=llm, search=FakeSearch(EVAL_CORPUS))
    [result] = evaluate_answers([item], ["traditional"], context, lambda: llm.sent)
    assert result.outcome_correct and result.citation_recall == 1.0 and result.llm_calls == 1
    assert not result.cached and result.conversation == "User: biometric technology for banks?\nUser: and technology"
    assert len(llm.calls) == 2  # the earlier turn ran first, in the same conversation


def test_a_failing_item_is_recorded_as_an_error_and_the_run_goes_on():
    items = [
        GoldenItem(
            id="a",
            category="paraphrased",
            question="technology",
            expected_paragraphs=["ekyc:8.4"],
            expected_outcome="answer",
        ),
        GoldenItem(
            id="b",
            category="paraphrased",
            question="technology",
            expected_paragraphs=["ekyc:8.4"],
            expected_outcome="answer",
        ),
    ]
    llm = CountingLLM(generator=[RuntimeError("boom"), GOOD])
    results = evaluate_answers(
        items, ["traditional"], AgentContext(llm=llm, search=FakeSearch(EVAL_CORPUS)), lambda: llm.sent
    )
    assert results[0].observed_outcome == "error" and "boom" in results[0].error
    assert results[1].outcome_correct


def test_a_cached_replay_is_marked_so_its_latency_is_ignored():
    item = GoldenItem(
        id="a",
        category="paraphrased",
        question="technology",
        expected_paragraphs=["ekyc:8.4"],
        expected_outcome="answer",
    )
    llm = CountingLLM(generator=[GOOD])
    [result] = evaluate_answers(
        [item], ["traditional"], AgentContext(llm=llm, search=FakeSearch(EVAL_CORPUS)), lambda: 0
    )
    assert result.cached


def test_judging_isolates_failures_and_stops_the_cross_check_at_the_first_outage():
    turns = [turn(item_id=str(i), answer=GOOD) for i in range(3)]
    judge = CountingLLM(judge=[judgement(1), LLMUnavailableError("quota", 2), judgement(1)])
    cross = CountingLLM(judge=[judgement(1), LLMUnavailableError("quota", 1)])
    judge_turns(turns, judge, cross, cross_check=5)
    assert [t.faithfulness for t in turns] == [1.0, None, 1.0]
    assert [bool(t.cross_check_verdicts) for t in turns] == [True, False, False] and cross.sent == 2


def test_the_judge_sees_the_users_conversation_and_the_cited_passages():
    llm = CountingLLM(judge=[judgement(1)])
    evidence = merge_evidence([], to_evidence([candidate("e84", EVAL_CORPUS["e84"])], "search", "q"))
    judge_answer(llm, "User: earlier\nUser: and now?", GOOD, evidence)
    prompt = llm.calls[0][1][-1].content
    assert "User: earlier\nUser: and now?" in prompt and "[E1] ekyc para 8.4 (S)" in prompt


def test_ablation_stages_on_a_fake_search():
    class Hit:
        def __init__(self, cid, pid):
            self.id = cid
            self.payload = {"doc_code": "ekyc", "scope": "main", "paragraph_id": pid, "embed_text": pid, "text": pid}

    class Repo:
        def search_dense(self, vector, limit):
            return [Hit("a", "8.1"), Hit("b", "8.4")]

        def search_sparse(self, vector, limit):
            return [Hit("b", "8.4"), Hit("a", "8.1")]

        def search_hybrid(self, dense, sparse, limit, exclude_tags, documents=()):
            return self.search_dense(dense, limit), self.search_sparse(sparse, limit)

        def get_by_paragraph_ids(self, ids, exclude_tags, documents=()):
            return [Hit("b", "8.4")] if "8.4" in ids else []

    class Rerank:
        def rerank(self, query, documents):
            return [RerankScore(i, 1.0 if d == "8.4" else 0.1) for i, d in enumerate(documents)]

    item = GoldenItem(
        id="x",
        category="scenario",
        question="what does para 8.4 say",
        expected_paragraphs=["ekyc:8.4"],
        expected_outcome="answer",
    )
    scores, errors = evaluate_retrieval([item], SearchParts(Repo(), lambda q: [0.0], Rerank()))
    by_stage = {s.stage: s for s in scores}
    assert not errors and set(by_stage) == set(STAGES)
    assert by_stage["dense only"].mrr == 0.5 and by_stage["+ exact lookup"].ranked[0] == "ekyc:8.4"
    assert by_stage["+ rerank"].mrr == 1.0 and by_stage["+ selection"].hit == 1.0


def test_a_judge_that_also_answered_the_turn_is_flagged_as_self_judged():
    # The fake judge reports "fake-judge": a turn whose answer came (partly) from that model is self-judged.
    fell_back = turn(item_id="a", answer=GOOD, models=["deepseek:deepseek-flash", "fake-judge"])
    independent = turn(item_id="b", answer=GOOD, models=["deepseek:deepseek-flash"])
    judge_turns([fell_back, independent], CountingLLM(judge=[judgement(1), judgement(1)]), None, cross_check=0)
    assert fell_back.self_judged and not independent.self_judged


def test_answers_that_used_a_fallback_model_are_counted():
    primaries = frozenset({"deepseek:deepseek-flash"})
    turns = [
        turn(item_id="a", models=["deepseek:deepseek-flash"]),
        turn(item_id="b", models=["deepseek:deepseek-flash", "google_genai:gemini-3.5-flash-lite"]),
    ]
    assert mode_summary(turns, primaries)["fallback_answers"] == 1


@pytest.mark.parametrize("deepseek_key", ["", "ds-key"])
def test_the_estimate_books_calls_on_the_models_that_will_answer(monkeypatch, deepseek_key):
    monkeypatch.setattr(settings, "deepseek_api_key", deepseek_key)
    calls = estimate([load_golden()[0]], ["agentic"], judge=True, cross_check=0)
    planner = quota_name(role_models("planner")[0])
    judge = quota_name(role_models("judge")[0])
    assert f"expected {planner}" in calls and f"expected {judge}" in calls
    assert (planner == "deepseek-flash") is bool(deepseek_key)  # no key: the Gemini fallback answers
