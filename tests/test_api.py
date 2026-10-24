"""API tests: every route, with the real graphs driven by a scripted LLM and a
fake search, and the audit tables in an in-memory SQLite database. No
Gemini, Cohere, Qdrant or Postgres needed."""

import gzip
import json
import logging
import uuid

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app.agent.checkpoint import state_serializer
from app.agent.graph import build_agentic_graph
from app.agent.schemas import EvidenceReview, GroundedAnswer, QueryPlan, SearchRequest, Statement
from app.agent.state import AgentContext
from app.agent.traditional import build_traditional_graph
from app.api.deps import Services
from app.api.routes.health import readiness
from app.api.runner import TurnRunner, index_version
from app.api.views import source_view
from app.core.logging import RequestIdFilter
from app.db.audit_repository import AuditRepository
from app.db.models import Base
from app.db.session import make_session_factory
from app.main import create_app
from tests.fakes import FakeSearch, ScriptedLLM, payload

CORPUS = {"e84": payload("8.4", "S 8.4 A financial institution shall verify identity using biometric technology")}


def plan(question="Can banks use biometric technology?"):
    return QueryPlan(
        route="search",
        standalone_question=question,
        searches=[SearchRequest(query="technology")],
        reasoning="One topic, one search.",
    )


REVIEW = EvidenceReview(relevant_evidence=["E1"], is_sufficient=True)
ANSWER = GroundedAnswer(
    status="answered",
    summary="Yes: banks must verify identity with biometric technology.",
    statements=[
        Statement(
            text="Banks must verify identity using biometric technology.",
            obligation="must",
            evidence_ids=["E1"],
            quote="shall verify identity using biometric technology",
            topic="biometrics",
        )
    ],
)


class Harness:
    def __init__(self, llm, graphs=None):
        engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(engine)
        self.audit = AuditRepository(make_session_factory(engine))
        self.llm = llm
        saver = InMemorySaver(serde=state_serializer())
        graphs = graphs or {"agentic": build_agentic_graph(saver), "traditional": build_traditional_graph(saver)}
        runner = TurnRunner(graphs, AgentContext(llm=llm, search=FakeSearch(CORPUS)), self.audit, "test-index")
        services = Services(runner=runner, audit=self.audit, health=lambda: {"status": "ok"})
        self.client = TestClient(create_app(services))

    def chat(self, question, **body):
        response = self.client.post("/chat", json={"question": question, **body})
        assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")
        return [json.loads(block.split("data: ", 1)[1]) for block in response.text.strip().split("\n\n")]


@pytest.fixture
def harness():
    return Harness(ScriptedLLM(planner=[plan()], reviewer=[REVIEW], generator=[ANSWER]))


def test_chat_streams_start_steps_then_the_answer(harness):
    events = harness.chat("Can banks use biometric technology?", sector="banking")
    types = [e["type"] for e in events]
    assert types[0] == "start" and types[-1] == "answer" and set(types[1:-1]) == {"step"}
    assert [e["node"] for e in events if e["type"] == "step"][:2] == ["understand_question", "search_policies"]

    turn = events[-1]["turn"]
    assert turn["status"] == "answered" and turn["verified"] and turn["applies_to"] == "banking institutions"
    assert turn["statements"][0]["sources"] == ["E1"]
    assert turn["sources"][0]["pdf_url"] == "/documents/ekyc.pdf#page=6" and turn["sources"][0]["provision"] == "S"
    assert turn["llm_calls"] == 3 and "Citations verified ✓" in turn["steps"]


def test_every_turn_and_step_is_written_to_the_audit_log(harness):
    events = harness.chat("Can banks use biometric technology?")
    conversation_id = uuid.UUID(events[0]["conversation_id"])
    [turn] = harness.audit.conversation_turns(conversation_id)
    assert turn.prompt_version and turn.index_version == "test-index" and turn.models
    assert turn.evidence[0]["chunk_id"] == "e84" and turn.plan["route"] == "search"
    steps = [e for e in events if e["type"] == "step"]
    assert [e.node for e in harness.audit.turn_events([turn.id])[turn.id]] == [e["node"] for e in steps]


def test_a_follow_up_works_across_two_requests():
    llm = ScriptedLLM(
        planner=[plan(), plan("Can takaful operators use biometric technology?")],
        reviewer=[REVIEW, REVIEW],
        generator=[ANSWER, ANSWER],
    )
    harness = Harness(llm)
    first = harness.chat("Can banks use biometric technology?")
    conversation_id = first[0]["conversation_id"]
    second = harness.chat("what about takaful operators?", conversation_id=conversation_id)
    assert second[0]["conversation_id"] == conversation_id
    second_planner_prompt = [messages for role, messages in llm.calls if role == "planner"][1][-1].content
    assert "User: Can banks use biometric technology?" in second_planner_prompt
    assert second[-1]["turn"]["standalone_question"] == "Can takaful operators use biometric technology?"
    assert len(harness.audit.conversation_turns(uuid.UUID(conversation_id))) == 2


def test_pii_is_masked_before_the_llm_and_the_audit_log(harness):
    events = harness.chat("Customer 900101-14-5678 wants biometric technology onboarding")
    assert events[0]["pii_masked"] == ["NRIC"]
    prompts = " ".join(str(m.content) for _, messages in harness.llm.calls for m in messages)
    assert "900101" not in prompts and "[NRIC]" in prompts
    [turn] = harness.audit.conversation_turns(uuid.UUID(events[0]["conversation_id"]))
    assert "900101" not in turn.question and turn.pii_masked == ["NRIC"]


def test_a_failing_graph_reports_an_error_and_is_still_audited():
    class Broken:
        def stream(self, *args, **kwargs):
            raise RuntimeError("qdrant down")
            yield  # pragma: no cover

    harness = Harness(ScriptedLLM(), graphs={"agentic": Broken(), "traditional": Broken()})
    events = harness.chat("anything")
    assert [e["type"] for e in events] == ["start", "error"]
    [turn] = harness.audit.conversation_turns(uuid.UUID(events[0]["conversation_id"]))
    assert turn.status == "error" and "qdrant down" in turn.error


def test_conversations_are_listed_and_replayed(harness):
    events = harness.chat("Can banks use biometric technology?")
    conversation_id = events[0]["conversation_id"]
    listed = harness.client.get("/conversations").json()
    assert listed[0]["id"] == conversation_id and listed[0]["title"] == "Can banks use biometric technology?"
    [turn] = harness.client.get(f"/conversations/{conversation_id}").json()
    assert turn["summary"] == ANSWER.summary and turn["steps"]
    assert harness.client.get(f"/conversations/{uuid.uuid4()}").status_code == 404


def test_feedback_is_saved_for_a_real_turn_only(harness):
    turn_id = harness.chat("Can banks use biometric technology?")[0]["turn_id"]
    assert harness.client.post(f"/turns/{turn_id}/feedback", json={"rating": 1}).status_code == 204
    assert harness.client.post(f"/turns/{uuid.uuid4()}/feedback", json={"rating": -1}).status_code == 404
    assert harness.client.post(f"/turns/{turn_id}/feedback", json={"rating": 5}).status_code == 422


def test_documents_are_listed_and_served(harness):
    codes = [d["code"] for d in harness.client.get("/documents").json()]
    assert codes == ["ekyc", "ftfc", "amlcft"]
    pdf = harness.client.get("/documents/ekyc.pdf")
    assert pdf.status_code == 200 and pdf.headers["content-type"] == "application/pdf"
    assert harness.client.get("/documents/other.pdf").status_code == 404


def test_requests_are_validated_and_tagged(harness):
    assert harness.client.post("/chat", json={"question": ""}).status_code == 422
    assert harness.client.post("/chat", json={"question": "q", "mode": "react"}).status_code == 422
    response = harness.client.get("/health")
    assert response.json() == {"status": "ok"} and len(response.headers["x-request-id"]) == 8


def start_and_stream(harness, question, mode="agentic", conversation_id=None):
    runner = harness.client.app.state.services.runner
    start = runner.start(question, "banking", conversation_id)
    return start, runner.stream(start, mode, "banking")


def test_a_client_leaving_mid_stream_is_still_audited(harness):
    start, events = start_and_stream(harness, "Can banks use biometric technology?")
    assert next(events)["type"] == "start" and next(events)["node"] == "understand_question"
    events.close()  # what Starlette does when the browser disconnects
    [turn] = harness.audit.conversation_turns(start.conversation_id)
    assert turn.status == "cancelled" and turn.error == "client disconnected" and turn.llm_calls == 1
    assert [e.node for e in harness.audit.turn_events([turn.id])[turn.id]] == ["understand_question"]


def test_an_audit_write_failure_still_delivers_the_answer(harness, monkeypatch):
    def broken(*args):
        raise RuntimeError("postgres hiccup")

    monkeypatch.setattr(harness.audit, "record_turn", broken)
    events = harness.chat("Can banks use biometric technology?")
    assert events[-1]["type"] == "answer"
    assert "could not be saved" in events[-1]["turn"]["notes"][-1]


def test_traditional_mode_runs_through_the_api():
    harness = Harness(ScriptedLLM(generator=[ANSWER]))
    events = harness.chat("biometric technology", mode="traditional")
    nodes = [e["node"] for e in events if e["type"] == "step"]
    assert nodes == ["retrieve", "generate_answer", "verify_answer", "finish"]
    assert events[-1]["turn"]["mode"] == "traditional" and events[-1]["turn"]["llm_calls"] == 1


def test_the_two_modes_keep_separate_memory_in_one_conversation():
    llm = ScriptedLLM(planner=[plan(), plan()], reviewer=[REVIEW, REVIEW], generator=[ANSWER, ANSWER, ANSWER])
    harness = Harness(llm)
    conversation_id = harness.chat("Can banks use biometric technology?")[0]["conversation_id"]
    harness.chat("biometric technology", mode="traditional", conversation_id=conversation_id)
    harness.chat("and takaful operators?", conversation_id=conversation_id)
    last_planner_prompt = [messages for role, messages in llm.calls if role == "planner"][-1][-1].content
    assert last_planner_prompt.count("User:") == 1  # only the agentic thread's earlier question


def test_a_second_question_while_one_runs_is_refused(harness):
    start, first = start_and_stream(harness, "Can banks use biometric technology?")
    next(first)
    runner = harness.client.app.state.services.runner
    second = list(runner.stream(start, "agentic", "banking"))
    assert second[0]["type"] == "error" and "already being answered" in second[0]["message"]
    first.close()


def test_log_lines_during_a_turn_carry_its_ids(harness, caplog):
    caplog.set_level("INFO")
    start, events = start_and_stream(harness, "Can banks use biometric technology?")
    caplog.handler.addFilter(RequestIdFilter())
    for _ in events:
        logging.getLogger("app.test").info("inside the turn")
    assert {r.turn_id for r in caplog.records if r.name == "app.test"} == {str(start.turn_id)[:8]}


def test_readiness_reports_degraded_without_spending_quota():
    def qdrant_down():
        raise ConnectionError("qdrant")

    report = readiness(qdrant_down, lambda: True, "abc123")
    assert report["status"] == "degraded" and report["qdrant"]["points"] is None
    assert readiness(lambda: 1219, lambda: True, "abc123")["status"] == "ok"
    assert set(report["keys"]) == {"gemini", "cohere"} and all(isinstance(v, bool) for v in report["keys"].values())


def test_index_version_hashes_the_chunks_or_their_gzip(tmp_path):
    plain = tmp_path / "chunks.jsonl"
    plain.write_bytes(b'{"chunk": 1}\n')
    zipped = tmp_path / "zipped" / "chunks.jsonl"
    zipped.parent.mkdir()
    with gzip.open(zipped.with_suffix(".jsonl.gz"), "wb") as handle:
        handle.write(b'{"chunk": 1}\n')
    assert index_version(plain) == index_version(zipped) != "unknown"
    assert index_version(tmp_path / "missing.jsonl") == "unknown"


@pytest.mark.parametrize(
    ("evidence", "label"),
    [
        ({"document": "ekyc", "scope": "main", "paragraph_id": "8.4", "provision": "S"}, "e-KYC 8.4 (S)"),
        ({"document": "ekyc", "scope": "Appendix 2", "paragraph_id": "13", "provision": None}, "e-KYC App. 2 para 13"),
        ({"document": "amlcft", "scope": "main", "paragraph_id": None, "provision": None}, "AML/CFT table"),
    ],
)
def test_source_labels_are_short_and_unambiguous(evidence, label):
    full = evidence | {"id": "E1", "citation": "c", "text": "t", "page": 3}
    assert source_view(full).label == label
