"""The UI's local backend (Streamlit Cloud) and the pages themselves.

The backend wraps the same Services the API tests build: real graphs driven by
a scripted LLM, a fake search and an in-memory audit database. The pages run
headless with Streamlit's AppTest against that backend, so a UI change that
breaks a page fails here. No Gemini, Cohere, Qdrant or Postgres needed."""

import uuid
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import ui.backend
from app.api.runner import DAILY_LIMIT_MESSAGE
from app.core.config import settings
from app.db import vector_repository
from tests.fakes import ScriptedLLM
from tests.test_api import ANSWER, REVIEW, Harness, plan
from ui.backend import START_FAILED_MESSAGE
from ui.local_backend import STORE_DOWN_MESSAGE, LocalBackend, publish_pdfs

QUESTION = "Can banks use biometric technology?"
PAGES = Path(__file__).resolve().parents[1] / "ui" / "views"


def local(daily_limit=0, answers=1):
    llm = ScriptedLLM(planner=[plan()] * answers, reviewer=[REVIEW] * answers, generator=[ANSWER] * answers)
    harness = Harness(llm, daily_limit=daily_limit)
    return LocalBackend(harness.services), harness


def test_chat_speaks_the_api_event_contract():
    backend, _ = local()
    events = list(backend.chat(QUESTION, "agentic", "banking"))
    assert [e["type"] for e in events][0] == "start" and events[-1]["type"] == "answer"
    assert {e["title"] for e in events if e["type"] == "step"} >= {"Plan", "Review", "Write", "Verify"}
    assert events[-1]["turn"]["summary"] == ANSWER.summary


def test_history_is_only_the_visitors_and_replays():
    backend, harness = local()
    mine = list(backend.chat(QUESTION, "agentic", None))[0]["conversation_id"]
    harness.audit.ensure_conversation(None, "someone else's question", None)
    assert [c["id"] for c in backend.conversations([mine])] == [mine]
    assert backend.conversations([]) == []
    assert [t["question"] for t in backend.conversation(mine)] == [QUESTION]


def test_limits_and_outages_become_friendly_errors(monkeypatch):
    backend, _ = local(daily_limit=1)
    list(backend.chat(QUESTION, "agentic", None))
    assert list(backend.chat(QUESTION, "agentic", None)) == [{"type": "error", "message": DAILY_LIMIT_MESSAGE}]

    backend, harness = local()

    def down(*args):
        raise ConnectionError("neon unreachable")

    monkeypatch.setattr(harness.audit, "ensure_conversation", down)
    assert list(backend.chat(QUESTION, "agentic", None)) == [{"type": "error", "message": STORE_DOWN_MESSAGE}]


def test_feedback_and_pdf_links():
    backend, _ = local()
    turn_id = list(backend.chat(QUESTION, "agentic", None))[-1]["turn"]["turn_id"]
    assert backend.feedback(turn_id, 1) and not backend.feedback(str(uuid.uuid4()), 1)
    assert backend.pdf_link("/documents/ekyc.pdf#page=8") == "/app/static/pd-ekyc-apr2024.pdf#page=8"
    assert backend.pdf_link("/documents/amlcft.pdf") == "/app/static/PD_AMLCFTCPF_TFS_FI_Feb2024_v2.pdf"


def test_pdfs_are_published_once(tmp_path):
    source, static = tmp_path / "pdfs", tmp_path / "static"
    source.mkdir()
    for name in ("pd-ekyc-apr2024.pdf", "pd-ftfc-mar24.pdf", "PD_AMLCFTCPF_TFS_FI_Feb2024_v2.pdf"):
        (source / name).write_bytes(b"%PDF-1.7 " + name.encode())
    publish_pdfs(source, static)
    first = (static / "pd-ekyc-apr2024.pdf").stat().st_mtime_ns
    publish_pdfs(source, static)
    assert len(list(static.iterdir())) == 3 and (static / "pd-ekyc-apr2024.pdf").stat().st_mtime_ns == first


def use_backend(monkeypatch, backend):
    """Point the pages at this backend, with no per-visitor limit unless a test sets one
    (whatever the developer's .env says)."""
    monkeypatch.setattr(ui.backend, "backend", lambda: backend)
    monkeypatch.setattr(settings, "questions_per_session", 0)


@pytest.fixture
def pages(monkeypatch):
    backend, _ = local(answers=3)
    use_backend(monkeypatch, backend)
    return backend


def test_the_ask_page_answers_a_question(pages):
    app = AppTest.from_file(PAGES / "chat.py", default_timeout=30).run()
    assert not app.exception and app.title[0].value == "Policy assistant"
    app.chat_input[0].set_value(QUESTION).run()
    assert not app.exception
    shown = " ".join(m.value for m in app.markdown)
    assert ANSWER.summary in shown and ANSWER.statements[0].text in shown
    assert any("Your conversations" in c.value for c in app.sidebar.caption)


def test_an_example_question_is_asked_once(pages):
    app = AppTest.from_file(PAGES / "chat.py", default_timeout=30).run()
    app.pills[0].set_value(app.pills[0].options[0]).run()
    assert not app.exception and len(app.session_state.turns) == 1
    app.run()  # a rerun must not ask it again
    assert len(app.session_state.turns) == 1


def test_the_compare_page_answers_in_both_modes(pages):
    app = AppTest.from_file(PAGES / "compare.py", default_timeout=30).run()
    app.text_input[0].set_value(QUESTION)
    app.button[0].click().run()
    assert not app.exception
    results = app.session_state.compare_results
    assert results["traditional"]["mode"] == "traditional" and results["agentic"]["mode"] == "agentic"


def test_the_compare_page_explains_a_refusal(monkeypatch):
    use_backend(monkeypatch, local(daily_limit=1, answers=3)[0])
    app = AppTest.from_file(PAGES / "compare.py", default_timeout=30).run()
    app.text_input[0].set_value(QUESTION)
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state.compare_results["traditional"] is not None  # the first question fit the limit
    assert DAILY_LIMIT_MESSAGE in [e.value for e in app.error]  # the second one says why, after the rerun


def test_the_visitor_limit_is_explained(pages, monkeypatch):
    monkeypatch.setattr(settings, "questions_per_session", 1)
    app = AppTest.from_file(PAGES / "chat.py", default_timeout=30).run()
    app.chat_input[0].set_value(QUESTION).run()
    app.chat_input[0].set_value("And for insurers?").run()
    assert not app.exception and len(app.session_state.turns) == 1
    assert any("used this demo's questions" in i.value for i in app.info)


def test_a_backend_that_cannot_start_shows_a_message(monkeypatch):
    def unreachable():
        raise ConnectionError("qdrant cloud unreachable")

    monkeypatch.setattr(ui.backend, "_build", unreachable)
    app = AppTest.from_file(PAGES / "chat.py", default_timeout=30).run()
    assert not app.exception and [e.value for e in app.error] == [START_FAILED_MESSAGE]


def test_qdrant_cloud_gets_the_api_key(monkeypatch):
    seen = {}
    monkeypatch.setattr(vector_repository, "QdrantClient", lambda **kwargs: seen.update(kwargs))
    monkeypatch.setattr(settings, "qdrant_url", "https://cluster.cloud.qdrant.io:6333")
    monkeypatch.setattr(settings, "qdrant_api_key", "key")
    vector_repository.connect_qdrant()
    assert seen == {"url": "https://cluster.cloud.qdrant.io:6333", "api_key": "key"}
    monkeypatch.setattr(settings, "qdrant_api_key", "")
    vector_repository.connect_qdrant()
    assert seen["api_key"] is None  # the local container has no auth


@pytest.mark.parametrize("page", ["evaluation.py", "about.py"])
def test_the_reading_pages_render(page):
    app = AppTest.from_file(PAGES / page, default_timeout=30).run()
    assert not app.exception
