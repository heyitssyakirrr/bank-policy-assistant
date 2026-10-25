"""The UI's API client: SSE parsing and error handling, with a mock transport."""

import json

import httpx

from ui.api_client import ApiClient, parse_sse


def sse_body(*events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


def client(handler):
    return ApiClient("http://api", "http://localhost:8000", transport=httpx.MockTransport(handler))


def test_parse_sse_yields_one_event_per_block():
    lines = sse_body({"type": "start"}, {"type": "step", "lines": ["a"]}).splitlines()
    assert [e["type"] for e in parse_sse(lines)] == ["start", "step"]


def test_chat_streams_the_events_in_order():
    events = [{"type": "start", "turn_id": "t"}, {"type": "step", "lines": ["x"]}, {"type": "answer", "turn": {}}]

    def handler(request):
        assert json.loads(request.content)["mode"] == "agentic"
        return httpx.Response(200, text=sse_body(*events), headers={"content-type": "text/event-stream"})

    assert [e["type"] for e in client(handler).chat("q", "agentic", "banking")] == ["start", "step", "answer"]


def test_chat_turns_http_failures_into_an_error_event():
    refused = client(lambda request: httpx.Response(422, json={"detail": [{"msg": "too long"}]}))
    assert list(refused.chat("", "agentic", None)) == [
        {"type": "error", "message": "The API refused the question (422)."}
    ]
    limited = client(lambda request: httpx.Response(429, json={"detail": "Today's question limit…"}))
    assert list(limited.chat("q", "agentic", None)) == [{"type": "error", "message": "Today's question limit…"}]

    def unreachable(request):
        raise httpx.ConnectError("down")

    assert list(client(unreachable).chat("q", "agentic", None)) == [
        {"type": "error", "message": "Cannot reach the API (ConnectError)."}
    ]


def test_reads_fall_back_when_the_api_is_down():
    def unreachable(request):
        raise httpx.ConnectError("down")

    api = client(unreachable)
    assert api.conversations(["c1"]) == [] and api.health() == {"status": "unreachable"} and not api.feedback("t", 1)


def test_history_asks_only_for_the_visitors_conversations():
    asked = []

    def handler(request):
        asked.append(request.url.params.get_list("ids"))
        return httpx.Response(200, json=[{"id": "c1"}])

    api = client(handler)
    assert api.conversations([]) == [] and asked == []  # nothing to ask for: no request
    assert api.conversations(["c1", "c2"]) == [{"id": "c1"}] and asked == [["c1", "c2"]]


def test_pdf_links_use_the_browser_address():
    assert client(lambda r: httpx.Response(200)).pdf_link("/documents/ekyc.pdf#page=8") == (
        "http://localhost:8000/documents/ekyc.pdf#page=8"
    )
