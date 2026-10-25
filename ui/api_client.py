"""The UI's backend over HTTP: the FastAPI app (UI_BACKEND=api).

It sends a question and reads the Server-Sent Events stream: "start", one
"step" per graph node, then "answer" (or "error"). See ui/backend.py.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import httpx

STREAM_TIMEOUT = httpx.Timeout(10.0, read=120.0)  # an agentic answer can take ~30-60 s on the free tier
READ_TIMEOUT = httpx.Timeout(5.0)  # sidebar reads: a slow API must not freeze the page


def parse_sse(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """SSE text lines -> event dicts. An event ends at a blank line; its
    payload is the JSON on its `data:` line."""
    data: list[str] = []
    for line in lines:
        if line.startswith("data:"):
            data.append(line[5:].strip())
        elif not line.strip() and data:
            yield json.loads("\n".join(data))
            data = []
    if data:
        yield json.loads("\n".join(data))


def _refusal(response: httpx.Response) -> str:
    """The API's own reason when it has one (e.g. the daily limit, 429)."""
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    return detail if isinstance(detail, str) else f"The API refused the question ({response.status_code})."


class ApiClient:
    def __init__(self, base_url: str, public_url: str, transport: httpx.BaseTransport | None = None) -> None:
        self._http = httpx.Client(base_url=base_url, timeout=STREAM_TIMEOUT, transport=transport)
        self._public_url = public_url.rstrip("/")

    def chat(
        self, question: str, mode: str, sector: str | None, conversation_id: str | None = None
    ) -> Iterator[dict[str, Any]]:
        body = {"question": question, "mode": mode, "sector": sector, "conversation_id": conversation_id}
        try:
            with self._http.stream("POST", "/chat", json=body) as response:
                if response.status_code != 200:
                    response.read()
                    yield {"type": "error", "message": _refusal(response)}
                    return
                yield from parse_sse(response.iter_lines())
        except httpx.HTTPError as exc:
            yield {"type": "error", "message": f"Cannot reach the API ({type(exc).__name__})."}

    def conversations(self, ids: Sequence[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        return self._get("/conversations", default=[], params={"ids": list(ids)})

    def conversation(self, conversation_id: str) -> list[dict[str, Any]]:
        return self._get(f"/conversations/{conversation_id}", default=[])

    def health(self) -> dict[str, Any]:
        return self._get("/health", default={"status": "unreachable"})

    def feedback(self, turn_id: str, rating: int) -> bool:
        try:
            response = self._http.post(f"/turns/{turn_id}/feedback", json={"rating": rating}, timeout=READ_TIMEOUT)
            return response.status_code == 204
        except httpx.HTTPError:
            return False

    def pdf_link(self, pdf_url: str) -> str:
        return f"{self._public_url}{pdf_url}"

    def _get(self, path: str, default: Any, params: dict[str, Any] | None = None) -> Any:
        try:
            response = self._http.get(path, params=params, timeout=READ_TIMEOUT)
            return response.json() if response.status_code == 200 else default
        except httpx.HTTPError:
            return default
