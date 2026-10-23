"""POST /chat: answer a question, streaming each agent step as it happens.

Server-Sent Events: the response stays open and the server writes one
`event: <type>` + `data: <json>` block per event (start, step..., answer or
error). The UI renders steps live, then the answer.
"""

import json
from collections.abc import Iterable, Iterator
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.api.deps import ServicesDep
from app.api.schemas import ChatRequest

router = APIRouter()


def sse(events: Iterable[dict[str, Any]]) -> Iterator[str]:
    for event in events:
        yield f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/chat")
def chat(request: ChatRequest, services: ServicesDep) -> StreamingResponse:
    try:  # before the stream starts, so a database failure is a real HTTP error
        start = services.runner.start(request.question, request.sector, request.conversation_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="The conversation store is unavailable.") from exc
    events = services.runner.stream(start, request.mode, request.sector)
    return StreamingResponse(
        sse(events), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
