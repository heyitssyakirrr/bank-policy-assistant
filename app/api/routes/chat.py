from fastapi import APIRouter

from app.core.logging import logger
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    logger.info(f"Received message: {payload.message!r}")
    # Phase 0 stub — real retrieval + generation is built in Phase 1
    return ChatResponse(
        session_id=payload.session_id or "temp-session",
        answer=f"Echo: {payload.message}",
        sources=[],
        grounded=True,
    )