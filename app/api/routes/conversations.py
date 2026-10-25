"""The conversation list (sidebar) and a conversation's past turns."""

import uuid
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from app.api.deps import ServicesDep
from app.api.schemas import ConversationSummary
from app.api.views import TurnView, turn_view

router = APIRouter()


@router.get("/conversations", response_model=list[ConversationSummary])
def list_conversations(
    services: ServicesDep, ids: Annotated[list[uuid.UUID] | None, Query(max_length=100)] = None
) -> list[ConversationSummary]:
    """Most recent first. `?ids=…&ids=…` lists only those (one visitor's history)."""
    conversations = services.audit.list_conversations(ids=ids)
    return [ConversationSummary.model_validate(c, from_attributes=True) for c in conversations]


@router.get("/conversations/{conversation_id}", response_model=list[TurnView])
def conversation_turns(conversation_id: uuid.UUID, services: ServicesDep) -> list[TurnView]:
    turns = services.audit.conversation_turns(conversation_id)
    if not turns:
        raise HTTPException(status_code=404, detail="conversation not found")
    events = services.audit.turn_events([turn.id for turn in turns])
    return [turn_view(turn, events[turn.id]) for turn in turns]
