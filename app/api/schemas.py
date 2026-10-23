"""Request and response bodies of the API (the views of a turn are in views.py)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.agent.schemas import Mode, Sector
from app.core.config import settings


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=settings.max_question_chars)
    mode: Mode = "agentic"
    sector: Sector | None = None  # None = "Not sure: show all sectors"
    conversation_id: uuid.UUID | None = None  # None starts a new conversation


class FeedbackRequest(BaseModel):
    rating: Literal[1, -1]
    comment: str | None = Field(default=None, max_length=2000)


class ConversationSummary(BaseModel):
    id: uuid.UUID
    title: str
    user_sector: str | None
    created_at: datetime
    updated_at: datetime


class DocumentInfo(BaseModel):
    code: str
    title: str
    pdf_url: str
