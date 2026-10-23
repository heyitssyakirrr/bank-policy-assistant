"""Our own tables: the product's record of every conversation (PROJECT_PLAN §6).

The LangGraph checkpointer is the agent's working memory (owned by
LangGraph). These tables are what the product shows and what an auditor
reads: the conversation list, every turn with its evidence, models,
prompt and index versions, every graph step, and user feedback.

`turns` and `turn_events` are append-only: the migration installs Postgres
triggers that reject UPDATE and DELETE on them.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, Uuid
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[str | None] = mapped_column(String(128))  # empty until there is a login (SSO later)
    title: Mapped[str] = mapped_column(String(200))
    user_sector: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Turn(Base):
    """One question and its answer: enough to explain "why did it say X?"."""

    __tablename__ = "turns"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("conversations.id"), index=True)
    mode: Mapped[str] = mapped_column(String(16))  # "agentic" | "traditional"
    question: Mapped[str] = mapped_column(Text)  # after PII masking
    pii_masked: Mapped[list[Any]] = mapped_column(default=list)  # kinds masked, e.g. ["NRIC"]
    standalone_question: Mapped[str | None] = mapped_column(Text)
    sector: Mapped[str | None] = mapped_column(String(32))
    plan: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    answer: Mapped[dict[str, Any]] = mapped_column(JSON)
    outcome: Mapped[str] = mapped_column(String(16))  # "answer" | "out_of_scope" | "small_talk" | "clarify"
    status: Mapped[str] = mapped_column(String(32))  # the answer status, or "error"
    verification: Mapped[list[Any]] = mapped_column(default=list)  # every verifier report, in order
    evidence: Mapped[list[Any]] = mapped_column(default=list)  # every passage the model saw
    notes: Mapped[list[Any]] = mapped_column(default=list)
    models: Mapped[list[Any]] = mapped_column(default=list)
    prompt_version: Mapped[str] = mapped_column(String(32))
    index_version: Mapped[str] = mapped_column(String(32))
    llm_calls: Mapped[int] = mapped_column(Integer, default=0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class TurnEvent(Base):
    """One graph step of a turn: every search with its chunks and scores,
    every reviewer decision, fallbacks and errors."""

    __tablename__ = "turn_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    turn_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("turns.id"), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    node: Mapped[str] = mapped_column(String(64))
    lines: Mapped[list[Any]] = mapped_column(default=list)  # what the user saw in "Agent steps"
    output: Mapped[dict[str, Any]] = mapped_column(JSON)  # the node's state update (JSON-safe)
    ms: Mapped[int] = mapped_column(Integer)  # time since the turn started
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    turn_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("turns.id"), index=True)
    rating: Mapped[int] = mapped_column(Integer)  # +1 or -1
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
