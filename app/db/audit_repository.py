"""Everything that reads or writes our Postgres tables.

Turns and their events are only ever inserted (append-only audit log);
conversations get their `updated_at` bumped so the sidebar can sort them.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Conversation, Feedback, Turn, TurnEvent, utcnow

TITLE_CHARS = 80


class AuditRepository:
    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions

    def ensure_conversation(
        self, conversation_id: uuid.UUID | None, first_question: str, sector: str | None
    ) -> uuid.UUID:
        """The conversation to add a turn to: the given one if it exists, else
        a new one titled after its first question."""
        with self._sessions.begin() as session:
            if conversation_id is not None and session.get(Conversation, conversation_id) is not None:
                return conversation_id
            conversation = Conversation(
                id=conversation_id or uuid.uuid4(), title=first_question[:TITLE_CHARS], user_sector=sector
            )
            session.add(conversation)
            return conversation.id

    def record_turn(self, turn: Turn, events: Sequence[TurnEvent]) -> None:
        with self._sessions.begin() as session:
            session.add(turn)
            session.flush()  # the turn row must exist before its events reference it
            session.add_all(events)
            conversation = session.get(Conversation, turn.conversation_id)
            if conversation is not None:
                conversation.updated_at = utcnow()

    def list_conversations(self, limit: int = 50) -> list[Conversation]:
        with self._sessions() as session:
            query = select(Conversation).order_by(Conversation.updated_at.desc()).limit(limit)
            return list(session.scalars(query))

    def conversation_turns(self, conversation_id: uuid.UUID) -> list[Turn]:
        with self._sessions() as session:
            query = select(Turn).where(Turn.conversation_id == conversation_id).order_by(Turn.created_at)
            return list(session.scalars(query))

    def turn_events(self, turn_ids: Sequence[uuid.UUID]) -> dict[uuid.UUID, list[TurnEvent]]:
        """The steps of several turns in one query, each turn's in order."""
        events: dict[uuid.UUID, list[TurnEvent]] = {turn_id: [] for turn_id in turn_ids}
        with self._sessions() as session:
            query = select(TurnEvent).where(TurnEvent.turn_id.in_(turn_ids)).order_by(TurnEvent.turn_id, TurnEvent.seq)
            for event in session.scalars(query):
                events[event.turn_id].append(event)
        return events

    def add_feedback(self, turn_id: uuid.UUID, rating: int, comment: str | None) -> bool:
        """False if the turn doesn't exist."""
        with self._sessions.begin() as session:
            if session.get(Turn, turn_id) is None:
                return False
            session.add(Feedback(turn_id=turn_id, rating=rating, comment=comment))
            return True
