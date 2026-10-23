"""How graph state is saved by a checkpointer.

LangGraph only restores the Python types it has been told are safe (the
rest will be refused by a future version): a checkpoint is data loaded from
a database, so it must never be able to construct arbitrary classes. Every
Pydantic model we keep in AgentState is registered here, once, for both the
in-memory saver (scripts, tests) and the Postgres saver (the API).
"""

from __future__ import annotations

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from app.agent.evidence import Evidence
from app.agent.schemas import EvidenceReview, GroundedAnswer, ParagraphRef, QueryPlan, SearchRequest, Statement
from app.agent.verification import Issue, VerificationReport

STATE_TYPES = (
    QueryPlan,
    SearchRequest,
    ParagraphRef,
    EvidenceReview,
    GroundedAnswer,
    Statement,
    Evidence,
    VerificationReport,
    Issue,
)


def state_serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=[(t.__module__, t.__name__) for t in STATE_TYPES])
