"""The FastAPI app: the agent behind an API, so any front end can use it.

    POST /chat                     answer a question, streaming steps (SSE)
    GET  /conversations            the conversation list
    GET  /conversations/{id}       a conversation's turns
    POST /turns/{id}/feedback      thumbs up / down
    GET  /documents, /documents/{code}.pdf
    GET  /health                   readiness: Qdrant, Postgres, versions, models

Run: uvicorn app.main:app  (after `alembic upgrade head`)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import ExitStack, asynccontextmanager
from functools import partial

from fastapi import FastAPI
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from qdrant_client import QdrantClient

from app.agent.checkpoint import state_serializer
from app.agent.graph import build_agentic_graph
from app.agent.state import AgentContext
from app.agent.traditional import build_traditional_graph
from app.api.deps import Services
from app.api.routes import chat, conversations, documents, feedback, health
from app.api.routes.health import readiness
from app.api.runner import TurnRunner, index_version
from app.core.config import enable_tracing, settings
from app.core.logging import RequestContextMiddleware, setup_logging
from app.db.audit_repository import AuditRepository
from app.db.session import database_ok, make_engine, make_session_factory, psycopg_url
from app.db.vector_repository import VectorRepository
from app.llm.models import ChatModelClient
from app.services.search_service import build_search_service


def build_services(resources: ExitStack) -> Services:
    """The real stack. No response cache: a live user always gets a fresh answer.
    Tracing starts here, not at import, so tests never send runs to LangSmith."""
    enable_tracing()
    engine = make_engine()
    resources.callback(engine.dispose)
    audit = AuditRepository(make_session_factory(engine))

    pool = ConnectionPool(
        psycopg_url(), kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row}, open=True
    )
    resources.callback(pool.close)
    checkpointer = PostgresSaver(pool, serde=state_serializer())
    checkpointer.setup()  # LangGraph creates and migrates its own checkpoint tables

    graphs = {"agentic": build_agentic_graph(checkpointer), "traditional": build_traditional_graph(checkpointer)}
    context = AgentContext(llm=ChatModelClient(), search=build_search_service())
    version = index_version(settings.chunks_path)
    vectors = VectorRepository(QdrantClient(url=settings.qdrant_url), settings.qdrant_collection)
    report = partial(readiness, vectors.count, partial(database_ok, engine), version)
    return Services(runner=TurnRunner(graphs, context, audit, version), audit=audit, health=report)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    with ExitStack() as resources:
        if not hasattr(app.state, "services"):  # tests provide their own
            app.state.services = build_services(resources)
        yield


def create_app(services: Services | None = None) -> FastAPI:
    setup_logging(settings.log_level)
    app = FastAPI(title="Bank Policy Compliance Assistant", lifespan=lifespan)
    if services is not None:
        app.state.services = services
    app.add_middleware(RequestContextMiddleware)
    for module in (chat, conversations, feedback, documents, health):
        app.include_router(module.router)
    return app


app = create_app()
