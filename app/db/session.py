"""The Postgres connection: one SQLAlchemy engine for our tables, and the URL
form the LangGraph checkpointer (plain psycopg) expects."""

from __future__ import annotations

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings


def make_engine(url: str = settings.database_url) -> Engine:
    return create_engine(url, pool_pre_ping=True)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


def psycopg_url(url: str = settings.database_url) -> str:
    """'postgresql+psycopg://...' (SQLAlchemy) -> 'postgresql://...' (psycopg)."""
    return url.replace("postgresql+psycopg://", "postgresql://", 1)


def database_ok(engine: Engine) -> bool:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
