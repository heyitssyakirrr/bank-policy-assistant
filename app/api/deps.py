"""What the routes need, built once at startup and stored on the app.

Tests build the same `Services` with a fake graph and an in-memory database,
so every route runs without Gemini, Cohere, Qdrant or Postgres.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import Depends, Request

from app.api.runner import TurnRunner
from app.db.audit_repository import AuditRepository


@dataclass(frozen=True, slots=True)
class Services:
    runner: TurnRunner
    audit: AuditRepository
    health: Callable[[], dict[str, Any]]  # readiness report, spends no quota


def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]
