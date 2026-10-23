"""GET /health: is everything the demo needs ready? Spends no API quota:
it counts Qdrant points, runs SELECT 1 and reads names from settings."""

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter

from app.agent.prompts import PROMPT_VERSION
from app.api.deps import ServicesDep
from app.core.config import settings
from app.llm.models import role_models

router = APIRouter()


def readiness(count_points: Callable[[], int], database_ok: Callable[[], bool], index_version: str) -> dict[str, Any]:
    try:
        points: int | None = count_points()
    except Exception:
        points = None
    postgres = database_ok()
    return {
        "status": "ok" if points and postgres else "degraded",
        "qdrant": {"ok": bool(points), "points": points, "collection": settings.qdrant_collection},
        "postgres": {"ok": postgres},
        "index_version": index_version,
        "prompt_version": PROMPT_VERSION,
        "models": {role: role_models(role)[0] for role in ("planner", "reviewer", "generator")},
        "keys": {"gemini": bool(settings.gemini_api_key), "cohere": bool(settings.cohere_api_key)},
    }


@router.get("/health")
def health(services: ServicesDep) -> dict[str, Any]:
    return services.health()
