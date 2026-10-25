"""The UI's backend in its own process (UI_BACKEND=local, Streamlit Cloud).

Streamlit Community Cloud runs a single process, so instead of calling
FastAPI the UI calls the same TurnRunner the API uses, built by the same
`build_services()`: same graphs, search, verifier, audit log and limits.
Only the wire between the UI and the agent differs (see ui/backend.py).
"""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Iterator, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config

from app.api.deps import Services
from app.api.runner import DailyLimitReached
from app.api.schemas import ConversationSummary
from app.api.views import DOCUMENT_FILES, turn_view
from app.core.config import settings
from app.main import build_services

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"  # Streamlit serves it at /app/static/
STORE_DOWN_MESSAGE = "The conversation store is unavailable. Please try again in a minute."


def publish_pdfs(source_dir: Path, static_dir: Path) -> None:
    """Copy the policy PDFs where Streamlit's static file serving finds them,
    so a citation can open the PDF at its page."""
    static_dir.mkdir(exist_ok=True)
    for name in DOCUMENT_FILES.values():
        target = static_dir / name
        if not target.exists() or target.stat().st_size != (source_dir / name).stat().st_size:
            shutil.copyfile(source_dir / name, target)


def migrate_database() -> None:
    """`alembic upgrade head` on the configured database (Neon in the cloud).
    No ini file, so Alembic leaves the app's logging alone."""
    config = Config()
    config.set_main_option("script_location", str(ROOT / "migrations"))
    command.upgrade(config, "head")


def _uuids(ids: Sequence[str]) -> list[uuid.UUID]:
    return [uuid.UUID(i) for i in ids]


class LocalBackend:
    def __init__(self, services: Services) -> None:
        self._services = services

    def chat(
        self, question: str, mode: str, sector: str | None, conversation_id: str | None = None
    ) -> Iterator[dict[str, Any]]:
        runner = self._services.runner
        try:
            start = runner.start(question, sector, uuid.UUID(conversation_id) if conversation_id else None)
        except DailyLimitReached as exc:
            yield {"type": "error", "message": str(exc)}
            return
        except Exception:
            yield {"type": "error", "message": STORE_DOWN_MESSAGE}
            return
        yield from runner.stream(start, mode, sector)

    def conversations(self, ids: Sequence[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        try:
            found = self._services.audit.list_conversations(ids=_uuids(ids))
        except Exception:
            return []
        return [ConversationSummary.model_validate(c, from_attributes=True).model_dump(mode="json") for c in found]

    def conversation(self, conversation_id: str) -> list[dict[str, Any]]:
        audit = self._services.audit
        try:
            turns = audit.conversation_turns(uuid.UUID(conversation_id))
            events = audit.turn_events([turn.id for turn in turns])
        except Exception:
            return []
        return [turn_view(turn, events[turn.id]).model_dump(mode="json") for turn in turns]

    def health(self) -> dict[str, Any]:
        return self._services.health()

    def feedback(self, turn_id: str, rating: int) -> bool:
        try:
            return self._services.audit.add_feedback(uuid.UUID(turn_id), rating, None)
        except Exception:
            return False

    def pdf_link(self, pdf_url: str) -> str:
        """'/documents/ekyc.pdf#page=8' -> '/app/static/pd-ekyc-apr2024.pdf#page=8'."""
        path, _, page = pdf_url.partition("#")
        return f"/app/static/{DOCUMENT_FILES[Path(path).stem]}" + (f"#{page}" if page else "")


def build_local_backend() -> LocalBackend:
    """The full stack, kept open for the life of the Streamlit process. If any
    part fails to start, everything already opened (engine, pool) is closed."""
    migrate_database()
    publish_pdfs(settings.raw_pdf_dir, STATIC_DIR)
    with ExitStack() as resources:
        services = build_services(resources)
        resources.pop_all()  # started: keep it all open
    return LocalBackend(services)
