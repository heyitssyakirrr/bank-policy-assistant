"""What the pages need from the backend, whichever one serves them.

    UI_BACKEND=api    ApiClient: HTTP + Server-Sent Events to FastAPI (local, Docker)
    UI_BACKEND=local  LocalBackend: the same TurnRunner, in this process (Streamlit Cloud)

Both speak the API's event contract: "start", one "step" per graph node,
then "answer" (or "error"), and return turns as TurnView dicts. The pages
only ever see this protocol.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from typing import Any, Protocol

import streamlit as st

from app.core.config import settings


class Backend(Protocol):
    def chat(
        self, question: str, mode: str, sector: str | None, conversation_id: str | None = None
    ) -> Iterator[dict[str, Any]]: ...

    def conversations(self, ids: Sequence[str]) -> list[dict[str, Any]]:
        """These conversations (one visitor's), most recent first."""
        ...

    def conversation(self, conversation_id: str) -> list[dict[str, Any]]: ...

    def health(self) -> dict[str, Any]: ...

    def feedback(self, turn_id: str, rating: int) -> bool: ...

    def pdf_link(self, pdf_url: str) -> str:
        """A link the browser opens directly: the PDF at the cited page."""
        ...


START_FAILED_MESSAGE = (
    "The assistant could not start: one of its services (search index or database) is unreachable. "
    "Please try again in a few minutes."
)
logger = logging.getLogger("app.ui")


@st.cache_resource(show_spinner="Starting the assistant…")
def _build() -> Backend:
    """One per Streamlit process. The local backend is imported only when
    chosen: it loads the whole agent stack. A failure isn't cached, so the
    next visit tries again."""
    if settings.ui_backend == "local":
        from ui.local_backend import build_local_backend

        return build_local_backend()
    from ui.api_client import ApiClient

    return ApiClient(settings.api_url, settings.public_api_url)


def backend() -> Backend:
    """The page's backend, or a clear message and an end to the page."""
    try:
        return _build()
    except Exception:
        logger.exception("the backend failed to start")
        st.error(START_FAILED_MESSAGE, icon=":material/cloud_off:")
        st.stop()
