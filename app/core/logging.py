"""Structured logs: every line carries the request, conversation and turn IDs,
so one question can be followed across the API, the agent and the database."""

import logging
import time
import uuid
from contextvars import ContextVar

from starlette.types import ASGIApp, Message, Receive, Scope, Send

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
conversation_id_var: ContextVar[str] = ContextVar("conversation_id", default="-")
turn_id_var: ContextVar[str] = ContextVar("turn_id", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        record.conversation_id = conversation_id_var.get()[:8]
        record.turn_id = turn_id_var.get()[:8]
        return True


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | req=%(request_id)s conv=%(conversation_id)s turn=%(turn_id)s"
            " | %(name)s | %(message)s"
        )
    )
    handler.addFilter(RequestIdFilter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    # The SDK warns about "automatic function calling" on every structured call; we don't use it.
    logging.getLogger("google_genai.models").setLevel(logging.ERROR)


logger = logging.getLogger("app")


class RequestContextMiddleware:
    """Gives each HTTP request an ID (logged, and returned as X-Request-ID).
    A plain ASGI middleware, so the ID also covers a streamed (SSE) body."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = uuid.uuid4().hex[:8]
        token = request_id_var.set(request_id)
        started = time.perf_counter()

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                message["headers"] = [*message.get("headers", []), (b"x-request-id", request_id.encode())]
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            logger.info(
                f"{scope['method']} {scope['path']} completed in {(time.perf_counter() - started) * 1000:.0f}ms"
            )
            request_id_var.reset(token)
