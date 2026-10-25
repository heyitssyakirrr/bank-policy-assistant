"""Running one turn: the single entry point the API uses.

    route:  runner.start(...)   PII masked, conversation ensured (DB errors -> HTTP error)
    stream: runner.stream(...)  graph.stream ──► one "step" event per node
                                       └─► final state ──► audit rows ──► "answer" event

Events are plain dicts, so the API can send them as Server-Sent Events:
  {"type": "start",  "conversation_id", "turn_id", "mode", "pii_masked"}
  {"type": "step",   "node", "title", "lines", "ms"}
  {"type": "answer", "turn": <TurnView>}
  {"type": "error",  "message", "turn_id"}

Every turn is audited, whatever happens: an answer, a crash ("error"), or
the client leaving mid-stream ("cancelled", with the steps it got through).

A public deployment can cap the questions per quota day (all visitors
together): `start` counts today's turns in the audit log and refuses the
question with DailyLimitReached once the limit is reached.

A conversation is one checkpointer thread per mode ("<conversation>:agentic"),
so switching modes or the Compare page never mixes two graphs' state. One
turn at a time per thread: a second request while one runs is refused.
"""

from __future__ import annotations

import gzip
import hashlib
import logging
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from langgraph.graph.state import CompiledStateGraph
from pydantic_core import to_jsonable_python

from app.agent.prompts import PROMPT_VERSION
from app.agent.schemas import Mode
from app.agent.state import AgentContext, turn_input
from app.agent.steps import STEP_TITLES, describe_step
from app.api.views import turn_view
from app.core.logging import conversation_id_var, turn_id_var
from app.core.pii import mask_pii
from app.db.audit_repository import AuditRepository
from app.db.models import Turn, TurnEvent, utcnow

logger = logging.getLogger("app.runner")

BUSY_MESSAGE = "A question is already being answered in this conversation. Please wait for it to finish."
FAILED_MESSAGE = "Something went wrong answering this question."
DAILY_LIMIT_MESSAGE = (
    "Today's question limit for this demo has been reached. It resets at 3 pm Malaysia time "
    "(4 pm from November); "
    "questions asked before are still shown in your history."
)


class DailyLimitReached(Exception):
    """The deployment's questions for this quota day are used up."""


def quota_day_start(now: datetime, quota_timezone: str) -> datetime:
    """When the current quota day began: midnight in the quota's time zone (Gemini's
    free tier resets at midnight Pacific time, so daylight saving moves it), in UTC."""
    local = now.astimezone(ZoneInfo(quota_timezone))
    return local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(UTC)


@lru_cache(maxsize=4)
def index_version(chunks_path: Path) -> str:
    """A short hash of the indexed chunks, saved with every turn: which
    version of the documents an answer was based on."""
    for path, opener in ((chunks_path, open), (chunks_path.with_suffix(".jsonl.gz"), gzip.open)):
        if path.exists():
            with opener(path, "rb") as handle:
                return hashlib.sha256(handle.read()).hexdigest()[:12]
    logger.warning(f"{chunks_path} not found: turns will record index_version='unknown'")
    return "unknown"


def _jsonable(value: Any) -> Any:
    return to_jsonable_python(value, fallback=str)


def _step_error(update: dict[str, Any]) -> str | None:
    """A tool failure reported inside a step (see tools._failed)."""
    errors = [f"{r['kind']} '{r['query']}': {r['error']}" for r in update.get("search_log", []) if "error" in r]
    return "; ".join(errors) or None


@dataclass(frozen=True, slots=True)
class TurnStart:
    conversation_id: uuid.UUID
    turn_id: uuid.UUID
    question: str  # after PII masking
    pii_masked: list[str]


class TurnRunner:
    def __init__(
        self,
        graphs: dict[str, CompiledStateGraph],
        context: AgentContext,
        audit: AuditRepository,
        index_version: str,
        daily_limit: int = 0,
        quota_timezone: str = "America/Los_Angeles",
        now: Callable[[], datetime] = utcnow,
    ) -> None:
        self._graphs = graphs
        self._context = context
        self._audit = audit
        self._index_version = index_version
        self._daily_limit = daily_limit  # 0 = no limit
        self._quota_timezone = quota_timezone
        self._now = now
        self._thread_locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
        self._locks_guard = threading.Lock()

    def start(self, question: str, sector: str | None, conversation_id: uuid.UUID | None) -> TurnStart:
        """Check the daily limit, mask PII and make sure the conversation
        exists. Called before the response starts, so a refusal or a database
        error becomes a proper HTTP error."""
        if self._daily_limit:
            since = quota_day_start(self._now(), self._quota_timezone)
            if self._audit.count_turns_since(since) >= self._daily_limit:
                raise DailyLimitReached(DAILY_LIMIT_MESSAGE)
        masked = mask_pii(question)
        conversation_id = self._audit.ensure_conversation(conversation_id, masked.text, sector)
        return TurnStart(conversation_id, uuid.uuid4(), masked.text, masked.masked)

    def stream(self, start: TurnStart, mode: Mode, sector: str | None) -> Iterator[dict[str, Any]]:
        thread_id = f"{start.conversation_id}:{mode}"
        with self._locks_guard:
            lock = self._thread_locks[thread_id]
        if not lock.acquire(blocking=False):
            yield {"type": "error", "message": BUSY_MESSAGE, "turn_id": str(start.turn_id)}
            return
        try:
            yield from self._run(start, mode, sector, thread_id)
        finally:
            lock.release()

    def _run(self, start: TurnStart, mode: Mode, sector: str | None, thread_id: str) -> Iterator[dict[str, Any]]:
        def bind_log_context() -> None:  # each SSE chunk may resume in another thread's context copy
            conversation_id_var.set(str(start.conversation_id))
            turn_id_var.set(str(start.turn_id))

        bind_log_context()
        graph = self._graphs[mode]
        config = {
            "configurable": {"thread_id": thread_id},
            "run_name": f"{mode} turn",
            "metadata": {"conversation_id": str(start.conversation_id), "turn_id": str(start.turn_id), "mode": mode},
        }
        started = time.perf_counter()
        events: list[TurnEvent] = []
        error: str | None = None
        status: str | None = None  # None = the answer's own status
        graph_ran = False  # before it runs, the thread's saved state is the previous turn's
        try:  # from the first event on, so a client leaving at any point is audited
            yield {
                "type": "start",
                "conversation_id": str(start.conversation_id),
                "turn_id": str(start.turn_id),
                "mode": mode,
                "pii_masked": start.pii_masked,
            }
            bind_log_context()
            graph_ran = True
            for chunk in graph.stream(
                turn_input(start.question, sector), config, context=self._context, stream_mode="updates"
            ):
                for node, update in chunk.items():
                    update = update or {}
                    lines = describe_step(node, update)
                    ms = int((time.perf_counter() - started) * 1000)
                    event = TurnEvent(
                        turn_id=start.turn_id,
                        seq=len(events),
                        node=node,
                        lines=lines,
                        output=_jsonable(update),
                        ms=ms,
                        error=_step_error(update),
                    )
                    events.append(event)
                    yield {"type": "step", "node": node, "title": STEP_TITLES.get(node, node), "lines": lines, "ms": ms}
                    bind_log_context()
        except GeneratorExit:  # the client left: record what happened so far, then stop
            error, status = "client disconnected", "cancelled"
            raise
        except Exception as exc:
            logger.exception("turn failed")
            error, status = f"{type(exc).__name__}: {exc}", "error"
            ms = int((time.perf_counter() - started) * 1000)
            events.append(
                TurnEvent(turn_id=start.turn_id, seq=len(events), node="error", lines=[], output={}, ms=ms, error=error)
            )
        finally:
            state = self._final_state(graph, config) if graph_ran else {}
            turn = self._turn_row(start, mode, state, started, error, status)
            saved = self._save(turn, events)

        if error:
            yield {"type": "error", "message": FAILED_MESSAGE, "turn_id": str(start.turn_id)}
            return
        view = turn_view(turn, events).model_dump(mode="json")
        if not saved:
            view["notes"] = [*view["notes"], "This answer could not be saved to the conversation history."]
        yield {"type": "answer", "turn": view}

    def _final_state(self, graph: CompiledStateGraph, config: dict[str, Any]) -> dict[str, Any]:
        """The state at the end, or as far as the turn got (so a failed turn
        still records the LLM calls it spent)."""
        try:
            return dict(graph.get_state(config).values)
        except Exception:
            logger.exception("could not read the final state")
            return {}

    def _save(self, turn: Turn, events: list[TurnEvent]) -> bool:
        try:
            self._audit.record_turn(turn, events)
            return True
        except Exception:
            logger.exception("could not write the audit rows")
            return False

    def _turn_row(
        self,
        start: TurnStart,
        mode: str,
        state: dict[str, Any],
        started: float,
        error: str | None,
        status: str | None,
    ) -> Turn:
        answer = state.get("answer") if not error else None
        plan = state.get("plan")
        return Turn(
            id=start.turn_id,
            conversation_id=start.conversation_id,
            mode=mode,
            question=start.question,
            pii_masked=start.pii_masked,
            standalone_question=plan.standalone_question if plan else None,
            sector=state.get("sector"),
            plan=_jsonable(plan),
            answer=_jsonable(answer) or {},
            outcome=state.get("outcome", "answer"),
            status=status or (answer.status if answer else "error"),
            verification=_jsonable(state.get("verifications", [])),
            evidence=_jsonable(state.get("evidence", [])),
            notes=state.get("notes", []),
            models=state.get("models", []),
            prompt_version=PROMPT_VERSION,
            index_version=self._index_version,
            llm_calls=state.get("llm_calls", 0),
            input_tokens=state.get("input_tokens", 0),
            output_tokens=state.get("output_tokens", 0),
            latency_ms=int((time.perf_counter() - started) * 1000),
            error=error,
        )
