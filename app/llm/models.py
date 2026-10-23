"""The one adapter every LLM call goes through.

    role ("planner") ──► model from settings ──► rate limiter (shared per model)
         ──► cache? ──► call with live retries ──► validated Pydantic object
                     └─ malformed JSON: 1 repair retry
         primary model failing (quota, outage) ──► the role's fallback model

Nodes depend on the `StructuredLLM` protocol, never on a provider: the real
`ChatModelClient` in the app, a scripted fake in tests.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from langchain.chat_models import init_chat_model
from langchain_core.exceptions import ModelNotFoundError, ModelPermissionDeniedError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel

from app.core.config import settings
from app.core.retry import is_transient, retry_live
from app.llm.cache import ResponseCache, cache_key
from app.llm.rate_limit import shared_limiter

logger = logging.getLogger("app.llm")

Role = Literal["planner", "reviewer", "generator", "judge"]
REPAIR_ATTEMPTS = 1


@dataclass(frozen=True, slots=True)
class LLMResult[T: BaseModel]:
    output: T
    model: str  # the model that produced it (the fallback, if the primary failed)
    # LLM calls this result represents, including a repair retry. A cached
    # result reports the calls it originally took, so the request budget and
    # the graph's path are identical with or without the cache (`cached` says
    # whether any API request was actually sent).
    calls: int
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cached: bool = False


class StructuredLLM(Protocol):
    def invoke[T: BaseModel](self, role: Role, messages: Sequence[BaseMessage], schema: type[T]) -> LLMResult[T]: ...


class StructuredOutputError(Exception):
    """The model kept returning output that doesn't match the schema."""

    def __init__(self, message: str, calls: int) -> None:
        super().__init__(message)
        self.calls = calls


class LLMUnavailableError(Exception):
    """Both the role's model and its fallback failed. `calls` is what the
    failed attempts cost, so the request budget still counts them."""

    def __init__(self, message: str, calls: int) -> None:
        super().__init__(message)
        self.calls = calls


# Errors another model can avoid (besides quota and outages, which are
# transient): invalid output, or this model being unknown or not enabled.
_MODEL_SPECIFIC_ERRORS = (StructuredOutputError, ModelNotFoundError, ModelPermissionDeniedError)


def role_models(role: Role) -> tuple[str, str]:
    """(primary, fallback) for a role, from settings."""
    return getattr(settings, f"{role}_model"), getattr(settings, f"{role}_fallback_model")


def build_chat_model(model: str) -> BaseChatModel:
    """ "google_genai:gemini-3.1-flash-lite" -> a LangChain chat model. Retries
    are ours (retry_live), so the SDK makes a single attempt per request."""
    provider_options = (
        {"thinking_config": {"thinking_level": settings.llm_thinking_level}}
        if model.startswith("google_genai:")
        else {}
    )
    return init_chat_model(
        model,
        api_key=settings.gemini_api_key or None,
        timeout=settings.llm_timeout_s,
        max_retries=1,
        **provider_options,
    )


def quota_name(model: str) -> str:
    """'google_genai:gemini-3.1-flash-lite' -> 'gemini-3.1-flash-lite': quotas are per model."""
    return model.split(":", 1)[-1]


def _serialise(messages: Sequence[BaseMessage]) -> list[tuple[str, Any]]:
    return [(m.type, m.content) for m in messages]


def _tokens(raw: Any) -> tuple[int, int]:
    usage = getattr(raw, "usage_metadata", None) or {}
    return usage.get("input_tokens", 0), usage.get("output_tokens", 0)


class ChatModelClient:
    def __init__(
        self,
        cache: ResponseCache | None = None,
        build: Callable[[str], BaseChatModel] = build_chat_model,
        models_for: Callable[[Role], tuple[str, str]] = role_models,
        method: str = settings.structured_output_method,
        requests_per_minute: int = settings.llm_requests_per_minute,
    ) -> None:
        self._cache = cache
        self._build = build
        self._models_for = models_for
        self._method = method
        self._rpm = requests_per_minute
        self._chat_models: dict[str, BaseChatModel] = {}
        self._lock = threading.Lock()
        self.requests: Counter[str] = Counter()  # API requests sent per model (the quota ledger)

    def invoke[T: BaseModel](self, role: Role, messages: Sequence[BaseMessage], schema: type[T]) -> LLMResult[T]:
        primary, fallback = self._models_for(role)
        errors: list[str] = []
        failed_calls = 0
        for model in dict.fromkeys((primary, fallback)):  # same model twice = no fallback
            try:
                return self._invoke_model(model, messages, schema)
            except Exception as exc:
                if not (isinstance(exc, _MODEL_SPECIFIC_ERRORS) or is_transient(exc)):
                    raise  # a bad request or bad key won't be fixed by another model
                logger.warning(f"{role} on {model} failed ({type(exc).__name__}: {exc}); trying fallback")
                errors.append(f"{model}: {type(exc).__name__}")
                failed_calls += getattr(exc, "calls", 1)
        raise LLMUnavailableError(f"{role}: every model failed ({'; '.join(errors)})", failed_calls)

    def _invoke_model[T: BaseModel](self, model: str, messages: Sequence[BaseMessage], schema: type[T]) -> LLMResult[T]:
        request = {"schema": schema.model_json_schema(), "messages": _serialise(messages)}
        key = cache_key("llm", f"{model}|thinking={settings.llm_thinking_level}", request)
        if self._cache is not None and (hit := self._cache.get(key)) is not None:
            return LLMResult(schema.model_validate(hit["output"]), model, hit["calls"], *hit["tokens"], 0.0, True)

        runnable = self._chat_model(model).with_structured_output(schema, method=self._method, include_raw=True)
        conversation = list(messages)
        input_tokens = output_tokens = calls = 0
        started = time.perf_counter()
        for _ in range(1 + REPAIR_ATTEMPTS):
            response = self._call(model, runnable, conversation)
            calls += 1
            tokens = _tokens(response["raw"])
            input_tokens += tokens[0]
            output_tokens += tokens[1]
            parsed = response.get("parsed")
            if isinstance(parsed, schema):
                latency_ms = (time.perf_counter() - started) * 1000
                if self._cache is not None:
                    value = {"output": parsed.model_dump(mode="json"), "calls": calls}
                    self._cache.put(key, value | {"tokens": [input_tokens, output_tokens]})
                return LLMResult(parsed, model, calls, input_tokens, output_tokens, latency_ms)
            error = response.get("parsing_error") or "no structured output returned"
            logger.warning(f"{model} returned invalid {schema.__name__}: {error}; repairing")
            raw = response["raw"]
            # Plain text, without tool calls: a reply to a tool call would need a ToolMessage.
            previous = (raw.content or getattr(raw, "tool_calls", "")) if isinstance(raw, AIMessage) else raw
            conversation += [
                AIMessage(content=str(previous)),
                HumanMessage(
                    content=f"Your reply did not match the required JSON schema ({error}). "
                    f"Reply again with only a valid {schema.__name__} object."
                ),
            ]
        raise StructuredOutputError(f"{model}: invalid {schema.__name__} after {calls} attempts", calls)

    @retry_live()
    def _call(self, model: str, runnable: Any, conversation: list[BaseMessage]) -> dict[str, Any]:
        shared_limiter(quota_name(model), self._rpm).acquire()
        with self._lock:
            self.requests[model] += 1
        return runnable.invoke(conversation)

    def _chat_model(self, model: str) -> BaseChatModel:
        with self._lock:
            if model not in self._chat_models:
                self._chat_models[model] = self._build(model)
            return self._chat_models[model]
