"""The one adapter every LLM call goes through.

    role ("planner") ──► the role's models, a cooling one last ──► each: cached answer? ──► done
         ──► rate limiter (shared per model)
         ──► call with live retries ──► validated Pydantic object
                     └─ malformed JSON: 1 repair retry
         primary model failing (quota, outage) ──► the role's fallback model
         a model that just failed with an outage or overload is tried last for
         a cooldown, so every step doesn't pay its retries again

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

import openai
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
# transient): invalid output, this model being unknown or not enabled, or any
# refusal by the OpenAI-compatible API (DeepSeek: a bad key, 402 "insufficient
# balance", an option it rejects): the fallback is another provider.
_MODEL_SPECIFIC_ERRORS = (StructuredOutputError, ModelNotFoundError, ModelPermissionDeniedError, openai.APIStatusError)


def role_models(role: Role) -> tuple[str, str]:
    """(primary, fallback) for a role, from settings. A DeepSeek primary without
    a DeepSeek key is skipped, so a key-less setup (tests, CI, a fresh clone)
    runs on the Gemini fallback instead of failing every call."""
    primary, fallback = getattr(settings, f"{role}_model"), getattr(settings, f"{role}_fallback_model")
    return (primary, fallback) if configured(primary) else (fallback, fallback)


def configured(model: str) -> bool:
    return not is_deepseek(model) or bool(settings.deepseek_api_key)


def is_deepseek(model: str) -> bool:
    return model.startswith("deepseek:")


def role_thinking(role: Role) -> str:
    return getattr(settings, f"{role}_thinking_level")


# DeepSeek thinks unless told not to. With thinking on, it requires the model's
# reasoning to be sent back on any follow-up request, which LangChain doesn't do
# (the repair retry would be refused), so it is always off.
DEEPSEEK_THINKING = "disabled"


def thinking_for_model(model: str, role_level: str) -> str:
    """What the model actually uses: DeepSeek's fixed switch, or the role's Gemini level."""
    return DEEPSEEK_THINKING if is_deepseek(model) else role_level


def provider_options(model: str, thinking: str) -> dict[str, Any]:
    """Each provider's key, reasoning switch and SDK retries (ours are retry_live,
    so the SDK must not add its own: the OpenAI SDK's max_retries=1 is 2 attempts)."""
    if is_deepseek(model):
        return {
            "api_key": settings.deepseek_api_key or None,
            "extra_body": {"thinking": {"type": thinking}},
            "max_retries": 0,
        }
    if model.startswith("google_genai:"):
        return {
            "api_key": settings.gemini_api_key or None,
            "thinking_config": {"thinking_level": thinking},
            "max_retries": 1,  # the Gemini client's single attempt
        }
    return {}


def build_chat_model(model: str, thinking: str) -> BaseChatModel:
    """ "deepseek:deepseek-flash" or "google_genai:gemini-3.1-flash-lite" -> a LangChain chat model."""
    return init_chat_model(model, timeout=settings.llm_timeout_s, **provider_options(model, thinking))


def quota_name(model: str) -> str:
    """'google_genai:gemini-3.1-flash-lite' -> 'gemini-3.1-flash-lite': quotas are per model."""
    return model.split(":", 1)[-1]


def requests_per_minute(model: str) -> int:
    """The pace for a model: DeepSeek is paid with no fixed limit; Gemini's free tier allows 15."""
    return settings.deepseek_requests_per_minute if is_deepseek(model) else settings.llm_requests_per_minute


def _serialise(messages: Sequence[BaseMessage]) -> list[tuple[str, Any]]:
    return [(m.type, m.content) for m in messages]


def _tokens(raw: Any) -> tuple[int, int]:
    usage = getattr(raw, "usage_metadata", None) or {}
    return usage.get("input_tokens", 0), usage.get("output_tokens", 0)


class ChatModelClient:
    def __init__(
        self,
        cache: ResponseCache | None = None,
        build: Callable[[str, str], BaseChatModel] = build_chat_model,
        models_for: Callable[[Role], tuple[str, str]] = role_models,
        thinking_for: Callable[[Role], str] = role_thinking,
        method: str = settings.structured_output_method,
        rate_for: Callable[[str], int] = requests_per_minute,
        cooldown_s: float = settings.llm_cooldown_s,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Each model's cached answer is checked when that model's turn comes, in the
        order the models are tried: the primary first, unless it is cooling after
        a failure (then its fallback's cached answer is used straight away).
        Evaluation passes `cooldown_s=0`, so the primary is always asked first and
        every answer is attributed to the model that should give it."""
        self._cache = cache
        self._build = build
        self._models_for = models_for
        self._thinking_for = thinking_for
        self._method = method
        self._rate_for = rate_for
        self._cooldown_s = cooldown_s
        self._clock = clock
        self._cooling_until: dict[str, float] = {}  # model -> when it may be tried first again
        self._chat_models: dict[tuple[str, str], BaseChatModel] = {}
        self._lock = threading.Lock()
        self.requests: Counter[str] = Counter()  # API requests sent per model (the quota ledger)

    def invoke[T: BaseModel](self, role: Role, messages: Sequence[BaseMessage], schema: type[T]) -> LLMResult[T]:
        models = list(dict.fromkeys(self._models_for(role)))  # same model twice = no fallback
        thinking = {model: thinking_for_model(model, self._thinking_for(role)) for model in models}
        keys = {model: self._cache_key(model, thinking[model], messages, schema) for model in models}
        errors: list[str] = []
        failed_calls = 0
        for model in sorted(models, key=self._cooling):  # stable: a cooling model just moves last
            if (hit := self._cached(keys[model], schema, model)) is not None:
                return hit
            try:
                result = self._invoke_model(model, thinking[model], messages, schema, keys[model])
            except Exception as exc:
                transient = is_transient(exc)
                if not (transient or isinstance(exc, _MODEL_SPECIFIC_ERRORS)):
                    raise  # e.g. a bug in our own request: another model won't fix it
                # Overloaded, out of quota, or refused by the provider (key, balance): spare the next steps.
                if (transient or isinstance(exc, openai.APIStatusError)) and self._cooldown_s:
                    with self._lock:
                        self._cooling_until[model] = self._clock() + self._cooldown_s
                level = logging.WARNING if transient else logging.ERROR  # a permanent refusal needs fixing
                logger.log(level, f"{role} on {model} failed ({type(exc).__name__}: {exc}); trying fallback")
                errors.append(f"{model}: {type(exc).__name__}")
                failed_calls += getattr(exc, "calls", 1)
                continue
            with self._lock:  # it answered: back to its normal place
                self._cooling_until.pop(model, None)
            return result
        raise LLMUnavailableError(f"{role}: every model failed ({'; '.join(errors)})", failed_calls)

    def _cooling(self, model: str) -> bool:
        with self._lock:
            return self._cooling_until.get(model, 0.0) > self._clock()

    @staticmethod
    def _cache_key(model: str, thinking: str, messages: Sequence[BaseMessage], schema: type[BaseModel]) -> str:
        request = {"schema": schema.model_json_schema(), "messages": _serialise(messages)}
        return cache_key("llm", f"{model}|thinking={thinking}", request)

    def _cached[T: BaseModel](self, key: str, schema: type[T], model: str) -> LLMResult[T] | None:
        if self._cache is None or (hit := self._cache.get(key)) is None:
            return None
        return LLMResult(schema.model_validate(hit["output"]), model, hit["calls"], *hit["tokens"], 0.0, True)

    def _invoke_model[T: BaseModel](
        self, model: str, thinking: str, messages: Sequence[BaseMessage], schema: type[T], key: str
    ) -> LLMResult[T]:
        runnable = self._chat_model(model, thinking).with_structured_output(
            schema, method=self._method, include_raw=True
        )
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
        shared_limiter(quota_name(model), self._rate_for(model)).acquire()
        with self._lock:
            self.requests[model] += 1
        return runnable.invoke(conversation)

    def _chat_model(self, model: str, thinking: str) -> BaseChatModel:
        with self._lock:
            if (model, thinking) not in self._chat_models:
                self._chat_models[model, thinking] = self._build(model, thinking)
            return self._chat_models[model, thinking]
