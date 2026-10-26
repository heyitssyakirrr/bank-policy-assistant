"""The LLM adapter, rate limiter, cache and retry policy. No API calls: a
fake chat model returns scripted raw/parsed responses or raises."""

import threading

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from app.core.config import settings
from app.core.retry import is_transient
from app.llm.cache import ResponseCache
from app.llm.models import (
    ChatModelClient,
    LLMUnavailableError,
    build_chat_model,
    provider_options,
    requests_per_minute,
    role_models,
    thinking_for_model,
)
from app.llm.rate_limit import SlidingWindowLimiter


class Answer(BaseModel):
    text: str


class RateLimited(Exception):
    is_retryable = True


class Coded(Exception):
    def __init__(self, code):
        self.code = code


def ok(text="hi"):
    return {"raw": AIMessage(content="{}", usage_metadata=usage()), "parsed": Answer(text=text), "parsing_error": None}


def invalid():
    return {"raw": AIMessage(content="not json", usage_metadata=usage()), "parsed": None, "parsing_error": "bad json"}


def usage():
    return {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}


class FakeChat:
    def __init__(self, script):
        self.script = script
        self.conversations = []

    def with_structured_output(self, schema, method, include_raw):
        assert include_raw
        return self

    def invoke(self, conversation):
        self.conversations.append(conversation)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


@pytest.fixture(autouse=True)
def no_backoff_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)


def client(scripts, cache=None, clock=lambda: 0.0, cooldown_s=120):
    chats = {model: FakeChat(script) for model, script in scripts.items()}
    llm = ChatModelClient(
        cache=cache,
        build=lambda model, thinking: chats[model],
        models_for=lambda role: ("primary", "backup"),
        thinking_for=lambda role: "low",
        rate_for=lambda model: 1000,
        cooldown_s=cooldown_s,
        clock=clock,
    )
    return llm, chats


MESSAGES = [HumanMessage(content="question")]


def test_valid_output_is_returned_with_its_cost():
    llm, _ = client({"primary": [ok()], "backup": []})
    result = llm.invoke("planner", MESSAGES, Answer)
    assert result.output.text == "hi" and result.model == "primary"
    assert (result.calls, result.input_tokens, result.output_tokens) == (1, 10, 2)
    assert llm.requests == {"primary": 1}


def test_malformed_output_gets_one_repair_retry():
    llm, chats = client({"primary": [invalid(), ok()], "backup": []})
    result = llm.invoke("planner", MESSAGES, Answer)
    assert result.calls == 2 and result.input_tokens == 20
    repair = chats["primary"].conversations[1]
    assert len(repair) == 3 and "did not match" in repair[-1].content


def test_output_still_malformed_after_repair_falls_back_to_the_other_model():
    llm, _ = client({"primary": [invalid(), invalid()], "backup": [ok("from backup")]})
    result = llm.invoke("planner", MESSAGES, Answer)
    assert result.model == "backup" and result.output.text == "from backup"


def test_transient_errors_are_retried_three_times_then_fall_back():
    llm, _ = client({"primary": [RateLimited(), RateLimited(), RateLimited()], "backup": [ok()]})
    assert llm.invoke("planner", MESSAGES, Answer).model == "backup"
    assert llm.requests == {"primary": 3, "backup": 1}


def test_transient_error_then_success_stays_on_the_primary():
    llm, _ = client({"primary": [RateLimited(), ok()], "backup": []})
    assert llm.invoke("planner", MESSAGES, Answer).model == "primary"


def test_a_bad_request_is_not_retried_or_sent_to_the_fallback():
    llm, _ = client({"primary": [ValueError("400 bad request")], "backup": [ok()]})
    with pytest.raises(ValueError):
        llm.invoke("planner", MESSAGES, Answer)
    assert llm.requests == {"primary": 1}


def test_both_models_failing_raises_unavailable():
    llm, _ = client({"primary": [RateLimited()] * 3, "backup": [RateLimited()] * 3})
    with pytest.raises(LLMUnavailableError):
        llm.invoke("planner", MESSAGES, Answer)


def test_cached_response_costs_no_request(tmp_path):
    cache = ResponseCache(tmp_path / "cache.sqlite")
    llm, _ = client({"primary": [ok("first")], "backup": []}, cache)
    first = llm.invoke("planner", MESSAGES, Answer)
    second = llm.invoke("planner", MESSAGES, Answer)
    assert second.cached and second.output == first.output and second.calls == first.calls
    assert llm.requests == {"primary": 1}


def test_an_overloaded_model_is_tried_last_until_its_cooldown_ends():
    now = [0.0]
    llm, _ = client(
        {"primary": [RateLimited()] * 3 + [ok("primary again")], "backup": [ok("b1"), ok("b2")]},
        clock=lambda: now[0],
    )
    assert llm.invoke("planner", MESSAGES, Answer).model == "backup"  # primary failed 3 times
    assert llm.invoke("reviewer", MESSAGES, Answer).model == "backup"  # no new wait on the primary
    assert llm.requests == {"primary": 3, "backup": 2}
    now[0] = 121.0  # the cooldown is over: the primary comes first again
    assert llm.invoke("planner", MESSAGES, Answer).output.text == "primary again"


@pytest.mark.parametrize("outage", [Coded(503), Coded(504), TimeoutError()])
def test_overloads_and_timeouts_start_the_cooldown(outage):
    llm, _ = client({"primary": [outage] * 3, "backup": [ok(), ok()]})
    llm.invoke("planner", MESSAGES, Answer)
    llm.invoke("reviewer", MESSAGES, Answer)
    assert llm.requests == {"primary": 3, "backup": 2}


def test_the_cooldown_lasts_exactly_its_length():
    now = [0.0]
    llm, _ = client({"primary": [Coded(503)] * 3 + [ok("p")], "backup": [ok("b1"), ok("b2")]}, clock=lambda: now[0])
    llm.invoke("planner", MESSAGES, Answer)
    now[0] = 119.9  # still cooling just before the end
    assert llm.invoke("planner", MESSAGES, Answer).output.text == "b2"
    now[0] = 120.0  # over at exactly its length
    assert llm.invoke("planner", MESSAGES, Answer).output.text == "p"


def test_with_both_models_cooling_the_order_stays_and_a_success_ends_the_cooldown():
    llm, _ = client(
        {
            "primary": [Coded(503)] * 3 + [ok("primary is back"), ok("primary first again")],
            "backup": [Coded(503)] * 3 + [invalid(), invalid()],
        }
    )
    with pytest.raises(LLMUnavailableError):
        llm.invoke("planner", MESSAGES, Answer)  # both fail: both cooling
    # Both cooling: primary first; it answers and leaves its cooldown.
    assert llm.invoke("planner", MESSAGES, Answer).output.text == "primary is back"
    assert llm.invoke("planner", MESSAGES, Answer).output.text == "primary first again"


def test_a_cooling_primary_still_answers_when_the_fallback_breaks():
    llm, _ = client({"primary": [Coded(503)] * 3 + [ok("primary")], "backup": [ok("b"), invalid(), invalid()]})
    llm.invoke("planner", MESSAGES, Answer)  # primary cooling
    assert llm.invoke("planner", MESSAGES, Answer).output.text == "primary"  # backup malformed twice


def test_a_single_model_role_is_still_called_while_cooling():
    chat = FakeChat([Coded(503)] * 3 + [ok("cross-check")])
    llm = ChatModelClient(
        build=lambda model, thinking: chat, models_for=lambda role: ("only", "only"), thinking_for=lambda role: "low"
    )
    with pytest.raises(LLMUnavailableError):
        llm.invoke("judge", MESSAGES, Answer)
    assert llm.invoke("judge", MESSAGES, Answer).output.text == "cross-check"


def test_the_primarys_cached_answer_wins_when_both_are_cached(tmp_path):
    cache = ResponseCache(tmp_path / "cache.sqlite")
    down, _ = client({"primary": [Coded(503)] * 3, "backup": [ok("backup")]}, cache)
    down.invoke("planner", MESSAGES, Answer)
    up, _ = client({"primary": [ok("primary")], "backup": []}, cache)  # a new process: the primary first
    up.invoke("planner", MESSAGES, Answer)  # caches the primary's answer
    fresh, _ = client({"primary": [], "backup": []}, cache)
    assert fresh.invoke("planner", MESSAGES, Answer).model == "primary"


def test_after_an_outage_the_primary_is_asked_again_not_replaced_by_the_fallbacks_cache(tmp_path):
    cache = ResponseCache(tmp_path / "cache.sqlite")
    down, _ = client({"primary": [Coded(503)] * 3, "backup": [ok("self-judged")]}, cache)
    down.invoke("judge", MESSAGES, Answer)  # an outage: the fallback answered and was cached
    evaluation, _ = client({"primary": [ok("independent")], "backup": []}, cache, cooldown_s=0)
    result = evaluation.invoke("judge", MESSAGES, Answer)
    assert result.model == "primary" and result.output.text == "independent"


def test_a_malformed_answer_does_not_put_a_model_in_cooldown():
    llm, _ = client({"primary": [invalid(), invalid(), ok("primary")], "backup": [ok("backup")]})
    assert llm.invoke("planner", MESSAGES, Answer).model == "backup"
    assert llm.invoke("planner", [HumanMessage(content="another")], Answer).model == "primary"


def test_while_the_primary_cools_the_fallbacks_cached_answer_costs_nothing(tmp_path):
    cache = ResponseCache(tmp_path / "cache.sqlite")
    llm, _ = client({"primary": [RateLimited()] * 3, "backup": [ok("from backup")]}, cache)
    llm.invoke("planner", MESSAGES, Answer)  # the primary was down: the backup's answer is cached
    result = llm.invoke("planner", MESSAGES, Answer)  # the same question while the primary cools
    assert result.cached and result.model == "backup" and llm.requests == {"primary": 3, "backup": 1}


def test_a_different_prompt_is_a_cache_miss(tmp_path):
    cache = ResponseCache(tmp_path / "cache.sqlite")
    llm, _ = client({"primary": [ok("a"), ok("b")], "backup": []}, cache)
    llm.invoke("planner", MESSAGES, Answer)
    assert llm.invoke("planner", [HumanMessage(content="other")], Answer).output.text == "b"


# ---------------------------------------------------------------- rate limiter


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.slept = []

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def test_limiter_waits_only_when_the_window_is_full():
    clock = FakeClock()
    limiter = SlidingWindowLimiter(3, clock=lambda: clock.now, sleep=clock.sleep)
    for _ in range(3):
        assert limiter.acquire() == 0
    clock.now = 10
    waited = limiter.acquire()  # the 4th call waits until the 1st is 60 s old
    assert waited == pytest.approx(50.05)


def test_limiter_is_sliding_not_fixed_window():
    clock = FakeClock()
    limiter = SlidingWindowLimiter(2, clock=lambda: clock.now, sleep=clock.sleep)
    clock.now = 59
    limiter.acquire()
    limiter.acquire()
    clock.now = 61  # a fixed window would reset at 60 and allow a burst here
    assert limiter.acquire() > 0


def test_limiter_counts_every_call_from_parallel_threads():
    limiter = SlidingWindowLimiter(100)
    threads = [threading.Thread(target=limiter.acquire) for _ in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(limiter._calls) == 40


# ---------------------------------------------------------------- retry policy


@pytest.mark.parametrize(
    ("exc", "transient"),
    [(Coded(429), True), (Coded(503), True), (Coded(400), False), (TimeoutError(), True), (RateLimited(), True)],
)
def test_only_temporary_errors_are_transient(exc, transient):
    assert is_transient(exc) is transient


def test_each_role_gets_its_own_thinking_level_and_cache_key(tmp_path):
    built = []

    def build(model, thinking):
        built.append((model, thinking))
        return FakeChat([ok(thinking)])

    llm = ChatModelClient(
        cache=ResponseCache(tmp_path / "c.sqlite"),
        build=build,
        models_for=lambda role: ("same-model", "same-model"),
        thinking_for=lambda role: "medium" if role == "generator" else "low",
        rate_for=lambda model: 1000,
    )
    assert llm.invoke("planner", MESSAGES, Answer).output.text == "low"
    assert llm.invoke("generator", MESSAGES, Answer).output.text == "medium"  # same prompt, not a cache hit
    assert built == [("same-model", "low"), ("same-model", "medium")]


# ---------------------------------------------------------------- providers (DeepSeek, Gemini)

REQUEST = httpx.Request("POST", "https://api.deepseek.com/chat/completions")


def openai_error(cls, status):
    return cls("rejected", response=httpx.Response(status, request=REQUEST), body=None)


def test_deepseek_gets_its_key_thinking_off_and_no_sdk_retries(monkeypatch):
    monkeypatch.setattr(settings, "deepseek_api_key", "ds-key")
    thinking = thinking_for_model("deepseek:deepseek-flash", "medium")  # the role's Gemini level doesn't apply
    options = provider_options("deepseek:deepseek-flash", thinking)
    assert options == {"api_key": "ds-key", "extra_body": {"thinking": {"type": "disabled"}}, "max_retries": 0}
    built = build_chat_model("deepseek:deepseek-flash", thinking)
    assert built.max_retries == 0 and built.extra_body == {"thinking": {"type": "disabled"}}


def test_gemini_keeps_its_own_key_and_thinking_level(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "g-key")
    thinking = thinking_for_model("google_genai:gemini-3.1-flash-lite", "low")
    options = provider_options("google_genai:gemini-3.1-flash-lite", thinking)
    assert options == {"api_key": "g-key", "thinking_config": {"thinking_level": "low"}, "max_retries": 1}


def test_the_cache_key_carries_what_each_model_actually_uses(tmp_path):
    built = []

    def build(model, thinking):
        built.append((model, thinking))
        return FakeChat([openai_error(openai.BadRequestError, 400)] if model == "deepseek:x" else [ok(thinking)])

    llm = ChatModelClient(
        cache=ResponseCache(tmp_path / "c.sqlite"),
        build=build,
        models_for=lambda role: ("deepseek:x", "google_genai:y"),
        thinking_for=lambda role: "medium",
        rate_for=lambda model: 1000,
    )
    assert llm.invoke("generator", MESSAGES, Answer).output.text == "medium"
    assert built == [("deepseek:x", "disabled"), ("google_genai:y", "medium")]


def test_a_deepseek_role_without_a_key_runs_on_its_gemini_fallback(monkeypatch):
    monkeypatch.setattr(settings, "planner_model", "deepseek:deepseek-flash")
    monkeypatch.setattr(settings, "planner_fallback_model", "google_genai:gemini-3.1-flash-lite")
    monkeypatch.setattr(settings, "deepseek_api_key", "")
    assert role_models("planner") == ("google_genai:gemini-3.1-flash-lite",) * 2
    monkeypatch.setattr(settings, "deepseek_api_key", "ds-key")
    assert role_models("planner") == ("deepseek:deepseek-flash", "google_genai:gemini-3.1-flash-lite")


def test_each_provider_is_paced_by_its_own_limit(monkeypatch):
    monkeypatch.setattr(settings, "deepseek_requests_per_minute", 120)
    monkeypatch.setattr(settings, "llm_requests_per_minute", 15)
    assert requests_per_minute("deepseek:deepseek-flash") == 120
    assert requests_per_minute("google_genai:gemini-3.5-flash-lite") == 15


@pytest.mark.parametrize(
    ("exc", "transient"),
    [
        (openai_error(openai.RateLimitError, 429), True),
        (openai_error(openai.InternalServerError, 503), True),
        (openai.APITimeoutError(request=REQUEST), True),
        (openai.APIConnectionError(request=REQUEST), True),
        (openai_error(openai.BadRequestError, 400), False),
    ],
)
def test_deepseek_sdk_errors_are_classified(exc, transient):
    assert is_transient(exc) is transient


@pytest.mark.parametrize(
    ("cls", "status"),
    [
        (openai.AuthenticationError, 401),
        (openai.APIStatusError, 402),  # insufficient balance: no class of its own in the SDK
        (openai.NotFoundError, 404),
        (openai.BadRequestError, 400),
        (openai.UnprocessableEntityError, 422),
    ],
)
def test_a_request_deepseek_rejects_goes_to_the_fallback_and_cools(cls, status):
    llm, _ = client({"primary": [openai_error(cls, status)], "backup": [ok("from gemini"), ok("again")]})
    result = llm.invoke("planner", MESSAGES, Answer)
    assert result.model == "backup" and result.output.text == "from gemini"
    assert llm.requests == {"primary": 1, "backup": 1}  # not retried: it won't change
    llm.invoke("reviewer", MESSAGES, Answer)  # the refused provider isn't asked again for a while
    assert llm.requests == {"primary": 1, "backup": 2}
