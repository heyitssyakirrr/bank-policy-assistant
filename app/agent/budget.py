"""The request budget: a hard ceiling on what one question may cost.

PROJECT_PLAN §5.6: at most 5 LLM calls, 6 searches and 60 seconds per
question. The prompt never enforces a limit; code does. When the budget
runs out, the graph stops looping and answers with what it has, adding a
note so the user (and the audit log) can see why.

5 LLM calls = planner + 2 reviews + generator + 1 regeneration. A repair
retry for malformed output counts as a call too, so a repaired planner
leaves room for only one review or no regeneration.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

State = Mapping[str, Any]  # the graph state (AgentState); kept generic to avoid an import cycle

MAX_REVIEW_ROUNDS = 2
MAX_SEARCHES_PER_PLAN = 3


@dataclass(frozen=True, slots=True)
class Budget:
    max_llm_calls: int = 5
    max_searches: int = 6
    max_seconds: float = 60.0

    def llm_calls_left(self, state: State) -> int:
        return self.max_llm_calls - state.get("llm_calls", 0)

    def searches_left(self, state: State) -> int:
        return self.max_searches - len(state.get("search_log", []))

    def out_of_time(self, state: State, now: float) -> bool:
        return now - state.get("started_at", now) >= self.max_seconds
