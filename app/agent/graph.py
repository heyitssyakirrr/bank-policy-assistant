"""The agentic graph: the LLM decides the path, code enforces the bounds.

    understand_question ──► route_after_planning
        ├─ out of scope / small talk ──► respond_directly ─────────────────┐
        ├─ unclear ──────────────────► ask_clarification ────────────────┤
        └─ search_policies × N  +  lookup_paragraphs   (parallel, Send)    │
                         └──────► review_evidence ──► route_after_review   │
                                      ▲                 │ not enough      │
                                      └── more searches ┘ (≤ 2 rounds)    │
                                                        │ enough          │
                         generate_answer ◄──────────────┘                 │
                               ▼                                          │
                         verify_answer ── serious problem, once ──► regenerate
                               ▼                                          │
                             finish ◄─────────────────────────────────────┘

It shares `generate_answer`, `verify_answer` and `finish` with the
traditional graph. RECURSION_LIMIT is the last safety net behind the round
counter and the request budget: no path can run more steps than that.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.nodes import finish, generate_answer, route_after_verification, verify_answer
from app.agent.planning import understand_question
from app.agent.responses import ask_clarification, respond_directly
from app.agent.review import review_evidence
from app.agent.routing import route_after_planning, route_after_review
from app.agent.state import AgentContext, AgentState
from app.agent.tools import lookup_paragraphs, search_policies

RECURSION_LIMIT = 25  # the longest real path is 10 steps: plan, 2 x (tools, review), 2 x (generate, verify), finish


def build_agentic_graph(checkpointer: Any = None) -> CompiledStateGraph:
    graph = StateGraph(AgentState, context_schema=AgentContext)
    graph.add_node("understand_question", understand_question)
    graph.add_node("respond_directly", respond_directly)
    graph.add_node("ask_clarification", ask_clarification)
    graph.add_node("search_policies", search_policies)
    graph.add_node("lookup_paragraphs", lookup_paragraphs)
    graph.add_node("review_evidence", review_evidence)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("verify_answer", verify_answer)
    graph.add_node("finish", finish)

    tools_or_answer = ["search_policies", "lookup_paragraphs", "generate_answer"]
    graph.add_edge(START, "understand_question")
    graph.add_conditional_edges(
        "understand_question",
        route_after_planning,
        ["respond_directly", "ask_clarification", "search_policies", "lookup_paragraphs"],
    )
    graph.add_edge("search_policies", "review_evidence")
    graph.add_edge("lookup_paragraphs", "review_evidence")
    graph.add_conditional_edges("review_evidence", route_after_review, tools_or_answer)
    graph.add_edge("generate_answer", "verify_answer")
    graph.add_conditional_edges("verify_answer", route_after_verification, ["generate_answer", "finish"])
    graph.add_edge("respond_directly", "finish")
    graph.add_edge("ask_clarification", "finish")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer).with_config(recursion_limit=RECURSION_LIMIT)
