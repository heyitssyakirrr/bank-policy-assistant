"""The traditional RAG graph: one fixed line, the code decides everything.

    retrieve ──► generate_answer ──► verify_answer ──► finish ──► END
                        ▲                  │ serious problem (once)
                        └──── feedback ────┘

The search uses the question as typed, the user's sector, all documents,
and any paragraph numbers the question contains (found by regex).
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from app.agent.evidence import to_evidence
from app.agent.nodes import finish, generate_answer, route_after_verification, search_record, verify_answer
from app.agent.state import AgentContext, AgentState


def retrieve(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    result = runtime.context.search.search(state["question"], sector=state.get("sector"))
    return {
        "evidence": to_evidence(result.final, "search", result.question),
        "search_log": [search_record(result)],
    }


def build_traditional_graph(checkpointer: Any = None) -> CompiledStateGraph:
    graph = StateGraph(AgentState, context_schema=AgentContext)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("verify_answer", verify_answer)
    graph.add_node("finish", finish)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate_answer")
    graph.add_edge("generate_answer", "verify_answer")
    graph.add_conditional_edges("verify_answer", route_after_verification, ["generate_answer", "finish"])
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer)
