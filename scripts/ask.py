"""Ask a question from the command line and see the verified answer.

    python -m scripts.ask "What must a bank do if FAR exceeds 5%?"
    python -m scripts.ask "What does e-KYC 8.10 require?" --mode agentic --steps
    python -m scripts.ask "Can banks onboard remotely?" --mode agentic --follow-up "what about takaful operators?"
    python -m scripts.ask "..." --no-cache        # force fresh API calls

Uses the response cache by default, so asking the same question twice is
free. Prints the answer, every citation, the verification reports and the
cost (LLM calls, tokens, API requests per model). Follow-up questions run
in the same conversation (an in-memory checkpointer), as the API does.
"""

from __future__ import annotations

import argparse
import sys
import time

from langgraph.checkpoint.memory import InMemorySaver

from app.agent.checkpoint import state_serializer
from app.agent.graph import build_agentic_graph
from app.agent.prompts import applies_to
from app.agent.schemas import DISCLAIMER, SECTOR_LABELS
from app.agent.state import AgentContext, turn_input
from app.agent.steps import describe_step
from app.agent.traditional import build_traditional_graph
from app.core.config import enable_tracing, settings
from app.core.logging import setup_logging
from app.core.pii import mask_pii
from app.llm.cache import ResponseCache
from app.llm.models import ChatModelClient
from app.services.search_service import build_search_service

OBLIGATION_BADGE = {"must": "MUST", "should_or_may": "MAY ", "information": "INFO"}
BUILDERS = {"traditional": build_traditional_graph, "agentic": build_agentic_graph}


def print_answer(state: dict) -> None:
    answer, evidence = state["answer"], {e.id: e for e in state["evidence"]}
    label = answer.status if state["outcome"] == "answer" else state["outcome"]
    print(f"[{label.upper()}] {answer.summary}")
    for statement in answer.statements:
        print(f"  {OBLIGATION_BADGE[statement.obligation]}  {statement.text}  [{', '.join(statement.evidence_ids)}]")
        print(f'        "{statement.quote}"')
    if answer.gaps:
        print(f"  Gaps: {answer.gaps}")
    cited = dict.fromkeys(i for s in answer.statements for i in s.evidence_ids)
    if cited:
        print("Sources:")
        for evidence_id in cited:
            print(f"  {evidence_id}  {evidence[evidence_id].citation}  (PDF page {evidence[evidence_id].page})")
    for number, report in enumerate(state["verifications"], start=1):
        print(f"Verification {number}: {'passed' if report.passed else 'FAILED'}")
        for issue in report.fixes + report.serious:
            print(f"  - {issue.check}: {issue.detail}")
    for note in state["notes"]:
        print(f"Note: {note}")
    print(f"Applies to: {applies_to(state.get('sector'))}. {DISCLAIMER}")


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("question", nargs="+")
    cli.add_argument("--mode", choices=list(BUILDERS), default="traditional")
    cli.add_argument("--sector", choices=[*SECTOR_LABELS, "all"], default=settings.default_sector)
    cli.add_argument("--steps", action="store_true", help="print each graph step as it happens")
    cli.add_argument("--follow-up", action="append", default=[], help="another question in the same conversation")
    cli.add_argument("--no-cache", action="store_true", help="don't read or write the response cache")
    args = cli.parse_args()
    setup_logging("WARNING")
    enable_tracing()
    sys.stdout.reconfigure(encoding="utf-8")  # the Windows console defaults to cp1252

    cache = None if args.no_cache else ResponseCache(settings.response_cache_path)
    llm = ChatModelClient(cache=cache)
    context = AgentContext(llm=llm, search=build_search_service(cache))
    graph = BUILDERS[args.mode](checkpointer=InMemorySaver(serde=state_serializer()))
    config = {"configurable": {"thread_id": "cli"}}
    sector = None if args.sector == "all" else args.sector

    for typed in [" ".join(args.question), *args.follow_up]:
        question = mask_pii(typed).text  # before Gemini and LangSmith, as the API does
        print(f"\nQ: {question}  ({args.mode})")
        started = time.perf_counter()
        for chunk in graph.stream(turn_input(question, sector), config, context=context, stream_mode="updates"):
            for node, update in chunk.items():
                for line in describe_step(node, update or {}) if args.steps else []:
                    print(f"  {time.perf_counter() - started:5.1f}s · {line}")
        seconds = time.perf_counter() - started
        state = graph.get_state(config).values
        print_answer(state)
        print(
            f"{seconds:.1f}s | LLM calls {state['llm_calls']} | "
            f"tokens {state['input_tokens']} in / {state['output_tokens']} out | models {state['models']}"
        )
    print(f"\nAPI requests sent: {dict(llm.requests) or 'none (all cached)'}")
    if cache is not None:
        print(f"cache: {cache.hits} hits, {cache.misses} misses")


if __name__ == "__main__":
    main()
