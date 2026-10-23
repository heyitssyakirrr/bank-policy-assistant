"""Ask one question from the command line and see the verified answer.

    python -m scripts.ask "What must a bank do if FAR exceeds 5%?"
    python -m scripts.ask "Can a takaful operator onboard remotely?" --sector insurance_takaful
    python -m scripts.ask "..." --no-cache        # force fresh API calls

Uses the response cache by default, so asking the same question twice is
free. Prints the answer, every citation, the verification report and the
cost (LLM calls, tokens, API requests per model).
"""

from __future__ import annotations

import argparse
import time

from app.agent.schemas import DISCLAIMER, SECTOR_LABELS
from app.agent.state import AgentContext, turn_input
from app.agent.traditional import build_traditional_graph
from app.core.config import settings
from app.core.logging import setup_logging
from app.llm.cache import ResponseCache
from app.llm.models import ChatModelClient
from app.services.search_service import build_search_service

OBLIGATION_BADGE = {"must": "MUST", "should_or_may": "MAY ", "information": "INFO"}


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("question", nargs="+")
    cli.add_argument("--mode", choices=["traditional"], default="traditional")
    cli.add_argument("--sector", choices=[*SECTOR_LABELS, "all"], default=settings.default_sector)
    cli.add_argument("--no-cache", action="store_true", help="don't read or write the response cache")
    args = cli.parse_args()
    setup_logging("WARNING")

    question = " ".join(args.question)
    cache = None if args.no_cache else ResponseCache(settings.response_cache_path)
    llm = ChatModelClient(cache=cache)
    context = AgentContext(llm=llm, search=build_search_service(cache))
    sector = None if args.sector == "all" else args.sector

    started = time.perf_counter()
    state = build_traditional_graph().invoke(turn_input(question, sector), context=context)
    seconds = time.perf_counter() - started

    answer, evidence = state["answer"], {e.id: e for e in state["evidence"]}
    print(f"Q: {question}\n[{answer.status.upper()}] {answer.summary}\n")
    for statement in answer.statements:
        print(f"  {OBLIGATION_BADGE[statement.obligation]}  {statement.text}  [{', '.join(statement.evidence_ids)}]")
        print(f'        "{statement.quote}"')
    if answer.gaps:
        print(f"\n  Gaps: {answer.gaps}")
    cited = dict.fromkeys(i for s in answer.statements for i in s.evidence_ids)
    print("\nSources:")
    for evidence_id in cited:
        print(f"  {evidence_id}  {evidence[evidence_id].citation}  (PDF page {evidence[evidence_id].page})")

    for number, report in enumerate(state["verifications"], start=1):
        print(f"\nVerification {number}: {'passed' if report.passed else 'FAILED'}")
        for issue in report.fixes + report.serious:
            print(f"  - {issue.check}: {issue.detail}")
    for note in state["notes"]:
        print(f"Note: {note}")
    print(
        f"\n{DISCLAIMER}\nmode {args.mode} | {seconds:.1f}s | LLM calls {state.get('llm_calls', 0)} | "
        f"tokens {state.get('input_tokens', 0)} in / {state.get('output_tokens', 0)} out | "
        f"API requests {dict(llm.requests) or 'none (cached)'}"
    )
    if cache is not None:
        print(f"cache: {cache.hits} hits, {cache.misses} misses")


if __name__ == "__main__":
    main()
