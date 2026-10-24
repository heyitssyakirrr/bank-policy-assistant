"""Run the evaluation and write evals/results/<run>/report.md + results.json.

    python -m evals.run --full --estimate     the calls a full run is expected to make (spends nothing)
    python -m evals.run --smoke               8 retrieval + 5 answer items, both modes, judged (~25 Gemini calls)
    python -m evals.run --full                the whole golden set (Syakir runs it; see the estimate first)
    python -m evals.run --only xref-1 fact-6  just these items (after a fix)

Every external result is cached (app/data/interim/response_cache.sqlite):
re-running after a change pays only for the calls whose input changed.
Results are saved after each phase, so a quota failure late in a run
doesn't lose what was already measured.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from app.agent.prompts import PROMPT_VERSION
from app.agent.state import AgentContext
from app.core.config import enable_tracing, settings
from app.core.logging import setup_logging
from app.llm.cache import ResponseCache
from app.llm.models import ChatModelClient, LLMUnavailableError, quota_name, role_models
from app.services.retrieval_service import RetrievalService
from app.services.search_service import SearchService, build_search_parts
from evals.answer_eval import TurnResult, evaluate_answers
from evals.golden import GoldenItem, load_golden
from evals.judge import JudgeMismatchError, judge_answer, verdicts
from evals.metrics import faithfulness
from evals.report import build_report
from evals.retrieval_eval import StageScore, evaluate_retrieval, retrieval_items

REVIEW_PATH = Path("evals/golden_review.yaml")
SMOKE_RETRIEVAL = ("lookup-1", "lookup-3", "fact-1", "fact-4", "fact-6", "compare-1", "xref-1", "sector-1")
SMOKE_ANSWERS = ("fact-6", "xref-1", "followup-1", "scope-1", "ambiguous-1")
FULL_CROSS_CHECK = 15  # gemini-3.8-flash allows 20 calls a day

# Average LLM calls per turn, measured in the M1-M5 smoke runs (traditional: generator, sometimes a
# regeneration; agentic: planner, 1-2 reviews, generator, sometimes a regeneration).
EXPECTED_CALLS = {
    "traditional": {"answer": {"3.5": 1.3}, "refuse": {"3.5": 1.0}, "clarify": {"3.5": 1.0}},
    "agentic": {"answer": {"3.1": 2.5, "3.5": 1.3}, "refuse": {"3.1": 1.0}, "clarify": {"3.1": 1.0}},
}
WORST_CASE_CALLS = 5  # the request budget: no turn can make more LLM calls than this


def estimate(answer_items: Sequence[GoldenItem], modes: Sequence[str], judge: bool, cross_check: int) -> dict[str, Any]:
    """Expected Gemini calls on a cold cache (averages from the smoke runs), and the worst case."""
    lite_31 = lite_35 = 0.0
    turns = 0
    for item in answer_items:
        conversation = [*item.previous_turns, item.question]
        for mode in modes:
            for position in range(len(conversation)):
                outcome = item.expected_outcome if position == len(conversation) - 1 else "answer"
                lite_31 += EXPECTED_CALLS[mode][outcome].get("3.1", 0)
                lite_35 += EXPECTED_CALLS[mode][outcome].get("3.5", 0)
                turns += 1
    judged = len(answer_items) * len(modes) if judge else 0  # an unanswerable item answered by mistake is judged too
    return {
        "expected gemini-3.1-flash-lite (planner, reviewer, judge)": round(lite_31 + judged),
        "expected gemini-3.5-flash-lite (generator)": round(lite_35),
        "gemini-3.8-flash (judge cross-check)": min(cross_check, judged),
        "worst case, all Lite models": turns * WORST_CASE_CALLS + judged,
    }


def stratified(turns: Sequence[TurnResult], n: int) -> list[TurnResult]:
    """Up to n turns, taking one per category in turn (round-robin), so the
    cross-check covers comparisons and cross-references, not just easy lookups."""
    by_category: dict[str, list[TurnResult]] = {}
    for turn in turns:
        by_category.setdefault(turn.category, []).append(turn)
    picked: list[TurnResult] = []
    while len(picked) < n and any(by_category.values()):
        for queue in by_category.values():
            if queue and len(picked) < n:
                picked.append(queue.pop(0))
    return picked


def judge_turns(turns: list[TurnResult], llm: ChatModelClient, cross: ChatModelClient | None, cross_check: int) -> None:
    judgeable = [t for t in turns if t.answer is not None and t.answer.statements and not t.error]
    for turn in judgeable:
        try:
            result = judge_answer(llm, turn.conversation, turn.answer, turn.evidence)
            turn.verdicts = verdicts(result.output, len(turn.answer.statements))
        except (LLMUnavailableError, JudgeMismatchError) as exc:  # this turn stays unjudged; the run goes on
            print(f"  judge skipped {turn.item_id}/{turn.mode}: {type(exc).__name__}")
            continue
        turn.faithfulness = faithfulness(turn.verdicts)
        turn.relevancy = result.output.relevancy
        turn.judge_model = result.model
        turn.self_judged = quota_name(result.model) == quota_name(role_models("generator")[0])
    if cross is None:
        return
    for turn in stratified([t for t in judgeable if t.verdicts], cross_check):
        try:
            second = judge_answer(cross, turn.conversation, turn.answer, turn.evidence)
            turn.cross_check_verdicts = verdicts(second.output, len(turn.answer.statements))
        except LLMUnavailableError:  # gemini-3.8-flash allows 20 a day: stop at the first refusal
            print("  cross-check stopped: model unavailable (daily quota?)")
            return
        except JudgeMismatchError:
            continue


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def select_items(golden: list[GoldenItem], args: argparse.Namespace) -> tuple[list[GoldenItem], list[GoldenItem]]:
    by_id = {item.id: item for item in golden}
    if args.smoke:
        return [by_id[i] for i in SMOKE_RETRIEVAL], [by_id[i] for i in SMOKE_ANSWERS]
    if args.full:
        return golden, golden
    unknown = [i for i in args.only if i not in by_id]
    if unknown:
        raise SystemExit(f"Unknown item id(s) {unknown}. Valid ids: {', '.join(by_id)}")
    chosen = [by_id[i] for i in dict.fromkeys(args.only)]  # a repeated id would replay its conversation twice
    return chosen, chosen


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    scope = cli.add_mutually_exclusive_group(required=True)
    scope.add_argument("--smoke", action="store_true")
    scope.add_argument("--full", action="store_true")
    scope.add_argument("--only", nargs="+", metavar="ID")
    cli.add_argument("--modes", nargs="+", choices=["traditional", "agentic"], default=["traditional", "agentic"])
    cli.add_argument("--no-judge", action="store_true")
    cli.add_argument("--no-retrieval", action="store_true")
    cli.add_argument("--cross-check", type=int, default=None, help="answers re-judged by gemini-3.8-flash")
    cli.add_argument("--estimate", action="store_true", help="print the expected API calls and stop")
    args = cli.parse_args()
    setup_logging("WARNING")
    enable_tracing()
    sys.stdout.reconfigure(encoding="utf-8")

    golden = load_golden()
    retrieval_candidates, answer_items = select_items(golden, args)
    retrieval_set = [] if args.no_retrieval else retrieval_items(retrieval_candidates)
    default_cross = FULL_CROSS_CHECK if args.full else 0
    cross_check = 0 if args.no_judge else (default_cross if args.cross_check is None else args.cross_check)

    if args.estimate:
        print(f"{len(answer_items)} answer items x {len(args.modes)} modes; {len(retrieval_set)} retrieval items")
        for label, calls in estimate(answer_items, args.modes, not args.no_judge, cross_check).items():
            print(f"  {label}: {calls}")
        searches = len(retrieval_set) + len(answer_items) * (1 if args.modes == ["traditional"] else 3)
        print(f"  query embeddings and Cohere reranks: about {searches} each (many are cache hits)")
        return

    cache = ResponseCache(settings.response_cache_path)
    llm = ChatModelClient(cache=cache)
    cross = None
    if cross_check:
        cross_model = settings.judge_cross_check_model
        cross = ChatModelClient(cache=cache, models_for=lambda role: (cross_model, cross_model))
    parts = build_search_parts(cache)
    context = AgentContext(llm=llm, search=SearchService(RetrievalService(parts.repo, parts.embed), parts.reranker))

    run_name = datetime.now().strftime("%Y%m%d-%H%M") + ("-smoke" if args.smoke else "-full" if args.full else "-only")
    folder = settings.eval_results_dir / run_name
    folder.mkdir(parents=True, exist_ok=True)
    meta = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "commit": _commit(),
        "prompt_version": PROMPT_VERSION,
        "golden_size": len(golden),
        "models": {role: role_models(role)[0] for role in ("planner", "reviewer", "generator", "judge")}
        | {"judge cross-check": settings.judge_cross_check_model},
        "review": yaml.safe_load(REVIEW_PATH.read_text(encoding="utf-8")),
    }
    evaluated = list({item.id: item for item in [*retrieval_set, *answer_items]}.values())
    retrieval: list[StageScore] = []
    retrieval_errors: list[str] = []
    turns: list[TurnResult] = []

    def save() -> None:
        usage = {
            "llm_requests": dict(llm.requests) | (dict(cross.requests) if cross else {}),
            "embeddings": cache.misses_by_namespace["embed_query"],
            "reranks": cache.misses_by_namespace["rerank"],
            "cache_hits": cache.hits,
        }
        report, data = build_report(run_name, meta, evaluated, retrieval, turns, usage, retrieval_errors)
        (folder / "report.md").write_text(report, encoding="utf-8")
        (folder / "results.json").write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
        print(f"  saved {folder / 'report.md'} (cost so far: {usage})")

    print(f"Run {run_name}: {len(retrieval_set)} retrieval items, {len(answer_items)} answer items, modes {args.modes}")
    retrieval, retrieval_errors = evaluate_retrieval(
        retrieval_set, parts, on_item=lambda i: print(f"  retrieval {i.id}")
    )
    save()

    def show(turn: TurnResult) -> None:
        mark = "ok" if turn.outcome_correct else turn.observed_outcome.upper()
        recall = "" if turn.citation_recall is None else f" citation recall {turn.citation_recall:.2f}"
        print(f"  {turn.item_id:<12} {turn.mode:<11} {mark:<7} {turn.status}{recall} ({turn.llm_calls} calls)")

    turns = evaluate_answers(answer_items, args.modes, context, lambda: sum(llm.requests.values()), on_turn=show)
    save()
    if not args.no_judge:
        print("  judging...")
        judge_turns(turns, llm, cross, cross_check)
        save()


if __name__ == "__main__":
    main()
