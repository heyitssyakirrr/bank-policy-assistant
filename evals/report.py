"""The evaluation report: evals/results/<run>/report.md (shown on the
Evaluation page) and results.json (every number behind it)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from evals.answer_eval import TurnResult
from evals.golden import GoldenItem
from evals.judge import agreement
from evals.metrics import mean, percentile
from evals.retrieval_eval import StageScore, regressions, summarise

HOW_TO_READ = """\
- **Hit@5**: did any gold paragraph appear in the first 5 passages? **Recall@5**: what share of the gold
  paragraphs did. **MRR@5**: 1 / the position of the first gold paragraph (1.00 = first place).
- **Outcome**: answered when the documents cover the question, refused when they don't, asked a clarifying
  question when it is ambiguous. A crash or model outage is an **error**, counted apart, never as a refusal.
- **Evidence recall**: gold paragraphs the mode *retrieved*. **Citation recall**: gold paragraphs the answer
  *cited*. Retrieved-but-not-cited points at the generator; never-retrieved points at the search.
- **Faithfulness**: the judge's share of statements supported by their cited passages (partly = 0.5).
  A "must" citing a passage that only says "may" counts as unsupported."""


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.0f}%"


def _num(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def mode_summary(turns: Sequence[TurnResult], primaries: frozenset[str] = frozenset()) -> dict[str, Any]:
    """`primaries`: the roles' configured models; a turn that used any other model fell back."""
    scored = [t for t in turns if t.observed_outcome != "error"]  # outages are reported apart
    answerable = [t for t in scored if t.expected_outcome == "answer"]
    answered = [t for t in answerable if t.observed_outcome == "answer"]
    judged = [t for t in answered if t.faithfulness is not None]
    live = [t.latency_ms / 1000 for t in scored if not t.cached]
    return {
        "n": len(turns),
        "errors": len(turns) - len(scored),
        "outcome_accuracy": mean([float(t.outcome_correct) for t in scored]),
        "answer_rate": mean([float(t.observed_outcome == "answer") for t in answerable]),
        "refusal_accuracy": mean([float(t.outcome_correct) for t in scored if t.expected_outcome == "refuse"]),
        "clarify_accuracy": mean([float(t.outcome_correct) for t in scored if t.expected_outcome == "clarify"]),
        "citation_recall": mean([t.citation_recall for t in answered]),
        "evidence_recall": mean([t.evidence_recall for t in answerable]),
        "verified": mean([float(t.verified) for t in answered]),
        "faithfulness": mean([t.faithfulness for t in judged]),
        "relevancy": mean([t.relevancy for t in judged]),
        "self_judged": sum(t.self_judged for t in judged),
        "fallback_answers": sum(bool(set(t.models) - primaries) for t in scored),
        "judge_agreement": mean(
            [agreement(t.verdicts, t.cross_check_verdicts) for t in judged if t.cross_check_verdicts]
        ),
        "cross_checked": sum(bool(t.cross_check_verdicts) for t in judged),
        "llm_calls": mean([t.llm_calls for t in scored]),
        "tokens": mean([t.input_tokens + t.output_tokens for t in scored]),
        "latency_p50": percentile(live, 50),
        "latency_p95": percentile(live, 95),
        "latency_n": len(live),
    }


def _turn_problems(t: TurnResult, item: GoldenItem) -> list[str]:
    if t.error or t.observed_outcome == "error":
        return [f"error: {t.error or '; '.join(t.notes)}"]
    problems = []
    if not t.outcome_correct:
        problems.append(f"expected {t.expected_outcome}, got {t.observed_outcome} ({t.status})")
    if t.citation_recall is not None and t.observed_outcome == "answer" and t.citation_recall < 1:
        missing = [k for k in item.expected_paragraphs if k not in t.cited_keys]
        retrieved = [k for k in missing if k in t.evidence_keys]
        lost = [k for k in missing if k not in t.evidence_keys]
        if retrieved:
            problems.append(f"retrieved but not cited: {retrieved}")
        if lost:
            problems.append(f"never retrieved: {lost}")
    if t.faithfulness is not None and t.faithfulness < 1:
        problems.append(f"faithfulness {t.faithfulness:.2f} (verdicts {t.verdicts})")
    return problems


def _failures_by_category(turns: Sequence[TurnResult], items: dict[str, GoldenItem]) -> list[str]:
    lines = []
    for category in dict.fromkeys(t.category for t in turns):
        rows = [
            f"- **{t.item_id}** ({t.mode}): " + "; ".join(problems)
            for t in turns
            if t.category == category and (problems := _turn_problems(t, items[t.item_id]))
        ]
        if rows:
            lines += [f"\n**{category}**", *rows]
    return lines


def _review_line(review: dict[str, list[str]]) -> str:
    syakir = review.get("syakir") or []
    return (
        f"Claude Code re-read the gold paragraphs of {len(review.get('claude_code') or [])} items against their "
        f"questions. Syakir spot-checked {len(syakir)} item(s) against the PDFs"
        + (f": {', '.join(syakir)}." if syakir else " (not done yet: the plan asks for at least 10).")
    )


def build_report(
    run_name: str,
    meta: dict[str, Any],
    items: Sequence[GoldenItem],
    retrieval: Sequence[StageScore],
    turns: Sequence[TurnResult],
    usage: dict[str, Any],
    retrieval_errors: Sequence[str] = (),
) -> tuple[str, dict[str, Any]]:
    by_id = {item.id: item for item in items}
    modes = list(dict.fromkeys(t.mode for t in turns))
    roles = meta.get("models", {})
    primaries = frozenset(roles[r] for r in ("planner", "reviewer", "generator") if r in roles)
    summaries = {mode: mode_summary([t for t in turns if t.mode == mode], primaries) for mode in modes}
    ablation = summarise(retrieval)
    out = [
        f"# Evaluation: {run_name}",
        f"{meta['date']} · commit `{meta['commit']}` · prompts `{meta['prompt_version']}` · "
        f"{len(items)} items evaluated ({meta['golden_size']} in the golden set)",
        "\n" + _table(["Role", "Model"], list(meta["models"].items())),
        "\n**How the golden set was built:** questions written by Claude Code from the paragraphs in "
        "`chunks.jsonl` (gold labels = those paragraphs); hard cases (comparison, cross-reference, follow-up, "
        "out-of-scope, ambiguous) written by hand from the paragraph text. Claude writes the questions; the "
        "models in the table above answer and judge. " + _review_line(meta.get("review", {})),
        "\n## How to read this",
        HOW_TO_READ,
        "\n## Retrieval ablation (first 5 passages, single-turn answerable questions)",
        _table(
            ["Stage", "n", "Hit@5", "Recall@5", "MRR@5"],
            [[r["stage"], r["n"], _pct(r["hit@5"]), _pct(r["recall@5"]), _num(r["mrr@5"])] for r in ablation],
        ),
    ]
    if retrieval_errors:
        out.append("\nSkipped (search failed): " + "; ".join(retrieval_errors))
    drops = regressions(retrieval)
    out.append("\n**Where adding a stage made an item worse** (from hybrid onwards, each stage builds on the last):")
    out += [
        f"- {after.item_id} ({after.category}): {before.stage} → {after.stage}: MRR@5 {before.mrr:.2f} → "
        f"{after.mrr:.2f}; top 5 before {before.ranked}, after {after.ranked}"
        for before, after in drops
    ] or ["- none"]
    agentic = [t for t in turns if t.mode == "agentic" and t.expected_outcome == "answer" and not t.error]
    if agentic:
        out.append(
            f"\nAgentic mode, all evidence it gathered (planner searches, lookups, follow-ups): recall "
            f"{_pct(mean([t.evidence_recall for t in agentic]))} over {len(agentic)} answerable questions "
            "(includes follow-ups, so a different set from the table above; not limited to 5 passages)."
        )
    if modes:
        rows = [
            ("Outcome accuracy", "outcome_accuracy", _pct),
            ("Answered, when answerable", "answer_rate", _pct),
            ("Refused, when out of scope", "refusal_accuracy", _pct),
            ("Asked to clarify, when ambiguous", "clarify_accuracy", _pct),
            ("Citation recall", "citation_recall", _pct),
            ("Evidence recall", "evidence_recall", _pct),
            ("Answers passing every verifier check", "verified", _pct),
            ("Faithfulness (judge)", "faithfulness", _pct),
            ("Relevancy 1-5 (judge)", "relevancy", _num),
            ("Judge agreement with gemini-3.8-flash", "judge_agreement", _pct),
            ("Answers cross-checked", "cross_checked", str),
            ("Answers judged by a model that also answered them (fallback)", "self_judged", str),
            ("Answers that used a fallback model (the primary failed)", "fallback_answers", str),
            ("LLM calls per question", "llm_calls", _num),
            ("Tokens per question", "tokens", lambda v: _num(v, 0)),
            ("Latency p50 (s)", "latency_p50", _num),
            ("Latency p95 (s)", "latency_p95", _num),
            ("Errors (outages, crashes; excluded above)", "errors", str),
        ]
        latency_counts = ", ".join(f"{m}: {summaries[m]['latency_n']}" for m in modes)
        out += [
            "\n## Answers by mode",
            _table(["Metric", *modes], [[label, *(fmt(summaries[m][key]) for m in modes)] for label, key, fmt in rows]),
            f"\nLatency counts only questions answered with live API calls ({latency_counts}); cached replays "
            "are excluded.",
            "\n## By category (outcome accuracy · citation recall)",
        ]
        cells = []
        for category in dict.fromkeys(t.category for t in turns):
            row = [category]
            for mode in modes:
                of = [t for t in turns if t.mode == mode and t.category == category and t.observed_outcome != "error"]
                recall = mean([t.citation_recall for t in of if t.observed_outcome == "answer"])
                row.append(f"{_pct(mean([float(t.outcome_correct) for t in of]))} · {_pct(recall)}")
            cells.append(row)
        out.append(_table(["Category", *modes], cells))
        out += ["\n## Failures to analyse, by category", *(_failures_by_category(turns, by_id) or ["None."])]
    requests = usage["llm_requests"]
    out += [
        "\n## Cost of this run",
        _table(
            ["Calls", "Count"],
            [
                *([[model, count] for model, count in requests.items()] or [["LLM (all cached)", 0]]),
                ["query embeddings", usage["embeddings"]],
                ["Cohere reranks", usage["reranks"]],
                ["cache hits (free)", usage["cache_hits"]],
            ],
        ),
    ]
    data = {
        "run": run_name,
        "meta": meta,
        "usage": usage,
        "retrieval_ablation": ablation,
        "retrieval_errors": list(retrieval_errors),
        "answers_by_mode": summaries,
        "retrieval": [s.model_dump() for s in retrieval],
        "turns": [t.model_dump(mode="json") for t in turns],
    }
    return "\n".join(out) + "\n", data
