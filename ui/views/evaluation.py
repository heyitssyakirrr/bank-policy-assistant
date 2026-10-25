"""Evaluation: the measured results from evals/results/ (results.json + report.md)."""

import json

import streamlit as st

from app.core.config import settings

MODE_METRICS = {  # results.json key -> (label, is a share shown as %)
    "outcome_accuracy": ("Right outcome (answer / refuse / clarify)", True),
    "citation_recall": ("Gold paragraphs cited", True),
    "evidence_recall": ("Gold paragraphs retrieved", True),
    "faithfulness": ("Statements supported by their citation", True),
    "verified": ("Answers passing the citation checks", True),
    "relevancy": ("Relevancy (judge, 1 to 5)", False),
    "llm_calls": ("LLM calls per question", False),
    "tokens": ("Tokens per question", False),
    "latency_p50": ("Median time (s)", False),
}
MODES = {"traditional": "Traditional", "agentic": "Agentic"}


def _value(value: float | None, share: bool) -> str:
    if value is None:
        return "–"
    if share:
        return f"{value:.0%}"
    return f"{value:,.0f}" if value >= 100 else f"{value:.1f}"


st.title("Evaluation")
st.caption(
    "A golden set of 40 questions in 8 categories, each with its gold paragraphs and expected outcome. "
    "Retrieval is scored without an LLM; answers are scored by code checks and an LLM judge."
)

runs = sorted(settings.eval_results_dir.glob("*/results.json"))
if not runs:
    st.info("No evaluation has been run yet: `python -m evals.run --smoke`.", icon=":material/info:")
    st.stop()
# A partial re-run (--only) must not hide the last full evaluation: full runs come first.
full = [r for r in runs if r.parent.name.endswith("-full")]
ordered = [*reversed(full), *(r for r in reversed(runs) if r not in full)]
chosen = st.selectbox("Run", ordered, format_func=lambda path: path.parent.name)
results = json.loads(chosen.read_text(encoding="utf-8"))
meta = results["meta"]
by_mode = {mode: results["answers_by_mode"].get(mode, {}) for mode in MODES}  # a run may cover one mode
st.caption(f"{meta['date']} · commit {meta['commit']} · prompts {meta['prompt_version']}")

st.subheader("Traditional vs agentic")
headline = {  # tile label (short enough not to be cut) -> metric
    "Right outcome": "outcome_accuracy",
    "Gold cited": "citation_recall",
    "Faithful": "faithfulness",
    "LLM calls": "llm_calls",
}
for column, (tile, metric) in zip(st.columns(len(headline)), headline.items(), strict=True):
    label, share = MODE_METRICS[metric]
    agentic, traditional = by_mode["agentic"].get(metric), by_mode["traditional"].get(metric)
    column.metric(
        tile,
        _value(agentic, share),
        delta=None if agentic is None or traditional is None else f"traditional {_value(traditional, share)}",
        delta_color="off",
        delta_arrow="off",
        border=True,
        help=f"{label}. Agentic mode, with the traditional mode underneath.",
    )
st.dataframe(
    [
        {"Metric": label, **{name: _value(by_mode[mode].get(m), share) for mode, name in MODES.items()}}
        for m, (label, share) in MODE_METRICS.items()
    ],
    hide_index=True,
)
st.caption(" · ".join(f"{name}: {by_mode[mode].get('n', 0)} questions" for mode, name in MODES.items()))

st.subheader("Retrieval: what each search stage adds")
st.caption("MRR@5: 1 divided by the position of the first gold paragraph in the top 5 (1.00 = first place).")
ablation = results["retrieval_ablation"]
st.bar_chart(
    [{"Stage": row["stage"], "MRR@5": row["mrr@5"]} for row in ablation],
    x="Stage",
    y="MRR@5",
    horizontal=True,
    sort=False,
    color=None,
    height=260,
)
st.dataframe(
    [
        {
            "Stage": row["stage"],
            "Questions": row["n"],
            "Hit@5": f"{row['hit@5']:.0%}",
            "Recall@5": f"{row['recall@5']:.0%}",
            "MRR@5": f"{row['mrr@5']:.2f}",
        }
        for row in ablation
    ],
    hide_index=True,
)

st.subheader("Every question")
st.dataframe(
    [
        {
            "Question": turn["item_id"],
            "Category": turn["category"].replace("_", " "),
            "Mode": MODES[turn["mode"]],
            "Right outcome": "Yes" if turn["outcome_correct"] else "No",
            "Citation recall": "–" if turn["citation_recall"] is None else f"{turn['citation_recall']:.0%}",
            "Faithfulness": "–" if turn["faithfulness"] is None else f"{turn['faithfulness']:.0%}",
            "LLM calls": turn["llm_calls"],
        }
        for turn in results["turns"]
    ],
    hide_index=True,
)
report = chosen.with_name("report.md")
if report.exists():
    with st.expander("Full report", icon=":material/description:"):
        st.markdown(report.read_text(encoding="utf-8"))
