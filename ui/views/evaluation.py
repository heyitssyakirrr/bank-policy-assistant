"""Evaluation: the latest measured results from evals/results/."""

import streamlit as st

from app.core.config import settings

st.header("Evaluation results")
reports = sorted(settings.eval_results_dir.glob("*/report.md"))
if not reports:
    st.info("No evaluation has been run yet. Run `python -m evals.run --smoke`, then reload this page.")
else:
    # A partial re-run ("--only") must not hide the last full evaluation: full runs come first.
    full = [r for r in reports if r.parent.name.endswith("-full")]
    ordered = [*reversed(full), *(r for r in reversed(reports) if r not in full)]
    chosen = st.selectbox("Run", ordered, format_func=lambda path: path.parent.name)
    st.caption(f"{len(reports)} run(s) so far; full runs are listed first")
    st.markdown(chosen.read_text(encoding="utf-8"))
