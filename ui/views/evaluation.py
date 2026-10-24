"""Evaluation: the latest measured results from evals/results/."""

import streamlit as st

from app.core.config import settings

st.header("Evaluation results")
reports = sorted(settings.eval_results_dir.glob("*/report.md"))
if not reports:
    st.info("No evaluation has been run yet. Run `python -m evals.run --smoke`, then reload this page.")
else:
    latest = reports[-1]
    st.caption(f"Run: {latest.parent.name} · {len(reports)} run(s) so far")
    st.markdown(latest.read_text(encoding="utf-8"))
