"""Compare: the same question in both modes, side by side.

Each side starts a fresh conversation, so neither has history the other
lacks. Both share the search engine, the generator and the verifier, so
the difference is only in who decides the steps.
"""

import streamlit as st

from ui.backend import backend
from ui.components import can_ask, render_turn, result_badge, sector_setting, stream_turn

EXAMPLES = [
    "What does e-KYC paragraph 8.10 require?",
    "What does para 8.2 say?",
    "How do the e-KYC verification rules differ from AML's non-face-to-face rules?",
    "What is the interest rate cap on credit cards?",
]
SIDES = {
    "traditional": ("Traditional RAG", "One search with the question as typed, then the answer."),
    "agentic": ("Agentic RAG", "Plans the searches, checks the evidence, follows cross-references."),
}

api = backend()
results: dict = st.session_state.setdefault("compare_results", {})

with st.sidebar:
    sector = sector_setting()


def use_example() -> None:
    st.session_state.compare_question = st.session_state.compare_example
    st.session_state.compare_example = None


st.title("Compare modes")
st.caption(
    "The same question, answered both ways. Both modes share the search engine, the generator (which writes "
    "the answer) and the verifier (which checks the citations): the only difference is whether code or the "
    "model decides the steps."
)
st.pills("Examples", EXAMPLES, key="compare_example", on_change=use_example)
with st.form("compare", border=False):
    with st.container(horizontal=True, vertical_alignment="bottom"):
        question = st.text_input("Question", key="compare_question", placeholder="Ask a question to compare")
        submitted = st.form_submit_button("Compare", type="primary", icon=":material/compare_arrows:")

columns = dict(zip(SIDES, st.columns(2, gap="medium"), strict=True))
panels = {}
for mode, column in columns.items():
    title, description = SIDES[mode]
    panels[mode] = column.container(border=True)
    panels[mode].subheader(title)
    panels[mode].caption(description)

if submitted and question.strip() and can_ask(count=2):
    results.clear()
    results["question"] = question
    for mode, panel in panels.items():
        with panel:
            asked = stream_turn(api, question, mode, sector, conversation_id=None)
        results[mode], results[f"{mode}_error"] = asked.turn, asked.error
    st.rerun()

if results.get("question"):
    for mode, panel in panels.items():
        with panel:
            if results.get(mode):
                render_turn(results[mode], api)
            else:  # kept across the rerun, so a limit or an outage is explained, not just "no answer"
                st.error(results.get(f"{mode}_error") or "No answer.", icon=":material/error:")
    if results.get("traditional") and results.get("agentic"):
        st.subheader("What each mode did")
        rows = {
            SIDES[mode][0]: {
                "Result": result_badge(turn)[0],
                "Statements": len(turn["statements"]),
                "Sources cited": len(turn["sources"]),
                "Agent steps": len(turn["steps"]),
                "LLM calls": turn["llm_calls"],
                "Tokens": f"{turn['input_tokens'] + turn['output_tokens']:,}",
                "Time": f"{turn['latency_ms'] / 1000:.1f} s",
            }
            for mode in SIDES
            if (turn := results[mode])
        }
        st.table(rows)
