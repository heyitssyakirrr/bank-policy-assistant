"""Compare: the same question in both modes, side by side.

Each side starts a fresh conversation, so neither has history the other
lacks. The table underneath shows what the agent's decisions cost.
"""

import streamlit as st

from ui.components import api_client, render_turn, sector_sidebar, stream_turn

api = api_client()
sector = sector_sidebar()
results: dict = st.session_state.setdefault("compare_results", {})

st.header("Traditional vs agentic RAG")
st.caption(
    "Traditional: one search with the question as typed, then the answer. "
    "Agentic: the LLM plans the searches, runs them in parallel, reviews the evidence and searches again if needed. "
    "Both share the same search engine, generator and verifier."
)
with st.form("compare"):
    question = st.text_input("Question", placeholder="What does e-KYC paragraph 8.10 require?")
    submitted = st.form_submit_button("Ask both")

columns = dict(zip(("traditional", "agentic"), st.columns(2), strict=True))
for mode, column in columns.items():
    column.subheader(mode.capitalize())

if submitted and question.strip():
    results.clear()
    results["question"] = question
    for mode, column in columns.items():
        with column:
            results[mode], _ = stream_turn(api, question, mode, sector, conversation_id=None)

if results.get("question"):
    for mode, column in columns.items():
        with column:
            if results.get(mode):
                render_turn(results[mode], api, show_steps=True)
    if results.get("traditional") and results.get("agentic"):
        st.subheader("What each mode cost")
        st.table(
            {
                mode.capitalize(): {
                    "Latency (s)": round(turn["latency_ms"] / 1000, 1),
                    "LLM calls": turn["llm_calls"],
                    "Tokens": turn["input_tokens"] + turn["output_tokens"],
                    "Statements": len(turn["statements"]),
                    "Sources cited": len(turn["sources"]),
                    "Status": turn["status"].replace("_", " "),
                }
                for mode in columns
                if (turn := results[mode])
            }
        )
