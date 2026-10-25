"""How it works: the system in one page, for a first-time visitor."""

import streamlit as st

st.title("How it works")
st.caption(
    "An agentic RAG assistant over three Bank Negara Malaysia policy documents. It answers only from them, "
    "cites every statement to its paragraph and page, and says when the documents don't cover a question."
)

st.subheader("Two modes, one engine")
left, right = st.columns(2, gap="medium")
with left.container(border=True):
    st.markdown("**Traditional RAG**: code decides every step")
    st.markdown("Search → Write → Verify")
    st.caption("One search with the question as typed. Fast and cheap, but it can't notice what it missed.")
with right.container(border=True):
    st.markdown("**Agentic RAG**: the model decides, code sets the limits")
    st.markdown(
        "Plan → Search or look up, in parallel → Review → *search again if needed, at most 2 rounds* → Write → Verify"
    )
    st.caption(
        "The planner splits comparisons, rewrites follow-ups and refuses out-of-scope questions before "
        "searching. The reviewer follows cross-references such as e-KYC 8.10 → 8.4."
    )
st.markdown(
    "Both modes share the same search engine, the generator (which writes the answer) and the verifier "
    "(which checks the citations), so comparing them measures only the decision-making. Try it on the "
    "**Compare modes** page."
)

st.subheader("The search engine")
st.markdown(
    "- **1,219 passages**, one per numbered paragraph or table, each labelled with its document, section, "
    "S/G marker, page and sector.\n"
    "- **Hybrid search**: meaning (Gemini embeddings) and exact words (BM25, which keeps paragraph numbers "
    "like 14A.9.6 whole), merged with reciprocal rank fusion.\n"
    "- **Exact lookup** when a paragraph is named, then a **reranker** (Cohere) re-scores the top 30.\n"
    "- **Sector handling**: the AML rules repeated for banks (14A), insurers and takaful operators (14B), "
    "money services (14C) and e-money (14D) are collapsed to the copy for your institution."
)

st.subheader("Why you can trust the citations")
st.markdown(
    "- The model never types a citation: it refers to passages by label (E1, E2…) and code turns each "
    "label into the paragraph and page.\n"
    "- Code checks every statement: the label exists, the quote really is in that passage, a MUST is not "
    "backed only by guidance (G), and a passage the answer depends on is not left out. One failed check "
    "means one rewrite with the problems listed; whatever still fails is removed.\n"
    "- MyKad (NRIC) numbers are masked before anything reaches the model, the logs or the database, and "
    "every answer is stored in an append-only audit log."
)

st.subheader("Known limits")
st.markdown(
    "- Agentic answers take longer (often 20 to 40 seconds) on the free model tier.\n"
    "- The same question can be phrased slightly differently each time; the checks keep it grounded.\n"
    "- This is a demo on free tiers, so the number of questions per day is limited."
)
st.caption("Based on BNM policy documents. Not legal advice.")
