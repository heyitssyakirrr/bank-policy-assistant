"""The Streamlit app: Ask, Compare modes, Evaluation and How it works.

    streamlit run ui/app.py      (UI_BACKEND=api needs the API: uvicorn app.main:app)

The pages reach the agent only through ui/backend.py: over HTTP (api) or in
this process (local, for Streamlit Community Cloud).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `ui` and `app` import when run as a script

import streamlit as st  # noqa: E402

st.set_page_config(page_title="BNM Policy Assistant", page_icon=":material/policy:", layout="wide")
pages = [
    st.Page("views/chat.py", title="Ask", icon=":material/forum:", default=True),
    st.Page("views/compare.py", title="Compare modes", icon=":material/compare_arrows:"),
    st.Page("views/evaluation.py", title="Evaluation", icon=":material/analytics:"),
    st.Page("views/about.py", title="How it works", icon=":material/account_tree:"),
]
with st.sidebar:
    st.markdown("**BNM Policy Assistant**")
    st.caption("e-KYC · Fair Treatment · AML/CFT")
st.navigation(pages).run()
