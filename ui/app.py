"""The Streamlit demo: Chat, Compare and Evaluation pages.

    streamlit run ui/app.py      (with the API running: uvicorn app.main:app)

The UI only talks to the API over HTTP; it never imports the agent.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `ui` and `app.core` import when run as a script

import streamlit as st  # noqa: E402

st.set_page_config(page_title="BNM Policy Assistant", page_icon="⚖️", layout="wide")
pages = [
    st.Page("views/chat.py", title="Chat", icon="💬", default=True),
    st.Page("views/compare.py", title="Compare modes", icon="⚖️"),
    st.Page("views/evaluation.py", title="Evaluation", icon="📊"),
]
st.navigation(pages).run()
