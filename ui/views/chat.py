"""Chat: ask questions, watch the agent's steps live, read cited answers."""

import streamlit as st

from ui.components import api_client, remembered, render_turn, sector_sidebar, stream_turn

MODES = {"Agentic": "agentic", "Traditional": "traditional"}

api = api_client()
state = st.session_state
state.setdefault("conversation_id", None)
state.setdefault("turns", [])


@st.cache_data(ttl=30)
def health() -> dict:
    return api.health()


with st.sidebar:
    st.title("BNM Policy Assistant")
    sector = sector_sidebar()
    mode = MODES[
        remembered(
            st.radio,
            "Mode",
            list(MODES),
            "mode_label",
            help="Agentic: the LLM plans, searches in parallel and reviews the evidence. "
            "Traditional: one search, then the answer.",
        )
    ]
    if st.button("New chat", use_container_width=True):
        state.conversation_id, state.turns = None, []
        st.rerun()
    st.subheader("History")
    for conversation in api.conversations()[:15]:
        if st.button(conversation["title"], key=f"conv-{conversation['id']}", use_container_width=True):
            state.conversation_id = conversation["id"]
            state.turns = api.conversation(conversation["id"])
            st.rerun()
    status = health()
    st.caption(
        f"API: {status['status']}"
        + (f" · {status['qdrant']['points']:,} passages indexed" if status.get("qdrant", {}).get("ok") else "")
    )

st.header("Ask about e-KYC, Fair Treatment or AML/CFT")
st.caption("Answers come only from three Bank Negara Malaysia policy documents, with paragraph citations.")

for turn in state.turns:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        render_turn(turn, api, show_steps=True)

if question := st.chat_input("e.g. Can a customer be onboarded without meeting face to face?"):
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        turn, state.conversation_id = stream_turn(api, question, mode, sector, state.conversation_id)
        if turn is not None:
            state.turns.append(turn)
            st.rerun()  # redraw from history: the new conversation appears in the sidebar
