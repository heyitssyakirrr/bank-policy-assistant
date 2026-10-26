"""Ask: questions with cited answers, follow-ups, and the agent's steps live."""

import streamlit as st

from app.core.config import settings
from ui.backend import backend
from ui.components import (
    can_ask,
    health_line,
    mode_setting,
    my_conversations,
    questions_left,
    render_turn,
    sector_setting,
    stream_turn,
)
from ui.guide_content import DOCUMENTS

# Each shows something the agent does: a cross-reference, a sector, a table, a comparison.
EXAMPLES = {
    "What does e-KYC paragraph 8.10 require?": "Follows a cross-reference to 8.4",
    "Can a takaful operator onboard customers without meeting them face to face?": "Picks the takaful rules",
    "What must a bank do if FAR exceeds 5%?": "Reads the FAR table",
    "How do the e-KYC verification rules differ from AML's non-face-to-face rules?": "Compares two documents",
}

api = backend()
state = st.session_state
state.setdefault("conversation_id", None)
state.setdefault("turns", [])


def ask_example() -> None:
    """A clicked example is asked once: the pill is cleared so a rerun can't ask it again."""
    state.example_question, state.example = state.example, None


with st.sidebar:
    sector = sector_setting()
    mode = mode_setting()
    if st.button("New conversation", icon=":material/add:", width="stretch"):
        state.conversation_id, state.turns = None, []
        st.rerun()
    conversations = my_conversations(api)
    if conversations:
        st.caption("Your conversations")
        for conversation in conversations[:15]:
            current = conversation["id"] == state.conversation_id
            if st.button(
                conversation["title"],
                key=f"conv-{conversation['id']}",
                type="secondary" if current else "tertiary",
                width="stretch",
            ):
                state.conversation_id = conversation["id"]
                state.turns = api.conversation(conversation["id"])
                st.rerun()
    health_line(api)

st.title("Policy assistant")
st.caption(
    "Answers only from three Bank Negara Malaysia policy documents. Every statement cites its paragraph and "
    "says whether it is an obligation (MUST) or guidance (MAY). Not legal advice."
)

for turn in state.turns:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        render_turn(turn, api)

question = st.chat_input(
    "Ask about e-KYC, fair treatment or AML/CFT rules",
    max_chars=settings.max_question_chars,
    submit_mode="disable",
)

if not state.turns:
    intro = st.empty()
    with intro.container():
        with st.container(horizontal=True, gap="medium"):
            for name, scope in DOCUMENTS.items():
                with st.container(border=True):
                    st.markdown(f"**{name}**")
                    st.caption(scope)
        st.pills(
            "Try a question",
            list(EXAMPLES),
            key="example",
            on_change=ask_example,
            help=" ".join(f"{q} {what}." for q, what in EXAMPLES.items()),
        )
        st.caption(
            ":material/menu_book: Not sure what to ask, or how to tell a good answer? The **Guide** in the "
            "sidebar explains the three documents and lists questions with their expected answers."
        )
        left = questions_left()
        if left is not None:
            st.caption(f"{left} questions left in this session.")
    question = question or state.pop("example_question", None)
    if question:
        intro.empty()

if question and can_ask():
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        asked = stream_turn(api, question, mode, sector, state.conversation_id)
    state.conversation_id = asked.conversation_id
    if asked.turn is not None:
        state.turns.append(asked.turn)
        st.rerun()  # redraw from history: the conversation appears in the sidebar
