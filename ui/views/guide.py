"""Guide: the three documents in plain English, and questions to try with what
a correct answer says (content in ui/guide_content.py)."""

import streamlit as st

from ui.components import questions_left
from ui.guide_content import DOCUMENTS, QUESTIONS, READING_AN_ANSWER, TERMS


def ask(question: str, agentic_only: bool) -> None:
    """Ask this question in a new conversation on the Ask page."""
    left = questions_left()
    if left is not None and left < 1:  # keep the visitor's conversation on screen
        st.info("You have used this demo's questions for this session.", icon=":material/hourglass_empty:")
        return
    if agentic_only:  # these questions show what only the agent does (plan, clarify, refuse)
        st.session_state.mode_label = "Agentic"
    st.session_state.conversation_id, st.session_state.turns = None, []
    st.session_state.example_question = question
    st.switch_page("views/chat.py")


st.title("Guide")
st.caption(
    "You don't need to read the 265 pages to use the assistant or to judge its answers. Here is what each "
    "document covers, the words they use, and questions to try with what a correct answer says."
)

st.subheader("The three documents")
for column, (name, summary) in zip(st.columns(3, gap="medium"), DOCUMENTS.items(), strict=True):
    with column.container(border=True):
        st.markdown(f"**{name}**")
        st.caption(summary)

st.subheader("Questions to try")
st.caption(
    "Click **Ask** to ask it on the Ask page, then compare the answer with the expected one and its paragraph. "
    "The questions in *What the agent adds* are asked in Agentic mode."
)
for tab, (topic, question_set) in zip(st.tabs(list(QUESTIONS)), QUESTIONS.items(), strict=True):
    with tab:
        for i, item in enumerate(question_set.questions):
            with st.container(border=True, horizontal=True, vertical_alignment="center"):
                with st.container():
                    st.markdown(f"**{item.question}**")
                    st.caption(f"{item.expected}  \nCheck: {item.check}")
                if st.button("Ask", key=f"ask-{topic}-{i}", icon=":material/arrow_forward:"):
                    ask(item.question, question_set.agentic_only)

with st.expander("Words you'll meet", icon=":material/translate:"):
    st.table({"Term": list(TERMS), "Meaning": list(TERMS.values())})

st.caption(READING_AN_ANSWER)
