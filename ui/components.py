"""Pieces shared by the Chat and Compare pages: how a turn looks.

A turn (the API's TurnView) is drawn as: a status badge, the summary, the
statements with MUST / MAY badges and citation links, the sources, the
agent steps, the cost, and thumbs up/down.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from app.core.config import settings
from ui.api_client import ApiClient

SECTORS: dict[str, str | None] = {
    "Banking": "banking",
    "Insurance / takaful": "insurance_takaful",
    "Money services": "money_services",
    "E-money": "non_bank_e_money",
    "Not sure: show all sectors": None,
}
STATUS_BADGES = {
    "answered": ("Answered", "green"),
    "partially_answered": ("Partially answered", "orange"),
    "not_covered": ("Not covered", "gray"),
    "error": ("Error", "red"),
}
OUTCOME_BADGES = {
    "clarify": ("Clarifying question", "blue"),
    "out_of_scope": ("Outside the documents", "gray"),
    "small_talk": ("Hello", "blue"),
}
OBLIGATION_BADGES = {"must": ("MUST", "red"), "should_or_may": ("MAY", "blue"), "information": ("INFO", "gray")}
PROVISION_LABELS = {"S": "S: standard (obligation)", "G": "G: guidance"}


@st.cache_resource
def api_client() -> ApiClient:
    return ApiClient(settings.api_url, settings.public_api_url)


def remembered(widget, label: str, options: list[str], store: str, **kwargs):
    """A sidebar widget whose choice survives switching pages. Streamlit drops
    a widget's state when its page isn't shown, so the choice is kept under a
    plain session key and given back as the default."""
    st.session_state.setdefault(store, options[0])
    widget_key = f"_{store}"

    def keep() -> None:
        st.session_state[store] = st.session_state[widget_key]

    return widget(
        label, options, index=options.index(st.session_state[store]), key=widget_key, on_change=keep, **kwargs
    )


def sector_sidebar() -> str | None:
    """The institution type picker: the user's sector (None = all sectors)."""
    label = remembered(
        st.sidebar.selectbox,
        "Institution type",
        list(SECTORS),
        "sector_label",
        help="The rules for your sector are preferred. A sector named in the question overrides this.",
    )
    return SECTORS[label]


def _badge(text: str, color: str) -> str:
    return f":{color}-badge[{text}]"


def stream_turn(
    api: ApiClient, question: str, mode: str, sector: str | None, conversation_id: str | None
) -> tuple[dict[str, Any] | None, str | None]:
    """Ask the API and show the agent steps live. Returns (the answer's turn,
    or None after showing an error; the conversation ID, kept even when the
    turn failed so the next question stays in the same conversation)."""
    started: dict[str, Any] = {}
    with st.status("Agent steps: starting…", expanded=True) as status:
        for event in api.chat(question, mode, sector, conversation_id):
            match event["type"]:
                case "start":
                    started = event
                    if event["pii_masked"]:
                        st.info(f"Masked before processing: {', '.join(event['pii_masked'])}")
                case "step":
                    for line in event["lines"]:
                        status.update(label=f"Agent steps: {line}")
                        st.write(f"`{event['ms'] / 1000:5.1f}s` {line}")
                case "answer":
                    turn = event["turn"]
                    status.update(
                        label=f"Agent steps · {len(turn['steps'])} steps · {turn['latency_ms'] / 1000:.1f} s",
                        state="complete",
                        expanded=False,
                    )
                    return turn, turn["conversation_id"]
                case "error":
                    status.update(label="Agent steps: failed", state="error")
                    st.error(event["message"])
                    return None, started.get("conversation_id", conversation_id)
    st.error("The answer stream ended early." if started else "No answer received.")
    return None, started.get("conversation_id", conversation_id)


def render_turn(turn: dict[str, Any], api: ApiClient, show_steps: bool = False) -> None:
    sources = {s["id"]: s for s in turn["sources"]}
    text, color = OUTCOME_BADGES.get(turn["outcome"]) or STATUS_BADGES.get(turn["status"], ("?", "gray"))
    st.markdown(f"{_badge(text, color)} {turn['summary']}")

    topics = list(dict.fromkeys(s["topic"] for s in turn["statements"]))
    grouped = 1 < len(topics) < len(turn["statements"])  # headings only when a topic has several statements
    for topic in topics:
        if grouped:
            st.markdown(f"**{topic}**")
        for statement in (s for s in turn["statements"] if s["topic"] == topic):
            label, badge_color = OBLIGATION_BADGES[statement["obligation"]]
            links = " ".join(
                f'[{sources[i]["label"]}]({api.public_link(sources[i]["pdf_url"])} "{sources[i]["citation"]}")'
                for i in statement["sources"]
                if i in sources
            )
            st.markdown(f"{_badge(label, badge_color)} {statement['text']} {links}")
    if turn.get("gaps"):
        st.caption(f"Not in the documents: {turn['gaps']}")

    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for source in sources.values():
                provision = PROVISION_LABELS.get(source["provision"] or "", "no S/G marker")
                st.markdown(f"**{source['label']}** · {source['citation']} · _{provision}_")
                st.caption(source["snippet"])
                st.markdown(f"[Open PDF at page]({api.public_link(source['pdf_url'])})")
    for note in turn["notes"]:
        st.warning(note)
    if show_steps and turn["steps"]:
        with st.expander(f"Agent steps ({len(turn['steps'])})"):
            for line in turn["steps"]:
                st.write(line)

    verified = " · citations verified ✓" if turn["verified"] else ""
    st.caption(
        f"Applies to: {turn['applies_to']}{verified} · {turn['mode']} · {turn['latency_ms'] / 1000:.1f} s · "
        f"{turn['llm_calls']} LLM calls · {turn['input_tokens'] + turn['output_tokens']:,} tokens  \n"
        f"{turn['disclaimer']}"
    )
    _feedback(turn, api)


def _feedback(turn: dict[str, Any], api: ApiClient) -> None:
    """Posts each new rating (feedback rows are append-only: the latest wins)."""
    key = f"feedback-{turn['turn_id']}"
    rating = st.feedback("thumbs", key=key)
    sent: dict[str, int] = st.session_state.setdefault("feedback_sent", {})
    if rating is None or sent.get(key) == rating:
        return
    if api.feedback(turn["turn_id"], 1 if rating == 1 else -1):
        sent[key] = rating
    else:
        st.toast("Feedback could not be saved. Please try again.")
