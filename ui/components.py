"""Pieces shared by the pages: the sidebar settings, the visitor's limits and
history, the live step timeline, and how an answered turn looks.

A turn (a TurnView dict) is drawn as: a status badge and who it applies to,
the summary, the statements with MUST / MAY tags and citation links, the
sources, the agent steps, the cost line and thumbs up / down.
"""

from __future__ import annotations

import re
from typing import Any, NamedTuple

import streamlit as st

from app.core.config import settings
from ui.backend import Backend

SECTORS: dict[str, str | None] = {
    "Banking": "banking",
    "Insurance and takaful": "insurance_takaful",
    "Money services": "money_services",
    "E-money": "non_bank_e_money",
    "All sectors": None,
}
MODES = {"Agentic": "agentic", "Traditional": "traditional"}
STATUS_BADGES = {  # (label, colour, icon)
    "answered": ("Answered", "green", ":material/check_circle:"),
    "partially_answered": ("Partly answered", "orange", ":material/rule:"),
    "not_covered": ("Not in the documents", "gray", ":material/block:"),
    "cancelled": ("Stopped", "gray", ":material/stop_circle:"),
    "error": ("Error", "red", ":material/error:"),
}
OUTCOME_BADGES = {
    "clarify": ("Needs clarification", "blue", ":material/help:"),
    "out_of_scope": ("Outside the documents", "gray", ":material/block:"),
    "small_talk": ("Hello", "gray", ":material/waving_hand:"),
}
OBLIGATION_TAGS = {"must": ("MUST", "red"), "should_or_may": ("MAY", "blue"), "information": ("INFO", "gray")}
PROVISION_LABELS = {"S": "Standard (S): an obligation", "G": "Guidance (G)"}
MODE_HELP = (
    "Agentic: the model plans the searches, runs them in parallel, checks the evidence and searches again "
    "if something is missing. Traditional: one search, then the answer."
)


# ---------------------------------------------------------------- sidebar


def _remembered(widget, label: str, options: list[str], store: str, **kwargs) -> str:
    """A sidebar widget whose choice survives switching pages (Streamlit drops
    a widget's state when its page isn't shown, so it's kept under `store`)."""
    st.session_state.setdefault(store, options[0])
    key = f"_{store}"

    def keep() -> None:
        if st.session_state[key] is not None:  # a segmented control can be clicked off
            st.session_state[store] = st.session_state[key]

    kwargs |= {"index": options.index(st.session_state[store])} if widget is st.selectbox else {}
    if widget is st.segmented_control:
        kwargs |= {"default": st.session_state[store], "required": True}
    widget(label, options, key=key, on_change=keep, **kwargs)
    return st.session_state[store]


def sector_setting() -> str | None:
    label = _remembered(
        st.selectbox,
        "Your institution",
        list(SECTORS),
        "sector_label",
        help="Rules for your sector come first. A sector named in the question takes priority.",
    )
    return SECTORS[label]


def mode_setting() -> str:
    return MODES[_remembered(st.segmented_control, "Answer mode", list(MODES), "mode_label", help=MODE_HELP)]


@st.cache_data(ttl=30, show_spinner=False)
def _health(_backend: Backend) -> dict[str, Any]:
    """Counted every 30 s at most, not on every click (it queries Qdrant and Postgres)."""
    return _backend.health()


def health_line(backend: Backend) -> None:
    status = _health(backend)
    if status.get("status") == "ok":
        points = status.get("qdrant", {}).get("points")
        st.caption(":material/check_circle: Ready" + (f" · {points:,} passages indexed" if points else ""))
    else:
        st.caption(":material/warning: Some services are unavailable: answers may fail.")


# ---------------------------------------------------------------- the visitor's limits and history


def questions_left() -> int | None:
    """None = no per-visitor limit."""
    if not settings.questions_per_session:
        return None
    return max(settings.questions_per_session - st.session_state.get("questions_asked", 0), 0)


def can_ask(count: int = 1) -> bool:
    """Shows why not when the visitor's limit is used up."""
    left = questions_left()
    if left is not None and left < count:
        st.info(
            "You have used this demo's questions for this session. Thank you for trying it! "
            "Answers you received are still in your history.",
            icon=":material/hourglass_empty:",
        )
        return False
    return True


def _count_question() -> None:
    """Called when a question has really started, so refusals and outages don't use up the allowance."""
    st.session_state["questions_asked"] = st.session_state.get("questions_asked", 0) + 1


HISTORY_SHOWN = 50  # the API's list limit too


def remember_conversation(conversation_id: str | None) -> None:
    """The visitor's own conversations (this browser session), newest first."""
    if conversation_id:
        mine: list[str] = st.session_state.setdefault("my_conversations", [])
        if conversation_id in mine:
            mine.remove(conversation_id)
        mine.insert(0, conversation_id)


def my_conversations(backend: Backend) -> list[dict[str, Any]]:
    return backend.conversations(st.session_state.get("my_conversations", [])[:HISTORY_SHOWN])


# ---------------------------------------------------------------- a turn, live


class Asked(NamedTuple):
    turn: dict[str, Any] | None  # None: it failed, and `error` says why
    conversation_id: str | None  # kept even when the turn failed, so the next question stays in it
    error: str | None = None


INTERRUPTED_MESSAGE = "The answer was interrupted. Please ask again."


def stream_turn(backend: Backend, question: str, mode: str, sector: str | None, conversation_id: str | None) -> Asked:
    """Ask, showing each agent step as it finishes (and an error, if any)."""
    started: dict[str, Any] = {}
    previous_ms = 0
    with st.status(":shimmer[Reading the question]", type="compact", expanded=True) as status:
        for event in backend.chat(question, mode, sector, conversation_id):
            match event["type"]:
                case "start":
                    started = event
                    _count_question()
                    remember_conversation(event["conversation_id"])
                    if event["pii_masked"]:
                        st.caption(f":material/shield: Masked before processing: {', '.join(event['pii_masked'])}")
                case "step":
                    _step(event, event["ms"] - previous_ms)
                    previous_ms = event["ms"]
                    status.update(label=f":shimmer[{(event['lines'] or [event['title']])[-1]}]")
                case "answer":
                    turn = event["turn"]
                    status.update(label=_steps_label(turn), state="complete", expanded=False)
                    return Asked(turn, turn["conversation_id"])
                case "error":
                    status.update(label="Could not finish", state="error", expanded=False)
                    st.error(event["message"], icon=":material/error:")
                    return Asked(None, started.get("conversation_id", conversation_id), event["message"])
        status.update(label="Could not finish", state="error", expanded=False)
    st.error(INTERRUPTED_MESSAGE, icon=":material/error:")
    return Asked(None, started.get("conversation_id", conversation_id), INTERRUPTED_MESSAGE)


def _step(step: dict[str, Any], took_ms: int) -> None:
    """One entry of the step timeline (consecutive "step" containers are joined),
    with the time since the previous step finished."""
    if not step["lines"]:
        return
    with st.status(f"{step['title']} · {took_ms / 1000:.1f} s", type="step", state="complete"):
        for line in step["lines"]:
            st.markdown(line)


def _steps_label(turn: dict[str, Any]) -> str:
    return f"Agent steps · {len(turn['steps'])} · {turn['latency_ms'] / 1000:.1f} s"


# ---------------------------------------------------------------- a turn, answered


LIST_START_RE = re.compile(r"^(\d{1,2})\.", re.M)


def _quote(snippet: str) -> str:
    """A passage as a quote block. "11. In respect of…" would render as list item 11, so the dot is escaped."""
    return "> " + LIST_START_RE.sub(r"\1\\.", snippet).replace("\n", "\n> ")


def render_turn(turn: dict[str, Any], backend: Backend, show_steps: bool = True) -> None:
    sources = {s["id"]: s for s in turn["sources"]}
    label, color, icon = OUTCOME_BADGES.get(turn["outcome"]) or STATUS_BADGES.get(
        turn["status"], ("Unknown", "gray", None)
    )
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.badge(label, color=color, icon=icon)
        if turn["outcome"] == "answer" and turn["statements"]:
            st.caption(f"Applies to {turn['applies_to']}")
    if turn["summary"]:
        st.markdown(turn["summary"])

    topics = list(dict.fromkeys(s["topic"] for s in turn["statements"]))
    grouped = 1 < len(topics) < len(turn["statements"])  # headings only when a topic has several statements
    for topic in topics:
        if grouped:
            st.markdown(f"**{topic[:1].upper()}{topic[1:]}**")
        for statement in (s for s in turn["statements"] if s["topic"] == topic):
            tag, tag_color = OBLIGATION_TAGS[statement["obligation"]]
            links = " · ".join(
                f'[{sources[i]["label"]}]({backend.pdf_link(sources[i]["pdf_url"])} "{sources[i]["citation"]}")'
                for i in statement["sources"]
                if i in sources
            )
            st.markdown(f":{tag_color}-badge[{tag}] {statement['text']} {links}")
    if turn.get("gaps"):
        st.caption(f"Not covered by the documents: {turn['gaps']}")
    for note in turn["notes"]:
        st.caption(f":material/info: {note}")

    if sources:
        with st.expander(f"Sources · {len(sources)}", icon=":material/menu_book:"):
            for source in sources.values():
                provision = PROVISION_LABELS.get(source["provision"] or "", "Appendix, table or text: no S/G marker")
                st.markdown(f"**{source['citation']}**  \n:gray[{provision}]")
                st.markdown(_quote(source["snippet"]))
                st.markdown(f"[Open in the PDF]({backend.pdf_link(source['pdf_url'])})")
    if show_steps and turn["steps"]:
        with st.status(_steps_label(turn), type="compact", state="complete"):
            previous_ms = 0
            for step in turn["steps"]:
                _step(step, step["ms"] - previous_ms)
                previous_ms = step["ms"]

    facts = ["Citations verified"] if turn["verified"] else []
    facts += [
        f"{turn['mode'].capitalize()} mode",
        f"{turn['llm_calls']} LLM call{'s' if turn['llm_calls'] != 1 else ''}",
        f"{turn['input_tokens'] + turn['output_tokens']:,} tokens",
        f"{turn['latency_ms'] / 1000:.1f} s",
    ]
    st.caption(" · ".join(facts) + f"  \n{turn['disclaimer']}")
    _feedback(turn, backend)


def _feedback(turn: dict[str, Any], backend: Backend) -> None:
    """Posts each new rating (feedback rows are append-only: the latest wins)."""
    key = f"feedback-{turn['turn_id']}"
    rating = st.feedback("thumbs", key=key)
    sent: dict[str, int] = st.session_state.setdefault("feedback_sent", {})
    if rating is None or sent.get(key) == rating:
        return
    if backend.feedback(turn["turn_id"], 1 if rating == 1 else -1):
        sent[key] = rating
        st.toast("Thanks for the feedback.", icon=":material/check:")
    else:
        st.toast("Feedback could not be saved. Please try again.", icon=":material/error:")
