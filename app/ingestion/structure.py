"""Blocks -> AnnotatedBlocks: decide what each block MEANS.

Docling tells us what a block looks like (heading, list item, text). It
does not reliably tell us what it is in a BNM policy document: S 8.1 came
back as a list item but G 8.10 as plain text, and "A. General requirements"
came back as a list item although it is a heading. So the decisions are
made here, from the TEXT, with rules written for how BNM documents are
numbered:

  PART B ...                  -> heading, level 0 (resets everything below)
  Appendix 2: ...             -> heading, level 0, and opens a new scope
  8 e-KYC implementation      -> heading, level 1 (numbered section)
  Role and responsibility ... -> heading, level 2 (unnumbered sub-heading)
  29.1 Definition and ...     -> heading, level 2 (numbered, but a title)
  A. General requirements     -> heading, level 3 (lettered)
  S 8.1 A financial ...       -> paragraph start, id 8.1, provision S
  1.1 Supported by ...        -> paragraph start, id 1.1, no S/G (Part A)
  5. In determining ...       -> paragraph start, id 5 (only inside an appendix)
  (i) / i. / (a) ...          -> continuation of the open paragraph
  8.10 and 8.13 of this ...   -> continuation (a wrapped cross-reference, not para 8.10)
  Good practices / Poor ...   -> callout (FTFC boxes): illustrative text tied to the
                                 paragraph above, tagged "callout", not a heading

Everything until the next paragraph start or heading belongs to the
current paragraph: its sub-items, the sentence after its list, and the
part that continues onto the next page.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from app.ingestion.manifest import DocumentSpec
from app.ingestion.models import AnnotatedBlock, Block, BlockKind, Role

# 8.1 | 10.2.3 | 14C.10.7 | 4.0  (one or two digits, optional letter, then .number groups)
PARAGRAPH_ID = r"\d{1,2}[A-Z]?(?:\.\d{1,2}){1,3}"
PARAGRAPH_START_RE = re.compile(rf"^(?:(?P<provision>[SG])\s+)?(?P<pid>{PARAGRAPH_ID})(?=\s|$)\s*(?P<body>.*)$", re.S)
APPENDIX_PARAGRAPH_RE = re.compile(r"^(?P<pid>\d{1,2})\.\s+(?P<body>\S.*)$", re.S)

PART_RE = re.compile(r"^(?:PART\s+[A-Z]\b|APPENDICES$)", re.IGNORECASE)
APPENDIX_RE = re.compile(r"^APPENDIX\s+(?P<name>\d{1,2}[a-z]?)\b", re.IGNORECASE)
SECTION_RE = re.compile(r"^\d{1,2}[A-Z]?\s+[A-Za-z(]")
LETTERED_RE = re.compile(r"^[A-Z]\.\s+\S")
LONE_PROVISION_RE = re.compile(r"^[SG]$")
# Trailing number allowed: a box title can carry a footnote marker ("Good practices 8").
CALLOUT_RE = re.compile(r"^(?:(?:good|poor)\s+practices?|examples?|illustrations?)(?:\s+\d{1,3})?$", re.IGNORECASE)

MAX_HEADING_WORDS = 12  # longer than this, a "numbered title" is really a sentence
SENTENCE_END = (".", ";", ":", ",", "-", "—")

LEVEL_PART, LEVEL_SECTION, LEVEL_SUBHEADING, LEVEL_LETTERED = 0, 1, 2, 3


def is_cross_reference(body: str) -> bool:
    """'8.10 and 8.13 of this policy document' is the tail of '...under
    paragraphs' wrapped onto a new line, not paragraph 8.10. Real
    paragraphs start with a capital letter or '(' — never 'and', 'of', ','."""
    first = body.lstrip()[:1]
    return bool(first) and (first.islower() or first in ",.;:)")


def looks_like_title(text: str) -> bool:
    """Short and not ending like a sentence: 'Definition and Interpretation',
    not 'The Board shall ensure ... as follows:'."""
    text = text.strip()
    return 0 < len(text.split()) <= MAX_HEADING_WORDS and not text.endswith(SENTENCE_END)


def heading_level(text: str) -> int:
    if PART_RE.match(text) or APPENDIX_RE.match(text):
        return LEVEL_PART
    if SECTION_RE.match(text) and not PARAGRAPH_START_RE.match(text):
        return LEVEL_SECTION
    if LETTERED_RE.match(text):
        return LEVEL_LETTERED
    return LEVEL_SUBHEADING


@dataclass(slots=True)
class _State:
    """The reader's position while walking the document top to bottom."""

    headings: dict[int, str] = field(default_factory=dict)  # level -> heading text
    scope: str = "main"
    paragraph_id: str | None = None
    provision: str | None = None
    pending_provision: str | None = None  # an "S"/"G" the parser split off from its paragraph number
    callout: str | None = None  # "Good practices" while inside such a box

    @property
    def section_path(self) -> tuple[str, ...]:
        path = tuple(self.headings[level] for level in sorted(self.headings))
        return path + (self.callout,) if self.callout else path

    def set_heading(self, level: int, text: str) -> None:
        for deeper in [lvl for lvl in self.headings if lvl >= level]:
            del self.headings[deeper]
        self.headings[level] = text
        if level == LEVEL_PART:
            appendix = APPENDIX_RE.match(text)
            self.scope = f"Appendix {appendix.group('name')}" if appendix else "main"
        # A heading closes the open paragraph: text after it is new content.
        self.paragraph_id = None
        self.provision = None
        self.pending_provision = None
        self.callout = None


def annotate(blocks: Iterable[Block], spec: DocumentSpec) -> list[AnnotatedBlock]:
    state = _State()
    annotated: list[AnnotatedBlock] = []

    for block in blocks:

        def emit(role: Role, reason: str, block: Block = block) -> None:
            # Records the block with the reader's state AFTER the rule ran
            # (so a paragraph that closes a callout box is not tagged "callout").
            tags = spec.tags_for_page(block.page) + (("callout",) if state.callout else ())
            annotated.append(
                AnnotatedBlock(
                    block=block,
                    role=role,
                    section_path=state.section_path,
                    paragraph_id=state.paragraph_id,
                    provision=state.provision,
                    scope=state.scope,
                    tags=tags,
                    reason=reason,
                )
            )

        if block.page in spec.skip_pages:
            emit(Role.SKIP, "manifest skip_pages")
            continue
        if block.kind in (BlockKind.CAPTION, BlockKind.OTHER):
            emit(Role.SKIP, f"{block.kind.value}: no indexable text (captions are stored on their table)")
            continue
        if block.kind is BlockKind.FOOTNOTE:
            emit(Role.FOOTNOTE, "parser label footnote")
            continue
        if block.kind is BlockKind.TABLE:
            emit(Role.TABLE, "parser label table")
            continue

        text = block.full_text

        # Rule 0: a lone "S"/"G" (the marker column split off by the parser) applies to the next paragraph.
        if LONE_PROVISION_RE.match(text):
            state.pending_provision = text
            emit(Role.SKIP, "standalone S/G marker, applied to the next paragraph number")
            continue

        # Rule 0b: a callout box title ("Good practices"). The box illustrates the
        # paragraph above, so we keep that paragraph's id but start a separate,
        # tagged block of text. It is NOT a heading: the next paragraph must not
        # inherit "Good practices" in its section path.
        if CALLOUT_RE.match(text):
            state.callout = text
            emit(Role.CALLOUT, "callout box title")
            continue

        # Rule 1: a numbered provision ("S 8.1 ...", "1.1 ..."), whatever the parser called it.
        match = PARAGRAPH_START_RE.match(text)
        if match and match.group("provision") is None and is_cross_reference(match.group("body")):
            match = None  # fall through: it's text that happens to start with a paragraph number
        if match:
            if (
                match.group("provision") is None
                and not state.pending_provision
                and looks_like_title(match.group("body"))
            ):
                state.set_heading(LEVEL_SUBHEADING, text)
                emit(Role.HEADING, "numbered title (short, no S/G, no sentence ending)")
            else:
                state.paragraph_id = match.group("pid")
                state.provision = match.group("provision") or state.pending_provision
                state.pending_provision = None
                state.callout = None
                emit(Role.PARAGRAPH_START, "paragraph number" + (" + S/G marker" if state.provision else ""))
            continue

        # Rule 2: inside an appendix, paragraphs are numbered "1." "2." ...
        if state.scope != "main" and block.kind in (BlockKind.LIST_ITEM, BlockKind.TEXT):
            match = APPENDIX_PARAGRAPH_RE.match(text)
            if match:
                state.paragraph_id = match.group("pid")
                state.provision = None
                state.callout = None
                emit(Role.PARAGRAPH_START, "appendix paragraph number")
                continue

        # Rule 3: headings, by parser label, or a short lettered list item ("A. General requirements").
        is_lettered_item = block.kind is BlockKind.LIST_ITEM and LETTERED_RE.match(text) and looks_like_title(text)
        if block.kind is BlockKind.HEADING or is_lettered_item:
            level = heading_level(text)
            state.set_heading(level, text)
            emit(Role.HEADING, f"heading level {level}" + (" (lettered list item)" if is_lettered_item else ""))
            continue

        # Rule 4: everything else belongs to whatever is open.
        emit(Role.CONTINUATION, "continues open paragraph" if state.paragraph_id else "text under heading")

    return annotated
