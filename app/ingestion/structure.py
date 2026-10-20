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
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace

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

MARKER_ALIGN_TOLERANCE = 8.0  # points: an "S" and its paragraph number sit on the same line
MAX_HEADING_WORDS = 12  # longer than this, a "numbered title" is really a sentence
SENTENCE_END = (".", ";", ":", ",", "-", "—")

LEVEL_PART, LEVEL_SECTION, LEVEL_SUBHEADING, LEVEL_LETTERED = 0, 1, 2, 3


def is_cross_reference(body: str) -> bool:
    """'8.10 and 8.13 of this policy document' is the tail of '...under
    paragraphs' wrapped onto a new line, not paragraph 8.10. Real
    paragraphs start with a capital letter or '(' — never 'and', 'of', ','."""
    first = body.lstrip()[:1]
    return bool(first) and (first.islower() or first in ",.;:)")


def _dedupe(cells: Iterable[str]) -> list[str]:
    """Drop empty cells and cells repeated from the cell before (merged cells
    are copied into every column they span)."""
    out: list[str] = []
    for cell in cells:
        if cell and (not out or out[-1] != cell):
            out.append(cell)
    return out


def _is_number_cell(cell: str, in_appendix: bool) -> bool:
    return bool(re.fullmatch(PARAGRAPH_ID, cell) or (in_appendix and re.fullmatch(r"\d{1,2}\.", cell)))


def _is_numbered_row(cells: list[str], in_appendix: bool) -> bool:
    """| S | 10.3.1 | text |  or  | 11. | text |  or  | 10.4 PF Risk Assessment |"""
    if len(cells) >= 2 and cells[0] in ("S", "G") and _is_number_cell(cells[1], in_appendix):
        return True
    return _is_number_cell(cells[0], in_appendix) or bool(PARAGRAPH_START_RE.match(cells[0]))


def unpack_layout_table(block: Block, in_appendix: bool) -> list[Block] | None:
    """Docling sometimes reads ordinary numbered paragraphs as a TABLE because
    of a column layout. AML puts S/G, the number and the text in three
    columns, so whole pages came back as | S | 10.3.1 | Reporting ... |.
    e-KYC Appendix 2 para 11 came back as | 11. | In respect of ... |.

    If the first row, or at least half the rows, start with a paragraph
    number, the "table" is really text: each row becomes one text block
    ("S 10.3.1 Reporting ...") and the normal rules take it from there.
    Real tables (FAR thresholds, glossary terms) don't start rows with
    paragraph numbers, so they are left alone."""
    if block.kind is not BlockKind.TABLE or not block.table_rows:
        return None
    rows = [cells for cells in (_dedupe(row) for row in block.table_rows) if cells]
    if not rows:
        return None
    numbered = sum(_is_numbered_row(cells, in_appendix) for cells in rows)
    if not (_is_numbered_row(rows[0], in_appendix) or numbered * 2 >= len(rows)):
        return None
    return [
        replace(
            block,
            # A lone short row with no number is a sub-heading whose number Docling
            # dropped ("| | Internal Suspicious Transaction Reports | ... |").
            kind=BlockKind.HEADING if len(cells) == 1 and looks_like_title(cells[0]) else BlockKind.TEXT,
            text=" ".join(cells),
            table_rows=(),
            caption="",
        )
        for cells in rows
    ]


def align_markers(blocks: list[Block]) -> tuple[dict[int, str], set[int]]:
    """Pair each standalone "S"/"G" with the paragraph number on the same line.

    In AML the S/G sits in its own column, far left of the number, and
    Docling emits that column in an unreliable order: sometimes before the
    paragraphs, sometimes after. Reading order can't be trusted, but
    position can: the marker and its paragraph share a line, i.e. the same
    distance from the top of the page.

    Returns ({paragraph block index: "S"/"G"}, {marker block indexes used})."""
    markers: dict[int, list[Block]] = {}
    candidates: dict[int, list[Block]] = {}
    for block in blocks:
        if block.top is None or block.kind is BlockKind.TABLE:
            continue
        text = block.full_text
        if LONE_PROVISION_RE.match(text):
            markers.setdefault(block.page, []).append(block)
            continue
        match = PARAGRAPH_START_RE.match(text)
        if match and match.group("provision") is None and not is_cross_reference(match.group("body")):
            candidates.setdefault(block.page, []).append(block)

    assigned: dict[int, str] = {}
    used: set[int] = set()
    for page, page_markers in markers.items():
        pairs = sorted(
            (abs(m.top - c.top), m.index, c.index, m.full_text)
            for m in page_markers
            for c in candidates.get(page, [])
            if abs(m.top - c.top) <= MARKER_ALIGN_TOLERANCE
        )
        for _distance, marker_index, para_index, marker in pairs:  # closest pairs first
            if marker_index not in used and para_index not in assigned:
                used.add(marker_index)
                assigned[para_index] = marker
    return assigned, used


def _unpack_tables(blocks: Iterable[Block], state: _State) -> Iterator[tuple[Block, str]]:
    """Yields (block, note). Lazy on purpose: whether "11." counts as a
    paragraph number depends on being inside an appendix, which is only
    known once the blocks before this one have been read."""
    for block in blocks:
        rows = unpack_layout_table(block, in_appendix=state.scope != "main")
        if rows is None:
            yield block, ""
        else:
            for row in rows:
                yield row, "table row re-read as text (row starts with a paragraph number); "


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
    # "S"/"G" markers the parser split off from their paragraph numbers, in
    # order. AML's marker column often comes out as several "S" blocks at the
    # top of a page, before the paragraphs they belong to: a queue, per page.
    pending_provisions: list[tuple[str, int]] = field(default_factory=list)
    callout: str | None = None  # "Good practices" while inside such a box

    @property
    def section_path(self) -> tuple[str, ...]:
        path = tuple(self.headings[level] for level in sorted(self.headings))
        return path + (self.callout,) if self.callout else path

    def take_pending_provision(self, page: int) -> str | None:
        """Oldest queued marker from this page; markers from earlier pages are stale."""
        self.pending_provisions = [(m, p) for m, p in self.pending_provisions if p == page]
        return self.pending_provisions.pop(0)[0] if self.pending_provisions else None

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
        self.callout = None


def annotate(blocks: Iterable[Block], spec: DocumentSpec) -> list[AnnotatedBlock]:
    state = _State()
    annotated: list[AnnotatedBlock] = []
    blocks = list(blocks)
    aligned_provision, aligned_markers = align_markers(blocks)

    for block, note in _unpack_tables(blocks, state):

        def emit(role: Role, reason: str, block: Block = block, note: str = note) -> None:
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
                    reason=note + reason,
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
            if block.index in aligned_markers:
                emit(Role.SKIP, "standalone S/G marker, matched by position to the paragraph on its line")
            else:  # no position to match on: fall back to reading order
                state.pending_provisions.append((text, block.page))
                emit(Role.SKIP, "standalone S/G marker, no position match: queued for the next paragraph on this page")
            continue

        # Rule 0b: a callout box title ("Good practices"). The box illustrates the
        # paragraph above, so we keep that paragraph's id but start a separate,
        # tagged block of text. It is NOT a heading: the next paragraph must not
        # inherit "Good practices" in its section path.
        if CALLOUT_RE.match(text):
            state.callout = re.sub(r"\s+\d{1,3}$", "", text)  # "Good practices 8" -> drop footnote marker
            emit(Role.CALLOUT, "callout box title")
            continue

        # Rule 1: a numbered provision ("S 8.1 ...", "1.1 ..."), whatever the parser called it.
        match = PARAGRAPH_START_RE.match(text)
        if match and match.group("provision") is None and is_cross_reference(match.group("body")):
            match = None  # fall through: it's text that happens to start with a paragraph number
        if match:
            if match.group("provision") is None and looks_like_title(match.group("body")):
                state.set_heading(LEVEL_SUBHEADING, text)
                emit(Role.HEADING, "numbered title (short, no S/G, no sentence ending)")
            else:
                state.paragraph_id = match.group("pid")
                state.provision = (
                    match.group("provision")
                    or aligned_provision.get(block.index)
                    or state.take_pending_provision(block.page)
                )
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
