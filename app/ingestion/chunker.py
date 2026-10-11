"""AnnotatedBlocks -> Chunks.

Strategy: one chunk per numbered paragraph (a provision is the unit people
ask about and cite), one chunk per table, and one chunk per run of
unnumbered text under a heading. Then three clean-ups:

  1. Stitch tables that continue onto the next page (the FAR table).
  2. Attach each footnote to the chunk whose text references it.
  3. Split anything over `max_words`, on block boundaries first, so a part
     never ends mid-sentence unless one block alone is too long.

Every chunk's `embed_text` starts with a context header (document >
section path > paragraph). A chunk that just says "the Board shall..."
is ambiguous on its own; with the header, both the embedding and BM25
know which document and section it belongs to. This is often called
"contextual chunk headers".
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field, replace

from app.ingestion.manifest import DocumentSpec
from app.ingestion.models import AnnotatedBlock, Chunk, ChunkType, ParsedDocument, Role

CHUNK_NAMESPACE = uuid.UUID("6f1c2b0e-7f3a-4c1e-9a55-2b7d8e4f1a10")  # fixed forever: ids must stay stable
FOOTNOTE_NUMBER_RE = re.compile(r"^(\d{1,3})\s+(\S.*)$", re.S)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;:])\s+(?=[A-Z(])")
# Numbers after these words are references, not footnote markers ("paragraph 3 of this Appendix").
REFERENCE_WORDS = frozenset(
    "paragraph paragraphs para appendix part section table level page and or to of than "
    "within first last least every per item items".split()
)


def references_footnote(text: str, number: str) -> bool:
    """True if `text` contains footnote marker `number`: a word, a paragraph
    number or a closing bracket, a space, then the number ("issued ID 3 by",
    "masks 4 )", "paragraph 8.22 11 ."). Docling separates superscripts from
    the word before them with a space."""
    pattern = re.compile(rf"(?P<word>[A-Za-z]+|\d+[A-Z]?(?:\.\d+)+|[)\]’'\"])\s+{number}(?=[\s.,;:)]|$)")
    for match in pattern.finditer(text):
        if match.group("word").lower() not in REFERENCE_WORDS:
            return True
    return False


LEAD_IN_WORDS = 60
WORD_RE = re.compile(r"\w+")


def count_words(text: str) -> int:
    """Real words only. A markdown table's "|" and "---" separators are not
    content and must not push a table over the size limit."""
    return len(WORD_RE.findall(text))


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    max_words: int = 350  # ~465 tokens; the embedding model accepts far more, but focused chunks retrieve better


@dataclass(slots=True)
class _Unit:
    """A group of blocks that will become one chunk (or several parts)."""

    chunk_type: ChunkType
    first: AnnotatedBlock
    blocks: list[AnnotatedBlock] = field(default_factory=list)
    table_rows: list[tuple[str, ...]] = field(default_factory=list)
    caption: str = ""
    lead_in: str = ""  # TABLE only: the start of the paragraph that introduces it
    footnotes: list[str] = field(default_factory=list)

    @property
    def pages(self) -> list[int]:
        return [ab.block.page for ab in self.blocks]

    @property
    def text(self) -> str:
        if self.chunk_type is ChunkType.TABLE:
            return _render_table(self.table_rows, self.caption, self.lead_in)
        return "\n".join(ab.block.full_text for ab in self.blocks)


def _truncate_words(text: str, limit: int) -> str:
    words = text.split()
    return text if len(words) <= limit else " ".join(words[:limit]) + " …"


def _render_table(rows: list[tuple[str, ...]], caption: str, lead_in: str = "") -> str:
    intro = [line for line in (lead_in, caption) if line]
    if not rows:
        return "\n".join(intro)
    width = max(len(row) for row in rows)
    padded = [row + ("",) * (width - len(row)) for row in rows]
    fmt = lambda row: "| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |"  # noqa: E731
    lines = intro
    lines += [fmt(padded[0]), "|" + "---|" * width] + [fmt(row) for row in padded[1:]]
    return "\n".join(lines)


def _group_into_units(
    annotated: list[AnnotatedBlock],
) -> tuple[list[_Unit], list[AnnotatedBlock], list[tuple[AnnotatedBlock, _Unit]]]:
    """Returns (units, footnotes, heading_owners). heading_owners maps each
    heading to the first unit under it, so a footnote marker sitting on a
    heading ("C. IDV through e-KYC for legal person 5") has somewhere to go."""
    units: list[_Unit] = []
    footnotes: list[AnnotatedBlock] = []
    heading_owners: list[tuple[AnnotatedBlock, _Unit]] = []
    pending_headings: list[AnnotatedBlock] = []
    open_text: _Unit | None = None  # the paragraph / text unit that continuations attach to
    last_content: _Unit | None = None  # last unit that received content (for table stitching)

    def add(unit: _Unit) -> _Unit:
        units.append(unit)
        heading_owners.extend((heading, unit) for heading in pending_headings)
        pending_headings.clear()
        return unit

    for ab in annotated:
        role = ab.role
        if role in (Role.SKIP, Role.HEADING, Role.CALLOUT):
            if role is not Role.SKIP:
                open_text = None  # headings and callout titles close the open paragraph/text
                last_content = None
                pending_headings.append(ab)
            continue
        if role is Role.FOOTNOTE:
            footnotes.append(ab)
            continue  # footnotes sit at page bottoms and must not break a paragraph or table

        if role is Role.TABLE:
            rows = list(ab.block.table_rows)
            prev = last_content
            if (
                prev is not None
                and prev.chunk_type is ChunkType.TABLE
                and ab.block.page == prev.pages[-1] + 1
                and rows
                and prev.table_rows
                and len(rows[0]) == len(prev.table_rows[0])
            ):
                # Continuation of the table on the previous page. Its first row
                # is data unless the PDF repeated the header row.
                if rows[0] == prev.table_rows[0]:
                    rows = rows[1:]
                # A row cut by the page break (AML glossary: "beneficial owner"
                # continues on the next page with an empty first cell) is glued
                # back onto the row it came from.
                if rows and not rows[0][0] and len(prev.table_rows) > 1:
                    last = prev.table_rows[-1]
                    prev.table_rows[-1] = tuple(
                        " ".join(filter(None, pair)) for pair in zip(last, rows[0], strict=True)
                    )  # same width: checked above
                    rows = rows[1:]
                prev.table_rows.extend(rows)
                prev.blocks.append(ab)
                continue
            unit = add(_Unit(ChunkType.TABLE, first=ab, blocks=[ab], table_rows=rows, caption=ab.block.caption))
            if open_text is not None and open_text.chunk_type is ChunkType.PARAGRAPH:
                # "10. ... shall adopt a tiered approach ... as follows:" + Table 1.
                # Without the lead-in the table chunk doesn't say what it is for.
                unit.lead_in = _truncate_words(open_text.text, LEAD_IN_WORDS)
            last_content = unit
            # The paragraph stays open: text after a table usually still belongs to it.
            continue

        if role is Role.PARAGRAPH_START:
            open_text = add(_Unit(ChunkType.PARAGRAPH, first=ab, blocks=[ab]))
            last_content = open_text
            continue

        # CONTINUATION
        if open_text is None:
            open_text = add(_Unit(ChunkType.TEXT, first=ab, blocks=[]))
        open_text.blocks.append(ab)
        last_content = open_text

    return units, footnotes, heading_owners


@dataclass(slots=True)
class _Footnote:
    number: str | None
    text: str
    page: int
    blocks: list[AnnotatedBlock]


def _merge_footnote_parts(footnotes: list[AnnotatedBlock]) -> list[_Footnote]:
    """A footnote can span several blocks: "11 This requirement must be
    completed ... unless:" followed by "(i) ..." and "(ii) ...". A block that
    doesn't start with a number continues the footnote before it."""
    merged: list[_Footnote] = []
    for ab in footnotes:
        match = FOOTNOTE_NUMBER_RE.match(ab.block.text)
        if match:
            merged.append(_Footnote(match.group(1), match.group(2), ab.block.page, [ab]))
        elif merged and merged[-1].page in (ab.block.page, ab.block.page - 1):
            merged[-1].text += "\n" + ab.block.text
            merged[-1].blocks.append(ab)
        else:
            merged.append(_Footnote(None, ab.block.text, ab.block.page, [ab]))
    return merged


def _attach_footnotes(
    units: list[_Unit],
    footnotes: list[AnnotatedBlock],
    heading_owners: list[tuple[AnnotatedBlock, _Unit]],
) -> list[AnnotatedBlock]:
    """Put each footnote on the unit that references it ("issued ID 3 by")
    on the same page. Returns the blocks of footnotes nothing referenced."""
    # page -> [(unit, text on that page)], so each footnote only checks its own page
    by_page: dict[int, list[tuple[_Unit, str]]] = {}
    for unit in units:
        for ab in unit.blocks:
            text = unit.text if unit.chunk_type is ChunkType.TABLE else ab.block.full_text
            by_page.setdefault(ab.block.page, []).append((unit, text))
    for heading, unit in heading_owners:  # checked last: body text wins over a heading
        by_page.setdefault(heading.block.page, []).append((unit, heading.block.full_text))

    unreferenced: list[AnnotatedBlock] = []
    for fn in _merge_footnote_parts(footnotes):
        owner = None
        if fn.number is not None:
            owner = next(
                (unit for unit, text in by_page.get(fn.page, []) if references_footnote(text, fn.number)),
                None,
            )
        if owner is None:
            unreferenced.extend(fn.blocks[:1])  # report/keep it once, as its first block
            if len(fn.blocks) > 1:  # keep the continuation text with it
                first = fn.blocks[0]
                unreferenced[-1] = replace(
                    first, block=replace(first.block, text=f"{fn.number or ''} {fn.text}".strip())
                )
        else:
            owner.footnotes.append(f"[{fn.number}] {fn.text}")
    return unreferenced


def _split_words(text: str, max_words: int) -> list[str]:
    """Split one over-long block on sentence boundaries, words as a last resort."""
    parts: list[str] = []
    current: list[str] = []
    for sentence in SENTENCE_SPLIT_RE.split(text):
        words = sentence.split()
        if len(words) > max_words:  # a single monster sentence
            if current:
                parts.append(" ".join(current))
                current = []
            parts.extend(" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words))
            continue
        if len(current) + len(words) > max_words and current:
            parts.append(" ".join(current))
            current = []
        current.extend(words)
    if current:
        parts.append(" ".join(current))
    return parts


def _split_unit_text(unit: _Unit, max_words: int) -> list[tuple[str, list[int]]]:
    """Returns [(text, pages_covered)], each under max_words where possible."""
    if unit.chunk_type is ChunkType.TABLE:
        header, body = unit.table_rows[:1], unit.table_rows[1:]
        whole = unit.text
        if count_words(whole) <= max_words:
            return [(whole, unit.pages)]
        if not body:  # a one-row "table" that is too big: plain word split
            return [(piece, unit.pages) for piece in _split_words(whole, max_words)]
        parts, current = [], []
        header_words = count_words(_render_table(header, unit.caption, unit.lead_in))
        for row in body:
            row_words = count_words(" ".join(row))
            if header_words + row_words > max_words:
                # One row alone is too big (AML Appendix 8b): flush, then split the row's text.
                if current:
                    parts.append(_render_table(header + current, unit.caption, unit.lead_in))
                    current = []
                budget = max(max_words - header_words, max_words // 2)
                for piece in _split_words(" | ".join(cell for cell in row if cell), budget):
                    parts.append(_render_table(header + [(piece,)], unit.caption, unit.lead_in))
                continue
            if current and header_words + sum(count_words(" ".join(r)) for r in current) + row_words > max_words:
                parts.append(_render_table(header + current, unit.caption, unit.lead_in))
                current = []
            current.append(row)
        if current:
            parts.append(_render_table(header + current, unit.caption, unit.lead_in))
        return [(part, unit.pages) for part in parts]  # row->page mapping is lost on stitch; keep the full span

    pieces: list[tuple[str, int]] = []  # (text, page)
    for ab in unit.blocks:
        text = ab.block.full_text
        if count_words(text) > max_words:
            pieces.extend((part, ab.block.page) for part in _split_words(text, max_words))
        else:
            pieces.append((text, ab.block.page))

    parts: list[tuple[str, list[int]]] = []
    current_texts: list[str] = []
    current_pages: list[int] = []
    current_words = 0
    for text, page in pieces:
        words = count_words(text)
        if current_texts and current_words + words > max_words:
            parts.append(("\n".join(current_texts), current_pages))
            current_texts, current_pages, current_words = [], [], 0
        current_texts.append(text)
        current_pages.append(page)
        current_words += words
    if current_texts:
        parts.append(("\n".join(current_texts), current_pages))
    return parts


def _context_header(
    spec: DocumentSpec, unit_first: AnnotatedBlock, chunk_type: ChunkType, part: int, total: int
) -> str:
    path = " > ".join(unit_first.section_path) or "(no section)"
    bits = [spec.title, path]
    if unit_first.paragraph_id:
        label = f"para {unit_first.paragraph_id}"
        if unit_first.scope != "main":
            label = f"{unit_first.scope}, {label}"
        if unit_first.provision:
            label += f" ({'Standard' if unit_first.provision == 'S' else 'Guidance'})"
        bits.append(label)
    if chunk_type is ChunkType.TABLE:
        bits.append("table")
    if total > 1:
        bits.append(f"part {part} of {total}")
    return "[" + " | ".join(bits) + "]"


def build_chunks(
    parsed: ParsedDocument,
    annotated: list[AnnotatedBlock],
    spec: DocumentSpec,
    config: ChunkingConfig | None = None,
) -> tuple[list[Chunk], dict]:
    """Returns (chunks, stats). Stats feed the debug report."""
    config = config or ChunkingConfig()
    units, footnotes, heading_owners = _group_into_units(annotated)
    unreferenced = _attach_footnotes(units, footnotes, heading_owners)
    for fn in unreferenced:  # keep them searchable rather than silently losing content
        units.append(_Unit(ChunkType.FOOTNOTE, first=fn, blocks=[fn]))

    chunks: list[Chunk] = []
    seen_keys: dict[str, int] = {}
    ordinal_by_owner: dict[tuple, int] = {}
    stats = {
        "split_units": 0,
        "duplicate_keys": [],  # main body: the same paragraph number twice means a rule misfired
        "restarted_numbering": [],  # appendices: numbering restarting per sub-section is normal
        "unreferenced_footnotes": [fn.block.text[:80] for fn in unreferenced],
    }

    for unit in units:
        first = unit.first
        parts = _split_unit_text(unit, config.max_words)
        if len(parts) > 1:
            stats["split_units"] += 1

        # Identity: numbered paragraphs are keyed by their number; everything
        # else by its position under that paragraph/heading. Scope keeps
        # "1.1" in Appendix 1 apart from "1.1" in the main body.
        if unit.chunk_type is ChunkType.PARAGRAPH:
            ident = first.paragraph_id
        else:
            # Numbered per anchor: para 10.4's "Good practices" box is 10.4#1,
            # its "Poor practices" box 10.4#2.
            anchor = first.paragraph_id or (first.section_path[-1] if first.section_path else "root")
            owner = (first.scope, anchor, unit.chunk_type)
            ordinal_by_owner[owner] = ordinal_by_owner.get(owner, 0) + 1
            ident = f"{anchor}#{ordinal_by_owner[owner]}"
        base_key = f"{spec.code}|{first.scope}|{unit.chunk_type.value}|{ident}"
        if base_key in seen_keys:
            seen_keys[base_key] += 1
            stats["duplicate_keys" if first.scope == "main" else "restarted_numbering"].append(base_key)
            base_key = f"{base_key}~{seen_keys[base_key]}"
        else:
            seen_keys[base_key] = 1

        tags = sorted({tag for ab in unit.blocks for tag in ab.tags})
        for part_index, (text, pages) in enumerate(parts, start=1):
            key = f"{base_key}|{part_index}"
            header = _context_header(spec, first, unit.chunk_type, part_index, len(parts))
            # Footnotes go on the last part, beside the text that references them.
            notes = unit.footnotes if part_index == len(parts) else []
            embed_text = "\n".join([header, text] + (["Footnotes:"] + notes if notes else []))
            page_start, page_end = min(pages), max(pages)
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid5(CHUNK_NAMESPACE, key)),
                    chunk_key=key,
                    doc_code=spec.code,
                    doc_title=spec.title,
                    source_file=parsed.source_file,
                    source_sha256=parsed.source_sha256,
                    chunk_type=unit.chunk_type,
                    paragraph_id=first.paragraph_id,
                    provision=first.provision,
                    scope=first.scope,
                    section_path=list(first.section_path),
                    page_start=page_start,
                    page_end=page_end,
                    printed_page_start=parsed.printed_pages.get(page_start),
                    printed_page_end=parsed.printed_pages.get(page_end),
                    part_index=part_index,
                    part_count=len(parts),
                    tags=tags,
                    text=text,
                    footnotes=list(notes),
                    embed_text=embed_text,
                    word_count=count_words(text),
                    content_hash=hashlib.sha256(embed_text.encode("utf-8")).hexdigest()[:16],
                    block_indexes=[ab.block.index for ab in unit.blocks],
                )
            )
    return chunks, stats
