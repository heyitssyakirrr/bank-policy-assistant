"""Data models shared by every ingestion stage.

The pipeline is a chain of small, pure transformations:

    PDF --parser--> ParsedDocument (Blocks) --structure--> AnnotatedBlocks --chunker--> Chunks

Each stage only depends on the models in this file, never on the stage
before it. That is what lets us swap the parser (Docling -> something else)
or the chunker (paragraph -> fixed-size, for the Phase 2 experiment)
without touching the rest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum


class BlockKind(StrEnum):
    """What the PARSER says a block is. Parser-neutral on purpose:
    Docling's labels are mapped onto these, so nothing downstream ever
    sees a Docling type."""

    HEADING = "heading"
    TEXT = "text"
    LIST_ITEM = "list_item"
    TABLE = "table"
    FOOTNOTE = "footnote"
    CAPTION = "caption"
    OTHER = "other"  # pictures, table of contents, form fields: no indexable text


class Role(StrEnum):
    """What OUR RULES decide a block means in the document's structure."""

    PARAGRAPH_START = "paragraph_start"  # "S 8.1 A financial institution shall..."
    CONTINUATION = "continuation"  # sub-items (i)/(ii), trailing text: belongs to the open paragraph
    HEADING = "heading"  # updates the section path, produces no chunk itself
    CALLOUT = "callout"  # "Good practices" box title: starts an illustrative block tied to the paragraph above
    TABLE = "table"
    FOOTNOTE = "footnote"
    SKIP = "skip"  # captions, pictures, manifest-skipped pages


class ChunkType(StrEnum):
    PARAGRAPH = "paragraph"  # one numbered provision (the normal case)
    TEXT = "text"  # unnumbered text under a heading (intros, callout boxes)
    TABLE = "table"
    FOOTNOTE = "footnote"  # a footnote no chunk referenced, kept so it stays searchable


@dataclass(frozen=True, slots=True)
class Block:
    """One item exactly as the parser read it, in reading order."""

    index: int  # position in the document, stable across runs
    kind: BlockKind
    text: str  # tables: rendered markdown
    page: int  # 1-based PDF page index
    marker: str = ""  # list marker as printed, e.g. "(i)", "1.", "A."
    source_label: str = ""  # the parser's own label, kept only for debugging
    table_rows: tuple[tuple[str, ...], ...] = ()  # TABLE only: cell text, row by row
    caption: str = ""  # TABLE only
    top: float | None = None  # distance from the top of the page (points); lines up side-by-side columns

    @property
    def full_text(self) -> str:
        """Marker + text, the way a human reads the line. Docling sometimes
        puts the number in `marker` and sometimes leaves it in `text`;
        normalising here means the rules only ever look at one string."""
        if self.marker and not self.text.startswith(self.marker):
            return f"{self.marker} {self.text}".strip()
        return self.text


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    doc_code: str
    source_file: str
    source_sha256: str
    page_count: int
    blocks: tuple[Block, ...]
    printed_pages: dict[int, str]  # PDF page -> page number printed on it ("6")
    furniture_words: dict[int, int]  # PDF page -> words removed as header/footer


@dataclass(frozen=True, slots=True)
class AnnotatedBlock:
    block: Block
    role: Role
    section_path: tuple[str, ...]
    paragraph_id: str | None = None  # "8.1", "14C.10.7", or "5" inside an appendix
    provision: str | None = None  # "S" (standard, must comply) / "G" (guidance)
    scope: str = "main"  # "main" or the appendix name; paragraph ids repeat across scopes
    tags: tuple[str, ...] = ()  # from the manifest, e.g. ("form_template",)
    reason: str = ""  # which rule fired; printed in the debug file


@dataclass(slots=True)
class Chunk:
    """The unit we embed, index and cite. Everything the retriever, the
    reranker and the citation UI need later is on here, so the next steps
    never have to go back to the PDF."""

    chunk_id: str  # UUID derived from chunk_key: same input -> same id (idempotent upserts)
    chunk_key: str  # human-readable identity, e.g. "ekyc|main|para|8.1|1"
    doc_code: str
    doc_title: str
    source_file: str
    source_sha256: str
    chunk_type: ChunkType
    paragraph_id: str | None
    provision: str | None
    scope: str
    section_path: list[str]
    page_start: int
    page_end: int
    printed_page_start: str | None
    printed_page_end: str | None
    part_index: int  # 1-based; >1 only when an oversized unit was split
    part_count: int
    tags: list[str]
    text: str  # the provision text itself, shown to the user / LLM
    footnotes: list[str]
    embed_text: str  # context header + text + footnotes: what we embed and BM25-index
    word_count: int
    content_hash: str  # changes when embed_text changes -> tells us what to re-embed
    block_indexes: list[int] = field(default_factory=list)  # provenance, for debugging

    def citation(self) -> str:
        """e.g. 'e-KYC, para 8.2 (S), p. 6' — the format the UI will show."""
        if self.paragraph_id:
            where = f"para {self.paragraph_id}" + (f" ({self.provision})" if self.provision else "")
            if self.scope != "main":
                where = f"{self.scope}, {where}"
        else:
            where = self.section_path[-1] if self.section_path else ""
        if self.chunk_type is ChunkType.TABLE:
            where = f"{where}, table" if where else "table"
        if "callout" in self.tags and self.section_path:
            where = f"{where}, {self.section_path[-1]}"
        page = self.printed_page_start or f"pdf {self.page_start}"
        return f"{self.doc_title}, {where}, p. {page}"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["chunk_type"] = self.chunk_type.value
        return data
