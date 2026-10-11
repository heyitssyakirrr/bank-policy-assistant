"""PDF -> ParsedDocument, using Docling.

This is the ONLY module that imports Docling. It translates Docling's
labels into our own BlockKind, so the structure rules and the chunker never
depend on Docling's types. Swapping parsers later means rewriting this file
only.

Two things make re-runs cheap:
  * The converter (which loads layout + table models) is created once and
    reused for every document.
  * Docling's output is cached as JSON, keyed by the PDF's SHA-256. Editing
    a regex rule and re-running takes seconds instead of re-parsing for
    minutes; replacing a PDF changes its hash, so the cache can't go stale.
"""

from __future__ import annotations

import hashlib
import html
import logging
import re
from pathlib import Path

from docling_core.types.doc import (
    ContentLayer,
    DocItemLabel,
    DoclingDocument,
    TableItem,
)

from app.ingestion.models import Block, BlockKind, ParsedDocument

logger = logging.getLogger("app.ingestion.parser")

# "Page 6 of 28" (e-KYC) or "145 of 178" (AML/FTFC headers)
PRINTED_PAGE_RE = re.compile(r"(?:Page\s+)?(\d{1,4})\s+of\s+\d{1,4}\b", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")

_LABEL_TO_KIND: dict[DocItemLabel, BlockKind] = {
    DocItemLabel.SECTION_HEADER: BlockKind.HEADING,
    DocItemLabel.TITLE: BlockKind.HEADING,
    DocItemLabel.TEXT: BlockKind.TEXT,
    DocItemLabel.PARAGRAPH: BlockKind.TEXT,
    DocItemLabel.LIST_ITEM: BlockKind.LIST_ITEM,
    DocItemLabel.TABLE: BlockKind.TABLE,
    DocItemLabel.FOOTNOTE: BlockKind.FOOTNOTE,
    DocItemLabel.CAPTION: BlockKind.CAPTION,
}
_FURNITURE_LABELS = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_text(text: str) -> str:
    """Collapse whitespace and undo HTML escapes (`&amp;` -> `&`)."""
    return WHITESPACE_RE.sub(" ", html.unescape(text or "")).strip()


def table_rows(item: TableItem) -> tuple[tuple[str, ...], ...]:
    rows = (tuple(clean_text(cell.text) for cell in row) for row in item.data.grid)
    return tuple(row for row in rows if any(row))


def rows_to_markdown(rows: tuple[tuple[str, ...], ...]) -> str:
    """First row is treated as the header row."""
    if not rows:
        return ""

    def fmt(row: tuple[str, ...]) -> str:
        return "| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |"

    width = max(len(row) for row in rows)
    padded = [row + ("",) * (width - len(row)) for row in rows]
    lines = [fmt(padded[0]), "|" + "---|" * width]
    lines += [fmt(row) for row in padded[1:]]
    return "\n".join(lines)


def to_parsed_document(doc: DoclingDocument, *, doc_code: str, source_file: str, source_sha256: str) -> ParsedDocument:
    """Translate a DoclingDocument into our parser-neutral model.
    Kept separate from DoclingParser so it can be unit-tested with a
    hand-built DoclingDocument, without running any models."""
    blocks: list[Block] = []
    printed_pages: dict[int, str] = {}
    furniture_words: dict[int, int] = {}
    unknown_labels: dict[str, int] = {}

    layers = {ContentLayer.BODY, ContentLayer.FURNITURE}
    for item, _level in doc.iterate_items(included_content_layers=layers):
        prov = getattr(item, "prov", None)
        if not prov:
            continue  # group containers carry no page and no text
        page = prov[0].page_no
        label = item.label

        # Page furniture: never indexed, but read for the printed page number.
        if label in _FURNITURE_LABELS or item.content_layer == ContentLayer.FURNITURE:
            text = clean_text(getattr(item, "text", ""))
            furniture_words[page] = furniture_words.get(page, 0) + len(text.split())
            match = PRINTED_PAGE_RE.search(text)
            if match and page not in printed_pages:
                printed_pages[page] = match.group(1)
            continue

        kind = _LABEL_TO_KIND.get(label, BlockKind.OTHER)
        if kind is BlockKind.OTHER and label not in (DocItemLabel.PICTURE, DocItemLabel.DOCUMENT_INDEX):
            unknown_labels[label.value] = unknown_labels.get(label.value, 0) + 1

        if kind is BlockKind.TABLE and isinstance(item, TableItem):
            rows = table_rows(item)
            blocks.append(
                Block(
                    index=len(blocks),
                    kind=kind,
                    text=rows_to_markdown(rows),
                    page=page,
                    source_label=label.value,
                    table_rows=rows,
                    caption=clean_text(item.caption_text(doc)),
                )
            )
            continue

        text = clean_text(getattr(item, "text", ""))
        if not text and kind is not BlockKind.OTHER:
            continue
        blocks.append(
            Block(
                index=len(blocks),
                kind=kind,
                text=text,
                page=page,
                marker=clean_text(getattr(item, "marker", "") or ""),
                source_label=label.value,
            )
        )

    if unknown_labels:
        logger.info(f"{doc_code}: labels treated as non-indexable: {unknown_labels}")

    return ParsedDocument(
        doc_code=doc_code,
        source_file=source_file,
        source_sha256=source_sha256,
        page_count=doc.num_pages(),
        blocks=tuple(blocks),
        printed_pages=printed_pages,
        furniture_words=furniture_words,
    )


class DoclingParser:
    def __init__(self, cache_dir: Path, *, use_cache: bool = True) -> None:
        self._cache_dir = cache_dir
        self._use_cache = use_cache
        self._converter = None  # built lazily: loading models takes a while

    def _get_converter(self):
        if self._converter is None:
            # Imported here so the rest of the app (and the unit tests) can
            # import this module with only docling-core installed.
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption

            options = PdfPipelineOptions()
            options.do_ocr = False  # digital PDFs: the text layer is exact, OCR only adds noise
            options.do_table_structure = True
            self._converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
            )
        return self._converter

    def _load_or_convert(self, pdf_path: Path, sha256: str, doc_code: str) -> DoclingDocument:
        cache_file = self._cache_dir / f"{doc_code}-{sha256[:12]}.docling.json"
        if self._use_cache and cache_file.exists():
            logger.info(f"{doc_code}: using cached parse {cache_file.name}")
            return DoclingDocument.load_from_json(cache_file)

        logger.info(f"{doc_code}: parsing {pdf_path.name} with Docling (first run is slow)")
        result = self._get_converter().convert(pdf_path)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        result.document.save_as_json(cache_file)
        return result.document

    def parse(self, pdf_path: Path, *, doc_code: str) -> ParsedDocument:
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        sha256 = file_sha256(pdf_path)
        doc = self._load_or_convert(pdf_path, sha256, doc_code)
        return to_parsed_document(doc, doc_code=doc_code, source_file=pdf_path.name, source_sha256=sha256)
